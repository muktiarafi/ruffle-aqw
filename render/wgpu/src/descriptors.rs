use crate::filters::{FilterVertex, Filters};
use crate::layouts::BindLayouts;
use crate::pipelines::VERTEX_BUFFERS_DESCRIPTION_POS_UV;
use crate::shaders::Shaders;
use crate::{
    BitmapSamplers, Pipelines, PosColorVertex, PosUvVertex, PosVertex, create_buffer_with_data,
};
use fnv::FnvHashMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::Backend;

pub struct Descriptors {
    pub wgpu_instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub limits: wgpu::Limits,
    pub backend: Backend,
    pub queue: wgpu::Queue,
    pub bitmap_samplers: BitmapSamplers,
    pub bind_layouts: BindLayouts,
    pub quad: Quad,
    copy_pipeline: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    pub shaders: Shaders,
    /// AQW build: one cell per combination, built outside the map lock, so a combination
    /// being built (e.g. by `warm_up`) doesn't block the others.
    pipelines: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), Arc<OnceLock<Arc<Pipelines>>>>>,
    pub filters: Filters,
    warm_up_started: AtomicBool,
}

impl Debug for Descriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Descriptors")
    }
}

impl Descriptors {
    pub fn new(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let limits = device.limits();
        let bind_layouts = BindLayouts::new(&device);
        let bitmap_samplers = BitmapSamplers::new(&device);
        let shaders = Shaders::new(&device);
        let quad = Quad::new(&device);
        let filters = Filters::new(&device);
        let backend = adapter.get_info().backend;

        Self {
            wgpu_instance: instance,
            adapter,
            device,
            limits,
            backend,
            queue,
            bitmap_samplers,
            bind_layouts,
            quad,
            copy_pipeline: Default::default(),
            shaders,
            pipelines: Default::default(),
            filters,
            warm_up_started: AtomicBool::new(false),
        }
    }

    /// AQW build: on DirectX 12, creating a render pipeline compiles its shaders (with the
    /// slow FXC compiler), and Ruffle creates ~90 of them the first time it draws with a new
    /// MSAA sample count, plus filter pipelines on first use: multi-second freezes when
    /// entering a room or seeing an effect for the first time (2.8 s entering Yulgar on a
    /// Ryzen 9). Builds them all on background threads at startup, while AQW is loading.
    /// Vulkan doesn't need this (no freezes there), so it does nothing on other backends.
    pub fn warm_up(self: &Arc<Self>, target_format: wgpu::TextureFormat) {
        if self.backend != Backend::Dx12 || self.warm_up_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut formats = vec![target_format];
        if !formats.contains(&wgpu::TextureFormat::Rgba8Unorm) {
            formats.push(wgpu::TextureFormat::Rgba8Unorm);
        }
        // AQW's quality: High = 4x, Medium = 2x, Low = 1x (offscreen layers too).
        let mut sample_counts: Vec<u32> = [4, 2, 1]
            .into_iter()
            .map(|count| crate::utils::supported_sample_count(&self.adapter, count, target_format))
            .collect();
        sample_counts.dedup();
        let started = std::time::Instant::now();
        let mut jobs: Vec<Box<dyn FnOnce(&Descriptors) + Send>> = Vec::new();
        for &format in &formats {
            for &count in &sample_counts {
                jobs.push(Box::new(move |d: &Descriptors| {
                    let pipelines = d.pipelines(count, format);
                    // The blends read from an atlas page have pipelines of their own, each
                    // made on first use otherwise: on an Intel UHD 620 that froze the
                    // picture for 50 to 120 ms per blend mode in the middle of play (571 ms
                    // once). After the others, the modes AQW draws most first.
                    if crate::surface::commands::page_blends_enabled() {
                        use crate::blend::ComplexBlend;
                        for blend in [
                            ComplexBlend::Overlay,
                            ComplexBlend::Multiply,
                            ComplexBlend::HardLight,
                            ComplexBlend::Lighten,
                            ComplexBlend::Darken,
                            ComplexBlend::Difference,
                            ComplexBlend::Invert,
                        ] {
                            pipelines.complex_blend_page(d, blend);
                        }
                    }
                }));
            }
            jobs.push(Box::new(move |d: &Descriptors| {
                d.copy_pipeline(format, 1);
            }));
        }
        jobs.push(Box::new(|d: &Descriptors| d.filters.warm_up(d, 1)));
        let remaining = Arc::new(std::sync::atomic::AtomicUsize::new(jobs.len()));
        for (index, job) in jobs.into_iter().enumerate() {
            let descriptors = self.clone();
            let remaining = remaining.clone();
            let _ = std::thread::Builder::new()
                .name(format!("dx12 warm-up {index}"))
                .spawn(move || {
                    crate::perf_counters::timers::untimed_thread();
                    job(&descriptors);
                    if remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                        tracing::info!(
                            target: "ruffle_aqw_perf",
                            "DirectX 12 pipelines ready after {:.1} s",
                            started.elapsed().as_secs_f64()
                        );
                    }
                });
        }
    }

    pub fn copy_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .copy_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                let _timer = crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::PIPELINE);
                let copy_texture_pipeline_layout =
                    &self
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: create_debug_label!("Copy pipeline layout").as_deref(),
                            bind_group_layouts: &[
                                Some(&self.bind_layouts.globals),
                                None,
                                Some(&self.bind_layouts.bitmap),
                            ],
                            immediate_size: 0,
                        });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Copy pipeline").as_deref(),
                        layout: Some(copy_texture_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &VERTEX_BUFFERS_DESCRIPTION_POS_UV,
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                // All of our blending has been done by now, so we want
                                // to overwrite the target pixels without any blending
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview_mask: None,
                        cache: None,
                    })
            })
            .clone()
    }

    pub fn pipelines(&self, msaa_sample_count: u32, format: wgpu::TextureFormat) -> Arc<Pipelines> {
        let cell = self
            .pipelines
            .lock()
            .expect("Pipelines should not be already locked")
            .entry((msaa_sample_count, format))
            .or_default()
            .clone();
        cell.get_or_init(|| {
            let _timer = crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::PIPELINE);
            Arc::new(Pipelines::new(
                &self.device,
                &self.shaders,
                format,
                msaa_sample_count,
                &self.bind_layouts,
            ))
        })
        .clone()
    }
}

pub struct Quad {
    pub vertices_pos: wgpu::Buffer,
    pub vertices_pos_uv: wgpu::Buffer,
    pub vertices_pos_color: wgpu::Buffer,
    pub filter_vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub indices_line: wgpu::Buffer,
    pub indices_line_rect: wgpu::Buffer,
}

impl Quad {
    pub fn new(device: &wgpu::Device) -> Self {
        let vertices_pos = [
            PosVertex {
                position: [0.0, 0.0],
            },
            PosVertex {
                position: [1.0, 0.0],
            },
            PosVertex {
                position: [1.0, 1.0],
            },
            PosVertex {
                position: [0.0, 1.0],
            },
        ];
        let vertices_pos_uv = [
            PosUvVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0, 1.0],
            },
            PosUvVertex {
                position: [1.0, 0.0],
                uv: [1.0, 0.0, 1.0],
            },
            PosUvVertex {
                position: [1.0, 1.0],
                uv: [1.0, 1.0, 1.0],
            },
            PosUvVertex {
                position: [0.0, 1.0],
                uv: [0.0, 1.0, 1.0],
            },
        ];
        let vertices_pos_color = [
            PosColorVertex {
                position: [0.0, 0.0],
                color: [255, 255, 255, 255],
            },
            PosColorVertex {
                position: [1.0, 0.0],
                color: [255, 255, 255, 255],
            },
            PosColorVertex {
                position: [1.0, 1.0],
                color: [255, 255, 255, 255],
            },
            PosColorVertex {
                position: [0.0, 1.0],
                color: [255, 255, 255, 255],
            },
        ];
        let filter_vertices = [
            FilterVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 0.0],
                uv: [1.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
            FilterVertex {
                position: [0.0, 1.0],
                uv: [0.0, 1.0],
            },
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let indices_line: [u32; 2] = [0, 1];
        let indices_line_rect: [u32; 5] = [0, 1, 2, 3, 0];

        let vbo_pos = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&vertices_pos),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (pos)"),
        );

        let vbo_pos_uv = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&vertices_pos_uv),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (pos & uv)"),
        );

        let vbo_pos_color = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&vertices_pos_color),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (pos & color)"),
        );

        let vbo_filter = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&filter_vertices),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (filter)"),
        );

        let ibo = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Quad ibo"),
        );
        let ibo_line = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices_line),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Line ibo"),
        );
        let ibo_line_rect = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices_line_rect),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Line rect ibo"),
        );

        Self {
            vertices_pos: vbo_pos,
            vertices_pos_uv: vbo_pos_uv,
            vertices_pos_color: vbo_pos_color,
            filter_vertices: vbo_filter,
            indices: ibo,
            indices_line: ibo_line,
            indices_line_rect: ibo_line_rect,
        }
    }
}
