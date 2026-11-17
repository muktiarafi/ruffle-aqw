//! AQW build: blur and glow passes for many cacheAsBitmap objects at once.
//!
//! The objects' contents are drawn side by side into an atlas page (see `crate::cache_batch`);
//! these passes then run each filter step over every slot of the page in one render pass, each
//! slot with its own arguments (instanced quads). Per slot, the math and the sampled texels are
//! the same as `BlurFilter`/`GlowFilter` on the object's own texture: samples are clamped to the
//! slot as the sampler clamped them to the texture's edges.

use crate::backend::RenderTargetMode;
use crate::buffer_pool::TexturePool;
use crate::descriptors::Descriptors;
use crate::filters::FilterVertex;
use crate::surface::target::CommandTarget;
use bytemuck::{Pod, Zeroable};
use std::sync::OnceLock;
use swf::{BlurFilter as BlurFilterArgs, GlowFilter as GlowFilterArgs};
use wgpu::util::StagingBelt;
use wgpu::vertex_attr_array;

/// The most slots one page (and one instanced draw) holds.
pub const MAX_SLOTS: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlurInstance {
    rect: [f32; 4],
    page_dir: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GlowInstance {
    rect: [f32; 4],
    page_offset: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 4] = vertex_attr_array![
    2 => Float32x4,
    3 => Float32x4,
    4 => Float32x4,
    5 => Float32x4,
];

/// A slot of the page: where it is, in texels.
#[derive(Copy, Clone, Debug)]
pub struct Slot {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Slot {
    fn rect(&self) -> [f32; 4] {
        [
            self.x as f32,
            self.y as f32,
            self.width as f32,
            self.height as f32,
        ]
    }
}

/// The glow step of one slot (a glow, or a drop shadow's glow with its offset).
pub struct SlotGlow<'a> {
    pub slot: Slot,
    pub filter: &'a GlowFilterArgs,
    pub blur_offset: (f32, f32),
}

pub struct FilterBatch {
    blur_layout: wgpu::BindGroupLayout,
    blur_pipeline_layout: wgpu::PipelineLayout,
    blur_pipeline: OnceLock<wgpu::RenderPipeline>,
    glow_layout: wgpu::BindGroupLayout,
    glow_pipeline_layout: wgpu::PipelineLayout,
    glow_pipeline: OnceLock<wgpu::RenderPipeline>,
    instances: wgpu::Buffer,
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, filtering: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(if filtering {
            wgpu::SamplerBindingType::Filtering
        } else {
            wgpu::SamplerBindingType::NonFiltering
        }),
        count: None,
    }
}

impl FilterBatch {
    pub fn new(device: &wgpu::Device) -> Self {
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: create_debug_label!("Batched blur binds").as_deref(),
            entries: &[texture_entry(0, true), sampler_entry(1, true)],
        });
        let glow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: create_debug_label!("Batched glow binds").as_deref(),
            entries: &[
                texture_entry(0, false),
                sampler_entry(1, false),
                texture_entry(2, false),
            ],
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&blur_layout)],
            immediate_size: 0,
        });
        let glow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&glow_layout)],
            immediate_size: 0,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: create_debug_label!("Batched filter instances").as_deref(),
            size: (MAX_SLOTS * size_of::<BlurInstance>().max(size_of::<GlowInstance>())) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            blur_layout,
            blur_pipeline_layout,
            blur_pipeline: OnceLock::new(),
            glow_layout,
            glow_pipeline_layout,
            glow_pipeline: OnceLock::new(),
            instances,
        }
    }

    fn pipeline(
        descriptors: &Descriptors,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let _timer =
            crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::PIPELINE);
        descriptors
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: create_debug_label!("{label}").as_deref(),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("main_vertex"),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: size_of::<FilterVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                        }),
                        Some(wgpu::VertexBufferLayout {
                            array_stride: size_of::<BlurInstance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &INSTANCE_ATTRIBUTES,
                        }),
                    ],
                    compilation_options: Default::default(),
                },
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
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("main_fragment"),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Builds the pipelines ahead of time (see `Descriptors::warm_up`).
    pub fn warm_up(&self, descriptors: &Descriptors) {
        self.blur_pipeline(descriptors);
        self.glow_pipeline(descriptors);
    }

    fn blur_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.blur_pipeline.get_or_init(|| {
            Self::pipeline(
                descriptors,
                &self.blur_pipeline_layout,
                &descriptors.shaders.batch_blur_filter,
                "Batched blur filter",
            )
        })
    }

    fn glow_pipeline(&self, descriptors: &Descriptors) -> &wgpu::RenderPipeline {
        self.glow_pipeline.get_or_init(|| {
            Self::pipeline(
                descriptors,
                &self.glow_pipeline_layout,
                &descriptors.shaders.batch_glow_filter,
                "Batched glow filter",
            )
        })
    }

    fn new_page_target(
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
    ) -> CommandTarget {
        CommandTarget::new(
            descriptors,
            texture_pool,
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            draw_encoder,
        )
    }

    /// One render pass drawing `count` instances (already written) over `target`.
    #[expect(clippy::too_many_arguments)]
    fn draw_instances(
        &self,
        descriptors: &Descriptors,
        draw_encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        target: &CommandTarget,
        count: usize,
        label: &str,
    ) {
        crate::perf_counters::inc(crate::perf_counters::RENDER_PASS);
        let _ = label;
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!("{label}").as_deref(),
            color_attachments: &[target.color_attachments()],
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, descriptors.quad.filter_vertices.slice(..));
        render_pass.set_vertex_buffer(1, self.instances.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..6, 0, 0..count as u32);
    }

    /// Blurs every slot of `source` with its own blur arguments (`None`: leave it as it is),
    /// like `BlurFilter::apply` on each slot's own texture. Returns `None` if no slot needed a
    /// single pass (`BlurFilter::apply` returns `None` then too).
    #[expect(clippy::too_many_arguments)]
    pub fn blur(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &wgpu::Texture,
        page: (u32, u32),
        slots: &[(Slot, Option<BlurFilterArgs>)],
    ) -> Option<CommandTarget> {
        debug_assert!(slots.len() <= MAX_SLOTS);
        let max_passes = slots
            .iter()
            .filter_map(|(_, filter)| filter.as_ref().map(|f| f.num_passes()))
            .max()
            .unwrap_or(0);
        let pipeline = self.blur_pipeline(descriptors);
        let source_view = source.create_view(&Default::default());
        let mut flip = Self::new_page_target(descriptors, texture_pool, draw_encoder, page);
        let mut flop = Self::new_page_target(descriptors, texture_pool, draw_encoder, page);
        let mut first = true;
        let mut instances = Vec::with_capacity(slots.len());
        for pass in 0..max_passes {
            for horizontal in [true, false] {
                instances.clear();
                let mut any = false;
                for (slot, filter) in slots {
                    let full_size = filter.as_ref().and_then(|f| {
                        (pass < f.num_passes()).then(|| {
                            if horizontal {
                                f.blur_x.to_f32()
                            } else {
                                f.blur_y.to_f32()
                            }
                            .min(255.0)
                        })
                    });
                    let mut instance = BlurInstance {
                        rect: slot.rect(),
                        page_dir: [
                            page.0 as f32,
                            page.1 as f32,
                            if horizontal { 1.0 } else { 0.0 },
                            if horizontal { 0.0 } else { 1.0 },
                        ],
                        params: [1.0, 0.0, 0.0, 0.0],
                        params2: [0.0, 0.0, 1.0, 0.0],
                    };
                    // Same arguments as `BlurFilter::apply`; a width of 1 or less is skipped there.
                    if let Some(full_size) = full_size.filter(|&s| s > 1.0) {
                        any = true;
                        let radius = (full_size - 1.0) / 2.0;
                        let m = radius.ceil() - 1.0;
                        let alpha = ((radius - m) * 255.0).floor() / 255.0;
                        let last_offset = 1.0 / ((1.0 / alpha) + 1.0);
                        let last_weight = alpha + 1.0;
                        instance.params = [full_size, m, m * 2.0, alpha];
                        instance.params2 = [last_offset, last_weight, 0.0, 0.0];
                    }
                    instances.push(instance);
                }
                if !any {
                    // Every slot skips this pass: nothing changes (no pass in `BlurFilter` either).
                    continue;
                }
                let previous_view = if first { &source_view } else { flip.color_view() };
                first = false;
                staging_belt
                    .write_buffer(
                        draw_encoder,
                        &self.instances,
                        0,
                        wgpu::BufferSize::new(
                            (instances.len() * size_of::<BlurInstance>()) as u64,
                        )
                        .expect("at least one slot"),
                    )
                    .copy_from_slice(bytemuck::cast_slice(&instances));
                let bind_group = descriptors
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: create_debug_label!("Batched blur group").as_deref(),
                        layout: &self.blur_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(previous_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    descriptors.bitmap_samplers.get_sampler(false, true),
                                ),
                            },
                        ],
                    });
                self.draw_instances(
                    descriptors,
                    draw_encoder,
                    pipeline,
                    &bind_group,
                    &flop,
                    instances.len(),
                    "Batched blur",
                );
                std::mem::swap(&mut flip, &mut flop);
            }
        }
        if first { None } else { Some(flip) }
    }

    /// The glow step for every slot, like `GlowFilter::apply` after its blur: `source` is the
    /// page before the filter, `blurred` the blurred page (or `source` if nothing was blurred).
    #[expect(clippy::too_many_arguments)]
    pub fn glow(
        &self,
        descriptors: &Descriptors,
        texture_pool: &mut TexturePool,
        draw_encoder: &mut wgpu::CommandEncoder,
        staging_belt: &mut StagingBelt,
        source: &wgpu::Texture,
        blurred: &wgpu::Texture,
        page: (u32, u32),
        slots: &[SlotGlow],
    ) -> CommandTarget {
        debug_assert!(!slots.is_empty() && slots.len() <= MAX_SLOTS);
        let pipeline = self.glow_pipeline(descriptors);
        let instances: Vec<GlowInstance> = slots
            .iter()
            .map(|glow| {
                let filter = glow.filter;
                let flag = |set: bool| if set { 1.0 } else { 0.0 };
                GlowInstance {
                    rect: glow.slot.rect(),
                    page_offset: [
                        page.0 as f32,
                        page.1 as f32,
                        glow.blur_offset.0,
                        glow.blur_offset.1,
                    ],
                    color: [
                        f32::from(filter.color.r) / 255.0,
                        f32::from(filter.color.g) / 255.0,
                        f32::from(filter.color.b) / 255.0,
                        f32::from(filter.color.a) / 255.0,
                    ],
                    params: [
                        filter.strength.to_f32(),
                        flag(filter.is_inner()),
                        flag(filter.is_knockout()),
                        flag(filter.composite_source()),
                    ],
                }
            })
            .collect();
        staging_belt
            .write_buffer(
                draw_encoder,
                &self.instances,
                0,
                wgpu::BufferSize::new((instances.len() * size_of::<GlowInstance>()) as u64)
                    .expect("at least one slot"),
            )
            .copy_from_slice(bytemuck::cast_slice(&instances));
        let target = Self::new_page_target(descriptors, texture_pool, draw_encoder, page);
        let source_view = source.create_view(&Default::default());
        let blurred_view = blurred.create_view(&Default::default());
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!("Batched glow group").as_deref(),
                layout: &self.glow_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&blurred_view),
                    },
                ],
            });
        self.draw_instances(
            descriptors,
            draw_encoder,
            pipeline,
            &bind_group,
            &target,
            instances.len(),
            "Batched glow",
        );
        target
    }
}
