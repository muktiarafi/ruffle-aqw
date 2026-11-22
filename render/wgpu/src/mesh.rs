use crate::backend::WgpuRenderBackend;
use crate::target::RenderTarget;
use crate::{Descriptors, GradientUniforms, PosColorVertex, PosUvVertex, as_texture};
use fnv::FnvHashMap;
use std::any::Any;
use std::ops::Range;
use std::sync::{Arc, Weak};
use wgpu::util::DeviceExt;

use crate::buffer_builder::BufferBuilder;
use ruffle_render::backend::{ShapeHandle, ShapeHandleImpl};
use ruffle_render::bitmap::{BitmapHandle, BitmapSource};
use ruffle_render::tessellator::{Bitmap, Draw as LyonDraw, DrawType as TessDrawType, Gradient};
use swf::GradientInterpolation;

/// How big to make gradient textures. Larger will keep more detail, but be slower and use more memory.
const GRADIENT_SIZE: usize = 256;

#[derive(Debug)]
pub struct Mesh {
    pub draws: Vec<Draw>,
    /// AQW build: slices of the shared shape buffers (see `crate::arena`); draw ranges are
    /// absolute offsets into them. `None` for a shape without geometry.
    pub vertices: Option<crate::arena::ArenaSlice>,
    pub indices: Option<crate::arena::ArenaSlice>,
    /// AQW build: the gradient ramps its draws use, shared between shapes (see
    /// `GradientCache`); kept alive for the draws' bind groups.
    pub gradients: Vec<Arc<SharedGradient>>,
    /// AQW build: counted in the perf log until the shape is freed.
    pub live: crate::perf_counters::LiveMesh,
    /// AQW build: the shape's bounds (in its own coordinates, including strokes), used to size
    /// blend layers to their content (see `surface::commands::layer_rect_and_exact_size`).
    pub bounds: swf::Rectangle<swf::Twips>,
    /// AQW build: a process-wide id, never reused (see `layer_cache`; a mesh is never changed
    /// in place, so the id alone identifies its geometry), and the bitmaps its draws sample
    /// (their contents can change in place: see `Texture::stamp`).
    pub id: u64,
    pub bitmaps: Vec<BitmapHandle>,
}

impl Mesh {
    /// AQW build: the next mesh id (see the `id` field).
    pub fn new_id() -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl ShapeHandleImpl for Mesh {}

pub fn as_mesh(handle: &ShapeHandle) -> &Mesh {
    <dyn Any>::downcast_ref(&*handle.0).expect("Shape handle must be a WGPU ShapeData")
}

#[derive(Debug)]
pub struct PendingDraw {
    pub draw_type: PendingDrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<wgpu::BufferAddress>,
    pub index_format: wgpu::IndexFormat,
    pub num_indices: u32,
    pub num_mask_indices: u32,
}

/// AQW build: a draw's indices are 16-bit when its vertices fit (nearly every shape): half the
/// index memory, and indices were the larger part of a shape's GPU geometry (each vertex is
/// used by several triangles). Shape geometry is what grows over a long AQW session.
fn index_format(vertex_count: usize) -> wgpu::IndexFormat {
    if vertex_count <= 1 << 16 {
        wgpu::IndexFormat::Uint16
    } else {
        wgpu::IndexFormat::Uint32
    }
}

impl PendingDraw {
    pub fn finish(
        self,
        vertex_base: wgpu::BufferAddress,
        index_base: wgpu::BufferAddress,
        gradients: &[Arc<SharedGradient>],
    ) -> Draw {
        Draw {
            draw_type: self.draw_type.finish(gradients),
            vertices: self.vertices.start + vertex_base..self.vertices.end + vertex_base,
            indices: self.indices.start + index_base..self.indices.end + index_base,
            index_format: self.index_format,
            num_indices: self.num_indices,
            num_mask_indices: self.num_mask_indices,
        }
    }
}

#[derive(Debug)]
pub struct Draw {
    pub draw_type: DrawType,
    pub vertices: Range<wgpu::BufferAddress>,
    pub indices: Range<wgpu::BufferAddress>,
    /// AQW build: 16-bit or 32-bit (see `index_format`).
    pub index_format: wgpu::IndexFormat,
    pub num_indices: u32,
    pub num_mask_indices: u32,
}

impl PendingDraw {
    pub fn new<T: RenderTarget>(
        backend: &mut WgpuRenderBackend<T>,
        source: &dyn BitmapSource,
        draw: LyonDraw,
        vertex_buffer: &mut BufferBuilder,
        index_buffer: &mut BufferBuilder,
    ) -> Option<Self> {
        Self::new_with(draw, vertex_buffer, index_buffer, |bitmap| {
            PendingDrawType::bitmap(bitmap, source, backend)
        })
    }

    /// AQW build: `new`, with a bitmap fill's texture and bind group looked up by
    /// `bitmap_fill` (see `crate::shape_uploads`: there the backend isn't always at hand).
    pub fn new_with(
        draw: LyonDraw,
        vertex_buffer: &mut BufferBuilder,
        index_buffer: &mut BufferBuilder,
        bitmap_fill: impl FnOnce(Bitmap) -> Option<PendingDrawType>,
    ) -> Option<Self> {
        let index_format = index_format(draw.vertices.len());
        let vertices = match &draw.draw_type {
            TessDrawType::Color => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(PosColorVertex::from)
                    .collect();
                vertex_buffer
                    .add_strided(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
            TessDrawType::Gradient { matrix, .. } => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(|v| PosUvVertex::from_tessellator(v, matrix))
                    .collect();
                vertex_buffer
                    .add_strided(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
            TessDrawType::Bitmap(bitmap) => {
                let vertices: Vec<_> = draw
                    .vertices
                    .into_iter()
                    .map(|v| PosUvVertex::from_tessellator(v, &bitmap.matrix))
                    .collect();
                vertex_buffer
                    .add_strided(&vertices)
                    .expect("Mesh vertex buffer was too large!")
            }
        };

        // Each draw's indices start at a multiple of their own size, so a draw can be bound
        // with its offset (see `CommandRenderer::render_shape`).
        let indices = match index_format {
            wgpu::IndexFormat::Uint16 => {
                let short: Vec<u16> = draw.indices.iter().map(|&i| i as u16).collect();
                index_buffer.add_strided(&short)
            }
            wgpu::IndexFormat::Uint32 => index_buffer.add_strided(&draw.indices),
        }
        .expect("Mesh index buffer was too large!");

        let index_count = draw.indices.len() as u32;
        let draw_type = match draw.draw_type {
            TessDrawType::Color => PendingDrawType::color(),
            TessDrawType::Gradient {
                matrix: _,
                gradient,
            } => PendingDrawType::gradient(gradient),
            TessDrawType::Bitmap(bitmap) => bitmap_fill(bitmap)?,
        };
        Some(PendingDraw {
            draw_type,
            vertices,
            indices,
            index_format,
            num_indices: index_count,
            num_mask_indices: draw.mask_index_count,
        })
    }
}

#[derive(Debug)]
pub enum PendingDrawType {
    Color,
    Gradient {
        gradient_index: usize,
    },
    Bitmap {
        binds: BitmapBinds,
        /// AQW build: the bitmap sampled, for the mesh's `bitmaps`.
        handle: BitmapHandle,
    },
}

/// Converts an RGBA color from sRGB space to linear color space.
fn srgb_to_linear(color: f32) -> f32 {
    if color <= 0.04045 {
        color / 12.92
    } else {
        f32::powf((color + 0.055) / 1.055, 2.4)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl PendingDrawType {
    pub fn color() -> Self {
        PendingDrawType::Color
    }

    pub fn gradient(gradient_index: usize) -> Self {
        PendingDrawType::Gradient { gradient_index }
    }

    pub fn bitmap<T: RenderTarget>(
        bitmap: Bitmap,
        source: &dyn BitmapSource,
        backend: &mut WgpuRenderBackend<T>,
    ) -> Option<Self> {
        let handle = source.bitmap_handle(bitmap.bitmap_id, backend)?;
        let texture = as_texture(&handle);
        let binds = texture.bind_group(
            bitmap.is_repeating,
            bitmap.is_smoothed,
            &backend.descriptors.device,
            &backend.descriptors.bind_layouts.bitmap,
            handle.clone(),
            &backend.descriptors.bitmap_samplers,
        );

        Some(PendingDrawType::Bitmap {
            binds: binds.clone(),
            handle,
        })
    }

    pub fn finish(self, gradients: &[Arc<SharedGradient>]) -> DrawType {
        match self {
            PendingDrawType::Color => DrawType::Color,
            PendingDrawType::Gradient { gradient_index } => DrawType::Gradient {
                bind_group: gradients[gradient_index].bind_group.clone(),
            },
            PendingDrawType::Bitmap { binds, .. } => DrawType::Bitmap { binds },
        }
    }
}

#[derive(Debug)]
pub enum DrawType {
    Color,
    Gradient { bind_group: wgpu::BindGroup },
    Bitmap { binds: BitmapBinds },
}

/// AQW build: what a gradient's GPU side is made of: its 256-pixel ramp and its uniforms
/// (type, focal point, interpolation, repeat). Two fills with the same key draw exactly the
/// same, so they share one `SharedGradient`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GradientKey {
    colors: Box<[u8; GRADIENT_SIZE * 4]>,
    uniforms: [u8; std::mem::size_of::<GradientUniforms>()],
}

impl GradientKey {
    pub fn of(gradient: &Gradient) -> Self {
        let uniforms = *bytemuck::bytes_of(&GradientUniforms::from(gradient.clone()))
            .first_chunk()
            .expect("GradientUniforms bytes");
        Self {
            colors: Box::new(ramp(gradient)),
            uniforms,
        }
    }

    /// The uniforms, as uploaded.
    pub fn uniform_bytes(&self) -> &[u8] {
        &self.uniforms
    }
}

/// The gradient's 256-pixel RGBA ramp (what stock Ruffle made a texture of per gradient).
fn ramp(gradient: &Gradient) -> [u8; GRADIENT_SIZE * 4] {
    if gradient.records.is_empty() {
        [0; GRADIENT_SIZE * 4]
    } else {
        let mut colors = [0; GRADIENT_SIZE * 4];
        let mut last = 0;
        let mut next;

        let convert = if gradient.interpolation == GradientInterpolation::LinearRgb {
            |c| srgb_to_linear(c / 255.0) * 255.0
        } else {
            |c| c
        };

        for t in 0..GRADIENT_SIZE {
            if last + 1 < gradient.records.len() && t > gradient.records[last + 1].ratio as usize {
                last += 1;
            }
            next = (last + 1).min(gradient.records.len() - 1);

            assert!(last == next || last + 1 == next);

            let last_record = &gradient.records[last];
            let next_record = &gradient.records[next];

            let a = if t <= last_record.ratio as usize || last_record.ratio == next_record.ratio {
                // We are before the first gradient record,
                // or this record's ratio is equal to the next one,
                // meaning we need to do a full stop of this color for 1 pixel.
                0.0
            } else if t > next_record.ratio as usize {
                // We are after the last record
                1.0
            } else {
                (t as f32 - last_record.ratio as f32)
                    / (next_record.ratio as f32 - last_record.ratio as f32)
            };

            colors[t * 4] = lerp(
                convert(last_record.color.r as f32),
                convert(next_record.color.r as f32),
                a,
            ) as u8;
            colors[(t * 4) + 1] = lerp(
                convert(last_record.color.g as f32),
                convert(next_record.color.g as f32),
                a,
            ) as u8;
            colors[(t * 4) + 2] = lerp(
                convert(last_record.color.b as f32),
                convert(next_record.color.b as f32),
                a,
            ) as u8;
            colors[(t * 4) + 3] =
                lerp(last_record.color.a as f32, next_record.color.a as f32, a) as u8;
        }

        colors
    }
}

/// AQW build: a gradient's texture, uniforms and bind group, shared by every tessellation of
/// every shape whose gradient has the same `GradientKey`. Stock Ruffle made all three per
/// gradient per tessellation: a long AQW session had thousands of tiny textures (each taking
/// an allocator page), views and bind groups, most of them identical. The texture and its
/// contents are the same as before, so shapes draw the same.
#[derive(Debug)]
pub struct SharedGradient {
    pub bind_group: wgpu::BindGroup,
    _texture_view: wgpu::TextureView,
    _uniforms: crate::arena::ArenaSlice,
}

impl SharedGradient {
    /// `uniforms` holds `key.uniform_bytes()`, uploaded.
    pub fn new(
        descriptors: &Descriptors,
        key: &GradientKey,
        uniforms: crate::arena::ArenaSlice,
    ) -> Self {
        let texture = descriptors.device.create_texture_with_data(
            &descriptors.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: GRADIENT_SIZE as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &key.colors[..],
        );
        let texture_view = texture.create_view(&Default::default());
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &descriptors.bind_layouts.gradient,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uniforms.buffer(),
                            offset: uniforms.offset(),
                            size: wgpu::BufferSize::new(
                                std::mem::size_of::<GradientUniforms>() as u64
                            ),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, true),
                        ),
                    },
                ],
                label: create_debug_label!("Gradient bindgroup").as_deref(),
            });
        Self {
            bind_group,
            _texture_view: texture_view,
            _uniforms: uniforms,
        }
    }
}

/// AQW build: the live `SharedGradient`s by key. Entries are weak: a gradient lives exactly as
/// long as some shape uses it, as before; a shape made later with the same gradient reuses it.
#[derive(Default)]
pub struct GradientCache {
    entries: FnvHashMap<GradientKey, Weak<SharedGradient>>,
    /// Entry count at which dead entries are next removed.
    next_prune: usize,
}

impl GradientCache {
    pub fn get(&self, key: &GradientKey) -> Option<Arc<SharedGradient>> {
        self.entries.get(key).and_then(Weak::upgrade)
    }

    pub fn insert(&mut self, key: GradientKey, gradient: &Arc<SharedGradient>) {
        // Dead entries keep their key (1 KB each); drop them now and then.
        if self.entries.len() >= self.next_prune.max(1024) {
            self.entries.retain(|_, weak| weak.strong_count() > 0);
            self.next_prune = self.entries.len() * 2;
        }
        self.entries.insert(key, Arc::downgrade(gradient));
    }

    /// For the perf log: how many gradients are live.
    pub fn live(&self) -> usize {
        self.entries
            .values()
            .filter(|weak| weak.strong_count() > 0)
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct BitmapBinds {
    pub bind_group: wgpu::BindGroup,
}

impl BitmapBinds {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        texture_view: wgpu::TextureView,
        label: Option<String>,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
            label: label.as_deref(),
        });
        Self { bind_group }
    }
}
