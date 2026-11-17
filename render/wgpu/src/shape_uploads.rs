//! AQW build: new shapes are uploaded through an encoder and a staging belt of their own (off
//! by default: `RUFFLE_SHAPE_UPLOADS`, a `shape-uploads` file next to the exe).
//!
//! A new shape's vertices, indices and gradient uniforms are copied into the shared arenas
//! (see `crate::arena`) by commands recorded into the picture's encoder, through the picture's
//! staging belt. So preparing a shape needed the picture's encoder, and ahead of it the wait
//! for the present thread to submit the last picture (`ensure_presented`). A shape tween
//! prepares shapes in every picture, while the stage is walked: the walk then began with that
//! wait, and a render thread recording the last picture has the encoder to itself.
//!
//! Here the copies are recorded into an encoder of their own, through a belt of their own,
//! behind one lock. Whoever submits the picture's command stream next takes that encoder
//! (`ShapeUploader::take`) and submits it in the same `queue.submit`, ahead of the stream's
//! own encoders. That order is right:
//! - a draw of a new shape is recorded after the shape was prepared, so it is submitted with
//!   its copies or after them, and the GPU runs it after them;
//! - the copies write ranges that nothing submitted with them or before them reads: a freed
//!   range waits `REUSE_DELAY_FRAMES` pictures before it is handed out again, and a new chunk
//!   is a buffer of its own.
//!
//! Nothing here waits for the present thread: the copies write staging buffers of this belt
//! only (new ones, or ones the GPU is done with) and record commands; a gradient's texture is
//! a new texture.
//!
//! The meshes are the ones `WgpuRenderBackend::upload_shape` makes without the switch: the
//! same bytes, allocated in the same order.

use crate::arena::{ArenaOutOfMemory, ArenaSlice, BufferArena};
use crate::buffer_builder::BufferBuilder;
use crate::descriptors::Descriptors;
use crate::mesh::{GradientCache, GradientKey, Mesh, PendingDraw, PendingDrawType, SharedGradient};
use ruffle_render::tessellator::{Bitmap, Gradient};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Whether new shapes are uploaded apart from the picture (`RUFFLE_SHAPE_UPLOADS`).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var_os("RUFFLE_SHAPE_UPLOADS").is_some();
        if enabled {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "New shapes are uploaded through their own encoder (RUFFLE_SHAPE_UPLOADS)"
            );
        }
        enabled
    })
}

/// The belt's chunk size: the picture's belt's.
const BELT_CHUNK: wgpu::BufferAddress = 65536;

/// A belt keeps every staging buffer it ever needed at once. With more than this uploaded
/// between two submits (a room loading, the quality changed: tens of MB), the belt is let go
/// of after the submit instead, and with it those buffers.
const BELT_KEEP: u64 = 4 << 20;

pub type SharedUploader = Arc<Mutex<ShapeUploader>>;

// With a render thread, shapes are prepared on one thread while the other takes the uploads
// (see `crate::render_thread`): the uploader must be `Send` by what it is made of.
const _: fn() = || {
    fn sendable<T: Send>() {}
    sendable::<ShapeUploader>();
};

pub struct ShapeUploader {
    descriptors: Arc<Descriptors>,
    /// The backend's arenas (shared handles).
    mesh_arena: BufferArena,
    uniform_arena: BufferArena,
    /// Instead of the backend's (see `GradientCache`).
    gradient_cache: GradientCache,
    belt: wgpu::util::StagingBelt,
    /// The copies recorded since the last `take`, if any.
    encoder: Option<wgpu::CommandEncoder>,
    /// Bytes written through `belt` since the last `take`.
    written: u64,
    /// For the log: bursts after which the belt was let go of.
    belts_dropped: u64,
}

impl ShapeUploader {
    pub fn new(
        descriptors: Arc<Descriptors>,
        mesh_arena: BufferArena,
        uniform_arena: BufferArena,
    ) -> Self {
        let belt = wgpu::util::StagingBelt::new(descriptors.device.clone(), BELT_CHUNK);
        Self {
            descriptors,
            mesh_arena,
            uniform_arena,
            gradient_cache: GradientCache::default(),
            belt,
            encoder: None,
            written: 0,
            belts_dropped: 0,
        }
    }

    pub fn shared(self) -> SharedUploader {
        Arc::new(Mutex::new(self))
    }

    /// The shared GPU gradients for a tessellated shape's gradients, making the ones not live
    /// yet (as `WgpuRenderBackend::shared_gradients`). `None` if the GPU is out of memory.
    pub fn gradients(&mut self, gradients: &[Gradient]) -> Option<Vec<Arc<SharedGradient>>> {
        let mut shared = Vec::with_capacity(gradients.len());
        for gradient in gradients {
            let key = GradientKey::of(gradient);
            if let Some(found) = self.gradient_cache.get(&key) {
                shared.push(found);
                continue;
            }
            let descriptors = &self.descriptors;
            let encoder = self.encoder.get_or_insert_with(|| {
                descriptors
                    .device
                    .create_command_encoder(&Default::default())
            });
            self.written += key.uniform_bytes().len() as u64;
            let uniforms = self
                .uniform_arena
                .alloc(
                    &descriptors.device,
                    &mut self.belt,
                    encoder,
                    key.uniform_bytes(),
                )
                .ok()?
                .expect("gradient uniforms aren't empty");
            let (made, out_of_memory) = crate::catch_out_of_memory(&descriptors.device, || {
                SharedGradient::new(descriptors, &key, uniforms)
            });
            if out_of_memory {
                return None;
            }
            let made = Arc::new(made);
            self.gradient_cache.insert(key, &made);
            shared.push(made);
        }
        Some(shared)
    }

    /// Room in the shared shape buffers for a shape's vertices and indices, and their upload
    /// (one attempt of `WgpuRenderBackend::alloc_shape_buffers`).
    #[allow(clippy::type_complexity)]
    pub fn buffers(
        &mut self,
        vertices: &[u8],
        indices: &[u8],
    ) -> Result<(Option<ArenaSlice>, Option<ArenaSlice>), ArenaOutOfMemory> {
        let device = &self.descriptors.device;
        let encoder = self
            .encoder
            .get_or_insert_with(|| device.create_command_encoder(&Default::default()));
        self.written += (vertices.len() + indices.len()) as u64;
        Ok((
            // (A multiple of both vertex sizes, see `CommandRenderer::render_shape`.)
            self.mesh_arena.alloc_at_multiple(
                device,
                &mut self.belt,
                encoder,
                vertices,
                crate::surface::commands::VERTEX_BASE_MULTIPLE,
            )?,
            self.mesh_arena
                .alloc(device, &mut self.belt, encoder, indices)?,
        ))
    }

    /// The copies recorded since the last call, to be submitted ahead of the command stream
    /// of whoever calls (see the module's notes), in the same `queue.submit`. The belt's
    /// buffers come back by themselves once that has happened.
    pub fn take(&mut self) -> Option<wgpu::CommandEncoder> {
        let encoder = self.encoder.take()?;
        if std::mem::take(&mut self.written) > BELT_KEEP {
            // Its buffers are freed when the GPU is done with the copies.
            self.belt.finish();
            self.belt =
                wgpu::util::StagingBelt::new(self.descriptors.device.clone(), BELT_CHUNK);
            self.belts_dropped += 1;
        } else {
            self.belt.finish_and_recall_on_submit(&encoder);
        }
        Some(encoder)
    }

    /// For the perf log: how many gradients are live, and how many bursts the belt was let
    /// go of after.
    pub fn stats(&self) -> (usize, u64) {
        (self.gradient_cache.live(), self.belts_dropped)
    }
}

pub fn lock(uploader: &Mutex<ShapeUploader>) -> MutexGuard<'_, ShapeUploader> {
    uploader
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `take` on a shared uploader, if there is one.
pub fn take(uploader: &Option<SharedUploader>) -> Option<wgpu::CommandEncoder> {
    lock(uploader.as_ref()?).take()
}

/// `encoders` with the uploads' encoder ahead of them, if there is one.
pub fn ahead_of(
    uploads: Option<wgpu::CommandEncoder>,
    mut encoders: Vec<wgpu::CommandEncoder>,
) -> Vec<wgpu::CommandEncoder> {
    if let Some(uploads) = uploads {
        encoders.insert(0, uploads);
    }
    encoders
}

/// What uploading a shape needs from whoever has the backend.
pub trait UploadHost {
    /// A bitmap fill's texture and bind group (`PendingDrawType::bitmap`).
    fn bitmap_fill(&mut self, bitmap: Bitmap) -> Option<PendingDrawType>;

    /// The GPU is out of memory: frees the idle pooled textures and lets the GPU release
    /// them, before one more attempt.
    fn make_room(&mut self);
}

/// Builds the draws of a tessellated shape and uploads its geometry: what
/// `WgpuRenderBackend::upload_shape` does, through `uploader`. The lock is held for the
/// uploads only (not while `host` is asked for anything).
pub fn upload_shape(
    uploader: &Mutex<ShapeUploader>,
    descriptors: &Descriptors,
    bounds: swf::Rectangle<swf::Twips>,
    lyon_mesh: ruffle_render::tessellator::Mesh,
    host: &mut dyn UploadHost,
) -> Mesh {
    let mut vertex_buffer = BufferBuilder::new(&descriptors.limits, 0);
    let mut index_buffer = BufferBuilder::new(&descriptors.limits, 0);
    // If the GPU is out of memory, free idle pooled textures and try once more; if it still
    // fails, the shape draws nothing (as without the switch).
    let mut gradients = None;
    for attempt in 0..2 {
        // (Its own statement: see the buffers below.)
        let made = lock(uploader).gradients(&lyon_mesh.gradients);
        if let Some(made) = made {
            gradients = Some(made);
            break;
        }
        if attempt == 0 {
            tracing::warn!(
                target: "ruffle_aqw_perf",
                "GPU out of memory for shape gradients; freeing idle textures and retrying"
            );
            host.make_room();
        }
    }
    let Some(gradients) = gradients else {
        tracing::warn!(
            target: "ruffle_aqw_perf",
            "GPU still out of memory for shape gradients; shape not drawn"
        );
        return Mesh {
            draws: Vec::new(),
            bounds,
            live: crate::perf_counters::LiveMesh::new(0),
            vertices: None,
            indices: None,
            gradients: Vec::new(),
            id: Mesh::new_id(),
            bitmaps: Vec::new(),
        };
    };

    let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
    let mut bitmaps = Vec::new();
    for draw in lyon_mesh.draws {
        if let Some(draw) =
            PendingDraw::new_with(draw, &mut vertex_buffer, &mut index_buffer, |bitmap| {
                host.bitmap_fill(bitmap)
            })
        {
            if let PendingDrawType::Bitmap { handle, .. } = &draw.draw_type
                && !bitmaps.contains(handle)
            {
                bitmaps.push(handle.clone());
            }
            draws.push(draw);
        }
    }

    let mut slices = None;
    for attempt in 0..2 {
        // (Its own statement: the lock must not be held while `host` is asked to make room,
        // which can wait for a thread that wants the lock.)
        let made = lock(uploader).buffers(vertex_buffer.bytes(), index_buffer.bytes());
        match made {
            Ok(made) => {
                slices = Some(made);
                break;
            }
            Err(_) if attempt == 0 => {
                tracing::warn!(
                    target: "ruffle_aqw_perf",
                    "GPU out of memory for shape buffers; freeing idle textures and retrying"
                );
                host.make_room();
            }
            Err(_) => {
                tracing::warn!(
                    target: "ruffle_aqw_perf",
                    "GPU still out of memory for shape buffers; shape not drawn"
                );
            }
        }
    }
    let (vertices, indices, draws) = match slices {
        Some((vertices, indices)) => (vertices, indices, draws),
        // Out of GPU memory even after freeing idle textures: the shape draws nothing.
        None => (None, None, Vec::new()),
    };
    let vertex_base = vertices.as_ref().map_or(0, |s| s.offset());
    let index_base = indices.as_ref().map_or(0, |s| s.offset());

    let draws = draws
        .into_iter()
        .map(|d| d.finish(vertex_base, index_base, &gradients))
        .collect();

    let bytes =
        vertices.as_ref().map_or(0, |s| s.size()) + indices.as_ref().map_or(0, |s| s.size());
    Mesh {
        draws,
        bounds,
        live: crate::perf_counters::LiveMesh::new(bytes),
        vertices,
        indices,
        gradients,
        id: Mesh::new_id(),
        bitmaps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device};
    use crate::mesh::as_mesh;
    use crate::surface::commands::VERTEX_BASE_MULTIPLE;
    use crate::target::TextureTarget;
    use ruffle_render::backend::RenderBackend;
    use ruffle_render::backend::null::NullBitmapSource;
    use ruffle_render::commands::{CommandHandler, CommandList};
    use ruffle_render::transform::Transform;

    /// A device for the tests, if this machine has one (`WGPU_BACKEND` picks the API); the
    /// tests pass without doing anything where there is none. With it, a guard that keeps the
    /// tests of this module one at a time, from making a device to letting go of it: three
    /// threads looking for adapters at once got stuck inside a graphics driver.
    fn descriptors() -> Option<(MutexGuard<'static, ()>, Arc<Descriptors>)> {
        static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
        let guard = ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let instance = create_wgpu_instance(backends, wgpu::BackendOptions::default(), None);
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backends,
            &instance,
            None,
            Default::default(),
        ))
        .ok()?;
        Some((
            guard,
            Arc::new(Descriptors::new(instance, adapter, device, queue)),
        ))
    }

    fn bytes(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// A readable arena (the backend's can't be copied from).
    fn arena() -> BufferArena {
        BufferArena::new(
            "Test geometry",
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_SRC,
            4,
            1 << 20,
        )
    }

    fn uploader(descriptors: &Arc<Descriptors>) -> ShapeUploader {
        let alignment = descriptors.limits.min_uniform_buffer_offset_alignment as u64;
        ShapeUploader::new(
            descriptors.clone(),
            arena(),
            BufferArena::new(
                "Test uniforms",
                wgpu::BufferUsages::UNIFORM,
                alignment,
                1 << 16,
            ),
        )
    }

    type Slices = (Option<ArenaSlice>, Option<ArenaSlice>);

    /// Records, into `encoder`, copies of what the slices hold.
    fn copy_out(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        slices: &[Slices],
    ) -> Vec<wgpu::Buffer> {
        let mut copies = Vec::new();
        for (vertices, indices) in slices {
            for slice in [vertices, indices].into_iter().flatten() {
                let copy = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: slice.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(
                    slice.buffer(),
                    slice.offset(),
                    &copy,
                    0,
                    slice.size(),
                );
                copies.push(copy);
            }
        }
        copies
    }

    fn read(device: &wgpu::Device, copies: &[wgpu::Buffer]) -> Vec<Vec<u8>> {
        for copy in copies {
            copy.slice(..)
                .map_async(wgpu::MapMode::Read, |result| result.expect("mapped"));
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
        copies
            .iter()
            .map(|copy| copy.slice(..).get_mapped_range().expect("mapped").to_vec())
            .collect()
    }

    fn places(slices: &[Slices]) -> Vec<Option<(u64, u64)>> {
        slices
            .iter()
            .flat_map(|(vertices, indices)| [vertices, indices])
            .map(|slice| slice.as_ref().map(|slice| (slice.offset(), slice.size())))
            .collect()
    }

    /// The uploads' own encoder, submitted ahead of a stream that reads the buffers, leaves
    /// in them what uploads recorded into the frame's encoder leave: the same bytes at the
    /// same places.
    #[test]
    fn uploads_apart_are_the_frames_uploads() {
        let Some((_one_at_a_time, descriptors)) = descriptors() else {
            return;
        };
        let device = &descriptors.device;
        let shapes = [
            (bytes(120, 1), bytes(12, 2)),
            (bytes(60_000, 3), bytes(30_000, 4)),
            (bytes(24, 5), Vec::new()),
            // (More than half a chunk: a buffer of its own.)
            (bytes(600_000, 6), bytes(4, 7)),
            (bytes(1200, 8), bytes(600, 9)),
        ];

        // As part of the frame (what `WgpuRenderBackend::alloc_shape_buffers` records).
        let frame_arena = arena();
        let mut belt = wgpu::util::StagingBelt::new(device.clone(), BELT_CHUNK);
        let mut encoder = device.create_command_encoder(&Default::default());
        let in_frame: Vec<Slices> = shapes
            .iter()
            .map(|(vertices, indices)| {
                (
                    frame_arena
                        .alloc_at_multiple(
                            device,
                            &mut belt,
                            &mut encoder,
                            vertices,
                            VERTEX_BASE_MULTIPLE,
                        )
                        .expect("memory"),
                    frame_arena
                        .alloc(device, &mut belt, &mut encoder, indices)
                        .expect("memory"),
                )
            })
            .collect();
        let frame_copies = copy_out(device, &mut encoder, &in_frame);
        belt.finish();
        descriptors.queue.submit([encoder.finish()]);
        belt.recall();

        // Apart: the stream that reads them is recorded first, and submitted after.
        let mut uploader = uploader(&descriptors);
        assert!(uploader.take().is_none(), "nothing was uploaded yet");
        let apart: Vec<Slices> = shapes
            .iter()
            .map(|(vertices, indices)| uploader.buffers(vertices, indices).expect("memory"))
            .collect();
        let mut picture = device.create_command_encoder(&Default::default());
        let apart_copies = copy_out(device, &mut picture, &apart);
        let uploads = uploader.take();
        assert!(uploads.is_some(), "the uploads were recorded");
        assert!(uploader.take().is_none(), "and are taken once");
        descriptors
            .queue
            .submit(crate::segments::finish_encoders(ahead_of(
                uploads,
                vec![picture],
            )));

        assert_eq!(places(&in_frame), places(&apart));
        let in_frame = read(device, &frame_copies);
        let apart = read(device, &apart_copies);
        assert_eq!(in_frame.len(), 9);
        assert!(in_frame == apart, "the same bytes");
        let sent = shapes
            .iter()
            .flat_map(|(vertices, indices)| [vertices, indices])
            .filter(|data| !data.is_empty());
        for (data, got) in sent.zip(&apart) {
            assert!(got.starts_with(data), "what was sent");
        }
    }

    /// The belt's staging buffers come back after each submit without being asked to, and
    /// after a burst they are let go of.
    #[test]
    fn the_belt_gives_back_its_buffers() {
        let Some((_one_at_a_time, descriptors)) = descriptors() else {
            return;
        };
        let device = &descriptors.device;
        let buffers = || {
            crate::release_gpu_memory(&descriptors);
            device.get_internal_counters().hal.buffers.read()
        };
        let mut uploader = uploader(&descriptors);
        let mut kept = Vec::new();
        let submit = |uploader: &mut ShapeUploader| {
            let uploads = uploader.take().expect("uploads");
            descriptors.queue.submit([uploads.finish()]);
        };

        // Small uploads, picture after picture: the same few staging buffers.
        kept.push(
            uploader
                .buffers(&bytes(6000, 1), &bytes(3000, 2))
                .expect("memory"),
        );
        submit(&mut uploader);
        let start = buffers();
        for round in 0..40 {
            kept.push(
                uploader
                    .buffers(&bytes(6000, round), &bytes(3000, round))
                    .expect("memory"),
            );
            submit(&mut uploader);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let steady = buffers();
        assert!(
            steady <= start + 3,
            "staging buffers pile up: {start} -> {steady}"
        );
        assert_eq!(uploader.stats().1, 0);

        // A burst (more than `BELT_KEEP`, in pieces of about a chunk each).
        for round in 0..100 {
            kept.push(
                uploader
                    .buffers(&bytes(60_000, round), &bytes(600, round))
                    .expect("memory"),
            );
        }
        submit(&mut uploader);
        assert_eq!(uploader.stats().1, 1, "the belt was let go of");
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let after = buffers();
        // (The geometry itself needed a few more chunks of the arena.)
        assert!(
            after <= steady + 10,
            "the burst's staging buffers stay: {steady} -> {after}"
        );
        // And it goes on working.
        kept.push(
            uploader
                .buffers(&bytes(6000, 1), &bytes(3000, 2))
                .expect("memory"),
        );
        submit(&mut uploader);
    }

    /// A gradient is made once and shared.
    #[test]
    fn gradients_are_shared() {
        let Some((_one_at_a_time, descriptors)) = descriptors() else {
            return;
        };
        let gradient = |color: u8| Gradient {
            gradient_type: ruffle_render::shape_utils::GradientType::Linear,
            repeat_mode: swf::GradientSpread::Pad,
            focal_point: swf::Fixed8::ZERO,
            interpolation: swf::GradientInterpolation::Rgb,
            records: vec![
                swf::GradientRecord {
                    ratio: 0,
                    color: swf::Color::from_rgb(0, 255),
                },
                swf::GradientRecord {
                    ratio: 255,
                    color: swf::Color {
                        r: color,
                        g: 0,
                        b: 0,
                        a: 255,
                    },
                },
            ],
        };
        let mut uploader = uploader(&descriptors);
        let first = uploader
            .gradients(&[gradient(10), gradient(20)])
            .expect("memory");
        assert!(uploader.take().is_some(), "their uniforms are uploaded");
        let again = uploader
            .gradients(&[gradient(20), gradient(10)])
            .expect("memory");
        assert!(uploader.take().is_none(), "nothing new to upload");
        assert!(Arc::ptr_eq(&first[0], &again[1]) && Arc::ptr_eq(&first[1], &again[0]));
        assert_eq!(uploader.stats().0, 2);
        drop((first, again));
        assert_eq!(
            uploader.stats().0,
            0,
            "a gradient lives as long as its shapes"
        );
    }

    fn square(id: u16, at: i32, size: i32, fill: swf::FillStyle) -> swf::Shape {
        let px = swf::Twips::from_pixels_i32;
        let bounds = swf::Rectangle {
            x_min: px(at),
            x_max: px(at + size),
            y_min: px(at),
            y_max: px(at + size),
        };
        let edge = |dx, dy| swf::ShapeRecord::StraightEdge {
            delta: swf::PointDelta::new(px(dx), px(dy)),
        };
        swf::Shape {
            version: 4,
            id,
            shape_bounds: bounds.clone(),
            edge_bounds: bounds,
            flags: swf::ShapeFlag::empty(),
            styles: swf::ShapeStyles {
                fill_styles: vec![fill],
                line_styles: vec![],
            },
            shape: vec![
                swf::ShapeRecord::StyleChange(Box::new(swf::StyleChangeData {
                    move_to: Some(swf::Point::new(px(at), px(at))),
                    fill_style_0: None,
                    fill_style_1: Some(1),
                    line_style: None,
                    new_styles: None,
                })),
                edge(size, 0),
                edge(0, size),
                edge(-size, 0),
                edge(0, -size),
            ],
        }
    }

    fn red() -> swf::FillStyle {
        swf::FillStyle::Color(swf::Color {
            r: 200,
            g: 40,
            b: 40,
            a: 255,
        })
    }

    fn blue_to_green(at: i32, size: i32) -> swf::FillStyle {
        // (A gradient's own square is 32768 twips wide, around its middle.)
        let px = swf::Twips::from_pixels_i32;
        let scale = swf::Fixed16::from_f64(f64::from(size) * 20.0 / 32768.0);
        swf::FillStyle::LinearGradient(swf::Gradient {
            matrix: swf::Matrix {
                tx: px(at + size / 2),
                ty: px(at + size / 2),
                ..swf::Matrix::scale(scale, scale)
            },
            spread: swf::GradientSpread::Pad,
            interpolation: swf::GradientInterpolation::Rgb,
            records: vec![
                swf::GradientRecord {
                    ratio: 0,
                    color: swf::Color {
                        r: 0,
                        g: 0,
                        b: 255,
                        a: 255,
                    },
                },
                swf::GradientRecord {
                    ratio: 255,
                    color: swf::Color {
                        r: 0,
                        g: 255,
                        b: 0,
                        a: 255,
                    },
                },
            ],
        })
    }

    fn backend(descriptors: &Arc<Descriptors>, apart: bool) -> WgpuRenderBackend<TextureTarget> {
        let target = TextureTarget::new(&descriptors.device, (64, 64)).expect("a target");
        let mut backend = WgpuRenderBackend::new(descriptors.clone(), target).expect("a backend");
        if apart {
            backend.upload_shapes_apart();
        }
        assert_eq!(backend.shape_uploader().is_some(), apart || enabled());
        backend
    }

    /// Registers `new` shapes, draws them after `shapes`, and returns what the new meshes
    /// are (as text) and the picture.
    fn picture(
        backend: &mut WgpuRenderBackend<TextureTarget>,
        shapes: &mut Vec<ruffle_render::backend::ShapeHandle>,
        new: &[swf::Shape],
    ) -> (String, image::RgbaImage) {
        let mut facts = String::new();
        for shape in new {
            let handle = backend.register_shape(shape.into(), &NullBitmapSource);
            let mesh = as_mesh(&handle);
            let place = |slice: &Option<ArenaSlice>| {
                slice.as_ref().map(|slice| (slice.offset(), slice.size()))
            };
            facts += &format!(
                "{:?} {:?} {} gradients;",
                place(&mesh.vertices),
                place(&mesh.indices),
                mesh.gradients.len()
            );
            for draw in &mesh.draws {
                let kind = match draw.draw_type {
                    crate::mesh::DrawType::Color => "color",
                    crate::mesh::DrawType::Gradient { .. } => "gradient",
                    crate::mesh::DrawType::Bitmap { .. } => "bitmap",
                };
                facts += &format!(
                    " {kind} {:?} {:?} {:?} {} {};",
                    draw.vertices,
                    draw.indices,
                    draw.index_format,
                    draw.num_indices,
                    draw.num_mask_indices
                );
            }
            shapes.push(handle);
        }
        let mut commands = CommandList::new();
        for shape in shapes.iter() {
            commands.render_shape(shape.clone(), Transform::default());
        }
        backend.submit_frame(swf::Color::WHITE, commands, Vec::new());
        (facts, backend.capture_frame().expect("a picture"))
    }

    /// Shapes uploaded apart are the same meshes and draw the same pictures as shapes
    /// uploaded with the frame: when drawn in the picture they were registered in, in later
    /// ones, and when their place in the buffers is one a freed shape had.
    #[test]
    fn a_shape_uploaded_apart_draws_the_same() {
        let Some((_one_at_a_time, descriptors)) = descriptors() else {
            return;
        };
        let mut with_frame = backend(&descriptors, false);
        let mut apart = backend(&descriptors, true);
        let (mut frame_shapes, mut apart_shapes) = (Vec::new(), Vec::new());
        let mut compare =
            |frame_shapes: &mut Vec<_>, apart_shapes: &mut Vec<_>, new: &[swf::Shape]| {
                let (frame_facts, frame_picture) = picture(&mut with_frame, frame_shapes, new);
                let (apart_facts, apart_picture) = picture(&mut apart, apart_shapes, new);
                assert_eq!(frame_facts, apart_facts);
                assert!(frame_picture == apart_picture, "the pictures differ");
                apart_picture
            };

        // Drawn in the picture they are registered in.
        let first = [
            square(1, 4, 20, red()),
            square(2, 30, 24, blue_to_green(30, 24)),
        ];
        let drawn = compare(&mut frame_shapes, &mut apart_shapes, &first);
        assert_eq!(drawn.get_pixel(1, 1).0, [255, 255, 255, 255]);
        assert_eq!(drawn.get_pixel(12, 12).0, [200, 40, 40, 255]);
        let middle = drawn.get_pixel(42, 42).0;
        assert!(
            middle[0] == 0 && middle[1] > 60 && middle[2] > 60,
            "a gradient: {middle:?}"
        );

        // More of them later, the first ones still drawn.
        let second = [
            square(3, 2, 10, blue_to_green(2, 10)),
            square(4, 50, 12, red()),
        ];
        let drawn = compare(&mut frame_shapes, &mut apart_shapes, &second);
        assert_eq!(drawn.get_pixel(56, 56).0, [200, 40, 40, 255]);

        // The first ones freed; pictures go by until their places are handed out again.
        frame_shapes.drain(..2);
        apart_shapes.drain(..2);
        for _ in 0..6 {
            let drawn = compare(&mut frame_shapes, &mut apart_shapes, &[]);
            assert_eq!(drawn.get_pixel(12, 12).0, [255, 255, 255, 255]);
        }
        let third = [
            square(5, 20, 20, red()),
            square(6, 34, 8, blue_to_green(34, 8)),
        ];
        let drawn = compare(&mut frame_shapes, &mut apart_shapes, &third);
        assert_eq!(drawn.get_pixel(25, 25).0, [200, 40, 40, 255]);
        assert_eq!(drawn.get_pixel(56, 56).0, [200, 40, 40, 255]);
    }
}
