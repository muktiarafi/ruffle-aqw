// Remove this when we decide on how to handle multithreaded rendering (especially on wasm)
#![allow(clippy::arc_with_non_send_sync)]

use crate::backend::ActiveFrame;
use crate::bitmaps::BitmapSamplers;
use crate::buffer_pool::{BufferPool, PoolEntry};
use crate::mesh::BitmapBinds;
use crate::pipelines::Pipelines;
use crate::target::{RenderTarget, SwapChainTarget};
use crate::utils::{
    BufferDimensions, capture_image, create_buffer_with_data, format_list, get_backend_names,
};
use bytemuck::{Pod, Zeroable};
use descriptors::Descriptors;
use enum_map::Enum;
use ruffle_render::backend::RawTexture;
use ruffle_render::bitmap::{BitmapHandle, BitmapHandleImpl, PixelRegion, SyncHandle};
use ruffle_render::shape_utils::GradientType;
use ruffle_render::tessellator::{Gradient as TessGradient, Vertex as TessVertex};
use std::any::Any;
use std::cell::{Cell, OnceCell};
use std::sync::Arc;
use swf::GradientSpread;
pub use wgpu;
pub use wgpu_profiler;

type Error = Box<dyn std::error::Error>;

#[macro_use]
pub mod utils;
/// Turu: [`segments::ThreadState`] and [`segments::swap`], for several players on one thread.
pub mod segments;
mod arena;

mod bitmaps;
mod context3d;
mod globals;
mod pipelines;
mod pixel_bender;
pub mod target;

pub mod backend;
mod blend;
mod buffer_builder;
mod buffer_pool;
#[cfg(feature = "clap")]
pub mod clap;
pub mod descriptors;
mod dynamic_transforms;
mod filters;
mod cache_batch;
mod packing;
mod layouts;
mod mesh;
mod shaders;
mod surface;

impl BitmapHandleImpl for Texture {}

pub fn as_texture(handle: &BitmapHandle) -> &Texture {
    <dyn Any>::downcast_ref(&*handle.0).unwrap()
}

pub fn raw_texture_as_texture(handle: &dyn RawTexture) -> &wgpu::Texture {
    <dyn Any>::downcast_ref(handle).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum MaskState {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Transforms {
    world_matrix: [[f32; 4]; 4],
    mult_color: [f32; 4],
    add_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosVertex {
    position: [f32; 2],
}

impl From<TessVertex> for PosVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosUvVertex {
    position: [f32; 2],
    uv: [f32; 3],
}

impl PosUvVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, t: f32) -> Self {
        let position = [x, y];
        let uv = [u, v, t];
        Self { position, uv }
    }

    pub fn from_tessellator(vertex: TessVertex, texture_matrix: &[[f32; 3]; 3]) -> Self {
        let position = [vertex.x, vertex.y];
        let mut uv = Self::transform_uv(texture_matrix, vertex.x, vertex.y);
        // AQW build: gradient and bitmap fills are white; their alpha is the coverage of a
        // Flash-style soft edge (1 everywhere else), passed in `uv.z`.
        uv[2] = f32::from(vertex.color.a) / 255.0;
        Self { position, uv }
    }

    fn transform_uv(matrix: &[[f32; 3]; 3], x: f32, y: f32) -> [f32; 3] {
        [
            matrix[0][0] * x + matrix[1][0] * y + matrix[2][0],
            matrix[0][1] * x + matrix[1][1] * y + matrix[2][1],
            1.0,
        ]
    }
}

/// AQW build: the color is 8-bit RGBA (`Unorm8x4`; the shader still reads 0..1 floats), as
/// SWF colors and Flash-style edge coverage are 8-bit anyway: 12 bytes a vertex instead of
/// 24, for most of AQW's shape geometry (solid fills).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosColorVertex {
    position: [f32; 2],
    color: [u8; 4],
}

impl From<TessVertex> for PosColorVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            color: [vertex.color.r, vertex.color.g, vertex.color.b, vertex.color.a],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GradientUniforms {
    focal_point: f32,
    interpolation: i32,
    shape: i32,
    repeat: i32,
}

impl From<TessGradient> for GradientUniforms {
    fn from(gradient: TessGradient) -> Self {
        Self {
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
            shape: match gradient.gradient_type {
                GradientType::Linear => 1,
                GradientType::Radial => 2,
                GradientType::Focal => 3,
            },
            repeat: match gradient.repeat_mode {
                GradientSpread::Pad => 1,
                GradientSpread::Reflect => 2,
                GradientSpread::Repeat => 3,
            },
        }
    }
}

#[derive(Debug)]
pub enum QueueSyncHandle {
    AlreadyCopied {
        index: Option<wgpu::SubmissionIndex>,
        buffer: PoolEntry<wgpu::Buffer, BufferDimensions>,
        copy_dimensions: BufferDimensions,
        descriptors: Arc<Descriptors>,
    },
    NotCopied {
        handle: BitmapHandle,
        copy_area: PixelRegion,
        descriptors: Arc<Descriptors>,
        pool: Arc<BufferPool<wgpu::Buffer, BufferDimensions>>,
    },
}

impl SyncHandle for QueueSyncHandle {}

impl QueueSyncHandle {
    pub fn capture<R, F: FnOnce(&[u8], u32) -> R>(
        self,
        with_rgba: F,
        frame: &mut ActiveFrame,
    ) -> R {
        match self {
            QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors,
            } => capture_image(
                &descriptors.device,
                &buffer,
                &copy_dimensions,
                index,
                with_rgba,
            ),
            QueueSyncHandle::NotCopied {
                handle,
                copy_area,
                descriptors,
                pool,
            } => {
                let texture = as_texture(&handle);

                let buffer_dimensions = BufferDimensions::new(
                    copy_area.width() as usize,
                    copy_area.height() as usize,
                    texture.texture.format(),
                );

                let buffer = pool.take(&descriptors, buffer_dimensions.clone());
                frame.command_encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy_area.x_min,
                            y: copy_area.y_min,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(buffer_dimensions.padded_bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: copy_area.width(),
                        height: copy_area.height(),
                        depth_or_array_layers: 1,
                    },
                );
                let index = frame.submit_direct(&descriptors);

                let image = capture_image(
                    &descriptors.device,
                    &buffer,
                    &buffer_dimensions,
                    Some(index),
                    with_rgba,
                );

                // After we've read pixels from a texture enough times, we'll store this buffer so that
                // future reads will be faster (it'll copy as part of the draw process instead)
                texture
                    .copy_count
                    .set(texture.copy_count.get().saturating_add(1));

                image
            }
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    /// AQW build: counted in the perf log until the bitmap is freed.
    live: Option<perf_counters::LiveBitmap>,
    pub(crate) texture: wgpu::Texture,
    repeating_linear: OnceCell<BitmapBinds>,
    repeating_nearest: OnceCell<BitmapBinds>,
    clamped_linear: OnceCell<BitmapBinds>,
    clamped_nearest: OnceCell<BitmapBinds>,
    copy_count: Cell<u8>,
}

impl Texture {
    pub fn bind_group(
        &self,
        repeating: bool,
        smoothed: bool,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        handle: BitmapHandle,
        samplers: &BitmapSamplers,
    ) -> &BitmapBinds {
        let bind = match (repeating, smoothed) {
            (true, true) => &self.repeating_linear,
            (true, false) => &self.repeating_nearest,
            (false, true) => &self.clamped_linear,
            (false, false) => &self.clamped_nearest,
        };
        bind.get_or_init(|| {
            BitmapBinds::new(
                device,
                layout,
                samplers.get_sampler(repeating, smoothed),
                self.texture.create_view(&Default::default()),
                create_debug_label!("Bitmap {:?} bind group (smoothed: {})", handle.0, smoothed),
            )
        })
    }
}

/// AQW build: turns the shared layer atlas on or off at runtime (for A/B comparisons in the
/// same scene; `RUFFLE_NO_ATLAS` turns it off for good). See `surface::commands`.
pub fn set_atlas_enabled(enabled: bool) {
    ATLAS_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// AQW build: turns batched cacheAsBitmap filters on or off at runtime (see `cache_batch`).
pub fn set_cache_batch_enabled(enabled: bool) {
    cache_batch::set_enabled(enabled);
}

pub fn cache_batch_enabled_now() -> bool {
    cache_batch::enabled()
}

pub fn atlas_enabled_now() -> bool {
    ATLAS_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) static ATLAS_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// AQW build: counts costly render operations so the desktop perf log can show what a
/// crowded scene spends its render passes on. Relaxed atomics; read and reset per log window.
pub mod perf_counters {
    use std::sync::atomic::{AtomicU32, Ordering};

    /// AQW build: running totals of time spent in a few expensive kinds of work (never reset),
    /// so the frontend can say what a slow frame spent its time on (`snapshot` before and
    /// after it). Times from several threads add up.
    pub mod timers {
        use std::sync::atomic::{AtomicU64, Ordering};

        pub const TEXTURE_CREATE: usize = 0;
        pub const UPLOAD: usize = 1;
        /// Finishing the frame's command buffers (waits for parallel encoding).
        pub const FINISH: usize = 2;
        pub const SUBMIT: usize = 3;
        pub const GPU_WAIT: usize = 4;
        /// Creating render pipelines (compiles shaders on DirectX 12).
        pub const PIPELINE: usize = 5;
        /// Tessellating and uploading new shapes (`register_shape*`), wherever it happens.
        pub const SHAPES: usize = 6;
        /// Recording the frame's cacheAsBitmap redraws and their filters.
        pub const CACHE_DRAWS: usize = 7;
        /// Recording the frame's main commands (the stage).
        pub const MAIN_DRAW: usize = 8;
        /// Parts of `SHAPES` in a batch (`register_shapes*`): tessellating on all cores, then
        /// uploading the meshes on this thread.
        pub const SHAPE_TESSELLATE: usize = 9;
        pub const SHAPE_UPLOAD: usize = 10;
        pub const LEN: usize = 11;
        pub const NAMES: [&str; LEN] = [
            "texture_create",
            "upload",
            "finish_encoding",
            "submit",
            "gpu_wait",
            "pipeline_create",
            "shape_prepare",
            "cache_draws",
            "main_draw",
            "of which tessellate",
            "of which shape_upload",
        ];

        static MICROS: [AtomicU64; LEN] = [const { AtomicU64::new(0) }; LEN];

        #[inline]
        pub fn time<R>(timer: usize, f: impl FnOnce() -> R) -> R {
            let start = std::time::Instant::now();
            let result = f();
            MICROS[timer].fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            result
        }

        pub fn add_micros(timer: usize, micros: u64) {
            MICROS[timer].fetch_add(micros, Ordering::Relaxed);
        }

        pub fn snapshot() -> [u64; LEN] {
            std::array::from_fn(|i| MICROS[i].load(Ordering::Relaxed))
        }

        /// Adds the time until it's dropped.
        pub struct Scoped(usize, std::time::Instant);

        impl Scoped {
            pub fn new(timer: usize) -> Self {
                Self(timer, std::time::Instant::now())
            }
        }

        impl Drop for Scoped {
            fn drop(&mut self) {
                MICROS[self.0].fetch_add(self.1.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }

    pub const RENDER_PASS: usize = 0;
    pub const BLEND_LAYER: usize = 1;
    pub const BLEND_TRIVIAL: usize = 2;
    pub const FILTER_GLOW: usize = 3;
    pub const FILTER_BLUR: usize = 4;
    pub const FILTER_SHADOW: usize = 5;
    pub const FILTER_OTHER: usize = 6;
    pub const OFFSCREEN: usize = 7;
    pub const MULTIPLY_FAST: usize = 8;
    pub const ALPHA_ERASE_SKIPPED: usize = 9;
    pub const PASS_MERGED: usize = 10;
    /// Followed by one counter per `ComplexBlend`, in declaration order.
    pub const COMPLEX_BASE: usize = 11;
    pub const SEGMENTS: usize = 20;
    /// Milliseconds spent waiting for the GPU to catch up (see `ActiveFrame::submit_for_target`).
    pub const GPU_WAIT_MS: usize = 21;
    /// Multisample resolves (see `CommandTarget::set_lazy_resolve`).
    pub const RESOLVES: usize = 22;
    /// Blend layers sized to their content (see `surface::commands::layer_rect_and_exact_size`).
    pub const BOUNDED_LAYERS: usize = 23;
    /// Shapes tessellated in parallel batches / one at a time (incl. re-tessellation at a new
    /// scale), and the main thread's time spent registering shapes, in microseconds.
    pub const SHAPES_BATCHED: usize = 24;
    pub const SHAPES_SINGLE: usize = 25;
    pub const SHAPE_REGISTER_US: usize = 26;
    /// Shapes re-tessellated for a new scale (`register_shape_with_scale`).
    pub const SHAPES_RESCALED: usize = 27;
    /// Measurements for a shared layer atlas (see `surface::commands::AtlasSim`): blend
    /// layers whose content has no blends, alpha masks or Stage3D inside (`atlas_leaf`, of
    /// which complex blends like Overlay: `atlas_leaf_complex`), bounded layers with such
    /// content inside (`atlas_nested`), full-size layers (`atlas_full`), leaf layers too big
    /// for a page (`atlas_too_big`); the leaves' exact area and today's rounded layer area in
    /// thousands of pixels; and how many 1024x2048 pages they would fill.
    pub const ATLAS_LEAF: usize = 28;
    pub const ATLAS_LEAF_COMPLEX: usize = 29;
    pub const ATLAS_NESTED: usize = 30;
    pub const ATLAS_FULL: usize = 31;
    pub const ATLAS_TOO_BIG: usize = 32;
    pub const ATLAS_KPX: usize = 33;
    pub const LAYER_KPX: usize = 34;
    pub const ATLAS_PAGES: usize = 35;
    /// Textures taken from the layer pool / of those, newly created (no idle one of that
    /// size); the same for the cacheAsBitmap pool.
    pub const LAYER_POOL_TAKE: usize = 36;
    pub const LAYER_POOL_NEW: usize = 37;
    pub const CACHE_POOL_TAKE: usize = 38;
    pub const CACHE_POOL_NEW: usize = 39;
    /// What the frame's glow/blur/shadow filters look like, to design batching them: glows
    /// (and shadows) by quality (blur passes), inner/knockout ones, blur sizes (largest of x/y
    /// in px: <=8, <=24, more), and the cacheAsBitmap redraws (count, thousands of pixels).
    pub const GLOW_Q1: usize = 40;
    pub const GLOW_Q2: usize = 41;
    pub const GLOW_Q3: usize = 42;
    pub const GLOW_INNER_KNOCKOUT: usize = 43;
    pub const BLUR_SMALL: usize = 44;
    pub const BLUR_MID: usize = 45;
    pub const BLUR_BIG: usize = 46;
    pub const CAB_ENTRIES: usize = 47;
    pub const CAB_KPX: usize = 48;
    /// cacheAsBitmap redraws done in batches, and the batch pages drawn (see `cache_batch`).
    pub const CAB_BATCHED: usize = 49;
    pub const CAB_PAGES: usize = 50;
    /// Area of the atlas pages drawn (blend layers and batched filters) and of the slots used in
    /// them, in thousands of pixels: how tightly they're packed (see `crate::packing`).
    pub const PAGE_KPX: usize = 51;
    pub const SLOT_KPX: usize = 52;
    const LEN: usize = 53;
    pub const NAMES: [&str; LEN] = [
        "passes",
        "layer",
        "add/screen",
        "glow",
        "blur",
        "shadow",
        "filter_other",
        "offscreen",
        "multiply_fast",
        "alpha/erase_skipped",
        "merged_passes",
        "multiply_slow",
        "lighten",
        "darken",
        "difference",
        "invert",
        "alpha",
        "erase",
        "overlay",
        "hardlight",
        "segments",
        "gpu_wait_ms",
        "resolves",
        "bounded_layers",
        "shapes_batched",
        "shapes_single",
        "shape_register_us",
        "shapes_rescaled",
        "atlas_leaf",
        "atlas_leaf_complex",
        "atlas_nested",
        "atlas_full",
        "atlas_too_big",
        "atlas_kpx",
        "layer_kpx",
        "atlas_pages",
        "layer_pool_take",
        "layer_pool_new",
        "cache_pool_take",
        "cache_pool_new",
        "glow_q1",
        "glow_q2",
        "glow_q3",
        "glow_inner_knockout",
        "blur_small",
        "blur_mid",
        "blur_big",
        "cab_entries",
        "cab_kpx",
        "cab_batched",
        "cab_pages",
        "page_kpx",
        "slot_kpx",
    ];

    static COUNTS: [AtomicU32; LEN] = [const { AtomicU32::new(0) }; LEN];

    /// Render passes recorded since startup (never reset), for segment cutting.
    static PASSES_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[inline]
    pub fn add(counter: usize, amount: u32) {
        COUNTS[counter].fetch_add(amount, Ordering::Relaxed);
    }

    pub fn inc(counter: usize) {
        COUNTS[counter].fetch_add(1, Ordering::Relaxed);
        if counter == RENDER_PASS {
            PASSES_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }

    static LIVE_MESHES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    static LIVE_MESH_BYTES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    static LIVE_BITMAPS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    static LIVE_BITMAP_BYTES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

    /// Counts a live shape mesh (and its vertex/index buffer bytes) until dropped.
    #[derive(Debug)]
    pub struct LiveMesh(i64);

    impl LiveMesh {
        pub fn new(bytes: u64) -> Self {
            let bytes = bytes as i64;
            LIVE_MESHES.fetch_add(1, Ordering::Relaxed);
            LIVE_MESH_BYTES.fetch_add(bytes, Ordering::Relaxed);
            Self(bytes)
        }
    }

    impl Drop for LiveMesh {
        fn drop(&mut self) {
            LIVE_MESHES.fetch_sub(1, Ordering::Relaxed);
            LIVE_MESH_BYTES.fetch_sub(self.0, Ordering::Relaxed);
        }
    }

    /// Counts a live bitmap texture (and its approximate bytes) until dropped.
    #[derive(Debug)]
    pub struct LiveBitmap(i64);

    impl LiveBitmap {
        pub fn new(texture: &wgpu::Texture) -> Self {
            let bytes = texture.width() as i64 * texture.height() as i64 * 4;
            LIVE_BITMAPS.fetch_add(1, Ordering::Relaxed);
            LIVE_BITMAP_BYTES.fetch_add(bytes, Ordering::Relaxed);
            Self(bytes)
        }
    }

    impl Drop for LiveBitmap {
        fn drop(&mut self) {
            LIVE_BITMAPS.fetch_sub(1, Ordering::Relaxed);
            LIVE_BITMAP_BYTES.fetch_sub(self.0, Ordering::Relaxed);
        }
    }

    /// (live meshes, mesh bytes, live bitmaps, bitmap bytes)
    pub fn live() -> (i64, i64, i64, i64) {
        (
            LIVE_MESHES.load(Ordering::Relaxed),
            LIVE_MESH_BYTES.load(Ordering::Relaxed),
            LIVE_BITMAPS.load(Ordering::Relaxed),
            LIVE_BITMAP_BYTES.load(Ordering::Relaxed),
        )
    }

    /// Draw commands recorded into render passes since startup (never reset), so segment
    /// cutting can follow the work and not just the pass count (atlas pages hold many draws).
    static DRAWS_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub fn add_draws(draws: usize) {
        DRAWS_TOTAL.fetch_add(draws as u64, Ordering::Relaxed);
    }

    pub fn total_draws() -> u64 {
        DRAWS_TOTAL.load(Ordering::Relaxed)
    }

    pub fn total_passes() -> u64 {
        PASSES_TOTAL.load(Ordering::Relaxed)
    }

    /// Counts an atlas page of `page` pixels holding `slots` (width, height).
    pub fn count_page(page: (u32, u32), slots: impl Iterator<Item = (u32, u32)>) {
        let kpx = |w: u32, h: u32| ((w as u64 * h as u64).div_ceil(1000)) as u32;
        add(PAGE_KPX, kpx(page.0, page.1));
        add(SLOT_KPX, slots.map(|(w, h)| kpx(w, h)).sum());
    }

    pub fn take() -> [u32; LEN] {
        std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed))
    }
}

/// AQW build: whether the error scope caught an out-of-memory error (the scope must have been
/// pushed with `ErrorFilter::OutOfMemory`).
/// On native wgpu the popped scope's future is ready immediately, so it's polled once right
/// here (not with an executor's `block_on`, which panics when called from inside another
/// executor, e.g. while a loader task runs).
pub(crate) fn aqw_out_of_memory(scope: wgpu::ErrorScopeGuard) -> bool {
    let out_of_memory = pop_scope(scope).is_some();
    if out_of_memory {
        gpu_oom::note();
    }
    out_of_memory
}

fn pop_scope(scope: wgpu::ErrorScopeGuard) -> Option<wgpu::Error> {
    let future = std::pin::pin!(scope.pop());
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future, &mut context) {
        std::task::Poll::Ready(error) => error,
        std::task::Poll::Pending => None,
    }
}

/// AQW build: runs `f` (which creates GPU resources) and reports whether the GPU ran out of
/// memory. Also catches the validation errors `f` itself then causes by using what failed (e.g.
/// creating a view of a texture that couldn't be allocated); a validation error without an
/// out-of-memory error is a bug and stays fatal.
pub(crate) fn catch_out_of_memory<R>(device: &wgpu::Device, f: impl FnOnce() -> R) -> (R, bool) {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let result = f();
    let out_of_memory = aqw_out_of_memory(memory);
    if let Some(error) = pop_scope(validation)
        && !out_of_memory
    {
        panic!("wgpu error: {error}");
    }
    (result, out_of_memory)
}

/// AQW build: after dropping textures to make room, waits until the GPU is done with them so
/// their memory is really released (including resources only kept alive by finished command
/// buffers waiting on wgpu's background drop thread).
pub(crate) fn release_gpu_memory(descriptors: &descriptors::Descriptors) {
    let _ = descriptors
        .device
        .poll(wgpu::PollType::wait_indefinitely());
    #[cfg(not(target_family = "wasm"))]
    wgpu_core::aqw_flush_encoder_drops();
}

/// AQW build: running out of GPU memory is not fatal.
///
/// wgpu reports errors to a handler that, by default, panics on any error. On small GPUs (2 GB
/// laptop cards in crowded AQW rooms) an allocation that doesn't fit gave "Out of Memory" and
/// crashed. The main allocations free idle textures and retry (`create_texture_making_room`,
/// `alloc_shape_buffers`, gradients, the texture pools); for everything else this handler logs
/// the error and asks the backend to free idle textures before the next frame. The failed
/// object is invalid, so for a few seconds after an out-of-memory error the validation errors of
/// work that used it are logged instead of fatal too: at worst a frame misses something. Any
/// other error is still fatal, as before.
pub(crate) mod gpu_oom {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    static PENDING: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    /// Why the device was lost, once it was.
    static LOST: Mutex<Option<String>> = Mutex::new(None);

    /// Why the GPU device was lost ("Unknown: ..."), or `None` while it works. Once lost, it
    /// stays lost: frames aren't drawn and GPU errors are tolerated.
    pub fn device_lost() -> Option<String> {
        LOST.lock().ok().and_then(|lost| lost.clone())
    }

    /// How long after an out-of-memory error its follow-up validation errors are tolerated.
    const TOLERANCE: Duration = Duration::from_secs(3);

    pub fn install_error_handler(device: &wgpu::Device) {
        // AQW build: the GPU can drop the device (wgpu loses it when the driver runs out of
        // memory, on phones with 4 GB). Nothing drawn after that shows, but the game and its
        // scripts keep running: remember it, so the app stops drawing and tells the user, and
        // tolerate the errors of the work still sent to the dead device.
        device.set_device_lost_callback(|reason, message| {
            // Inside wgpu, possibly with its locks held: only remember it.
            if let Ok(mut lost) = LOST.lock()
                && lost.is_none()
            {
                *lost = Some(format!("{reason:?}: {message}"));
            }
        });
        device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
            if device_lost().is_some() {
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 50 {
                    tracing::warn!(target: "ruffle_aqw_perf", "GPU error tolerated (device lost): {error}");
                }
                return;
            }
            let recent = LAST
                .lock()
                .ok()
                .and_then(|last| *last)
                .is_some_and(|last| last.elapsed() < TOLERANCE);
            match &error {
                wgpu::Error::OutOfMemory { .. } => {
                    PENDING.store(true, Ordering::Relaxed);
                    if let Ok(mut last) = LAST.lock() {
                        *last = Some(Instant::now());
                    }
                }
                wgpu::Error::Validation { .. } if recent => {}
                _ => panic!("wgpu error: {error}"),
            }
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 50 {
                tracing::warn!(target: "ruffle_aqw_perf", "GPU error tolerated (out of memory): {error}");
            }
        }));
    }

    /// An out-of-memory error was caught by an error scope: tolerate its follow-up errors too.
    pub fn note() {
        PENDING.store(true, Ordering::Relaxed);
        if let Ok(mut last) = LAST.lock() {
            *last = Some(Instant::now());
        }
    }

    /// Whether an out-of-memory error happened since the last call.
    pub fn take_pending() -> bool {
        PENDING.swap(false, Ordering::Relaxed)
    }
}

/// AQW build: why the GPU device was lost, or `None` while it works (see `gpu_oom`). After a
/// loss the renderer draws nothing, and the app should neither draw its own UI on that device
/// nor wait on it.
pub fn device_lost() -> Option<String> {
    gpu_oom::device_lost()
}

/// AQW build: while `true` (the GPU's memory is nearly full), the texture pools keep at most
/// 64 MB of spare textures instead of 512 MB.
pub fn set_memory_pressure(on: bool) {
    buffer_pool::MEMORY_PRESSURE.store(on, std::sync::atomic::Ordering::Relaxed);
}

static TEXTURE_KINDS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// AQW build: turns `texture_kind` off (a `no-transient-memory` file next to the exe, at
/// startup), for A/B comparisons.
pub fn set_transient_textures_enabled(enabled: bool) {
    TEXTURE_KINDS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn transient_textures_enabled() -> bool {
    TEXTURE_KINDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// AQW build: GPU memory blocks of their own for textures that come and go all the time
/// (pooled layers): see `wgpu_core::aqw_texture_kind`.
pub(crate) const TRANSIENT_TEXTURES: u8 = 1;
/// AQW build: the same for textures that live as long as some object (cacheAsBitmap).
pub(crate) const CACHED_TEXTURES: u8 = 2;

/// AQW build: textures created inside `f` get GPU memory blocks of their own for their `kind`
/// (`TRANSIENT_TEXTURES` or `CACHED_TEXTURES`).
pub(crate) fn texture_kind<R>(kind: u8, f: impl FnOnce() -> R) -> R {
    #[cfg(not(target_family = "wasm"))]
    if transient_textures_enabled() {
        return wgpu_core::aqw_texture_kind(kind, f);
    }
    let _ = kind;
    f()
}
