use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{BufferPool, TexturePool};
use crate::context3d::WgpuContext3D;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::{GradientCache, GradientKey, Mesh, PendingDraw, SharedGradient};
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::{LayerRef, Surface};
use crate::target::{MaybeOwnedBuffer, TextureTarget};
use crate::target::{RenderTargetFrame, TextureBufferInfo};
use crate::utils::{BufferDimensions, run_copy_pipeline};
use crate::{
    Descriptors, Error, QueueSyncHandle, RenderTarget, SwapChainTarget, Texture, as_texture,
    format_list, get_backend_names,
};
use image::imageops::FilterType;
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
};
use ruffle_render::backend::{NO_STRETCH, RenderBackend, ShapeHandle, Stretch, ViewportDimensions};
use ruffle_render::bitmap::{
    Bitmap, BitmapFormat, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error as BitmapError;
use ruffle_render::filters::Filter;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use ruffle_render::tessellator::ShapeTessellator;
use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::Color;
use tracing::instrument;
use wgpu::SubmissionIndex;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Creates a wgpu instance with Ruffle's required configuration.
///
/// This disables indirect call validation because wgpu's validation runs a compute
/// shader that uses `array<u32>`, which requires the `DYNAMIC_ARRAY_SIZE` feature.
/// However, wgpu runs this shader without first checking if the device supports
/// that feature, causing device creation to fail on GPUs that lack it.
/// Since Ruffle doesn't use indirect draws, disabling this validation has no
/// functional impact.
///
/// See <https://github.com/gfx-rs/wgpu/issues/8799>
pub fn create_wgpu_instance(
    backends: wgpu::Backends,
    backend_options: wgpu::BackendOptions,
    display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
) -> wgpu::Instance {
    let descriptor = match display {
        Some(display) => wgpu::InstanceDescriptor::new_with_display_handle(display),
        None => wgpu::InstanceDescriptor::new_without_display_handle(),
    };
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        // AQW build: no debug names on driver objects (only graphics debuggers show them, and
        // naming every per-pass command list/buffer showed up in DirectX 12 profiles), unless
        // this is a `render_debug_labels` build.
        flags: wgpu::InstanceFlags::default()
            .difference(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
            .union(if cfg!(feature = "render_debug_labels") {
                wgpu::InstanceFlags::empty()
            } else {
                wgpu::InstanceFlags::DISCARD_HAL_LABELS
            })
            .with_env(),
        backend_options,
        // AQW build: `RUFFLE_TEST_GPU_BUDGET_PERCENT=7` makes allocations fail once video memory
        // use passes that share of the GPU's budget (7% of a 32 GB card ~ a 2 GB laptop GPU),
        // to test running out of GPU memory.
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds {
            for_resource_creation: std::env::var("RUFFLE_TEST_GPU_BUDGET_PERCENT")
                .ok()
                .and_then(|v| v.parse().ok()),
            for_device_loss: None,
        },
        ..descriptor
    })
}

/// AQW build: returns the graphics-memory budget and current usage exposed by the native API.
/// (Not on Windows, where `video_memory` asks DXGI.)
#[cfg(not(windows))]
pub fn adapter_memory(adapter: &wgpu::Adapter) -> Option<(u64, u64)> {
    unsafe {
        #[cfg(target_os = "linux")]
        {
            return adapter
                .as_hal::<wgpu::hal::api::Vulkan>()
                .and_then(|adapter| adapter.memory_budget());
        }
        #[cfg(target_os = "macos")]
        {
            return adapter
                .as_hal::<wgpu::hal::api::Metal>()
                .and_then(|adapter| adapter.memory_budget());
        }
        #[allow(unreachable_code)]
        None
    }
}

pub struct WgpuRenderBackend<T: RenderTarget> {
    pub(crate) descriptors: Arc<Descriptors>,
    /// AQW build: shape vertices/indices and gradient uniforms (see `crate::arena`). Per
    /// backend, not in the shared `Descriptors`: freed ranges are reused after this backend's
    /// own frames have finished on the GPU.
    mesh_arena: crate::arena::BufferArena,
    uniform_arena: crate::arena::BufferArena,
    /// AQW build: gradient textures shared between shapes (see `GradientCache`).
    gradient_cache: GradientCache,
    /// AQW build: shapes being tessellated in the background (see `start_shapes`).
    background_shapes: BackgroundShapes,
    target: T,
    surface: Surface,
    meshes: Vec<Mesh>,
    shape_tessellator: ShapeTessellator,
    // This is currently unused - we just store it to report in
    // `get_viewport_dimensions`
    viewport_scale_factor: f64,
    texture_pool: TexturePool,
    offscreen_texture_pool: TexturePool,
    /// AQW build: freed cacheAsBitmap textures, reused at the same size (see `TextureRecycler`).
    cab_recycler: Arc<crate::TextureRecycler>,
    pub(crate) offscreen_buffer_pool: Arc<BufferPool<wgpu::Buffer, BufferDimensions>>,
    dynamic_transforms: DynamicTransforms,
    active_frame: ActiveFrame,
    profiler: GpuProfiler,
}

impl<T: RenderTarget> WgpuRenderBackend<T> {
    /// AQW build, for the perf log: live shape/bitmap counts and the GPU allocator totals.
    pub fn aqw_memory_report(&self) -> String {
        const MB: f64 = 1024.0 * 1024.0;
        let (meshes, mesh_bytes, bitmaps, bitmap_bytes) = crate::perf_counters::live();
        let (cabs, cab_live_bytes) = crate::perf_counters::live_cab();
        let gpu = self
            .descriptors
            .device
            .generate_allocator_report()
            .map(|r| {
                // What the GPU memory is, by allocation label with numbers removed (labels need
                // the `render_debug_labels` feature; without it: unlabeled textures vs buffers).
                let mut by_label: std::collections::HashMap<String, (u32, u64)> =
                    Default::default();
                for allocation in &r.allocations {
                    let label: String = allocation
                        .name
                        .chars()
                        .map(|c| if c.is_ascii_digit() { '#' } else { c })
                        .collect();
                    let entry = by_label.entry(label).or_default();
                    entry.0 += 1;
                    entry.1 += allocation.size;
                }
                let mut by_label: Vec<_> = by_label.into_iter().collect();
                by_label.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
                let top = by_label
                    .iter()
                    .take(10)
                    .map(|(label, (count, bytes))| {
                        format!("{label}: {count} ({:.0} MB)", *bytes as f64 / MB)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                // Blocks by texture kind (each kind has its own allocator): count, used/size.
                // (An empty block counts as "other".)
                let mut kinds = [(0u32, 0u64, 0u64); 3];
                for block in &r.blocks {
                    let allocations = r.allocations.get(block.allocations.clone()).unwrap_or(&[]);
                    let kind = match allocations.first().map(|a| a.name.as_str()) {
                        Some(name) if name.starts_with("Transient") => 0,
                        Some(name) if name.starts_with("Cached") => 1,
                        _ => 2,
                    };
                    kinds[kind].0 += 1;
                    kinds[kind].1 += allocations.iter().map(|a| a.size).sum::<u64>();
                    kinds[kind].2 += block.size;
                }
                let kinds = ["pooled", "cached", "other"]
                    .iter()
                    .zip(kinds)
                    .map(|(name, (blocks, used, size))| {
                        format!(
                            "{name}: {blocks} blocks, {:.0}/{:.0} MB",
                            used as f64 / MB,
                            size as f64 / MB
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "{:.0} MB used / {:.0} MB reserved in {} allocations, {} blocks ({kinds}) [by label: {top}]",
                    r.total_allocated_bytes as f64 / MB,
                    r.total_reserved_bytes as f64 / MB,
                    r.allocations.len(),
                    r.blocks.len(),
                )
            })
            .unwrap_or_else(|| "n/a".to_string());
        let (pool_sizes, pool_textures, pool_bytes) = self.texture_pool.stats();
        let (mesh_chunks, mesh_reserved, mesh_used) = self.mesh_arena.stats();
        let (uniform_chunks, uniform_reserved, uniform_used) =
            self.uniform_arena.stats();
        let (cab_sizes, cab_textures, cab_bytes) = self.offscreen_texture_pool.stats();
        let gradient_ramps = self.gradient_cache.live();
        let hal = self.descriptors.device.get_internal_counters().hal;
        let gpu = format!(
            "{gpu}; vulkan objects: {} buffers, {} textures, {} views, {} bind groups, {} samplers, {} command encoders, {} pipelines, {} memory allocations",
            hal.buffers.read(),
            hal.textures.read(),
            hal.texture_views.read(),
            hal.bind_groups.read(),
            hal.samplers.read(),
            hal.command_encoders.read(),
            hal.render_pipelines.read(),
            hal.memory_allocations.read(),
        );
        format!(
            "{meshes} shapes ({:.0} MB), {bitmaps} bitmaps ({:.0} MB; cacheAsBitmap {cabs}, {:.0} MB), shape buffers: {mesh_chunks} geometry chunks ({:.0}/{:.0} MB used), {uniform_chunks} uniform chunks ({:.1}/{:.0} MB), {gradient_ramps} gradient ramps, layer pool: {pool_textures} textures in {pool_sizes} sizes ({:.0} MB), cache pool: {cab_textures} textures in {cab_sizes} sizes ({:.0} MB); gpu allocator: {gpu}",
            mesh_bytes as f64 / MB,
            bitmap_bytes as f64 / MB,
            cab_live_bytes as f64 / MB,
            mesh_used as f64 / MB,
            mesh_reserved as f64 / MB,
            uniform_used as f64 / MB,
            uniform_reserved as f64 / MB,
            pool_bytes as f64 / MB,
            cab_bytes as f64 / MB,
        )
    }
}

impl WgpuRenderBackend<SwapChainTarget> {
    #[cfg(target_family = "wasm")]
    pub async fn for_canvas(
        canvas: web_sys::HtmlCanvasElement,
        webgpu: bool,
    ) -> Result<Self, Error> {
        let backends = if webgpu {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::GL
        };
        let instance = create_wgpu_instance(
            backends,
            wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    // See <https://github.com/gfx-rs/wgpu/releases/tag/v25.0.0>
                    fence_behavior: wgpu::GlFenceBehavior::AutoFinish,
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        );
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;
        let (adapter, device, queue) = request_adapter_and_device(
            backends,
            &instance,
            Some(&surface),
            wgpu::PowerPreference::HighPerformance,
        )
        .await?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target =
            SwapChainTarget::new(surface, &descriptors.adapter, (1, 1), &descriptors.device);
        Self::new(Arc::new(descriptors), target)
    }

    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    ///
    /// Since wgpu 29, a display handle is needed at instance creation time:
    /// pass one via `display`, or make sure the `window` target carries a raw
    /// display handle (note that `SurfaceTargetUnsafe::from_window` does not
    /// provide one). Prefer passing `display` - some backends (e.g. GL via
    /// EGL) select their platform when the instance is created, before the
    /// target's display handle is seen.
    #[cfg(not(target_family = "wasm"))]
    pub unsafe fn for_window_unsafe(
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
        display: Option<Box<dyn wgpu::wgt::WgpuHasDisplayHandle>>,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), display);
        let surface = unsafe { instance.create_surface_unsafe(window)? };
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            Some(&surface),
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target = SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device);
        Self::new(Arc::new(descriptors), target)
    }

    /// # Safety
    ///  See [`wgpu::SurfaceTargetUnsafe`] variants for safety requirements.
    #[cfg(not(target_family = "wasm"))]
    pub unsafe fn recreate_surface_unsafe(
        &mut self,
        window: wgpu::SurfaceTargetUnsafe,
        size: (u32, u32),
    ) -> Result<(), Error> {
        let descriptors = &self.descriptors;
        let surface = unsafe { descriptors.wgpu_instance.create_surface_unsafe(window)? };
        self.target =
            SwapChainTarget::new(surface, &descriptors.adapter, size, &descriptors.device);
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
impl WgpuRenderBackend<crate::target::TextureTarget> {
    pub fn for_offscreen(
        size: (u32, u32),
        backend: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Error> {
        if wgpu::Backends::SECONDARY.contains(backend) {
            tracing::warn!(
                "{} graphics backend support may not be fully supported.",
                format_list(&get_backend_names(backend), "and")
            );
        }
        let instance = create_wgpu_instance(backend, wgpu::BackendOptions::default(), None);
        let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
            backend,
            &instance,
            None,
            power_preference,
        ))?;
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let target = crate::target::TextureTarget::new(&descriptors.device, size)?;
        Self::new(Arc::new(descriptors), target)
    }

    pub fn capture_frame(&self) -> Option<image::RgbaImage> {
        use crate::utils::buffer_to_image;
        if let Some(buffer) = &self.target.buffer {
            let (buffer, dimensions) = buffer.buffer.inner();
            Some(buffer_to_image(
                &self.descriptors.device,
                buffer,
                dimensions,
                None,
                self.target.size,
            ))
        } else {
            None
        }
    }
}

impl<T: RenderTarget> WgpuRenderBackend<T> {
    pub fn new(descriptors: Arc<Descriptors>, target: T) -> Result<Self, Error> {
        if target.width() > descriptors.limits.max_texture_dimension_2d
            || target.height() > descriptors.limits.max_texture_dimension_2d
        {
            return Err(format!(
                "Render target texture cannot be larger than {}px on either dimension (requested {} x {})",
                descriptors.limits.max_texture_dimension_2d,
                target.width(),
                target.height()
            )
                .into());
        }

        // AQW build: on DirectX 12, build the render pipelines in the background now.
        descriptors.warm_up(target.format());

        let surface = Surface::new(
            &descriptors,
            StageQuality::Low,
            target.width(),
            target.height(),
            target.format(),
        );

        let offscreen_buffer_pool = BufferPool::new(Box::new(
            |descriptors: &Descriptors, dimensions: &BufferDimensions| {
                descriptors.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: dimensions.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            },
        ));

        let transforms = DynamicTransforms::new(&descriptors);
        let active_frame = ActiveFrame::new(&descriptors);

        let profiler_settings = GpuProfilerSettings {
            enable_timer_queries: cfg!(feature = "profile-with-tracy"),
            enable_debug_groups: cfg!(feature = "render_debug_labels"),
            ..Default::default()
        };
        #[cfg(feature = "profile-with-tracy")]
        let profiler = GpuProfiler::new_with_tracy_client(
            profiler_settings,
            descriptors.backend,
            &descriptors.device,
            &descriptors.queue,
        )?;
        #[cfg(not(feature = "profile-with-tracy"))]
        let profiler = GpuProfiler::new(&descriptors.device, profiler_settings)?;

        let uniform_alignment = descriptors.limits.min_uniform_buffer_offset_alignment as u64;
        Ok(Self {
            descriptors,
            target,
            surface,
            meshes: Vec::new(),
            shape_tessellator: ShapeTessellator::new(),
            viewport_scale_factor: 1.0,
            texture_pool: TexturePool::new_counted(crate::perf_counters::LAYER_POOL_TAKE, crate::perf_counters::LAYER_POOL_NEW),
            offscreen_texture_pool: TexturePool::new_counted(crate::perf_counters::CACHE_POOL_TAKE, crate::perf_counters::CACHE_POOL_NEW),
            cab_recycler: Default::default(),
            mesh_arena: crate::arena::BufferArena::new(
                "Shape geometry",
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
                4,
                // `RUFFLE_NO_SHARED_SHAPES`: one buffer per shape again, for A/B comparisons.
                if std::env::var_os("RUFFLE_NO_SHARED_SHAPES").is_some() {
                    4
                } else {
                    8 << 20
                },
            ),
            uniform_arena: crate::arena::BufferArena::new(
                "Shape uniforms",
                wgpu::BufferUsages::UNIFORM,
                uniform_alignment,
                1 << 20,
            ),
            gradient_cache: GradientCache::default(),
            background_shapes: BackgroundShapes::new(),
            offscreen_buffer_pool: Arc::new(offscreen_buffer_pool),
            dynamic_transforms: transforms,
            active_frame,
            profiler,
        })
    }

    /// AQW build: frees every spare pooled texture and lets the GPU release the memory, for
    /// when the GPU's memory is nearly full (see Mugetsu's memory watch).
    pub fn aqw_trim_memory(&mut self) {
        self.cab_recycler.clear();
        self.texture_pool.clear_idle();
        self.offscreen_texture_pool.clear_idle();
        crate::release_gpu_memory(&self.descriptors);
    }

    /// AQW build: creates a texture; if the GPU is out of memory (2 GB laptop cards in crowded
    /// AQW rooms), frees every idle pooled texture, lets the GPU release them, and tries once
    /// more. `None` if it still doesn't fit: callers treat that like a too-large bitmap (a
    /// cacheAsBitmap object is then drawn without its cache) instead of a fatal error.
    fn create_texture_making_room(
        &mut self,
        desc: &wgpu::TextureDescriptor,
    ) -> Option<wgpu::Texture> {
        let scope = self
            .descriptors
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let texture = crate::perf_counters::timers::time(
            crate::perf_counters::timers::TEXTURE_CREATE,
            || self.descriptors.device.create_texture(desc),
        );
        if !crate::aqw_out_of_memory(scope) {
            return Some(texture);
        }
        drop(texture);
        self.cab_recycler.clear();
        tracing::warn!(
            target: "ruffle_aqw_perf",
            "GPU out of memory for a {}x{} bitmap; freeing idle textures and retrying",
            desc.size.width,
            desc.size.height
        );
        self.texture_pool.clear_idle();
        self.offscreen_texture_pool.clear_idle();
        crate::release_gpu_memory(&self.descriptors);
        let scope = self
            .descriptors
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let texture = self.descriptors.device.create_texture(desc);
        if !crate::aqw_out_of_memory(scope) {
            return Some(texture);
        }
        tracing::warn!(
            target: "ruffle_aqw_perf",
            "GPU still out of memory for a {}x{} bitmap; not creating it",
            desc.size.width,
            desc.size.height
        );
        None
    }

    /// AQW build: uploads a shape's uniforms, vertices and indices into the shared arenas. If
    /// the GPU is out of memory for a new arena chunk, frees every idle pooled texture, lets the
    /// GPU release them, and tries once more; `None` if it still fails.
    #[allow(clippy::type_complexity)]
    fn alloc_shape_buffers(
        &mut self,
        vertices: &[u8],
        indices: &[u8],
    ) -> Option<(
        Option<crate::arena::ArenaSlice>,
        Option<crate::arena::ArenaSlice>,
    )> {
        for attempt in 0..2 {
            let device = &self.descriptors.device;
            let frame = &mut self.active_frame;
            let result = (|| {
                Ok::<_, crate::arena::ArenaOutOfMemory>((
                    // (A multiple of both vertex sizes, see `CommandRenderer::render_shape`.)
                    self.mesh_arena.alloc_at_multiple(
                        device,
                        &mut frame.staging_belt,
                        &mut frame.command_encoder,
                        vertices,
                        crate::surface::commands::VERTEX_BASE_MULTIPLE,
                    )?,
                    self.mesh_arena.alloc(
                        device,
                        &mut frame.staging_belt,
                        &mut frame.command_encoder,
                        indices,
                    )?,
                ))
            })();
            match result {
                Ok(slices) => return Some(slices),
                Err(_) if attempt == 0 => {
                    tracing::warn!(
                        target: "ruffle_aqw_perf",
                        "GPU out of memory for shape buffers; freeing idle textures and retrying"
                    );
                    self.texture_pool.clear_idle();
                    self.offscreen_texture_pool.clear_idle();
                    crate::release_gpu_memory(&self.descriptors);
                }
                Err(_) => {}
            }
        }
        tracing::warn!(
            target: "ruffle_aqw_perf",
            "GPU still out of memory for shape buffers; shape not drawn"
        );
        None
    }

    /// AQW build: draws the cached objects waiting in `batch` (see `cache_batch`), a page at a
    /// time: their contents side by side, each filter step once for the whole page, then each
    /// slot copied into the object's own texture.
    fn flush_cache_batch(&mut self, batch: &mut crate::cache_batch::CacheBatch) {
        use crate::cache_batch as cb;
        if batch.is_empty() {
            return;
        }
        for (signature, entries) in batch.take() {
            let mut queue: std::collections::VecDeque<_> = entries.into();
            while !queue.is_empty() {
                let sizes: Vec<(u32, u32)> = queue.iter().map(|(_, size)| *size).collect();
                let (page, slots) = cb::pack(&sizes);
                crate::perf_counters::count_page(
                    page,
                    slots.iter().map(|slot| (slot.width, slot.height)),
                );
                debug_assert!(!slots.is_empty());
                let mut lists = Vec::with_capacity(slots.len());
                let mut handles = Vec::with_capacity(slots.len());
                let mut filters = Vec::with_capacity(slots.len());
                for slot in &slots {
                    let (entry, _) = queue.pop_front().expect("one entry per slot");
                    for filter in &entry.filters {
                        crate::filters::count_filter(filter);
                    }
                    lists.push((entry.commands, cb::layer_rect(slot)));
                    handles.push(entry.handle);
                    filters.push(entry.filters);
                }
                crate::perf_counters::inc(crate::perf_counters::CAB_PAGES);
                crate::perf_counters::add(crate::perf_counters::CAB_BATCHED, slots.len() as u32);

                let surface = Surface::new(
                    &self.descriptors,
                    self.surface.quality(),
                    page.0,
                    page.1,
                    wgpu::TextureFormat::Rgba8Unorm,
                );
                let mut scope = self
                    .profiler
                    .scope("Batched filters", &mut self.active_frame.command_encoder);
                let content = crate::surface::draw_into_slots(
                    &surface,
                    &self.descriptors,
                    &self.meshes,
                    &mut self.active_frame.staging_belt,
                    &self.dynamic_transforms,
                    &mut scope,
                    &mut self.offscreen_texture_pool,
                    page.0,
                    page.1,
                    lists,
                );
                // Every target stays alive until the copies below are recorded.
                let mut targets = vec![content];
                let batched = &self.descriptors.filters.batch;
                for (step, &kind) in signature.iter().enumerate() {
                    let source = targets
                        .last()
                        .expect("the content")
                        .color_texture()
                        .clone();
                    if cb::is_blur_step(kind) {
                        let blur_slots: Vec<_> = slots
                            .iter()
                            .zip(&filters)
                            .map(|(slot, filters)| match &filters[step] {
                                Filter::BlurFilter(blur) => (*slot, Some(blur.clone())),
                                _ => unreachable!("same filter chain"),
                            })
                            .collect();
                        if let Some(blurred) = batched.blur(
                            &self.descriptors,
                            &mut self.offscreen_texture_pool,
                            &mut scope,
                            &mut self.active_frame.staging_belt,
                            &source,
                            page,
                            &blur_slots,
                        ) {
                            targets.push(blurred);
                        }
                    } else {
                        let glows: Vec<_> = filters
                            .iter()
                            .map(|filters| cb::glow_step(&filters[step]))
                            .collect();
                        let blur_slots: Vec<_> = slots
                            .iter()
                            .zip(&glows)
                            .map(|(slot, (glow, _))| (*slot, Some(glow.inner_blur_filter())))
                            .collect();
                        let blurred = batched.blur(
                            &self.descriptors,
                            &mut self.offscreen_texture_pool,
                            &mut scope,
                            &mut self.active_frame.staging_belt,
                            &source,
                            page,
                            &blur_slots,
                        );
                        let blurred_texture = blurred
                            .as_ref()
                            .map(|blurred| blurred.color_texture().clone())
                            .unwrap_or_else(|| source.clone());
                        targets.extend(blurred);
                        let glowed = batched.glow(
                            &self.descriptors,
                            &mut self.offscreen_texture_pool,
                            &mut scope,
                            &mut self.active_frame.staging_belt,
                            &source,
                            &blurred_texture,
                            page,
                            &cb::slot_glows(&slots, &glows),
                        );
                        targets.push(glowed);
                    }
                }
                let result = targets.last().expect("the content").color_texture();
                for (handle, slot) in handles.iter().zip(&slots) {
                    scope.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: result,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: slot.x,
                                y: slot.y,
                                z: 0,
                            },
                            aspect: Default::default(),
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &as_texture(handle).texture,
                            mip_level: 0,
                            origin: Default::default(),
                            aspect: Default::default(),
                        },
                        wgpu::Extent3d {
                            width: slot.width,
                            height: slot.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                drop(targets);
                drop(scope);
                crate::segments::cut_now(&self.descriptors, &mut self.active_frame.command_encoder);
            }
        }
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
        stretch: Stretch,
    ) -> Mesh {
        let shape_id = shape.id;
        let bounds = shape.shape_bounds.clone();
        let lyon_mesh =
            self.shape_tessellator
                .tessellate_shape_stretched(shape, bitmap_source, scale, stretch);
        self.upload_shape(shape_id, bounds, lyon_mesh, bitmap_source)
    }

    /// The GPU half of `register_shape_internal`: builds the draws for a tessellated shape and
    /// uploads its geometry. Must run on the thread that owns the frame (main thread).
    fn upload_shape(
        &mut self,
        shape_id: swf::CharacterId,
        bounds: swf::Rectangle<swf::Twips>,
        lyon_mesh: ruffle_render::tessellator::Mesh,
        bitmap_source: &dyn BitmapSource,
    ) -> Mesh {
        let mut vertex_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        let mut index_buffer = BufferBuilder::new(&self.descriptors.limits, 0);
        // AQW build: gradients are shared between shapes (see `GradientCache`); a new one is a
        // small texture plus its uniforms. If the GPU is out of memory, free idle pooled
        // textures, let the GPU release them, and try once more (see
        // `create_texture_making_room`). If it still fails, the shape draws nothing.
        let mut gradients = None;
        for attempt in 0..2 {
            if let Some(made) = self.shared_gradients(&lyon_mesh.gradients) {
                gradients = Some(made);
                break;
            }
            if attempt == 0 {
                tracing::warn!(
                    target: "ruffle_aqw_perf",
                    "GPU out of memory for shape gradients; freeing idle textures and retrying"
                );
                self.texture_pool.clear_idle();
                self.offscreen_texture_pool.clear_idle();
                crate::release_gpu_memory(&self.descriptors);
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
            };
        };

        let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
        for draw in lyon_mesh.draws {
            if let Some(draw) = PendingDraw::new(
                self,
                bitmap_source,
                draw,
                &mut vertex_buffer,
                &mut index_buffer,
            ) {
                draws.push(draw);
            }
        }

        // AQW build: slices of the shared shape buffers instead of three buffers per shape
        // (see `crate::arena`).
        let _ = shape_id;
        let (vertices, indices, draws) =
            match self.alloc_shape_buffers(vertex_buffer.bytes(), index_buffer.bytes()) {
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

        let bytes = vertices.as_ref().map_or(0, |s| s.size()) + indices.as_ref().map_or(0, |s| s.size());
        Mesh {
            draws,
            bounds,
            live: crate::perf_counters::LiveMesh::new(bytes),
            vertices,
            indices,
            gradients,
        }
    }

    /// AQW build: the shared GPU gradients for a tessellated shape's gradients, making the
    /// ones not live yet (see `GradientCache`). `None` if the GPU is out of memory.
    fn shared_gradients(
        &mut self,
        gradients: &[ruffle_render::tessellator::Gradient],
    ) -> Option<Vec<Arc<SharedGradient>>> {
        let mut shared = Vec::with_capacity(gradients.len());
        for gradient in gradients {
            let key = GradientKey::of(gradient);
            if let Some(found) = self.gradient_cache.get(&key) {
                shared.push(found);
                continue;
            }
            let descriptors = &self.descriptors;
            let frame = &mut self.active_frame;
            let uniforms = self
                .uniform_arena
                .alloc(
                    &descriptors.device,
                    &mut frame.staging_belt,
                    &mut frame.command_encoder,
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

    fn clamp_bitmap(&self, bitmap: &mut Bitmap) -> bool {
        let max_size = self.descriptors.limits.max_texture_dimension_2d;
        if bitmap.width() > max_size || bitmap.height() > max_size {
            let image =
                image::RgbaImage::from_raw(bitmap.width(), bitmap.height(), bitmap.data().to_vec())
                    .expect("Width and height of bitmap must match bitmap data");

            let ratio = bitmap.width() as f32 / bitmap.height() as f32;
            let mut width = bitmap.width();
            let mut height = bitmap.height();
            if width > max_size {
                width = max_size;
                height = (max_size as f32 / ratio) as u32;
            }
            if height > max_size {
                height = max_size;
                width = (max_size as f32 * ratio) as u32;
            }
            let resized = image::imageops::resize(&image, width, height, FilterType::CatmullRom);
            *bitmap = Bitmap::new(width, height, BitmapFormat::Rgba, resized.into_raw());
            true
        } else {
            false
        }
    }

    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn target(&self) -> &T {
        &self.target
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.descriptors.device
    }

    pub fn make_queue_sync_handle(
        &self,
        target: TextureTarget,
        index: Option<SubmissionIndex>,
        destination: BitmapHandle,
        copy_area: PixelRegion,
    ) -> Box<QueueSyncHandle> {
        match target.take_buffer() {
            None => Box::new(QueueSyncHandle::NotCopied {
                handle: destination,
                copy_area,
                descriptors: self.descriptors.clone(),
                pool: self.offscreen_buffer_pool.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Borrowed(buffer, copy_dimensions),
                ..
            }) => Box::new(QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors: self.descriptors.clone(),
            }),
            Some(TextureBufferInfo {
                buffer: MaybeOwnedBuffer::Owned(..),
                ..
            }) => unreachable!("Buffer must be Borrowed as it was set to be Borrowed earlier"),
        }
    }
}

impl<T: RenderTarget + 'static> RenderBackend for WgpuRenderBackend<T> {
    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        // Avoid panics from creating 0-sized framebuffers.
        // TODO: find a way to bubble an error when the size is too large
        let width = std::cmp::max(
            std::cmp::min(
                dimensions.width,
                self.descriptors.limits.max_texture_dimension_2d,
            ),
            1,
        );
        let height = std::cmp::max(
            std::cmp::min(
                dimensions.height,
                self.descriptors.limits.max_texture_dimension_2d,
            ),
            1,
        );
        self.target.resize(&self.descriptors.device, width, height);

        self.surface = Surface::new(
            &self.descriptors,
            self.surface.quality(),
            width,
            height,
            self.target.format(),
        );

        self.viewport_scale_factor = dimensions.scale_factor;
        self.texture_pool = TexturePool::new_counted(crate::perf_counters::LAYER_POOL_TAKE, crate::perf_counters::LAYER_POOL_NEW);
    }

    fn create_context3d(
        &mut self,
        profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Ok(Box::new(WgpuContext3D::new(
            self.descriptors.clone(),
            profile,
        )))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        let mut result = vec![];
        result.push("Renderer: wgpu".to_string());

        let info = self.descriptors.adapter.get_info();
        result.push(format!("Adapter Backend: {:?}", info.backend));
        result.push(format!("Adapter Name: {:?}", info.name));
        result.push(format!("Adapter Device Type: {:?}", info.device_type));
        result.push(format!("Adapter Driver Name: {:?}", info.driver));
        result.push(format!("Adapter Driver Info: {:?}", info.driver_info));

        let enabled_features = self.descriptors.device.features();
        let available_features = self.descriptors.adapter.features() - enabled_features;
        let current_limits = &self.descriptors.limits;

        result.push(format!("Enabled features: {enabled_features:?}"));
        result.push(format!("Available features: {available_features:?}"));
        result.push(format!("Current limits: {current_limits:?}"));
        result.push(format!("Surface quality: {}", self.surface.quality()));
        result.push(format!("Surface samples: {}", self.surface.sample_count()));
        result.push(format!("Surface size: {:?}", self.surface.size()));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        if cfg!(target_family = "wasm") {
            let info = self.descriptors.adapter.get_info();
            if info.backend == wgpu::Backend::BrowserWebGpu {
                "webgpu"
            } else {
                "wgpu-webgl"
            }
        } else {
            "wgpu"
        }
    }

    fn set_quality(&mut self, quality: StageQuality) {
        self.surface = Surface::new(
            &self.descriptors,
            quality,
            self.surface.size().width,
            self.surface.size().height,
            self.target.format(),
        );
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.target.width(),
            height: self.target.height(),
            scale_factor: self.viewport_scale_factor,
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        let start = std::time::Instant::now();
        let mesh = self.register_shape_internal(shape, bitmap_source, 1.0, NO_STRETCH);
        crate::perf_counters::inc(crate::perf_counters::SHAPES_SINGLE);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        ShapeHandle(Arc::new(mesh))
    }

    #[instrument(level = "debug", skip_all)]
    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> ShapeHandle {
        self.register_shape_stretched(shape, bitmap_source, scale, NO_STRETCH)
    }

    fn register_shape_stretched(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
        stretch: Stretch,
    ) -> ShapeHandle {
        let start = std::time::Instant::now();
        let mesh = self.register_shape_internal(shape, bitmap_source, scale, stretch);
        crate::perf_counters::inc(crate::perf_counters::SHAPES_RESCALED);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        ShapeHandle(Arc::new(mesh))
    }

    /// AQW build: tessellates the shapes on all cores, then uploads them in order on this
    /// thread. Tessellation is deterministic and only asks the bitmap source for bitmap sizes,
    /// which are looked up here first, so the meshes are identical to `register_shape`'s.
    /// `RUFFLE_SERIAL_TESSELLATION` tessellates one at a time instead.
    fn register_shapes(
        &mut self,
        shapes: &[&swf::Shape],
        bitmap_source: &dyn BitmapSource,
    ) -> Vec<ShapeHandle> {
        use rayon::prelude::*;
        static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let serial = *SERIAL
            .get_or_init(|| std::env::var_os("RUFFLE_SERIAL_TESSELLATION").is_some());
        if serial || shapes.len() < 2 {
            return shapes
                .iter()
                .map(|shape| self.register_shape((*shape).into(), bitmap_source))
                .collect();
        }

        let start = std::time::Instant::now();
        let sizes = BitmapSizes::new(shapes, bitmap_source);
        let meshes: Vec<_> = shapes
            .par_iter()
            .map_init(ShapeTessellator::new, |tessellator, shape| {
                let shape: DistilledShape = (*shape).into();
                let id = shape.id;
                let bounds = shape.shape_bounds.clone();
                (id, bounds, tessellator.tessellate_shape(shape, &sizes))
            })
            .collect();
        let handles = meshes
            .into_iter()
            .map(|(id, bounds, lyon_mesh)| {
                ShapeHandle(Arc::new(self.upload_shape(id, bounds, lyon_mesh, bitmap_source)))
            })
            .collect();
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, shapes.len() as u32);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        handles
    }

    /// AQW build: tessellates the shapes on all cores, then uploads them in order on this
    /// thread, like `register_shapes` (same meshes as `register_shape_with_scale` for each).
    fn register_shapes_with_scales(
        &mut self,
        jobs: &[(&swf::Shape, f32, Stretch, &dyn BitmapSource)],
    ) -> Vec<ShapeHandle> {
        use rayon::prelude::*;
        if jobs.len() < 2 {
            return jobs
                .iter()
                .map(|(shape, scale, stretch, source)| {
                    self.register_shape_stretched((*shape).into(), *source, *scale, *stretch)
                })
                .collect();
        }

        let start = std::time::Instant::now();
        let sizes: Vec<BitmapSizes> = jobs
            .iter()
            .map(|(shape, _, _, source)| BitmapSizes::new(&[*shape], *source))
            .collect();
        // The bitmap sources aren't `Sync`; tessellation only needs the sizes.
        let work: Vec<(&swf::Shape, f32, Stretch)> = jobs
            .iter()
            .map(|(shape, scale, stretch, _)| (*shape, *scale, *stretch))
            .collect();
        let tessellating = crate::perf_counters::timers::Scoped::new(
            crate::perf_counters::timers::SHAPE_TESSELLATE,
        );
        let meshes: Vec<_> = work
            .par_iter()
            .zip(sizes.par_iter())
            .map_init(ShapeTessellator::new, |tessellator, ((shape, scale, stretch), sizes)| {
                let shape: DistilledShape = (*shape).into();
                let id = shape.id;
                let bounds = shape.shape_bounds.clone();
                let mesh = tessellator.tessellate_shape_stretched(shape, sizes, *scale, *stretch);
                (id, bounds, mesh)
            })
            .collect();
        drop(tessellating);
        let _uploading =
            crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::SHAPE_UPLOAD);
        let handles = meshes
            .into_iter()
            .zip(jobs)
            .map(|((id, bounds, lyon_mesh), (_, _, _, source))| {
                ShapeHandle(Arc::new(self.upload_shape(id, bounds, lyon_mesh, *source)))
            })
            .collect();
        crate::perf_counters::add(crate::perf_counters::SHAPES_RESCALED, jobs.len() as u32);
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, jobs.len() as u32);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        handles
    }

    /// AQW build: each shape tessellated on its own on the thread pool (see
    /// `BackgroundShapes`), with its bitmap sizes looked up here first, as in
    /// `register_shapes_with_scales` (same meshes).
    fn start_shapes(
        &mut self,
        jobs: Vec<(u64, Arc<swf::Shape>, f32, Stretch, &dyn BitmapSource)>,
    ) -> Vec<(u64, ruffle_render::backend::TessellatedShape)> {
        for (ticket, shape, scale, stretch, source) in jobs {
            let sizes = BitmapSizes::new(&[&*shape], source);
            let done = self.background_shapes.sender.clone();
            self.background_shapes.running.insert(ticket);
            rayon::spawn(move || {
                let distilled: DistilledShape = (&*shape).into();
                let id = distilled.id;
                let bounds = distilled.shape_bounds.clone();
                let start = std::time::Instant::now();
                let mesh = ShapeTessellator::new()
                    .tessellate_shape_stretched(distilled, &sizes, scale, stretch);
                let micros = start.elapsed().as_micros() as u64;
                // (The receiver is gone only when the backend is.)
                let _ = done.send((ticket, Tessellated { id, bounds, mesh, micros }));
            });
        }
        Vec::new()
    }

    fn collect_shapes(
        &mut self,
        deadline: Option<std::time::Instant>,
        wait_for: &[u64],
        needed: &[u64],
    ) -> Vec<(u64, ruffle_render::backend::TessellatedShape)> {
        let start = std::time::Instant::now();
        let shapes = &mut self.background_shapes;
        let mut done = Vec::new();
        let take = |running: &mut std::collections::HashSet<u64>,
                    done: &mut Vec<(u64, ruffle_render::backend::TessellatedShape)>,
                    (ticket, shape): (u64, Tessellated)| {
            running.remove(&ticket);
            let (micros, (vertices, indices)) = (shape.micros, shape.size());
            let data = Box::new(shape);
            let shape = ruffle_render::backend::TessellatedShape { data, micros, vertices, indices };
            done.push((ticket, shape));
        };
        while let Ok(received) = shapes.receiver.try_recv() {
            take(&mut shapes.running, &mut done, received);
        }
        loop {
            let waiting_for_needed = needed.iter().any(|t| shapes.running.contains(t));
            if !waiting_for_needed && !wait_for.iter().any(|t| shapes.running.contains(t)) {
                break;
            }
            let received = match deadline {
                Some(deadline) if !waiting_for_needed => {
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    shapes.receiver.recv_timeout(deadline - now).ok()
                }
                _ => shapes.receiver.recv().ok(),
            };
            match received {
                Some(received) => take(&mut shapes.running, &mut done, received),
                None => break,
            }
        }
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        crate::perf_counters::timers::add_micros(
            crate::perf_counters::timers::SHAPE_TESSELLATE,
            micros,
        );
        done
    }

    fn upload_tessellated(
        &mut self,
        shape: ruffle_render::backend::TessellatedShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Option<ShapeHandle> {
        let shape = shape.data.downcast::<Tessellated>().ok()?;
        let start = std::time::Instant::now();
        let Tessellated { id, bounds, mesh, .. } = *shape;
        let handle = ShapeHandle(Arc::new(self.upload_shape(id, bounds, mesh, bitmap_source)));
        crate::perf_counters::add(crate::perf_counters::SHAPES_RESCALED, 1);
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, 1);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPE_UPLOAD, micros);
        Some(handle)
    }

    #[instrument(level = "debug", skip_all)]
    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        // AQW build: the GPU dropped the device (`crate::device_lost`): nothing drawn would show,
        // and every pass would only make errors.
        if crate::device_lost().is_some() {
            return;
        }
        // AQW build: an allocation ran out of GPU memory since the last frame (see
        // `crate::gpu_oom`): make room before drawing this one.
        if crate::gpu_oom::take_pending() {
            self.texture_pool.clear_idle();
            self.offscreen_texture_pool.clear_idle();
            crate::release_gpu_memory(&self.descriptors);
        }
        let Some(frame_output) = self.target.get_next_texture() else {
            // Attempt to recreate the swap chain in this case.
            self.target.resize(
                &self.descriptors.device,
                self.target.width(),
                self.target.height(),
            );
            return;
        };

        // AQW build: see the segment cut at the end of the loop.
        const CAB_ENTRIES_PER_SEGMENT: usize = 8;
        let mut cab_entries = 0usize;
        let cache_timer =
            crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::CACHE_DRAWS);
        let mut batch = crate::cache_batch::CacheBatch::default();
        for entry in cache_entries {
            let texture = as_texture(&entry.handle);
            crate::perf_counters::inc(crate::perf_counters::CAB_ENTRIES);
            crate::perf_counters::add(
                crate::perf_counters::CAB_KPX,
                (texture.texture.width() * texture.texture.height()).div_ceil(1000),
            );
            // AQW build: batched filter redraws (see `cache_batch`). Whatever this object draws
            // must already be up to date, so a batch it refers to is drawn first.
            let size = (texture.texture.width(), texture.texture.height());
            if batch.references_pending(&entry.commands) {
                self.flush_cache_batch(&mut batch);
            }
            if let Some(signature) = crate::cache_batch::signature(&entry, size.0, size.1) {
                batch.add(signature, entry, size);
                if batch.len() >= 4 * crate::filters::batch::MAX_SLOTS {
                    self.flush_cache_batch(&mut batch);
                }
                continue;
            }
            let texture = as_texture(&entry.handle);
            let surface = Surface::new(
                &self.descriptors,
                self.surface.quality(),
                texture.texture.width(),
                texture.texture.height(),
                wgpu::TextureFormat::Rgba8Unorm,
            );
            if entry.filters.is_empty() {
                surface.draw_commands(
                    RenderTargetMode::ExistingWithColor(
                        texture.texture.clone(),
                        wgpu::Color {
                            r: f64::from(entry.clear.r) / 255.0,
                            g: f64::from(entry.clear.g) / 255.0,
                            b: f64::from(entry.clear.b) / 255.0,
                            a: f64::from(entry.clear.a) / 255.0,
                        },
                    ),
                    &self.descriptors,
                    &self.meshes,
                    entry.commands,
                    &mut self.active_frame.staging_belt,
                    &self.dynamic_transforms,
                    &mut self
                        .profiler
                        .scope("Draw to CAB", &mut self.active_frame.command_encoder),
                    LayerRef::None,
                    &mut self.offscreen_texture_pool,
                );
            } else {
                let mut scope = self
                    .profiler
                    .scope("Filters", &mut self.active_frame.command_encoder);
                // We're relying on there being no impotent filters here,
                // so that we can safely start by using the actual CAB texture.
                // It's guaranteed that at least one filter would have used it and moved the target to something else,
                // letting us safely copy back to it later.
                let mut target = surface.draw_commands(
                    RenderTargetMode::ExistingWithColor(
                        texture.texture.clone(),
                        wgpu::Color {
                            r: f64::from(entry.clear.r) / 255.0,
                            g: f64::from(entry.clear.g) / 255.0,
                            b: f64::from(entry.clear.b) / 255.0,
                            a: f64::from(entry.clear.a) / 255.0,
                        },
                    ),
                    &self.descriptors,
                    &self.meshes,
                    entry.commands,
                    &mut self.active_frame.staging_belt,
                    &self.dynamic_transforms,
                    &mut scope.scope("Draw to CAB"),
                    LayerRef::None,
                    &mut self.offscreen_texture_pool,
                );
                for filter in entry.filters {
                    target = self.descriptors.filters.apply(
                        &self.descriptors,
                        &mut scope.scope(filter.name()),
                        &mut self.offscreen_texture_pool,
                        &mut self.active_frame.staging_belt,
                        FilterSource::for_entire_texture(target.color_texture()),
                        filter,
                    );
                }
                run_copy_pipeline(
                    &self.descriptors,
                    texture.texture.format(),
                    &texture.texture.create_view(&Default::default()),
                    target.color_view(),
                    target.globals(),
                    target.color_texture().sample_count(),
                    &mut scope.scope("Copy filtered to CAB"),
                );
            }
            // AQW build: glow/blur-heavy skills redraw hundreds of cached objects per frame
            // (each several filter passes), all recorded into one encoder that a single core
            // then had to finish (~70% of 100 ms frames). Cut it into parallel segments too.
            cab_entries += 1;
            if cab_entries % CAB_ENTRIES_PER_SEGMENT == 0 {
                crate::segments::cut_now(
                    &self.descriptors,
                    &mut self.active_frame.command_encoder,
                );
            }
            // Periodically flush GPU work to prevent OOM when many cache entries
            // accumulate (e.g. when a large container's cacheAsBitmap is skipped
            // but its hundreds of children each have their own bitmap caches).
            self.active_frame.maybe_flush(&self.descriptors);
        }
        self.flush_cache_batch(&mut batch);
        drop(cache_timer);

        let main_timer =
            crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::MAIN_DRAW);
        self.surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithColor(wgpu::Color {
                r: f64::from(clear.r) / 255.0,
                g: f64::from(clear.g) / 255.0,
                b: f64::from(clear.b) / 255.0,
                a: f64::from(clear.a) / 255.0,
            }),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &self.dynamic_transforms,
            &mut self
                .profiler
                .scope("Frame commands", &mut self.active_frame.command_encoder),
            &self.meshes,
            commands,
            LayerRef::None,
            &mut self.texture_pool,
        );
        drop(main_timer);
        self.profiler
            .resolve_queries(&mut self.active_frame.command_encoder);
        self.active_frame.staging_belt.finish();

        self.active_frame
            .submit_for_target(&self.descriptors, &self.target, frame_output);
        // AQW build: this pool (cacheAsBitmap and their filters) used to be thrown away every
        // frame, recreating ~150 textures per frame in crowded AQW rooms. Besides the CPU cost,
        // the graphics driver leaked host memory on that churn (10-90 MB/s). Keep it across
        // frames like the main pool. Cached objects change size as they animate (thousands of
        // distinct sizes), so sizes unused for ~1 s at 24 fps are freed.
        if std::env::var_os("RUFFLE_FRESH_CACHE_POOL").is_some() {
            // Stock behavior, for A/B comparisons.
            self.offscreen_texture_pool = TexturePool::new_counted(crate::perf_counters::CACHE_POOL_TAKE, crate::perf_counters::CACHE_POOL_NEW);
        } else {
            self.offscreen_texture_pool.end_frame(24);
        }
        // AQW build: free layer/filter textures of sizes unused for ~10 s at 24 fps.
        self.texture_pool.end_frame(240);
        // AQW build: cacheAsBitmap textures not reused within ~1 s, and within 64 MB (16 MB
        // while the GPU's memory is nearly full; Android always says so).
        let recycle_mb = if crate::buffer_pool::MEMORY_PRESSURE.load(std::sync::atomic::Ordering::Relaxed) {
            16
        } else {
            64
        };
        self.cab_recycler.end_frame(24, recycle_mb << 20);
        self.mesh_arena.end_frame();
        self.uniform_arena.end_frame();
        self.profiler
            .end_frame()
            .expect("Frame should end successfully");
        let timestamp_period = self.descriptors.queue.get_timestamp_period();
        self.profiler.process_finished_frame(timestamp_period);
    }

    #[instrument(level = "debug", skip_all)]
    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        let mut bitmap = bitmap.to_rgba();

        self.clamp_bitmap(&mut bitmap);

        let extent = wgpu::Extent3d {
            width: bitmap.width(),
            height: bitmap.height(),
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        crate::perf_counters::inc(crate::perf_counters::BITMAP_NEW);
        let texture = self
            .create_texture_making_room(&wgpu::TextureDescriptor {
                label: texture_label.as_deref(),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            })
            .ok_or(BitmapError::TooLarge)?;

        let _upload_timer = crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::UPLOAD);
        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Default::default(),
                aspect: wgpu::TextureAspect::All,
            },
            bitmap.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * extent.width),
                rows_per_image: None,
            },
            extent,
        );

        let handle = BitmapHandle(Arc::new(Texture {
            recycle: None,
            live: Some(crate::perf_counters::LiveBitmap::new(&texture)),
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        }));

        Ok(handle)
    }

    #[instrument(level = "debug", skip_all)]
    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        mut region: PixelRegion,
    ) -> Result<(), BitmapError> {
        if region.width() == 0 || region.height() == 0 {
            // Nothing to do. It's important to bail out now, as the
            // write_texture call panics when the source buffer is of zero size.
            return Ok(());
        }

        let texture = as_texture(handle);

        let mut bitmap = bitmap.to_rgba();
        if self.clamp_bitmap(&mut bitmap) {
            // If we're updating a resized texture, just redo the whole thing.
            // We can't trivially map pixel regions as we use a filter to resize.
            region = PixelRegion::for_whole_size(bitmap.width(), bitmap.height());
        }

        let extent = wgpu::Extent3d {
            width: region.width(),
            height: region.height(),
            depth_or_array_layers: 1,
        };

        self.active_frame.submit_direct(&self.descriptors);
        let _upload_timer = crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::UPLOAD);
        self.descriptors.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x_min,
                    y: region.y_min,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap.data()[(region.y_min * texture.texture.width() * 4) as usize
                ..(region.y_max * texture.texture.width() * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: (region.x_min * 4) as wgpu::BufferAddress,
                bytes_per_row: Some(4 * texture.texture.width()),
                rows_per_image: None,
            },
            extent,
        );

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        commands: CommandList,
        quality: StageQuality,
        bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        crate::perf_counters::inc(crate::perf_counters::OFFSCREEN);
        let texture = as_texture(&handle);

        let extent = wgpu::Extent3d {
            width: texture.texture.width(),
            height: texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let mut target = TextureTarget {
            size: extent,
            texture: texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let frame_output = target
            .get_next_texture()
            .expect("TextureTargetFrame.get_next_texture is infallible");

        let surface = Surface::new(
            &self.descriptors,
            quality,
            texture.texture.width(),
            texture.texture.height(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        surface.draw_commands_and_copy_to(
            frame_output.view(),
            RenderTargetMode::FreshWithTexture(target.get_texture()),
            &self.descriptors,
            &mut self.active_frame.staging_belt,
            &self.dynamic_transforms,
            &mut self
                .profiler
                .scope("Offscreen commands", &mut self.active_frame.command_encoder),
            &self.meshes,
            commands,
            LayerRef::Current,
            &mut self.offscreen_texture_pool,
        );

        self.active_frame.maybe_flush(&self.descriptors);
        Some(self.make_queue_sync_handle(target, None, handle, bounds))
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        matches!(
            filter,
            Filter::BlurFilter(_)
                | Filter::GlowFilter(_)
                | Filter::DropShadowFilter(_)
                | Filter::ColorMatrixFilter(_)
                | Filter::ShaderFilter(_)
                | Filter::BevelFilter(_)
                | Filter::DisplacementMapFilter(_)
        )
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn apply_filter(
        &mut self,
        source: BitmapHandle,
        source_point: (u32, u32),
        source_size: (u32, u32),
        destination: BitmapHandle,
        dest_point: (i32, i32),
        filter: Filter,
    ) -> Option<Box<dyn SyncHandle>> {
        let source_texture = as_texture(&source);
        let dest_texture = as_texture(&destination);

        let copy_area = PixelRegion::for_whole_size(
            dest_texture.texture.width(),
            dest_texture.texture.height(),
        );

        let target = TextureTarget {
            size: wgpu::Extent3d {
                width: dest_texture.texture.width(),
                height: dest_texture.texture.height(),
                depth_or_array_layers: 1,
            },
            texture: dest_texture.texture.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            buffer: None,
        };

        let applied_filter = self.descriptors.filters.apply(
            &self.descriptors,
            &mut self.active_frame.command_encoder,
            &mut self.offscreen_texture_pool,
            &mut self.active_frame.staging_belt,
            FilterSource {
                texture: &source_texture.texture,
                point: source_point,
                size: source_size,
            },
            filter,
        );

        let (dest_x, dest_y) = dest_point;

        let src_offset_x = dest_x.min(0).unsigned_abs();
        let src_offset_y = dest_y.min(0).unsigned_abs();

        let final_dest_x = dest_x.max(0) as u32;
        let final_dest_y = dest_y.max(0) as u32;

        let available_width = applied_filter.width().saturating_sub(src_offset_x);
        let available_height = applied_filter.height().saturating_sub(src_offset_y);
        let dest_available_width = dest_texture.texture.width().saturating_sub(final_dest_x);
        let dest_available_height = dest_texture.texture.height().saturating_sub(final_dest_y);

        let copy_width = available_width.min(dest_available_width);
        let copy_height = available_height.min(dest_available_height);

        if copy_width == 0 || copy_height == 0 {
            return None;
        }

        self.active_frame.command_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: applied_filter.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: src_offset_x,
                    y: src_offset_y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dest_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: final_dest_x,
                    y: final_dest_y,
                    z: 0,
                },
                aspect: Default::default(),
            },
            wgpu::Extent3d {
                width: copy_width,
                height: copy_height,
                depth_or_array_layers: 1,
            },
        );

        self.active_frame.maybe_flush(&self.descriptors);
        Some(self.make_queue_sync_handle(target, None, destination, copy_area))
    }

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, BitmapError> {
        self.compile_pixelbender_shader_impl(shader)
    }

    fn run_pixelbender_shader(
        &mut self,
        shader: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        let output_channels = shader
            .0
            .parsed_shader()
            .output_channels()
            .expect("No output parameter");
        let has_padding = output_channels == 3;

        let texture_format =
            crate::pixel_bender::temporary_texture_format_for_channels(output_channels as u32);

        let target_handle = match target {
            PixelBenderTarget::Bitmap(handle) => handle.clone(),
            PixelBenderTarget::Bytes { width, height } => {
                let extent = wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                };
                // FIXME - cache this texture somehow. We might also want to consider using
                // a compute shader
                let texture_label = create_debug_label!("Temporary pixelbender output texture");
                let texture = self
                    .descriptors
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: texture_label.as_deref(),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: texture_format,
                        view_formats: &[texture_format],
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                    });
                BitmapHandle(Arc::new(Texture {
                    recycle: None,
                    live: Some(crate::perf_counters::LiveBitmap::new(&texture)),
                    texture,
                    repeating_linear: Default::default(),
                    repeating_nearest: Default::default(),
                    clamped_linear: Default::default(),
                    clamped_nearest: Default::default(),
                    copy_count: Cell::new(0),
                }))
            }
        };

        let target_texture = as_texture(&target_handle);

        let extent = wgpu::Extent3d {
            width: target_texture.texture.width(),
            height: target_texture.texture.height(),
            depth_or_array_layers: 1,
        };

        let copy_dimensions = BufferDimensions::new(
            target_texture.texture.width() as usize,
            target_texture.texture.height() as usize,
            target_texture.texture.format(),
        );
        let buffer_info = Some(TextureBufferInfo {
            buffer: MaybeOwnedBuffer::Borrowed(
                self.offscreen_buffer_pool
                    .take(&self.descriptors, copy_dimensions.clone()),
                copy_dimensions,
            ),
            copy_area: PixelRegion::for_whole_size(
                target_texture.texture.width(),
                target_texture.texture.height(),
            ),
        });

        let mut texture_target = TextureTarget {
            size: extent,
            texture: target_texture.texture.clone(),
            format: target_texture.texture.format(),
            buffer: buffer_info,
        };

        let frame_output = texture_target
            .get_next_texture()
            .expect("TextureTargetFrame.get_next_texture is infallible");

        run_pixelbender_shader_impl(
            &self.descriptors,
            shader,
            ShaderMode::ShaderJob,
            arguments,
            &target_texture.texture,
            &mut self.active_frame.command_encoder,
            Some(wgpu::RenderPassColorAttachment {
                view: frame_output.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }),
            1,
            // When running a standalone shader, we always process the entire image
            &FilterSource::for_entire_texture(&target_texture.texture),
        )?;

        let index = Some(self.active_frame.submit_for_target(
            &self.descriptors,
            &texture_target,
            frame_output,
        ));

        let sync_handle = self.make_queue_sync_handle(
            texture_target,
            index,
            target_handle,
            PixelRegion::for_whole_size(extent.width, extent.height),
        );

        match target {
            PixelBenderTarget::Bitmap(_) => Ok(PixelBenderOutput::Bitmap(sync_handle)),
            PixelBenderTarget::Bytes { width, .. } => {
                let mut output = None;
                self.resolve_sync_handle(
                    sync_handle,
                    Box::new(|raw_pixels, buffer_width| {
                        let width = *width as usize;

                        if buffer_width as usize
                            != width * output_channels * std::mem::size_of::<f32>()
                        {
                            let mut new_pixels = Vec::new();
                            for row in raw_pixels.chunks(buffer_width as usize) {
                                let actual_row = &row[0..(width * std::mem::size_of::<[f32; 4]>())];

                                for pixel in actual_row
                                    .as_chunks::<{ std::mem::size_of::<[f32; 4]>() }>()
                                    .0
                                {
                                    if has_padding {
                                        // Take the first three channels
                                        new_pixels.extend_from_slice(
                                            &pixel[0..(3 * std::mem::size_of::<f32>())],
                                        );
                                    } else {
                                        // Copy the pixel as-is
                                        new_pixels.extend_from_slice(pixel);
                                    }
                                }
                            }
                            output = Some(new_pixels);
                        } else {
                            output = Some(raw_pixels.to_vec());
                        };
                    }),
                )?;
                Ok(PixelBenderOutput::Bytes(output.unwrap()))
            }
        }
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        let width = width.get();
        let height = height.get();

        if width > self.descriptors.limits.max_texture_dimension_2d
            || height > self.descriptors.limits.max_texture_dimension_2d
        {
            return Err(BitmapError::TooLarge);
        }

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture_label = create_debug_label!("Bitmap");
        // AQW build: only cacheAsBitmap uses this; those pictures are remade whenever their
        // object changes size and freed with it, so they get GPU memory of their own, and a
        // freed one of the same size is reused (`TextureRecycler`). Its old picture doesn't
        // matter: a cache redraw clears or overwrites the whole texture.
        let recycled = self.cab_recycler.take(width, height);
        crate::perf_counters::inc(if recycled.is_some() {
            crate::perf_counters::CAB_TEXTURE_REUSED
        } else {
            crate::perf_counters::CAB_TEXTURE_NEW
        });
        let texture = match recycled {
            Some(texture) => texture,
            None => crate::texture_kind(crate::CACHED_TEXTURES, || {
                self.create_texture_making_room(&wgpu::TextureDescriptor {
                    label: texture_label.as_deref(),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                })
            })
            .ok_or(BitmapError::TooLarge)?,
        };
        Ok(BitmapHandle(Arc::new(Texture {
            recycle: Some(self.cab_recycler.clone()),
            live: Some(crate::perf_counters::LiveBitmap::new_cab(&texture)),
            texture,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
        })))
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        let handle = Box::<dyn Any>::downcast::<QueueSyncHandle>(handle).unwrap();
        handle.capture(with_rgba, &mut self.active_frame);
        Ok(())
    }
}

pub async fn request_adapter_and_device(
    backend: wgpu::Backends,
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
    power_preference: wgpu::PowerPreference,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), Error> {
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface: surface,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }).await
        .map_err(|_e| {
            let names = get_backend_names(backend);
            if names.is_empty() {
                "Ruffle requires hardware acceleration, but no compatible graphics device was found (no backend provided?)".to_string()
            } else if cfg!(target_vendor = "apple") {
                "Ruffle does not support OpenGL on macOS/iOS.".to_string()
            } else {
                format!("Ruffle requires hardware acceleration, but no compatible graphics device was found supporting {}", format_list(&names, "or"))
            }
        })?;

    let (device, queue) = request_device(&adapter).await?;
    crate::gpu_oom::install_error_handler(&device);
    Ok((adapter, device, queue))
}

// We try to request the highest limits we can get away with
async fn request_device(
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    // We start off with the lowest limits we actually need - basically GL-ES 3.0
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
    // Then we increase parts of it to the maximum supported by the adapter, to take advantage of
    // more powerful hardware or capabilities
    limits = limits.using_resolution(adapter.limits());
    limits = limits.using_alignment(adapter.limits());
    limits.max_uniform_buffer_binding_size = adapter.limits().max_uniform_buffer_binding_size;
    limits.max_inter_stage_shader_variables = adapter.limits().max_inter_stage_shader_variables;
    // This will be a default limit in a future wgpu version (down from 8).
    // It's required for some WebGL devices to be supported.
    limits.max_color_attachments = 4;

    let mut features = Default::default();

    let optional_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::FLOAT32_FILTERABLE
        | GpuProfiler::ALL_WGPU_TIMER_FEATURES;

    features |= optional_features & adapter.features();

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            // AQW build: smaller GPU memory blocks (8-64 MB instead of 128-256 MB). Textures
            // come and go all the time in AQW (layers, cached bitmaps, filters) and left the big
            // blocks mostly empty: 6-10 GB reserved for ~2 GB used after a session.
            // `RUFFLE_GPU_MEMORY_HINTS=performance` restores the old block sizes.
            memory_hints: if std::env::var("RUFFLE_GPU_MEMORY_HINTS")
                .is_ok_and(|v| v.eq_ignore_ascii_case("performance"))
            {
                wgpu::MemoryHints::Performance
            } else {
                wgpu::MemoryHints::MemoryUsage
            },
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
}

/// Determines how we choose our frame buffer
#[derive(Clone)]
pub enum RenderTargetMode {
    // Construct a new frame buffer, clearng it with the provided color.
    // This is used when rendering to the actual display,
    // or when applying a filter. In both cases, we have a fixed background color,
    // and don't need to blend with anything else
    FreshWithColor(wgpu::Color),
    // Construct a new frame buffer, cleared with an existing texture.
    // we will blend with the previous contents of the texture.
    // This is used in `render_offscreen`, as we need to blend with the previous
    // contents of our `BitmapData` texture
    FreshWithTexture(wgpu::Texture),
    // Use the provided texture as our frame buffer, and clear it with the given color.
    ExistingWithColor(wgpu::Texture, wgpu::Color),
}

impl RenderTargetMode {
    pub fn color(&self) -> Option<wgpu::Color> {
        match self {
            RenderTargetMode::FreshWithColor(color) => Some(*color),
            RenderTargetMode::FreshWithTexture(_) => None,
            RenderTargetMode::ExistingWithColor(_, color) => Some(*color),
        }
    }
}

pub struct ActiveFrame {
    pub staging_belt: wgpu::util::StagingBelt,
    pub command_encoder: wgpu::CommandEncoder,
    draws_since_flush: u32,
    /// AQW build: frames submitted to the GPU and possibly not finished yet.
    in_flight: std::collections::VecDeque<SubmissionIndex>,
}

impl ActiveFrame {
    const MAX_DRAWS_PER_FLUSH: u32 = 100;

    pub fn new(descriptors: &Descriptors) -> Self {
        Self {
            command_encoder: descriptors
                .device
                .create_command_encoder(&Default::default()),
            staging_belt: wgpu::util::StagingBelt::new(descriptors.device.clone(), 65536),
            draws_since_flush: 0,
            in_flight: Default::default(),
        }
    }

    pub fn submit_for_target<T: RenderTarget>(
        &mut self,
        descriptors: &Descriptors,
        target: &T,
        frame: T::Frame,
    ) -> SubmissionIndex {
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        use crate::perf_counters::timers;
        let command_buffers = timers::time(timers::FINISH, || {
            crate::segments::finish_all(descriptors, draw_encoder)
        });
        let index = timers::time(timers::SUBMIT, || {
            target.submit(&descriptors.device, &descriptors.queue, command_buffers, frame)
        });
        self.staging_belt.recall();

        // AQW build: when the GPU is the bottleneck (2000+ passes of full-screen layers),
        // nothing made the CPU wait for it, so submitted frames queued up without limit, and
        // so did everything they hold (process memory grew ~90 MB/s). Keep at most
        // MAX_FRAMES_IN_FLIGHT frames queued, like games do; this never delays a frame the
        // GPU keeps up with, and shortens the queue (latency) when it doesn't.
        const MAX_FRAMES_IN_FLIGHT: usize = 2;
        self.in_flight.push_back(index.clone());
        while self.in_flight.len() > MAX_FRAMES_IN_FLIGHT {
            let oldest = self.in_flight.pop_front().expect("queue is not empty");
            let start = std::time::Instant::now();
            timers::time(timers::GPU_WAIT, || {
                let _ = descriptors.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(oldest),
                    timeout: Some(std::time::Duration::from_secs(2)),
                });
            });
            crate::perf_counters::add(
                crate::perf_counters::GPU_WAIT_MS,
                start.elapsed().as_millis() as u32,
            );
        }
        index
    }

    pub fn submit_direct(&mut self, descriptors: &Descriptors) -> SubmissionIndex {
        self.draws_since_flush = 0;
        self.staging_belt.finish();
        let draw_encoder = std::mem::replace(
            &mut self.command_encoder,
            descriptors
                .device
                .create_command_encoder(&Default::default()),
        );
        let index = descriptors
            .queue
            .submit(crate::segments::finish_all(descriptors, draw_encoder));
        self.staging_belt.recall();
        index
    }

    pub fn maybe_flush(&mut self, descriptors: &Descriptors) {
        // [NA] This is kind of a hack.
        // If we do "too much" during one frame, the submission ends up being way too large and goes OutOfMemory.
        // What it is that we're OOMing on is likely buffers and temporary textures and such from render_offscreen
        // Hard to track that though... so let's just flush it out if we do more than X draws per frame
        self.draws_since_flush += 1;

        if self.draws_since_flush > Self::MAX_DRAWS_PER_FLUSH {
            self.submit_direct(descriptors);
        }
    }
}

/// AQW build: the sizes of every bitmap a batch of shapes fills with, looked up up front so
/// tessellation (which only needs `bitmap_size`) can run on worker threads.
struct BitmapSizes(std::collections::HashMap<u16, Option<ruffle_render::bitmap::BitmapSize>>);

/// AQW build: a shape tessellated on the thread pool (see `RenderBackend::start_shapes`):
/// what `upload_shape` needs, and how long it took (for the log).
struct Tessellated {
    id: swf::CharacterId,
    bounds: swf::Rectangle<swf::Twips>,
    mesh: ruffle_render::tessellator::Mesh,
    micros: u64,
}

impl Tessellated {
    /// Its vertex and index counts (for the log).
    fn size(&self) -> (usize, usize) {
        (
            self.mesh.draws.iter().map(|draw| draw.vertices.len()).sum(),
            self.mesh.draws.iter().map(|draw| draw.indices.len()).sum(),
        )
    }
}

/// AQW build: shapes being tessellated on the thread pool: each sends its result here when
/// done; `running` are the tickets not received yet.
struct BackgroundShapes {
    sender: std::sync::mpsc::Sender<(u64, Tessellated)>,
    receiver: std::sync::mpsc::Receiver<(u64, Tessellated)>,
    running: std::collections::HashSet<u64>,
}

impl BackgroundShapes {
    fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self {
            sender,
            receiver,
            running: Default::default(),
        }
    }
}

impl BitmapSizes {
    fn new(shapes: &[&swf::Shape], bitmap_source: &dyn BitmapSource) -> Self {
        let mut sizes = std::collections::HashMap::new();
        let mut add = |styles: &swf::ShapeStyles| {
            let fills = styles.fill_styles.iter();
            let lines = styles.line_styles.iter().map(|line| line.fill_style());
            for fill in fills.chain(lines) {
                if let swf::FillStyle::Bitmap { id, .. } = fill {
                    sizes
                        .entry(*id)
                        .or_insert_with(|| bitmap_source.bitmap_size(*id));
                }
            }
        };
        for shape in shapes {
            add(&shape.styles);
            for record in &shape.shape {
                if let swf::ShapeRecord::StyleChange(change) = record
                    && let Some(styles) = &change.new_styles
                {
                    add(styles);
                }
            }
        }
        Self(sizes)
    }
}

impl BitmapSource for BitmapSizes {
    fn bitmap_size(&self, id: u16) -> Option<ruffle_render::bitmap::BitmapSize> {
        // Every id a shape uses was looked up in `new`.
        self.0.get(&id).copied().flatten()
    }

    fn bitmap_handle(&self, _id: u16, _renderer: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        unreachable!("tessellation only asks for bitmap sizes")
    }
}
