//! AQW build: the render thread (OFF unless `RUFFLE_RENDER_THREAD` is set; a `render-thread`
//! file next to the exe).
//!
//! Turning a picture's command list into GPU commands (`WgpuRenderBackend::submit_frame`: the
//! cacheAsBitmap redraws, their filters, the stage's render passes) was over a third of the
//! main thread's work in a crowded room, done before the next game frame's scripts or the next
//! picture's walk could start. With the render thread, the player holds a `ThreadedBackend`
//! instead of the backend itself:
//!
//! - `submit_frame` hands the picture to the render thread as a job and returns. The job runs
//!   the backend's own `submit_frame`, unchanged: the same GPU commands in the same order.
//!   One picture at a time: the next `submit_frame` waits for the last one's job.
//! - Every other backend call first waits until the picture's job is done, then runs on the
//!   calling (main) thread as before. So the backend is only ever inside ONE thread at a time,
//!   and none of its code needs to know about this: a bitmap update must not reach the queue
//!   before the picture that was built with the old pixels, and so on; all of that stays in
//!   the order it has without the thread.
//! - New shapes are the exception, when they are uploaded apart from the picture
//!   (`crate::shape_uploads`, `RUFFLE_SHAPE_UPLOADS`): a shape tween prepares shapes in every
//!   picture, while the stage is walked, and each of those calls waited for the picture
//!   before (and the walk with it). The proxy then prepares them itself, on the calling
//!   thread, with a tessellator of its own, through the uploader the backend shares with it
//!   (see `DirectShapes`). Without `RUFFLE_SHAPE_UPLOADS` they wait like every other call.
//!   A cached object's new texture (`create_empty_texture`) is the other thing nearly every
//!   walk of a crowded room asks for: the proxy makes it itself too (new GPU memory that
//!   nothing reads yet).
//! - The window's frame (the movie's picture and the menu, drawn and shown from the present
//!   thread, see `crate::present`) must be sent to the present thread AFTER the movie's own
//!   submit, which is now sent from the render thread: the frontend sends it through
//!   `run_after_picture`.
//!
//! What crosses between the threads (the command list and its handles, and the backend) is
//! not `Send` by type; see `AssertSend` for why sending it is sound here.

use crate::backend::{
    BackendUploadHost, BackgroundShapes, BitmapSizes, Tessellated, WgpuRenderBackend,
};
use crate::descriptors::Descriptors;
use crate::mesh::{Mesh, PendingDrawType};
use crate::present::{Job, PresentThread};
use crate::shape_uploads::{SharedUploader, UploadHost};
use crate::target::RenderTarget;
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, NO_STRETCH, PixelBenderOutput,
    PixelBenderTarget, RenderBackend, ShapeHandle, Stretch, TessellatedShape, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error;
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// The render thread's stack: what the main thread has for the same recursion (a blend
/// layer inside a blend layer inside...).
const RENDER_STACK: usize = 8 << 20;

/// Whether pictures are recorded on a render thread (see the module).
pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var_os("RUFFLE_RENDER_THREAD").is_some();
        if enabled {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "pictures are recorded on a render thread (RUFFLE_RENDER_THREAD)"
            );
        }
        enabled
    })
}

/// With the render thread, recording a picture can submit work of its own at any moment (a
/// flush in the middle of a heavy picture, see `ActiveFrame::maybe_flush`), from the render
/// thread. The present thread reconfigures the window's surface when it finds it out of date
/// (the window was resized), and wgpu can't do that while another thread submits: it waits
/// for the GPU, finds more work queued and fails ("Failed to wait for GPU to come idle
/// before reconfiguring the Surface"), which is fatal. So those submits (`submitting`) and
/// the reconfiguring (`reconfiguring`) keep each other out. Nothing is locked without the
/// render thread.
static SUBMIT_GATE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Held while a picture's recording submits work itself (not through the present thread).
pub fn submitting() -> Option<std::sync::RwLockReadGuard<'static, ()>> {
    enabled().then(|| {
        SUBMIT_GATE
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    })
}

/// Held while the window's surface is reconfigured (see `submitting`).
pub fn reconfiguring() -> Option<std::sync::RwLockWriteGuard<'static, ()>> {
    enabled().then(|| {
        SUBMIT_GATE
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    })
}

/// Lets a value that isn't `Send` by type go to the render thread and back.
///
/// SAFETY of every use in this module rests on one rule, which `ThreadedBackend` keeps: the
/// backend, and everything reachable only through it or through a command list given to it,
/// is used by ONE thread at a time, with a mutex or a finished job (a happens-before) at
/// every change of hands:
/// - a picture's command list and cache entries are moved into its job, used there and
///   dropped there; the main thread keeps nothing of them but clones of the handles in them;
/// - the backend is used by the job (under its mutex), and by the calling thread only after
///   the job is done (`ThreadedBackend::wait`, then the same mutex);
/// - handles (`Arc<dyn ShapeHandleImpl>` and the like) are shared between the threads, but
///   only their reference counts (atomic) are touched outside the backend; what is inside
///   them and not thread-safe (a texture's lazily made bind groups, a pixel bender shader's
///   pipelines) is only touched by backend code, so by the rule above by one thread at a
///   time. What their `Drop` reaches is thread-safe (arena slices and pooled textures go
///   back through mutexes, counters are atomics, wgpu objects are thread-safe). There is no
///   `Rc` in this crate outside `context3d`, and a Stage3D context turns the render thread
///   off (`ThreadedBackend::create_context3d`);
/// - new shapes prepared by the calling thread while a picture is recorded (`DirectShapes`)
///   touch nothing of the backend: the uploader is `Send` by type and behind its own mutex,
///   the arenas behind theirs, and a mesh is made of wgpu objects, arena slices and counters.
///   The one thing in a mesh that is the backend's, a bitmap fill's texture and its lazily
///   made bind group, is asked of the backend itself, so after the picture (`ProxyHost`).
struct AssertSend<T>(T);

// SAFETY: see the type.
unsafe impl<T> Send for AssertSend<T> {}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A `WgpuRenderBackend` whose pictures are recorded on a render thread (see the module).
pub struct ThreadedBackend<T: RenderTarget + 'static> {
    backend: Arc<Mutex<AssertSend<WgpuRenderBackend<T>>>>,
    /// The render thread: jobs run in the order sent. `None` only while dropping.
    thread: Option<Arc<PresentThread>>,
    /// The ticket of the picture's job, until it's known to be done.
    pending: Option<u64>,
    /// Encoder segments the render thread was left with (see `crate::segments`): they go on
    /// with whichever thread records next.
    carried: Arc<Mutex<AssertSend<Vec<wgpu::CommandEncoder>>>>,
    /// The backend's viewport, answered without waiting for a picture.
    viewport: ViewportDimensions,
    /// Pictures are recorded on the calling thread from now on (a Stage3D context exists:
    /// its objects are used by scripts directly, outside the backend's calls).
    sync_only: bool,
    /// How long the last picture's job took, without its wait for the present thread.
    picture_micros: Arc<AtomicU64>,
    /// All the time spent waiting for a picture's job so far.
    waited_micros: u64,
    /// The backend's own `waited_micros` when last seen.
    backend_waited: Cell<u64>,
    /// Counts every time the backend's target may have changed (see `target_state`).
    target_state: u64,
    /// The backend's descriptors (the device), for what this thread makes without waiting.
    descriptors: Arc<Descriptors>,
    /// What new shapes are prepared with on the calling thread, without waiting for the
    /// picture being recorded: only with shapes uploaded apart from the picture.
    shapes: Option<DirectShapes>,
    /// For the log: the calls that waited for a picture being recorded.
    waits: WaitStats,
}

/// See `ThreadedBackend::shapes`: the uploader the backend takes the uploads from when it
/// submits (see `crate::shape_uploads`), and what tessellating needs, kept apart from the
/// backend's own (which the render thread may be using).
struct DirectShapes {
    uploads: SharedUploader,
    tessellator: ShapeTessellator,
    background: BackgroundShapes,
}

/// Which calls waited for a picture being recorded, how many times and for how long in all
/// (microseconds), since the last report.
#[derive(Default)]
struct WaitStats(Vec<(&'static str, u32, u64)>);

impl WaitStats {
    fn add(&mut self, call: &'static str, micros: u64) {
        match self.0.iter_mut().find(|(name, _, _)| *name == call) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += micros;
            }
            None => self.0.push((call, 1, micros)),
        }
    }

    /// "call: times (ms in all), ...", the longest wait first; `None` if nothing waited.
    fn take_report(&mut self) -> Option<String> {
        let mut waits = std::mem::take(&mut self.0);
        if waits.is_empty() {
            return None;
        }
        waits.sort_by_key(|(_, _, micros)| std::cmp::Reverse(*micros));
        Some(
            waits
                .iter()
                .map(|(call, times, micros)| {
                    format!("{call}: {times} ({:.1} ms)", *micros as f64 / 1000.0)
                })
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

/// What uploading a shape from the calling thread asks the backend for (see `DirectShapes`):
/// both wait for the picture being recorded first.
struct ProxyHost<'a, T: RenderTarget + 'static> {
    proxy: &'a mut ThreadedBackend<T>,
    source: &'a dyn BitmapSource,
}

impl<T: RenderTarget + 'static> UploadHost for ProxyHost<'_, T> {
    fn bitmap_fill(
        &mut self,
        bitmap: ruffle_render::tessellator::Bitmap,
    ) -> Option<PendingDrawType> {
        let mut backend = self.proxy.backend_for("a shape's bitmap fill");
        BackendUploadHost {
            backend: &mut backend.0,
            source: self.source,
        }
        .bitmap_fill(bitmap)
    }

    fn make_room(&mut self) {
        let mut backend = self.proxy.backend_for("out of memory for a shape");
        BackendUploadHost {
            backend: &mut backend.0,
            source: self.source,
        }
        .make_room()
    }
}

/// `RUFFLE_SERIAL_TESSELLATION`, as `WgpuRenderBackend::register_shapes` reads it.
fn serial_tessellation() -> bool {
    static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SERIAL.get_or_init(|| std::env::var_os("RUFFLE_SERIAL_TESSELLATION").is_some())
}

impl<T: RenderTarget + 'static> ThreadedBackend<T> {
    pub fn new(backend: WgpuRenderBackend<T>) -> Self {
        let viewport = backend.viewport_dimensions();
        let descriptors = backend.descriptors().clone();
        let shapes = backend.shape_uploader().map(|uploads| DirectShapes {
            uploads,
            tessellator: ShapeTessellator::new(),
            background: BackgroundShapes::new(),
        });
        if shapes.is_some() {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "new shapes are prepared without waiting for the picture being recorded"
            );
        }
        Self {
            backend: Arc::new(Mutex::new(AssertSend(backend))),
            thread: Some(PresentThread::with_stack("render", Some(RENDER_STACK))),
            pending: None,
            carried: Arc::new(Mutex::new(AssertSend(Vec::new()))),
            viewport,
            sync_only: false,
            picture_micros: Arc::new(AtomicU64::new(0)),
            waited_micros: 0,
            backend_waited: Cell::new(0),
            target_state: 0,
            descriptors,
            shapes,
            waits: WaitStats::default(),
        }
    }

    fn thread(&self) -> &PresentThread {
        self.thread.as_ref().expect("the render thread is there until the backend is dropped")
    }

    /// Waits until the picture handed to the render thread has been recorded. How long that
    /// took (microseconds), if there was one.
    fn wait(&mut self) -> Option<u64> {
        let ticket = self.pending.take()?;
        let started = std::time::Instant::now();
        self.thread().wait_for(ticket);
        let micros = started.elapsed().as_micros() as u64;
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::RENDER_WAIT, micros);
        self.waited_micros += micros;
        // What the render thread was left with goes on with this thread's recording.
        let carried = std::mem::take(&mut lock(&self.carried).0);
        crate::segments::restore(carried);
        Some(micros)
    }

    /// The backend, once the picture in progress is done.
    fn backend(&mut self) -> MutexGuard<'_, AssertSend<WgpuRenderBackend<T>>> {
        self.backend_for("other")
    }

    /// `backend`, for `call` (see `WaitStats`).
    fn backend_for(
        &mut self,
        call: &'static str,
    ) -> MutexGuard<'_, AssertSend<WgpuRenderBackend<T>>> {
        if let Some(micros) = self.wait() {
            self.waits.add(call, micros);
        }
        lock(&self.backend)
    }

    /// Builds the draws of a tessellated shape and uploads its geometry from this thread,
    /// without waiting for the picture being recorded (see `DirectShapes`).
    fn upload_direct(
        &mut self,
        bounds: swf::Rectangle<swf::Twips>,
        lyon_mesh: ruffle_render::tessellator::Mesh,
        source: &dyn BitmapSource,
    ) -> Mesh {
        let shapes = self.shapes.as_ref().expect("only with direct shapes");
        let (uploads, descriptors) = (shapes.uploads.clone(), self.descriptors.clone());
        let mut host = ProxyHost {
            proxy: self,
            source,
        };
        crate::shape_uploads::upload_shape(&uploads, &descriptors, bounds, lyon_mesh, &mut host)
    }

    /// `WgpuRenderBackend::register_shape_internal`, from this thread (see `DirectShapes`).
    fn register_direct(
        &mut self,
        shape: DistilledShape,
        source: &dyn BitmapSource,
        scale: f32,
        stretch: Stretch,
    ) -> Mesh {
        let bounds = shape.shape_bounds.clone();
        let shapes = self.shapes.as_mut().expect("only with direct shapes");
        let lyon_mesh = shapes
            .tessellator
            .tessellate_shape_stretched(shape, source, scale, stretch);
        self.upload_direct(bounds, lyon_mesh, source)
    }

    /// Waits until the render thread has nothing left to do: the picture in progress, and
    /// what was queued behind it (`after`: the window's frames on their way to the present
    /// thread).
    pub fn settle(&mut self) {
        let _ = self.wait();
        let thread = self.thread();
        thread.wait_for(thread.run(Box::new(|| {})));
    }

    /// Runs `f` with the backend itself (after the picture in progress is done).
    pub fn with_backend<R>(&mut self, f: impl FnOnce(&mut WgpuRenderBackend<T>) -> R) -> R {
        let result = f(&mut self.backend().0);
        self.viewport = lock(&self.backend).0.viewport_dimensions();
        self.target_state += 1;
        result
    }

    /// Runs `job` on the render thread after the picture in progress (at once, here, when
    /// pictures are recorded on this thread): for what must reach the present thread after
    /// the picture's own submit (the window's frame).
    pub fn after(&mut self, job: Job) {
        if self.sync_only {
            let _ = self.wait();
            job();
        } else {
            self.thread().run(job);
        }
    }

    /// How long the render thread took for the last picture (microseconds), without its wait
    /// for the present thread: what the frontend's pacer budgets the thread by.
    pub fn picture_micros(&self) -> u64 {
        self.picture_micros.load(Ordering::Relaxed)
    }
}

impl<T: RenderTarget + 'static> Drop for ThreadedBackend<T> {
    fn drop(&mut self) {
        // The picture in progress is finished; the thread lets go of what its storage holds
        // of the device (see `segments::clear`) and ends; only then is the backend dropped,
        // on this thread, as without the render thread.
        let started = std::time::Instant::now();
        let _ = self.wait();
        // (What `wait` took over from the render thread is this thread's now: let go of it
        // too, as the frontend did for this thread before it dropped the player.)
        crate::segments::clear();
        let waited = started.elapsed();
        if let Some(thread) = self.thread.take() {
            thread.run(Box::new(|| {
                crate::segments::clear();
            }));
            drop(thread);
        }
        tracing::info!(
            target: "ruffle_aqw_perf",
            "closing: the render thread's last picture was waited for {} ms; the thread ended {} ms later",
            waited.as_millis(),
            (started.elapsed() - waited).as_millis()
        );
    }
}

/// Runs `f` with the wgpu backend of `renderer`, behind a render thread or not. `None` if
/// it's another kind of backend.
pub fn with_wgpu_backend<T: RenderTarget + 'static, R>(
    renderer: &mut dyn RenderBackend,
    f: impl FnOnce(&mut WgpuRenderBackend<T>) -> R,
) -> Option<R> {
    let renderer: &mut dyn Any = renderer;
    if renderer.is::<ThreadedBackend<T>>() {
        renderer
            .downcast_mut::<ThreadedBackend<T>>()
            .map(|threaded| threaded.with_backend(f))
    } else {
        renderer.downcast_mut::<WgpuRenderBackend<T>>().map(f)
    }
}

/// Runs `job` once the picture `renderer` is recording has been sent on its way (see
/// `ThreadedBackend::after`); at once if `renderer` has no render thread.
pub fn run_after_picture<T: RenderTarget + 'static>(renderer: &mut dyn RenderBackend, job: Job) {
    let renderer: &mut dyn Any = renderer;
    match renderer.downcast_mut::<ThreadedBackend<T>>() {
        Some(threaded) => threaded.after(job),
        None => job(),
    }
}

/// Waits until `renderer`'s render thread, if it has one, has nothing left to do (see
/// `ThreadedBackend::settle`): before something no other thread may submit work during, like
/// reconfiguring the window's surface.
pub fn settle<T: RenderTarget + 'static>(renderer: &mut dyn RenderBackend) {
    let renderer: &mut dyn Any = renderer;
    if let Some(threaded) = renderer.downcast_mut::<ThreadedBackend<T>>() {
        threaded.settle();
    }
}

/// With a render thread: a number that changes whenever the backend's target may have (a
/// resize, or anything done through `with_wgpu_backend`): what a frontend keeps its copy of
/// the target's state by, so reading it doesn't wait for a picture being recorded. `None`
/// without a render thread (read the target itself then).
pub fn target_state<T: RenderTarget + 'static>(renderer: &dyn RenderBackend) -> Option<u64> {
    let renderer: &dyn Any = renderer;
    renderer
        .downcast_ref::<ThreadedBackend<T>>()
        .map(|threaded| threaded.target_state)
}

/// For the log: the calls that waited for a picture being recorded since the last report
/// ("call: times (ms in all), ..."). `None` without a render thread, or if nothing waited.
pub fn waits_report<T: RenderTarget + 'static>(renderer: &mut dyn RenderBackend) -> Option<String> {
    let renderer: &mut dyn Any = renderer;
    renderer
        .downcast_mut::<ThreadedBackend<T>>()
        .and_then(|threaded| threaded.waits.take_report())
}

/// See `ThreadedBackend::picture_micros`; `None` without a render thread.
pub fn picture_micros<T: RenderTarget + 'static>(renderer: &dyn RenderBackend) -> Option<u64> {
    let renderer: &dyn Any = renderer;
    renderer
        .downcast_ref::<ThreadedBackend<T>>()
        .filter(|threaded| !threaded.sync_only)
        .map(|threaded| threaded.picture_micros())
}

impl<T: RenderTarget + 'static> RenderBackend for ThreadedBackend<T> {
    fn viewport_dimensions(&self) -> ViewportDimensions {
        self.viewport
    }

    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        let mut backend = self.backend_for("set_viewport_dimensions");
        backend.0.set_viewport_dimensions(dimensions);
        let viewport = backend.0.viewport_dimensions();
        drop(backend);
        self.viewport = viewport;
        self.target_state += 1;
    }

    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        if self.shapes.is_none() {
            return self
                .backend_for("register_shape")
                .0
                .register_shape(shape, bitmap_source);
        }
        // (As `WgpuRenderBackend::register_shape`.)
        let start = std::time::Instant::now();
        let mesh = self.register_direct(shape, bitmap_source, 1.0, NO_STRETCH);
        crate::perf_counters::inc(crate::perf_counters::SHAPES_SINGLE);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        ShapeHandle(Arc::new(mesh))
    }

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
        if self.shapes.is_none() {
            return self
                .backend_for("register_shape_stretched")
                .0
                .register_shape_stretched(shape, bitmap_source, scale, stretch);
        }
        // (As `WgpuRenderBackend::register_shape_stretched`.)
        let start = std::time::Instant::now();
        let mesh = self.register_direct(shape, bitmap_source, scale, stretch);
        crate::perf_counters::inc(crate::perf_counters::SHAPES_RESCALED);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        ShapeHandle(Arc::new(mesh))
    }

    fn waited_micros(&self) -> u64 {
        // (The backend's own waits are read when it's free; the last value seen otherwise.)
        if let Ok(backend) = self.backend.try_lock() {
            self.backend_waited.set(backend.0.waited_micros());
        }
        self.waited_micros + self.backend_waited.get()
    }

    fn register_shapes(
        &mut self,
        shapes: &[&swf::Shape],
        bitmap_source: &dyn BitmapSource,
    ) -> Vec<ShapeHandle> {
        if self.shapes.is_none() {
            return self
                .backend_for("register_shapes")
                .0
                .register_shapes(shapes, bitmap_source);
        }
        // (As `WgpuRenderBackend::register_shapes`.)
        use rayon::prelude::*;
        if serial_tessellation() || shapes.len() < 2 {
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
                let bounds = shape.shape_bounds.clone();
                (bounds, tessellator.tessellate_shape(shape, &sizes))
            })
            .collect();
        let handles = meshes
            .into_iter()
            .map(|(bounds, lyon_mesh)| {
                ShapeHandle(Arc::new(self.upload_direct(bounds, lyon_mesh, bitmap_source)))
            })
            .collect();
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, shapes.len() as u32);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        handles
    }

    fn register_shapes_with_scales(
        &mut self,
        jobs: &[(&swf::Shape, f32, Stretch, &dyn BitmapSource)],
    ) -> Vec<ShapeHandle> {
        if self.shapes.is_none() {
            return self
                .backend_for("register_shapes_with_scales")
                .0
                .register_shapes_with_scales(jobs);
        }
        // (As `WgpuRenderBackend::register_shapes_with_scales`.)
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
            .map_init(
                ShapeTessellator::new,
                |tessellator, ((shape, scale, stretch), sizes)| {
                    let shape: DistilledShape = (*shape).into();
                    let bounds = shape.shape_bounds.clone();
                    let mesh =
                        tessellator.tessellate_shape_stretched(shape, sizes, *scale, *stretch);
                    (bounds, mesh)
                },
            )
            .collect();
        drop(tessellating);
        let _uploading =
            crate::perf_counters::timers::Scoped::new(crate::perf_counters::timers::SHAPE_UPLOAD);
        let handles = meshes
            .into_iter()
            .zip(jobs)
            .map(|((bounds, lyon_mesh), (_, _, _, source))| {
                ShapeHandle(Arc::new(self.upload_direct(bounds, lyon_mesh, *source)))
            })
            .collect();
        crate::perf_counters::add(crate::perf_counters::SHAPES_RESCALED, jobs.len() as u32);
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, jobs.len() as u32);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        handles
    }

    fn start_shapes(
        &mut self,
        jobs: Vec<(u64, Arc<swf::Shape>, f32, Stretch, &dyn BitmapSource)>,
    ) -> Vec<(u64, TessellatedShape)> {
        // (The shapes started here are collected here: see `DirectShapes`.)
        match &mut self.shapes {
            Some(shapes) => shapes.background.start(jobs),
            None => self.backend_for("start_shapes").0.start_shapes(jobs),
        }
    }

    fn collect_shapes(
        &mut self,
        deadline: Option<std::time::Instant>,
        wait_for: &[u64],
        needed: &[u64],
    ) -> Vec<(u64, TessellatedShape)> {
        match &mut self.shapes {
            Some(shapes) => shapes.background.collect(deadline, wait_for, needed),
            None => self
                .backend_for("collect_shapes")
                .0
                .collect_shapes(deadline, wait_for, needed),
        }
    }

    fn upload_tessellated(
        &mut self,
        shape: TessellatedShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Option<ShapeHandle> {
        if self.shapes.is_none() {
            return self
                .backend_for("upload_tessellated")
                .0
                .upload_tessellated(shape, bitmap_source);
        }
        // (As `WgpuRenderBackend::upload_tessellated`.)
        let shape = shape.data.downcast::<Tessellated>().ok()?;
        let start = std::time::Instant::now();
        let Tessellated { bounds, mesh, .. } = *shape;
        let handle = ShapeHandle(Arc::new(self.upload_direct(bounds, mesh, bitmap_source)));
        crate::perf_counters::add(crate::perf_counters::SHAPES_RESCALED, 1);
        crate::perf_counters::add(crate::perf_counters::SHAPES_BATCHED, 1);
        let micros = start.elapsed().as_micros() as u64;
        crate::perf_counters::add(crate::perf_counters::SHAPE_REGISTER_US, micros as u32);
        crate::perf_counters::timers::add_micros(crate::perf_counters::timers::SHAPES, micros);
        crate::perf_counters::timers::add_micros(
            crate::perf_counters::timers::SHAPE_UPLOAD,
            micros,
        );
        Some(handle)
    }

    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        commands: CommandList,
        quality: StageQuality,
        bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        self.backend_for("render_offscreen")
            .0
            .render_offscreen(handle, commands, quality, bounds)
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
        self.backend_for("apply_filter").0.apply_filter(
            source,
            source_point,
            source_size,
            destination,
            dest_point,
            filter,
        )
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        // (Asked for every filtered object of every picture: answered without the backend.)
        crate::backend::filter_supported(filter)
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn submit_frame(
        &mut self,
        clear: swf::Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        if self.sync_only {
            self.backend_for("submit_frame").0.submit_frame(clear, commands, cache_entries);
            return;
        }
        // One picture at a time.
        if let Some(micros) = self.wait() {
            self.waits.add("the next picture", micros);
        }
        // Segments cut on this thread since the last picture (offscreen draws that weren't
        // flushed) come before the picture's own (see `crate::segments`).
        let earlier = AssertSend(crate::segments::take_pending());
        let picture = AssertSend((clear, commands, cache_entries));
        let backend = self.backend.clone();
        let carried = self.carried.clone();
        let picture_micros = self.picture_micros.clone();
        let ticket = self.thread().run(Box::new(move || {
            // (Whole values, not their fields: it is the wrappers that are `Send`.)
            let (earlier, picture) = (earlier, picture);
            use crate::perf_counters::timers;
            let started = std::time::Instant::now();
            let waited = timers::snapshot()[timers::PRESENT_WAIT];
            let mut backend = lock(&backend);
            crate::segments::restore(earlier.0);
            let (clear, commands, cache_entries) = picture.0;
            backend.0.submit_frame(clear, commands, cache_entries);
            // Nearly always nothing: a picture's submit takes every segment with it.
            let left = crate::segments::take_pending();
            if !left.is_empty() {
                lock(&carried).0.extend(left);
            }
            drop(backend);
            // (All of it this job's: nothing else is inside the backend meanwhile.)
            let waited = timers::snapshot()[timers::PRESENT_WAIT] - waited;
            timers::add_micros(timers::RENDER_PRESENT_WAIT, waited);
            let micros = (started.elapsed().as_micros() as u64).saturating_sub(waited);
            picture_micros.store(micros, Ordering::Relaxed);
            timers::add_micros(timers::RENDER_THREAD, micros);
        }));
        self.pending = Some(ticket);
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, Error> {
        // A cached object's texture is new GPU memory that nothing reads yet: made here,
        // without waiting for the picture being recorded (crowded rooms make one in nearly
        // every walk). The sizes too large are the backend's to refuse, and out of memory it
        // frees idle textures before it tries again: both wait.
        let limit = self.descriptors.limits.max_texture_dimension_2d;
        if !self.sync_only
            && width.get() <= limit
            && height.get() <= limit
            && let Some(texture) =
                crate::backend::empty_texture_apart(&self.descriptors, width.get(), height.get())
        {
            return Ok(texture);
        }
        self.backend_for("create_empty_texture")
            .0
            .create_empty_texture(width, height)
    }

    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, Error> {
        self.backend_for("register_bitmap").0.register_bitmap(bitmap)
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        region: PixelRegion,
    ) -> Result<(), Error> {
        self.backend_for("update_texture").0.update_texture(handle, bitmap, region)
    }

    fn create_context3d(&mut self, profile: Context3DProfile) -> Result<Box<dyn Context3D>, Error> {
        // A Stage3D context's objects are used by scripts directly, at any time: pictures
        // are recorded on this thread from now on.
        if !self.sync_only {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "Stage3D in use: pictures are recorded on the main thread again"
            );
        }
        self.sync_only = true;
        self.backend_for("create_context3d").0.create_context3d(profile)
    }

    fn debug_info(&self) -> Cow<'static, str> {
        lock(&self.backend).0.debug_info()
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn set_quality(&mut self, quality: StageQuality) {
        self.backend_for("set_quality").0.set_quality(quality)
    }

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, Error> {
        self.backend_for("compile_pixelbender_shader").0.compile_pixelbender_shader(shader)
    }

    fn run_pixelbender_shader(
        &mut self,
        handle: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, Error> {
        self.backend_for("run_pixelbender_shader")
            .0
            .run_pixelbender_shader(handle, arguments, target)
    }

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), Error> {
        self.backend_for("resolve_sync_handle").0.resolve_sync_handle(handle, with_rgba)
    }
}
