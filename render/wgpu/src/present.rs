//! AQW build: the present thread.
//!
//! The main thread used to finish the frame's command buffers, submit them, wait for the GPU
//! when two frames were already queued, then draw the window (the movie's picture and the
//! menu) and present it, all before it could run the next SWF frame's scripts. On a laptop
//! CPU that was 4-5 ms of a 43 ms frame in a crowded room. Now it hands those steps to this
//! thread as jobs (`run`) and goes on with the scripts; the jobs run in the order they were
//! sent, so the window is drawn after the movie it shows was submitted, and the GPU receives
//! exactly the same commands in the same order as before.
//!
//! ONE thread, on purpose. A present can block until the monitor's next refresh (with frame
//! interpolation every refresh is one), and the next movie submit then queues behind it, so
//! the main thread waits for it (`present_wait`, 0.3-3 ms per picture at 360 Hz). That wait
//! is the same vsync wait the main thread paid before, and it's what keeps the queue of
//! window frames bounded. Separate submit and present threads were tried and reverted: the
//! window frames piled up without limit (the main thread redraws twice per refresh, the
//! monitor shows one), and the movie's next picture could reach its single texture before
//! the window copied the previous one, so pictures were skipped or doubled. The pacer just
//! leaves the wait out of a picture's cost (see the desktop's redraw handler).
//!
//! What the main thread must not do while a frame is still on its way to the GPU queue:
//! write GPU memory the frame reads (upload a shape into a reused arena range, update a
//! bitmap, draw into a `BitmapData`), or submit work of its own before it (a flush of the
//! next frame in progress): the GPU would run those first and the frame would show the
//! newer data. So the renderer waits for the frame's submit (`wait_for`) before any of
//! those, which is normally over long before the next frame's scripts are (see
//! `WgpuRenderBackend::ensure_presented`; the wait is timed as `present_wait`).
//!
//! `RUFFLE_SERIAL_PRESENT` keeps everything on the main thread, for A/B comparisons. Only on
//! Windows so far: Linux (where OpenGL ties its context to a thread) and macOS haven't run
//! it; `RUFFLE_PRESENT_THREAD` turns it on there, to try.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;

pub type Job = Box<dyn FnOnce() + Send + 'static>;

struct Shared {
    /// Jobs finished (a job that panicked counts, so nobody waits for it forever).
    done: Mutex<u64>,
    done_changed: Condvar,
    /// What a job panicked with; the thread stops after that, and the next `run` or
    /// `wait_for` panics with it on the main thread (as the job would have before). Only in
    /// builds that unwind (dev): release and dist builds use `panic = "abort"`, so a panic
    /// ends the process right after the panic hook, on whichever thread it happens.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

pub struct PresentThread {
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    sent: AtomicU64,
    shared: Arc<Shared>,
}

impl PresentThread {
    /// Whether frames are submitted and shown from a present thread (see the module).
    pub fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("RUFFLE_SERIAL_PRESENT").is_none()
                && !cfg!(feature = "profile-with-tracy")
                && (cfg!(windows) || std::env::var_os("RUFFLE_PRESENT_THREAD").is_some())
        })
    }

    pub fn new(name: &str) -> Arc<Self> {
        Self::with_stack(name, None)
    }

    /// `new`, with a stack of `stack` bytes for the thread (the render thread records
    /// pictures as deeply nested as the main thread did: see `crate::render_thread`).
    pub fn with_stack(name: &str, stack: Option<usize>) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel::<Job>();
        let shared = Arc::new(Shared {
            done: Mutex::new(0),
            done_changed: Condvar::new(),
            panic: Mutex::new(None),
        });
        let worker = shared.clone();
        let mut builder = std::thread::Builder::new().name(name.into());
        if let Some(stack) = stack {
            builder = builder.stack_size(stack);
        }
        let thread = builder
            .spawn(move || {
                for job in receiver {
                    let result = catch_unwind(AssertUnwindSafe(job));
                    let failed = result.is_err();
                    if let Err(payload) = result {
                        *lock(&worker.panic) = Some(payload);
                    }
                    *lock(&worker.done) += 1;
                    worker.done_changed.notify_all();
                    if failed {
                        break;
                    }
                }
            })
            .unwrap_or_else(|error| panic!("spawning the {name} thread: {error}"));
        Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            thread: Mutex::new(Some(thread)),
            sent: AtomicU64::new(0),
            shared,
        })
    }

    /// Runs `job` after the jobs sent before it; returns its ticket (see `wait_for`).
    pub fn run(&self, job: Job) -> u64 {
        self.check_panic();
        let ticket = self.sent.fetch_add(1, Ordering::Relaxed) + 1;
        let sent = lock(&self.sender)
            .as_ref()
            .is_some_and(|sender| sender.send(job).is_ok());
        if !sent {
            self.check_panic();
            panic!("a submit/present thread has stopped");
        }
        ticket
    }

    /// Waits until the job with `ticket` is done.
    pub fn wait_for(&self, ticket: u64) {
        let mut done = lock(&self.shared.done);
        while *done < ticket {
            if lock(&self.shared.panic).is_some() {
                drop(done);
                self.check_panic();
                return;
            }
            done = self
                .shared
                .done_changed
                .wait(done)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(done);
        self.check_panic();
    }

    fn check_panic(&self) {
        if let Some(payload) = lock(&self.shared.panic).take() {
            resume_unwind(payload);
        }
    }
}

impl Drop for PresentThread {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop once the jobs sent are done.
        lock(&self.sender).take();
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
