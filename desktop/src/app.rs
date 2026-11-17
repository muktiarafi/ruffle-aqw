use crate::custom_event::{OpenType, RuffleEvent};
use crate::gui::{GuiController, MENU_HEIGHT};
use crate::player::{LaunchOptions, PlayerController};
use crate::preferences::GlobalPreferences;
use crate::util::{
    get_screen_size, gilrs_button_to_gamepad_button, winit_input_to_ruffle_key_descriptor,
    winit_to_ruffle_text_control,
};
use anyhow::Error;
use gilrs::{Event, EventType, Gilrs};
use ruffle_core::FloatDuration;
use ruffle_core::PlayerEvent;
use ruffle_core::events::{ImeEvent, ImeNotification, PlayerNotification};
use ruffle_core::swf::HeaderExt;
use ruffle_frontend_utils::content::ContentDescriptor;
use ruffle_render::backend::ViewportDimensions;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize, Size};
use winit::event::{ElementState, Ime, KeyEvent, Modifiers, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Icon, WindowAttributes, WindowId};

struct MainWindow {
    preferences: GlobalPreferences,
    gui: GuiController,
    player: PlayerController,
    minimized: bool,
    /// Set from `WindowEvent::Occluded`, which winit only reports on macOS and X11
    /// (and on X11 only without a compositor). Always false elsewhere.
    occluded: bool,
    mouse_pos: PhysicalPosition<f64>,
    /// Whether the cursor moved since the last hit-test. `CursorMoved` arrives at the
    /// mouse polling rate (1000+ Hz on gaming mice) and each `MouseMove` hit-tests the
    /// whole display list, which stalls crowded scenes. Moves are coalesced and applied
    /// at most once per movie frame, plus right before any click.
    mouse_move_pending: bool,
    /// AQW build (testing): a pointer fed to the movie instead of the real one (see
    /// `TestPointer`).
    test_pointer: Option<TestPointer>,
    /// AQW build (testing): see `TestHooks`.
    test_hooks: crate::test_hooks::TestHooks,
    modifiers: Modifiers,
    /// AQW build: frames shown since `fps_since`, for the FPS counter (see `count_fps`).
    fps_frames: u32,
    fps_since: Instant,
    min_window_size: LogicalSize<u32>,
    max_window_size: PhysicalSize<u32>,
    no_gui: bool,
    preferred_width: Option<f64>,
    preferred_height: Option<f64>,
    start_fullscreen: bool,
    loaded: LoadingState,
    time: Instant,
    next_frame_time: Option<Instant>,
    event_loop_proxy: EventLoopProxy<RuffleEvent>,
    perf: PerfStats,
    /// A SWF frame ran since the last draw; see `check_redraw`.
    frame_since_render: bool,
    last_redraw: Instant,
    /// With frame interpolation on: when the next in-between draw is due (see `pace_interpolation`).
    next_draw_time: Option<Instant>,
    /// When the last draw started.
    draw_started: Instant,
    /// Recent averages of a draw's and a SWF frame's time on the main thread (ms). AQW build:
    /// a draw of a game frame's picture (`draw_cost_ms`: it walks the stage, and more of its
    /// layers are new) and of an in-between picture (`between_cost_ms`: replayed, mostly
    /// cached layers) cost differently, so they're budgeted apart (see `pace_interpolation`).
    draw_cost_ms: f64,
    between_cost_ms: f64,
    /// AQW build: with the render thread, what a picture costs THAT thread (ms, averaged;
    /// see `pace_interpolation`). 0 without it.
    thread_cost_ms: f64,
    /// AQW build: with the render thread, how the pictures really followed each other: the
    /// time from one new picture to the next, and how long this thread waited for the render
    /// thread before it could hand a picture over (ms, averaged); when the last one started;
    /// and what was learned from them (see `PipelineLimit` and `pace_interpolation`).
    picture_gap_ms: f64,
    render_wait_ms: f64,
    picture_at: Option<Instant>,
    pipeline_limit: Option<PipelineLimit>,
    /// Since when the pictures have not followed each other as fast as aimed for.
    saturated_since: Option<Instant>,
    /// A step to fewer pictures made for that, while it is on trial: when, and the divisor
    /// before it. And after a trial failed: how long, and until when, none is made.
    limit_trial: Option<(Instant, u32, u64)>,
    limit_quiet: Duration,
    limit_quiet_until: Option<Instant>,
    /// When it was last logged that a step would cost too many pictures (once a minute).
    limit_note_at: Option<Instant>,
    /// Every frame sent to the window so far (a trial counts them).
    presents_total: u64,
    frame_cost_ms: f64,
    /// Draw a new picture on every n-th refresh of the monitor.
    draw_divisor: u32,
    /// AQW build: with the render thread, the window frames sent since the last picture that
    /// showed the same picture again (see the redraw handler).
    repeats_since_picture: u32,
    /// While interpolating: when the last new picture was due (see `pace_interpolation`).
    picture_due: Instant,
    /// AQW build: "uncapped" frame pacing (see `mugetsu::uncapped`): whether frames now wait
    /// for the monitor's refresh (they do not while uncapped), whether the pacer was in that
    /// mode on its last turn, and the time between pictures that an FPS limit asks for.
    vsync: bool,
    uncapped: bool,
    uncapped_interval: Option<Duration>,
    /// Since when `draw_divisor` would have gone up / down (it changes only if that lasts).
    divisor_up_since: Option<Instant>,
    divisor_down_since: Option<Instant>,
    /// The monitor's refresh interval, and when it was last looked up.
    refresh_interval: Duration,
    refresh_checked: Option<Instant>,
}

/// Logs frame cost and pacing every few seconds (target `ruffle_aqw_perf`), for comparing builds.
#[derive(Default)]
struct PerfStats {
    window_start: Option<Instant>,
    last_frame: Option<Instant>,
    frames: u32,
    game_frames: u64,
    windows_logged: u32,
    /// Highest orphaned-avatar threshold a retention report was logged for.
    retention_reports_at: usize,
    frame_ms_sum: f64,
    frame_ms_max: f64,
    gap_ms_max: f64,
    /// Gaps between frames of 1.5x the frame time or more: long enough to see.
    hitches: u32,
    renders: u32,
    /// Pictures sent to the screen (movie renders plus GUI-only repaints).
    presents: u32,
    render_ms_sum: f64,
    render_ms_max: f64,
    /// `perf_counters::timers` at the start of this window.
    timers_at: Option<[u64; ruffle_render_wgpu::perf_counters::timers::LEN]>,
    /// `ruffle_core::render_walk_micros` at the start of this window.
    walk_at: Option<u64>,
    /// `ruffle_core::interpolation_replay::stats` at the start of this window.
    replay_at: Option<[u64; 6]>,
    /// Drawn frames per second of the last logged window.
    last_drawn_fps: f64,
    /// With frame interpolation on: the draws per second aimed for, and what the pacer goes
    /// by (ms: a game frame, its picture, an in-between picture; see `pace_interpolation`).
    interpolation_rate: f64,
    interpolation_costs: [f64; 3],
    /// Uncapped frame pacing: `interpolation_rate` is then the FPS limit (infinite: none).
    interpolation_uncapped: bool,
}

impl PerfStats {
    /// `game_frames`: SWF frames this tick ran (more than 1 when catching up).
    fn frame(&mut self, now: Instant, frame_ms: f64, frame_time_ms: f64, game_frames: u64) {
        if let Some(last) = self.last_frame {
            let gap = now.duration_since(last).as_secs_f64() * 1000.0;
            self.gap_ms_max = self.gap_ms_max.max(gap);
            if gap >= frame_time_ms * 1.5 {
                self.hitches += 1;
            }
        }
        self.last_frame = Some(now);
        self.frames += 1;
        self.game_frames += game_frames;
        self.frame_ms_sum += frame_ms;
        self.frame_ms_max = self.frame_ms_max.max(frame_ms);
    }

    fn render(&mut self, render_ms: f64) {
        self.renders += 1;
        self.render_ms_sum += render_ms;
        self.render_ms_max = self.render_ms_max.max(render_ms);
    }

    /// Logs object counts every 12th window (~1 min); counting walks the whole display tree.
    /// Every 12th window (~1 min): logs object counts, and the first time orphaned avatars pass
    /// 40 and again past 120, a retention report showing what keeps them alive.
    fn log_object_counts(&mut self, player: &mut ruffle_core::Player) {
        self.windows_logged += 1;
        // TEST SWITCH: `RUFFLE_TEST_MEMORY_EVERY=<n>` logs every n-th window (5 s each)
        // instead of every 12th, so a memory test needs half a minute, not two.
        static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let every = *EVERY.get_or_init(|| {
            std::env::var("RUFFLE_TEST_MEMORY_EVERY")
                .ok()
                .and_then(|every| every.parse().ok())
                .filter(|every| *every >= 1)
                .unwrap_or(12)
        });
        if self.windows_logged as u64 % every != 1 % every {
            return;
        }
        let (orphans, on_stage, under_orphans, classes) = player.display_object_counts();
        tracing::info!(
            target: "ruffle_aqw_perf",
            "objects: {on_stage} on stage, {orphans} orphans holding {under_orphans} more; weak listeners registered: {:?} (method, anonymous); top orphan classes: {classes:?}",
            ruffle_core::Player::weak_listener_counts()
        );
        let renderer_report = ruffle_render_wgpu::render_thread::with_wgpu_backend(
            player.renderer_mut(),
            |r: &mut ruffle_render_wgpu::backend::WgpuRenderBackend<crate::gui::MovieView>| {
                r.aqw_memory_report()
            },
        )
        .unwrap_or_default();
        let (encoder_backlog, in_flight) = wgpu_core::aqw_stats();
        let (working_set, commit) = process_memory_mb();
        // AQW build: what Windows counts against the graphics chip's memory (see
        // `mugetsu::video_memory`).
        let video_memory = crate::mugetsu::video_memory()
            .map(|(budget, usage)| {
                format!(
                    "video memory {} / {} MB budget (transient textures apart: {}); ",
                    usage >> 20,
                    budget >> 20,
                    if ruffle_render_wgpu::transient_textures_enabled() { "on" } else { "off" }
                )
            })
            .unwrap_or_default();
        tracing::info!(
            target: "ruffle_aqw_perf",
            "memory: {video_memory}{}; {renderer_report}; process commit {commit} MB, working set {working_set} MB, rust heap {} MB, {}; wgpu: {encoder_backlog} encoders waiting to drop, {in_flight} submissions in flight",
            player.aqw_memory_report(),
            crate::HEAP_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1 << 20),
            crate::memory_split::report(),
        );
        if let Some((copies, draws)) = ruffle_core::Player::sharp_snapshot_counts() {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "sharp snapshots: {copies} made, {draws} draws so far"
            );
        }
        crate::memory_split::collect_if_flagged(|| {
            let mut commit = 0usize;
            let mut zero = [0usize; 7];
            // SAFETY: plain out-pointers.
            unsafe {
                let [a, b, c, d, e, f, g] = &mut zero;
                libmimalloc_sys::mi_process_info(a, b, c, d, e, &mut commit, f, g);
            }
            commit as u64
        });
        crate::heap_census::report();
        // `RUFFLE_RETENTION_TEST=<class>` runs one report at startup, to check the tool itself.
        if self.windows_logged == 1
            && let Ok(class) = std::env::var("RUFFLE_RETENTION_TEST")
        {
            player.log_retention_paths(&class, 3);
        }
        let avatars = classes
            .iter()
            .find(|(class, _)| class == "AvatarMC")
            .map_or(0, |(_, n)| *n);
        // The report walks the whole heap (a noticeable pause), so it only runs when asked for
        // with `RUFFLE_RETENTION_REPORTS` (it found the avatar leak; kept for future leaks).
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled =
            *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_RETENTION_REPORTS").is_some());
        let threshold = [40, 120]
            .into_iter()
            .filter(|_| enabled)
            .find(|&t| self.retention_reports_at < t && avatars >= t);
        if let Some(threshold) = threshold {
            self.retention_reports_at = threshold;
            player.log_retention_paths("AvatarMC", 3);
        }
        // A `heap-census` file next to the exe: what's in memory, at 2 minutes and then every
        // 30 (a short pause each), to find what grows in long sessions.
        static CENSUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CENSUS.get_or_init(|| crate::mugetsu::exe_flag("heap-census"))
            && (self.windows_logged == 25 || self.windows_logged % 360 == 1 && self.windows_logged > 1)
        {
            player.log_heap_census(40);
        }
    }

    /// Returns whether a window was logged.
    fn maybe_log(&mut self, now: Instant) -> bool {
        use ruffle_render_wgpu::perf_counters::timers;
        let start = *self.window_start.get_or_insert(now);
        let timers_at = *self
            .timers_at
            .get_or_insert_with(timers::snapshot);
        let walk_at = *self.walk_at.get_or_insert_with(ruffle_core::render_walk_micros);
        let replay_at = *self
            .replay_at
            .get_or_insert_with(ruffle_core::interpolation_replay::stats);
        let secs = now.duration_since(start).as_secs_f64();
        if secs < 5.0 {
            return false;
        }
        let avg = |sum: f64, n: u32| if n == 0 { 0.0 } else { sum / n as f64 };
        tracing::info!(
            target: "ruffle_aqw_perf",
            "game fps {:.1}, drawn {:.1}, presented {:.1} | frame avg {:.1}ms max {:.1}ms | worst gap {:.0}ms, hitches {} | render avg {:.1}ms max {:.1}ms",
            self.game_frames as f64 / secs,
            self.renders as f64 / secs,
            self.presents as f64 / secs,
            avg(self.frame_ms_sum, self.frames),
            self.frame_ms_max,
            self.gap_ms_max,
            self.hitches,
            avg(self.render_ms_sum, self.renders),
            self.render_ms_max,
        );
        let counts = ruffle_render_wgpu::perf_counters::take();
        let per_render = ruffle_render_wgpu::perf_counters::NAMES
            .iter()
            .zip(counts)
            .enumerate()
            .filter(|&(i, (_, n))| i == 0 || n > 0)
            .map(|(_, (name, n))| format!("{name} {:.1}", avg(n as f64, self.renders)))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(target: "ruffle_aqw_perf", "per drawn frame: {per_render}");
        // Where the render time went, per drawn frame (the rest of "render avg" is recording
        // the frame's commands on the main thread).
        let timers_now = timers::snapshot();
        let parts = timers::NAMES
            .iter()
            .zip(timers_now.iter().zip(timers_at))
            .filter(|(_, (a, b))| *a > b)
            .map(|(name, (a, b))| {
                format!("{name} {:.2}", avg((a - b) as f64 / 1000.0, self.renders))
            })
            .collect::<Vec<_>>()
            .join(", ");
        // AQW build: a `no-atlas` file next to the exe turns the layer atlas off from the next
        // window on, for A/B comparisons in the same scene.
        let atlas_was_on = ruffle_render_wgpu::atlas_enabled_now();
        ruffle_render_wgpu::set_atlas_enabled(!crate::mugetsu::exe_flag("no-atlas"));
        // Same for the cacheAsBitmap size steps (`no-cache-steps`).
        let steps_were_on = ruffle_core::cache_size_steps();
        ruffle_core::set_cache_size_steps(!crate::mugetsu::exe_flag("no-cache-steps"));
        // And for batched cacheAsBitmap filters (`no-glow-batch`).
        let batch_was_on = ruffle_render_wgpu::cache_batch_enabled_now();
        ruffle_render_wgpu::set_cache_batch_enabled(!crate::mugetsu::exe_flag("no-glow-batch"));
        // And for cached blend layers (`no-layer-cache`, see `ruffle_render_wgpu::layer_cache`).
        let layers_were_on = ruffle_render_wgpu::layer_cache::enabled();
        ruffle_render_wgpu::layer_cache::set_enabled(!crate::mugetsu::exe_flag("no-layer-cache"));
        tracing::info!(
            target: "ruffle_aqw_perf",
            "render time per drawn frame (ms): build_commands {:.2}, {parts}; atlas {}; cache steps {}; glow batch {}; present thread {}; render thread {}; layer cache {}; interpolation {}",
            avg((ruffle_core::render_walk_micros() - walk_at) as f64 / 1000.0, self.renders),
            if atlas_was_on { "on" } else { "off" },
            if steps_were_on { "on" } else { "off" },
            if batch_was_on { "on" } else { "off" },
            if ruffle_render_wgpu::present::PresentThread::enabled() { "on" } else { "off" },
            if ruffle_render_wgpu::render_thread::enabled() { "on" } else { "off" },
            if layers_were_on { "on" } else { "off" },
            if ruffle_core::frame_interpolation::enabled() {
                let [frame, picture, between] = self.interpolation_costs;
                let aim = if !self.interpolation_uncapped {
                    format!("aiming for {:.0}/s", self.interpolation_rate)
                } else if self.interpolation_rate.is_finite() {
                    format!("uncapped, at most {:.0}/s", self.interpolation_rate)
                } else {
                    "uncapped".to_string()
                };
                format!(
                    "on, {aim} (ms: a game frame {frame:.1}, its picture {picture:.1}, an in-between {between:.1}; {:.0} a second to spend)",
                    interpolation_budget_ms()
                )
            } else {
                "off".to_string()
            }
        );
        // In-between pictures drawn from the last recorded one (see `replay`); a `no-replay`
        // file next to the exe walks the stage for every picture instead (A/B comparisons).
        use ruffle_core::interpolation_replay as replay;
        let replay_now = replay::stats();
        if replay_now != replay_at {
            let counts = replay::STAT_NAMES
                .iter()
                .zip(replay_now.iter().zip(replay_at))
                .filter(|(_, (a, b))| *a > b)
                .map(|(name, (a, b))| format!("{name} {}", a - b))
                .collect::<Vec<_>>()
                .join(", ");
            let reason = replay::take_reason()
                .map(|why| format!("; last reason: {why}"))
                .unwrap_or_default();
            tracing::info!(target: "ruffle_aqw_perf", "interpolation replay: {counts}{reason}");
        }
        replay::set_enabled(!crate::mugetsu::exe_flag("no-replay"));
        *self = PerfStats {
            window_start: Some(now),
            last_frame: self.last_frame,
            windows_logged: self.windows_logged,
            retention_reports_at: self.retention_reports_at,
            timers_at: Some(timers_now),
            walk_at: Some(ruffle_core::render_walk_micros()),
            replay_at: Some(replay_now),
            last_drawn_fps: self.renders as f64 / secs,
            interpolation_rate: self.interpolation_rate,
            interpolation_costs: self.interpolation_costs,
            interpolation_uncapped: self.interpolation_uncapped,
            ..Default::default()
        };
        true
    }
}

/// AQW build: when Mugetsu uses most of the graphics memory Windows allows it (85%), it frees
/// its spare textures and keeps far fewer from then on, until usage has stayed under 65% for a
/// minute (going back sooner refilled the spares within seconds and needed freeing again: each
/// round made the spares be recreated, costing hitches). Past the budget, Windows moves
/// graphics memory to system RAM (2-3 FPS on a 4 GB card after a long session), and Vulkan runs
/// out of memory instead (gear of players arriving not drawn). Checked every 5 s; does nothing
/// while there's room, so big cards are unaffected.
use crate::mugetsu::process_memory_mb;

/// AQW build: what the memory limit is compared with, in MB: the figure Task Manager shows
/// (see `mugetsu::task_manager_mb`; it was the committed memory until 2026-11-16).
fn process_commit_mb() -> u64 {
    crate::mugetsu::task_manager_mb()
}

/// What one `free_memory` did, for the log.
struct Freed {
    /// Shapes not drawn in the last 30 seconds, now gone.
    shapes: usize,
    /// (working set, committed) in MB, before and after.
    before: (u64, u64),
    after: (u64, u64),
    /// How long the three steps took on this thread, in ms: the shapes, the spare textures,
    /// the heap pages.
    took_ms: [f32; 3],
    /// Whether the heap's unused pages were returned on this thread (`after` has them) or
    /// are being returned in the background (a log line of its own says when and how much).
    heap_waited: bool,
    /// The thorough way only: (loaded SWFs nothing used any more, now freed; the
    /// milliseconds that took).
    libraries: Option<(usize, f64)>,
}

impl std::fmt::Display for Freed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [shapes, textures, heap] = self.took_ms;
        write!(
            f,
            "{} -> {} MB committed (working set {} -> {} MB); {} shapes not drawn in the last 30 s, the spare textures and unused heap pages freed in {:.1} ms (shapes {shapes:.1}, textures {textures:.1}, heap pages {heap:.1}){}",
            self.before.1,
            self.after.1,
            self.before.0,
            self.after.0,
            self.shapes,
            shapes + textures + heap,
            if self.heap_waited { "" } else { "; the heap's pages go back in the background" },
        )?;
        if let Some((libraries, ms)) = self.libraries {
            write!(f, "; before that, {libraries} loaded SWFs nothing used any more freed in {ms:.0} ms")?;
        }
        Ok(())
    }
}

/// TEST SWITCH: `RUFFLE_HEAP_RETURN=wait`: freeing memory returns the heap's unused pages on
/// the main thread and waits for every worker, as it did (for comparisons).
fn heap_return_waits() -> bool {
    static WAITS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WAITS.get_or_init(|| std::env::var("RUFFLE_HEAP_RETURN").is_ok_and(|value| value == "wait"))
}

/// AQW build: frees everything that can be freed without the picture changing: the shapes
/// not drawn in the last 30 seconds (what is on screen is drawn in every picture), the spare
/// textures and the layer cache, and the allocator's free pages. For the memory limit and
/// "Free memory now".
///
/// The heap's pages were returned here, with a forced collect on this thread and on every
/// worker, waited for: 10-16 ms in a test room, but 27-33 ms of a 37-42 ms freeing in a real
/// session with a 3 GB limit (8 thousand pages, 5 thousand of them abandoned by their
/// threads): a game frame late every time the limit worked. Returning memory to the system
/// is the allocator's business for the whole process, not this thread's: now every thread
/// only hands its own free pages back to the allocator (cheap; the workers when they are next
/// free), and a thread started for it does the forced collect that gives them to the system.
///
/// That returns less: measured in a test room, the working set fell by 30-55 MB instead of
/// 110-140 (only a forced collect on the main thread takes back the pages other threads
/// abandoned), while the committed memory, which is mostly shapes and textures, fell as
/// before. So the memory limit, which works by itself during play, frees this way, and
/// "Free memory now", a click in the menu where a pause of 40 ms is not seen, frees the
/// `thorough` way: forced, on this thread and every worker, waited for.
fn free_memory(player: &mut ruffle_core::Player, thorough: bool) -> Freed {
    let before = process_memory_mb();
    // (A click only: every loaded SWF nothing uses any more goes now, not within the next
    // GC cycles: two whole cycles in one go, a pause the limit's own freeing must not make.)
    let libraries = thorough.then(|| player.free_unused_libraries_now());
    let started = Instant::now();
    let shapes = ruffle_core::evict_unused_shapes(30);
    let shapes_done = Instant::now();
    ruffle_render_wgpu::render_thread::with_wgpu_backend(
        player.renderer_mut(),
        |r: &mut ruffle_render_wgpu::backend::WgpuRenderBackend<crate::gui::MovieView>| {
            r.aqw_trim_memory()
        },
    );
    let textures_done = Instant::now();
    let heap_waited = thorough || heap_return_waits();
    // SAFETY: plain calls into the allocator, each on a thread that allocates with it.
    if heap_waited {
        ruffle_render_wgpu::on_every_worker(|| unsafe {
            libmimalloc_sys::mi_collect(true);
        });
        unsafe { libmimalloc_sys::mi_collect(true) };
    } else {
        unsafe { libmimalloc_sys::mi_collect(false) };
        ruffle_render_wgpu::on_every_worker_later(|| unsafe {
            libmimalloc_sys::mi_collect(false);
        });
        let spawned = std::thread::Builder::new()
            .name("free memory".to_string())
            .spawn(|| {
                // (The workers get to theirs first.)
                std::thread::sleep(Duration::from_millis(100));
                let started = Instant::now();
                unsafe { libmimalloc_sys::mi_collect(true) };
                let took = started.elapsed().as_secs_f32() * 1000.0;
                let (working_set, commit) = process_memory_mb();
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "the heap's unused pages went back to the system in {took:.1} ms in the background: now {commit} MB committed (working set {working_set} MB)"
                );
            });
        if let Err(error) = spawned {
            tracing::warn!("free memory: no thread for the heap's pages ({error})");
        }
    }
    let heap_done = Instant::now();
    let ms = |from: Instant, to: Instant| (to - from).as_secs_f32() * 1000.0;
    Freed {
        shapes,
        before,
        after: process_memory_mb(),
        took_ms: [
            ms(started, shapes_done),
            ms(shapes_done, textures_done),
            ms(textures_done, heap_done),
        ],
        heap_waited,
        libraries,
    }
}

/// How long after freeing memory for the player's limit the next freeing may come, at the
/// least, in seconds.
const LIMIT_WAIT_SECS: u64 = 20;
/// And at the most, while the freeing does not hold (see `memory_watch`).
const LIMIT_WAIT_MAX_SECS: u64 = 300;

fn memory_watch(player: &mut ruffle_core::Player) {
    struct Watch {
        pressure: bool,
        low_since: Option<Instant>,
        last_log: Option<Instant>,
        last_sweep: Option<Instant>,
        /// When memory was last freed for the player's memory limit.
        last_limit_trim: Option<Instant>,
        /// How long to wait after that before freeing again, in seconds.
        limit_wait_secs: u64,
        /// The limit those two belong to: another limit starts afresh.
        limit_seen_mb: u64,
    }
    static WATCH: std::sync::Mutex<Watch> = std::sync::Mutex::new(Watch {
        pressure: false,
        low_since: None,
        last_log: None,
        last_sweep: None,
        last_limit_trim: None,
        limit_wait_secs: LIMIT_WAIT_SECS,
        limit_seen_mb: 0,
    });
    let mut watch = WATCH.lock().unwrap_or_else(|e| e.into_inner());
    // AQW build: on every GPU (and every OS), once a minute, free the tessellations of shapes
    // not drawn for a while: other maps, the gear of players who left. AQW keeps every SWF
    // it has loaded, so shape geometry otherwise grows for the whole session (800 MB in four
    // minutes of town hopping on a 32 GB card, where the memory-pressure path below never
    // runs). A shape drawn again later is tessellated again, as on its first draw. A
    // `no-shape-eviction` file next to the exe turns it off (A/B).
    //
    // "A while" was always 10 minutes, and ten minutes of going from room to room held 69
    // thousand shapes, 1 GB of geometry (which is the system's RAM on built-in graphics). So
    // the fuller the store, the sooner an unused shape goes: 10 minutes while the store is
    // small, 3 minutes above `mugetsu::shape_store_mb`, 2 minutes above twice that. What is
    // on screen is drawn every picture and is never among them.
    if crate::mugetsu::shape_eviction_enabled()
        && watch
            .last_sweep
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(60))
    {
        watch.last_sweep = Some(Instant::now());
        let store_mb = (ruffle_render_wgpu::perf_counters::live().1.max(0) as u64) >> 20;
        let soft_mb = crate::mugetsu::shape_store_mb();
        let unused_secs = if soft_mb == 0 || store_mb <= soft_mb {
            600
        } else if store_mb <= soft_mb * 2 {
            180
        } else {
            120
        };
        let freed = ruffle_core::evict_unused_shapes(unused_secs);
        if freed > 0 {
            let now_mb = (ruffle_render_wgpu::perf_counters::live().1.max(0) as u64) >> 20;
            tracing::info!(
                target: "ruffle_aqw_perf",
                "freed {freed} shapes not drawn for {} minutes (shape geometry: {store_mb} -> {now_mb} MB)",
                unused_secs / 60
            );
        }
    }
    // AQW build: the player's own memory limit (see `mugetsu::memory_limit_mb`): over it,
    // everything that can be freed is, at most every 20 seconds (what is freed and needed
    // again is made again, and freeing in every window would do nothing else).
    //
    // A scene can need more than the limit: with 2 GB set, towns and boss rooms on DirectX 12
    // held 2.2-2.35 GB right after each freeing and 2.5 GB twenty seconds later, so the same
    // spare textures and heap pages were freed and made again every 20 seconds for as long as
    // the limit stood, each round a game frame arriving 15 ms late in the boss rooms (the
    // user's session of 2026-11-15). So when memory is over the limit again at the first
    // chance after a freeing, that freeing did not hold, and the wait before the next one
    // doubles, up to 5 minutes; a freeing that held for a while puts it back to 20 seconds.
    let limit_mb = u64::from(crate::mugetsu::memory_limit_mb());
    if limit_mb != watch.limit_seen_mb {
        watch.limit_seen_mb = limit_mb;
        watch.limit_wait_secs = LIMIT_WAIT_SECS;
        watch.last_limit_trim = None;
    }
    if limit_mb != 0
        && watch
            .last_limit_trim
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(watch.limit_wait_secs))
        && process_commit_mb() > limit_mb
    {
        let held = watch
            .last_limit_trim
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(watch.limit_wait_secs * 2 + 10));
        watch.limit_wait_secs = if held {
            LIMIT_WAIT_SECS
        } else {
            (watch.limit_wait_secs * 2).min(LIMIT_WAIT_MAX_SECS)
        };
        watch.last_limit_trim = Some(Instant::now());
        let freed = free_memory(player, false);
        tracing::info!(
            target: "ruffle_aqw_perf",
            "memory limit {limit_mb} MB: {freed}{}; the next freeing in {} s at the soonest",
            if held { "" } else { "; the last freeing did not hold" },
            watch.limit_wait_secs
        );
    }
    let Some((budget, usage)) = crate::mugetsu::video_memory() else {
        return;
    };
    let budget = crate::mugetsu::test_video_budget().map_or(budget, |test| test.min(budget));
    if usage > budget / 100 * 85 {
        // Shapes not drawn for 2 minutes (gear of players who left, other maps) are
        // tessellated again if they come back.
        let shapes = ruffle_core::evict_unused_shapes(120);
        ruffle_render_wgpu::render_thread::with_wgpu_backend(
            player.renderer_mut(),
            |r: &mut ruffle_render_wgpu::backend::WgpuRenderBackend<crate::gui::MovieView>| {
                r.aqw_trim_memory()
            },
        );
        ruffle_render_wgpu::set_memory_pressure(true);
        let was = std::mem::replace(&mut watch.pressure, true);
        watch.low_since = None;
        if !was || watch.last_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(60)) {
            watch.last_log = Some(Instant::now());
            let after = crate::mugetsu::video_memory().map_or(0, |(_, usage)| usage);
            tracing::info!(
                target: "ruffle_aqw_perf",
                "graphics memory nearly full: {} of {} MB; freed spare textures and {shapes} unused shapes, now {} MB",
                usage >> 20,
                budget >> 20,
                after >> 20
            );
        }
    } else if watch.pressure {
        if usage < budget / 100 * 65 {
            let low_since = *watch.low_since.get_or_insert_with(Instant::now);
            if low_since.elapsed() >= Duration::from_secs(60) {
                watch.pressure = false;
                watch.low_since = None;
                ruffle_render_wgpu::set_memory_pressure(false);
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "graphics memory back to normal: {} of {} MB",
                    usage >> 20,
                    budget >> 20
                );
            }
        } else {
            watch.low_since = None;
        }
    }
}

/// AQW build: the time each `perf_counters::timers` timer added since `before`, in ms
/// (only the nonzero ones).
/// AQW build: how much of each second (ms) the render thread may spend recording pictures
/// (see `MainWindow::pace_interpolation`).
const RENDER_THREAD_BUDGET_MS: f64 = 850.0;

/// AQW build: with the render thread a picture passes three stages after this thread: the
/// render thread records it, the present thread submits it, the GPU draws it. The pacer
/// budgets this thread's time and the render thread's, but nothing told it when a later
/// stage could not keep up. In a crowded Yulgar (250 blended layers a picture) the GPU got
/// through 190 pictures a second; this thread's share of a picture was 1-1.6 ms and the
/// render thread's 2.2, so the pacer aimed for 360, and each picture went out once, whenever
/// it was through: 190 uneven pictures (and 190 on the FPS counter) where 180 pictures shown
/// twice each was the even choice it makes without the render thread.
///
/// The sign of it: this thread waits for the render thread before every picture
/// (`render_wait`; the render thread waits for the present thread, which waits for the GPU)
/// and the pictures follow each other slower than aimed for. The time between them then IS
/// what a picture takes to get through, so it's remembered, as a multiple of the render
/// thread's own time for a picture (which follows the scene into lighter and heavier rooms),
/// and the pacer aims for no more than fits (see `pace_interpolation`). Nothing measures the
/// later stages while they keep up, so the limit is believed for a while only (`hold`): after
/// that the pacer may try more pictures again, and finds the limit again within a third of a
/// second if it still stands; every time it does, it is believed twice as long (10 s to
/// 5 minutes).
struct PipelineLimit {
    per_thread_ms: f64,
    learned: Instant,
    hold: Duration,
}

/// How much of each second (ms) pictures may take to get through at the pace learned (see
/// `PipelineLimit`): that pace was measured with nothing to spare, so nearly all of it.
const PIPELINE_BUDGET_MS: f64 = 970.0;

/// How long a step to fewer pictures made for a `PipelineLimit` is on trial: if the pictures
/// fall behind again within this time, it did not help and is undone (see
/// `MainWindow::pace_interpolation`). Finding that takes about a second at 60-90 pictures.
const LIMIT_TRIAL: Duration = Duration::from_secs(4);

/// How far the pictures must fall behind the pace aimed for to count as falling behind (the
/// time between them, as a multiple of the aim).
const LIMIT_SHORTFALL: f64 = 1.025;

/// The most a step to an even pace may cost, as a share of the pictures getting through now.
/// What you see move is the new pictures, and the next even pace can be far below what gets
/// through (360 to 180 a second on a 360 Hz monitor): a 30-player room that got 340 of 360
/// through (one in 17 a refresh late) is better left at 340 than paced at an even 180, though
/// the FPS counter then reads 340, not 360; a built-in Radeon that got 64 through is better
/// left at 64 than at an even 51. A crowded Yulgar got 190 of 360 through: an even 180 costs
/// a twentieth and every picture stays up the same time.
const LIMIT_LOSS: f64 = 1.0 / 6.0;

/// AQW build: how long this thread waited for the present thread and for the render thread
/// since `before` (ms): not a frame's or a picture's own cost. (The render thread's wait for
/// the present thread is timed in the same `present_wait` and taken out again.)
fn main_thread_wait_ms(before: &[u64; ruffle_render_wgpu::perf_counters::timers::LEN]) -> f64 {
    use ruffle_render_wgpu::perf_counters::timers;
    let now = timers::snapshot();
    let since = |timer: usize| (now[timer] - before[timer]) as f64 / 1000.0;
    (since(timers::PRESENT_WAIT) - since(timers::RENDER_PRESENT_WAIT)).max(0.0)
        + since(timers::RENDER_WAIT)
}

fn timer_parts(before: &[u64; ruffle_render_wgpu::perf_counters::timers::LEN]) -> String {
    use ruffle_render_wgpu::perf_counters::timers;
    let now = timers::snapshot();
    // AQW build: the present thread's timers ("present: ...") ran alongside this thread, so
    // they're not part of its time: listed after it.
    let list = |on_present_thread: bool| {
        timers::NAMES
            .iter()
            .zip(now.iter().zip(before))
            .filter(|(name, (a, b))| {
                (name.starts_with("present: ") || name.starts_with("render thread: "))
                    == on_present_thread
                    && *a > *b
            })
            .map(|(name, (a, b))| format!("{name} {:.1}", (a - b) as f64 / 1000.0))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let (here, meanwhile) = (list(false), list(true));
    let here = if here.is_empty() { "none of the timed work".to_string() } else { here };
    if meanwhile.is_empty() {
        here
    } else {
        format!("{here}; meanwhile on the present and render threads: {meanwhile}")
    }
}

impl MainWindow {
    pub fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        if matches!(event, WindowEvent::RedrawRequested) {
            // Don't render when nobody can see the result: while minimized, the
            // surface also keeps the size it had before, as `GuiController::resize`
            // refuses to reconfigure it to a zero size.
            if !self.minimized && !self.occluded {
                let started = Instant::now();
                let timers_before = ruffle_render_wgpu::perf_counters::timers::snapshot();
                let walk_before = ruffle_core::render_walk_micros();
                let mut player = self.player.get();
                let frame_budget_ms = player
                    .as_ref()
                    .map_or(1000.0 / 24.0, |player| 1000.0 / player.frame_rate());
                let mut rendered_movie = false;
                let mut replayed_picture = false;
                let mut interpolating = false;
                let mut uncapped = false;
                if let Some(ref mut player) = player {
                    // AQW build: egui asks for a repaint on every mouse move, and each repaint
                    // used to re-render the whole movie too. The movie view keeps the last
                    // frame, so only re-render it when a new SWF frame ran (or it was resized).
                    // Even if the movie is paused, user interaction with debug tools can change the render output
                    // With frame interpolation on, a new in-between picture goes out on the
                    // first refresh after it's due, every n refreshes' time; the refreshes in
                    // between show the last one again (see `pace_interpolation`). Timed, not
                    // counted: a long draw (e.g. 11 ms in Yulgar) makes presents miss refreshes.
                    interpolating = self.next_draw_time.is_some();
                    // Uncapped (see `pace_interpolation`): a new picture every time, or as
                    // often as the FPS limit allows.
                    uncapped = interpolating && self.uncapped;
                    let new_picture_due = if uncapped {
                        match self.uncapped_interval {
                            None => true,
                            Some(interval) => {
                                let due = self.picture_due + interval;
                                if started + Duration::from_micros(300) >= due {
                                    self.picture_due =
                                        if started < due + interval { due } else { started };
                                    true
                                } else {
                                    false
                                }
                            }
                        }
                    } else if interpolating {
                        let interval = self.refresh_interval * self.draw_divisor;
                        let due = self.picture_due + interval;
                        if started + self.refresh_interval / 2 >= due {
                            // Keep the rhythm, unless it fell more than a picture behind.
                            self.picture_due = if started < due + interval { due } else { started };
                            true
                        } else {
                            false
                        }
                    } else {
                        self.frame_since_render
                    };
                    if new_picture_due || !player.is_playing() || extra_renders_enabled() {
                        let replayed_before = ruffle_core::interpolation_replay::stats()[1];
                        player.render();
                        rendered_movie = true;
                        replayed_picture =
                            ruffle_core::interpolation_replay::stats()[1] > replayed_before;
                    }
                }
                let movie_ms = started.elapsed().as_secs_f64() * 1000.0;
                // AQW build: with the render thread, what the last picture cost that thread.
                let thread_micros = player.as_deref().and_then(|player| {
                    ruffle_render_wgpu::render_thread::picture_micros::<crate::gui::MovieView>(
                        player.renderer(),
                    )
                });

                // AQW build: with the render thread, this thread has time to spare between
                // pictures and sent more window frames than the monitor shows. Each takes a
                // refresh on the present thread, and the pictures' submits queue behind
                // them: the extra ones cost pictures (168 a second instead of 180 in a
                // 30-player room). So a picture is followed by one window frame for each
                // further refresh it stays up, no more (unless the menus want one). Without
                // the render thread this thread is paced by its own work, as before.
                // Uncapped: every frame sent is a new picture, none is shown again (the
                // menus still get the frames they ask for). With an FPS limit the menus'
                // own frames keep to it as well: the counter counts every frame sent, and
                // with the limit's "Custom" field open (a field with the keyboard asks for
                // a frame all the time) it read 346 for a limit of 144, as if the limit had
                // not been taken. Not under 60 a second though, so that a menu stays smooth
                // at a low limit.
                if uncapped && !rendered_movie {
                    let menus_want = self.gui.needs_render() && self.gui.wants_redraws();
                    // (TEST SWITCH: `RUFFLE_TEST_MENU_FRAMES_FREE=1`: as it was, for comparisons.)
                    static FREE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let free = *FREE.get_or_init(|| {
                        std::env::var_os("RUFFLE_TEST_MENU_FRAMES_FREE").is_some()
                    });
                    let menus_due = match self.uncapped_interval {
                        Some(interval) if !free => {
                            self.last_redraw.elapsed() >= interval.min(Duration::from_micros(16_600))
                        }
                        _ => true,
                    };
                    if !(menus_want && menus_due) {
                        self.draw_started = started;
                        return;
                    }
                }
                if interpolating && !uncapped && thread_micros.is_some() {
                    if rendered_movie {
                        self.repeats_since_picture = 0;
                    } else if self.repeats_since_picture + 1 >= self.draw_divisor
                        && !(self.gui.needs_render() && self.gui.wants_redraws())
                    {
                        // (Asked for again half a refresh from now: see `pace_interpolation`.)
                        self.draw_started = started;
                        return;
                    } else {
                        self.repeats_since_picture += 1;
                    }
                }

                self.gui.render(player);
                self.perf.presents += 1;
                self.presents_total += 1;
                self.count_fps();
                self.frame_since_render = false;
                self.last_redraw = Instant::now();
                self.draw_started = started;
                if rendered_movie {
                    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
                    // Drawing the movie only (presenting can include waiting for the refresh);
                    // one-off slow draws, e.g. of new shapes, count at most 3x the average.
                    // With the present thread, the wait for the refresh lands inside the
                    // movie's draw instead (the thread is still presenting the last window
                    // frame when the next picture's submit is handed to it, see
                    // `ruffle_render_wgpu::present`): it isn't the picture's cost either, so
                    // it's left out, or the pacer drew half the pictures it could (0.3-3 ms
                    // per picture at 360 Hz, against a real cost of ~2 ms).
                    // With the render thread (`ruffle_render_wgpu::render_thread`) the wait
                    // for the refresh is the thread's; what this thread waits for is the
                    // thread itself (`render_wait`), left out as well: the thread's own time
                    // is budgeted apart (see `pace_interpolation`).
                    let waited_ms = main_thread_wait_ms(&timers_before);
                    if let Some(micros) = thread_micros {
                        self.thread_cost_ms =
                            self.thread_cost_ms * 0.9 + micros as f64 / 1000.0 * 0.1;
                        // How the pictures really follow each other (see `PipelineLimit`).
                        use ruffle_render_wgpu::perf_counters::timers;
                        let waited = timers::snapshot()[timers::RENDER_WAIT]
                            .saturating_sub(timers_before[timers::RENDER_WAIT]);
                        self.render_wait_ms =
                            self.render_wait_ms * 0.95 + waited as f64 / 1000.0 * 0.05;
                        if let Some(last) = self.picture_at.replace(started)
                            && interpolating
                        {
                            let gap = started.duration_since(last).as_secs_f64() * 1000.0;
                            self.picture_gap_ms = if self.picture_gap_ms > 0.0 {
                                // Averaged over some 30 pictures: falling 6 % short is 16
                                // pictures on time and one a refresh late, and a quicker
                                // average dipped under the mark between the late ones (340
                                // pictures a second went unnoticed for 10 s).
                                // (A pause, as a long game frame, is not the pictures' pace.)
                                self.picture_gap_ms * 0.97
                                    + gap.min(self.picture_gap_ms * 2.0) * 0.03
                            } else {
                                // (Not the first gap: a movie's first pictures are 50-100 ms
                                // apart, and this average kept that for a second.)
                                (self.refresh_interval * self.draw_divisor).as_secs_f64() * 1000.0
                            };
                        }
                    }
                    let average = if replayed_picture {
                        &mut self.between_cost_ms
                    } else {
                        &mut self.draw_cost_ms
                    };
                    let cost = (movie_ms - waited_ms).max(0.0).min(*average * 3.0);
                    *average = *average * 0.9 + cost * 0.1;
                    self.perf.render(total_ms);
                    self.test_hooks.picture(total_ms);
                    // AQW build: say what a slow render (longer than 1.2 frames) spent its
                    // time on.
                    if total_ms > frame_budget_ms * 0.9 {
                        let parts = timer_parts(&timers_before);
                        let walk_ms =
                            (ruffle_core::render_walk_micros() - walk_before) as f64 / 1000.0;
                        tracing::info!(
                            target: "ruffle_aqw_perf",
                            "slow render {total_ms:.0} ms: movie {movie_ms:.0} ms, window/present {:.0} ms; build_commands {walk_ms:.1}, {parts} (ms)",
                            total_ms - movie_ms
                        );
                    }
                }
            }

            // Important that we return here, or we'll get a feedback loop with egui
            // (winit says redraw, egui hears redraw and says redraw, we hear redraw and tell winit to redraw...)
            return;
        }

        // AQW build: the window's surface can't be reconfigured while another thread is
        // submitting work: wgpu waits for the GPU, finds more work queued and fails ("Failed
        // to wait for GPU to come idle before reconfiguring the Surface"), which is fatal.
        // The present thread submits the movie's pictures, so that could happen without the
        // render thread too (once in 120 quick resizes on DirectX 12); with it a picture is
        // on its way nearly all the time (the second resize failed on Vulkan). So first
        // everything on its way is let through: the picture being recorded and the frames
        // queued behind it, then what the present thread was sent.
        if matches!(event, WindowEvent::Resized(_)) {
            if let Some(mut player) = self.player.get() {
                ruffle_render_wgpu::render_thread::settle::<crate::gui::MovieView>(
                    player.renderer_mut(),
                );
            }
            self.gui.settle_present();
        }
        if self.gui.handle_event(&event) {
            // Event consumed by GUI.
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                // AQW build: shutdown steps are logged, to find where a slow exit waits.
                tracing::info!("closing: window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                // The movie view must be re-rendered at the new size.
                self.frame_since_render = true;
                // How minimizing shows up on Windows; winit still has no
                // `WindowEvent::Minimize`. (`Window::is_minimized` does exist by
                // now, but it has to be polled, and it wouldn't tell us anything
                // new: macOS reports minimized windows as occluded instead, and
                // on Wayland it always returns `None`.)
                self.minimized = size.width == 0 && size.height == 0;

                if let Some(mut player) = self.player.get() {
                    let viewport_scale_factor = self.gui.window().scale_factor();
                    // AQW build: (Performance mode: fewer pixels for the same stage, see
                    // `mugetsu::Performance`; the scale factor goes down with them, so the
                    // stage keeps its size in the movie's own units.)
                    player.set_viewport_dimensions(ViewportDimensions {
                        width: crate::mugetsu::drawn_len(size.width),
                        height: crate::mugetsu::drawn_len(
                            size.height.saturating_sub(self.gui.height_offset() as u32),
                        ),
                        scale_factor: viewport_scale_factor * crate::mugetsu::render_scale(),
                    });
                }
                self.gui.window().request_redraw();
                if matches!(self.loaded, LoadingState::WaitingForResize) {
                    self.loaded = LoadingState::Loaded;
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    // Nothing repaints us on its own if the movie is paused.
                    self.gui.window().request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }
                // (Testing: the movie only sees the test pointer.)
                if self.test_pointer.is_some() {
                    return;
                }

                // This is needed because some platforms (like macOS)
                // will fire this event at the same time as WindowEvent::MouseInput,
                // which may cause inaccurate behavior in the movie
                if position == self.mouse_pos {
                    return;
                }

                self.mouse_pos = position;
                self.mouse_move_pending = true;
                // Test switch: hit-test every mouse move, like stock Ruffle.
                if unbatched_mouse_moves() {
                    self.flush_mouse_move();
                }
            }
            WindowEvent::DroppedFile(file) => {
                if let Some(content_descriptor) = ContentDescriptor::new_local(&file, None) {
                    self.gui.create_movie(
                        &mut self.player,
                        LaunchOptions::from(&self.preferences),
                        content_descriptor,
                    );
                }
            }
            WindowEvent::Focused(true) => {
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                {
                    self.modifiers = Default::default();
                }
                self.player.handle_event(PlayerEvent::FocusGained);
            }
            WindowEvent::Focused(false) => {
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                {
                    self.modifiers = Default::default();
                }
                self.player.handle_event(PlayerEvent::FocusLost);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                // Keep MouseMove -> MouseDown ordering and an up-to-date hover target.
                self.flush_mouse_move();

                use ruffle_core::events::MouseButton as RuffleMouseButton;
                use winit::event::MouseButton;
                let (x, y) = self.gui.window_to_movie_position(self.mouse_pos);
                let button = match button {
                    MouseButton::Left => RuffleMouseButton::Left,
                    MouseButton::Right => RuffleMouseButton::Right,
                    MouseButton::Middle => RuffleMouseButton::Middle,
                    _ => RuffleMouseButton::Unknown,
                };
                let event = match state {
                    // TODO We should get information about click index from the OS,
                    //   but winit does not support that yet.
                    ElementState::Pressed => PlayerEvent::MouseDown {
                        x,
                        y,
                        button,
                        index: None,
                    },
                    ElementState::Released => PlayerEvent::MouseUp { x, y, button },
                };
                let handled = self.player.handle_event(event);
                if !handled && state == ElementState::Pressed && button == RuffleMouseButton::Right
                {
                    // Show context menu.
                    if let Some(mut player) = self.player.get() {
                        let context_menu = player.prepare_context_menu();
                        // AQW build: Mugetsu's settings too (AQW turns Flash's menu off).
                        let settings = Some(crate::mugetsu::MenuSettings::now());

                        // MouseUp event will be ignored when the context menu is shown,
                        // but it has to be dispatched when the menu closes.
                        let close_event = PlayerEvent::MouseUp {
                            x,
                            y,
                            button: RuffleMouseButton::Right,
                        };
                        self.gui.show_context_menu(context_menu, settings, close_event);
                    }
                }
                self.check_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                use ruffle_core::events::MouseWheelDelta;
                use winit::event::MouseScrollDelta;
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, dy) => MouseWheelDelta::Lines(dy.into()),
                    MouseScrollDelta::PixelDelta(pos) => MouseWheelDelta::Pixels(pos.y),
                };
                let event = PlayerEvent::MouseWheel { delta };
                self.player.handle_event(event);
                self.check_redraw();
            }
            WindowEvent::CursorEntered { .. } => {
                if let Some(mut player) = self.player.get() {
                    player.set_mouse_in_stage(true);
                    if player.needs_render() {
                        self.gui.window().request_redraw();
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                // (Testing: the test pointer stays where it is.)
                if self.test_pointer.is_some() {
                    return;
                }
                self.mouse_move_pending = false;
                if let Some(mut player) = self.player.get() {
                    player.set_mouse_in_stage(false);
                }
                self.player.handle_event(PlayerEvent::MouseLeave);
                self.check_redraw();
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.gui.is_context_menu_visible() {
                    return;
                }

                // Handle escaping from fullscreen.
                if let KeyEvent {
                    state: ElementState::Pressed,
                    logical_key: Key::Named(NamedKey::Escape),
                    ..
                } = event
                {
                    let _ = self
                        .event_loop_proxy
                        .send_event(RuffleEvent::ExitFullScreen);
                }

                let key = winit_input_to_ruffle_key_descriptor(&event);
                match event.state {
                    ElementState::Pressed => {
                        self.player.handle_event(PlayerEvent::KeyDown { key });
                        if let Some(control_code) =
                            winit_to_ruffle_text_control(&event, self.modifiers)
                        {
                            self.player
                                .handle_event(PlayerEvent::TextControl { code: control_code });
                        } else if let Some(text) = event.text {
                            for codepoint in text.chars() {
                                self.player
                                    .handle_event(PlayerEvent::TextInput { codepoint });
                            }
                        }
                    }
                    ElementState::Released => {
                        self.player.handle_event(PlayerEvent::KeyUp { key });
                    }
                };
                self.check_redraw();
            }
            WindowEvent::Ime(ime) => match ime {
                Ime::Enabled => {}
                Ime::Preedit(text, cursor) => {
                    self.player
                        .handle_event(PlayerEvent::Ime(ImeEvent::Preedit(text, cursor)));
                }
                Ime::Commit(text) => {
                    self.player
                        .handle_event(PlayerEvent::Ime(ImeEvent::Commit(text)));
                }
                Ime::Disabled => {}
            },
            _ => (),
        }
    }

    /// AQW build: counts shown frames for the FPS counter (Mugetsu's settings), updated twice a
    /// second; it's drawn with the next frame, so it causes no extra redraws.
    fn count_fps(&mut self) {
        self.fps_frames += 1;
        let elapsed = self.fps_since.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            let fps = self.fps_frames as f32 / elapsed;
            crate::mugetsu::SHOWN_FPS.store(fps.to_bits(), std::sync::atomic::Ordering::Relaxed);
            self.fps_frames = 0;
            self.fps_since = Instant::now();
        }
    }

    /// AQW build: a change in Mugetsu's settings (the game's right-click menu).
    fn apply_setting(&mut self, setting: crate::mugetsu::Setting) {
        use crate::mugetsu::{self, Setting};
        match setting {
            Setting::Interpolation(on) => {
                ruffle_core::frame_interpolation::set_enabled(on);
                mugetsu::save_interpolation(on);
                tracing::info!("Frame interpolation turned {}", if on { "on" } else { "off" });
                // The render thread comes with interpolation, but only from the start.
                self.gui
                    .offer_interpolation_restart(on && mugetsu::render_thread_needs_restart());
                #[cfg(windows)]
                crate::windows::sync_menu_checks(self.gui.window());
            }
            Setting::Graphics(choice) => {
                if (cfg!(windows) || cfg!(target_os = "linux")) && choice != mugetsu::saved_graphics() {
                    self.gui.ask_graphics_restart(choice);
                }
            }
            Setting::RestartWithGraphics(choice) => {
                #[cfg(any(windows, target_os = "linux"))]
                mugetsu::save_graphics(choice);
                tracing::info!(
                    "Graphics changed to {}; starting Mugetsu again",
                    choice.map_or("Automatic", mugetsu::Graphics::name)
                );
                // Closed like any other time (the player and renderer let go of while the
                // driver is loaded), then started again at the end of `exiting`.
                mugetsu::start_again_after_exit();
                let _ = self.event_loop_proxy.send_event(RuffleEvent::ExitRequested);
            }
            Setting::Restart => {
                tracing::info!("Starting Mugetsu again (asked for in a prompt)");
                mugetsu::start_again_after_exit();
                let _ = self.event_loop_proxy.send_event(RuffleEvent::ExitRequested);
            }
            Setting::ShowFps(on) => {
                mugetsu::set_show_fps(on);
                self.gui.window().request_redraw();
            }
            Setting::MemoryLimit(mb) => {
                mugetsu::set_memory_limit_mb(mb);
                if mb == 0 {
                    tracing::info!("Memory limit: off");
                } else {
                    tracing::info!("Memory limit: {mb} MB");
                }
            }
            Setting::FreeMemory => {
                if let Some(mut player) = self.player.get() {
                    // (A click in the menu: the pause of the thorough way is not seen.)
                    let freed = free_memory(&mut player, true);
                    tracing::info!(target: "ruffle_aqw_perf", "free memory now: {freed}");
                    // (Diagnostic: what the allocator holds right after, in full.)
                    tracing::info!(
                        target: "ruffle_aqw_perf",
                        "allocator statistics after it: {}",
                        crate::memory_split::allocator_json_line()
                    );
                }
                self.gui.window().request_redraw();
            }
            Setting::Performance(level) => {
                mugetsu::set_performance(level);
                tracing::info!(
                    "Performance mode: {level:?} (the game is drawn at {:.0} % of the window's width and height)",
                    level.scale() * 100.0
                );
                // As after a resize: the movie is drawn at its new size from the next picture.
                let size = self.gui.window().inner_size();
                if let Some(mut player) = self.player.get() {
                    let viewport_scale_factor = self.gui.window().scale_factor();
                    player.set_viewport_dimensions(ViewportDimensions {
                        width: mugetsu::drawn_len(size.width),
                        height: mugetsu::drawn_len(
                            size.height.saturating_sub(self.gui.height_offset() as u32),
                        ),
                        scale_factor: viewport_scale_factor * mugetsu::render_scale(),
                    });
                }
                self.frame_since_render = true;
                self.gui.window().request_redraw();
            }
            Setting::UpdateReminders(on) => mugetsu::set_update_reminders(on),
            // (The frames' way to the window changes on the pacer's next turn: see
            // `pace_interpolation`.)
            // (Set as picked, without a question first: the menu's hint says what it is for.)
            Setting::Uncapped(on) => {
                mugetsu::set_uncapped(on);
                tracing::info!("Frame pacing set to {}", if on { "uncapped" } else { "v-sync" });
            }
            Setting::FpsLimit(limit) => {
                mugetsu::set_fps_limit(limit);
                tracing::info!("FPS limit set to {limit:?}");
            }
        }
    }

    /// AQW build: frames sent to the window wait for the monitor's refresh, or do not
    /// ("uncapped" frame pacing, see `mugetsu::uncapped`). The window's surface is configured
    /// again for it, so everything on its way is let through first, as before a resize (see
    /// the `Resized` event). The pacer starts afresh.
    fn set_vsync(&mut self, vsync: bool) {
        if let Some(mut player) = self.player.get() {
            ruffle_render_wgpu::render_thread::settle::<crate::gui::MovieView>(
                player.renderer_mut(),
            );
        }
        self.gui.settle_present();
        let got = self.gui.set_vsync(vsync);
        self.vsync = vsync;
        tracing::info!(
            target: "ruffle_aqw_perf",
            "frame pacing: {}; frames are sent {got}",
            if vsync { "v-sync" } else { "uncapped" }
        );
        self.draw_divisor = 1;
        self.repeats_since_picture = 0;
        self.divisor_up_since = None;
        self.divisor_down_since = None;
        self.pipeline_limit = None;
        self.limit_trial = None;
        self.limit_quiet = Duration::ZERO;
        self.limit_quiet_until = None;
        self.saturated_since = None;
        self.picture_gap_ms = 0.0;
        self.render_wait_ms = 0.0;
    }

    fn on_metadata(&mut self, swf_header: HeaderExt) {
        let height_offset = if self.gui.window().fullscreen().is_some() || self.no_gui {
            0.0
        } else {
            MENU_HEIGHT as f64
        };

        // To prevent issues like waiting on resize indefinitely (#11364) or desyncing the window state on Windows,
        // do not resize while window is maximized.
        let should_resize = !self.gui.window().is_maximized();

        let (viewport_size, state) = if should_resize {
            let movie_width = swf_header.stage_size().width().to_pixels();
            let movie_height = swf_header.stage_size().height().to_pixels();

            let window_size: Size = match (self.preferred_width, self.preferred_height) {
                (None, None) => LogicalSize::new(movie_width, movie_height + height_offset).into(),
                (Some(width), None) => {
                    let scale = width / movie_width;
                    let height = movie_height * scale;
                    PhysicalSize::new(
                        width.max(1.0),
                        height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                    )
                    .into()
                }
                (None, Some(height)) => {
                    let scale = height / movie_height;
                    let width = movie_width * scale;
                    PhysicalSize::new(
                        width.max(1.0),
                        height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                    )
                    .into()
                }
                (Some(width), Some(height)) => PhysicalSize::new(
                    width.max(1.0),
                    height.max(1.0) + height_offset * self.gui.window().scale_factor(),
                )
                .into(),
            };

            let window_size = Size::clamp(
                window_size,
                self.min_window_size.into(),
                self.max_window_size.into(),
                self.gui.window().scale_factor(),
            );

            let viewport_size = self.gui.window().inner_size();
            let mut window_resize_denied = false;

            if let Some(new_viewport_size) = self.gui.window().request_inner_size(window_size) {
                if new_viewport_size != viewport_size {
                    self.gui.resize(new_viewport_size);
                } else {
                    tracing::warn!("Unable to resize window");
                    window_resize_denied = true;
                }
            }

            let viewport_size = self.gui.window().inner_size();

            // On X11 (and possibly other platforms), the window size is not updated immediately.
            // On a successful resize request, wait for the window to be resized to the requested size
            // before we start running the SWF (which can observe the viewport size in "noScale" mode)
            let state = if !window_resize_denied && window_size != viewport_size.into() {
                LoadingState::WaitingForResize
            } else {
                LoadingState::Loaded
            };

            (viewport_size, state)
        } else {
            (self.gui.window().inner_size(), LoadingState::Loaded)
        };

        self.loaded = state;

        self.gui.window().set_fullscreen(if self.start_fullscreen {
            Some(Fullscreen::Borderless(None))
        } else {
            None
        });
        self.gui.window().set_visible(true);

        let viewport_scale_factor = self.gui.window().scale_factor();
        if let Some(mut player) = self.player.get() {
            player.set_viewport_dimensions(ViewportDimensions {
                width: crate::mugetsu::drawn_len(viewport_size.width),
                height: crate::mugetsu::drawn_len(
                    viewport_size.height - (height_offset * viewport_scale_factor) as u32,
                ),
                scale_factor: viewport_scale_factor * crate::mugetsu::render_scale(),
            });
        }
    }

    fn about_to_wait(&mut self, gilrs: Option<&mut Gilrs>) {
        if let Some(Event { event, .. }) = gilrs.and_then(|gilrs| gilrs.next_event()) {
            match event {
                EventType::ButtonPressed(button, _) => {
                    if let Some(button) = gilrs_button_to_gamepad_button(button) {
                        self.player
                            .handle_event(PlayerEvent::GamepadButtonDown { button });
                        self.check_redraw();
                    }
                }
                EventType::ButtonReleased(button, _) => {
                    if let Some(button) = gilrs_button_to_gamepad_button(button) {
                        self.player
                            .handle_event(PlayerEvent::GamepadButtonUp { button });
                        self.check_redraw();
                    }
                }
                _ => {}
            }
        }

        // Core loop
        // [NA] This used to be called `MainEventsCleared`, but I think the behaviour is different now.
        // We should look at changing our tick to happen somewhere else if we see any behavioural problems.
        if matches!(self.loaded, LoadingState::Loaded) {
            let new_time = Instant::now();
            // AQW build (testing): where the test pointer is now.
            if let Some(pointer) = &mut self.test_pointer
                && let Some((x, y)) = pointer.position(new_time)
            {
                let size = self.gui.window().inner_size();
                let top = self.gui.movie_to_window_position(0.0, 0.0).y;
                self.mouse_pos = PhysicalPosition::new(
                    x * f64::from(size.width),
                    top + y * (f64::from(size.height) - top),
                );
                self.mouse_move_pending = true;
                if let Some(mut player) = self.player.get() {
                    player.set_mouse_in_stage(true);
                }
            }
            // AQW build (testing): things done to the window at set times.
            for action in self.test_hooks.window_actions(new_time) {
                use crate::test_hooks::WindowAction;
                match action {
                    WindowAction::Menu => {
                        let settings = Some(crate::mugetsu::MenuSettings::now());
                        let close_event = PlayerEvent::MouseUp {
                            x: 0.0,
                            y: 0.0,
                            button: ruffle_core::events::MouseButton::Right,
                        };
                        self.gui.show_context_menu(Vec::new(), settings, close_event);
                        self.gui.window().request_redraw();
                    }
                    WindowAction::OfferRestart => self.gui.offer_interpolation_restart(true),
                    WindowAction::Interpolation(on) => {
                        self.apply_setting(crate::mugetsu::Setting::Interpolation(on));
                    }
                    _ => {
                        let mut player = self.player.get();
                        action.apply(self.gui.window(), player.as_deref_mut());
                    }
                }
            }
            // This runs after every batch of OS events (so ~1000 times a second while a
            // gaming mouse moves); only hit-test when a movie frame is actually due.
            if self.next_frame_time.is_none_or(|t| new_time >= t) {
                self.flush_mouse_move();
            }
            let dt = FloatDuration::from_std(new_time.duration_since(self.time));
            if dt.as_millis() > 0.0 {
                self.time = new_time;
                let perf = &mut self.perf;
                let frame_since_render = &mut self.frame_since_render;
                let frame_cost_ms = &mut self.frame_cost_ms;
                let test_hooks = &mut self.test_hooks;
                let events = &self.event_loop_proxy;
                self.next_frame_time = self.player.get().map(|mut player| {
                    let frames_before = player.frames_run();
                    let timers_before = ruffle_render_wgpu::perf_counters::timers::snapshot();
                    // AQW build (testing).
                    test_hooks.tick(new_time, &mut player);
                    player.tick(dt);
                    if player.frames_run() != frames_before {
                        *frame_since_render = true;
                        let frame_ms = new_time.elapsed().as_secs_f64() * 1000.0;
                        test_hooks.frame(frame_ms);
                        // One-off slow frames, e.g. a room loading, count at most 3x the
                        // average (as draws do): one 300 ms frame made interpolation think every
                        // frame cost 30 ms and drop to once per frame for seconds.
                        // A shape uploaded by the scripts waits for the present thread too
                        // (see the redraw handler): not the frame's cost either.
                        let waited_ms = main_thread_wait_ms(&timers_before);
                        let cost = (frame_ms - waited_ms).max(0.0).min(*frame_cost_ms * 3.0);
                        *frame_cost_ms = *frame_cost_ms * 0.9 + cost * 0.1;
                        // AQW build: say what a slow frame (scripts etc., longer than 1.2
                        // frames) spent its time on, e.g. when a crowded room loads.
                        if frame_ms > 1.2 * 1000.0 / player.frame_rate() {
                            tracing::info!(
                                target: "ruffle_aqw_perf",
                                "slow frame {frame_ms:.0} ms ({} frames run); of that: {} (ms)",
                                player.frames_run() - frames_before,
                                timer_parts(&timers_before)
                            );
                        }
                        perf.frame(
                            new_time,
                            frame_ms,
                            1000.0 / player.frame_rate(),
                            player.frames_run() - frames_before,
                        );
                    }
                    if perf.maybe_log(new_time) {
                        // AQW build: with the render thread, the calls that had to wait for
                        // a picture being recorded since the last report.
                        if let Some(waits) = ruffle_render_wgpu::render_thread::waits_report::<
                            crate::gui::MovieView,
                        >(player.renderer_mut())
                        {
                            tracing::info!(
                                target: "ruffle_aqw_perf",
                                "render thread: calls that waited for a picture: {waits}"
                            );
                        }
                        memory_watch(&mut player);
                        perf.log_object_counts(&mut player);
                        crate::mugetsu::low_quality_tick(&mut player, perf.last_drawn_fps, events);
                        crate::mugetsu::performance_offer_tick(
                            &mut player,
                            perf.last_drawn_fps,
                            events,
                        );
                        crate::mugetsu::test_restart_tick(events);
                        crate::mugetsu::test_mouse_tick(&mut player);
                    }
                    new_time + player.time_til_next_frame()
                });
                self.check_redraw();
            }
            self.pace_interpolation();
        }
    }

    /// AQW build: with frame interpolation on (title-bar menu), sends a picture on every refresh
    /// of the monitor: vsync makes each one wait for its refresh, which paces this exactly (a
    /// timer is ~1-2 ms late, a lot at 360 Hz). A new in-between picture is drawn on every n-th
    /// of them, the most the main thread can afford, so each stays up the same time; the others
    /// show the last one again, which costs little. Needs at least 1.5 new pictures per SWF
    /// frame, else it draws once per frame as without interpolation.
    fn pace_interpolation(&mut self) {
        self.next_draw_time = None;
        self.uncapped = false;
        if self.minimized || self.occluded {
            // (A step on trial cannot be judged while nothing is shown: it stands.)
            self.limit_trial = None;
            return;
        }
        // AQW build: "uncapped" frame pacing (see `mugetsu::uncapped`): frames then do not
        // wait for the monitor's refresh. Looked at here, so that it follows the setting
        // whoever changed it (and frame interpolation being turned off or on).
        let uncapped = crate::mugetsu::uncapped_now();
        if uncapped == self.vsync {
            self.set_vsync(!uncapped);
        }
        if !ruffle_core::frame_interpolation::enabled() {
            self.limit_trial = None;
            return;
        }
        let now = Instant::now();
        if self
            .refresh_checked
            .is_none_or(|checked| now.duration_since(checked) > Duration::from_secs(2))
        {
            let first = self.refresh_checked.is_none();
            self.refresh_checked = Some(now);
            let millihertz = self
                .gui
                .window()
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz())
                .unwrap_or(60_000)
                .clamp(24_000, 360_000);
            let interval = Duration::from_secs_f64(1000.0 / millihertz as f64);
            if interval != self.refresh_interval || first {
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "frame interpolation: monitor refreshes at {:.1} Hz",
                    millihertz as f64 / 1000.0
                );
            }
            self.refresh_interval = interval;
        }
        let frame_rate = match self.player.get() {
            Some(player) if player.is_playing() => player.frame_rate(),
            _ => return,
        };
        // Draw on every refresh if the main thread can afford it (keeping a third of its time
        // spare), else on every 2nd, 3rd, ... refresh: an even rate looks smoother than an
        // uneven higher one. Each change of rate is a visible change of rhythm, so it only steps
        // to fewer pictures after needing to for half a second, and back to more after 3 seconds
        // of clearly affording it; 1 second for a big step back (at least twice the pictures,
        // or out of drawing once per frame), as after a room has loaded.
        let refresh_hz = 1.0 / self.refresh_interval.as_secs_f64();
        if uncapped {
            // No refreshes to count: each new picture goes out once, as soon as it is ready,
            // as many as get through, or as many as the FPS limit allows, evenly by the
            // clock. The monitor shows each for as long as it took (variable refresh) or
            // tears (fixed). A game frame that is due still runs first: the loop looks
            // after every picture. No budget and no steps: that is what "uncapped" asks for.
            use crate::mugetsu::FpsLimit;
            let limit = match crate::mugetsu::fps_limit() {
                FpsLimit::Off => None,
                // Three under the monitor's rate: at the rate itself a variable-refresh
                // monitor stops following.
                FpsLimit::Screen => Some((refresh_hz - 3.0).max(refresh_hz * 0.9)),
                FpsLimit::Fixed(fps) => Some(fps as f64),
            };
            self.perf.interpolation_uncapped = true;
            self.perf.interpolation_rate = limit.unwrap_or(f64::INFINITY);
            self.perf.interpolation_costs =
                [self.frame_cost_ms, self.draw_cost_ms, self.between_cost_ms];
            if limit.is_some_and(|fps| fps < frame_rate * 1.5) {
                // Not even 1.5 pictures per game frame: one per frame, as without
                // interpolation.
                return;
            }
            self.uncapped = true;
            self.uncapped_interval = limit.map(|fps| Duration::from_secs_f64(1.0 / fps));
            let due = match self.uncapped_interval {
                None => now,
                Some(interval) => self.picture_due + interval,
            };
            if now + Duration::from_micros(300) >= due {
                self.gui.window().request_redraw();
            }
            self.next_draw_time = Some(due.max(now));
            return;
        }
        self.perf.interpolation_uncapped = false;
        let spare_ms = (interpolation_budget_ms() - self.frame_cost_ms * frame_rate).max(100.0);
        // AQW build: one picture per game frame costs `draw_cost`; the in-between pictures
        // (`between_cost` each) get what's left of the spare time. With both costs equal this
        // is the old spare / cost pictures per second.
        // With the render thread, recording the pictures is that thread's work: it gets a
        // budget of its own (850 ms a second), by what a picture costs it.
        let thread_cost_ms = self.thread_cost_ms;
        // And what a picture takes to get through the stages after that thread, while a
        // limit found there is believed (see `PipelineLimit`): it only keeps the pacer from
        // going back to more pictures. Going to fewer because of it takes seeing the
        // pictures fall behind (below), not this estimate alone.
        let pipeline = self
            .pipeline_limit
            .as_ref()
            .filter(|limit| now.duration_since(limit.learned) < limit.hold)
            .map(|limit| limit.per_thread_ms);
        let divisor_for = |draw_cost_ms: f64,
                           between_cost_ms: f64,
                           thread_cost_ms: f64,
                           pipeline: Option<f64>| {
            let left_ms = (spare_ms - frame_rate * draw_cost_ms).max(0.0);
            let mut per_second = frame_rate + left_ms / between_cost_ms.max(0.5);
            if thread_cost_ms > 0.0 {
                per_second = per_second.min(RENDER_THREAD_BUDGET_MS / thread_cost_ms.max(0.5));
                if let Some(per_thread_ms) = pipeline {
                    per_second = per_second
                        .min(PIPELINE_BUDGET_MS / (thread_cost_ms * per_thread_ms).max(0.5));
                }
            }
            (refresh_hz / per_second).ceil().clamp(1.0, 16.0) as u32
        };
        let needed = divisor_for(self.draw_cost_ms, self.between_cost_ms, thread_cost_ms, None);
        let fitting = divisor_for(
            self.draw_cost_ms,
            self.between_cost_ms,
            thread_cost_ms,
            pipeline,
        );
        let lasted = |since: &mut Option<Instant>, wanted: bool, secs: f64| {
            if !wanted {
                *since = None;
                return false;
            }
            now.duration_since(*since.get_or_insert(now)).as_secs_f64() >= secs
        };
        let affordable = divisor_for(
            self.draw_cost_ms * 1.3,
            self.between_cost_ms * 1.3,
            thread_cost_ms * 1.3,
            pipeline,
        );
        let once_per_frame =
            (self.refresh_interval * self.draw_divisor).as_secs_f64() * 1.5 > 1.0 / frame_rate;
        let big_step = once_per_frame || affordable * 2 <= self.draw_divisor;
        // With the render thread: the pictures follow each other slower than aimed for, and
        // this thread spends much of the time between them waiting for the render thread
        // (see `PipelineLimit`). Much of it: 50-70 % where the later stages were the limit
        // (a 30-player room at 340 a second, Yulgar at 190); while a movie or a room loads,
        // the pictures are late because THIS thread is busy, and the wait was 15 % of it.
        let aim_ms = self.refresh_interval.as_secs_f64() * 1000.0 * self.draw_divisor as f64;
        // A step made for this is on trial (`LIMIT_TRIAL`). It worked if the pictures kept
        // the new pace (else: below) AND the monitor got a frame on most refreshes: the point
        // of fewer pictures is that each stays up the same number of refreshes, and a GPU
        // that cannot show them again in between gives the loss without the gain (a built-in
        // Radeon: 156 window frames a second of 360, with 51 new pictures where 64 went
        // through before the step).
        let mut undo: Option<(u32, String)> = None;
        if let Some((at, before, presents)) = self.limit_trial
            && now.duration_since(at) >= LIMIT_TRIAL
        {
            self.limit_trial = None;
            let shown =
                (self.presents_total - presents) as f64 / now.duration_since(at).as_secs_f64();
            if shown < refresh_hz * 0.7 {
                undo = Some((
                    before,
                    format!("the monitor got {shown:.0} frames a second of {refresh_hz:.0}"),
                ));
            } else {
                self.limit_quiet = Duration::ZERO;
            }
        }
        let saturated = thread_cost_ms > 0.0
            && !once_per_frame
            && self.limit_quiet_until.is_none_or(|until| now >= until)
            && self.picture_gap_ms > aim_ms * LIMIT_SHORTFALL
            && self.render_wait_ms > self.picture_gap_ms * 0.4;
        let fell_behind = lasted(&mut self.saturated_since, saturated, 0.35);
        if undo.is_none()
            && fell_behind
            && let Some((_, before, _)) = self.limit_trial.take()
        {
            undo = Some((
                before,
                format!(
                    "the pictures fell behind again ({:.1} ms apart where {aim_ms:.1} was aimed for)",
                    self.picture_gap_ms
                ),
            ));
        }
        if let Some((before, why)) = undo {
            // Fewer new pictures did not help. Showing a picture again is not free, and on a
            // GPU that is far too slow the window frames that show the last picture again
            // took the time the new ones needed: every step to fewer made them later still,
            // down to one picture per game frame (30 where 64 went through). So the step is
            // undone, the pictures go out as fast as they get through, as before, and this
            // is left alone for a while (15 s, twice as long each time it fails again, up to
            // 10 minutes).
            self.limit_quiet = (self.limit_quiet * 2)
                .clamp(Duration::from_secs(15), Duration::from_secs(600));
            self.limit_quiet_until = Some(now + self.limit_quiet);
            tracing::info!(
                target: "ruffle_aqw_perf",
                "frame interpolation: a picture every {} refreshes did not help: {why}: back to every {before}, as fast as they get through; left alone for {} s",
                self.draw_divisor,
                self.limit_quiet.as_secs()
            );
            self.draw_divisor = before;
            self.pipeline_limit = None;
            self.picture_gap_ms =
                self.refresh_interval.as_secs_f64() * 1000.0 * self.draw_divisor as f64;
            self.render_wait_ms = 0.0;
            self.saturated_since = None;
            self.divisor_up_since = None;
            self.divisor_down_since = None;
        } else if fell_behind {
            let gap_ms = self.picture_gap_ms;
            let before = self.draw_divisor;
            let fits = (refresh_hz * gap_ms / PIPELINE_BUDGET_MS).ceil().clamp(1.0, 16.0) as u32;
            let target = fits.max(before + 1).max(needed).min(16);
            // What that pace gives (one picture per game frame once it would be fewer than
            // 1.5), against what gets through now (see `LIMIT_LOSS`).
            let refresh_secs = self.refresh_interval.as_secs_f64();
            let pace = if refresh_secs * target as f64 * 1.5 > 1.0 / frame_rate {
                frame_rate
            } else {
                refresh_hz / target as f64
            };
            let through = 1000.0 / gap_ms;
            if 1.0 - pace / through > LIMIT_LOSS {
                // Left as it is; looked at again in a few seconds (the room may fill up).
                self.limit_quiet_until = Some(now + Duration::from_secs(5));
                self.saturated_since = None;
                if self
                    .limit_note_at
                    .is_none_or(|at| now.duration_since(at) > Duration::from_secs(60))
                {
                    self.limit_note_at = Some(now);
                    tracing::info!(
                        target: "ruffle_aqw_perf",
                        "frame interpolation: {through:.0} pictures a second get through where {:.0} are aimed for; the even pace that fits ({pace:.0}) would lose {:.0} % of them: left as it is",
                        1000.0 / aim_ms,
                        (1.0 - pace / through) * 100.0
                    );
                }
            } else {
                let per_thread_ms = (gap_ms / thread_cost_ms.max(0.2)).clamp(1.0, 50.0);
                let hold = match &self.pipeline_limit {
                    Some(old)
                        if now.duration_since(old.learned) < old.hold * 3
                            && (per_thread_ms / old.per_thread_ms - 1.0).abs() < 0.3 =>
                    {
                        (old.hold * 2).min(Duration::from_secs(300))
                    }
                    _ => Duration::from_secs(10),
                };
                self.draw_divisor = target;
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "frame interpolation: the pictures followed each other every {gap_ms:.1} ms where {aim_ms:.1} was aimed for (this thread waited {:.1} ms for the render thread before each): a picture now every {} refreshes, not every {before}; getting one through takes {per_thread_ms:.1} times the render thread's {thread_cost_ms:.1} ms, believed for {} s",
                    self.render_wait_ms,
                    self.draw_divisor,
                    hold.as_secs()
                );
                self.pipeline_limit = Some(PipelineLimit {
                    per_thread_ms,
                    learned: now,
                    hold,
                });
                // On trial: see above.
                self.limit_trial = Some((now, before, self.presents_total));
                // Measured anew at the new pace.
                self.picture_gap_ms =
                    self.refresh_interval.as_secs_f64() * 1000.0 * self.draw_divisor as f64;
                self.render_wait_ms = 0.0;
                self.saturated_since = None;
                self.divisor_up_since = None;
                self.divisor_down_since = None;
            }
        } else if lasted(&mut self.divisor_up_since, needed > self.draw_divisor, 0.5) {
            self.draw_divisor = needed;
            self.divisor_up_since = None;
            self.divisor_down_since = None;
        } else if lasted(
            &mut self.divisor_down_since,
            affordable < self.draw_divisor,
            if big_step { 1.0 } else { 3.0 },
        ) {
            // (`fitting` is `needed` unless a limit found behind the render thread is
            // believed; it is never above `affordable`.)
            self.draw_divisor = fitting;
            self.divisor_up_since = None;
            self.divisor_down_since = None;
        }
        let interval = self.refresh_interval * self.draw_divisor;
        self.perf.interpolation_rate = refresh_hz / self.draw_divisor as f64;
        self.perf.interpolation_costs =
            [self.frame_cost_ms, self.draw_cost_ms, self.between_cost_ms];
        if interval.as_secs_f64() * 1.5 > 1.0 / frame_rate {
            // Not even 1.5 draws per SWF frame: draw once per frame, as without interpolation.
            return;
        }
        // Requested right away, keeping the screen's queue of pictures full so a busy moment on
        // the main thread doesn't miss a refresh; presenting waits for vsync. The timer only
        // limits this if presenting doesn't wait.
        let early = self.refresh_interval / 2;
        let mut due = self.draw_started + early;
        if now >= due {
            self.gui.window().request_redraw();
            due = now + early;
        }
        self.next_draw_time = Some(due);
    }

    fn flush_mouse_move(&mut self) {
        if !std::mem::take(&mut self.mouse_move_pending) {
            return;
        }
        let (x, y) = self.gui.window_to_movie_position(self.mouse_pos);
        self.player.handle_event(PlayerEvent::MouseMove { x, y });
        self.check_redraw();
    }

    fn check_redraw(&self) {
        // AQW build: nothing is drawn while the window is minimized (see the redraw handler),
        // so nothing is asked for either. A redraw that was asked for came back at once,
        // was skipped, left the frame "not drawn yet", and was asked for again on the next
        // turn of the loop: a minimized player kept one processor core busy (890-950 ms a
        // second of the main thread, with or without frame interpolation). When the window
        // is back, the next game frame asks.
        if self.minimized || self.occluded {
            return;
        }
        let player = self.player.get();
        // AQW build: like Flash Player's normal pacing, only draw once a new SWF frame has run.
        // AQW calls `updateAfterEvent()` from timers and hovers change between frames, which
        // made Ruffle redraw the same frame 2-9x per frame interval; in combat those extra
        // draws cost the main thread the time the next frame needed. A paused movie still
        // redraws on demand.
        let player_needs_render = player
            .map(|p| {
                p.needs_render()
                    && (self.frame_since_render || !p.is_playing() || extra_renders_enabled())
            })
            .unwrap_or_default();
        // AQW build: egui asks for a repaint on every mouse move. Each repaint presents, and
        // presenting faster than the monitor refreshes blocks the main thread, so moving the
        // mouse made frames late. With the menu bar hidden there is usually nothing to update,
        // so GUI-only repaints are capped at ~60/s (popups and dialogs still work).
        let gui_needs_render = self.gui.needs_render()
            && self.gui.wants_redraws()
            && (!self.no_gui || self.last_redraw.elapsed() >= Duration::from_millis(16));
        if player_needs_render || gui_needs_render {
            self.gui.window().request_redraw();
        }
    }
}

pub struct App {
    main_window: Option<MainWindow>,
    runtime: Option<tokio::runtime::Runtime>,
    gilrs: Option<Gilrs>,
    event_loop_proxy: EventLoopProxy<RuffleEvent>,
    preferences: GlobalPreferences,
    font_database: fontdb::Database,
}

/// Enters the tokio runtime context.
/// This cannot be a method, as the borrow-checker would complain.
macro_rules! enter_runtime {
    ($this:expr) => {
        let _guard = $this.runtime.as_ref().map(|runtime| runtime.enter());
    };
}

impl App {
    pub fn new(preferences: GlobalPreferences) -> Result<(Self, EventLoop<RuffleEvent>), Error> {
        let event_loop = EventLoop::with_user_event().build()?;
        // AQW build: Ruffle never reads raw `DeviceEvent`s, but winit registers for raw input
        // while focused, so a 1000 Hz mouse woke the loop (and ran a player tick) ~1000x/s.
        event_loop.listen_device_events(winit::event_loop::DeviceEvents::Never);

        let mut font_database = fontdb::Database::default();
        font_database.load_system_fonts();

        let gilrs = Gilrs::new()
            .inspect_err(|err| {
                tracing::warn!("Gamepad support could not be initialized: {err}");
            })
            .ok();
        let event_loop_proxy = event_loop.create_proxy();
        let runtime = tokio::runtime::Runtime::new()?;

        Ok((
            Self {
                main_window: None,
                runtime: Some(runtime),
                gilrs,
                event_loop_proxy,
                font_database,
                preferences,
            },
            event_loop,
        ))
    }
}

impl ApplicationHandler<RuffleEvent> for App {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        enter_runtime!(self);

        if cause == StartCause::Init {
            let movie_url = self.preferences.cli.movie_url.clone();
            // AQW build: Mugetsu's icon (64 px so the taskbar copy stays sharp).
            let icon_bytes = include_bytes!("../assets/mugetsu-icon-64.rgba");
            let icon =
                Icon::from_rgba(icon_bytes.to_vec(), 64, 64).expect("App icon should be correct");

            let no_gui = self.preferences.cli.no_gui;
            let min_window_size = if no_gui {
                (16, 16)
            } else {
                (350, MENU_HEIGHT + 16)
            }
            .into();
            let preferred_width = self.preferences.cli.width;
            let preferred_height = self.preferences.cli.height;
            let start_fullscreen = self.preferences.cli.fullscreen;

            #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
            let mut window_attributes = WindowAttributes::default()
                .with_visible(false)
                .with_title(crate::window_title()) // AQW build
                .with_window_icon(Some(icon))
                .with_min_inner_size(min_window_size);

            #[cfg(target_os = "linux")]
            {
                use winit::platform::startup_notify::{
                    self, EventLoopExtStartupNotify, WindowAttributesExtStartupNotify,
                };
                use winit::platform::wayland::WindowAttributesExtWayland;
                window_attributes = window_attributes.with_name("rs.ruffle.Ruffle", "main");
                if let Some(token) = event_loop.read_token_from_env() {
                    startup_notify::reset_activation_token_env();
                    window_attributes = window_attributes.with_activation_token(token);
                }
            }

            let event_loop_proxy = self.event_loop_proxy.clone();
            let preferences = self.preferences.clone();
            let window = event_loop
                .create_window(window_attributes)
                .expect("Window should be created");
            let max_window_size = get_screen_size(&window);
            window.set_max_inner_size(Some(max_window_size));
            let window = Arc::new(window);
            let font_database = self.font_database.clone();

            let mut gui = GuiController::new(
                window.clone(),
                event_loop_proxy.clone(),
                preferences.clone(),
                &font_database,
                movie_url.clone(),
                no_gui,
            )
            .expect("GUI controller should be created");

            let mut player = PlayerController::new(
                event_loop_proxy.clone(),
                window.clone(),
                gui.descriptors().clone(),
                font_database,
                preferences.clone(),
                gui.file_picker(),
            );

            if let Some(movie_url) = &movie_url {
                gui.create_movie(
                    &mut player,
                    LaunchOptions::from(&preferences),
                    ContentDescriptor {
                        url: movie_url.clone(),
                        root_content_path: None,
                    },
                );
            } else {
                gui.show_open_dialog();
            }

            let mut loaded = LoadingState::Loading;

            if movie_url.is_none() {
                // No SWF provided on command line; show window with dummy movie immediately.
                window.set_visible(true);
                loaded = LoadingState::Loaded;
            }

            self.main_window = Some(MainWindow {
                preferences,
                gui,
                player,
                min_window_size,
                max_window_size,
                no_gui,
                preferred_width,
                preferred_height,
                start_fullscreen,
                loaded,
                minimized: false,
                occluded: false,
                mouse_pos: PhysicalPosition::new(0.0, 0.0),
                mouse_move_pending: false,
                test_pointer: TestPointer::from_env(),
                test_hooks: crate::test_hooks::TestHooks::from_env(),
                fps_frames: 0,
                fps_since: Instant::now(),
                modifiers: Modifiers::default(),
                time: Instant::now(),
                next_frame_time: None,
                event_loop_proxy,
                perf: PerfStats::default(),
                frame_since_render: true,
                last_redraw: Instant::now(),
                next_draw_time: None,
                draw_started: Instant::now(),
                draw_cost_ms: 2.0,
                between_cost_ms: 2.0,
                thread_cost_ms: 0.0,
                picture_gap_ms: 0.0,
                render_wait_ms: 0.0,
                picture_at: None,
                pipeline_limit: None,
                saturated_since: None,
                limit_trial: None,
                limit_quiet: Duration::ZERO,
                limit_quiet_until: None,
                limit_note_at: None,
                presents_total: 0,
                frame_cost_ms: 3.0,
                draw_divisor: 1,
                repeats_since_picture: 0,
                picture_due: Instant::now(),
                vsync: true,
                uncapped: false,
                uncapped_interval: None,
                divisor_up_since: None,
                divisor_down_since: None,
                refresh_interval: Duration::from_secs_f64(1.0 / 60.0),
                refresh_checked: None,
            });
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: RuffleEvent) {
        enter_runtime!(self);

        match (&mut self.main_window, event) {
            (Some(main_window), RuffleEvent::TaskPoll(task)) => main_window.player.poll(task),

            (Some(main_window), RuffleEvent::OnMetadata(swf_header)) => {
                main_window.on_metadata(swf_header)
            }

            (Some(main_window), RuffleEvent::ContextMenuItemClicked(index)) => {
                if let Some(mut player) = main_window.player.get() {
                    player.run_context_menu_callback(index);
                }
            }

            (Some(main_window), RuffleEvent::MugetsuSetting(setting)) => {
                main_window.apply_setting(setting);
            }

            (Some(main_window), RuffleEvent::MugetsuNotice(notice)) => {
                main_window.gui.show_notice(notice);
            }

            (Some(main_window), RuffleEvent::BrowseAndOpen(options, open_type)) => {
                let event_loop = main_window.event_loop_proxy.clone();
                let picker = main_window.gui.file_picker();
                tokio::spawn(async move {
                    let picked = match open_type {
                        OpenType::File => {
                            picker.pick_ruffle_file(None).await.map(|file| (file, None))
                        }
                        OpenType::Directory => picker
                            .pick_ruffle_directory_and_content(None)
                            .await
                            .map(|(dir, file)| (file, Some(dir))),
                    };

                    if let Some(desc) =
                        picked.and_then(|(file, dir)| ContentDescriptor::new_local(&file, dir))
                    {
                        let _ = event_loop.send_event(RuffleEvent::Open(desc, options));
                    }
                });
            }

            (Some(main_window), RuffleEvent::Open(desc, options)) => {
                main_window
                    .gui
                    .create_movie(&mut main_window.player, *options, desc);
            }

            (Some(main_window), RuffleEvent::OpenDialog(descriptor)) => {
                main_window.gui.open_dialog(descriptor);
            }

            (Some(main_window), RuffleEvent::CloseFile) => {
                main_window.gui.window().set_title(&crate::window_title()); // Reset title since file has been closed.
                main_window.gui.close_movie(&mut main_window.player);
            }

            (Some(main_window), RuffleEvent::ExportBundle) => {
                main_window.gui.export_bundle();
            }

            (Some(main_window), RuffleEvent::EnterFullScreen) => {
                if let Some(mut player) = main_window.player.get()
                    && player.is_playing()
                {
                    player.set_fullscreen(true);
                }
            }

            (Some(main_window), RuffleEvent::ExitFullScreen) => {
                if let Some(mut player) = main_window.player.get()
                    && player.is_playing()
                {
                    player.set_fullscreen(false);
                }
            }

            (Some(main_window), RuffleEvent::PlayerNotification(notification)) => {
                match notification {
                    PlayerNotification::ImeNotification(ImeNotification::ImeReady {
                        purpose,
                        cursor_area,
                    }) => {
                        let ime_enabled = main_window.preferences.ime_enabled().unwrap_or(false);
                        main_window.gui.set_ime_allowed(ime_enabled);
                        main_window.gui.set_ime_purpose(purpose);
                        main_window.gui.set_ime_cursor_area(cursor_area);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImePurposeUpdated(
                        purpose,
                    )) => {
                        main_window.gui.set_ime_purpose(purpose);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImeCursorAreaUpdated(
                        cursor_area,
                    )) => {
                        main_window.gui.set_ime_cursor_area(cursor_area);
                    }
                    PlayerNotification::ImeNotification(ImeNotification::ImeNotReady) => {
                        main_window.gui.set_ime_allowed(false);
                    }
                }
            }

            (_, RuffleEvent::ExitRequested) => {
                event_loop.exit();
            }

            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        enter_runtime!(self);

        if let Some(main_window) = &mut self.main_window {
            main_window.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        enter_runtime!(self);

        if let Some(main_window) = &mut self.main_window {
            main_window.about_to_wait(self.gilrs.as_mut());

            // The event loop is finished; let's find out how long we need to wait for.
            // We don't need to worry about earlier update requests, as it's the
            // only place where we're setting control flow, and events cancel wait.
            // Note: the control flow might be set to `ControlFlow::WaitUntil` with a
            // timestamp in the past! Take that into consideration when changing this code.
            let wake = match (main_window.next_frame_time, main_window.next_draw_time) {
                (Some(frame), Some(draw)) => Some(frame.min(draw)),
                (frame, draw) => frame.or(draw),
            };
            if let Some(wake) = wake {
                event_loop.set_control_flow(ControlFlow::WaitUntil(wake));
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let started = Instant::now();
        crate::mugetsu::set_exit_phase(1);
        tracing::info!("closing: dropping the player and renderer");
        // AQW build: what this thread's storage holds of the graphics device is let go of
        // now, while the driver is alive: thread storage is freed last of all at exit, when
        // Windows has already unloaded the driver (a laptop on DirectX 12 crashed there,
        // 0x87A in KERNELBASE, after "closing: done"). Shapes still being tessellated keep
        // their caches, and with them meshes and the device; recorded segments keep the
        // device; the video memory query keeps a DXGI adapter.
        let shapes = ruffle_core::shutdown_deferred_shapes();
        let segments = ruffle_render_wgpu::segments::clear();
        crate::mugetsu::release_video_memory_adapter();
        // `MainWindow` needs a tokio context to properly drop.
        {
            enter_runtime!(self);
            let _ = self.main_window.take();
        }
        // AQW build: how the close divides (a slow one was seen under a heavy load: where).
        let dropped = started.elapsed();
        // AQW build: the last finished command buffers are dropped on wgpu's background thread
        // and can hold the last reference to the graphics device. Waited for here (2 s at
        // most), so the device is gone before the rest of the shutdown, not at some point
        // during it: on DirectX 12 that crashed two closes out of three (see wgpu-hal's
        // `DynLib`), and the process could end with the driver in the middle of it.
        wgpu_core::aqw_flush_encoder_drops();
        crate::mugetsu::set_exit_phase(2);
        tracing::info!(
            "closing: player and renderer dropped after {} ms ({shapes} shapes still being drawn and {segments} recorded segments were let go of; the window and the player took {} ms, the last command buffers {} ms)",
            started.elapsed().as_millis(),
            dropped.as_millis(),
            (started.elapsed() - dropped).as_millis()
        );

        // Manually stop the tokio runtime: this makes sure that any pending Player-bound futures
        // are properly cancelled and put back on the winit event loop before it closes, preventing
        // them from being dropped on the wrong thread and causing a panic.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(std::time::Duration::from_secs(1));
        }
        crate::mugetsu::set_exit_phase(3);
        tracing::info!("closing: done after {} ms", started.elapsed().as_millis());
        if crate::mugetsu::should_start_again() {
            crate::mugetsu::start_again_now();
        }
    }
}

enum LoadingState {
    Loading,
    WaitingForResize,
    Loaded,
}

/// How much of each second (ms) the main thread may spend on game frames and pictures
/// together with frame interpolation on; in-between pictures get what the game frames and
/// their pictures leave of it (see `MainWindow::pace_interpolation`). 650: a third stays spare.
/// `RUFFLE_INTERP_BUDGET` sets it (300-950), to compare: in a crowd, where scripts and the
/// game frames' own pictures take ~500 ms, 650 leaves room for ~20-30 in-betweens a second.
fn interpolation_budget_ms() -> f64 {
    static BUDGET: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let set = std::env::var("RUFFLE_INTERP_BUDGET")
            .ok()
            .and_then(|ms| ms.parse::<f64>().ok())
            .map(|ms| ms.clamp(300.0, 950.0));
        if let Some(ms) = set {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "frame interpolation: the main thread's budget is {ms:.0} ms per second (650 normally)"
            );
        }
        set.unwrap_or(650.0)
    })
}

/// `RUFFLE_EXTRA_RENDERS` restores redrawing between SWF frames (on `updateAfterEvent`, hover
/// changes, ...), for A/B comparisons.
fn extra_renders_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_EXTRA_RENDERS").is_some())
}

/// AQW build (testing): `RUFFLE_TEST_POINTER` feeds the movie a pointer of its own, so mouse
/// picking can be tested and timed without moving the user's cursor (the real cursor is then
/// ignored). `x,y`: a fixed place, as fractions of the movie's width and height (`0.5,0.6`).
/// `sweep` or `sweep:COLUMNSxROWS:MS`: across a grid of places (default 16x10), row by row,
/// one every MS milliseconds (default 150), round and round. The first place is given two
/// seconds after the movie loaded.
struct TestPointer {
    fixed: Option<(f64, f64)>,
    columns: u32,
    rows: u32,
    step: std::time::Duration,
    started: Option<Instant>,
    last: Option<u64>,
}

impl TestPointer {
    fn from_env() -> Option<Self> {
        let value = std::env::var("RUFFLE_TEST_POINTER").ok()?;
        let mut pointer = Self {
            fixed: None,
            columns: 16,
            rows: 10,
            step: std::time::Duration::from_millis(150),
            started: None,
            last: None,
        };
        if let Some(rest) = value.strip_prefix("sweep") {
            let mut parts = rest.trim_start_matches(':').split(':');
            if let Some((columns, rows)) = parts.next().and_then(|grid| grid.split_once('x')) {
                pointer.columns = columns.parse().ok().filter(|n| *n > 0)?;
                pointer.rows = rows.parse().ok().filter(|n| *n > 0)?;
            }
            if let Some(ms) = parts.next() {
                pointer.step =
                    std::time::Duration::from_millis(ms.parse().ok().filter(|n| *n > 0)?);
            }
        } else {
            let (x, y) = value.split_once(',')?;
            pointer.fixed = Some((x.trim().parse().ok()?, y.trim().parse().ok()?));
        }
        tracing::info!("Test pointer: {value}");
        Some(pointer)
    }

    /// The place to move to now, if it is time for another.
    fn position(&mut self, now: Instant) -> Option<(f64, f64)> {
        let started = *self.started.get_or_insert(now);
        let since = now
            .duration_since(started)
            .checked_sub(std::time::Duration::from_secs(2))?;
        let index = match self.fixed {
            Some(_) => 0,
            None => (since.as_millis() / self.step.as_millis().max(1)) as u64,
        };
        if self.last == Some(index) {
            return None;
        }
        self.last = Some(index);
        Some(match self.fixed {
            Some(place) => place,
            None => {
                let cell = index % u64::from(self.columns * self.rows);
                let (column, row) = (
                    cell % u64::from(self.columns),
                    cell / u64::from(self.columns),
                );
                (
                    (column as f64 + 0.5) / f64::from(self.columns),
                    (row as f64 + 0.5) / f64::from(self.rows),
                )
            }
        })
    }
}

/// `RUFFLE_UNBATCHED_MOUSE` (test switch): send every mouse move to the movie as it arrives
/// instead of at most once per frame.
fn unbatched_mouse_moves() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_UNBATCHED_MOUSE").is_some())
}
