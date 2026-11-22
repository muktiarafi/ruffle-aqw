//! AQW build (testing): things a test run makes the player do or report by itself, switched
//! on by environment variables. Nothing here does anything without them.

use ruffle_core::Player;
use ruffle_render::quality::StageQuality;
use ruffle_render_wgpu::perf_counters::timers;
use std::time::{Duration, Instant};

/// What a stretch of play cost, for the log: the time spent preparing shapes and waiting for
/// the present thread (ms in all), and the longest game frame and picture.
struct Costs {
    since: Instant,
    timers: [u64; timers::LEN],
    longest_frame_ms: f64,
    longest_picture_ms: f64,
    pictures: u32,
}

impl Costs {
    fn from(since: Instant, timers: [u64; timers::LEN]) -> Self {
        Self {
            since,
            timers,
            longest_frame_ms: 0.0,
            longest_picture_ms: 0.0,
            pictures: 0,
        }
    }

    fn report(&self) -> String {
        let now = timers::snapshot();
        let ms = |timer: usize| (now[timer] - self.timers[timer]) as f64 / 1000.0;
        format!(
            "shape_prepare {:.1} ms (tessellate {:.1}, shape_upload {:.1}), present_wait {:.1} ms, {} pictures, longest game frame {:.1} ms, longest picture {:.1} ms",
            ms(timers::SHAPES),
            ms(timers::SHAPE_TESSELLATE),
            ms(timers::SHAPE_UPLOAD),
            ms(timers::PRESENT_WAIT),
            self.pictures,
            self.longest_frame_ms,
            self.longest_picture_ms,
        )
    }
}

/// - `RUFFLE_TEST_START=SECONDS`: that long after the movie loaded, the log says what
///   everything up to then cost (entering a room: its shapes prepared for the first time).
/// - `RUFFLE_TEST_QUALITY=SECONDS`: changes the stage quality every SECONDS seconds (Medium,
///   High, Medium, ...; the first time 12 seconds after the movie loaded): with Flash-style
///   edges every shape on the stage is then prepared again. Four seconds after each change
///   the log says what those four seconds cost.
/// - `RUFFLE_TEST_WINDOW=SECONDS:ACTION,...`: that long after the movie loaded, something
///   done to the window (see `WindowAction`), for example `8:size:900x600,12:min,20:restore`.
#[derive(Default)]
pub struct TestHooks {
    loaded: Option<Instant>,
    /// What is still to be done to the window, and when (in order).
    window: std::collections::VecDeque<(Duration, WindowAction)>,
    /// When to report on the start, and its costs so far.
    start: Option<(Duration, Costs)>,
    /// How often the quality changes, and how many times it has.
    quality_every: Option<Duration>,
    quality_changes: u32,
    /// The quality change being measured.
    quality: Option<(StageQuality, Costs)>,
}

impl TestHooks {
    const FIRST_QUALITY_AFTER: Duration = Duration::from_secs(12);
    const QUALITY_MEASURED: Duration = Duration::from_secs(4);

    pub fn from_env() -> Self {
        let seconds = |name: &str, least: f64| {
            let seconds: f64 = std::env::var(name).ok()?.trim().parse().ok()?;
            (seconds >= least).then(|| Duration::from_secs_f64(seconds))
        };
        Self {
            // (The timers count from the start of the process.)
            start: seconds("RUFFLE_TEST_START", 0.5)
                .map(|after| (after, Costs::from(Instant::now(), [0; timers::LEN]))),
            quality_every: seconds("RUFFLE_TEST_QUALITY", 5.0),
            window: WindowAction::list_from_env(),
            ..Default::default()
        }
    }

    /// What is to be done to the window now (see `WindowAction::apply`).
    pub fn window_actions(&mut self, now: Instant) -> Vec<WindowAction> {
        let Some(loaded) = self.loaded else {
            return Vec::new();
        };
        let since = now.duration_since(loaded);
        let mut due = Vec::new();
        while self.window.front().is_some_and(|(at, _)| *at <= since) {
            due.extend(self.window.pop_front().map(|(_, action)| action));
        }
        due
    }

    /// A game frame took this long.
    pub fn frame(&mut self, ms: f64) {
        let costs = [
            self.start.as_mut().map(|(_, costs)| costs),
            self.quality.as_mut().map(|(_, costs)| costs),
        ];
        for costs in costs.into_iter().flatten() {
            costs.longest_frame_ms = costs.longest_frame_ms.max(ms);
        }
    }

    /// A picture of the movie took this long.
    pub fn picture(&mut self, ms: f64) {
        let costs = [
            self.start.as_mut().map(|(_, costs)| costs),
            self.quality.as_mut().map(|(_, costs)| costs),
        ];
        for costs in costs.into_iter().flatten() {
            costs.longest_picture_ms = costs.longest_picture_ms.max(ms);
            costs.pictures += 1;
        }
    }

    /// Before each tick of a loaded movie.
    pub fn tick(&mut self, now: Instant, player: &mut Player) {
        if self.start.is_none() && self.quality_every.is_none() && self.window.is_empty() {
            return;
        }
        let loaded = *self.loaded.get_or_insert(now);
        let since_loaded = now.duration_since(loaded);
        if let Some((after, costs)) = &self.start
            && since_loaded >= *after
        {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "test start: up to {:.1} s after the movie loaded ({:.1} s after the player started): {}",
                after.as_secs_f64(),
                now.duration_since(costs.since).as_secs_f64(),
                costs.report()
            );
            self.start = None;
        }
        if let Some((quality, costs)) = &self.quality
            && now.duration_since(costs.since) >= Self::QUALITY_MEASURED
        {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "test quality change to {quality:?}: in the {} s after it: {}",
                Self::QUALITY_MEASURED.as_secs(),
                costs.report()
            );
            self.quality = None;
        }
        let Some(every) = self.quality_every else {
            return;
        };
        let Some(since) = since_loaded.checked_sub(Self::FIRST_QUALITY_AFTER) else {
            return;
        };
        let due = (since.as_secs_f64() / every.as_secs_f64()) as u32 + 1;
        if due <= self.quality_changes {
            return;
        }
        self.quality_changes = due;
        let quality = if due % 2 == 1 {
            StageQuality::Medium
        } else {
            StageQuality::High
        };
        self.quality = Some((quality, Costs::from(now, timers::snapshot())));
        player.set_quality(quality);
    }
}

/// One thing a test does to the window (`RUFFLE_TEST_WINDOW`):
/// - `size:WIDTHxHEIGHT`: its inner size, in pixels;
/// - `min` and `restore`: minimised, and back;
/// - `full` and `windowed`: full screen (borderless), and back;
/// - `lose`: the graphics device is destroyed under the player, as when a driver is
///   restarted or the card is unplugged;
/// - `uncapped` and `vsync`: the frame pacing setting, as picked in the menu
///   (it is saved, so only with a `--config` folder of the test's own);
/// - `menu`: Mugetsu's right-click menu opens (at the pointer, or the window's corner), for
///   pictures of it;
/// - `interp-off` and `interp-on`: frame interpolation, as switched in the menu.
#[derive(Clone, Copy, Debug)]
pub enum WindowAction {
    Size(u32, u32),
    Minimise(bool),
    Fullscreen(bool),
    LoseDevice,
    Uncapped(bool),
    /// (Done by the app: see where the actions are applied.)
    Menu,
    Interpolation(bool),
    /// The offer to restart after frame interpolation was turned on.
    OfferRestart,
}

impl WindowAction {
    fn list_from_env() -> std::collections::VecDeque<(Duration, WindowAction)> {
        let Ok(value) = std::env::var("RUFFLE_TEST_WINDOW") else {
            return Default::default();
        };
        let parse = |entry: &str| {
            let (seconds, action) = entry.trim().split_once(':')?;
            let at = Duration::from_secs_f64(seconds.parse().ok().filter(|s: &f64| *s >= 0.0)?);
            let action = match action {
                "min" => WindowAction::Minimise(true),
                "restore" => WindowAction::Minimise(false),
                "full" => WindowAction::Fullscreen(true),
                "windowed" => WindowAction::Fullscreen(false),
                "lose" => WindowAction::LoseDevice,
                "uncapped" => WindowAction::Uncapped(true),
                "vsync" => WindowAction::Uncapped(false),
                "menu" => WindowAction::Menu,
                "offer-restart" => WindowAction::OfferRestart,
                "interp-off" => WindowAction::Interpolation(false),
                "interp-on" => WindowAction::Interpolation(true),
                size => {
                    let (width, height) = size.strip_prefix("size:")?.split_once('x')?;
                    WindowAction::Size(width.parse().ok()?, height.parse().ok()?)
                }
            };
            Some((at, action))
        };
        let mut list: Vec<_> = value.split(',').filter_map(parse).collect();
        list.sort_by_key(|(at, _)| *at);
        tracing::info!("Test window actions: {list:?}");
        list.into()
    }

    pub fn apply(self, window: &winit::window::Window, player: Option<&mut Player>) {
        tracing::info!(target: "ruffle_aqw_perf", "test window action: {self:?}");
        match self {
            WindowAction::Size(width, height) => {
                let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
            }
            WindowAction::Minimise(minimised) => window.set_minimized(minimised),
            WindowAction::Fullscreen(full) => {
                window.set_fullscreen(full.then_some(winit::window::Fullscreen::Borderless(None)));
            }
            WindowAction::Uncapped(on) => crate::mugetsu::set_uncapped(on),
            WindowAction::Menu | WindowAction::OfferRestart | WindowAction::Interpolation(_) => {}
            WindowAction::LoseDevice => {
                if let Some(player) = player {
                    ruffle_render_wgpu::render_thread::with_wgpu_backend(
                        player.renderer_mut(),
                        |renderer: &mut ruffle_render_wgpu::backend::WgpuRenderBackend<
                            crate::gui::MovieView,
                        >| renderer.descriptors().device.destroy(),
                    );
                }
            }
        }
    }
}
