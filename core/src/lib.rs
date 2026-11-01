// This lint is helpful, but right now we have too many instances of it.
// TODO: Remove this once all instances are fixed.
#![allow(clippy::needless_pass_by_ref_mut)]
// This lint is good in theory, but in AVMs we often need to do `let x = args.get(0); let y = args.get(1);` etc.
// It'd make those much less readable and consistent.
#![allow(clippy::get_first)]

#[macro_use]
mod display_object;
pub use display_object::{StageAlign, StageDisplayState, StageScaleMode};
/// AQW build: frees GPU geometry of shapes not drawn for a while (when GPU memory is low).
pub use tessellation_cache::evict_unused as evict_unused_shapes;
/// AQW build: how long a frame waits for its new shapes (see `deferred_shapes`).
pub use display_object::graphic::deferred_shapes::set_wait as set_shape_wait;
/// AQW build: frame interpolation (an option, off by default).
pub use display_object::interp as frame_interpolation;

/// Turu: the thread-local state that belongs to one player: its queued and running shape
/// tessellations (`deferred_shapes`), the change count its recorded pictures are checked
/// against (`replay`) and its frame interpolation's frame count. With several players on one
/// thread, the one in front keeps its state in the thread; each other player keeps its own
/// here, swapped in while its code runs, so a hidden player's frames neither spoil the shown
/// one's replays and interpolation nor leave tessellations only its renderer can finish.
///
/// A new player's state is `default()`, which is also what the thread starts with.
#[derive(Default)]
pub struct PlayerThreadState {
    shapes: display_object::graphic::deferred_shapes::State,
    replay_changes: u64,
    interp: display_object::interp::ThreadState,
}

impl PlayerThreadState {
    /// Swaps this state with the thread's.
    pub fn swap(&mut self) {
        display_object::graphic::deferred_shapes::swap(&mut self.shapes);
        display_object::replay::swap_changes(&mut self.replay_changes);
        display_object::interp::swap(&mut self.interp);
    }
}

#[macro_use]
extern crate num_derive;

#[macro_use]
mod aqw_retention;
#[cfg(test)]
mod aqw_bench;
mod avm1;
mod avm2;
mod avm_rng;
mod binary_data;
pub mod bitmap;
mod character;
pub mod context;
pub mod context_menu;
mod drawing;
mod ecma_conversions;
pub mod events;
pub mod focus_tracker;
pub mod font;
mod frame_lifecycle;
pub mod fte;
mod html;
mod input;
mod library;
pub mod limits;
pub mod loader;
mod local_connection;
mod net_connection;
mod orphan_manager;
pub mod pixel_bender;
mod player;
mod prelude;
pub mod socket;
mod streams;
pub mod string;
mod system_properties;
pub mod tag_utils;
mod tessellation_cache;
pub mod timer;
mod types;
mod vminterface;

pub mod backend;
pub mod compatibility_rules;
pub mod config;
#[cfg(feature = "egui")]
pub mod debug_ui;
pub mod external;
pub mod i18n;
pub mod stub;

pub use context_menu::ContextMenuItem;
pub use events::PlayerEvent;
pub use indexmap;
pub use loader::LoadBehavior;
pub use player::{Player, PlayerBuilder, PlayerMode, PlayerRuntime, StaticCallstack};
pub use ruffle_common::duration::FloatDuration;
pub use ruffle_render::backend::ViewportDimensions;
pub use swf;
pub use swf::Color;
pub use ttf_parser;

/// The newest Flash Player version known to Ruffle.
pub const NEWEST_PLAYER_VERSION: u8 = 51;

/// The default Flash Player version that Ruffle will emulate.
pub const DEFAULT_PLAYER_VERSION: u8 = 32;

/// Path to the raw playerglobal ABC, for use as an `asc.jar` import library.
pub const PLAYERGLOBAL_ABC_PATH: &str = env!("RUFFLE_PLAYERGLOBAL_ABC_PATH");

/// AQW build: whether every movie sees the Flash APIs of newer SWF versions than its own (see
/// `UpdateContext::set_root_movie`). The Mugetsu app turns this on at startup.
static SHOW_ALL_APIS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_show_all_apis(enabled: bool) {
    SHOW_ALL_APIS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn show_all_apis() -> bool {
    SHOW_ALL_APIS.load(std::sync::atomic::Ordering::Relaxed)
}

/// AQW build: total time spent building render commands (walking the display list in
/// `Player::render`, incl. deferred shape tessellation), in microseconds, for the perf log.
static RENDER_WALK_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn add_render_walk_micros(micros: u64) {
    RENDER_WALK_MICROS.fetch_add(micros, std::sync::atomic::Ordering::Relaxed);
}

/// AQW build: frame interpolation's replays (see `display_object::replay`).
pub mod interpolation_replay {
    pub use crate::display_object::replay::{STAT_NAMES, set_enabled, stats, take_reason};
}

pub fn render_walk_micros() -> u64 {
    RENDER_WALK_MICROS.load(std::sync::atomic::Ordering::Relaxed)
}

/// AQW build: whether cacheAsBitmap textures are sized in steps (see
/// `display_object::cache_texture_size`). A runtime switch for A/B comparisons.
static CACHE_SIZE_STEPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_cache_size_steps(enabled: bool) {
    CACHE_SIZE_STEPS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// `RUFFLE_NO_CACHE_STEPS` turns the steps off for good (A/B comparisons in tests).
pub fn cache_size_steps() -> bool {
    static ENV_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    CACHE_SIZE_STEPS.load(std::sync::atomic::Ordering::Relaxed)
        && !*ENV_OFF.get_or_init(|| std::env::var_os("RUFFLE_NO_CACHE_STEPS").is_some())
}
