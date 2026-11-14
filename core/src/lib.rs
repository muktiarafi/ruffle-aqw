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
/// AQW build: the size detailed shapes are tessellated for at least (see `tessellation_cache`).
pub use tessellation_cache::set_min_size as set_shape_min_px;
/// AQW build: how long a frame waits for its new shapes (see `deferred_shapes`).
pub use display_object::graphic::deferred_shapes::set_wait as set_shape_wait;
/// AQW build: lets go of the shapes still being tessellated, at shutdown (see
/// `deferred_shapes`).
pub use display_object::graphic::deferred_shapes::shutdown as shutdown_deferred_shapes;
/// AQW build: frame interpolation (an option, off by default).
pub use display_object::interp as frame_interpolation;
/// Turu: a text field's selection, for Turu's Android build, which keeps the game's focused field
/// in step with a native text field above the soft keyboard.
pub use display_object::TextSelection;

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

/// Turu, for the perf log: the uncaught AS3 errors since the last call, and how many of them
/// were logged (a repeat is only counted, see `avm2::aqw_errors`).
pub use avm2::aqw_errors::take as take_error_stats;

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

/// AQW build: whether a child may be removed again from inside its own removal events without
/// an error (see `remove_child_from_displaylist` in the AVM2 display object container). Flash
/// Player throws #2094 there, and so does Ruffle since October 2026 (it started a second
/// removal before). AQW's skill effects do exactly that half-way through their clean-up
/// (`SpellW.killSpell`, run from their own `removedFromStage` listener when the game takes
/// one off: `parent.removeChild(this)`, then `stop()` and dropping what they point to): with
/// the error they are never stopped, play on off the stage with their frame scripts failing
/// (#1009), and keep their references until they are collected. The Mugetsu app turns this on
/// for AQW, which is how every Mugetsu so far has behaved; `RUFFLE_STRICT_REMOVAL` (testing)
/// keeps Flash's error.
static FORGIVING_REMOVAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_forgiving_removal(enabled: bool) {
    FORGIVING_REMOVAL.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn forgiving_removal() -> bool {
    static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    FORGIVING_REMOVAL.load(std::sync::atomic::Ordering::Relaxed)
        && !*STRICT.get_or_init(|| std::env::var_os("RUFFLE_STRICT_REMOVAL").is_some())
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