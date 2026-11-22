#![deny(clippy::unwrap_used)]
// By default, Windows creates an additional console window for our program.
//
//
// This is silently ignored on non-windows systems.
// See https://docs.microsoft.com/en-us/cpp/build/reference/subsystem?view=msvc-160 for details.
#![windows_subsystem = "windows"]

mod app;
mod backends;
mod cli;
mod custom_event;
mod dbus;
mod gui;
mod heap_census;
mod log;
mod memory_split;
mod mugetsu;
mod player;
mod preferences;
mod test_hooks;
#[cfg(feature = "tracy")]
mod tracy;
mod util;
#[cfg(windows)]
mod windows;

use crate::preferences::GlobalPreferences;
use anyhow::{Context, Error};
use app::App;
#[cfg(feature = "shell-completions")]
use clap::CommandFactory;
use clap::Parser;
use cli::Opt;
pub use mugetsu::window_title;
use ruffle_core::StaticCallstack;
use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::panic::PanicHookInfo;
use tracing_subscriber::fmt::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

thread_local! {
    /// AQW build: set by `mugetsu_args` when this start uses DirectX 12 because the Vulkan
    /// driver crashed before (the crashed file), to log it once logging is set up.
    static VULKAN_CRASHED: RefCell<Option<std::path::PathBuf>> = RefCell::default();
    /// AQW build: set by `mugetsu_args` when this start uses DirectX 12 because the graphics
    /// chip is Intel's (its name), to log it once logging is set up.
    static INTEL_DX12: RefCell<Option<String>> = RefCell::default();
    /// AQW build: set by `mugetsu_args` when this start uses Vulkan because DirectX 12
    /// crashed while starting before (the crashed file), to log it once logging is set up.
    static DX12_CRASHED: RefCell<Option<std::path::PathBuf>> = RefCell::default();
    static CALLSTACK: RefCell<Option<StaticCallstack>> = RefCell::default();
    static RENDER_INFO: RefCell<Option<String>> = RefCell::default();
    static SWF_INFO: RefCell<Option<String>> = RefCell::default();
}

#[cfg(feature = "tracy")]
#[global_allocator]
static GLOBAL: tracing_tracy::client::ProfiledAllocator<std::alloc::System> =
    tracing_tracy::client::ProfiledAllocator::new(std::alloc::System, 0);

// AQW build: since command encoding and wgpu cleanup moved to other threads, the main thread
// spent ~1/3 of its time in crowded scenes contending for the Windows process heap lock (its
// many small per-frame allocations racing the workers'). mimalloc gives every thread its own
// heap.
#[cfg(not(feature = "tracy"))]
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// AQW build: mimalloc, plus a count of live Rust heap bytes for the perf log (to tell a leak
/// in Ruffle/wgpu from one in the graphics driver). Counted per thread and published every
/// 1 MB of change, so it costs no shared atomics on the hot path.
#[cfg(not(feature = "tracy"))]
pub struct CountingAlloc;

#[cfg(not(feature = "tracy"))]
pub static HEAP_BYTES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

#[cfg(not(feature = "tracy"))]
impl CountingAlloc {
    #[inline]
    fn record(delta: i64) {
        thread_local! {
            static PENDING: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
        }
        let _ = PENDING.try_with(|pending| {
            let value = pending.get() + delta;
            if value.abs() >= 1 << 20 {
                HEAP_BYTES.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
                pending.set(0);
            } else {
                pending.set(value);
            }
        });
    }
}

#[cfg(not(feature = "tracy"))]
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if !ptr.is_null() {
            Self::record(layout.size() as i64);
        }
        heap_census::on_alloc(ptr, layout.size());
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::record(layout.size() as i64);
        }
        heap_census::on_alloc(ptr, layout.size());
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // (Before freeing: once freed, another thread can be given the same address and
        // record it, and the census would then forget that allocation instead.)
        heap_census::on_dealloc(ptr, layout.size());
        unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) };
        Self::record(-(layout.size() as i64));
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let stack = heap_census::before_realloc(ptr, layout.size(), new_size);
        let new = unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            Self::record(new_size as i64 - layout.size() as i64);
        }
        heap_census::after_realloc(ptr, layout.size(), new, new_size, stack);
        new
    }
}

static RUFFLE_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "-",
    env!("CFG_RELEASE_CHANNEL"),
    " (",
    env!("VERGEN_GIT_SHA"),
    " ",
    env!("VERGEN_GIT_COMMIT_DATE"),
    ")"
);

fn panic_hook(info: &PanicHookInfo) {
    CALLSTACK.with(|callstack| {
        if let Some(callstack) = &*callstack.borrow() {
            callstack.avm2(|callstack| println!("AVM2 stack trace: {callstack}"))
        }
    });

    let message = info.payload_as_str().unwrap_or("panic occurred");

    // AQW build: keep the crash and where it happened in ruffle.log.
    tracing::error!(
        "fatal error: {info}\n{}",
        std::backtrace::Backtrace::force_capture()
    );

    // AQW build (Mugetsu): no "report to the Ruffle developers" link (this is a modified build;
    // its crashes aren't theirs to triage). Say plainly what happened instead.
    const GPU_RESET_HINTS: [&str; 6] = [
        "lost",
        "suspended",
        "removed",
        "Acquiring a texture failed",
        "Out of Memory",
        "out of memory",
    ];
    let description = if GPU_RESET_HINTS.iter().any(|hint| message.contains(hint)) {
        "The graphics driver stopped responding or was reset by Windows (for example after a \
        driver update or a driver crash, or because the graphics card ran out of memory), so \
        Mugetsu has to close.\n\n\
        Please start Mugetsu again. If this keeps happening in crowded rooms, set Quality to LOW \
        in AQW's options."
            .to_string()
    } else {
        format!(
            "Mugetsu ran into an error and has to close.\n\n{message}\n\n\
            Details were saved to %LOCALAPPDATA%\\ruffle\\log\\ruffle.log (it is replaced \
            the next time Mugetsu starts)."
        )
    };
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("Mugetsu")
        .set_description(description)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

/// AQW build (Mugetsu): the release ships this player as Mugetsu.exe, started by double-click
/// without arguments. Without a movie it plays AdventureQuest Worlds with Mugetsu's settings (the
/// ones Artix's own launcher uses for its Ruffle mode); options given on the command line win over
/// those defaults. `--no-obs` turns off OBS's Game Capture hook (its Vulkan layer) for the game.
fn mugetsu_args() -> Vec<std::ffi::OsString> {
    const AQW_LOADER_URL: &str = "https://game.aq.com/game/gamefiles/Loader3.swf?ver=a";

    let mut args: Vec<std::ffi::OsString> = env::args_os().collect();
    let is = |arg: &std::ffi::OsString, flag: &str| {
        arg.to_str().is_some_and(|a| a.eq_ignore_ascii_case(flag))
    };
    if args.iter().any(|a| is(a, "--no-obs")) {
        args.retain(|a| !is(a, "--no-obs"));
        // SAFETY: this runs first thing in `main`, before any other thread exists.
        unsafe { env::set_var("DISABLE_VULKAN_OBS_CAPTURE", "1") };
    }
    let gui = args.iter().any(|a| is(a, "--gui"));
    args.retain(|a| !is(a, "--gui"));
    let has_movie = Opt::try_parse_from(&args).map_or(true, |opt| opt.movie_url.is_some());
    if has_movie {
        // A movie was given (or the arguments don't parse; let clap report that): plain Ruffle.
        // (`RUFFLE_TEST_INTERPOLATION=1`, testing: with frame interpolation, as AQW can have.)
        if env::var_os("RUFFLE_TEST_INTERPOLATION").is_some() {
            ruffle_core::frame_interpolation::set_enabled(true);
        }
        // (`RUFFLE_TEST_SHAPE_WAIT=1`, testing: a frame waits for its new shapes only as long
        // as it does in AQW; a movie on the command line otherwise waits for them all.)
        if env::var_os("RUFFLE_TEST_SHAPE_WAIT").is_some() {
            ruffle_core::set_shape_wait(mugetsu::shape_wait());
        }
        // (`RUFFLE_TEST_FORGIVING_REMOVAL=1`, testing: a child removed again from inside its
        // own removal events is let through, as for AQW.)
        if env::var_os("RUFFLE_TEST_FORGIVING_REMOVAL").is_some() {
            ruffle_core::set_forgiving_removal(true);
        }
        // (`RUFFLE_TEST_UNCAPPED=1`, `RUFFLE_TEST_FPS_LIMIT=screen|N`, testing: uncapped
        // frame pacing, as AQW can have; see `mugetsu::load_pacing`.)
        if env::var_os("RUFFLE_TEST_UNCAPPED").is_some()
            || env::var_os("RUFFLE_TEST_FPS_LIMIT").is_some()
        {
            mugetsu::load_pacing();
        }
        // (`RUFFLE_TEST_PERFORMANCE=balanced|fast`, testing: Performance mode.)
        if env::var_os("RUFFLE_TEST_PERFORMANCE").is_some() {
            mugetsu::load_performance();
        }
        // (`RUFFLE_TEST_MEMORY_LIMIT_MB=<n>`, testing: the memory limit.)
        if env::var_os("RUFFLE_TEST_MEMORY_LIMIT_MB").is_some() {
            mugetsu::load_memory_limit();
        }
        // (`RUFFLE_TEST_SHOW_FPS=1`, testing: the FPS counter is shown, as a player can have
        // it; its changing number makes the window be drawn between the movie's pictures.)
        if env::var_os("RUFFLE_TEST_SHOW_FPS").is_some() {
            mugetsu::show_fps_for_test();
        }
        return args;
    }
    mugetsu::set_plays_aqw();
    ruffle_core::frame_interpolation::set_enabled(mugetsu::saved_interpolation());
    ruffle_core::set_shape_wait(mugetsu::shape_wait());
    ruffle_core::set_forgiving_removal(true);
    // Flash-style edge smoothing (much lighter on laptops). MSAA was a choice in the menu
    // until 2.0.0 (the user: not needed any more); an `msaa` file next to the exe still draws
    // with it, as the way back should an item look wrong. What an earlier version saved
    // (`mugetsu-edges.txt`) is not read any more.
    ruffle_render::tessellator::set_flash_aa(!mugetsu::exe_flag("msaa"));
    mugetsu::load_show_fps();
    mugetsu::load_pacing();
    mugetsu::load_performance();
    mugetsu::load_memory_limit();
    heap_census::enable_if_flagged();
    // AQW build: `layer-cache-check` file next to the exe = compare sampled cached layers with
    // fresh renders (see `ruffle_render_wgpu::layer_cache`).
    ruffle_render_wgpu::layer_cache::set_check_enabled(mugetsu::exe_flag("layer-cache-check"));
    let given = |flag: &str| {
        args.iter().skip(1).any(|a| {
            a.to_str()
                .is_some_and(|a| a == flag || a.starts_with(&format!("{flag}=")))
        })
    };
    // DirectX 12 is the default on Windows (since 2.0: as fast or faster than Vulkan in the
    // game, with far less memory held by the graphics driver). Vulkan is the default on Linux,
    // what the menu's other choice is, and what Windows uses where DirectX 12 crashed while
    // Mugetsu was starting (see `mugetsu::dx12_crashed_before`) or cannot be started at all
    // (see `GuiController::new`). The older rules stay for what they log and say in the
    // title: Intel/Qualcomm graphics (see `mugetsu::intel_graphics`), and a PC where the
    // Vulkan driver crashed while starting (see `mugetsu::vulkan_crashed_before`).
    let low_power = args
        .windows(2)
        .any(|pair| (is(&pair[0], "--power") || is(&pair[0], "-p")) && is(&pair[1], "low"))
        || args.iter().any(|a| is(a, "--power=low"));
    mugetsu::apply_drawing_switches();
    let graphics = if given("--graphics") || given("-g") {
        "vulkan" // unused: the given option wins
    } else if let Some(chosen) = mugetsu::saved_graphics() {
        #[cfg(target_os = "linux")]
        mugetsu::add_title_note(chosen.name());
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        mugetsu::add_title_note(format!("{} · right-click here to change", chosen.name()));
        #[cfg(target_os = "macos")]
        mugetsu::add_title_note(chosen.name());
        chosen.arg()
    } else if cfg!(target_os = "macos") {
        // Metal is the only graphics backend macOS offers (wgpu's Vulkan port needs MoltenVK).
        "metal"
    } else if let Some(module) = mugetsu::vulkan_crashed_before() {
        mugetsu::add_title_note("DirectX 12: Vulkan driver crashed on this PC");
        VULKAN_CRASHED.with(|crashed| *crashed.borrow_mut() = Some(module));
        "dx12"
    } else if let Some(module) = mugetsu::dx12_crashed_before() {
        // (Before the Intel rule: a DirectX 12 that crashes at start is no use there either.)
        mugetsu::add_title_note("Vulkan: DirectX 12 crashed on this PC");
        DX12_CRASHED.with(|crashed| *crashed.borrow_mut() = Some(module));
        "vulkan"
    } else if let Some(name) = mugetsu::intel_graphics(low_power) {
        INTEL_DX12.with(|intel| *intel.borrow_mut() = Some(name));
        "dx12"
    } else if cfg!(windows) {
        "dx12"
    } else {
        "vulkan"
    };
    // (option, short form, value)
    let mut defaults: Vec<(&str, Option<&str>, Option<&str>)> = vec![
        ("--tcp-connections", None, Some("allow")),
        ("--player-version", None, Some("9")),
    ];
    // `--width`/`--height` are physical pixels. On a Retina display 960x580 would make the
    // window half the apparent size of the 960x550 game, so let the loader SWF's stage size
    // decide there instead.
    #[cfg(not(target_os = "macos"))]
    defaults.extend([
        ("--width", None, Some("960")),
        ("--height", None, Some("580")),
    ]);
    defaults.extend([
        // Draw on the dedicated GPU when there is one.
        ("--power", Some("-p"), Some("high")),
        // No menu bar: the window is just the game.
        ("--no-gui", None, None),
        ("--graphics", Some("-g"), Some(graphics)),
    ]);
    let mut full = vec![args[0].clone(), AQW_LOADER_URL.into()];
    for (flag, short, value) in defaults {
        if (flag == "--no-gui" && gui) || given(flag) || short.is_some_and(given) {
            continue;
        }
        full.push(flag.into());
        if let Some(value) = value {
            full.push(value.into());
        }
    }
    full.extend(args.into_iter().skip(1));
    full
}

fn main() -> Result<(), Error> {
    // AQW build: started again after a graphics change: the previous Mugetsu ends first.
    mugetsu::wait_for_restarted_from();
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        prev_hook(info);
        panic_hook(info);
    }));
    #[cfg(windows)]
    windows::install_crash_handler();

    // AQW (since its 2026-10-10 update) uses Flash APIs newer than its loader's SWF version;
    // show them to every movie.
    ruffle_core::set_show_all_apis(true);

    #[cfg(windows)]
    let _console = windows::Console::attach();
    #[cfg(windows)]
    windows::aqw_tune_process();

    let opt = Opt::parse_from(mugetsu_args());

    #[cfg(feature = "shell-completions")]
    if let Some(shell) = opt.completions {
        let mut cmd = Opt::command();
        clap_complete::generate(shell, &mut cmd, "ruffle", &mut std::io::stdout());
        return Ok(());
    }

    let preferences = GlobalPreferences::load(opt)?;

    let logs_path = &preferences.cli.cache_directory.join("log");
    mugetsu::set_log_folder(logs_path.clone());
    let log_path = preferences.log_filename_pattern().create_path(logs_path);
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Err(err) = migrate_logs(&preferences, logs_path) {
        tracing::warn!("Failed to migrate logs: {}", err);
    }

    // [NA] `_guard` cannot be `_` or it'll immediately drop
    // https://docs.rs/tracing-appender/latest/tracing_appender/non_blocking/index.html
    let (non_blocking_file, _file_guard) = tracing_appender::non_blocking(File::create(log_path)?);
    let (non_blocking_stdout, _stdout_guard) = tracing_appender::non_blocking(std::io::stdout());

    // AQW build (Mugetsu): by default the log keeps warnings, crash details and Mugetsu's
    // performance/memory summaries, but not AQW's own trace output (it includes the login
    // response with the account's email and username) or its script errors (thousands per
    // session). A `verbose-log` file next to the executable restores Ruffle's usual logging;
    // RUST_LOG still overrides both.
    let verbose_log = mugetsu::exe_flag("verbose-log");
    let default_filter = if verbose_log {
        "warn,ruffle=info,avm_trace=info"
    } else {
        "warn,ruffle_desktop=info,ruffle_aqw_perf=info,ruffle_core=warn,ruffle_core::avm2=off,ruffle_core::stub=off,avm_trace=off"
    };
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .parse_lossy(env::var("RUST_LOG").as_deref().unwrap_or(default_filter));

    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(Layer::new().with_writer(non_blocking_stdout))
        .with(Layer::new().with_writer(non_blocking_file).with_ansi(false));

    #[cfg(feature = "tracy")]
    let subscriber = {
        let tracy_subscriber = tracing_tracy::TracyLayer::new(tracy::RuffleTracyConfig::default());
        subscriber.with(tracy_subscriber)
    };

    subscriber.init();

    tracing::info!("Mugetsu {} (Ruffle {RUFFLE_VERSION})", mugetsu::VERSION);
    // AQW build: the test switches this run was started with, so a log says which launcher
    // (or comparison) it is from.
    let mut switches: Vec<String> = env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.to_str()?.to_string();
            name.starts_with("RUFFLE_")
                .then(|| format!("{name}={}", value.to_string_lossy()))
        })
        .collect();
    if !switches.is_empty() {
        switches.sort();
        tracing::info!("Test switches: {}", switches.join(", "));
    }
    INTEL_DX12.with(|intel| {
        if let Some(name) = &*intel.borrow() {
            tracing::info!(
                "Using DirectX 12 on {name}: its Vulkan driver is less reliable (Intel, Qualcomm)"
            );
        }
    });
    VULKAN_CRASHED.with(|crashed| {
        if let Some(module) = &*crashed.borrow() {
            tracing::warn!(
                "Using DirectX 12: the Vulkan driver crashed on this PC before ({})",
                module.display()
            );
        }
    });
    DX12_CRASHED.with(|crashed| {
        if let Some(module) = &*crashed.borrow() {
            tracing::warn!(
                "Using Vulkan: DirectX 12 crashed on this PC while starting before ({})",
                module.display()
            );
        }
    });

    let result = App::new(preferences).and_then(|(mut app, event_loop)| {
        let result = event_loop.run_app(&mut app).context("Event loop failure");
        // AQW build: where a crash while closing happened (see `windows::crash_filter`).
        mugetsu::set_exit_phase(4);
        result
    });
    mugetsu::set_exit_phase(5);

    #[cfg(windows)]
    if let Err(error) = &result {
        eprintln!("{:?}", error)
    }

    // AQW build: returning normally runs every thread-local and static destructor while
    // Windows is already unloading DLLs; on a laptop (Intel, DirectX 12) that crashed inside
    // KERNELBASE (0x87A) after everything of ours had shut down and "closing: done" was logged.
    // Everything is dropped by now: flush the log and end the process here.
    if result.is_ok() {
        drop(_file_guard);
        drop(_stdout_guard);
        mugetsu::set_exit_phase(6);
        // A crash on another thread right now gets its report written first.
        #[cfg(windows)]
        windows::wait_for_crash_report();
        std::process::exit(0);
    }

    result
}

/// Move logs from config directory into proper log directory.
///
/// This exists because in older versions Ruffle created log files in the config directory.
///
/// TODO Remove this after some time.
fn migrate_logs(
    preferences: &GlobalPreferences,
    log_path: &std::path::Path,
) -> std::io::Result<()> {
    tracing::debug!("Migrating logs from config directory to log directory");
    let config_path = &preferences.cli.config;
    let paths = std::fs::read_dir(config_path)?;

    for dir in paths.flatten() {
        if dir.file_name().as_encoded_bytes().ends_with(b".log") {
            let src = dir.path();
            let dest = log_path.join(dir.file_name());
            tracing::info!(
                "Moving log file {} -> {}",
                src.to_string_lossy(),
                dest.to_string_lossy()
            );
            if let Err(err) = std::fs::rename(&src, &dest) {
                tracing::warn!(
                    "Failed to move log file {} -> {}: {err}",
                    src.to_string_lossy(),
                    dest.to_string_lossy()
                )
            }
        }
    }

    Ok(())
}
