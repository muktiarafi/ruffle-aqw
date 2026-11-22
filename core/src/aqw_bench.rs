//! AQW build: an offline load benchmark (a test, run on demand).
//!
//! Loads real AQW files from `AQW_CORPUS` (a folder, searched for .swf files) into one player
//! the way the game loads gear: each through AS3's `Loader.loadBytes`, the loader added to the
//! stage so the loaded movie runs its first frames, `AQW_REPEAT` times over (default 3, as
//! several players wear the same item). Then reports the heap the loads left (every Rust
//! allocation is counted), the time they took and the shapes shared. The same files in the
//! same order every run, without a login or a server: before/after comparisons are exact.
//!
//! `AQW_CORPUS=<folder> cargo test --release --locked -p ruffle_core aqw_load_bench -- --ignored --nocapture`

use crate::avm2::bytearray::ByteArrayStorage;
use crate::avm2::object::ByteArrayObject;
use crate::avm2::{Activation, FunctionArgs, Value};
use crate::display_object::TDisplayObject;
use crate::player::PlayerBuilder;
use crate::string::AvmString;
use crate::tag_utils::SwfMovieData;
use ruffle_common::duration::FloatDuration;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

/// Bytes allocated and not freed, by the whole test binary.
static HEAP: AtomicI64 = AtomicI64::new(0);

struct Counting;

// SAFETY: forwards to the system allocator, only counting.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            HEAP.fetch_add(layout.size() as i64, Ordering::Relaxed);
            census::on_alloc(ptr, layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            HEAP.fetch_add(layout.size() as i64, Ordering::Relaxed);
            census::on_alloc(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        census::on_free(ptr);
        unsafe { System.dealloc(ptr, layout) };
        HEAP.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        census::on_free(ptr);
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            HEAP.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
            census::on_alloc(new, new_size);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `AQW_CENSUS=1`: what holds the heap the loads left, by call stack. A sample: one allocation
/// for every `STEP` bytes allocated, counted as that many bytes, with its call stack, until
/// it is freed; `report` adds the live ones up by the function that asked and by stack (names
/// from the test binary's .pdb). Windows only.
#[cfg(windows)]
mod census {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const DEPTH: usize = 30;
    const STEP: usize = 1024;
    type Stack = [usize; DEPTH];

    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    static LIVE: Mutex<Option<HashMap<usize, (usize, Stack)>>> = Mutex::new(None);

    thread_local! {
        /// Set while this thread is inside the census: its own allocations are not sampled.
        static BUSY: Cell<bool> = const { Cell::new(false) };
    }

    #[repr(C)]
    struct SymbolInfo {
        size_of_struct: u32,
        type_index: u32,
        reserved: [u64; 2],
        index: u32,
        size: u32,
        mod_base: u64,
        flags: u32,
        value: u64,
        address: u64,
        register: u32,
        scope: u32,
        tag: u32,
        name_len: u32,
        max_name_len: u32,
        name: [u8; 4],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn RtlCaptureStackBackTrace(
            skip: u32,
            capture: u32,
            trace: *mut *mut c_void,
            hash: *mut u32,
        ) -> u16;
        fn GetCurrentProcess() -> *mut c_void;
    }

    #[link(name = "dbghelp", kind = "raw-dylib")]
    unsafe extern "system" {
        fn SymSetOptions(options: u32) -> u32;
        fn SymInitialize(process: *mut c_void, search_path: *const u8, invade: i32) -> i32;
        fn SymFromAddr(
            process: *mut c_void,
            address: u64,
            displacement: *mut u64,
            symbol: *mut SymbolInfo,
        ) -> i32;
    }

    pub fn start() {
        BUSY.with(|busy| busy.set(true));
        *LIVE.lock().expect("census") = Some(HashMap::new());
        BUSY.with(|busy| busy.set(false));
        ENABLED.store(true, Ordering::Release);
    }

    #[inline]
    pub fn on_alloc(ptr: *mut u8, size: usize) {
        if !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let before = ALLOCATED.fetch_add(size, Ordering::Relaxed);
        let steps = (before + size) / STEP - before / STEP;
        if steps == 0 || BUSY.try_with(|busy| busy.replace(true)).unwrap_or(true) {
            return;
        }
        let mut stack: Stack = [0; DEPTH];
        // SAFETY: `stack` has room for `DEPTH` pointers.
        unsafe {
            RtlCaptureStackBackTrace(
                2,
                DEPTH as u32,
                stack.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
        }
        if let Ok(mut live) = LIVE.lock()
            && let Some(live) = live.as_mut()
        {
            live.insert(ptr as usize, (steps * STEP, stack));
        }
        let _ = BUSY.try_with(|busy| busy.set(false));
    }

    #[inline]
    pub fn on_free(ptr: *mut u8) {
        if !ENABLED.load(Ordering::Relaxed)
            || BUSY.try_with(|busy| busy.replace(true)).unwrap_or(true)
        {
            return;
        }
        if let Ok(mut live) = LIVE.lock()
            && let Some(live) = live.as_mut()
        {
            live.remove(&(ptr as usize));
        }
        let _ = BUSY.try_with(|busy| busy.set(false));
    }

    /// Frames that only pass an allocation on: the caller is what asked for the memory.
    fn passes_on(name: &str) -> bool {
        const WRAPPERS: [&str; 22] = [
            "alloc::",
            "core::",
            "std::",
            "hashbrown::",
            "indexmap::",
            "smallvec::",
            "gc_arena::",
            "bytemuck",
            "__rust",
            "__rdl",
            "__rg_",
            "aqw_bench",
            "RawVec",
            "impl$",
            "enum2$",
            "tuple$",
            "fnv::",
            "slotmap::",
            "ruffle_wstr::",
            "ruffle_core::string::",
            "ruffle_core::avm2::object::script_object::",
            "?",
        ];
        std::env::var_os("AQW_CENSUS_RAW").is_none()
            && WRAPPERS.iter().any(|wrapper| name.starts_with(wrapper))
    }

    pub fn report(loads: usize) {
        ENABLED.store(false, Ordering::Release);
        BUSY.with(|busy| busy.set(true));
        let live = LIVE.lock().expect("census").take().unwrap_or_default();
        // SAFETY: plain calls into dbghelp with this process's pseudo-handle.
        // (The test binary names its .pdb without a folder: dbghelp is told where it is.)
        let folder = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.display().to_string()))
            .unwrap_or_default();
        let folder = std::ffi::CString::new(folder).unwrap_or_default();
        let process = unsafe {
            SymSetOptions(0x2);
            let process = GetCurrentProcess();
            let ok = SymInitialize(process, folder.as_ptr() as *const u8, 1);
            if ok == 0 {
                println!("  census: SymInitialize failed");
            }
            process
        };
        let mut names: HashMap<usize, String> = HashMap::new();
        let mut name_of = |addr: usize| -> String {
            names
                .entry(addr)
                .or_insert_with(|| {
                    let mut buffer = vec![0u64; (size_of::<SymbolInfo>() + 1024) / 8];
                    let symbol = buffer.as_mut_ptr() as *mut SymbolInfo;
                    let mut displacement = 0u64;
                    // SAFETY: the buffer holds a `SymbolInfo` and 1000 more bytes for the name.
                    unsafe {
                        (*symbol).size_of_struct = 88;
                        (*symbol).max_name_len = 1000;
                        if SymFromAddr(process, addr as u64 - 1, &mut displacement, symbol) == 0 {
                            return format!("?{addr:x}");
                        }
                        let name = std::slice::from_raw_parts(
                            (symbol as *const u8).add(84),
                            (*symbol).name_len.min(1000) as usize,
                        );
                        let name = String::from_utf8_lossy(name).into_owned();
                        match name.rfind("::h") {
                            Some(at) if name.len() - at == 19 => name[..at].to_owned(),
                            _ => name,
                        }
                    }
                })
                .clone()
        };

        let mut by_stack: HashMap<Stack, usize> = HashMap::new();
        let mut total = 0usize;
        for (bytes, stack) in live.values() {
            *by_stack.entry(*stack).or_default() += *bytes;
            total += *bytes;
        }
        let mut by_site: HashMap<String, usize> = HashMap::new();
        let mut by_area: HashMap<String, usize> = HashMap::new();
        let mut stacks: Vec<(usize, Vec<String>)> = Vec::new();
        for (stack, bytes) in &by_stack {
            let named: Vec<String> = stack
                .iter()
                .take_while(|addr| **addr != 0)
                .map(|addr| name_of(*addr))
                .filter(|name| !passes_on(name))
                .collect();
            let site = named
                .first()
                .cloned()
                .unwrap_or_else(|| "(unknown)".to_owned());
            let area = site.split("::").take(3).collect::<Vec<_>>().join("::");
            *by_site.entry(site).or_default() += *bytes;
            *by_area.entry(area).or_default() += *bytes;
            stacks.push((*bytes, named));
        }
        let mb = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        let per_load = |bytes: usize| bytes as f64 / 1024.0 / loads.max(1) as f64;
        println!(
            "  census: {:.1} MB alive in {} sampled allocations ({} stacks), {:.1} KB a load",
            mb(total),
            live.len(),
            by_stack.len(),
            per_load(total)
        );
        let mut areas: Vec<_> = by_area.into_iter().collect();
        areas.sort_by(|a, b| b.1.cmp(&a.1));
        println!("  census, by area:");
        for (area, bytes) in areas.iter().take(30) {
            println!(
                "    {:7.1} MB {:5.1}% {:6.1} KB/load  {area}",
                mb(*bytes),
                *bytes as f64 * 100.0 / total.max(1) as f64,
                per_load(*bytes)
            );
        }
        let mut sites: Vec<_> = by_site.into_iter().collect();
        sites.sort_by(|a, b| b.1.cmp(&a.1));
        println!("  census, by the function that asked:");
        for (site, bytes) in sites.iter().take(60) {
            println!(
                "    {:7.1} MB {:5.1}% {:6.1} KB/load  {site}",
                mb(*bytes),
                *bytes as f64 * 100.0 / total.max(1) as f64,
                per_load(*bytes)
            );
        }
        stacks.sort_by(|a, b| b.0.cmp(&a.0));
        println!("  census, the biggest stacks:");
        for (bytes, named) in stacks.iter().take(40) {
            println!(
                "    {:7.1} MB {:5.1}%  {}",
                mb(*bytes),
                *bytes as f64 * 100.0 / total.max(1) as f64,
                named
                    .iter()
                    .take(7)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" < ")
            );
        }
        BUSY.with(|busy| busy.set(false));
    }
}

#[cfg(not(windows))]
mod census {
    pub fn start() {}
    pub fn on_alloc(_: *mut u8, _: usize) {}
    pub fn on_free(_: *mut u8) {}
    pub fn report(_: usize) {}
}

const MB: f64 = 1024.0 * 1024.0;

fn heap() -> i64 {
    HEAP.load(Ordering::Relaxed)
}

/// Every .swf file under `dir`, sorted (the same order every run).
fn swf_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            swf_files(&path, files);
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("swf"))
        {
            files.push(path);
        }
    }
}

/// A one-frame AS3 movie to be the root (the game's own client is the root in AQW).
fn root_movie() -> SwfMovieData {
    let header = swf::Header {
        compression: swf::Compression::None,
        version: 32,
        stage_size: swf::Rectangle {
            x_min: swf::Twips::ZERO,
            x_max: swf::Twips::from_pixels(960.0),
            y_min: swf::Twips::ZERO,
            y_max: swf::Twips::from_pixels(550.0),
        },
        frame_rate: swf::Fixed8::from_f32(24.0),
        num_frames: 1,
    };
    let tags = [
        swf::Tag::FileAttributes(swf::FileAttributes::IS_ACTION_SCRIPT_3),
        swf::Tag::ShowFrame,
    ];
    let mut bytes = Vec::new();
    swf::write_swf(&header, &tags, &mut bytes).expect("root movie");
    SwfMovieData::from_data(&bytes, "file:///aqw_bench.swf".to_string(), None, None).expect("root")
}

#[test]
#[ignore]
fn aqw_load_bench() {
    let dir = std::env::var("AQW_CORPUS").expect("AQW_CORPUS: a folder of AQW .swf files");
    let repeat: usize = std::env::var("AQW_REPEAT")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(3);
    let mut files = Vec::new();
    swf_files(std::path::Path::new(&dir), &mut files);
    files.sort();
    let data: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter_map(|path| Some((path.display().to_string(), std::fs::read(path).ok()?)))
        .collect();
    let file_bytes: usize = data.iter().map(|(_, bytes)| bytes.len()).sum();

    let start_heap = heap();
    let player = PlayerBuilder::new().with_movie(root_movie()).build();
    let mut player = player.lock().expect("player");
    player.set_is_playing(true);
    let frame = FloatDuration::from_millis(1000.0 / 24.0);
    for _ in 0..3 {
        player.tick(frame);
    }
    let base_heap = heap();
    let census = std::env::var_os("AQW_CENSUS").is_some();
    if census {
        census::start();
    }

    let started = Instant::now();
    let (mut loaded, mut failed) = (0usize, 0usize);
    for round in 0..repeat {
        for (name, bytes) in &data {
            let ok = player.mutate_with_update_context(|context| {
                // As the game client's code (its domain), which is what calls `Loader` in AQW.
                let domain = context.avm2.stage_domain();
                let mut activation = Activation::from_domain(context, domain);
                let load = |activation: &mut Activation<'_, '_>| -> Result<(), String> {
                    let loader_class = activation.avm2().classes().loader;
                    let loader = loader_class
                        .construct(activation, &[])
                        .map_err(|e| format!("{e:?}"))?;
                    let storage = ByteArrayStorage::from_vec(activation.context, bytes.clone());
                    let byte_array = ByteArrayObject::from_storage(activation.context, storage);
                    let load_bytes = AvmString::new_utf8(activation.gc(), "loadBytes");
                    loader
                        .call_public_property(
                            load_bytes,
                            FunctionArgs::from_slice(&[byte_array.into()]),
                            activation,
                        )
                        .map_err(|e| format!("{e:?}"))?;
                    let stage = activation.context.stage.object2().ok_or("no AS3 stage")?;
                    let add_child = AvmString::new_utf8(activation.gc(), "addChild");
                    Value::from(stage)
                        .call_public_property(
                            add_child,
                            FunctionArgs::from_slice(&[loader]),
                            activation,
                        )
                        .map_err(|e| format!("{e:?}"))?;
                    Ok(())
                };
                match load(&mut activation) {
                    Ok(()) => true,
                    Err(e) => {
                        if round == 0 {
                            eprintln!("load failed: {name}: {e}");
                        }
                        false
                    }
                }
            });
            if ok {
                loaded += 1;
            } else {
                failed += 1;
            }
            // A few frames for each, as the game runs while gear streams in: preloading,
            // the class definitions, the first frames.
            for _ in 0..2 {
                player.tick(frame);
            }
        }
    }
    // Settle: the last loads finish, and the garbage collector catches up.
    for _ in 0..48 {
        player.tick(frame);
    }
    let elapsed = started.elapsed();
    let end_heap = heap();
    let (parsed, shared) = crate::display_object::graphic::shared_shapes::counts();
    let movies = player.mutate_with_update_context(|context| context.library.loaded_movie_count());

    println!(
        "AQW load benchmark: {} files ({:.1} MB), {repeat} rounds",
        data.len(),
        file_bytes as f64 / MB
    );
    println!(
        "  loads: {loaded} ok, {failed} failed, {:.1} s ({:.2} ms per load)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / loaded.max(1) as f64
    );
    println!(
        "  player alone: {:.1} MB",
        (base_heap - start_heap) as f64 / MB
    );
    println!(
        "  heap from the loads: {:.1} MB ({:.2} MB per load), {movies} movies alive",
        (end_heap - base_heap) as f64 / MB,
        (end_heap - base_heap) as f64 / MB / loaded.max(1) as f64
    );
    println!("  shape tags: parsed {parsed}, shared {shared}");
    if census {
        census::report(loaded);
    }

    // `AQW_STAY_FRAMES=<n>`: everything stays on the stage for n more frames: what testing
    // libraries that are all in use costs (their revival at the end of each GC cycle).
    if let Some(frames) = std::env::var("AQW_STAY_FRAMES")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
    {
        let (tested0, revived0) = crate::library::library_gc_counts();
        let (_, sum0, count0) = crate::player::gc_finalize_micros(true);
        let stay_started = std::time::Instant::now();
        let mut longest_tick = 0.0f64;
        for _ in 0..frames {
            let tick = std::time::Instant::now();
            player.tick(frame);
            longest_tick = longest_tick.max(tick.elapsed().as_secs_f64() * 1000.0);
        }
        let (tested, revived) = crate::library::library_gc_counts();
        let (longest, sum, count) = crate::player::gc_finalize_micros(true);
        let (alive, freed) = player.mutate_with_update_context(|context| {
            (
                context.library.loaded_movie_count(),
                context.library.freed_movie_count(),
            )
        });
        println!(
            "  all on stage, {frames} frames ({:.1} s): {} GC cycles, {} libraries tested, {} revived, {freed} freed, {alive} alive; steps at a cycle's end: longest {:.2} ms, {:.2} ms a cycle; longest frame {:.1} ms",
            stay_started.elapsed().as_secs_f64(),
            count - count0,
            tested - tested0,
            revived - revived0,
            longest as f64 / 1000.0,
            (sum - sum0) as f64 / 1000.0 / (count - count0).max(1) as f64,
            longest_tick
        );
    }

    // `AQW_UNLOAD=1`: takes every loader off the stage again and lets the collector run
    // (`AQW_UNLOAD_FRAMES` frames, default 2000; set `RUFFLE_GC_CYCLE_SECONDS=0.25` so it
    // gets through enough cycles): how many of the movies' libraries are freed (see
    // `Library::finalize_movie_libraries`), and what is left of the heap.
    if std::env::var_os("AQW_UNLOAD").is_some() {
        player.mutate_with_update_context(|context| {
            let domain = context.avm2.stage_domain();
            let mut activation = Activation::from_domain(context, domain);
            if let Some(stage) = activation.context.stage.object2() {
                let remove_children = AvmString::new_utf8(activation.gc(), "removeChildren");
                if let Err(e) = Value::from(stage).call_public_property(
                    remove_children,
                    FunctionArgs::from_slice(&[]),
                    &mut activation,
                ) {
                    eprintln!("removeChildren failed: {e:?}");
                }
            }
        });
        // `AQW_COLLECT_NOW=1`: what "Free memory now" does (`free_unused_libraries_now`),
        // two frames after the loaders went.
        if std::env::var_os("AQW_COLLECT_NOW").is_some() {
            for _ in 0..2 {
                player.tick(frame);
            }
            let heap_before = heap();
            let (freed, ms) = player.free_unused_libraries_now();
            let alive =
                player.mutate_with_update_context(|context| context.library.loaded_movie_count());
            println!(
                "  free now: {freed} libraries freed in {ms:.0} ms, {alive} movies alive, heap {:.1} -> {:.1} MB",
                (heap_before - base_heap) as f64 / MB,
                (heap() - base_heap) as f64 / MB
            );
        }
        let frames: usize = std::env::var("AQW_UNLOAD_FRAMES")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(2000);
        let (_, _, count0) = crate::player::gc_finalize_micros(true);
        let mut curve = Vec::new();
        for i in 0..frames {
            player.tick(frame);
            if (i + 1) % (frames / 16).max(1) == 0 {
                let alive = player
                    .mutate_with_update_context(|context| context.library.loaded_movie_count());
                let (_, _, count) = crate::player::gc_finalize_micros(false);
                curve.push(format!("{} cycles: {alive}", count - count0));
            }
        }
        let (alive, freed) = player.mutate_with_update_context(|context| {
            (
                context.library.loaded_movie_count(),
                context.library.freed_movie_count(),
            )
        });
        let (longest, _, _) = crate::player::gc_finalize_micros(false);
        println!(
            "  off the stage again, {frames} frames later: {alive} movies alive, {freed} libraries freed, heap from the loads {:.1} MB; alive after {}; longest step at a cycle's end {:.2} ms",
            (heap() - base_heap) as f64 / MB,
            curve.join(", "),
            longest as f64 / 1000.0
        );
    }
}

/// The offline proof for `frame_lifecycle::early_scripts`, with a real colour-custom item:
/// `AQW_CC_FILE=<its .swf> AQW_CC_CLASS=<one of its part classes>` (a part whose coloured
/// pieces colour themselves from their first frame scripts). The part is made and put on the
/// stage between two frames, as the game builds an avatar's parts when a load finishes or its
/// load throttle's timer fires; then a picture is drawn, then a frame runs. Prints how many
/// frame scripts had run at each point and how many clips the picture drew with their script
/// still waiting (those are the ones seen in the colours they were drawn in). Run it once as
/// it is and once with `RUFFLE_EARLY_SCRIPTS=1`.
#[test]
#[ignore]
fn aqw_early_scripts() {
    use crate::frame_lifecycle::early_scripts;
    let file = std::env::var("AQW_CC_FILE").expect("AQW_CC_FILE: a colour-custom item's .swf");
    let class = std::env::var("AQW_CC_CLASS").expect("AQW_CC_CLASS: a part class in it");
    let bytes = std::fs::read(&file).expect("the file");

    let player = PlayerBuilder::new().with_movie(root_movie()).build();
    let mut player = player.lock().expect("player");
    player.set_is_playing(true);
    let frame = FloatDuration::from_millis(1000.0 / 24.0);
    for _ in 0..3 {
        player.tick(frame);
    }
    let loaded = player.update(|context| -> Result<(), String> {
        let domain = context.avm2.stage_domain();
        let mut activation = Activation::from_domain(context, domain);
        let loader_class = activation.avm2().classes().loader;
        let loader = loader_class
            .construct(&mut activation, &[])
            .map_err(|e| format!("{e:?}"))?;
        let storage = ByteArrayStorage::from_vec(activation.context, bytes.clone());
        let byte_array = ByteArrayObject::from_storage(activation.context, storage);
        let load_bytes = AvmString::new_utf8(activation.gc(), "loadBytes");
        loader
            .call_public_property(
                load_bytes,
                FunctionArgs::from_slice(&[byte_array.into()]),
                &mut activation,
            )
            .map_err(|e| format!("{e:?}"))?;
        let stage = activation.context.stage.object2().ok_or("no AS3 stage")?;
        let add_child = AvmString::new_utf8(activation.gc(), "addChild");
        Value::from(stage)
            .call_public_property(
                add_child,
                FunctionArgs::from_slice(&[loader]),
                &mut activation,
            )
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    });
    loaded.expect("the load");
    for _ in 0..12 {
        player.tick(frame);
    }

    let scripts_before = early_scripts::all_scripts();
    let (_, early_before, drawn_before) = early_scripts::counts();
    // Between two frames: `new Part()`, `stage.addChild(part)`.
    let made = player.update(|context| -> Result<(), String> {
        let domain = context.avm2.stage_domain();
        let mut activation = Activation::from_domain(context, domain);
        let stage = Value::from(activation.context.stage.object2().ok_or("no AS3 stage")?);
        fn name<'gc>(activation: &mut Activation<'_, 'gc>, text: &str) -> AvmString<'gc> {
            AvmString::new_utf8(activation.gc(), text)
        }
        let get_child_at = name(&mut activation, "getChildAt");
        let loader = stage
            .call_public_property(
                get_child_at,
                FunctionArgs::from_slice(&[1.into()]),
                &mut activation,
            )
            .map_err(|e| format!("{e:?}"))?;
        let mut value = loader;
        for property in ["contentLoaderInfo", "applicationDomain"] {
            let property = name(&mut activation, property);
            value = value
                .get_public_property(property, &mut activation)
                .map_err(|e| format!("{e:?}"))?;
        }
        let get_definition = name(&mut activation, "getDefinition");
        let class_name = name(&mut activation, &class);
        let part_class = value
            .call_public_property(
                get_definition,
                FunctionArgs::from_slice(&[class_name.into()]),
                &mut activation,
            )
            .map_err(|e| format!("{e:?}"))?;
        let part = part_class
            .construct(&mut activation, FunctionArgs::from_slice(&[]))
            .map_err(|e| format!("{e:?}"))?;
        let add_child = name(&mut activation, "addChild");
        stage
            .call_public_property(
                add_child,
                FunctionArgs::from_slice(&[part]),
                &mut activation,
            )
            .map_err(|e| format!("{e:?}"))?;
        Ok(())
    });
    made.expect("the part");
    let scripts_made = early_scripts::all_scripts();
    let (_, early_made, _) = early_scripts::counts();

    player.render();
    let (_, _, drawn_picture) = early_scripts::counts();

    player.tick(frame);
    let scripts_frame = early_scripts::all_scripts();
    player.render();
    let (_, _, drawn_after) = early_scripts::counts();
    for _ in 0..5 {
        player.tick(frame);
    }
    let scripts_later = early_scripts::all_scripts();

    println!(
        "early scripts {}: frame scripts run when the part was on the stage: {} ({} of them early); clips in the first picture with their script still waiting: {}; scripts run by the next frame: {}; waiting in the picture after it: {}; scripts run in five more frames: {}",
        if early_scripts::enabled() {
            "ON"
        } else {
            "OFF"
        },
        scripts_made - scripts_before,
        early_made - early_before,
        drawn_picture - drawn_before,
        scripts_frame - scripts_made,
        drawn_after - drawn_picture,
        scripts_later - scripts_frame
    );
}
