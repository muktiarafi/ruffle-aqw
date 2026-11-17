//! AQW build (diagnostic): a census of the large heap allocations alive, by call stack.
//!
//! The perf log's "rust heap" number says how much the player's own code holds, not what.
//! With a `heap-census` file next to the exe, every allocation of `MIN_SIZE` bytes or more
//! is recorded with its call stack (`RtlCaptureStackBackTrace`: a few microseconds and no
//! allocation of its own) until it is freed; after each memory line, the live ones are added
//! up by stack and the biggest stacks are logged, with symbols from dbghelp (the build's
//! .pdb), resolved on a thread of its own. Small allocations (GC objects, strings) are not
//! covered: their count is in the memory line. Windows only; off without the file, at no cost
//! beyond one size comparison per allocation.
#![cfg_attr(not(windows), allow(dead_code))]

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Allocations smaller than this aren't recorded: 64 KB, or the number of KB written in the
/// `heap-census` file (4 covers most of the heap, at a few percent of CPU while loading).
static MIN_SIZE: AtomicUsize = AtomicUsize::new(64 * 1024);
/// Call stack frames kept per allocation.
const DEPTH: usize = 24;
/// Stacks logged per report.
const TOP: usize = 10;
/// Functions logged per report (bytes added up by the innermost function of each stack).
const TOP_SITES: usize = 15;
/// Frames shown per stack (after the allocator's own are skipped).
const SHOWN: usize = 8;

type Stack = [usize; DEPTH];

static ENABLED: AtomicBool = AtomicBool::new(false);
static REPORTING: AtomicBool = AtomicBool::new(false);
/// Live recorded allocations: address -> (size, stack).
static LIVE: Mutex<Option<HashMap<usize, (usize, Stack)>>> = Mutex::new(None);

thread_local! {
    /// Set while this thread is inside the census, so the census's own allocations (the
    /// map's) aren't recorded, and the lock isn't taken twice.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// Turns the census on when a `heap-census` file sits next to the exe.
pub fn enable_if_flagged() {
    if !cfg!(windows) || !crate::mugetsu::exe_flag("heap-census") {
        return;
    }
    if let Some(kb) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("heap-census")))
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| text.trim().parse::<usize>().ok())
        .filter(|kb| *kb > 0)
    {
        MIN_SIZE.store(kb * 1024, Ordering::Relaxed);
    }
    if let Ok(mut live) = LIVE.lock() {
        *live = Some(HashMap::new());
    }
    ENABLED.store(true, Ordering::Release);
    tracing::info!(
        target: "ruffle_aqw_perf",
        "heap census on: allocations of {} KB and more are recorded with their call stacks; a report follows each memory line",
        min_size() / 1024
    );
}

#[inline]
fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
fn min_size() -> usize {
    MIN_SIZE.load(Ordering::Relaxed)
}

/// Runs `f` unless this thread is already inside the census.
fn guarded(f: impl FnOnce()) {
    let _ = BUSY.try_with(|busy| {
        if busy.get() {
            return;
        }
        busy.set(true);
        f();
        busy.set(false);
    });
}

#[cfg(windows)]
fn capture() -> Stack {
    let mut stack = [0usize; DEPTH];
    // SAFETY: `stack` has `DEPTH` slots and the call writes at most that many.
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(
            2,
            DEPTH as u32,
            stack.as_mut_ptr().cast(),
            std::ptr::null_mut(),
        );
    }
    stack
}

#[cfg(not(windows))]
fn capture() -> Stack {
    [0; DEPTH]
}

#[inline]
pub fn on_alloc(ptr: *mut u8, size: usize) {
    if size < min_size() || ptr.is_null() || !enabled() {
        return;
    }
    guarded(|| {
        let stack = capture();
        if let Ok(mut live) = LIVE.lock()
            && let Some(map) = live.as_mut()
        {
            map.insert(ptr as usize, (size, stack));
        }
    });
}

#[inline]
pub fn on_dealloc(ptr: *mut u8, size: usize) {
    if size < min_size() || !enabled() {
        return;
    }
    guarded(|| {
        if let Ok(mut live) = LIVE.lock()
            && let Some(map) = live.as_mut()
        {
            map.remove(&(ptr as usize));
        }
    });
}

/// A reallocation, in two steps around it: `before_realloc` forgets the old block (before it
/// can be freed and its address given to another thread), `after_realloc` records the new one
/// with the old stack (or the old block again, if the reallocation failed).
#[inline]
pub fn before_realloc(old: *mut u8, old_size: usize, new_size: usize) -> Option<Stack> {
    if (old_size < min_size() && new_size < min_size()) || !enabled() {
        return None;
    }
    let mut stack = None;
    guarded(|| {
        if let Ok(mut live) = LIVE.lock()
            && let Some(map) = live.as_mut()
        {
            stack = map.remove(&(old as usize)).map(|(_, stack)| stack);
        }
    });
    stack
}

#[inline]
pub fn after_realloc(
    old: *mut u8,
    old_size: usize,
    new: *mut u8,
    new_size: usize,
    stack: Option<Stack>,
) {
    if (old_size < min_size() && new_size < min_size()) || !enabled() {
        return;
    }
    // Failed: the old block is still there.
    let (ptr, size) = if new.is_null() { (old, old_size) } else { (new, new_size) };
    if size < min_size() {
        return;
    }
    guarded(|| {
        // Grew past the threshold: the stack of the growth is as good a name.
        let stack = stack.unwrap_or_else(capture);
        if let Ok(mut live) = LIVE.lock()
            && let Some(map) = live.as_mut()
        {
            map.insert(ptr as usize, (size, stack));
        }
    });
}

/// Logs the biggest stacks (on a thread of its own; skipped while the last report runs).
pub fn report() {
    if !enabled() || REPORTING.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut snapshot: Vec<(usize, Stack)> = Vec::new();
    guarded(|| {
        if let Ok(live) = LIVE.lock()
            && let Some(map) = live.as_ref()
        {
            snapshot = map.values().copied().collect();
        }
    });
    let spawned = std::thread::Builder::new()
        .name("heap-census".into())
        .spawn(move || {
            report_snapshot(&snapshot);
            REPORTING.store(false, Ordering::Release);
        });
    if spawned.is_err() {
        REPORTING.store(false, Ordering::Release);
    }
}

fn report_snapshot(snapshot: &[(usize, Stack)]) {
    const MB: f64 = 1024.0 * 1024.0;
    let mut by_stack: HashMap<Stack, (usize, usize)> = HashMap::new();
    let mut total = 0usize;
    for (size, stack) in snapshot {
        total += size;
        let entry = by_stack.entry(*stack).or_default();
        entry.0 += size;
        entry.1 += 1;
    }
    let mut rows: Vec<_> = by_stack.into_iter().collect();
    rows.sort_by_key(|(_, (bytes, _))| std::cmp::Reverse(*bytes));
    tracing::info!(
        target: "ruffle_aqw_perf",
        "heap census: {} allocations of {} KB and more alive, {:.0} MB in {} distinct stacks; the biggest:",
        snapshot.len(),
        min_size() / 1024,
        total as f64 / MB,
        rows.len()
    );
    let mut resolver = Resolver::new();
    for (stack, (bytes, count)) in rows.iter().take(TOP) {
        tracing::info!(
            target: "ruffle_aqw_perf",
            "  {:.1} MB in {count}: {}",
            *bytes as f64 / MB,
            resolver.describe(stack).join(" < ")
        );
    }
    // The same bytes added up by the innermost function that isn't the allocator's own
    // (inlined frames included), which merges the stacks that differ only in how they got
    // there.
    let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
    for (stack, (bytes, count)) in &rows {
        let site = resolver
            .describe(stack)
            .into_iter()
            .next()
            .unwrap_or_else(|| "?".to_string());
        let entry = by_site.entry(site).or_default();
        entry.0 += bytes;
        entry.1 += count;
    }
    let mut sites: Vec<_> = by_site.into_iter().collect();
    sites.sort_by_key(|(_, (bytes, _))| std::cmp::Reverse(*bytes));
    tracing::info!(target: "ruffle_aqw_perf", "heap census by function:");
    for (site, (bytes, count)) in sites.iter().take(TOP_SITES) {
        tracing::info!(target: "ruffle_aqw_perf", "  {:.1} MB in {count}: {site}", *bytes as f64 / MB);
    }
}

/// Frames that are the allocator's own, or too generic to say anything. Library paths are
/// matched at the start of the name (`<` of a trait impl skipped): matched anywhere,
/// `core::` also hid every `ruffle_core::` frame.
fn is_noise(name: &str) -> bool {
    const PREFIXES: [&str; 11] = [
        "alloc::alloc",
        "alloc::raw_vec",
        "alloc::vec::",
        "alloc::boxed",
        "alloc::string",
        "alloc::collections",
        "hashbrown",
        "core::",
        "std::",
        "mi_",
        "mimalloc",
    ];
    const ANYWHERE: [&str; 7] = [
        "heap_census",
        "CountingAlloc",
        "__rust_alloc",
        "__rust_realloc",
        "__rust_dealloc",
        "RtlCapture",
        "finish_grow",
    ];
    let start = name.trim_start_matches('<');
    PREFIXES.iter().any(|prefix| start.starts_with(prefix))
        || ANYWHERE.iter().any(|part| name.contains(part))
}

#[cfg(windows)]
struct Resolver {
    process: windows_sys::Win32::Foundation::HANDLE,
    /// Per return address: its frames, innermost (inlined) first, each "function (file:line)".
    cache: HashMap<usize, Vec<String>>,
}

#[cfg(windows)]
impl Resolver {
    fn new() -> Self {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            SYMOPT_DEFERRED_LOADS, SYMOPT_LOAD_LINES, SYMOPT_UNDNAME, SymInitializeW,
            SymSetOptions,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: plain dbghelp calls on the current process; initializing twice (the standard
        // library may have, for a panic's backtrace) just returns false.
        let process = unsafe {
            let process = GetCurrentProcess();
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            SymInitializeW(process, std::ptr::null(), 1);
            process
        };
        Self {
            process,
            cache: HashMap::new(),
        }
    }

    /// The frames at a return address, innermost first: the functions inlined there (from
    /// the PDB's inline records; with LTO and PGO most callers are inlined) and then the one
    /// containing it, each with the file and line of the call.
    fn names(&mut self, ip: usize) -> Vec<String> {
        if let Some(names) = self.cache.get(&ip) {
            return names.clone();
        }
        use windows_sys::Win32::System::Diagnostics::Debug::{
            IMAGEHLP_LINEW64, SYMBOL_INFOW, SymAddrIncludeInlineTrace, SymFromAddrW,
            SymFromInlineContextW, SymGetLineFromAddrW64, SymGetLineFromInlineContextW,
            SymQueryInlineTrace,
        };
        const NAME_CHARS: usize = 512;
        let words = (std::mem::size_of::<SYMBOL_INFOW>() + NAME_CHARS * 2).div_ceil(8);
        let mut buffer = vec![0u64; words];
        let symbol = buffer.as_mut_ptr().cast::<SYMBOL_INFOW>();
        // The return address is one past the call: look up the call itself.
        let addr = ip as u64 - 1;
        let process = self.process;
        // SAFETY: the buffer holds a SYMBOL_INFOW plus NAME_CHARS name characters, as the
        // size fields say; the line struct is initialized with its size; dbghelp fills them.
        let symbol_name = |symbol: *mut SYMBOL_INFOW| unsafe {
            let len = ((*symbol).NameLen as usize).min(NAME_CHARS);
            let chars =
                std::slice::from_raw_parts(std::ptr::addr_of!((*symbol).Name).cast::<u16>(), len);
            format!("{:#}", rustc_demangle::demangle(&String::from_utf16_lossy(chars)))
        };
        let line_text = |line: &IMAGEHLP_LINEW64| unsafe {
            if line.FileName.is_null() {
                return String::new();
            }
            let mut len = 0;
            while *line.FileName.add(len) != 0 {
                len += 1;
            }
            let file = String::from_utf16_lossy(std::slice::from_raw_parts(line.FileName, len));
            let short = file.rsplit(['\\', '/']).next().unwrap_or(&file).to_string();
            format!(" ({short}:{})", line.LineNumber)
        };
        let mut names = Vec::new();
        unsafe {
            let inlined = SymAddrIncludeInlineTrace(process, addr);
            if inlined > 0 {
                let (mut context, mut index) = (0u32, 0u32);
                if SymQueryInlineTrace(process, addr, 0, addr, addr, &mut context, &mut index) != 0 {
                    for ctx in context..context + inlined {
                        (*symbol).SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
                        (*symbol).MaxNameLen = NAME_CHARS as u32;
                        let mut displacement = 0u64;
                        if SymFromInlineContextW(process, addr, ctx, &mut displacement, symbol) != 0 {
                            let mut line: IMAGEHLP_LINEW64 = std::mem::zeroed();
                            line.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
                            let mut disp = 0u32;
                            let text = if SymGetLineFromInlineContextW(
                                process, addr, ctx, 0, &mut disp, &mut line,
                            ) != 0
                            {
                                line_text(&line)
                            } else {
                                String::new()
                            };
                            names.push(format!("{}{text}", symbol_name(symbol)));
                        }
                    }
                }
            }
            (*symbol).SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
            (*symbol).MaxNameLen = NAME_CHARS as u32;
            let mut displacement = 0u64;
            if SymFromAddrW(process, addr, &mut displacement, symbol) != 0 {
                let mut line: IMAGEHLP_LINEW64 = std::mem::zeroed();
                line.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
                let mut disp = 0u32;
                let text = if SymGetLineFromAddrW64(process, addr, &mut disp, &mut line) != 0 {
                    line_text(&line)
                } else {
                    String::new()
                };
                names.push(format!("{}{text}", symbol_name(symbol)));
            }
        }
        if names.is_empty() {
            names.push(format!("{ip:#x}"));
        }
        self.cache.insert(ip, names.clone());
        names
    }

    fn describe(&mut self, stack: &Stack) -> Vec<String> {
        let names: Vec<String> = stack
            .iter()
            .take_while(|ip| **ip != 0)
            .flat_map(|ip| self.names(*ip))
            .collect();
        let shown: Vec<String> = names
            .iter()
            .filter(|name| !is_noise(name))
            .take(SHOWN)
            .cloned()
            .collect();
        if shown.is_empty() {
            names.into_iter().take(3).collect()
        } else {
            shown
        }
    }
}

#[cfg(not(windows))]
struct Resolver;

#[cfg(not(windows))]
impl Resolver {
    fn new() -> Self {
        Self
    }

    fn describe(&mut self, _stack: &Stack) -> Vec<String> {
        vec!["(stacks are only resolved on Windows)".to_string()]
    }
}
