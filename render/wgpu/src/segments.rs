//! AQW build: parallel command encoding.
//!
//! wgpu records render passes cheaply and does the expensive work (validation, resource
//! tracking, and the driver's own command recording) in `CommandEncoder::finish`. Crowded AQW
//! scenes record 1000-2000+ passes per frame, all in one encoder, so `finish` ran on one core.
//!
//! Instead, the frame's single command stream is cut into consecutive segments between render
//! passes; at submit time all segments are finished in parallel and submitted, in recording
//! order, in the same `queue.submit`. The GPU receives exactly the same commands in the same
//! order, within the same frame.

use crate::descriptors::Descriptors;
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

// Per thread, not in the shared `Descriptors`: one thread renders one player, and several
// players (on several threads) can share `Descriptors` (Ruffle's test runner does). A segment
// must only ever be submitted by the player that recorded it.
thread_local! {
    /// Recorded-but-unfinished segments of the current command stream, in order.
    static SEGMENTS: RefCell<Vec<wgpu::CommandEncoder>> = const { RefCell::new(Vec::new()) };
    /// Total render passes at the last cut.
    static CUT_AT: Cell<u64> = const { Cell::new(0) };
    /// Total draw commands at the last cut.
    static CUT_AT_DRAWS: Cell<u64> = const { Cell::new(0) };
}

/// Turu: segments finished and submitted together at most. On Metal, wgpu makes a command
/// buffer of every pass (and of the barriers before it) when an encoder is finished, and they
/// count against the queue's 4096 until submitted: a crowded frame with skill animations on
/// (2000+ passes) ran out in one submit, which wgpu treats as a lost device ("the graphics
/// driver stopped working"). A segment has about [`PASSES_PER_SEGMENT`] passes, so a batch
/// stays far below that. (Mugetsu's D3D12 merges a pass's command buffers; Metal doesn't.)
/// Only [`finish_in_batches`] batches: Mugetsu's present thread (Windows by default) finishes
/// a frame's segments all at once.
const SEGMENTS_PER_SUBMIT: usize = 16;

/// Render passes per segment: enough to amortize per-encoder overhead, few enough that a
/// heavy frame splits across all cores.
const PASSES_PER_SEGMENT: u64 = 48;
/// Or this many draw commands: an atlas page's passes hold the draws of dozens of layers, so a
/// segment of 48 of them would be far more work than one of 48 ordinary passes.
const DRAWS_PER_SEGMENT: u64 = 256;

fn cut_points() -> (u64, u64) {
    (
        crate::perf_counters::total_passes(),
        crate::perf_counters::total_draws(),
    )
}

fn set_cut(at: (u64, u64)) {
    CUT_AT.with(|c| c.set(at.0));
    CUT_AT_DRAWS.with(|c| c.set(at.1));
}

/// `RUFFLE_SERIAL_ENCODING` keeps the single-encoder behavior, for A/B comparisons.
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_SERIAL_ENCODING").is_none())
}

/// Called only where no render pass is open on `encoder` (between passes). If enough passes
/// were recorded since the last cut, moves them into a pending segment and continues
/// recording into a fresh encoder.
pub fn maybe_cut(descriptors: &Descriptors, encoder: &mut wgpu::CommandEncoder) {
    if !enabled() {
        return;
    }
    let now = cut_points();
    if now.0.saturating_sub(CUT_AT.with(Cell::get)) < PASSES_PER_SEGMENT
        && now.1.saturating_sub(CUT_AT_DRAWS.with(Cell::get)) < DRAWS_PER_SEGMENT
    {
        return;
    }
    set_cut(now);
    let recorded = std::mem::replace(
        encoder,
        descriptors
            .device
            .create_command_encoder(&Default::default()),
    );
    SEGMENTS.with(|s| s.borrow_mut().push(recorded));
    crate::perf_counters::inc(crate::perf_counters::SEGMENTS);
}

/// Cuts unconditionally (if enabled). For work that isn't counted in render passes, like the
/// cacheAsBitmap/filter phase of a frame. Only call where no render pass is open.
pub fn cut_now(descriptors: &Descriptors, encoder: &mut wgpu::CommandEncoder) {
    if !enabled() {
        return;
    }
    set_cut(cut_points());
    let recorded = std::mem::replace(
        encoder,
        descriptors
            .device
            .create_command_encoder(&Default::default()),
    );
    SEGMENTS.with(|s| s.borrow_mut().push(recorded));
    crate::perf_counters::inc(crate::perf_counters::SEGMENTS);
}

/// Finishes all pending segments followed by `tail` (the encoder still being recorded into),
/// in parallel, returning the command buffers in recording order for a single submit.
///
/// Turu: past [`SEGMENTS_PER_SUBMIT`], the earlier segments are finished and submitted here,
/// a batch at a time, in order; the last batch is returned. The staging belt must be finished
/// before this is called (as it always is), since the early batches reach the GPU first.
pub fn finish_all(
    descriptors: &Descriptors,
    tail: wgpu::CommandEncoder,
) -> Vec<wgpu::CommandBuffer> {
    finish_in_batches(descriptors, take_all(tail))
}

/// Turu: [`finish_encoders`], past [`SEGMENTS_PER_SUBMIT`] finishing and submitting the
/// earlier ones here, a batch at a time, in order; the last batch is returned (see
/// [`finish_all`]). Also for a frame whose new shapes' uploads go ahead of it
/// (`crate::shape_uploads`, on in Mugetsu's frontend).
pub fn finish_in_batches(
    descriptors: &Descriptors,
    mut encoders: Vec<wgpu::CommandEncoder>,
) -> Vec<wgpu::CommandBuffer> {
    while encoders.len() > SEGMENTS_PER_SUBMIT {
        let rest = encoders.split_off(SEGMENTS_PER_SUBMIT);
        descriptors.queue.submit(finish_encoders(encoders));
        encoders = rest;
    }
    finish_encoders(encoders)
}

/// Drops the segments recorded so far on this thread without submitting them: at shutdown
/// (they hold the device; this thread's storage is freed last of all, when the graphics
/// driver is already gone). Returns how many there were.
pub fn clear() -> usize {
    SEGMENTS.with(|s| std::mem::take(&mut *s.borrow_mut()).len())
}

/// Takes the segments recorded so far on this thread, in order, without finishing them: the
/// command stream goes on being recorded by another thread (see `crate::render_thread`),
/// which takes them over with `restore`.
pub fn take_pending() -> Vec<wgpu::CommandEncoder> {
    SEGMENTS.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Puts segments another thread recorded (`take_pending`) before this thread's own.
pub fn restore(mut earlier: Vec<wgpu::CommandEncoder>) {
    if earlier.is_empty() {
        return;
    }
    SEGMENTS.with(|s| {
        let mut segments = s.borrow_mut();
        earlier.append(&mut segments);
        *segments = earlier;
    });
}

/// Takes the pending segments followed by `tail`, in recording order, to be finished
/// elsewhere (`finish_encoders`; on the present thread, see `crate::present`). Only on the
/// thread that recorded them.
pub fn take_all(tail: wgpu::CommandEncoder) -> Vec<wgpu::CommandEncoder> {
    let mut encoders = SEGMENTS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    set_cut(cut_points());
    encoders.push(tail);
    encoders
}

/// Finishes `encoders` in parallel, returning their command buffers in the same order.
pub fn finish_encoders(encoders: Vec<wgpu::CommandEncoder>) -> Vec<wgpu::CommandBuffer> {
    if encoders.len() == 1 {
        return encoders.into_iter().map(|e| e.finish()).collect();
    }
    encoders.into_par_iter().map(|e| e.finish()).collect()
}
