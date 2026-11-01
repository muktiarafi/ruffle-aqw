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
// must only ever be submitted by the player that recorded it. A host that runs several players
// on one thread swaps each one's state in while it runs ([`swap`]).
thread_local! {
    /// Recorded-but-unfinished segments of the current command stream, in order.
    static SEGMENTS: RefCell<Vec<wgpu::CommandEncoder>> = const { RefCell::new(Vec::new()) };
    /// Total render passes at the last cut.
    static CUT_AT: Cell<u64> = const { Cell::new(0) };
    /// Total draw commands at the last cut.
    static CUT_AT_DRAWS: Cell<u64> = const { Cell::new(0) };
}

/// Turu: one player's segments and how far it got since its last cut, kept here while another
/// player on the same thread runs (see `ruffle_core::PlayerThreadState`). Otherwise a player's
/// submit takes the segments another player recorded (a hidden tab's `BitmapData.draw`), whose
/// staging belt that player hasn't finished: wgpu then fails the submit ("StagingBelt staging
/// buffer is still mapped"). A new player's state is `default()`, which is also what the
/// thread starts with.
#[derive(Default)]
pub struct ThreadState {
    segments: Vec<wgpu::CommandEncoder>,
    /// Passes and draws recorded since the last cut. Relative, not the totals at the cut: the
    /// totals count every player's passes, so a player swapped back in after others drew would
    /// otherwise cut between nearly every pass (a new encoder each time; seven tabs reached
    /// Metal's 4096 outstanding command buffers, which wgpu treats as a lost device).
    since_cut: (u64, u64),
}

/// Turu: swaps this player's state with the thread's.
pub fn swap(state: &mut ThreadState) {
    SEGMENTS.with(|s| std::mem::swap(&mut *s.borrow_mut(), &mut state.segments));
    let thread_cut = (CUT_AT.with(Cell::get), CUT_AT_DRAWS.with(Cell::get));
    let (cut, since) = rebase(cut_points(), thread_cut, state.since_cut);
    set_cut(cut);
    state.since_cut = since;
}

/// Turu: at the totals `now`, the thread's cut point for a player `since` past its last cut,
/// and how far past its cut the player that had `thread_cut` got.
fn rebase(now: (u64, u64), thread_cut: (u64, u64), since: (u64, u64)) -> ((u64, u64), (u64, u64)) {
    (
        (now.0.saturating_sub(since.0), now.1.saturating_sub(since.1)),
        (
            now.0.saturating_sub(thread_cut.0),
            now.1.saturating_sub(thread_cut.1),
        ),
    )
}

/// Turu: pending segments at most; past this, recording goes on in the current encoder. Each
/// unsubmitted segment holds one of Metal's 4096 command buffers, shared by every player, and a
/// player that isn't drawn submits only every 100 offscreen draws.
const MAX_PENDING_SEGMENTS: usize = 128;

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
    if (now.0.saturating_sub(CUT_AT.with(Cell::get)) < PASSES_PER_SEGMENT
        && now.1.saturating_sub(CUT_AT_DRAWS.with(Cell::get)) < DRAWS_PER_SEGMENT)
        || pending_full()
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
    if !enabled() || pending_full() {
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

/// Turu: [`MAX_PENDING_SEGMENTS`] are waiting.
fn pending_full() -> bool {
    SEGMENTS.with(|s| s.borrow().len() >= MAX_PENDING_SEGMENTS)
}

/// Finishes all pending segments followed by `tail` (the encoder still being recorded into),
/// in parallel, returning the command buffers in recording order for a single submit.
pub fn finish_all(
    descriptors: &Descriptors,
    tail: wgpu::CommandEncoder,
) -> Vec<wgpu::CommandBuffer> {
    let _ = descriptors;
    let mut encoders = SEGMENTS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    set_cut(cut_points());
    encoders.push(tail);
    if encoders.len() == 1 {
        return encoders.into_iter().map(|e| e.finish()).collect();
    }
    encoders.into_par_iter().map(|e| e.finish()).collect()
}

#[cfg(test)]
mod tests {
    use super::rebase;

    #[test]
    fn a_player_swapped_back_in_keeps_its_own_progress_since_its_cut() {
        // The shown tab A cut at 1000 passes and recorded 10 more; hidden B, 20 past its own
        // cut, is swapped in.
        let (cut, a_since) = rebase((1010, 0), (1000, 0), (20, 0));
        assert_eq!((cut, a_since), ((990, 0), (10, 0)));
        // B records 30 passes and is swapped out: A is back 10 past its cut, B 50 past its own.
        let (cut, b_since) = rebase((1040, 0), cut, a_since);
        assert_eq!((cut, b_since), ((1030, 0), (50, 0)));
        // A's frames record 4000 passes, its last cut at 5000. B swapped in again is still 50
        // past its cut, not 4050 (which would cut it between nearly every pass).
        let (cut, a_since) = rebase((5040, 0), (5000, 0), b_since);
        assert_eq!((cut, a_since), ((4990, 0), (40, 0)));
    }
}
