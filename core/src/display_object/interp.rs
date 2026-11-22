//! AQW build (Mugetsu): frame interpolation, an option that is off by default.
//!
//! AQW runs its scripts and timelines at 24-30 frames a second. With interpolation on, the
//! window is redrawn at the monitor's refresh rate, and each redraw places every object between
//! where it was on the previous SWF frame and where it is now, by how much of the time between
//! the two frames has passed. Only drawing uses the blended transforms: scripts, hit tests and
//! everything else only ever see the real frames. The price is that each frame shows up to one
//! frame later than without interpolation.
//!
//! Objects that appeared, were swapped for another, flipped, turned or jumped far since the
//! previous frame are drawn where they are (see `blend_matrix`).

use crate::prelude::*;
use ruffle_render::filters::Filter;
use ruffle_render::transform::Transform;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use swf::{ColorTransform, Fixed8, Fixed16};
use web_time::Instant;

/// Moves longer than this (in the parent's pixels) within one frame are jumps, not motion.
const JUMP_PX: f64 = 100.0;

/// Turns sharper than this within one frame (radians) are drawn as they are.
const MAX_TURN: f32 = 2.1;

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// SWF frames run so far (see `new_frame`).
    static FRAME: Cell<u64> = const { Cell::new(1) };
    /// When the current SWF frame finished running (its picture became ready).
    static FRAME_FINISHED: Cell<Option<Instant>> = const { Cell::new(None) };
    /// While the screen is drawn with interpolation: how far it is from the previous SWF frame
    /// (0) to the current one (1).
    static BLEND: Cell<Option<f32>> = const { Cell::new(None) };
    /// The length of a SWF frame in ms, as last given to `begin_render` (0: not yet).
    static FRAME_MS: Cell<f64> = const { Cell::new(0.0) };
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Called when a SWF frame starts running.
pub fn new_frame() {
    FRAME.with(|f| f.set(f.get() + 1));
}

/// Called when a SWF frame finished running. Blending is timed from here, so how long the
/// frame's scripts took doesn't make the motion skip ahead.
pub fn frame_finished() {
    FRAME_FINISHED.with(|f| f.set(Some(Instant::now())));
}

/// The number of the current SWF frame.
pub fn frame() -> u64 {
    FRAME.with(Cell::get)
}

/// Called before the screen is drawn; `frame_ms` is the length of a SWF frame.
pub fn begin_render(frame_ms: f64) {
    FRAME_MS.with(|f| f.set(frame_ms));
    let blend = if enabled() && frame_ms > 0.0 {
        FRAME_FINISHED
            .with(Cell::get)
            .map(|started| {
                (started.elapsed().as_secs_f64() * 1000.0 / frame_ms).clamp(0.0, 1.0) as f32
            })
            // (`RUFFLE_TEST_INTERP_AMOUNT=0.5`, testing: every picture is drawn that far
            // between two frames, so two runs' in-between pictures can be compared.)
            .map(|amount| test_amount().unwrap_or(amount))
    } else {
        None
    };
    BLEND.with(|b| b.set(blend));
}

fn test_amount() -> Option<f32> {
    static AMOUNT: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *AMOUNT.get_or_init(|| {
        std::env::var("RUFFLE_TEST_INTERP_AMOUNT")
            .ok()
            .and_then(|amount| amount.parse::<f32>().ok())
            .filter(|amount| (0.0..=1.0).contains(amount))
    })
}

/// Called after the screen was drawn (other drawing, such as `BitmapData.draw`, is never blended).
pub fn end_render() {
    BLEND.with(|b| b.set(None));
}

/// Whether the screen is being drawn with interpolation right now.
pub fn active() -> bool {
    BLEND.with(Cell::get).is_some()
}

/// While the screen is drawn with interpolation: how far it is from the previous SWF frame (0)
/// to the current one (1).
pub fn amount() -> Option<f32> {
    BLEND.with(Cell::get)
}

/// The transform an object is drawn with `t` of the way from how it looked on the previous
/// frame (`from`) to its real `transform` (changed into it).
pub fn blend_look(from: (Matrix, ColorTransform), transform: &mut Transform, t: f32) {
    transform.matrix = blend_matrix(&from.0, &transform.matrix, t);
    transform.color_transform = blend_color(&from.1, &transform.color_transform, t);
}

/// What an object looked like: its transform, and its ratio (how far a shape tween is).
type Look = (Matrix, ColorTransform, u16);

struct State {
    /// The SWF frame this object was last drawn in, and how it looked then.
    frame: u64,
    seen: Look,
    /// How it looked at the end of the frame before `frame`, if it was drawn then.
    previous: Option<Look>,
    /// The same for its filters (glows, shadows, ...), kept only for objects that have some.
    seen_filters: Option<Box<[Filter]>>,
    previous_filters: Option<Box<[Filter]>>,
    /// For a shape tween that isn't moved on every frame (see `Tracker::ratio`): the ratio it
    /// was drawn at when its ratio last changed, and the frame that was in (0: it hasn't
    /// changed since it came on screen).
    ratio_from: u16,
    ratio_changed: u64,
    /// Over how many frames that change is drawn (1: within its own frame, as usual), and how
    /// many frames lay between the last two changes (0: not known, or it turned round then).
    ratio_span: u8,
    ratio_gap: u8,
}

/// A tween moved on by a script every few frames, up to this long apart, is drawn moving
/// through the whole interval; one moved on less often jumps, as it's meant to. (AQW's aura
/// timers move on in the first frame that comes 100 ms or more after the last time: every 3
/// or 4 frames at 30 fps, whichever way the milliseconds fall.)
const MAX_HELD_MS: f64 = 140.0;

/// `MAX_HELD_MS` in SWF frames: 4 at 30 fps, 3 at 24.
fn max_held_frames() -> u8 {
    let frame_ms = FRAME_MS.with(Cell::get);
    if frame_ms > 0.0 {
        (MAX_HELD_MS / frame_ms).round().clamp(2.0, 60.0) as u8
    } else {
        4
    }
}

/// What's known about how the instances of one shape tween are moved on: kept with the tween
/// (`MorphShapeShared`), so an instance that has just come on screen, or moves for the first
/// time, can fall in with the others (see `State::join`). AQW's aura timers are four clips of
/// one tween, a quarter turn each: every quarter a new instance takes over, and learning the
/// rhythm again by itself took it three moves, drawn in jumps.
#[derive(Default)]
pub struct Rhythm(Cell<Option<Beat>>);

/// An instance moved on in `frame`, drawn over `span` frames (the longest, if several did).
#[derive(Clone, Copy)]
struct Beat {
    frame: u64,
    span: u8,
}

/// A new instance of a tween whose others have a rhythm is drawn from the tween's start if
/// it's no further from it than this (an eighth of the tween; see `State::join`).
const NEAR_START: u16 = 8192;

impl State {
    fn previous_at(&self, frame: u64) -> Option<Look> {
        if self.frame == frame {
            self.previous
        } else if self.frame + 1 == frame {
            Some(self.seen)
        } else {
            None
        }
    }

    /// Called at the object's first draw in `frame`, before `seen` is replaced, with the ratio
    /// it has now: notes when a tween moved on and how regularly it does.
    fn note_ratio(&mut self, frame: u64, ratio: u16) {
        if self.frame + 1 != frame {
            // Not drawn in the frame before: nothing is known about its rhythm.
            self.ratio_span = 1;
            self.ratio_gap = 0;
            self.ratio_changed = 0;
            return;
        }
        if ratio == self.seen.2 {
            return;
        }
        // Where it was drawn when the last frame ended: the next move starts from there.
        let from = self.held_at(frame - 1, 1.0).unwrap_or(self.seen.2);
        let gap = match self.ratio_changed {
            0 => 255,
            changed => frame.saturating_sub(changed).min(255) as u8,
        };
        let max = max_held_frames();
        let regular = |gap: u8| (2..=max).contains(&gap);
        // Going back the way it came (a timer started over, say) is a jump, not its rhythm.
        let way = |from: u16, to: u16| (i32::from(to) - i32::from(from)).signum();
        let turned = way(self.seen.2, ratio) * way(self.ratio_from, self.seen.2) < 0;
        self.ratio_span = if turned || gap < 2 || !by_distance() {
            1
        } else if self.ratio_span > 1 && gap <= max.saturating_mul(2) {
            // Its rhythm goes on, a beat it sat out included (a slow timer has nowhere new to
            // go on some beats). Drawn over the longest interval it has had: drawn over the
            // last one, it stood still for a frame whenever the next move came a frame later,
            // and on an uneven rhythm (3 or 4 frames) that was every few moves. This way it
            // starts each move from where it has got to, so it's always under way, and one
            // that moves on evenly in time is drawn at an even speed.
            if gap <= max {
                self.ratio_span.max(gap)
            } else {
                self.ratio_span
            }
        } else if regular(gap) && regular(self.ratio_gap) {
            // A rhythm: two intervals in a row of a few frames each.
            gap.max(self.ratio_gap)
        } else {
            1
        };
        self.ratio_from = from;
        self.ratio_gap = if turned { 0 } else { gap };
        self.ratio_changed = frame;
    }

    /// Called for a shape tween each time it's drawn in `frame` (after `note_ratio`), with
    /// what's known of how its other instances are moved on; `drawn_before` is whether it was
    /// drawn in the frame before.
    fn join(&mut self, rhythm: &Rhythm, frame: u64, drawn_before: bool) {
        let moved = self.ratio_changed == frame;
        let beat = rhythm.0.get();
        if moved && self.ratio_span > 1 {
            let span = match beat {
                Some(beat) if beat.frame == frame => beat.span.max(self.ratio_span),
                _ => self.ratio_span,
            };
            rhythm.0.set(Some(Beat { frame, span }));
            return;
        }
        let Some(beat) = beat else {
            return;
        };
        // Recent: within two intervals. (The timer's next quarter may come a beat after the
        // last one's final move: its clip is empty on its first frame.)
        let recent = u64::from(max_held_frames()) * 2;
        if frame.saturating_sub(beat.frame) > recent || !by_distance() {
            return;
        }
        if moved {
            // Moved on, without a rhythm of its own yet, a few frames or more after its last
            // move (not right after it, and not turned round): drawn like the others.
            if self.ratio_gap >= 2 {
                self.ratio_span = beat.span;
            }
        } else if !drawn_before
            && self.ratio_changed == 0
            && (1..=NEAR_START).contains(&self.seen.2)
        {
            // New on screen, a little way in from the tween's start: it takes over from another
            // instance (a timer's next quarter), which is drawn an interval late. So is this
            // one: from the start to where it is, over the interval.
            self.ratio_from = 0;
            self.ratio_changed = frame;
            self.ratio_span = beat.span;
            self.ratio_gap = beat.span;
        }
    }

    /// For a tween drawn through several frames: the ratio it's at `t` of the way through
    /// `frame` (not rounded to a step), on its way from `ratio_from` to its real ratio.
    fn held_at(&self, frame: u64, t: f32) -> Option<u16> {
        if self.ratio_span <= 1 {
            return None;
        }
        let frames = frame.saturating_sub(self.ratio_changed) as f32 + t;
        let p = (frames / f32::from(self.ratio_span)).clamp(0.0, 1.0);
        let (from, to) = (f32::from(self.ratio_from), f32::from(self.seen.2));
        Some((from + (to - from) * p).round() as u16)
    }
}

/// Per display object: what it looked like on the previous frame. Boxed so it costs one pointer
/// while interpolation is off. Copies (duplicated clips) start without a history.
#[derive(Default)]
pub struct Tracker(Cell<Option<Box<State>>>);

impl Clone for Tracker {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Tracker {
    /// Records `transform` and `ratio` (the object's real ones) and blends the transform with
    /// the previous frame's, while the screen is drawn with interpolation. Returns the previous
    /// frame's matrix and color transform it was blended from, if any (see `replay`).
    pub fn blend(&self, transform: &mut Transform, ratio: u16) -> Option<(Matrix, ColorTransform)> {
        let Some(t) = BLEND.with(Cell::get) else {
            return None;
        };
        let frame = frame();
        let current = (transform.matrix, transform.color_transform, ratio);
        let mut state = self.0.take().unwrap_or_else(|| {
            Box::new(State {
                frame: 0,
                seen: current,
                previous: None,
                seen_filters: None,
                previous_filters: None,
                ratio_from: ratio,
                ratio_changed: 0,
                ratio_span: 1,
                ratio_gap: 0,
            })
        });
        let previous = state.previous_at(frame);
        if state.frame != frame {
            state.note_ratio(frame, ratio);
            state.previous_filters = if state.frame + 1 == frame {
                state.seen_filters.take()
            } else {
                None
            };
            state.seen_filters = None;
        }
        state.frame = frame;
        state.seen = current;
        state.previous = previous;
        self.0.set(Some(state));
        let (matrix, color, _) = previous?;
        blend_look((matrix, color), transform, t);
        Some((matrix, color))
    }

    /// Lean render (see `replay::lean_render`): the object was drawn by the last recorded
    /// walks without being visited (from their recordings, as it was): the last of them in SWF
    /// frame `drawn_frame`, and (`frame_before`) one in the frame before that as well. Leaves
    /// what `blend` would have left by being called in each of them with the look last seen.
    /// (Not the rhythm of a shape tween or the filters: objects with those are always
    /// visited.)
    pub fn catch_up(&self, drawn_frame: u64, frame_before: bool) {
        if BLEND.with(Cell::get).is_none() {
            return;
        }
        let Some(mut state) = self.0.take() else {
            return;
        };
        if state.frame < drawn_frame {
            let continuous = state.frame + 1 == drawn_frame || frame_before;
            state.previous = continuous.then_some(state.seen);
            state.previous_filters = None;
            state.seen_filters = None;
            state.frame = drawn_frame;
        }
        self.0.set(Some(state));
    }

    /// The matrix `matrix` (the object's real one) is drawn with right now, without recording.
    pub fn peek(&self, matrix: Matrix) -> Matrix {
        let Some(t) = BLEND.with(Cell::get) else {
            return matrix;
        };
        let state = self.0.take();
        let previous = state.as_ref().and_then(|s| s.previous_at(frame()));
        self.0.set(state);
        match previous {
            Some((previous, _, _)) => blend_matrix(&previous, &matrix, t),
            None => matrix,
        }
    }

    /// Records `filters` (the object's real ones, after `blend`) and blends them with the
    /// previous frame's, e.g. a glow that pulses. Returns whether any changed.
    pub fn blend_filters(&self, filters: &mut [Filter]) -> bool {
        let Some(t) = BLEND.with(Cell::get) else {
            return false;
        };
        let Some(mut state) = self.0.take() else {
            return false;
        };
        let mut changed = false;
        if state.frame == frame() {
            if state.seen_filters.as_deref() != Some(&*filters) {
                state.seen_filters = Some(filters.to_vec().into_boxed_slice());
            }
            if let Some(previous) = &state.previous_filters
                && previous.len() == filters.len()
            {
                for (filter, previous) in filters.iter_mut().zip(previous.iter()) {
                    if let Some(blended) = blend_filter(previous, filter, t)
                        && blended != *filter
                    {
                        *filter = blended;
                        changed = true;
                    }
                }
            }
        }
        self.0.set(Some(state));
        changed
    }

    /// The ratio a shape tween (`MorphShape`) is drawn at right now, given its real one: AQW's
    /// skill cooldown wipes, for example, are shape tweens stepped once per frame.
    /// `rhythm` is the tween's own (see `Rhythm`). `travel_px` (only asked when it matters) is
    /// how far on screen, in pixels, the point of the tween that moves the most travels over
    /// the whole tween, or 0 for a tween that gets no in-betweens for moving slowly (see
    /// `tween_steps`).
    pub fn ratio(
        &self,
        ratio: u16,
        rhythm: &Rhythm,
        travel_px: impl FnOnce() -> f32,
    ) -> TweenRatio {
        let Some(t) = BLEND.with(Cell::get) else {
            return TweenRatio::Real(ratio);
        };
        let frame = frame();
        let mut state = self.0.take();
        let previous = state.as_ref().and_then(|s| s.previous_at(frame));
        // (Only if `blend` recorded it in this frame, at this ratio.)
        if let Some(s) = state.as_mut() {
            if s.frame == frame && s.seen.2 == ratio {
                s.join(rhythm, frame, previous.is_some());
            }
        }
        // A tween a script moves on every few frames (see `State::note_ratio`), drawn in this
        // frame at its real ratio `ratio`: where it's on its way from, and how far it is.
        let held = state.as_ref().and_then(|s| {
            (s.ratio_span > 1 && s.frame == frame && s.seen.2 == ratio).then(|| {
                let frames = frame.saturating_sub(s.ratio_changed) as f32 + t;
                (s.ratio_from, frames / f32::from(s.ratio_span))
            })
        });
        // Where one that moved on in this frame was drawn before (short of the previous
        // frame's ratio, if it was still on its way there).
        let moved_from = state
            .as_ref()
            .and_then(|s| (s.frame == frame && s.ratio_changed == frame).then_some(s.ratio_from));
        self.0.set(state);
        if let Some((from, p)) = held {
            let delta = (i32::from(ratio) - i32::from(from)).unsigned_abs();
            let steps = if delta == 0 {
                1
            } else {
                tween_steps(delta, travel_px)
            };
            if steps > 1 {
                return held_ratio(from, ratio, p, steps);
            }
            // A single step leaves nothing to draw on the way. As usual then: the previous
            // frame's ratio during the frame it changed in.
            return match previous {
                Some((_, _, previous)) if previous != ratio => TweenRatio::Real(previous),
                _ => TweenRatio::Real(ratio),
            };
        }
        let Some((_, _, previous)) = previous else {
            return TweenRatio::Real(ratio);
        };
        if let Some(from) = moved_from.filter(|from| *from != previous) {
            // It was still on its way to the previous frame's ratio (its rhythm ended): it
            // goes on from where it was drawn. That's no ratio of its own, so it's only ever
            // drawn as an in-between (`TweenRatio::Real` ratios are tessellated for keeps,
            // however slow that is).
            let delta = (i32::from(ratio) - i32::from(from)).unsigned_abs();
            let steps = if delta == 0 {
                1
            } else {
                tween_steps(delta, travel_px)
            };
            return held_ratio(from, ratio, t, steps);
        }
        let delta = (i32::from(ratio) - i32::from(previous)).unsigned_abs();
        if delta == 0 {
            return TweenRatio::Real(ratio);
        }
        let steps = tween_steps(delta, travel_px) as f32;
        blend_ratio(previous, ratio, (t * steps).floor() / steps)
    }
}

/// The ratio of a tween that a script moves on every few frames (AQW's aura timers: every
/// 100 ms, 3 or 4 frames): `p` of the way (in `steps` steps) from `from`, where it was drawn
/// when it moved on, to `to`, its real ratio, over the frames it's given (the longest of its
/// intervals; it may move on again before it gets there, and goes on from where it is then).
/// Blended within the one frame it changed in, it moved for 33 ms and stood for 67. It's drawn
/// that interval late instead, like everything else is one frame late.
///
/// Its ratios on the way are in-betweens (`from` may be one itself), with the real ratio to
/// draw if those are too slow to tessellate.
fn held_ratio(from: u16, to: u16, p: f32, steps: u32) -> TweenRatio {
    let delta = i32::from(to) - i32::from(from);
    // A jump of most of the tween (e.g. it started over) isn't motion.
    if delta == 0 || delta.abs() > 49152 || p >= 1.0 {
        return TweenRatio::Real(to);
    }
    let steps = steps as f32;
    let blended =
        i32::from(from) + (delta as f32 * ((p.max(0.0) * steps).floor() / steps)).round() as i32;
    let stepped = ((blended + 8) / 16 * 16).clamp(0, 65535);
    if (stepped - i32::from(to)) * delta.signum() >= 0 {
        TweenRatio::Real(to)
    } else {
        TweenRatio::Between {
            ratio: stepped as u16,
            previous: to,
        }
    }
}

/// The most steps a tween moves on in during one frame.
const MAX_TWEEN_STEPS: u32 = 8;

/// A tween that changes little per frame still gets in-betweens if its fastest point moves
/// more than this many pixels on screen in the frame: each step is then at most this long.
const TWEEN_STEP_PX: f32 = 1.0;

/// In how many steps a tween moves on during a frame in which its ratio changed by `delta`
/// (see `Tracker::ratio` for `travel_px`). Each new in-between ratio is a new tessellation, so
/// it's a few steps per frame, however often the screen is drawn:
/// - by how much of the tween the frame covers: one per 1024 of ratio, so a fast one, such as
///   the quick cooldown sweep all other skills get when one is used, has several;
/// - by how far it moves on screen: a slow one (a long cooldown's wipe, an aura's timer) on a
///   big window moved 2-3 pixels once per frame, next to motion drawn at the monitor's rate.
///   One that moves a pixel or less per frame is as smooth as it gets and keeps 1 step: it's
///   drawn at the previous frame's ratio until the frame is over.
///
/// `RUFFLE_NO_SLOW_TWEEN_STEPS` (testing) leaves the second rule out, as before it existed.
fn tween_steps(delta: u32, travel_px: impl FnOnce() -> f32) -> u32 {
    let by_ratio = delta.div_ceil(1024).clamp(1, MAX_TWEEN_STEPS);
    if by_ratio == MAX_TWEEN_STEPS || !by_distance() {
        return by_ratio;
    }
    steps_for(by_ratio, delta as f32 / 65535.0 * travel_px())
}

/// Whether slow tweens get in-betweens by how far they move, and held ones are drawn through
/// their interval (see `tween_steps`, `held_ratio`); not with `RUFFLE_NO_SLOW_TWEEN_STEPS`.
fn by_distance() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RUFFLE_NO_SLOW_TWEEN_STEPS").is_none())
}

/// `tween_steps` for a tween that gets `by_ratio` steps for its speed and whose fastest point
/// moves `moved_px` pixels on screen in the frame.
fn steps_for(by_ratio: u32, moved_px: f32) -> u32 {
    // (A size that isn't a number, e.g. from a broken matrix, counts as no movement.)
    let by_distance = (moved_px / TWEEN_STEP_PX)
        .ceil()
        .clamp(1.0, MAX_TWEEN_STEPS as f32);
    by_ratio.max(if by_distance.is_nan() {
        1
    } else {
        by_distance as u32
    })
}

/// How a shape tween is drawn right now (see `Tracker::ratio`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TweenRatio {
    /// At one of its real ratios: the current frame's, or the previous frame's. These are
    /// tessellated without interpolation too, and kept.
    Real(u16),
    /// Between the previous frame's ratio and the current one, which only interpolation draws.
    Between { ratio: u16, previous: u16 },
}

/// Ratios are blended in steps of 16 (of 65535), so nearby draws can share a tessellation.
fn blend_ratio(from: u16, to: u16, t: f32) -> TweenRatio {
    let delta = i32::from(to) - i32::from(from);
    // A jump of most of the tween (e.g. it started over) isn't motion.
    if delta == 0 || delta.abs() > 49152 {
        return TweenRatio::Real(to);
    }
    // Whether `a` is at `b` or past it, going from `from` to `to`.
    let reached = |a: i32, b: i32| (a - b) * delta.signum() >= 0;
    let blended = i32::from(from) + (delta as f32 * t).round() as i32;
    let stepped = (blended + 8) / 16 * 16;
    // Never past the real ratios, and exactly them at the ends.
    if reached(blended, i32::from(to)) || reached(stepped, i32::from(to)) {
        TweenRatio::Real(to)
    } else if reached(i32::from(from), blended) || reached(i32::from(from), stepped) {
        TweenRatio::Real(from)
    } else {
        TweenRatio::Between {
            ratio: stepped as u16,
            previous: from,
        }
    }
}

fn lerp(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}

fn lerp_twips(from: Twips, to: Twips, t: f32) -> Twips {
    Twips::new(from.get() + ((to.get() - from.get()) as f32 * t).round() as i32)
}

fn blend_matrix(from: &Matrix, to: &Matrix, t: f32) -> Matrix {
    if from == to {
        return *to;
    }
    let dx = (to.tx - from.tx).to_pixels();
    let dy = (to.ty - from.ty).to_pixels();
    if dx.hypot(dy) > JUMP_PX {
        return *to;
    }
    let mut matrix = *to;
    matrix.tx = lerp_twips(from.tx, to.tx, t);
    matrix.ty = lerp_twips(from.ty, to.ty, t);
    if (from.a, from.b, from.c, from.d) != (to.a, to.b, to.c, to.d) {
        // Mirrored on one frame and not the other (e.g. a character turning round): no blend.
        let det_from = from.a * from.d - from.b * from.c;
        let det_to = to.a * to.d - to.b * to.c;
        if (det_from < 0.0) != (det_to < 0.0) {
            return *to;
        }
        // Each axis turns and stretches, so rotations keep their size halfway through.
        let (Some((a, b)), Some((c, d))) = (
            blend_axis((from.a, from.b), (to.a, to.b), t),
            blend_axis((from.c, from.d), (to.c, to.d), t),
        ) else {
            return *to;
        };
        (matrix.a, matrix.b, matrix.c, matrix.d) = (a, b, c, d);
    }
    matrix
}

fn blend_axis(from: (f32, f32), to: (f32, f32), t: f32) -> Option<(f32, f32)> {
    let from_len = from.0.hypot(from.1);
    let to_len = to.0.hypot(to.1);
    if from_len < 1e-4 || to_len < 1e-4 {
        return Some((lerp(from.0, to.0, t), lerp(from.1, to.1, t)));
    }
    let from_angle = from.1.atan2(from.0);
    let mut turn = to.1.atan2(to.0) - from_angle;
    if turn > std::f32::consts::PI {
        turn -= std::f32::consts::TAU;
    } else if turn < -std::f32::consts::PI {
        turn += std::f32::consts::TAU;
    }
    if turn.abs() > MAX_TURN {
        return None;
    }
    let angle = from_angle + turn * t;
    let len = lerp(from_len, to_len, t);
    Some((len * angle.cos(), len * angle.sin()))
}

fn lerp16(from: Fixed16, to: Fixed16, t: f32) -> Fixed16 {
    if from == to {
        to
    } else {
        Fixed16::from_f32(lerp(from.to_f32(), to.to_f32(), t))
    }
}

fn lerp8(from: Fixed8, to: Fixed8, t: f32) -> Fixed8 {
    if from == to {
        to
    } else {
        Fixed8::from_f32(lerp(from.to_f32(), to.to_f32(), t))
    }
}

fn lerp_rgba(from: Color, to: Color, t: f32) -> Color {
    let c = |a: u8, b: u8| lerp(f32::from(a), f32::from(b), t).round() as u8;
    Color::from_rgba(u32::from_be_bytes([
        c(from.a, to.a),
        c(from.r, to.r),
        c(from.g, to.g),
        c(from.b, to.b),
    ]))
}

/// The filter between `from` and `to`, if they're the same kind with the same options.
fn blend_filter(from: &Filter, to: &Filter, t: f32) -> Option<Filter> {
    Some(match (from, to) {
        (Filter::GlowFilter(from), Filter::GlowFilter(to)) if from.flags == to.flags => {
            Filter::GlowFilter(swf::GlowFilter {
                color: lerp_rgba(from.color, to.color, t),
                blur_x: lerp16(from.blur_x, to.blur_x, t),
                blur_y: lerp16(from.blur_y, to.blur_y, t),
                strength: lerp8(from.strength, to.strength, t),
                flags: to.flags,
            })
        }
        (Filter::DropShadowFilter(from), Filter::DropShadowFilter(to))
            if from.flags == to.flags =>
        {
            Filter::DropShadowFilter(swf::DropShadowFilter {
                color: lerp_rgba(from.color, to.color, t),
                blur_x: lerp16(from.blur_x, to.blur_x, t),
                blur_y: lerp16(from.blur_y, to.blur_y, t),
                angle: lerp16(from.angle, to.angle, t),
                distance: lerp16(from.distance, to.distance, t),
                strength: lerp8(from.strength, to.strength, t),
                flags: to.flags,
            })
        }
        (Filter::BlurFilter(from), Filter::BlurFilter(to)) if from.flags == to.flags => {
            Filter::BlurFilter(swf::BlurFilter {
                blur_x: lerp16(from.blur_x, to.blur_x, t),
                blur_y: lerp16(from.blur_y, to.blur_y, t),
                flags: to.flags,
            })
        }
        (Filter::BevelFilter(from), Filter::BevelFilter(to)) if from.flags == to.flags => {
            Filter::BevelFilter(swf::BevelFilter {
                shadow_color: lerp_rgba(from.shadow_color, to.shadow_color, t),
                highlight_color: lerp_rgba(from.highlight_color, to.highlight_color, t),
                blur_x: lerp16(from.blur_x, to.blur_x, t),
                blur_y: lerp16(from.blur_y, to.blur_y, t),
                angle: lerp16(from.angle, to.angle, t),
                distance: lerp16(from.distance, to.distance, t),
                strength: lerp8(from.strength, to.strength, t),
                flags: to.flags,
            })
        }
        _ => return None,
    })
}

fn blend_color(from: &ColorTransform, to: &ColorTransform, t: f32) -> ColorTransform {
    if from == to {
        return *to;
    }
    let mul = |a: Fixed8, b: Fixed8| Fixed8::from_f32(lerp(a.to_f32(), b.to_f32(), t));
    let add = |a: i16, b: i16| lerp(a as f32, b as f32, t).round() as i16;
    ColorTransform {
        r_multiply: mul(from.r_multiply, to.r_multiply),
        g_multiply: mul(from.g_multiply, to.g_multiply),
        b_multiply: mul(from.b_multiply, to.b_multiply),
        a_multiply: mul(from.a_multiply, to.a_multiply),
        r_add: add(from.r_add, to.r_add),
        g_add: add(from.g_add, to.g_add),
        b_add: add(from.b_add, to.b_add),
        a_add: add(from.a_add, to.a_add),
    }
}

#[cfg(test)]
mod tests {
    /// Lean render: an object that stands still isn't visited by the walks that draw it from
    /// their recordings; when it moves, `catch_up` must leave its tracker as if `blend` had
    /// been called in every one of those walks.
    #[test]
    fn catching_up_is_blending_in_every_walk() {
        use crate::prelude::*;
        use ruffle_render::transform::Transform;
        let at = |x: f64| Transform {
            matrix: Matrix::translate(Twips::from_pixels(x), Twips::ZERO),
            ..Default::default()
        };
        // Bit i: whether SWF frame i + 2 has a recorded walk (frame 1 always has one).
        for walks in 0u32..256 {
            for twice in [false, true] {
                super::BLEND.with(|b| b.set(Some(0.25)));
                let (visited, skipped) = (super::Tracker::default(), super::Tracker::default());
                super::FRAME.with(|f| f.set(1));
                visited.blend(&mut at(10.0), 0);
                skipped.blend(&mut at(10.0), 0);
                let mut walked = vec![1u64];
                for i in 0..8 {
                    if walks & (1 << i) != 0 {
                        let frame = i as u64 + 2;
                        super::FRAME.with(|f| f.set(frame));
                        visited.blend(&mut at(10.0), 0);
                        if twice {
                            visited.blend(&mut at(10.0), 0);
                        }
                        walked.push(frame);
                    }
                }
                // It moves in frame 10, and is walked.
                super::FRAME.with(|f| f.set(10));
                let last = *walked.last().unwrap();
                skipped.catch_up(last, walked.contains(&(last - 1)));
                let (mut a, mut b) = (at(40.0), at(40.0));
                let (from_a, from_b) = (visited.blend(&mut a, 0), skipped.blend(&mut b, 0));
                assert_eq!(from_a, from_b, "walks {walks:#b}");
                assert!(a.same(&b), "walks {walks:#b}");
                // And once more in the next frame.
                super::FRAME.with(|f| f.set(11));
                let (mut a, mut b) = (at(70.0), at(70.0));
                assert_eq!(visited.blend(&mut a, 0), skipped.blend(&mut b, 0));
                assert!(a.same(&b));
                super::BLEND.with(|b| b.set(None));
            }
        }
    }

    use super::*;

    #[test]
    fn rotation_keeps_its_size() {
        let from = Matrix::IDENTITY;
        let to = Matrix::rotate(1.0);
        let half = blend_matrix(&from, &to, 0.5);
        assert!((half.a.hypot(half.b) - 1.0).abs() < 1e-5);
        assert!((half.b.atan2(half.a) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn jumps_and_flips_are_not_blended() {
        let from = Matrix::translate(Twips::ZERO, Twips::ZERO);
        let far = Matrix::translate(Twips::from_pixels(500.0), Twips::ZERO);
        assert_eq!(blend_matrix(&from, &far, 0.5), far);
        let flipped = Matrix::scale(-1.0, 1.0);
        assert_eq!(blend_matrix(&from, &flipped, 0.5), flipped);
        let near = Matrix::translate(Twips::from_pixels(10.0), Twips::ZERO);
        assert_eq!(blend_matrix(&from, &near, 0.5).tx, Twips::from_pixels(5.0));
    }

    #[test]
    fn glows_blend_and_other_kinds_do_not() {
        let glow = |blur: f32, alpha: u8| {
            Filter::GlowFilter(swf::GlowFilter {
                color: Color::from_rgba(u32::from_be_bytes([alpha, 255, 0, 0])),
                blur_x: Fixed16::from_f32(blur),
                blur_y: Fixed16::from_f32(blur),
                strength: Fixed8::from_f32(1.0),
                flags: swf::GlowFilterFlags::empty(),
            })
        };
        assert_eq!(
            blend_filter(&glow(2.0, 0), &glow(6.0, 200), 0.5),
            Some(glow(4.0, 100))
        );
        let blur = Filter::BlurFilter(swf::BlurFilter {
            blur_x: Fixed16::from_f32(4.0),
            blur_y: Fixed16::from_f32(4.0),
            flags: swf::BlurFilterFlags::empty(),
        });
        assert_eq!(blend_filter(&blur, &glow(2.0, 0), 0.5), None);
    }

    #[test]
    fn ratios_blend_in_steps_and_skip_restarts() {
        let between = |ratio, previous| TweenRatio::Between { ratio, previous };
        assert_eq!(blend_ratio(736, 1473, 0.5), between(1104, 736));
        assert_eq!(blend_ratio(1473, 736, 0.5), between(1104, 1473));
        assert_eq!(blend_ratio(736, 1473, 1.0), TweenRatio::Real(1473));
        assert_eq!(blend_ratio(65535, 0, 0.5), TweenRatio::Real(0));
        assert_eq!(blend_ratio(500, 500, 0.3), TweenRatio::Real(500));
        // At the start, the previous frame's own ratio, not one rounded to a step near it.
        assert_eq!(blend_ratio(1473, 2209, 0.0), TweenRatio::Real(1473));
        assert_eq!(blend_ratio(1473, 2209, 0.001), TweenRatio::Real(1473));
        // Rounded to a step at or past either end: that end.
        assert_eq!(blend_ratio(1473, 1500, 0.9), TweenRatio::Real(1500));
    }

    #[test]
    fn slow_tweens_step_by_how_far_they_move() {
        // A pixel or less in the frame: no in-betweens, however slow or fast the tween.
        assert_eq!(steps_for(1, 0.0), 1);
        assert_eq!(steps_for(1, 1.0), 1);
        // More: steps of at most a pixel, up to 8.
        assert_eq!(steps_for(1, 1.2), 2);
        assert_eq!(steps_for(1, 2.4), 3);
        assert_eq!(steps_for(1, 300.0), 8);
        // Never fewer than its speed alone gives (e.g. a tween that only changes colour).
        assert_eq!(steps_for(3, 0.0), 3);
        assert_eq!(steps_for(3, 5.5), 6);
        assert_eq!(steps_for(2, f32::NAN), 2);
        // A 180 px wipe over 150 frames (a 5 s cooldown) moves 1.2 px per frame.
        assert_eq!(steps_for(1, 437.0 / 65535.0 * 180.0), 2);
        // The fastest tweens don't ask for the size at all.
        assert_eq!(tween_steps(8 * 1024, || panic!("not needed")), 8);
    }

    /// The ratio `tracker`'s tween is drawn at `t` of the way through the current frame, where
    /// its real ratio is `ratio` (drawn first, as a walk does).
    fn drawn(tracker: &Tracker, ratio: u16, t: f32) -> u16 {
        drawn_among(tracker, &Rhythm::default(), ratio, t, 0.0)
    }

    /// `drawn` for an instance of a tween with the given rhythm, which travels `travel_px` on
    /// screen (a long way: 8 steps per move, so it's seen to move within a frame).
    fn drawn_among(tracker: &Tracker, rhythm: &Rhythm, ratio: u16, t: f32, travel_px: f32) -> u16 {
        BLEND.with(|b| b.set(Some(t)));
        let mut transform = Transform::default();
        tracker.blend(&mut transform, ratio);
        let drawn = match tracker.ratio(ratio, rhythm, || travel_px) {
            TweenRatio::Real(ratio) | TweenRatio::Between { ratio, .. } => ratio,
        };
        BLEND.with(|b| b.set(None));
        drawn
    }

    /// Runs one frame for `tracker`'s tween at `ratio`: where it's drawn as the frame starts
    /// and as it ends.
    fn frame_of(tracker: &Tracker, rhythm: &Rhythm, ratio: u16) -> (u16, u16) {
        new_frame();
        let start = drawn_among(tracker, rhythm, ratio, 0.0, 10_000.0);
        (start, drawn_among(tracker, rhythm, ratio, 0.99, 10_000.0))
    }

    #[test]
    fn a_tween_on_an_uneven_rhythm_never_stands_still() {
        // An aura's timer at 30 fps: moved on every 3rd or 4th frame, by 745 for each frame
        // since the last time.
        let (tracker, rhythm) = (Tracker::default(), Rhythm::default());
        let mut ratio = 10_000u16;
        let mut since = 3;
        let mut ends = Vec::new();
        for frames in [3u16, 3, 4, 3, 4, 4, 3, 3, 3, 4, 3, 4, 4, 3] {
            ratio += since * 745;
            ends.extend((0..frames).map(|_| frame_of(&tracker, &rhythm, ratio).1));
            since = frames;
        }
        // Its rhythm is known from its third move (frame 10), and it has settled a few moves
        // later: from then on it moves about 745 in every frame, whichever way the intervals
        // fall. (Drawn over the last interval, it stood still in the 4th frame of every
        // interval that followed one of 3.)
        let steps: Vec<i32> = ends[20..]
            .windows(2)
            .map(|pair| i32::from(pair[1]) - i32::from(pair[0]))
            .collect();
        assert!(
            steps.iter().all(|step| (600..=890).contains(step)),
            "{steps:?} {ends:?}"
        );
        // And it's never ahead of its real ratio.
        assert!(ends.iter().all(|end| *end <= ratio));
    }

    #[test]
    fn a_tween_keeps_its_rhythm_over_a_beat_it_sat_out() {
        // A slow timer: one frame of its clip (745) per beat of 3 frames, none on one beat.
        let (tracker, rhythm) = (Tracker::default(), Rhythm::default());
        let mut ratio = 10_000u16;
        let mut last = (0, 0);
        for frames in [3, 3, 3, 3, 6] {
            ratio += 745;
            for _ in 0..frames {
                last = frame_of(&tracker, &rhythm, ratio);
            }
        }
        assert_eq!(last, (ratio, ratio));
        // The move after the long interval is still drawn through its 3 frames.
        ratio += 745;
        let first = frame_of(&tracker, &rhythm, ratio);
        assert!(
            first.0 <= ratio - 745 + 16 && first.1 < ratio - 400,
            "{first:?}"
        );
        frame_of(&tracker, &rhythm, ratio);
        frame_of(&tracker, &rhythm, ratio);
        assert_eq!(frame_of(&tracker, &rhythm, ratio), (ratio, ratio));
        // After a rest of more than two beats it's a new start: within the frame, as ever.
        for _ in 0..6 {
            frame_of(&tracker, &rhythm, ratio);
        }
        ratio += 745;
        frame_of(&tracker, &rhythm, ratio);
        assert_eq!(frame_of(&tracker, &rhythm, ratio).0, ratio);
    }

    #[test]
    fn a_held_tween_that_turns_round_jumps() {
        let (tracker, rhythm) = (Tracker::default(), Rhythm::default());
        let mut ratio = 10_000u16;
        for _ in 0..5 {
            ratio += 2235;
            for _ in 0..3 {
                frame_of(&tracker, &rhythm, ratio);
            }
        }
        // (It's held: a move forward is nowhere near done after one frame.)
        ratio += 2235;
        assert!(frame_of(&tracker, &rhythm, ratio).1 < ratio - 1000);
        frame_of(&tracker, &rhythm, ratio);
        frame_of(&tracker, &rhythm, ratio);
        // The timer starts over: there by the end of the frame, like any tween's frame.
        ratio = 3_000;
        frame_of(&tracker, &rhythm, ratio);
        assert_eq!(frame_of(&tracker, &rhythm, ratio), (ratio, ratio));
    }

    #[test]
    fn only_real_ratios_are_drawn_as_real() {
        // A tween on an uneven rhythm (drawn over 4 frames, moved on after 3) that then
        // moves on in the very next frame, and one that turns round: wherever it had got to,
        // it's drawn there as an in-between, with its real ratio to fall back on.
        for turn in [false, true] {
            let (tracker, rhythm) = (Tracker::default(), Rhythm::default());
            let mut ratio = 10_000u16;
            let mut real = vec![ratio];
            for frames in [3, 3, 4, 3, 3] {
                ratio += 2235;
                real.push(ratio);
                for _ in 0..frames {
                    frame_of(&tracker, &rhythm, ratio);
                }
            }
            ratio = if turn { ratio - 900 } else { ratio + 745 };
            real.push(ratio);
            if !turn {
                // (Right after its last move: it had three more frames to go.)
                ratio += 2235;
                real.push(ratio);
                frame_of(&tracker, &rhythm, ratio - 2235);
            }
            new_frame();
            for t in [0.0, 0.3, 0.7, 0.99] {
                BLEND.with(|b| b.set(Some(t)));
                tracker.blend(&mut Transform::default(), ratio);
                match tracker.ratio(ratio, &rhythm, || 10_000.0) {
                    TweenRatio::Real(drawn) => assert!(real.contains(&drawn), "{drawn}"),
                    TweenRatio::Between { previous, .. } => assert_eq!(previous, ratio),
                }
                BLEND.with(|b| b.set(None));
            }
            // And it's at its real ratio once the frame is over.
            assert_eq!(frame_of(&tracker, &rhythm, ratio), (ratio, ratio));
        }
    }

    #[test]
    fn a_new_instance_falls_in_with_the_others() {
        // One instance of the tween moves on every 3 frames (a timer's first quarter).
        let rhythm = Rhythm::default();
        let first = Tracker::default();
        let mut ratio = 40_000u16;
        for _ in 0..5 {
            ratio += 2235;
            for _ in 0..3 {
                frame_of(&first, &rhythm, ratio);
            }
        }
        // On the next beat another takes over, a little way in: drawn from the start to
        // there over the interval, and its own first move goes on from there.
        let second = Tracker::default();
        let shown: Vec<(u16, u16)> = (0..3).map(|_| frame_of(&second, &rhythm, 1490)).collect();
        assert!(shown[0].0 <= 16 && shown[0].1 < 700, "{shown:?}");
        assert!(
            shown
                .windows(2)
                .all(|pair| pair[0].1 <= pair[1].0 && pair[1].0 < pair[1].1)
        );
        let moved = frame_of(&second, &rhythm, 1490 + 2235);
        assert!(
            moved.0.abs_diff(1490) <= 16 && moved.1 < 1490 + 1000,
            "{moved:?}"
        );

        // One that comes on screen far from the start is drawn where it is; its first move a
        // few frames later (the others' rhythm still fresh) is drawn through the interval.
        let third = Tracker::default();
        assert_eq!(frame_of(&third, &rhythm, 30_000), (30_000, 30_000));
        frame_of(&second, &rhythm, 1490 + 2235);
        frame_of(&third, &rhythm, 30_000);
        frame_of(&second, &rhythm, 1490 + 2235);
        frame_of(&third, &rhythm, 30_000);
        frame_of(&second, &rhythm, 1490 + 2 * 2235);
        let moved = drawn_among(&third, &rhythm, 32_235, 0.99, 10_000.0);
        assert!(moved < 31_200, "{moved}");

        // With nobody moving for a while, a new instance is drawn where it is.
        for _ in 0..10 {
            new_frame();
        }
        let fourth = Tracker::default();
        assert_eq!(frame_of(&fourth, &rhythm, 1490), (1490, 1490));
    }

    #[test]
    fn a_tween_moved_on_every_few_frames_is_drawn_through_the_interval() {
        // An aura's timer: 3 frames of its clip (3 x 745) every 3 game frames.
        let tracker = Tracker::default();
        let mut ratio = 10_000u16;
        let mut shown = Vec::new();
        for frame in 0..18 {
            new_frame();
            if frame % 3 == 0 && frame > 0 {
                ratio += 2235;
            }
            shown.push((drawn(&tracker, ratio, 0.0), drawn(&tracker, ratio, 0.5)));
        }
        // Once its rhythm is known (the third move, frame 9), it moves in every frame and
        // never backwards, one interval behind its real ratio.
        let held = &shown[9..];
        assert!(
            held.windows(2)
                .all(|pair| pair[0].1 <= pair[1].0 && pair[1].0 <= pair[1].1)
        );
        let moves = held.windows(2).filter(|pair| pair[1].0 > pair[0].0).count();
        assert!(moves >= 7, "{held:?}");
        // Right after a move it's still near where it was, and reaches it by the next move.
        assert!(shown[12].0 < 10_000 + 3 * 2235 + 800, "{shown:?}");
        assert!(shown[15].0.abs_diff(10_000 + 4 * 2235) <= 8, "{shown:?}");
        // Before that (the first two moves): within the frame it changed in, as ever.
        assert_eq!(shown[4].0, 10_000 + 2235);

        // One that stops moving ends at its real ratio and stays there.
        for _ in 0..6 {
            new_frame();
            shown.push((drawn(&tracker, ratio, 0.0), drawn(&tracker, ratio, 0.5)));
        }
        assert_eq!(shown.last().copied(), Some((ratio, ratio)));

        // A tween moved on every frame (a timeline's) is never held.
        let tracker = Tracker::default();
        for frame in 0..8u16 {
            new_frame();
            let ratio = 20_000 + frame * 745;
            let at_start = drawn(&tracker, ratio, 0.0);
            if frame > 0 {
                assert_eq!(at_start, ratio - 745);
            }
        }
    }

    #[test]
    fn held_ratios_step_to_the_real_one() {
        let between = |ratio| TweenRatio::Between {
            ratio,
            previous: 3235,
        };
        // (1000 -> 3235 in 3 steps; rounded to 16.)
        assert_eq!(held_ratio(1000, 3235, 0.0, 3), between(1008));
        assert_eq!(held_ratio(1000, 3235, 0.4, 3), between(1744));
        assert_eq!(held_ratio(1000, 3235, 0.7, 3), between(2496));
        assert_eq!(held_ratio(1000, 3235, 1.0, 3), TweenRatio::Real(3235));
        assert_eq!(held_ratio(1000, 3235, 1.7, 3), TweenRatio::Real(3235));
        assert_eq!(held_ratio(65000, 300, 0.5, 3), TweenRatio::Real(300));
    }
}
