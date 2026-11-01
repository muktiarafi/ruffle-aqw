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
}

/// Turu: one player's frame count and blend while another player's are the thread's (see
/// `crate::PlayerThreadState`).
pub struct ThreadState {
    frame: u64,
    frame_finished: Option<Instant>,
    blend: Option<f32>,
}

impl Default for ThreadState {
    fn default() -> Self {
        Self {
            frame: 1,
            frame_finished: None,
            blend: None,
        }
    }
}

/// Turu: swaps `state` with the thread's.
pub(crate) fn swap(state: &mut ThreadState) {
    state.frame = FRAME.replace(state.frame);
    state.frame_finished = FRAME_FINISHED.replace(state.frame_finished);
    state.blend = BLEND.replace(state.blend);
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
    let blend = if enabled() && frame_ms > 0.0 {
        FRAME_FINISHED
            .with(Cell::get)
            .map(|started| (started.elapsed().as_secs_f64() * 1000.0 / frame_ms).clamp(0.0, 1.0) as f32)
    } else {
        None
    };
    BLEND.with(|b| b.set(blend));
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
}

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
            })
        });
        let previous = state.previous_at(frame);
        if state.frame != frame {
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
    pub fn ratio(&self, ratio: u16) -> u16 {
        let Some(t) = BLEND.with(Cell::get) else {
            return ratio;
        };
        let state = self.0.take();
        let previous = state.as_ref().and_then(|s| s.previous_at(frame()));
        self.0.set(state);
        let Some((_, _, previous)) = previous else {
            return ratio;
        };
        // Each new in-between ratio is a new tessellation, so a tween moves on in a few steps
        // per frame (however often the screen is drawn): more for a fast one, such as the quick
        // cooldown sweep all other skills get when one is used, up to 8.
        let delta = (i32::from(ratio) - i32::from(previous)).unsigned_abs();
        let steps = delta.div_ceil(1024).clamp(1, 8) as f32;
        blend_ratio(previous, ratio, (t * steps).floor() / steps)
    }
}

/// Ratios are blended in steps of 16 (of 65535), so nearby draws can share a tessellation.
fn blend_ratio(from: u16, to: u16, t: f32) -> u16 {
    let delta = i32::from(to) - i32::from(from);
    // A jump of most of the tween (e.g. it started over) isn't motion.
    if delta == 0 || delta.abs() > 49152 {
        return to;
    }
    let blended = i32::from(from) + (delta as f32 * t).round() as i32;
    if blended == i32::from(to) {
        return to;
    }
    let stepped = (blended + 8) / 16 * 16;
    // Never past the real ratio, and exactly it at the end.
    let stepped = if delta > 0 {
        stepped.min(i32::from(to))
    } else {
        stepped.max(i32::from(to))
    };
    stepped.clamp(0, 65535) as u16
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
        assert_eq!(blend_filter(&glow(2.0, 0), &glow(6.0, 200), 0.5), Some(glow(4.0, 100)));
        let blur = Filter::BlurFilter(swf::BlurFilter {
            blur_x: Fixed16::from_f32(4.0),
            blur_y: Fixed16::from_f32(4.0),
            flags: swf::BlurFilterFlags::empty(),
        });
        assert_eq!(blend_filter(&blur, &glow(2.0, 0), 0.5), None);
    }

    #[test]
    fn ratios_blend_in_steps_and_skip_restarts() {
        assert_eq!(blend_ratio(736, 1473, 0.5), 1104);
        assert_eq!(blend_ratio(736, 1473, 1.0), 1473);
        assert_eq!(blend_ratio(65535, 0, 0.5), 0);
        assert_eq!(blend_ratio(500, 500, 0.3), 500);
    }
}
