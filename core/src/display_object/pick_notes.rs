//! AQW build: notes that let the mouse pick skip branches the pointer cannot hit.
//!
//! OFF unless `RUFFLE_PICK_BOUNDS` is set (a `pick-bounds` file next to the exe).
//!
//! `MovieClip::mouse_pick_avm2` visits every object under the stage: with the pointer over a
//! crowded room that is tens of thousands of objects per pick, for a result that depends on
//! the handful of objects the pointer is over. Upstream does not skip a clip the pointer is
//! outside of because "a child button can have an invisible hit area outside the parent's
//! bounds". So the skip is limited to branches where that cannot happen.
//!
//! A clip is PLAIN when everything a pick could hit below it lies inside bounds that can be
//! worked out from the tree: it is an AVM2 clip without a scroll rect or a `hitArea`, and
//! each child is a plain clip, or a shape, morph shape, bitmap or static text (their hit
//! tests all start with "is the point in my world bounds", as does a clip's own drawing),
//! none of them with a scroll rect. Buttons (their hit state is not part of their bounds),
//! text fields, videos, loaders, AVM1 content and anything else make a clip not plain: such
//! a clip is entered as before, and its own plain children are skipped one by one.
//!
//! For a plain clip the note holds bounds in the clip's own space that contain everything
//! below it, rounded outwards at every level. The pick skips a plain child when the pointer
//! is outside those bounds mapped to the stage, by a margin wider than the rounding the
//! exact hit tests can differ by (they multiply matrices down from the stage in `f32` and
//! round each translation to a twip).
//!
//! A note is good until something below the clip changes: a matrix, a list of children, a
//! shape, a drawing, a morph ratio, a scroll rect, a `hitArea`. Each of these marks the clip
//! and its ancestors dirty (`mark_from`; a dirty clip's ancestors are dirty already, so the
//! walk up stops at the first dirty clip). Notes are rebuilt when a pick asks for them.
//!
//! `RUFFLE_PICK_VERIFY` (testing) runs the full pick for every skipped branch as well, and
//! rebuilds every note it uses from nothing: a skipped branch that would have been hit and a
//! note that no longer covers its branch are logged (`RUFFLE_PICK_VERIFY=panic`: a panic,
//! for test runs). `RUFFLE_PICK_FAULT` (testing the test) skips every plain branch wherever
//! the pointer is, so that verify mode has something to report.

use super::{BoundsMode, DisplayObject, MovieClip, TDisplayObject, TDisplayObjectContainer};
use ruffle_render::matrix::Matrix;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use swf::{Point, Rectangle, Twips};

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var_os("RUFFLE_PICK_BOUNDS").is_some();
        if enabled {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "the mouse pick skips plain branches the pointer is outside of (RUFFLE_PICK_BOUNDS)"
            );
        }
        enabled
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    Off,
    Log,
    Panic,
}

pub fn verify() -> Verify {
    static VERIFY: OnceLock<Verify> = OnceLock::new();
    *VERIFY.get_or_init(|| match std::env::var("RUFFLE_PICK_VERIFY") {
        Ok(value) if value == "panic" => Verify::Panic,
        Ok(_) => Verify::Log,
        Err(_) => Verify::Off,
    })
}

/// `RUFFLE_PICK_FAULT` (testing the test): skip every plain branch. Only ever with
/// `RUFFLE_PICK_VERIFY`, to see that it reports the wrong skips.
fn fault() -> bool {
    static FAULT: OnceLock<bool> = OnceLock::new();
    *FAULT.get_or_init(|| std::env::var_os("RUFFLE_PICK_FAULT").is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Something below changed since the note was made (or it never was).
    Dirty,
    Plain,
    NotPlain,
}

/// What a clip remembers about its branch for the pick.
#[derive(Clone, Copy, Debug)]
pub struct Note {
    state: State,
    /// How many levels of clips the branch has below this one.
    levels: u16,
    /// The largest coordinate (twips, in this clip's space) any step of mapping the branch's
    /// content into this clip's space goes through, before translations cancel: what the
    /// `f32` error of the exact hit tests is proportional to.
    reach: f32,
    /// Everything below, in this clip's space.
    bounds: Rectangle<Twips>,
}

impl Default for Note {
    fn default() -> Self {
        Self {
            state: State::Dirty,
            levels: 0,
            reach: 0.0,
            bounds: Rectangle::INVALID,
        }
    }
}

/// Something about `node` changed that the notes of the clips at and above it depend on.
pub fn mark_from<'gc>(mut node: Option<DisplayObject<'gc>>) {
    if !enabled() {
        return;
    }
    while let Some(object) = node {
        if let DisplayObject::MovieClip(clip) = object {
            // A dirty clip's ancestors are dirty already.
            if clip.pick_note().state == State::Dirty {
                break;
            }
            clip.set_pick_note(Note::default());
        }
        node = object.parent();
    }
}

static SKIPPED: AtomicU64 = AtomicU64::new(0);
static ENTERED_PLAIN: AtomicU64 = AtomicU64::new(0);
static ENTERED_OTHER: AtomicU64 = AtomicU64::new(0);
static REBUILT: AtomicU64 = AtomicU64::new(0);
static STALE: AtomicU64 = AtomicU64::new(0);
static MISMATCHES: AtomicU64 = AtomicU64::new(0);

/// (branches skipped, plain branches entered, other clips entered, notes rebuilt, notes
/// found out of date, skipped branches that would have been hit), since the last call.
pub fn take_counters() -> [u64; 6] {
    [
        SKIPPED.swap(0, Ordering::Relaxed),
        ENTERED_PLAIN.swap(0, Ordering::Relaxed),
        ENTERED_OTHER.swap(0, Ordering::Relaxed),
        REBUILT.swap(0, Ordering::Relaxed),
        STALE.swap(0, Ordering::Relaxed),
        MISMATCHES.swap(0, Ordering::Relaxed),
    ]
}

/// The pick skipped a branch whose full pick is not a miss (verify mode).
pub fn mismatch(child: DisplayObject<'_>) {
    MISMATCHES.fetch_add(1, Ordering::Relaxed);
    let message = format!(
        "PICK MISMATCH: a skipped branch would have been hit: {}",
        path(child)
    );
    tracing::error!(target: "ruffle_aqw_perf", "{message}");
    if verify() == Verify::Panic {
        panic!("{message}");
    }
}

fn path(object: DisplayObject<'_>) -> String {
    let mut parts = Vec::new();
    let mut node = Some(object);
    while let Some(o) = node {
        if parts.len() == 8 {
            parts.push("…".to_string());
            break;
        }
        let name = o.name().map(|n| n.to_string()).unwrap_or_default();
        parts.push(format!("{}#{} '{name}'", kind(o), o.id()));
        node = o.parent();
    }
    parts.join(" < ")
}

fn kind(object: DisplayObject<'_>) -> &'static str {
    match object {
        DisplayObject::Stage(_) => "Stage",
        DisplayObject::Bitmap(_) => "Bitmap",
        DisplayObject::Avm1Button(_) => "Avm1Button",
        DisplayObject::Avm2Button(_) => "Avm2Button",
        DisplayObject::EditText(_) => "EditText",
        DisplayObject::TextLine(_) => "TextLine",
        DisplayObject::Graphic(_) => "Graphic",
        DisplayObject::MorphShape(_) => "MorphShape",
        DisplayObject::MovieClip(_) => "MovieClip",
        DisplayObject::Text(_) => "Text",
        DisplayObject::Video(_) => "Video",
        DisplayObject::LoaderDisplay(_) => "LoaderDisplay",
    }
}

/// Coordinates beyond this (twips) are left to the exact tests.
const FAR: f64 = 1.0e8;

/// The rectangle that `rect` covers after `matrix`, rounded outwards and a twip wider;
/// `None` if a number in it is not finite or too far out to trust.
fn map_out(matrix: &Matrix, rect: &Rectangle<Twips>) -> Option<Rectangle<Twips>> {
    let (a, b, c, d) = (
        f64::from(matrix.a),
        f64::from(matrix.b),
        f64::from(matrix.c),
        f64::from(matrix.d),
    );
    let (tx, ty) = (f64::from(matrix.tx.get()), f64::from(matrix.ty.get()));
    let (mut x_min, mut y_min) = (f64::INFINITY, f64::INFINITY);
    let (mut x_max, mut y_max) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in [
        (rect.x_min, rect.y_min),
        (rect.x_min, rect.y_max),
        (rect.x_max, rect.y_min),
        (rect.x_max, rect.y_max),
    ] {
        let (x, y) = (f64::from(x.get()), f64::from(y.get()));
        let (out_x, out_y) = (a * x + c * y + tx, b * x + d * y + ty);
        if !(out_x.is_finite() && out_y.is_finite()) || out_x.abs() > FAR || out_y.abs() > FAR {
            return None;
        }
        x_min = x_min.min(out_x);
        x_max = x_max.max(out_x);
        y_min = y_min.min(out_y);
        y_max = y_max.max(out_y);
    }
    Some(Rectangle {
        x_min: Twips::new(x_min.floor() as i32 - 1),
        x_max: Twips::new(x_max.ceil() as i32 + 1),
        y_min: Twips::new(y_min.floor() as i32 - 1),
        y_max: Twips::new(y_max.ceil() as i32 + 1),
    })
}

/// How much `matrix` can stretch a length.
fn stretch(matrix: &Matrix) -> f64 {
    let (a, b, c, d) = (
        f64::from(matrix.a),
        f64::from(matrix.b),
        f64::from(matrix.c),
        f64::from(matrix.d),
    );
    (a.abs() + c.abs()).max(b.abs() + d.abs())
}

fn largest(rect: &Rectangle<Twips>) -> f64 {
    [rect.x_min, rect.x_max, rect.y_min, rect.y_max]
        .iter()
        .map(|v| f64::from(v.get()).abs())
        .fold(0.0, f64::max)
}

struct Summary {
    bounds: Rectangle<Twips>,
    levels: u16,
    reach: f64,
}

/// What is below `clip`, from its note (rebuilt if it is dirty); `None` if it is not plain.
/// With `fresh`, worked out from nothing and no note is read or written.
fn summary<'gc>(clip: MovieClip<'gc>, fresh: bool) -> Option<Summary> {
    if !fresh {
        let note = clip.pick_note();
        match note.state {
            State::Plain => {
                return Some(Summary {
                    bounds: note.bounds,
                    levels: note.levels,
                    reach: f64::from(note.reach),
                });
            }
            State::NotPlain => return None,
            State::Dirty => {}
        }
        REBUILT.fetch_add(1, Ordering::Relaxed);
    }
    let found = build(clip, fresh);
    if !fresh {
        clip.set_pick_note(match &found {
            Some(summary) => Note {
                state: State::Plain,
                levels: summary.levels,
                // (Rounded up: `as f32` rounds to nearest.)
                reach: (summary.reach * 1.000_001) as f32,
                bounds: summary.bounds,
            },
            None => Note {
                state: State::NotPlain,
                ..Default::default()
            },
        });
    }
    found
}

fn build<'gc>(clip: MovieClip<'gc>, fresh: bool) -> Option<Summary> {
    if !clip.pick_plain_self() {
        return None;
    }
    // The clip's own drawing.
    let mut bounds = clip.self_bounds(BoundsMode::Engine);
    let mut levels = 0u16;
    let mut reach = if bounds.is_valid() {
        largest(&bounds)
    } else {
        0.0
    };
    for child in clip.iter_render_list() {
        // An object with a scroll rect has that rectangle as its bounds, and what is below
        // it is shifted by it: left to the exact tests.
        if child.has_scroll_rect() || child.scroll_rect().is_some() {
            return None;
        }
        // World matrices follow the parent links: a child listed here that hangs elsewhere
        // is not where this list says.
        if !child
            .parent()
            .is_some_and(|parent| DisplayObject::ptr_eq(parent, clip.into()))
        {
            return None;
        }
        let (child_bounds, child_levels, child_reach) = match child {
            DisplayObject::MovieClip(child_clip) => {
                let below = summary(child_clip, fresh)?;
                (below.bounds, below.levels.saturating_add(1), below.reach)
            }
            DisplayObject::Graphic(_)
            | DisplayObject::MorphShape(_)
            | DisplayObject::Bitmap(_)
            | DisplayObject::Text(_) => {
                let own = child.self_bounds(BoundsMode::Engine);
                let reach = if own.is_valid() { largest(&own) } else { 0.0 };
                (own, 1, reach)
            }
            _ => return None,
        };
        if !child_bounds.is_valid() {
            continue;
        }
        let matrix = child.base().matrix();
        let placed = map_out(&matrix, &child_bounds)?;
        bounds = bounds.union(&placed);
        levels = levels.max(child_levels);
        reach = reach.max(
            stretch(&matrix) * child_reach
                + f64::from(matrix.tx.get())
                    .abs()
                    .max(f64::from(matrix.ty.get()).abs()),
        );
    }
    if !reach.is_finite() || reach > FAR {
        return None;
    }
    Some(Summary {
        bounds,
        levels,
        reach,
    })
}

/// Whether `point` (stage) is outside what is below a plain clip, placed by `matrix` in a
/// parent whose world matrix is `to_global`.
fn outside(below: &Summary, matrix: &Matrix, to_global: &Matrix, point: Point<Twips>) -> bool {
    if !below.bounds.is_valid() {
        // Plain and empty: nothing in it can be hit.
        return true;
    }
    let Some(in_parent) = map_out(matrix, &below.bounds) else {
        return false;
    };
    let Some(global) = map_out(to_global, &in_parent) else {
        return false;
    };
    let (x, y) = (f64::from(point.x.get()), f64::from(point.y.get()));
    // The exact tests reach a leaf's world bounds through one `f32` matrix product per
    // level, each rounding its translation to a twip (half a twip of error per level), then
    // round every corner once more; and their `f32` error is proportional to the size of
    // the terms they add up. This margin is several times all of that, and at least a pixel.
    let levels = f64::from(below.levels) + 2.0;
    let terms = stretch(to_global)
        * (stretch(matrix) * below.reach
            + f64::from(matrix.tx.get())
                .abs()
                .max(f64::from(matrix.ty.get()).abs()));
    let margin = 20.0
        + 2.0 * levels
        + terms * levels * 1.0e-6
        + 1.0e-5 * largest(&global).max(x.abs()).max(y.abs());
    if !margin.is_finite() {
        return false;
    }
    x < f64::from(global.x_min.get()) - margin
        || x > f64::from(global.x_max.get()) + margin
        || y < f64::from(global.y_min.get()) - margin
        || y > f64::from(global.y_max.get()) + margin
}

/// Whether the pick can skip `child` of `parent` (whose world matrix is `to_global`): it is
/// a plain clip and `point` is outside everything below it, so its pick is a miss.
pub fn misses<'gc>(
    child: DisplayObject<'gc>,
    parent: DisplayObject<'gc>,
    to_global: &Matrix,
    point: Point<Twips>,
) -> bool {
    let DisplayObject::MovieClip(clip) = child else {
        return false;
    };
    if !child
        .parent()
        .is_some_and(|p| DisplayObject::ptr_eq(p, parent))
    {
        return false;
    }
    let Some(below) = summary(clip, false) else {
        ENTERED_OTHER.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if verify() != Verify::Off && !check_note(clip, &below) {
        return false;
    }
    let matrix = child.base().matrix();
    // (`RUFFLE_PICK_FAULT`, testing the test: every plain branch is skipped, wherever the
    // pointer is, so verify mode has wrong skips to find.)
    let skip = outside(&below, &matrix, to_global, point) || fault();
    if skip {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
    } else {
        ENTERED_PLAIN.fetch_add(1, Ordering::Relaxed);
    }
    skip
}

/// Verify mode: the note against the branch worked out from nothing; whether it holds.
fn check_note<'gc>(clip: MovieClip<'gc>, noted: &Summary) -> bool {
    let covers = match summary(clip, true) {
        // (The note says plain.)
        None => false,
        Some(now) => {
            !now.bounds.is_valid()
                || (noted.bounds.is_valid()
                    && noted.bounds.x_min <= now.bounds.x_min
                    && noted.bounds.y_min <= now.bounds.y_min
                    && noted.bounds.x_max >= now.bounds.x_max
                    && noted.bounds.y_max >= now.bounds.y_max
                    && noted.levels >= now.levels
                    && noted.reach * 1.001 >= now.reach)
        }
    };
    if !covers {
        STALE.fetch_add(1, Ordering::Relaxed);
        let message = format!(
            "PICK NOTE OUT OF DATE (a change below it did not mark it): {}",
            path(clip.into())
        );
        tracing::error!(target: "ruffle_aqw_perf", "{message}");
        if verify() == Verify::Panic {
            panic!("{message}");
        }
        // Carry on with a fresh note.
        clip.set_pick_note(Note::default());
    }
    covers
}
