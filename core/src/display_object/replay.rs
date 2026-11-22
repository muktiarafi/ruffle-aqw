//! AQW build (Mugetsu): frame interpolation's in-between pictures, drawn without walking the
//! display list again.
//!
//! Between two SWF frames nothing changes but how far the picture is between them, so every
//! in-between picture has the same commands, except for the transforms of the objects that
//! move. The first picture after a frame (or after something visible changed: an event, a
//! script) walks the display list as usual and records how each command was made (a
//! `Recorder`); the next ones (`Program::replay`) only blend the moving objects' transforms
//! again, and draw a few kinds of objects again as the walk would ("islands": shape tweens,
//! text, cached layers, masked objects), as they depend on more than their transform. The
//! commands come out the same as a walk's (checked by comparing both while developing it).

use crate::context::RenderContext;
use crate::display_object::graphic::Graphic;
use crate::display_object::{
    DisplayObject, DisplayObjectBase, RenderOptions, TDisplayObject, interp, render_base,
};
use crate::tessellation_cache::{Measure, TessellationCache};
use gc_arena::Collect;
use ruffle_render::blend::ExtendedBlendMode;
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::matrix::Matrix;
use ruffle_render::tessellator::EdgeMode;
use ruffle_render::transform::{Transform, TransformStack};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use swf::{ColorTransform, Twips};

const NONE: u32 = u32::MAX;

thread_local! {
    /// Changes to display objects so far (anything that may change how they're drawn).
    static CHANGES: Cell<u64> = const { Cell::new(0) };
}

/// Called whenever a display object changes in a way that may change how it's drawn.
#[inline]
pub fn changed() {
    CHANGES.with(|c| c.set(c.get().wrapping_add(1)));
}

fn changes() -> u64 {
    CHANGES.with(Cell::get)
}

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Replays on (default) or off (every picture walks the stage, for comparisons).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------------------------
// Lean render, OFF unless its switch is set: a recorded walk draws the branches that have not
// changed from the last recording instead of walking them.
//
// A walk costs every object on stage a transform, a blend with its last look, a handful of
// checks and its commands; in a room where most characters stand, nearly all of that comes out
// as it did the frame before. With `RUFFLE_LEAN_RENDER` (a `lean-render` file next to the exe)
// every recorded walk keeps the last program as well, and when `render_base` reaches an object
// - that was a node of the last program (its `Note` says which),
// - that is not dirty (nothing in its branch changed since that program's walk began:
//   `mark_dirty` is called wherever `changed()` is, and for a change of children),
// - whose branch was recorded whole, with the usual options, and with nothing in it moving
//   between two frames (`Segment::usable`),
// - whose own transform and blend mode are what they were, that isn't moving now, inside a
//   parent that isn't, and that would be drawn with the transform it was recorded with,
// - in the same mask state,
// the branch's ops are run again as a replay runs them (`run`: shapes are looked up, text,
// cached and masked objects and tweens are drawn again by their objects) and copied into the
// new recording with its nodes (`Recorder::reuse`). The walk goes on with the next object.
//
// Without frame interpolation every picture is a walk; with the switch on it is recorded too
// (nothing in it moves between frames), so the next one can draw from it, as long as most of
// the stage can be (see `Pace`; `RUFFLE_LEAN_RENDER=every` records every walk, for tests).
//
// What a walk does besides making commands, and how the branches that aren't walked get it:
// - `pre_render` clears each object's "cache invalidated" flag: an object in a reused branch
//   had it clear (nothing changed) and it stays so.
// - Tessellations are touched when looked up (so they aren't freed while on screen): `run`
//   looks every graphic's shape up, as the walk does.
// - The interpolation's trackers note each object's look every frame; the objects of a reused
//   branch aren't visited, so when one is walked again its tracker is caught up first
//   (`Recorder::catch_up`, `interp::Tracker::catch_up`): it was drawn, as it was, in the last
//   program.
//
// `RUFFLE_LEAN_RENDER_VERIFY` (`=panic` for tests; a `lean-render-verify` file) walks every
// branch that would be reused as well, without recording it, compares the two lists of
// commands, draws the walked one, and reports every difference ("LEAN RENDER VERIFY") with the
// deepest branch that differs. `RUFFLE_LEAN_RENDER_FAULT` breaks the marking on purpose, so
// the check has something to find.

fn switch(cell: &'static std::sync::OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| std::env::var_os(name).is_some())
}

pub fn lean_render() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch(&ON, "RUFFLE_LEAN_RENDER")
}

/// `RUFFLE_LEAN_RENDER=every` (for tests): without interpolation every walk is recorded, also
/// where that doesn't pay (see `Pace`).
fn every_walk() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUFFLE_LEAN_RENDER").is_ok_and(|value| value == "every"))
}

pub fn lean_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    switch(&ON, "RUFFLE_LEAN_RENDER_VERIFY")
}

/// `RUFFLE_LEAN_RENDER_FAULT` (testing the test): changes aren't marked, so branches are drawn
/// from the last recording that shouldn't be. Only ever with `RUFFLE_LEAN_RENDER_VERIFY`, to
/// see that it reports them.
#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    /// Nothing is marked.
    All,
    /// `=climb`: a change marks its object, but `mark_dirty` tells none of the objects above
    /// it (Ruffle's own "the cached picture is out of date" still climbs, and marks them).
    Climb,
    /// `=invalidate`: that climb of Ruffle's marks nothing; only the setters, and the changes
    /// of children, do.
    Invalidate,
}

fn fault() -> Fault {
    static FAULT: std::sync::OnceLock<Fault> = std::sync::OnceLock::new();
    *FAULT.get_or_init(
        || match std::env::var("RUFFLE_LEAN_RENDER_FAULT").as_deref() {
            Err(_) => Fault::None,
            Ok("climb") => Fault::Climb,
            Ok("invalidate") => Fault::Invalidate,
            Ok(_) => Fault::All,
        },
    )
}

/// Whether recordings are kept for lean render, and objects marked dirty.
pub fn lean_tracking() -> bool {
    #[cfg(test)]
    if let Some(forced) = FORCED_TRACKING.with(Cell::get) {
        return forced;
    }
    // (One flag: every setter of every display object asks.)
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| lean_render() || lean_verify())
}

#[cfg(test)]
thread_local! {
    /// For tests (each runs on a thread of its own): lean render's bookkeeping on or off.
    static FORCED_TRACKING: Cell<Option<bool>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn force_lean_tracking(on: Option<bool>) {
    FORCED_TRACKING.with(|f| f.set(on));
}

/// Per display object, for lean render: where it was in a recording (the recording's
/// generation and its node there), and when it or anything below it last changed: the
/// generation of the newest recording begun by then (see `mark_dirty`). Copies (instantiated
/// symbols) start with neither.
#[derive(Default)]
pub struct Note {
    stamp: Cell<u64>,
    marked: Cell<u32>,
}

impl Clone for Note {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Note {
    /// Whether the object or anything below it changed since the recording of `generation`
    /// was begun (so what that recorded of it may not be what it draws now).
    fn changed_since(&self, generation: u32) -> bool {
        self.marked.get() >= generation
    }

    fn stamp(&self, generation: u32, node: u32) {
        self.stamp
            .set((u64::from(generation) << 32) | u64::from(node));
    }

    /// Its node in the recording of `generation`, if it was stamped for it.
    fn node_in(&self, generation: u32) -> Option<u32> {
        let stamp = self.stamp.get();
        ((stamp >> 32) as u32 == generation).then_some(stamp as u32)
    }
}

/// The object of `base` changed in a way that may change how it or its branch is drawn: it
/// and its ancestors are marked with the generation of the newest recording begun so far, so
/// none of them is drawn from that recording or an older one (`Note::changed_since`).
///
/// An object's mark is never older than any mark below it: marking climbs to the root, and a
/// new parent is marked with its child (`set_parent`). So the climb can stop at the first
/// object that has this mark already, and an object whose mark is older than a recording has
/// nothing below it that changed since that was begun. (Nothing is ever cleaned: a walk that
/// doesn't reach an object, hidden or behind a cached picture, leaves nothing to go stale.)
pub(crate) fn mark_dirty(base: &DisplayObjectBase<'_>) {
    if !lean_tracking() {
        return;
    }
    let fault = fault();
    if fault == Fault::All {
        return;
    }
    let now = GENERATION.with(Cell::get);
    if base.render_note.marked.replace(now) == now || fault == Fault::Climb {
        return;
    }
    let mut node = base.parent();
    while let Some(object) = node {
        let base = object.base();
        if base.render_note.marked.replace(now) == now {
            break;
        }
        node = base.parent();
    }
}

/// `mark_dirty`, from `invalidate_cached_bitmap` (apart for `Fault::Invalidate`).
pub(crate) fn mark_invalidated(base: &DisplayObjectBase<'_>) {
    if lean_tracking() && fault() != Fault::Invalidate {
        mark_dirty(base);
    }
}

pub(crate) fn mark_dirty_from(object: Option<DisplayObject<'_>>) {
    if let Some(object) = object {
        mark_dirty(&object.base());
    }
}

#[derive(Clone, Copy, Default)]
struct LeanStats {
    /// Recorded walks, the nodes they walked, the branches they drew from the last recording
    /// and the nodes in those.
    walks: u64,
    walked: u64,
    reused: u64,
    reused_nodes: u64,
    frames: u64,
    /// What the check found, in all.
    found: u64,
}

thread_local! {
    static LEAN_STATS: Cell<LeanStats> = const { Cell::new(LeanStats { walks: 0, walked: 0, reused: 0, reused_nodes: 0, frames: 0, found: 0 }) };
    /// The last recording's generation (0: none; see `Note`).
    static GENERATION: Cell<u32> = const { Cell::new(0) };
}

fn lean_stats(change: impl FnOnce(&mut LeanStats)) {
    LEAN_STATS.with(|s| {
        let mut stats = s.get();
        change(&mut stats);
        s.set(stats);
    });
}

/// The check found a branch whose reused commands aren't what a walk gives.
fn lean_found(object: DisplayObject<'_>, what: String) {
    let mut count = 0;
    lean_stats(|s| {
        s.found += 1;
        count = s.found;
    });
    let kind = match object {
        DisplayObject::Stage(_) => "stage",
        DisplayObject::Bitmap(_) => "bitmap",
        DisplayObject::Avm1Button(_) => "AVM1 button",
        DisplayObject::Avm2Button(_) => "button",
        DisplayObject::EditText(_) => "text field",
        DisplayObject::TextLine(_) => "text line",
        DisplayObject::Graphic(_) => "shape",
        DisplayObject::MorphShape(_) => "shape tween",
        DisplayObject::MovieClip(_) => "clip",
        DisplayObject::Text(_) => "text",
        DisplayObject::Video(_) => "video",
        DisplayObject::LoaderDisplay(_) => "loader",
    };
    let name = object.name().map(|n| n.to_string()).unwrap_or_default();
    let message = format!(
        "LEAN RENDER VERIFY: the branch of {kind} {} {name:?} was drawn from the last recording, but {what}",
        object.id()
    );
    if count <= 40 {
        tracing::warn!(target: "ruffle_aqw_perf", "{message}");
    }
    if std::env::var_os("RUFFLE_LEAN_RENDER_VERIFY").is_some_and(|v| v == "panic") {
        panic!("{message}");
    }
}

fn describe(object: DisplayObject<'_>) -> String {
    let kind = match object {
        DisplayObject::Stage(_) => "stage",
        DisplayObject::Bitmap(_) => "bitmap",
        DisplayObject::Avm1Button(_) => "AVM1 button",
        DisplayObject::Avm2Button(_) => "button",
        DisplayObject::EditText(_) => "text field",
        DisplayObject::TextLine(_) => "text line",
        DisplayObject::Graphic(_) => "shape",
        DisplayObject::MorphShape(_) => "shape tween",
        DisplayObject::MovieClip(_) => "clip",
        DisplayObject::Text(_) => "text",
        DisplayObject::Video(_) => "video",
        DisplayObject::LoaderDisplay(_) => "loader",
    };
    let name = object.name().map(|n| n.to_string()).unwrap_or_default();
    format!("{kind} {} {name:?}", object.id())
}

/// The check: a branch of the last program (`node`) gave other commands than its walk. The
/// deepest node below it of which that is true too (the node itself, if none is).
fn blame<'gc>(context: &mut RenderContext<'_, 'gc>, prev: &Program<'gc>, node: u32) -> u32 {
    let end = prev.nodes[node as usize]
        .seg
        .nodes_end
        .min(prev.nodes.len() as u32);
    let drawn = Drawn {
        nodes: &prev.nodes,
        drawn: &prev.drawn,
        fresh: true,
    };
    let mut child = node + 1;
    while child < end {
        let seg = prev.nodes[child as usize].seg;
        let next = seg.nodes_end.max(child + 1);
        let parent = prev.nodes[child as usize].parent;
        if seg.usable && parent != NONE {
            let ops = &prev.list(seg.list)[seg.ops.0 as usize..seg.ops.1 as usize];
            let picture = std::mem::take(&mut context.commands);
            context.commands.set_maskers(seg.maskers);
            let result = run(ops, &prev.objects, &drawn, context);
            let reused = std::mem::take(&mut context.commands);
            context.commands.set_maskers(seg.maskers);
            context
                .transform_stack
                .push_absolute(&prev.nodes[parent as usize].recorded);
            render_base(
                prev.node_objects[child as usize],
                context,
                RenderOptions::default(),
            );
            context.transform_stack.pop();
            let walked = std::mem::replace(&mut context.commands, picture);
            if result.is_err() || difference(&reused.commands, &walked.commands).is_some() {
                return blame(context, prev, child);
            }
        }
        child = next;
    }
    node
}

/// Called when a frame's walks are done: a line in the log now and then.
pub fn lean_frame_done() {
    if !lean_tracking() {
        return;
    }
    let mut stats = LeanStats::default();
    lean_stats(|s| {
        s.frames += 1;
        stats = *s;
    });
    if stats.frames >= 150 {
        let per = |n: u64| n as f64 / stats.walks.max(1) as f64;
        tracing::info!(
            target: "ruffle_aqw_perf",
            "lean render ({}): {} recorded walks in {} frames; per walk: {:.0} objects walked, {:.0} branches drawn from the last recording ({:.0} objects in them); check: {} found in all",
            match (lean_render(), lean_verify()) {
                (_, true) => "check",
                (true, false) => "on",
                (false, false) => "off",
            },
            stats.walks,
            stats.frames,
            per(stats.walked),
            per(stats.reused),
            per(stats.reused_nodes),
            stats.found,
        );
        LEAN_STATS.with(|s| {
            s.set(LeanStats {
                found: stats.found,
                ..LeanStats::default()
            })
        });
    }
}

/// What two lists of commands differ in, if anything.
fn difference(reused: &[Command], walked: &[Command]) -> Option<String> {
    fn same_debug<T: std::fmt::Debug>(a: &T, b: &T) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }
    // (Also where one isn't a number: an object scaled by 0/0 is drawn so by both.)
    fn same_transform(a: &Transform, b: &Transform) -> bool {
        a.same(b) || same_debug(a, b)
    }
    for (index, pair) in reused.iter().zip(walked).enumerate() {
        let same = match pair {
            (
                Command::RenderBitmap {
                    bitmap: a,
                    transform: ta,
                    smoothing: sa,
                    pixel_snapping: pa,
                    region: ra,
                },
                Command::RenderBitmap {
                    bitmap: b,
                    transform: tb,
                    smoothing: sb,
                    pixel_snapping: pb,
                    region: rb,
                },
            ) => a == b && same_transform(ta, tb) && sa == sb && same_debug(pa, pb) && ra == rb,
            (
                Command::RenderStage3D {
                    bitmap: a,
                    transform: ta,
                },
                Command::RenderStage3D {
                    bitmap: b,
                    transform: tb,
                },
            ) => a == b && same_transform(ta, tb),
            (
                Command::RenderShape {
                    shape: a,
                    transform: ta,
                },
                Command::RenderShape {
                    shape: b,
                    transform: tb,
                },
            ) => crate::tessellation_cache::made_alike(a, b) && same_transform(ta, tb),
            (
                Command::RenderAlphaMask {
                    maskee_commands: a,
                    mask_commands: ma,
                },
                Command::RenderAlphaMask {
                    maskee_commands: b,
                    mask_commands: mb,
                },
            ) => {
                if let Some(inner) = difference(&a.commands, &b.commands)
                    .or_else(|| difference(&ma.commands, &mb.commands))
                {
                    return Some(format!("in the alpha mask at command {index}: {inner}"));
                }
                true
            }
            (
                Command::DrawRect {
                    color: a,
                    matrix: ma,
                },
                Command::DrawRect {
                    color: b,
                    matrix: mb,
                },
            )
            | (
                Command::DrawLine {
                    color: a,
                    matrix: ma,
                },
                Command::DrawLine {
                    color: b,
                    matrix: mb,
                },
            )
            | (
                Command::DrawLineRect {
                    color: a,
                    matrix: ma,
                },
                Command::DrawLineRect {
                    color: b,
                    matrix: mb,
                },
            ) => a == b && ma == mb,
            (Command::PushMask, Command::PushMask)
            | (Command::ActivateMask, Command::ActivateMask)
            | (Command::DeactivateMask, Command::DeactivateMask)
            | (Command::PopMask, Command::PopMask) => true,
            (Command::Blend(a, mode_a), Command::Blend(b, mode_b)) => {
                if !same_debug(mode_a, mode_b) {
                    return Some(format!("the blend at command {index} has another mode"));
                }
                if let Some(inner) = difference(&a.commands, &b.commands) {
                    return Some(format!("in the blend at command {index}: {inner}"));
                }
                true
            }
            _ => false,
        };
        if !same {
            return Some(format!(
                "command {index} is {} there and {} in a walk{}",
                command_name(pair.0),
                command_name(pair.1),
                match pair {
                    (
                        Command::RenderShape {
                            shape: a,
                            transform: ta,
                        },
                        Command::RenderShape {
                            shape: b,
                            transform: tb,
                        },
                    ) => format!(
                        " (the same tessellation: {}, the same transform: {})",
                        crate::tessellation_cache::made_alike(a, b),
                        ta.same(tb)
                    ),
                    _ => String::new(),
                }
            ));
        }
    }
    (reused.len() != walked.len()).then(|| {
        format!(
            "it has {} commands there and {} in a walk",
            reused.len(),
            walked.len()
        )
    })
}

/// For the perf log: pictures recorded, replayed, walked because recording wasn't possible,
/// replays given up (walked instead), recordings redone between two frames as something
/// visible changed and a redraw was asked for, and pictures replayed although something had
/// changed (it's shown with the next frame, see `Saved::plan`).
pub const STAT_NAMES: [&str; 6] = [
    "recorded",
    "replayed",
    "not recordable",
    "given up",
    "changed mid-frame",
    "left for the next frame",
];
const RECORDED: usize = 0;
const REPLAYED: usize = 1;
const NOT_RECORDABLE: usize = 2;
const GIVEN_UP: usize = 3;
const CHANGED_MID_FRAME: usize = 4;
const LEFT_FOR_NEXT_FRAME: usize = 5;
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static LAST_REASON: Mutex<Option<String>> = Mutex::new(None);

fn count(stat: usize) {
    STATS[stat].fetch_add(1, Ordering::Relaxed);
}

fn note(reason: String) {
    if let Ok(mut last) = LAST_REASON.lock() {
        *last = Some(reason);
    }
}

pub fn stats() -> [u64; 6] {
    std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed))
}

/// `RUFFLE_EAGER_REPLAY` (testing): every change between two frames is drawn with the next
/// picture, as before (see `Saved::plan`).
fn eager() -> bool {
    static EAGER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EAGER.get_or_init(|| std::env::var_os("RUFFLE_EAGER_REPLAY").is_some())
}

/// Why the last picture couldn't be recorded or replayed.
pub fn take_reason() -> Option<String> {
    LAST_REASON.lock().ok().and_then(|mut last| last.take())
}

/// Lean render: where a node's branch is in its recording (see `Recorder::reuse`).
#[derive(Clone, Copy)]
struct Segment {
    /// The list its ops are in (see `Recorder::list_parent`) and their range there.
    list: u32,
    ops: (u32, u32),
    /// Its nodes: its own index up to (not including) this.
    nodes_end: u32,
    /// Its objects (islands, drawings) and the lists nested in it, as ranges.
    objects: (u32, u32),
    lists: (u32, u32),
    /// The mask state it was drawn in, and its blend mode.
    maskers: u32,
    blend: ExtendedBlendMode,
    /// While recording: how many moving nodes there were before it.
    movers: u32,
    /// Recorded for lean render, whole, with the usual options, nothing in it moving and no
    /// tessellation in it still being made: it can be drawn again as it is.
    usable: bool,
    /// Copied from an older recording: the objects below its node don't know their nodes in
    /// this one (they're told when the node's object is walked again).
    copied: bool,
}

/// An object's transform in the recorded picture.
#[derive(Clone)]
struct Node {
    /// The node it's inside (`NONE`: none; it's drawn inside `outer`).
    parent: u32,
    outer: Option<Box<Transform>>,
    /// Its real transform, and the previous frame's look it's blended from.
    local: Transform,
    from: Option<(Matrix, ColorTransform)>,
    /// Whether it (or one it's inside) is drawn elsewhere as the picture's time moves on.
    moving: bool,
    /// What it was drawn with in the recorded picture.
    recorded: Transform,
    seg: Segment,
}

/// Where a graphic's tessellations come from (see `Graphic::handle_for`).
#[derive(Clone)]
pub(crate) struct ShapeSource {
    /// The movie by its address (see `Library::movie_addr`): a recording outlives the render
    /// it was made in.
    movie: usize,
    shape: Arc<swf::Shape>,
    cache: Rc<RefCell<TessellationCache>>,
}

/// How to make a command (or several) of an in-between picture.
pub(crate) enum Op {
    /// The same in every picture.
    Same(Command),
    /// Drawn with the transform of its node.
    Moved(Command, u32),
    /// A shape drawn with the transform of its node, as long as it's at the same scale (else
    /// it would be tessellated for it: the replay is given up).
    MovedScaled(Command, u32, Measure),
    /// A graphic's shape drawn with the transform of its node, its tessellation looked up (or
    /// made) as the walk does: the closest one may be another copy's, made since.
    Shape { node: u32, source: Rc<ShapeSource> },
    /// A drawing (`graphics`) drawn with the transform of its node; drawn again by its object
    /// when its size changes (it's tessellated for its size).
    Drawing {
        command: Command,
        node: u32,
        measure: Measure,
        object: u32,
    },
    /// An object drawn again as the walk would, inside the transform of node `node`, with the
    /// list in the same mask state (`maskers`).
    Island {
        object: u32,
        node: u32,
        options: RenderOptions,
        maskers: u32,
    },
    /// Commands drawn with a blend mode.
    Nest {
        ops: Vec<Op>,
        mode: RenderBlendMode,
        maskers: u32,
    },
}

#[derive(Default)]
struct List {
    ops: Vec<Op>,
    /// How many commands of the list being recorded have their op.
    accounted: usize,
    /// Lean render: which list it is (see `Recorder::list_parent`).
    id: u32,
}

/// An op for another recording, whose nodes and objects are `nodes` and `objects` further on.
fn shifted(op: &Op, nodes: u32, objects: u32) -> Op {
    match op {
        Op::Same(command) => Op::Same(command.clone()),
        Op::Moved(command, node) => Op::Moved(command.clone(), node.wrapping_add(nodes)),
        Op::MovedScaled(command, node, measure) => {
            Op::MovedScaled(command.clone(), node.wrapping_add(nodes), *measure)
        }
        Op::Shape { node, source } => Op::Shape {
            node: node.wrapping_add(nodes),
            source: source.clone(),
        },
        Op::Drawing {
            command,
            node,
            measure,
            object,
        } => Op::Drawing {
            command: command.clone(),
            node: node.wrapping_add(nodes),
            measure: *measure,
            object: object.wrapping_add(objects),
        },
        Op::Island {
            object,
            node,
            options,
            maskers,
        } => Op::Island {
            object: object.wrapping_add(objects),
            node: node.wrapping_add(nodes),
            options: options.clone(),
            maskers: *maskers,
        },
        Op::Nest { ops, mode, maskers } => Op::Nest {
            ops: ops.iter().map(|op| shifted(op, nodes, objects)).collect(),
            mode: mode.clone(),
            maskers: *maskers,
        },
    }
}

/// Records how the commands of a walk are made (see the module docs).
pub struct Recorder<'gc> {
    nodes: Vec<Node>,
    /// The nodes the walk is inside.
    path: Vec<u32>,
    /// The ops of the command list being recorded, and of those it's inside.
    lists: Vec<List>,
    objects: Vec<DisplayObject<'gc>>,
    /// Inside an island (its commands aren't recorded).
    suspended: u32,
    invalid: Option<String>,
    /// Lean render (see `lean_render`): on for this recording, and the last program, which
    /// branches are drawn from.
    lean: bool,
    prev: Option<Program<'gc>>,
    /// This recording's generation (see `Note`).
    generation: u32,
    /// Each node's object.
    node_objects: Vec<DisplayObject<'gc>>,
    /// For each list (by its id; 0 is the picture's own): the list it's nested in, and the
    /// index of its `Op::Nest` there.
    list_parent: Vec<(u32, u32)>,
    last_closed: u32,
    /// How many of the nodes so far move.
    moving_nodes: u32,
    /// For the log: nodes walked, branches reused and the nodes in them.
    counts: (u64, u64, u64),
    scene: Option<Scene>,
    /// Lean render: empty lists with room in them, for the program (see `Spare`).
    spare_drawn: Vec<Option<Transform>>,
    spare_moving: Vec<u32>,
}

/// Lean render: the big lists of the program before the last, emptied, for the next
/// recording. A walk in a crowded room fills megabytes of nodes and ops; getting that memory
/// from the system and giving it back for every walk cost more than the walk saved.
struct Spare {
    nodes: Vec<Node>,
    ops: Vec<Op>,
    drawn: Vec<Option<Transform>>,
    moving: Vec<u32>,
    list_parent: Vec<(u32, u32)>,
}

thread_local! {
    static SPARE: RefCell<Vec<Spare>> = const { RefCell::new(Vec::new()) };
}

/// A list kept for the next recording, unless it's far bigger than what the last one needed
/// (a crowded room was left).
fn kept<T>(mut list: Vec<T>, needed: usize) -> Vec<T> {
    list.clear();
    if list.capacity() > 4 * needed.max(4096) {
        Vec::new()
    } else {
        list
    }
}

/// What a walk's commands depend on besides the display objects: what is on screen (objects
/// outside aren't drawn), the stage's matrix and its quality. Lean render draws nothing from
/// a recording made with another.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Scene {
    pub view_bounds: swf::Rectangle<Twips>,
    pub viewport: Matrix,
    pub quality: ruffle_render::quality::StageQuality,
}

impl<'gc> Recorder<'gc> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            path: Vec::new(),
            lists: vec![List::default()],
            objects: Vec::new(),
            suspended: 0,
            invalid: None,
            lean: false,
            prev: None,
            generation: 0,
            node_objects: Vec::new(),
            list_parent: vec![(NONE, 0)],
            last_closed: 0,
            moving_nodes: 0,
            counts: (0, 0, 0),
            scene: None,
            spare_drawn: Vec::new(),
            spare_moving: Vec::new(),
        }
    }

    /// The stage is about to be walked, with `scene`. Returns how many commands the last
    /// lean recording's picture had (room to make for this one's).
    pub(crate) fn scene(&mut self, scene: Scene) -> usize {
        if self
            .prev
            .as_ref()
            .is_some_and(|prev| prev.scene != Some(scene))
        {
            self.prev = None;
        }
        self.scene = Some(scene);
        match &self.prev {
            Some(prev) if self.lean => prev.commands as usize,
            _ => 0,
        }
    }

    /// Lean render: the recording is over; the last program's lists are kept for the next.
    fn recycle(&mut self) {
        if !self.lean {
            return;
        }
        let Some(prev) = self.prev.take() else {
            return;
        };
        let Program {
            nodes,
            ops,
            drawn,
            moving,
            list_parent,
            ..
        } = prev;
        let needed = nodes.len();
        let spare = Spare {
            ops: kept(ops, needed),
            drawn: kept(drawn, needed),
            moving: kept(moving, needed),
            list_parent: kept(list_parent, needed),
            nodes: kept(nodes, needed),
        };
        SPARE.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < 2 {
                s.push(spare);
            }
        });
    }

    /// A video is about to draw (or not): lean render can't tell when it changes, so no
    /// branch it's in is drawn from this recording.
    pub(crate) fn video(&mut self) {
        if self.lean {
            for &node in &self.path {
                self.nodes[node as usize].seg.usable = false;
            }
        }
    }

    /// A recorder for lean render: branches that haven't changed are drawn from `prev`.
    fn lean(prev: Option<Program<'gc>>) -> Self {
        let generation = GENERATION.with(|g| {
            // (0 is "none".)
            let next = g.get().wrapping_add(1).max(1);
            g.set(next);
            next
        });
        let mut recorder = Self {
            lean: true,
            // (Tessellated for another quality: nothing of it can be drawn again.)
            prev: prev.filter(|p| {
                p.key.edge_mode == Some(ruffle_render::tessellator::edge_mode())
                    && !p.node_objects.is_empty()
            }),
            generation,
            ..Self::new()
        };
        if let Some(spare) = SPARE.with(|s| s.borrow_mut().pop()) {
            recorder.nodes = spare.nodes;
            recorder.lists[0].ops = spare.ops;
            recorder.spare_drawn = spare.drawn;
            recorder.spare_moving = spare.moving;
            let mut list_parent = spare.list_parent;
            list_parent.push((NONE, 0));
            recorder.list_parent = list_parent;
        }
        if let Some(prev) = &recorder.prev {
            recorder.nodes.reserve(prev.nodes.len());
            recorder.lists[0].ops.reserve(prev.ops.len());
            recorder.objects.reserve(prev.objects.len());
            recorder.node_objects.reserve(prev.node_objects.len());
        }
        recorder
    }

    /// Whether this walk is recorded for lean render.
    pub(crate) fn is_lean(&self) -> bool {
        self.lean
    }

    /// Lean render: `object` is about to be walked. If it was drawn in the last program, its
    /// interpolation tracker hears of it (it may not have been visited then).
    pub(crate) fn catch_up(&self, object: DisplayObject<'gc>) {
        if let Some(prev) = &self.prev
            && self.lean
            && object.base().render_note.node_in(prev.generation).is_some()
        {
            object
                .base()
                .interp
                .catch_up(prev.key.frame, prev.walked_previous_frame);
        }
    }

    /// Lean render: draws `object`'s branch from the last program (node `old_index` there)
    /// and copies it into this recording. `false`: it couldn't be drawn; nothing was done.
    fn reuse(
        &mut self,
        context: &mut RenderContext<'_, 'gc>,
        object: DisplayObject<'gc>,
        old_index: u32,
        outer: Transform,
        parent: Option<u32>,
    ) -> bool {
        self.account(&context.commands);
        let Recorder {
            prev,
            nodes,
            lists,
            objects,
            node_objects,
            list_parent,
            generation,
            invalid,
            counts,
            ..
        } = self;
        let prev = prev.as_ref().expect("the last program");
        let seg = prev.nodes[old_index as usize].seg;
        let ops = &prev.list(seg.list)[seg.ops.0 as usize..seg.ops.1 as usize];
        let drawn = Drawn {
            nodes: &prev.nodes,
            drawn: &prev.drawn,
            fresh: true,
        };
        if lean_verify() {
            // Drawn from the recording into a list of its own, then walked into another (not
            // recorded: the recorder isn't in the context here); the walked one is kept.
            let maskers = context.commands.maskers();
            let picture = std::mem::take(&mut context.commands);
            context.commands.set_maskers(maskers);
            let result = run(ops, &prev.objects, &drawn, context);
            let reused = std::mem::take(&mut context.commands);
            context.commands.set_maskers(maskers);
            render_base(object, context, RenderOptions::default());
            let walked = std::mem::replace(&mut context.commands, picture);
            match &result {
                Err(why) => lean_found(object, format!("drawing it gave up: {why}")),
                Ok(()) => {
                    if let Some(what) = difference(&reused.commands, &walked.commands) {
                        // Which object below it is it? (The deepest whose own branch differs.)
                        let culprit = blame(context, prev, old_index);
                        let mut path = String::new();
                        let mut at = Some(prev.node_objects[culprit as usize]);
                        while let Some(o) = at {
                            path = format!("/{}{path}", describe(o));
                            at = o.parent();
                        }
                        lean_found(
                            object,
                            format!("{what}; the deepest branch that differs: {path}"),
                        );
                    }
                }
            }
            context.commands.commands.extend(walked.commands);
            if result.is_err() && invalid.is_none() {
                // (What was walked has no ops: this recording can't be used.)
                *invalid =
                    Some("lean render's check: a branch couldn't be drawn again".to_string());
            }
        } else {
            let before = context.commands.commands.len();
            if let Err(why) = run(ops, &prev.objects, &drawn, context) {
                context.commands.commands.truncate(before);
                note(why);
                return false;
            }
        }

        // Its ops, nodes, objects and nested lists, as this recording's.
        let (first, end) = (old_index, seg.nodes_end);
        let list = lists.last_mut().expect("a list");
        let node_shift = (nodes.len() as u32).wrapping_sub(first);
        let object_shift = (objects.len() as u32).wrapping_sub(seg.objects.0);
        let list_shift = (list_parent.len() as u32).wrapping_sub(seg.lists.0);
        let op_shift = (list.ops.len() as u32).wrapping_sub(seg.ops.0);
        list.ops
            .extend(ops.iter().map(|op| shifted(op, node_shift, object_shift)));
        objects.extend_from_slice(&prev.objects[seg.objects.0 as usize..seg.objects.1 as usize]);
        node_objects.extend_from_slice(&prev.node_objects[first as usize..end as usize]);
        for nested in seg.lists.0..seg.lists.1 {
            let (inside, at) = prev.list_parent[nested as usize];
            list_parent.push(if inside == seg.list {
                (list.id, at.wrapping_add(op_shift))
            } else {
                (inside.wrapping_add(list_shift), at)
            });
        }
        let new_index = nodes.len() as u32;
        nodes.reserve((end - first) as usize);
        for index in first..end {
            let old = &prev.nodes[index as usize];
            let mut node = old.clone();
            if index == first {
                node.parent = parent.unwrap_or(NONE);
                node.outer = parent.is_none().then(|| Box::new(outer.clone()));
            } else {
                node.parent = old.parent.wrapping_add(node_shift);
            }
            let seg_here = &mut node.seg;
            if seg_here.list == seg.list {
                seg_here.list = list.id;
                seg_here.ops = (
                    seg_here.ops.0.wrapping_add(op_shift),
                    seg_here.ops.1.wrapping_add(op_shift),
                );
            } else {
                seg_here.list = seg_here.list.wrapping_add(list_shift);
            }
            seg_here.nodes_end = seg_here.nodes_end.wrapping_add(node_shift);
            seg_here.objects = (
                seg_here.objects.0.wrapping_add(object_shift),
                seg_here.objects.1.wrapping_add(object_shift),
            );
            seg_here.lists = (
                seg_here.lists.0.wrapping_add(list_shift),
                seg_here.lists.1.wrapping_add(list_shift),
            );
            seg_here.copied = true;
            nodes.push(node);
        }
        list.accounted = context.commands.commands.len();
        object.base().render_note.stamp(*generation, new_index);
        counts.1 += 1;
        counts.2 += u64::from(end - first);
        true
    }

    /// Lean render: the node `old` of the last program is walked again; the objects of the
    /// nodes right below it learn their nodes there (they were copied, not walked, last time).
    fn expose_children(prev: &Program<'gc>, old: u32) {
        let end = prev.nodes[old as usize].seg.nodes_end;
        let mut node = old + 1;
        while node < end.min(prev.nodes.len() as u32) {
            prev.node_objects[node as usize]
                .base()
                .render_note
                .stamp(prev.generation, node);
            node = prev.nodes[node as usize].seg.nodes_end.max(node + 1);
        }
    }

    /// Whether the walk is being recorded right now.
    pub(crate) fn recording(&self) -> bool {
        self.suspended == 0 && self.invalid.is_none()
    }

    fn fail(&mut self, why: impl FnOnce() -> String) {
        if self.invalid.is_none() {
            self.invalid = Some(why());
        }
    }

    /// Gives the commands added to the current list since last time their ops.
    fn account(&mut self, commands: &CommandList) {
        if !self.recording() {
            return;
        }
        let list = self.lists.last_mut().expect("a list");
        let node = self
            .path
            .last()
            .copied()
            .filter(|&n| self.nodes[n as usize].moving);
        let mut failed = None;
        let mut placeholder = false;
        for command in &commands.commands[list.accounted.min(commands.commands.len())..] {
            if self.lean
                && let Command::RenderShape { shape, .. } = command
                && crate::display_object::graphic::deferred_shapes::is_placeholder(shape)
            {
                placeholder = true;
            }
            let op = match node {
                None => Op::Same(command.clone()),
                Some(n) => {
                    let drawn = &self.nodes[n as usize].recorded;
                    match command {
                        Command::RenderShape { transform, .. } if transform.same(drawn) => {
                            Op::MovedScaled(command.clone(), n, Measure::of(&transform.matrix))
                        }
                        Command::RenderBitmap { transform, .. }
                        | Command::RenderStage3D { transform, .. }
                            if transform.same(drawn) =>
                        {
                            Op::Moved(command.clone(), n)
                        }
                        Command::PushMask
                        | Command::ActivateMask
                        | Command::DeactivateMask
                        | Command::PopMask => Op::Same(command.clone()),
                        other => {
                            let drew = match other {
                                Command::RenderShape { transform, .. }
                                | Command::RenderBitmap { transform, .. } => {
                                    format!(" with {:?}", transform.matrix)
                                }
                                Command::DrawRect { matrix, .. } => format!(" with {matrix:?}"),
                                _ => String::new(),
                            };
                            failed = Some(format!(
                                "a moving object drew apart from its transform ({}{drew}, its \
                                 transform {:?})",
                                command_name(other),
                                drawn.matrix
                            ));
                            break;
                        }
                    }
                }
            };
            list.ops.push(op);
        }
        list.accounted = commands.commands.len();
        if placeholder {
            // Lean render: the command holds a stand-in for a tessellation still being made
            // (swapped in the picture's list afterwards, not here): none of the branches it
            // is in can be drawn from this recording.
            for &node in &self.path {
                self.nodes[node as usize].seg.usable = false;
            }
        }
        if let Some(why) = failed {
            self.fail(|| why);
        }
    }

    /// Before `render_base` draws `object`: whether it's drawn again in every picture (an
    /// island), as it depends on more than its transform; then `island_end` follows.
    pub(crate) fn island_begin(
        &mut self,
        commands: &CommandList,
        object: DisplayObject<'gc>,
        options: &RenderOptions,
        cached: bool,
    ) -> bool {
        if !self.recording() {
            return false;
        }
        if object.as_video().is_some() {
            self.fail(|| "a video".to_string());
            return false;
        }
        // (A graphic in a 9-slice scaled clip is drawn stretched apart from its transform.)
        let sliced = matches!(object, DisplayObject::Graphic(_))
            && object.parent().is_some_and(|parent| {
                let grid = parent.scaling_grid();
                grid.width() > Twips::ZERO && grid.height() > Twips::ZERO
            });
        let island = cached
            || sliced
            || object.as_morph_shape().is_some()
            || object.as_text().is_some()
            || object.as_edit_text().is_some()
            || matches!(object, DisplayObject::TextLine(_))
            || object.masker().is_some()
            || object.scroll_rect().is_some()
            || object.opaque_background().is_some();
        if !island {
            return false;
        }
        self.account(commands);
        let Some(&node) = self.path.last() else {
            self.fail(|| "an island outside the stage".to_string());
            return false;
        };
        let list = self.lists.last_mut().expect("a list");
        list.ops.push(Op::Island {
            object: self.objects.len() as u32,
            node,
            options: options.clone(),
            maskers: commands.maskers(),
        });
        self.objects.push(object);
        self.suspended += 1;
        true
    }

    pub(crate) fn island_end(&mut self, commands: &CommandList) {
        self.suspended -= 1;
        if self.suspended == 0 {
            self.lists.last_mut().expect("a list").accounted = commands.commands.len();
        }
    }

    /// `render_base` pushed an object's transform: `real` blended from `from` into `drawn`,
    /// drawn inside `stack`'s current transform. Returns whether it was recorded (then
    /// `pop_node` follows).
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn push_node(
        &mut self,
        commands: &CommandList,
        stack: &TransformStack,
        real: Transform,
        from: Option<(Matrix, ColorTransform)>,
        drawn: &Transform,
        object: DisplayObject<'gc>,
        usual_options: bool,
    ) -> bool {
        if !self.recording() {
            return false;
        }
        self.account(commands);
        let parent = self.path.last().copied().unwrap_or(NONE);
        let outer = stack.transform();
        let recorded = outer.then(drawn);
        let changes = from
            .is_some_and(|(matrix, color)| matrix != real.matrix || color != real.color_transform);
        let moving = changes || (parent != NONE && self.nodes[parent as usize].moving);
        let index = self.nodes.len() as u32;
        let list = self.lists.last().expect("a list");
        let seg = Segment {
            list: list.id,
            ops: (list.ops.len() as u32, 0),
            nodes_end: 0,
            objects: (self.objects.len() as u32, 0),
            lists: (self.list_parent.len() as u32, 0),
            maskers: commands.maskers(),
            blend: object.blend_mode(),
            movers: self.moving_nodes,
            // (Settled in `pop_node`.)
            usable: self.lean && usual_options,
            copied: false,
        };
        if moving {
            self.moving_nodes += 1;
        }
        if self.lean {
            let note = &object.base().render_note;
            if let Some(prev) = &self.prev
                && let Some(old) = note.node_in(prev.generation)
                && prev.nodes[old as usize].seg.copied
            {
                Self::expose_children(prev, old);
            }
            note.stamp(self.generation, index);
            self.node_objects.push(object);
            self.counts.0 += 1;
        }
        self.path.push(index);
        self.nodes.push(Node {
            parent,
            outer: (parent == NONE).then(|| Box::new(outer)),
            local: real,
            from,
            moving,
            recorded,
            seg,
        });
        true
    }

    pub(crate) fn pop_node(&mut self, commands: &CommandList) {
        self.account(commands);
        if let Some(index) = self.path.pop() {
            let list = self.lists.last().expect("a list");
            let (nodes_end, objects_end, lists_end) = (
                self.nodes.len() as u32,
                self.objects.len() as u32,
                self.list_parent.len() as u32,
            );
            let seg = &mut self.nodes[index as usize].seg;
            seg.ops.1 = list.ops.len() as u32;
            seg.nodes_end = nodes_end;
            seg.objects.1 = objects_end;
            seg.lists.1 = lists_end;
            seg.usable = seg.usable && seg.list == list.id && seg.movers == self.moving_nodes;
        }
    }

    /// A graphic just drew its shape (the last command, if it was added).
    pub(crate) fn shape(
        &mut self,
        commands: &CommandList,
        movie: usize,
        shape: Arc<swf::Shape>,
        cache: Rc<RefCell<TessellationCache>>,
    ) {
        if !self.recording() {
            return;
        }
        let list = self.lists.last().expect("a list");
        let Some(&node) = self.path.last() else {
            self.account(commands);
            return;
        };
        let drawn = &self.nodes[node as usize].recorded;
        let is_it = commands.commands.len() == list.accounted + 1
            && matches!(commands.commands.last(), Some(Command::RenderShape { transform, .. }) if transform.same(drawn));
        if !is_it {
            self.account(commands);
            return;
        }
        let list = self.lists.last_mut().expect("a list");
        list.ops.push(Op::Shape {
            node,
            source: Rc::new(ShapeSource {
                movie,
                shape,
                cache,
            }),
        });
        list.accounted += 1;
    }

    /// `object` just drew its drawing (the last command, if it was added).
    pub(crate) fn drawing(&mut self, commands: &CommandList, object: DisplayObject<'gc>) {
        if !self.recording() {
            return;
        }
        let list = self.lists.last().expect("a list");
        let Some(&node) = self.path.last() else {
            self.account(commands);
            return;
        };
        let drawn = &self.nodes[node as usize];
        // (With lean render also where nothing moves: see `Drawn::fresh`. A replay draws the
        // command as it is then, as it did.)
        let is_it = (drawn.moving || self.lean)
            && commands.commands.len() == list.accounted + 1
            && matches!(commands.commands.last(), Some(Command::RenderShape { transform, .. }) if transform.same(&drawn.recorded));
        if !is_it {
            self.account(commands);
            return;
        }
        let command = commands.commands.last().expect("a command").clone();
        let measure = Measure::of(&drawn.recorded.matrix);
        let list = self.lists.last_mut().expect("a list");
        list.ops.push(Op::Drawing {
            command,
            node,
            measure,
            object: self.objects.len() as u32,
        });
        list.accounted += 1;
        self.objects.push(object);
    }

    /// `render_base` is about to draw into a new list, for a blend mode (`commands` is the
    /// current one). Returns whether it's recorded (then `nest_end` follows).
    pub(crate) fn nest_begin(&mut self, commands: &CommandList) -> bool {
        if !self.recording() {
            return false;
        }
        self.account(commands);
        let id = self.list_parent.len() as u32;
        let inside = self.lists.last().expect("a list").id;
        self.list_parent.push((inside, NONE));
        self.lists.push(List {
            id,
            ..List::default()
        });
        true
    }

    /// The nested list is done (`sub`), and back in the current one (`commands`, before the
    /// blend is added to it).
    pub(crate) fn nest_close(&mut self, sub: &CommandList) -> Vec<Op> {
        self.account(sub);
        let list = self.lists.pop();
        self.last_closed = list.as_ref().map_or(0, |list| list.id);
        list.map(|list| list.ops).unwrap_or_default()
    }

    /// The blend (if not empty) was added to `commands`, in mask state `maskers`.
    pub(crate) fn nest_end(
        &mut self,
        commands: &CommandList,
        ops: Vec<Op>,
        mode: RenderBlendMode,
        maskers: u32,
    ) {
        let list = self.lists.last_mut().expect("a list");
        if let Some(entry) = self.list_parent.get_mut(self.last_closed as usize) {
            entry.1 = list.ops.len() as u32;
        }
        list.ops.push(Op::Nest { ops, mode, maskers });
        list.accounted = commands.commands.len();
    }

    /// The walk is done (`commands` is its list).
    pub(crate) fn finish(&mut self, commands: &CommandList) -> Result<Program<'gc>, String> {
        self.account(commands);
        if let Some(why) = self.invalid.take() {
            return Err(why);
        }
        if !self.path.is_empty() || self.lists.len() != 1 || self.suspended != 0 {
            return Err("unbalanced".to_string());
        }
        let nodes = std::mem::take(&mut self.nodes);
        let mut moving = std::mem::take(&mut self.spare_moving);
        if self.moving_nodes != 0 {
            moving.extend((0..nodes.len() as u32).filter(|&n| nodes[n as usize].moving));
        }
        // Lean render: whether the frame before this one's had a recorded walk too (see
        // `interp::Tracker::catch_up`).
        let frame = interp::frame();
        let walked_previous_frame = match &self.prev {
            Some(prev) if prev.key.frame + 1 == frame => true,
            Some(prev) if prev.key.frame == frame => prev.walked_previous_frame,
            _ => false,
        };
        if self.lean {
            let counts = self.counts;
            lean_stats(|s| {
                s.walks += 1;
                s.walked += counts.0;
                s.reused += counts.1;
                s.reused_nodes += counts.2;
            });
        }
        Ok(Program {
            objects: std::mem::take(&mut self.objects),
            ops: std::mem::take(&mut self.lists[0].ops),
            // (With lean render it's sized when a replay first needs it: many programs are
            // never replayed.)
            drawn: if self.lean {
                std::mem::take(&mut self.spare_drawn)
            } else {
                vec![None; nodes.len()]
            },
            nodes,
            moving,
            key: Key::default(),
            node_objects: std::mem::take(&mut self.node_objects),
            list_parent: std::mem::take(&mut self.list_parent),
            generation: self.generation,
            walked_previous_frame,
            replayable: true,
            scene: self.scene,
            commands: commands.commands.len() as u32,
        })
    }
}

/// Lean render: draws `object`'s branch from the last recording instead of walking it, if
/// nothing it depends on changed (see the notes at `lean_render`). `real` is its own transform
/// now, `from` the look it's blended from. Returns whether it was drawn.
pub(crate) fn reuse_branch<'gc>(
    context: &mut RenderContext<'_, 'gc>,
    object: DisplayObject<'gc>,
    options: &RenderOptions,
    real: &Transform,
    from: Option<(Matrix, ColorTransform)>,
) -> bool {
    let Some(recorder) = context.recorder.as_deref() else {
        return false;
    };
    if !recorder.lean || !recorder.recording() {
        return false;
    }
    let Some(prev) = recorder.prev.as_ref() else {
        return false;
    };
    let note = &object.base().render_note;
    let Some(old_index) = note.node_in(prev.generation) else {
        return false;
    };
    if note.changed_since(prev.generation) {
        return false;
    }
    let Some(old) = prev.nodes.get(old_index as usize) else {
        return false;
    };
    let parent = recorder.path.last().copied();
    let outer = context.transform_stack.transform();
    let moving =
        from.is_some_and(|(matrix, color)| matrix != real.matrix || color != real.color_transform);
    if !old.seg.usable
        || moving
        || !(options.skip_masks && options.apply_transform && options.apply_matrix)
        || parent.is_some_and(|p| recorder.nodes[p as usize].moving)
        || !old.local.same(real)
        || !old.recorded.same(&outer.then(real))
        || old.seg.maskers != context.commands.maskers()
        || old.seg.blend != object.blend_mode()
    {
        return false;
    }
    // (Out of the context while the branch is drawn: what is drawn again by its object in it,
    // or walked for the check, must not be recorded here.)
    let Some(recorder) = context.recorder.take() else {
        return false;
    };
    let reused = recorder.reuse(context, object, old_index, outer, parent);
    context.recorder = Some(recorder);
    reused
}

impl Default for Recorder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// What a program was recorded for: it's replayed while all are the same.
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Key {
    /// The SWF frame.
    frame: u64,
    /// `changes()` after the last picture was drawn.
    changes: u64,
    edge_mode: Option<EdgeMode>,
}

impl Key {
    fn now() -> Self {
        Self {
            frame: interp::frame(),
            changes: changes(),
            edge_mode: Some(ruffle_render::tessellator::edge_mode()),
        }
    }
}

/// A recorded picture, to draw the next ones from (see the module docs).
#[derive(Collect)]
#[collect(no_drop)]
pub struct Program<'gc> {
    objects: Vec<DisplayObject<'gc>>,
    #[collect(require_static)]
    nodes: Vec<Node>,
    /// The nodes that move.
    #[collect(require_static)]
    moving: Vec<u32>,
    #[collect(require_static)]
    ops: Vec<Op>,
    /// Scratch: the moving nodes' transforms in the picture being drawn.
    #[collect(require_static)]
    drawn: Vec<Option<Transform>>,
    #[collect(require_static)]
    key: Key,
    /// Lean render (see `Recorder::reuse`): each node's object, where each nested list is,
    /// which recording this is, and whether the frame before its own had a recorded walk.
    node_objects: Vec<DisplayObject<'gc>>,
    #[collect(require_static)]
    list_parent: Vec<(u32, u32)>,
    #[collect(require_static)]
    generation: u32,
    #[collect(require_static)]
    walked_previous_frame: bool,
    /// Whether pictures can be replayed from it. (One recorded while tessellations were still
    /// being made can't; lean render keeps it to draw the next walk's unchanged branches from.)
    #[collect(require_static)]
    replayable: bool,
    #[collect(require_static)]
    scene: Option<Scene>,
    /// How many commands its picture had.
    #[collect(require_static)]
    commands: u32,
}

impl<'gc> Program<'gc> {
    /// Lean render: the ops of list `id` (0: the picture's own).
    fn list(&self, id: u32) -> &[Op] {
        if id == 0 {
            return &self.ops;
        }
        let (inside, at) = self.list_parent[id as usize];
        match self.list(inside).get(at as usize) {
            Some(Op::Nest { ops, .. }) => ops,
            _ => &[],
        }
    }

    /// Draws the picture `t` of the way between the previous frame and this one into
    /// `context.commands`. On an error, the commands are incomplete (walk instead).
    fn replay(&mut self, context: &mut RenderContext<'_, 'gc>, t: f32) -> Result<(), String> {
        if self.drawn.len() != self.nodes.len() {
            self.drawn.clear();
            self.drawn.resize(self.nodes.len(), None);
        }
        for &n in &self.moving {
            let node = &self.nodes[n as usize];
            let mut local = node.local.clone();
            if let Some(from) = node.from {
                interp::blend_look(from, &mut local, t);
            }
            let outer = match &node.outer {
                Some(outer) => outer,
                None => {
                    let parent = node.parent as usize;
                    self.drawn[parent]
                        .as_ref()
                        .unwrap_or(&self.nodes[parent].recorded)
                }
            };
            self.drawn[n as usize] = Some(outer.then(&local));
        }
        let drawn = Drawn {
            nodes: &self.nodes,
            drawn: &self.drawn,
            fresh: false,
        };
        run(&self.ops, &self.objects, &drawn, context)
    }
}

struct Drawn<'a> {
    nodes: &'a [Node],
    drawn: &'a [Option<Transform>],
    /// Lean render draws a branch as a walk would: a drawing is drawn by its object (which
    /// looks its tessellation up, and so keeps it), whatever its size.
    fresh: bool,
}

impl Drawn<'_> {
    fn get(&self, node: u32) -> &Transform {
        // (The list is empty until the program is first replayed.)
        self.drawn
            .get(node as usize)
            .and_then(Option::as_ref)
            .unwrap_or(&self.nodes[node as usize].recorded)
    }
}

fn with_transform(command: &Command, new: &Transform) -> Command {
    let mut command = command.clone();
    match &mut command {
        Command::RenderShape { transform, .. }
        | Command::RenderBitmap { transform, .. }
        | Command::RenderStage3D { transform, .. } => *transform = new.clone(),
        _ => {}
    }
    command
}

fn run<'gc>(
    ops: &[Op],
    objects: &[DisplayObject<'gc>],
    drawn: &Drawn<'_>,
    context: &mut RenderContext<'_, 'gc>,
) -> Result<(), String> {
    for op in ops {
        match op {
            Op::Same(command) => context.commands.commands.push(command.clone()),
            Op::Moved(command, node) => context
                .commands
                .commands
                .push(with_transform(command, drawn.get(*node))),
            Op::MovedScaled(command, node, measure) => {
                let transform = drawn.get(*node);
                if Measure::of(&transform.matrix) != *measure {
                    return Err("a drawing changed scale".to_string());
                }
                context
                    .commands
                    .commands
                    .push(with_transform(command, transform));
            }
            Op::Shape { node, source } => {
                let transform = drawn.get(*node);
                let movie = context
                    .library
                    .library_at(source.movie)
                    .map(|library| library.movie());
                if let Some(shape) = movie.and_then(|movie| {
                    Graphic::handle_for(
                        context,
                        movie,
                        &source.shape,
                        &source.cache,
                        Measure::of(&transform.matrix),
                    )
                }) {
                    context.commands.commands.push(Command::RenderShape {
                        shape,
                        transform: transform.clone(),
                    });
                }
            }
            Op::Drawing {
                command,
                node,
                measure,
                object,
            } => {
                let transform = drawn.get(*node);
                if !drawn.fresh && Measure::of(&transform.matrix) == *measure {
                    context
                        .commands
                        .commands
                        .push(with_transform(command, transform));
                } else {
                    context.transform_stack.push_absolute(transform);
                    match objects[*object as usize] {
                        DisplayObject::MovieClip(clip) => {
                            if let Some(drawing) = clip.drawing() {
                                drawing.render(context);
                            }
                        }
                        DisplayObject::Graphic(graphic) => graphic.render_drawing(context),
                        _ => {}
                    }
                    context.transform_stack.pop();
                }
            }
            Op::Island {
                object,
                node,
                options,
                maskers,
            } => {
                let saved = context.commands.maskers();
                context.commands.set_maskers(*maskers);
                context.transform_stack.push_absolute(drawn.get(*node));
                render_base(objects[*object as usize], context, options.clone());
                context.transform_stack.pop();
                context.commands.set_maskers(saved);
            }
            Op::Nest { ops, mode, maskers } => {
                let outer = std::mem::take(&mut context.commands);
                let result = run(ops, objects, drawn, context);
                let sub = std::mem::replace(&mut context.commands, outer);
                result?;
                if !sub.is_empty() {
                    let saved = context.commands.maskers();
                    context.commands.set_maskers(*maskers);
                    context.commands.blend(sub, mode.clone());
                    context.commands.set_maskers(saved);
                }
            }
        }
    }
    Ok(())
}

/// Lean render without frame interpolation: whether walks are recorded.
///
/// With interpolation a frame's walk is recorded anyway (the in-between pictures are drawn
/// from it), and drawing a branch from the last recording only saves. Without, recording is
/// what lean render adds to every walk: it pays when most of the stage stands still, and costs
/// where most of it moves (see the worker's notes for the rooms). So when a recorded walk
/// could draw less than two objects in three from the last recording, the next walks are
/// plain ones, 30 at first, up to 240 while it stays so; then two are recorded again (the
/// second can draw from the first) to see.
#[derive(Default)]
struct Pace {
    plain_left: u32,
    last_pause: u32,
}

impl Pace {
    /// Whether this picture's walk is a plain one.
    fn walk_plainly(&mut self) -> bool {
        if self.plain_left == 0 {
            return false;
        }
        self.plain_left -= 1;
        true
    }

    /// A recorded walk that had a last recording to draw from walked `walked` objects and
    /// drew `reused` from it.
    fn recorded(&mut self, walked: u64, reused: u64) {
        if reused < 2 * walked {
            self.last_pause = (self.last_pause * 2).clamp(30, 240);
            self.plain_left = self.last_pause;
        } else {
            self.last_pause = 0;
        }
    }
}

thread_local! {
    static PACE: RefCell<Pace> = const { RefCell::new(Pace { plain_left: 0, last_pause: 0 }) };
}

/// Where `Player::render` keeps the program between pictures.
#[derive(Collect, Default)]
#[collect(no_drop)]
pub struct Saved<'gc>(Option<Program<'gc>>);

/// How `Player::render` draws this picture.
pub enum Plan<'gc> {
    /// Walk as usual, without recording.
    Walk,
    /// Walk and record.
    Record(Recorder<'gc>),
    /// Replay the saved program.
    Replay(Program<'gc>),
}

impl<'gc> Saved<'gc> {
    /// `redraw_asked`: whether the movie asked to be drawn again since the last picture (a
    /// frame ran, `updateAfterEvent`, a button changed its state).
    ///
    /// Something that changes between two frames without that (a timer's handler, a network
    /// message) is drawn with the next frame's picture, which walks the stage anyway. Flash
    /// draws it then too: it only draws once per frame, or after `updateAfterEvent`. Walking
    /// again for the next in-between picture cost a whole walk per change: the 50 ms timer
    /// that pulses the glow of AQW's drops icon made 20 of them a second (245 walks in 5 s in
    /// a crowded town instead of 150, each ~5 ms). The program is still good to draw from:
    /// it draws the picture as it was, and what's drawn again in every picture (text, cached
    /// and masked objects, shape tweens: that glow too) is drawn as it is now.
    pub fn plan(&mut self, redraw_asked: bool) -> Plan<'gc> {
        let lean = lean_tracking();
        if !ENABLED.load(Ordering::Relaxed) || (interp::amount().is_none() && !lean) {
            self.0 = None;
            return Plan::Walk;
        }
        if interp::amount().is_none() {
            // Lean render without interpolation: every picture is a walk, recorded so the
            // next one can draw what hasn't changed from it, while that pays (see `Pace`).
            if !lean_verify() && !every_walk() && PACE.with(|pace| pace.borrow_mut().walk_plainly())
            {
                self.0 = None;
                return Plan::Walk;
            }
            return Plan::Record(Recorder::lean(self.0.take()));
        }
        let now = Key::now();
        let recorder = |program: Option<Program<'gc>>| {
            if lean {
                Recorder::lean(program)
            } else {
                Recorder::new()
            }
        };
        match self.0.take() {
            Some(program) if program.key == now && program.replayable => Plan::Replay(program),
            Some(program) => {
                if program.key.frame == now.frame && program.replayable {
                    if !redraw_asked && program.key.edge_mode == now.edge_mode && !eager() {
                        count(LEFT_FOR_NEXT_FRAME);
                        return Plan::Replay(program);
                    }
                    count(CHANGED_MID_FRAME);
                }
                Plan::Record(recorder(Some(program)))
            }
            None => Plan::Record(recorder(None)),
        }
    }

    /// Draws the picture with `program`, or walks (`walk`) if it can't.
    pub fn replay(
        &mut self,
        mut program: Program<'gc>,
        context: &mut RenderContext<'_, 'gc>,
        walk: impl FnOnce(&mut RenderContext<'_, 'gc>),
    ) {
        let t = interp::amount().unwrap_or(1.0);
        let outer = std::mem::take(&mut context.commands);
        let result = program.replay(context, t);
        let replayed = std::mem::replace(&mut context.commands, outer);
        match result {
            Ok(()) => {
                count(REPLAYED);
                context.commands = replayed;
            }
            Err(why) => {
                count(GIVEN_UP);
                note(why);
                walk(context);
            }
        }
        program.key = Key::now();
        self.0 = Some(program);
    }

    /// The recorded walk is done.
    pub fn recorded(&mut self, recorder: &mut Recorder<'gc>, commands: &CommandList) {
        // New tessellations are swapped in after the walk: record the next picture. (Lean
        // render keeps this recording all the same, to draw the next walk's unchanged
        // branches from: those with a stand-in in them are marked, see `Recorder::account`.)
        let replayable = !crate::display_object::graphic::deferred_shapes::has_jobs();
        if !replayable && !recorder.lean {
            return;
        }
        let result = recorder.finish(commands);
        if recorder.lean && interp::amount().is_none() && recorder.prev.is_some() {
            let (walked, _, reused) = recorder.counts;
            PACE.with(|pace| pace.borrow_mut().recorded(walked, reused));
        }
        recorder.recycle();
        match result {
            Ok(mut program) => {
                if replayable {
                    count(RECORDED);
                }
                program.replayable = replayable;
                program.key = Key::now();
                self.0 = Some(program);
            }
            Err(why) => {
                count(NOT_RECORDABLE);
                note(why);
            }
        }
    }
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::RenderBitmap { .. } => "bitmap",
        Command::RenderStage3D { .. } => "stage3d",
        Command::RenderShape { .. } => "shape",
        Command::RenderAlphaMask { .. } => "alpha mask",
        Command::DrawRect { .. } => "rect",
        Command::DrawLine { .. } => "line",
        Command::DrawLineRect { .. } => "line rect",
        Command::PushMask => "push mask",
        Command::ActivateMask => "activate mask",
        Command::DeactivateMask => "deactivate mask",
        Command::PopMask => "pop mask",
        Command::Blend(..) => "blend",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_object::{MovieClip, TDisplayObject};
    use crate::tag_utils::{SwfMovie, SwfMovieData};
    use gc_arena::Gc;
    use ruffle_render::backend::{ShapeHandle, ShapeHandleImpl};

    fn begin_recording(generation: u32) {
        GENERATION.with(|g| g.set(generation));
    }

    /// Lean render stands on this: a change marks the object and everything above it, so an
    /// object whose mark is older than a recording has nothing below it that changed since.
    #[test]
    fn a_change_marks_every_ancestor() {
        fn scenario<'gc>(mc: &gc_arena::Mutation<'gc>) {
            let movie = SwfMovie::new(mc, SwfMovieData::empty(10, None));
            let clip = || MovieClip::new(movie, mc);
            let (a, b, c, d, e) = (clip(), clip(), clip(), clip(), clip());
            let set_parent = |child: MovieClip<'gc>, parent: Option<MovieClip<'gc>>| {
                DisplayObjectBase::set_parent_ignoring_orphan_list(
                    Gc::write(mc, child.base()),
                    parent.map(Into::into),
                );
            };
            let marked = |clip: MovieClip<'gc>| clip.base().render_note.marked.get();
            let changed =
                |clip: MovieClip<'gc>, since: u32| clip.base().render_note.changed_since(since);
            // (Never older above than below.)
            let ordered = |chain: &[MovieClip<'gc>]| {
                chain
                    .windows(2)
                    .all(|pair| marked(pair[0]) >= marked(pair[1]))
            };

            // a > b > c, put together while recording 3 is the newest.
            begin_recording(3);
            set_parent(b, Some(a));
            set_parent(c, Some(b));
            assert_eq!((marked(a), marked(b), marked(c)), (3, 3, 3));

            // Recording 5 is made; nothing has changed since it was begun.
            begin_recording(5);
            assert!(!changed(a, 5) && !changed(b, 5) && !changed(c, 5));
            // The lowest changes: none of the three can be drawn from recording 5.
            mark_dirty(&c.base());
            assert!(changed(a, 5) && changed(b, 5) && changed(c, 5));

            // Recording 6 walks them. Then the middle one changes: it and the one above are
            // walked next time, the one below can still be drawn from recording 6.
            begin_recording(6);
            mark_dirty(&b.base());
            assert!(changed(a, 6) && changed(b, 6) && !changed(c, 6));
            // The lowest changes too. (The climb stops at b, marked already.)
            mark_dirty(&c.base());
            assert!(changed(c, 6));
            assert!(ordered(&[a, b, c]));

            // A new object, and one that changed long ago, join the lowest: everything above
            // them is marked.
            begin_recording(8);
            assert!(!changed(a, 8));
            set_parent(d, Some(c));
            assert!(changed(a, 8) && changed(b, 8) && changed(c, 8) && changed(d, 8));
            begin_recording(9);
            set_parent(e, Some(d));
            assert!(changed(a, 9) && changed(d, 9) && changed(e, 9));
            assert!(ordered(&[a, b, c, d, e]));

            // One leaves: the branch it leaves is marked (it draws one object less).
            begin_recording(10);
            assert!(!changed(a, 10));
            set_parent(e, None);
            assert!(changed(a, 10) && changed(d, 10) && changed(e, 10));
            // On its own, a change of it reaches nothing else.
            begin_recording(11);
            mark_dirty(&e.base());
            assert!(changed(e, 11) && !changed(d, 11) && !changed(a, 11));
        }
        force_lean_tracking(Some(true));
        gc_arena::arena::rootless_mutate(|mc| scenario(mc));
        force_lean_tracking(None);
    }

    /// With the switch off nothing is marked (the old path does no more than it did).
    #[test]
    fn nothing_is_marked_with_lean_render_off() {
        force_lean_tracking(Some(false));
        gc_arena::arena::rootless_mutate(|mc| {
            let movie = SwfMovie::new(mc, SwfMovieData::empty(10, None));
            let clip = MovieClip::new(movie, mc);
            begin_recording(4);
            mark_dirty(&clip.base());
            assert_eq!(clip.base().render_note.marked.get(), 0);
        });
        force_lean_tracking(None);
    }

    #[test]
    fn a_note_knows_its_node_only_in_its_recording() {
        let note = Note::default();
        note.stamp(7, 42);
        assert_eq!(note.node_in(7), Some(42));
        assert_eq!(note.node_in(6), None);
        assert_eq!(note.node_in(8), None);
        note.marked.set(7);
        assert!(note.changed_since(7) && !note.changed_since(8));
        // A copy (an instantiated symbol) starts without either.
        let copy = note.clone();
        assert_eq!(copy.node_in(7), None);
        assert!(!copy.changed_since(1));
    }

    /// Without interpolation, walks are recorded only while most of the stage can be drawn
    /// from the last recording.
    #[test]
    fn recording_pauses_where_most_of_the_stage_moves() {
        let mut pace = Pace::default();
        assert!(!pace.walk_plainly());
        // Two objects in three drawn from the last recording: it goes on.
        pace.recorded(1000, 2000);
        assert!(!pace.walk_plainly());
        // Fewer: 30 plain walks, then recorded ones again.
        pace.recorded(1000, 1999);
        assert_eq!((0..40).filter(|_| pace.walk_plainly()).count(), 30);
        // Still so: 60, 120, 240, 240.
        for pause in [60, 120, 240, 240] {
            pace.recorded(1000, 0);
            assert_eq!((0..300).filter(|_| pace.walk_plainly()).count(), pause);
        }
        // The stage has come to rest: recorded from now on, and the next pause is short again.
        pace.recorded(10, 5000);
        assert!(!pace.walk_plainly());
        pace.recorded(1000, 0);
        assert_eq!((0..300).filter(|_| pace.walk_plainly()).count(), 30);
    }

    #[derive(Debug)]
    struct Tessellation;
    impl ShapeHandleImpl for Tessellation {}

    fn shape(handle: &ShapeHandle, x: f32) -> Command {
        Command::RenderShape {
            shape: handle.clone(),
            transform: Transform {
                matrix: Matrix::scale(x, 1.0),
                ..Default::default()
            },
        }
    }

    /// The check's comparison: what a branch drawn from the last recording must share with
    /// its walk.
    #[test]
    fn the_check_compares_tessellations_transforms_and_nested_lists() {
        let one = ShapeHandle(Arc::new(Tessellation));
        let other = ShapeHandle(Arc::new(Tessellation));

        let walked = vec![Command::PushMask, shape(&one, 1.0), Command::ActivateMask];
        assert_eq!(difference(&walked.clone(), &walked), None);
        // Another tessellation, another transform, one command less.
        let reused = vec![Command::PushMask, shape(&other, 1.0), Command::ActivateMask];
        assert!(difference(&reused, &walked).unwrap().contains("command 1"));
        let reused = vec![Command::PushMask, shape(&one, 2.0), Command::ActivateMask];
        assert!(
            difference(&reused, &walked)
                .unwrap()
                .contains("the same transform: false")
        );
        assert!(
            difference(&walked[..2], &walked)
                .unwrap()
                .contains("2 commands there and 3")
        );
        // A transform that isn't a number is the same as itself.
        let lost = vec![shape(&one, f32::NAN)];
        assert_eq!(difference(&lost.clone(), &lost), None);
        // Inside a blend.
        let blend = |inner: Vec<Command>| {
            let mut list = CommandList::new();
            list.commands = inner;
            vec![Command::Blend(
                list,
                RenderBlendMode::Builtin(swf::BlendMode::Add),
            )]
        };
        assert_eq!(
            difference(
                &blend(vec![shape(&one, 1.0)]),
                &blend(vec![shape(&one, 1.0)])
            ),
            None
        );
        assert!(
            difference(
                &blend(vec![shape(&other, 1.0)]),
                &blend(vec![shape(&one, 1.0)])
            )
            .is_some()
        );
    }
}
