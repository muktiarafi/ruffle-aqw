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
use crate::display_object::{DisplayObject, RenderOptions, TDisplayObject, interp, render_base};
use crate::tag_utils::SwfMovie;
use crate::tessellation_cache::{Measure, TessellationCache};
use gc_arena::Collect;
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

/// Turu: swaps one player's change count with the thread's (see `crate::PlayerThreadState`).
pub(crate) fn swap_changes(changes: &mut u64) {
    *changes = CHANGES.replace(*changes);
}

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Replays on (default) or off (every picture walks the stage, for comparisons).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// For the perf log: pictures recorded, replayed, walked because recording wasn't possible,
/// replays given up (walked instead), and recordings redone between two frames as something
/// visible changed.
pub const STAT_NAMES: [&str; 5] = [
    "recorded",
    "replayed",
    "not recordable",
    "given up",
    "changed mid-frame",
];
const RECORDED: usize = 0;
const REPLAYED: usize = 1;
const NOT_RECORDABLE: usize = 2;
const GIVEN_UP: usize = 3;
const CHANGED_MID_FRAME: usize = 4;
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static LAST_REASON: Mutex<Option<String>> = Mutex::new(None);

fn count(stat: usize) {
    STATS[stat].fetch_add(1, Ordering::Relaxed);
}

fn note(reason: String) {
    if let Ok(mut last) = LAST_REASON.lock() {
        *last = Some(reason);
    }
}

pub fn stats() -> [u64; 5] {
    std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed))
}

/// Why the last picture couldn't be recorded or replayed.
pub fn take_reason() -> Option<String> {
    LAST_REASON.lock().ok().and_then(|mut last| last.take())
}

/// An object's transform in the recorded picture.
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
}

/// Where a graphic's tessellations come from (see `Graphic::handle_for`).
pub(crate) struct ShapeSource {
    movie: Arc<SwfMovie>,
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
    Shape { node: u32, source: Box<ShapeSource> },
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
        let node = self.path.last().copied().filter(|&n| self.nodes[n as usize].moving);
        let mut failed = None;
        for command in &commands.commands[list.accounted.min(commands.commands.len())..] {
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
    pub(crate) fn push_node(
        &mut self,
        commands: &CommandList,
        stack: &TransformStack,
        real: Transform,
        from: Option<(Matrix, ColorTransform)>,
        drawn: &Transform,
    ) -> bool {
        if !self.recording() {
            return false;
        }
        self.account(commands);
        let parent = self.path.last().copied().unwrap_or(NONE);
        let outer = stack.transform();
        let recorded = outer.then(drawn);
        let changes = from.is_some_and(|(matrix, color)| {
            matrix != real.matrix || color != real.color_transform
        });
        let moving = changes || (parent != NONE && self.nodes[parent as usize].moving);
        self.path.push(self.nodes.len() as u32);
        self.nodes.push(Node {
            parent,
            outer: (parent == NONE).then(|| Box::new(outer)),
            local: real,
            from,
            moving,
            recorded,
        });
        true
    }

    pub(crate) fn pop_node(&mut self, commands: &CommandList) {
        self.account(commands);
        self.path.pop();
    }

    /// A graphic just drew its shape (the last command, if it was added).
    pub(crate) fn shape(
        &mut self,
        commands: &CommandList,
        movie: Arc<SwfMovie>,
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
            source: Box::new(ShapeSource {
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
        let is_it = drawn.moving
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
        self.lists.push(List::default());
        true
    }

    /// The nested list is done (`sub`), and back in the current one (`commands`, before the
    /// blend is added to it).
    pub(crate) fn nest_close(&mut self, sub: &CommandList) -> Vec<Op> {
        self.account(sub);
        self.lists.pop().map(|list| list.ops).unwrap_or_default()
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
        let moving = (0..nodes.len() as u32)
            .filter(|&n| nodes[n as usize].moving)
            .collect();
        Ok(Program {
            objects: std::mem::take(&mut self.objects),
            ops: std::mem::take(&mut self.lists[0].ops),
            drawn: vec![None; nodes.len()],
            nodes,
            moving,
            key: Key::default(),
        })
    }
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
}

impl<'gc> Program<'gc> {
    /// Draws the picture `t` of the way between the previous frame and this one into
    /// `context.commands`. On an error, the commands are incomplete (walk instead).
    fn replay(&mut self, context: &mut RenderContext<'_, 'gc>, t: f32) -> Result<(), String> {
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
        };
        run(&self.ops, &self.objects, &drawn, context)
    }
}

struct Drawn<'a> {
    nodes: &'a [Node],
    drawn: &'a [Option<Transform>],
}

impl Drawn<'_> {
    fn get(&self, node: u32) -> &Transform {
        self.drawn[node as usize]
            .as_ref()
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
                context.commands.commands.push(with_transform(command, transform));
            }
            Op::Shape { node, source } => {
                let transform = drawn.get(*node);
                if let Some(shape) = Graphic::handle_for(
                    context,
                    &source.movie,
                    &source.shape,
                    &source.cache,
                    Measure::of(&transform.matrix),
                ) {
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
                if Measure::of(&transform.matrix) == *measure {
                    context.commands.commands.push(with_transform(command, transform));
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
    pub fn plan(&mut self) -> Plan<'gc> {
        if interp::amount().is_none() || !ENABLED.load(Ordering::Relaxed) {
            self.0 = None;
            return Plan::Walk;
        }
        let now = Key::now();
        match self.0.take() {
            Some(program) if program.key == now => Plan::Replay(program),
            Some(program) => {
                if program.key.frame == now.frame {
                    count(CHANGED_MID_FRAME);
                }
                Plan::Record(Recorder::new())
            }
            None => Plan::Record(Recorder::new()),
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
        if crate::display_object::graphic::deferred_shapes::has_jobs() {
            // New tessellations are swapped in after the walk: record the next picture.
            return;
        }
        match recorder.finish(commands) {
            Ok(mut program) => {
                count(RECORDED);
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
