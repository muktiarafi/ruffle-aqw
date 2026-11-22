//! Frame events management
//!
//! This module aids in keeping track of which frame execution phase we are in.
//!
//! For AVM2 code, display objects execute a series of discrete phases, and
//! each object is notified about the current frame phase in rendering order.
//! When objects are created, they are 'caught up' to the current frame phase
//! to ensure correct order of operations.
//!
//! AVM1 code (presumably, either on an AVM1 stage or within an `AVM1Movie`)
//! runs in one phase, with timeline operations executing with all phases
//! inline in the order that clips were originally created.

use crate::avm2::{Avm2, EventObject};
use crate::context::UpdateContext;
use crate::display_object::{DisplayObject, MovieClip, TDisplayObject};
use crate::loader::LoadManager;
use crate::orphan_manager::OrphanManager;
use tracing::instrument;

/// Which phase of the frame we're currently in.
///
/// AVM2 frames exist in one of four phases: `Enter`, `Construct`,
/// `FrameScripts`, or `Exit`. An additional `Idle` phase covers rendering and
/// event processing.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FramePhase {
    /// We're entering the next frame.
    ///
    /// When movie clips enter a new frame, they must do two things:
    ///
    ///  - Remove all children that should not exist on the next frame.
    ///  - Increment their current frame number.
    ///
    /// Once this phase ends, we fire `enterFrame` on the broadcast list.
    Enter,

    /// We're constructing children of existing display objects.
    ///
    /// All `PlaceObject` tags should execute at this time.
    ///
    /// Once we construct the frame, we fire `frameConstructed` on the
    /// broadcast list.
    Construct,

    /// We're running all queued frame scripts.
    ///
    /// Frame scripts are the AS3 equivalent of old-style `DoAction` tags. They
    /// are queued in the `Update` phase if the current timeline frame number
    /// differs from the prior frame's one.
    FrameScripts,

    /// We're finishing frame processing.
    ///
    /// When we exit a completed frame, we fire `exitFrame` on the broadcast
    /// list.
    Exit,

    /// We're not currently executing any frame code.
    ///
    /// At this point in time, event handlers are expected to run. No frame
    /// catch-up work should execute.
    #[default]
    Idle,
}

/// Run one frame according to AVM2 frame order.
/// NOTE: The `each_orphan_movie` calls are in really odd places,
/// but this is needed to match Flash Player's output. There may
/// still be lurking bugs, but the current code matches Flash's
/// output exactly for two complex test cases (see `avm2/orphan_movie*`)
#[instrument(level = "debug", skip_all)]
pub fn run_all_phases_avm2(context: &mut UpdateContext<'_>) {
    let stage = context.stage;
    // AQW build: settled marks only last for one frame (see `display_object::settle`).
    crate::display_object::settle::new_frame();

    if !stage.movie().is_action_script_3() {
        return;
    }

    *context.frame_phase = FramePhase::Enter;
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        orphan.enter_frame(context);
    });
    stage.enter_frame(context);

    *context.frame_phase = FramePhase::Construct;
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        orphan.construct_frame(context);
    });
    stage.construct_frame(context);
    broadcast_frame_constructed(context);

    *context.frame_phase = FramePhase::FrameScripts;
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        orphan.run_frame_scripts(context);
    });
    stage.run_frame_scripts(context);
    run_frame_script_cleanup(context);

    *context.frame_phase = FramePhase::Exit;
    broadcast_frame_exited(context);

    // The correct time to run context3DCreated events seems to be here
    stage.check_requested_context3ds(context);

    // We cannot easily remove dead `GcWeak` instances from the orphan list
    // inside `each_orphan_movie`, since the callback may modify the orphan list.
    // Instead, we do one cleanup at the end of the frame.
    // This performs special handling of clips which became orphaned as
    // a result of a RemoveObject tag - see `cleanup_dead_orphans` for details.
    context.orphan_manager.cleanup_dead_orphans(context.gc());

    *context.frame_phase = FramePhase::Idle;
    // AQW build: what the lean walks skipped (see `display_object::settle`).
    crate::display_object::settle::frame_done();
    crate::display_object::replay::lean_frame_done();
}

/// Like `run_all_phases_avm2`, but specialized for the "nested frame" triggered
/// by a goto. This is different enough to not be worth combining into a single
/// method with extra parameters.
///
/// During a goto, we run frame construction, framescripts, and frame exits for the *entire stage*.
/// This even extends to orphans - for example, calling `gotoAndStop` on an orphan will
/// cause frame construction to get run for the *current frame* of other objects on the timeline
/// (even if the goto was called from an enterFrame event handler).
pub fn run_inner_goto_frame<'gc>(
    context: &mut UpdateContext<'gc>,
    removed_frame_scripts: &[DisplayObject<'gc>],
    initial_clip: MovieClip<'gc>,
) {
    if initial_clip.swf_version() <= 9 && initial_clip.movie().is_action_script_3() {
        // We skip the next `enter_frame` call, so that we will still run the framescripts
        // queued for our target frame.
        initial_clip.base().set_skip_next_enter_frame(true);
        // AQW build: `enter_frame` has to come by again (see `display_object::settle`).
        initial_clip.wake();

        return;
    }

    let stage = context.stage;
    let old_phase = *context.frame_phase;

    // When performing goto, frame scripts behave the same as when entering a new frame
    // so no separate cleanup is performed on ones registered during frame script phase
    context.frame_script_cleanup_queue.clear();

    // Note - we do *not* call `enter_frame` or dispatch an `enterFrame` event

    *context.frame_phase = FramePhase::Construct;
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        orphan.construct_frame(context);
    });
    stage.construct_frame(context);
    broadcast_frame_constructed(context);

    *context.frame_phase = FramePhase::FrameScripts;
    stage.run_frame_scripts(context);
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        orphan.run_frame_scripts(context);
    });

    for child in removed_frame_scripts {
        child.run_frame_scripts(context);
    }

    *context.frame_phase = FramePhase::Exit;
    broadcast_frame_exited(context);

    // We cannot easily remove dead `GcWeak` instances from the orphan list
    // inside `each_orphan_movie`, since the callback may modify the orphan list.
    // Instead, we do one cleanup at the end of the frame.
    // This performs special handling of clips which became orphaned as
    // a result of a RemoveObject tag - see `cleanup_dead_orphans` for details.
    context.orphan_manager.cleanup_dead_orphans(context.gc());

    *context.frame_phase = old_phase;
}

/// Broadcast a `enterFrame` event to all `DisplayObject`s.
pub fn broadcast_frame_entered<'gc>(context: &mut UpdateContext<'gc>) {
    let enter_frame_evt = EventObject::bare_default_event(context, "enterFrame");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, enter_frame_evt, dobject_constr);
}

/// Broadcast a `frameConstructed` event to all `DisplayObject`s.
pub fn broadcast_frame_constructed<'gc>(context: &mut UpdateContext<'gc>) {
    let frame_constructed_evt = EventObject::bare_default_event(context, "frameConstructed");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, frame_constructed_evt, dobject_constr);
}

/// Broadcast a `exitFrame` event to all `DisplayObject`s.
pub fn broadcast_frame_exited<'gc>(context: &mut UpdateContext<'gc>) {
    let exit_frame_evt = EventObject::bare_default_event(context, "exitFrame");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, exit_frame_evt, dobject_constr);

    LoadManager::run_exit_frame(context);
}

/// Empty the `context.frame_script_cleanup_queue` by running frame scripts for
/// each clip in the queue.
fn run_frame_script_cleanup<'gc>(context: &mut UpdateContext<'gc>) {
    while let Some(clip) = context.frame_script_cleanup_queue.pop_front() {
        clip.set_has_pending_script(true);
        clip.set_last_queued_script_frame(None);
        clip.run_local_frame_scripts(context);
    }
}

/// AQW build: the frame scripts of clips put on the stage between frames don't wait for the
/// next frame.
///
/// Flash only draws after a frame. What an event between two frames puts on the stage (a
/// timer, data from the server, a finished load, a click) has run its first frame's scripts
/// by the time anybody sees it. Here a picture is drawn after every tick, and the tick handles
/// those events after its frame (`Player::tick`), and with a fast screen there are many
/// pictures between two frames: a clip made by such an event was drawn before its first frame
/// script had run. AQW colours every colour-custom part of an avatar from that part's own
/// first-frame script (`mcSetColor(this, "Trim", ...)`), so armour, hair, capes and helms
/// showed in the colours they were drawn in, for up to a frame, whenever the game built them
/// outside a frame (its load throttle is a timer).
///
/// So: once a clip was given a parent between frames, the frame scripts waiting on the stage
/// are run when that event's code has returned (`Player::update`, and before a picture is
/// drawn), before any picture. It is what a `gotoAndPlay` anywhere in that same code has
/// always done (`run_inner_goto_frame` runs the whole stage's waiting frame scripts, as Flash
/// does); a script that has run is not run again by the next frame. Clips that are not on the
/// stage keep waiting, as before: their scripts mostly ask for the stage.
///
/// It is NOT the order Flash runs them in: Flash runs such a script in the next frame, after
/// that frame's `enterFrame` and `frameConstructed` (the test `avm2/avm2_catchup_dobj` is
/// Flash's own trace of it; `from_shumway/as3-loader/events/loader-events` shows Flash
/// building a loaded movie inside a frame for the same reason). Running the next frame's
/// script phase early is the price of drawing between frames at all. So it is off unless
/// `RUFFLE_EARLY_SCRIPTS` is set: the desktop player sets it when AQW is played (a
/// `no-early-scripts` file next to the exe, or `safe-drawing`, leaves it off), and the tests
/// and other movies run in Flash's order.
pub mod early_scripts {
    use std::cell::Cell;

    thread_local! {
        /// A display object was given a parent, or a frame script was set for a clip's
        /// frame, at a time when the frame's own script phase will not come by it any more
        /// (see `changed`).
        static PARENTED: Cell<bool> = const { Cell::new(false) };
        /// How often that mark was set, by the phase it was set in (Enter, Construct,
        /// FrameScripts, Exit, Idle).
        static MARKS: Cell<[u64; 5]> = const { Cell::new([0; 5]) };
        /// Lines written about clips drawn before their script (see `drawn_waiting`).
        static LINES: Cell<u64> = const { Cell::new(0) };
        static RUNNING: Cell<bool> = const { Cell::new(false) };
        /// (runs that found a script, scripts run in them, clips drawn with their script
        /// still waiting)
        static COUNTS: Cell<(u64, u64, u64)> = const { Cell::new((0, 0, 0)) };
        /// Every frame script run, early or not (for the offline proof in `aqw_bench`).
        static ALL_SCRIPTS: Cell<u64> = const { Cell::new(0) };
        /// (walks of the stage for waiting scripts, microseconds they took)
        static WALKS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    /// For the log: what looking for waiting scripts costs: (walks of the stage, microseconds
    /// in all), so far.
    pub fn walks() -> (u64, u64) {
        WALKS.get()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn all_scripts() -> u64 {
        ALL_SCRIPTS.get()
    }

    /// Whether `run` has something to look at (asked before a picture is drawn: some code
    /// between frames does not come through `Player::update`, a finished load for one).
    pub fn waiting() -> bool {
        PARENTED.get() && enabled()
    }

    pub fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_EARLY_SCRIPTS").is_some())
    }

    /// A display object was given a parent, or a clip a script for the frame it is on. In
    /// the first two phases of a frame the frame's script phase is still to come and walks
    /// the whole stage. After that it may not come by any more: a script that makes clips
    /// during the script phase (often inside the nested frame of a `gotoAndPlay`), an
    /// `exitFrame` listener, and everything between frames.
    #[inline]
    pub fn changed(phase: super::FramePhase) {
        use super::FramePhase;
        if !matches!(phase, FramePhase::Enter | FramePhase::Construct) || RUNNING.get() {
            if !PARENTED.replace(true) {
                let index = match phase {
                    FramePhase::Enter => 0,
                    FramePhase::Construct => 1,
                    FramePhase::FrameScripts => 2,
                    FramePhase::Exit => 3,
                    FramePhase::Idle => 4,
                };
                let mut marks = MARKS.get();
                marks[index] += 1;
                MARKS.set(marks);
            }
        }
    }

    /// For the log: how often the mark was set first in the script phase, the exit phase and
    /// between frames.
    pub fn marks() -> (u64, u64, u64) {
        let marks = MARKS.get();
        (marks[2], marks[3], marks[4])
    }

    /// A frame script is about to run.
    #[inline]
    pub fn script() {
        ALL_SCRIPTS.set(ALL_SCRIPTS.get() + 1);
        if RUNNING.get() {
            COUNTS.with(|c| {
                let (runs, scripts, drawn) = c.get();
                c.set((runs, scripts + 1, drawn));
            });
        }
    }

    /// A clip is drawn while a frame script of its current frame has yet to run. Returns
    /// whether to write a line about it: the first sixty, then one in two thousand.
    pub fn drawn_waiting() -> bool {
        let drawn = COUNTS.with(|c| {
            let (runs, scripts, drawn) = c.get();
            c.set((runs, scripts, drawn + 1));
            drawn + 1
        });
        if !enabled() {
            return false;
        }
        let lines = LINES.get();
        if lines < 60 || drawn % 2000 == 0 {
            LINES.set(lines + 1);
            true
        } else {
            false
        }
    }

    /// Whether the mark is set (for those lines).
    pub fn marked() -> bool {
        PARENTED.get()
    }

    /// For the log: (runs that found a script, scripts run early, clips drawn with their
    /// script still waiting), so far.
    pub fn counts() -> (u64, u64, u64) {
        COUNTS.get()
    }

    /// Called when code that ran between frames has returned.
    pub fn run(context: &mut crate::context::UpdateContext<'_>) {
        use super::FramePhase;
        if *context.frame_phase != FramePhase::Idle || !PARENTED.replace(false) || !enabled() {
            return;
        }
        let stage = context.stage;
        if !crate::display_object::TDisplayObject::movie(stage).is_action_script_3() {
            return;
        }
        let before = COUNTS.get().1;
        let started = std::time::Instant::now();
        let mut walks = 0;
        RUNNING.set(true);
        // (A script run here can put more clips on the stage: a few rounds, and what is left
        // then waits for the frame.)
        for _ in 0..4 {
            walks += 1;
            *context.frame_phase = FramePhase::FrameScripts;
            crate::display_object::TDisplayObject::run_frame_scripts(stage, context);
            super::run_frame_script_cleanup(context);
            *context.frame_phase = FramePhase::Idle;
            if !PARENTED.replace(false) {
                break;
            }
        }
        RUNNING.set(false);
        PARENTED.set(false);
        let (all_walks, micros) = WALKS.get();
        WALKS.set((
            all_walks + walks,
            micros + started.elapsed().as_micros() as u64,
        ));
        if COUNTS.get().1 != before {
            COUNTS.with(|c| {
                let (runs, scripts, drawn) = c.get();
                c.set((runs + 1, scripts, drawn));
            });
        }
    }
}

/// Run all previously-executed frame phases on a newly-constructed display
/// object.
///
/// This is a no-op on AVM1, which has it's own catch-up logic.
pub fn catchup_display_object_to_frame<'gc>(
    context: &mut UpdateContext<'gc>,
    dobj: DisplayObject<'gc>,
) {
    if !dobj.movie().is_action_script_3() {
        return;
    }

    match *context.frame_phase {
        FramePhase::Enter => {
            dobj.enter_frame(context);
        }
        FramePhase::Construct | FramePhase::FrameScripts | FramePhase::Exit | FramePhase::Idle => {
            dobj.enter_frame(context);
            dobj.construct_frame(context);
        }
    }
}
