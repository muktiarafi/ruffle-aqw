use crate::avm1::Object as Avm1Object;
use crate::avm2::StageObject as Avm2StageObject;
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::graphic::deferred_shapes;
use crate::display_object::interp::TweenRatio;
use crate::display_object::{BoundsMode, DisplayObjectBase};
use crate::library::MovieLibrarySource;
use crate::prelude::*;
use crate::tag_utils::SwfMovie;
use crate::tessellation_cache::{MadeFor, Measure, TessellationCache};
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::ShapeHandle;
use ruffle_render::commands::CommandHandler;
use std::cell::{Cell, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;
use swf::{Fixed8, Fixed16};
use web_time::Instant;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct MorphShape<'gc>(Gc<'gc, MorphShapeData<'gc>>);

impl fmt::Debug for MorphShape<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MorphShape")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct MorphShapeData<'gc> {
    base: DisplayObjectBase<'gc>,
    shared: Lock<Gc<'gc, MorphShapeShared<'gc>>>,
    /// The AVM2 representation of this MorphShape.
    object: Lock<Option<Avm2StageObject<'gc>>>,
    /// AQW build: for a very detailed tween (see `MorphShapeShared::get_heavy_shape`).
    #[collect(require_static)]
    warming: RefCell<Warming>,
}

/// AQW build: an instance of a very detailed tween, while frames of it are being tessellated
/// in the background (see `MorphShapeShared::get_heavy_shape`).
#[derive(Clone, Default)]
struct Warming {
    /// The tessellation it was last drawn with.
    last: Option<ShapeHandle>,
    /// The frames it asked for that weren't ready, oldest first: each one's tessellations,
    /// and the size it was asked for.
    pending: Vec<(Rc<RefCell<TessellationCache>>, Measure)>,
}

impl<'gc> MorphShape<'gc> {
    /// AQW build: the shared data that every instance of this library symbol points to, used
    /// to tell whether the symbol (and so its movie's library) is still in use.
    pub fn aqw_shared(self) -> Gc<'gc, ()> {
        Gc::erase(self.0.shared.get())
    }

    pub fn from_swf_tag(
        gc_context: &Mutation<'gc>,
        tag: swf::DefineMorphShape,
        movie: SwfMovie<'gc>,
    ) -> Self {
        let shared = MorphShapeShared::from_swf_tag(&tag, movie);
        MorphShape(Gc::new(
            gc_context,
            MorphShapeData {
                base: Default::default(),
                shared: Lock::new(Gc::new(gc_context, shared)),
                object: Lock::new(None),
                warming: Default::default(),
            },
        ))
    }

    pub fn instantiate(self, gc_context: &Mutation<'gc>) -> Self {
        Self(Gc::new(gc_context, (*self.0).clone()))
    }
}

impl<'gc> TDisplayObject<'gc> for MorphShape<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn id(self) -> CharacterId {
        self.0.shared.get().id
    }

    fn replace_with(self, context: &mut UpdateContext<'gc>, id: CharacterId) {
        if let Some(new_morph_shape) = context
            .library
            .library_for_movie_mut(self.movie())
            .get_morph_shape(id)
        {
            unlock!(Gc::write(context.gc(), self.0), MorphShapeData, shared)
                .set(new_morph_shape.0.shared.get())
        } else {
            tracing::warn!("PlaceObject: expected morph shape at character ID {}", id);
        }
        self.invalidate_cached_bitmap();
    }

    fn object1(self) -> Option<crate::avm1::Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), MorphShapeData, object).set(Some(to))
    }

    /// Construct objects placed on this frame.
    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        if self.movie().is_action_script_3() && self.object2().is_none() {
            let class = context.avm2.classes().morphshape;
            let object = Avm2StageObject::for_display_object(context.gc(), self.into(), class);
            // We don't need to call the initializer method, as AVM2 can't link
            // a custom class to a MorphShape, and the initializer method for
            // MorphShape itself is a no-op
            self.set_object2(context, object);

            self.on_construction_complete(context);
        }
    }

    fn render_self(self, context: &mut RenderContext<'_, 'gc>) {
        let ratio = self.ratio();
        let shared = self.0.shared.get();
        // AQW build: with frame interpolation, drawn between the previous and current ratio,
        // unless that's too slow to tessellate (see `MorphShapeShared::get_blended_shape`). A
        // tween that changes little per frame gets in-betweens by how far it moves on screen,
        // if it's a simple one (see `MorphShapeShared::travel_px_if_simple`).
        let drawn = self.base().interp.ratio(ratio, &shared.rhythm, || {
            let matrix = &context.transform_stack.transform().matrix;
            shared.travel_px_if_simple() * Measure::of(matrix).scale
        });
        if trace_tweens() {
            // (Testing: where each instance is drawn in each picture, to measure how evenly
            // it moves; see tools/exit-tests in the Mugetsu folder.)
            tracing::info!(
                target: "ruffle_aqw_perf",
                "tween trace: {} {:p} frame {} at {:?} real {} drawn {:?}",
                shared.id,
                Gc::as_ptr(self.0),
                crate::display_object::interp::frame(),
                crate::display_object::interp::amount(),
                ratio,
                drawn,
            );
        }
        // AQW build: with Flash-style edges, tessellated at the size it's drawn (see
        // `Graphic`), so its soft edges are half a pixel wide on screen.
        let scale = if ruffle_render::tessellator::edge_aa() {
            // Without a stretch: morphs animate (see `TessellationCache::find_near_and_touch`).
            let measure = Measure::of(&context.transform_stack.transform().matrix);
            Measure {
                stretch: [1.0, 0.0, 1.0, measure.stretch[3]],
                ..measure
            }
        } else {
            Measure {
                scale: 1.0,
                stretch: ruffle_render::backend::NO_STRETCH,
            }
        };
        let (real, blended) = match drawn {
            TweenRatio::Real(ratio) => (ratio, None),
            TweenRatio::Between { ratio, previous } => {
                (previous, shared.get_blended_shape(context, ratio, scale))
            }
        };
        let shape_handle = match blended {
            Some(handle) => handle,
            None if shared.heavy() => shared.get_heavy_shape(context, &self.0.warming, real, scale),
            None => shared.get_shape(context, real, scale),
        };
        context
            .commands
            .render_shape(shape_handle, context.transform_stack.transform());
    }

    fn self_bounds(self, mode: BoundsMode) -> Rectangle<Twips> {
        let shared = self.0.shared.get();

        match mode {
            // For engine bounds, use the actual interpolated ratio
            BoundsMode::Engine => {
                let ratio = self.ratio();
                let frame = shared.get_frame(ratio);
                frame.bounds
            }

            // For getBounds() or hitTestObject(), use the `shape_bounds` that
            // were declared to be the bounds of the start shape
            BoundsMode::Script => shared.start.shape_bounds,

            // For getRect(), use the `edge_bounds` that were declared to be the
            // bounds of the start shape
            BoundsMode::ScriptWithoutStrokes => shared.start.edge_bounds,
        }
    }

    fn hit_test_shape(
        self,
        _context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        options: HitTestOptions,
    ) -> bool {
        if (!options.contains(HitTestOptions::SKIP_INVISIBLE) || self.visible())
            && self.world_bounds(BoundsMode::Engine).contains(point)
        {
            if let Some(frame) = self.0.shared.get().frames.borrow().get(&self.ratio()) {
                let Some(local_matrix) = self.global_to_local_matrix() else {
                    return false;
                };
                return ruffle_render::shape_utils::shape_hit_test(
                    &frame.shape,
                    local_matrix * point,
                    &local_matrix,
                );
            } else {
                tracing::warn!("Missing ratio for morph shape");
            }
        }

        false
    }

    fn post_instantiation(
        self,
        context: &mut UpdateContext<'gc>,
        _init_object: Option<Avm1Object<'gc>>,
        _instantiated_by: Instantiator,
        _run_frame: bool,
    ) {
        if self.movie().is_action_script_3() {
            self.set_default_instance_name(context);
        }
    }

    fn movie(self) -> SwfMovie<'gc> {
        self.0.shared.get().movie
    }
}

/// A precalculated intermediate frame for a morph shape.
/// AQW build: how many sizes a shape tween's frame keeps tessellations for (see `Frame`).
const SIZES_KEPT: usize = 4;

/// AQW build: an in-between ratio of frame interpolation that takes longer than this (ms) to
/// tessellate is slow. `SLOW_IN_BETWEENS` in a row (one could be the thread being interrupted)
/// and the tween is drawn at its real ratios only (see `MorphShapeShared::get_blended_shape`).
const SLOW_IN_BETWEEN_MS: f64 = 2.0;
const SLOW_IN_BETWEENS: u8 = 2;

/// AQW build: a real ratio that takes longer than this (ms) to tessellate is logged, once per
/// tween.
const SLOW_FRAME_MS: f64 = 20.0;

/// AQW build: a tween of more records than this, or one a frame of which took longer than
/// `HEAVY_FRAME_MS` to tessellate, is heavy: its frames are tessellated in the background (see
/// `MorphShapeShared::get_heavy_shape`). Records alone say little about the time (a staff's
/// tween of 3,139 took 3 ms a frame, a cape's of 4,629 took 22-32), but this way the first
/// frame of a very detailed one doesn't have to be waited for to find out.
const HEAVY_TWEEN_RECORDS: usize = 1000;
const HEAVY_FRAME_MS: f64 = 8.0;

/// AQW build: how many frames an instance of a heavy tween remembers asking for (see
/// `Warming`).
const WARMING_KEPT: usize = 8;

/// AQW build: only a tween of at most this many records gets in-betweens by how far it moves
/// on screen (see `MorphShapeShared::travel_px_if_simple`): interface wipes, bars and timers
/// (AQW's have 4 to 17; a 4-edge wipe takes 0.004 ms to tessellate). Those are up to 8
/// in-betweens per frame for every copy on screen, and a crowd has slow tweens of hundreds of
/// records in its weapons and effects too: those keep the steps their speed alone gives.
/// (By size, not by timing each tessellation: two slow measurements on a busy PC would have
/// switched a tween's in-betweens off for good.)
const SIMPLE_TWEEN_RECORDS: usize = 32;

/// AQW build: how many in-between tessellations of one tween are kept at most (see
/// `MorphShapeShared::blended`).
const IN_BETWEENS_KEPT: usize = 64;

struct Frame {
    /// (AQW build: with what it was tessellated for; the last used last. Up to
    /// `SIZES_KEPT`: copies of a tween drawn at different sizes, e.g. an effect spawning many,
    /// tessellated it again for each draw when only one was kept.)
    shape_handles: Vec<(MadeFor, ShapeHandle)>,
    shape: Arc<swf::Shape>,
    bounds: Rectangle<Twips>,
    /// AQW build: a heavy tween's tessellations of this frame, instead of `shape_handles`
    /// (see `MorphShapeShared::get_heavy_shape`).
    cache: Option<Rc<RefCell<TessellationCache>>>,
}

/// AQW build: `RUFFLE_TRACE_TWEENS` (testing) logs where every shape tween is drawn, in every
/// picture.
fn trace_tweens() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RUFFLE_TRACE_TWEENS").is_some())
}

/// Data shared between all instances of a morph shape.
#[derive(Collect)]
#[collect(no_drop)]
pub struct MorphShapeShared<'gc> {
    movie: SwfMovie<'gc>,
    id: CharacterId,

    #[collect(require_static)]
    start: swf::MorphShape,

    #[collect(require_static)]
    end: swf::MorphShape,

    #[collect(require_static)]
    frames: RefCell<fnv::FnvHashMap<u16, Frame>>,
    /// AQW build: the in-between ratios drawn with frame interpolation in this SWF frame or
    /// the one before (with the frame each was last drawn in), most recent last. Kept apart
    /// from `frames`, which keeps every ratio forever.
    #[collect(require_static)]
    blended: RefCell<Vec<(u16, MadeFor, ShapeHandle, u64)>>,
    /// AQW build: in-between ratios in a row that were slow to tessellate (see
    /// `SLOW_IN_BETWEENS`), and whether a slow real ratio was logged (see `get_shape`).
    #[collect(require_static)]
    slow_in_betweens: Cell<u8>,
    #[collect(require_static)]
    logged_slow_frame: Cell<bool>,
    /// AQW build: whether a frame took long to tessellate, and whether it was logged that
    /// the tween's frames are tessellated in the background (see `heavy`).
    #[collect(require_static)]
    slow_frames: Cell<bool>,
    #[collect(require_static)]
    logged_heavy: Cell<bool>,
    /// AQW build: see `travel_px`.
    #[collect(require_static)]
    travel_px: std::cell::OnceCell<f32>,
    /// AQW build: how its instances are moved on, for frame interpolation.
    #[collect(require_static)]
    rhythm: crate::display_object::interp::Rhythm,
}

impl<'gc> MorphShapeShared<'gc> {
    pub fn from_swf_tag(swf_tag: &swf::DefineMorphShape, movie: SwfMovie<'gc>) -> Self {
        Self {
            id: swf_tag.id,
            start: swf_tag.start.clone(),
            end: swf_tag.end.clone(),
            frames: RefCell::new(fnv::FnvHashMap::default()),
            blended: RefCell::new(Vec::new()),
            slow_in_betweens: Cell::new(0),
            logged_slow_frame: Cell::new(false),
            slow_frames: Cell::new(false),
            logged_heavy: Cell::new(false),
            travel_px: std::cell::OnceCell::new(),
            rhythm: Default::default(),
            movie,
        }
    }

    /// Retrieves the `Frame` for the given ratio.
    /// Lazily initializes the frame if it does not yet exist.
    fn get_frame(&self, ratio: u16) -> RefMut<'_, Frame> {
        let frames = self.frames.borrow_mut();
        RefMut::map(frames, |frames| {
            frames
                .entry(ratio)
                .or_insert_with(|| self.build_morph_frame(ratio))
        })
    }

    /// Retrieves the `ShapeHandle` for the given ratio.
    /// Lazily initializes and tessellates the shape if it does not yet exist.
    /// (AQW build: tessellated again when drawn at a scale too far from its tessellation's.)
    fn get_shape(
        &self,
        context: &mut RenderContext<'_, 'gc>,
        ratio: u16,
        scale: Measure,
    ) -> ShapeHandle {
        let mut frame = self.get_frame(ratio);
        // The one made for the closest size, of those that fit.
        if let Some(index) = frame
            .shape_handles
            .iter()
            .enumerate()
            .filter(|(_, (made_for, _))| made_for.fits(scale))
            .min_by(|(_, (a, _)), (_, (b, _))| a.deviation(scale).total_cmp(&b.deviation(scale)))
            .map(|(index, _)| index)
        {
            let entry = frame.shape_handles.remove(index);
            let handle = entry.1.clone();
            frame.shape_handles.push(entry);
            return handle;
        }
        let waited = context.renderer.waited_micros();
        let started = Instant::now();
        let library = context.library.library_for_movie(self.movie).unwrap();
        let handle = context.renderer.register_shape_stretched(
            (&*frame.shape).into(),
            &MovieLibrarySource { library },
            scale.scale,
            scale.stretch,
        );
        let ms = work_ms(started, context.renderer.waited_micros() - waited);
        if ms > HEAVY_FRAME_MS {
            self.slow_frames.set(true);
        }
        if ms > SLOW_FRAME_MS && !self.logged_slow_frame.replace(true) {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "shape tween {} of {} ({} records): a frame took {:.1} ms to tessellate, for {:.0} px across",
                self.id,
                self.movie.url(),
                self.start.shape.len(),
                ms,
                px_across(&frame.bounds, scale)
            );
        }
        if frame.shape_handles.len() >= SIZES_KEPT {
            frame.shape_handles.remove(0);
        }
        frame
            .shape_handles
            .push((MadeFor::now(scale), handle.clone()));
        handle
    }

    /// AQW build: whether this tween's frames are tessellated in the background (see
    /// `get_heavy_shape`).
    fn heavy(&self) -> bool {
        self.start.shape.len() > HEAVY_TWEEN_RECORDS || self.slow_frames.get()
    }

    /// AQW build: `get_shape` for a heavy tween (see `HEAVY_TWEEN_RECORDS`), for an instance
    /// with the given `warming`. A player's cape and scythe had four tweens of about 4,700
    /// records, 37 frames each, 22-32 ms a frame to tessellate: the first time their animation
    /// played, every frame of it froze the game for 25-120 ms. So a frame of such a tween is
    /// tessellated on the thread pool, like a detailed shape (`deferred_shapes`), and isn't
    /// waited for. Until it's done the instance is drawn as it was last drawn, or as the
    /// newest frame it asked for that has been finished since (without frame interpolation
    /// every frame is asked for once per loop, so that's what moves it during its first
    /// loop). Where that can't be (see `deferred_shapes::may_stand_in`), the frame's
    /// placeholder is drawn, as for any shape: left out, waited for, or its cached object
    /// redrawn when it's done. Outside a frame's render it's tessellated at once.
    fn get_heavy_shape(
        &self,
        context: &mut RenderContext<'_, 'gc>,
        warming: &RefCell<Warming>,
        ratio: u16,
        mut scale: Measure,
    ) -> ShapeHandle {
        let (shape, cache) = {
            let mut frame = self.get_frame(ratio);
            let cache = frame
                .cache
                .get_or_insert_with(TessellationCache::new_shared)
                .clone();
            (frame.shape.clone(), cache)
        };
        let found = cache.borrow_mut().find_near_and_touch(&mut scale);
        let deferring = deferred_shapes::is_active();
        let handle = match found {
            Some(handle) if deferring || !deferred_shapes::is_placeholder(&handle) => handle,
            found => {
                let made = if deferring {
                    if !self.logged_heavy.replace(true) {
                        tracing::info!(
                            target: "ruffle_aqw_perf",
                            "shape tween {} of {} ({} records): its frames are tessellated in the background",
                            self.id,
                            self.movie.url(),
                            self.start.shape.len()
                        );
                    }
                    let (movie, shape) = (self.movie, shape.clone());
                    deferred_shapes::queue(movie, shape, scale, cache.clone(), false)
                } else {
                    let library = context.library.library_for_movie(self.movie).unwrap();
                    context.renderer.register_shape_stretched(
                        (&*shape).into(),
                        &MovieLibrarySource { library },
                        scale.scale,
                        scale.stretch,
                    )
                };
                let mut cache = cache.borrow_mut();
                match found {
                    // (Being made in the background, but this isn't a frame's render.)
                    Some(placeholder) => cache.replace(&placeholder, Some(made.clone())),
                    None => cache.insert(scale, made.clone()),
                }
                made
            }
        };

        let mut warming = warming.borrow_mut();
        if !deferred_shapes::is_placeholder(&handle) {
            warming.pending.clear();
            warming.last = Some(handle.clone());
            return handle;
        }
        // The newest of the frames it asked for before that is done by now.
        let done = warming
            .pending
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| {
                let mut measure = entry.1;
                let found = entry.0.borrow_mut().find_near_and_touch(&mut measure);
                found
                    .filter(|handle| !deferred_shapes::is_placeholder(handle))
                    .map(|handle| (index, handle))
            });
        if let Some((index, done)) = done {
            warming.last = Some(done);
            warming.pending.drain(..=index);
        }
        if !warming
            .pending
            .iter()
            .any(|(pending, _)| Rc::ptr_eq(pending, &cache))
        {
            if warming.pending.len() >= WARMING_KEPT {
                warming.pending.remove(0);
            }
            warming.pending.push((cache, scale));
        }
        match &warming.last {
            Some(last) if deferred_shapes::may_stand_in() => last.clone(),
            _ => handle,
        }
    }

    /// AQW build: the `ShapeHandle` for an in-between ratio of frame interpolation, or `None`
    /// if this tween is too slow to tessellate for that: it's then drawn at its real ratios,
    /// which are kept for good (as in Flash, it changes shape once per frame; where it is and
    /// its size are still blended). In-betweens depend on when the screen is drawn, so each
    /// loop of a tween has new ones, and they aren't kept: another player's class effect took
    /// 30-160 ms per in-between, on every picture of every loop (every 3-4 s, for a minute).
    fn get_blended_shape(
        &self,
        context: &mut RenderContext<'_, 'gc>,
        ratio: u16,
        scale: Measure,
    ) -> Option<ShapeHandle> {
        // (A heavy tween's frames are tessellated in the background: it has none to spare
        // for in-betweens, which aren't kept.)
        if self.slow_in_betweens.get() >= SLOW_IN_BETWEENS || self.heavy() {
            return None;
        }
        let swf_frame = crate::display_object::interp::frame();
        if let Some(frame) = self.frames.borrow().get(&ratio)
            && let Some((_, handle)) = frame
                .shape_handles
                .iter()
                .rev()
                .find(|(made_for, _)| made_for.fits(scale))
        {
            return Some(handle.clone());
        }
        let mut blended = self.blended.borrow_mut();
        if let Some(index) = blended
            .iter()
            .position(|(r, made_for, _, _)| *r == ratio && made_for.fits(scale))
        {
            let mut entry = blended.remove(index);
            entry.3 = swf_frame;
            let handle = entry.2.clone();
            blended.push(entry);
            return Some(handle);
        }
        // Timed without the renderer's wait for the present thread before the upload, which
        // at 360 Hz took up to 2.8 ms and made 4-record interface tweens look slow. (Not
        // waited for first instead: the tessellation runs while that thread works.)
        let waited = context.renderer.waited_micros();
        let started = Instant::now();
        let frame = self.build_morph_frame(ratio);
        let library = context.library.library_for_movie(self.movie).unwrap();
        let handle = context.renderer.register_shape_stretched(
            (&*frame.shape).into(),
            &MovieLibrarySource { library },
            scale.scale,
            scale.stretch,
        );
        let ms = work_ms(started, context.renderer.waited_micros() - waited);
        if ms <= SLOW_IN_BETWEEN_MS {
            self.slow_in_betweens.set(0);
        } else {
            let slow = self.slow_in_betweens.get() + 1;
            self.slow_in_betweens.set(slow);
            if slow >= SLOW_IN_BETWEENS {
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "shape tween {} of {} ({} records): {} in-betweens in a row took over {} ms to tessellate (the last {:.1} ms, for {:.0} px across), drawn at its real frames from now on",
                    self.id,
                    self.movie.url(),
                    self.start.shape.len(),
                    SLOW_IN_BETWEENS,
                    SLOW_IN_BETWEEN_MS,
                    ms,
                    px_across(&frame.bounds, scale)
                );
                blended.clear();
                return Some(handle);
            }
        }
        // An in-between belongs to the frame it was made in: every copy of the tween on
        // screen (the wipes of all skills cooling down, every aura's timer) has its own, a new
        // one per step, and none is drawn again once its frame is over. So those not drawn in
        // this frame or the last go, however many copies there are (the last 16 were kept
        // before, which more copies than that would have tessellated again on every picture).
        blended.retain(|entry| entry.3 + 1 >= swf_frame);
        if blended.len() >= IN_BETWEENS_KEPT {
            blended.remove(0);
        }
        blended.push((ratio, MadeFor::now(scale), handle.clone(), swf_frame));
        Some(handle)
    }

    /// AQW build: how far the point of this tween that moves the most travels between its two
    /// ends, in pixels at its own scale. Frame interpolation gives a tween in-betweens by how
    /// far it moves on screen (see interp's `tween_steps`).
    fn travel_px(&self) -> f32 {
        *self.travel_px.get_or_init(|| {
            use swf::ShapeRecord;
            // Both ends have the same records in the same order (see `build_morph_frame`), so
            // their pens move in step.
            let (from, to) = (self.build_morph_frame(0), self.build_morph_frame(u16::MAX));
            let (from, to) = (&from.shape.shape, &to.shape.shape);
            let (mut from_x, mut from_y) = (Twips::ZERO, Twips::ZERO);
            let (mut to_x, mut to_y) = (Twips::ZERO, Twips::ZERO);
            let mut travel = 0.0f64;
            let mut reach = |from_x: Twips, from_y: Twips, to_x: Twips, to_y: Twips| {
                let moved = (to_x - from_x)
                    .to_pixels()
                    .hypot((to_y - from_y).to_pixels());
                travel = travel.max(moved);
            };
            for (from, to) in from.iter().zip(to) {
                // (A curve's control point moves too.)
                if let (
                    ShapeRecord::CurvedEdge {
                        control_delta: from_control,
                        ..
                    },
                    ShapeRecord::CurvedEdge {
                        control_delta: to_control,
                        ..
                    },
                ) = (from, to)
                {
                    reach(
                        from_x + from_control.dx,
                        from_y + from_control.dy,
                        to_x + to_control.dx,
                        to_y + to_control.dy,
                    );
                }
                Self::update_pos(&mut from_x, &mut from_y, from);
                Self::update_pos(&mut to_x, &mut to_y, to);
                reach(from_x, from_y, to_x, to_y);
            }
            travel as f32
        })
    }

    /// AQW build: `travel_px` for a simple tween, else 0: only a simple one gets in-betweens
    /// for moving slowly but visibly (see `SIMPLE_TWEEN_RECORDS`). A detailed one keeps those
    /// it gets for moving fast.
    fn travel_px_if_simple(&self) -> f32 {
        if self.start.shape.len() <= SIMPLE_TWEEN_RECORDS {
            self.travel_px()
        } else {
            0.0
        }
    }

    fn build_morph_frame(&self, ratio: u16) -> Frame {
        // Interpolate MorphShapes into a Shape.
        use swf::{FillStyle, LineStyle, ShapeRecord, ShapeStyles};
        // Start shape is ratio 65535, end shape is ratio 0.
        let b = f32::from(ratio) / 65535.0;
        let a = 1.0 - b;
        let fill_styles: Vec<FillStyle> = self
            .start
            .fill_styles
            .iter()
            .zip(self.end.fill_styles.iter())
            .map(|(start, end)| lerp_fill(start, end, a, b))
            .collect();
        let line_styles: Vec<LineStyle> = self
            .start
            .line_styles
            .iter()
            .zip(self.end.line_styles.iter())
            .map(|(start, end)| {
                start
                    .clone()
                    .with_width(lerp_twips(start.width(), end.width(), a, b))
                    .with_fill_style(lerp_fill(start.fill_style(), end.fill_style(), a, b))
            })
            .collect();

        let mut shape = Vec::with_capacity(self.start.shape.len());
        let mut start_iter = self.start.shape.iter();
        let mut end_iter = self.end.shape.iter();
        let mut start = start_iter.next();
        let mut end = end_iter.next();
        let mut start_x = Twips::ZERO;
        let mut start_y = Twips::ZERO;
        let mut end_x = Twips::ZERO;
        let mut end_y = Twips::ZERO;
        // TODO: Feels like this could be cleaned up a bit.
        // We step through both the start records and end records, interpolating edges pairwise.
        // Fill style/line style changes should only appear in the start records.
        // However, StyleChangeRecord move_to can appear it both start and end records,
        // and not necessarily in matching pairs; therefore, we have to keep track of the pen position
        // in case one side is missing a move_to; it will implicitly use the last pen position.
        while let (Some(s), Some(e)) = (start, end) {
            match (s, e) {
                (ShapeRecord::StyleChange(start_change), ShapeRecord::StyleChange(end_change)) => {
                    let mut style_change = start_change.clone();
                    if start_change.move_to.is_some() || end_change.move_to.is_some() {
                        if let Some(move_to) = &start_change.move_to {
                            start_x = move_to.x;
                            start_y = move_to.y;
                        }
                        if let Some(move_to) = &end_change.move_to {
                            end_x = move_to.x;
                            end_y = move_to.y;
                        }
                        style_change.move_to = Some(Point::new(
                            lerp_twips(start_x, end_x, a, b),
                            lerp_twips(start_y, end_y, a, b),
                        ));
                    }
                    shape.push(ShapeRecord::StyleChange(style_change));
                    start = start_iter.next();
                    end = end_iter.next();
                }
                (ShapeRecord::StyleChange(start_change), _) => {
                    let mut style_change = start_change.clone();
                    if let Some(move_to) = &start_change.move_to {
                        start_x = move_to.x;
                        start_y = move_to.y;
                        style_change.move_to = Some(Point::new(
                            lerp_twips(start_x, end_x, a, b),
                            lerp_twips(start_y, end_y, a, b),
                        ));
                    }
                    shape.push(ShapeRecord::StyleChange(style_change));
                    Self::update_pos(&mut start_x, &mut start_y, s);
                    start = start_iter.next();
                }
                (_, ShapeRecord::StyleChange(end_change)) => {
                    let mut style_change = end_change.clone();
                    if let Some(move_to) = &end_change.move_to {
                        end_x = move_to.x;
                        end_y = move_to.y;
                        style_change.move_to = Some(Point::new(
                            lerp_twips(start_x, end_x, a, b),
                            lerp_twips(start_y, end_y, a, b),
                        ));
                    }
                    shape.push(ShapeRecord::StyleChange(style_change));
                    Self::update_pos(&mut end_x, &mut end_y, s);
                    end = end_iter.next();
                    continue;
                }
                _ => {
                    shape.push(lerp_edges(
                        Point::new(start_x, start_y),
                        Point::new(end_x, end_y),
                        s,
                        e,
                        a,
                        b,
                    ));
                    Self::update_pos(&mut start_x, &mut start_y, s);
                    Self::update_pos(&mut end_x, &mut end_y, e);
                    start = start_iter.next();
                    end = end_iter.next();
                }
            }
        }

        let styles = ShapeStyles {
            fill_styles,
            line_styles,
        };

        let bounds = ruffle_render::shape_utils::calculate_shape_bounds(&shape);
        let shape = swf::Shape {
            version: 4,
            id: 0,
            shape_bounds: bounds,
            edge_bounds: bounds,
            flags: swf::ShapeFlag::HAS_SCALING_STROKES,
            styles,
            shape,
        };

        Frame {
            shape_handles: Vec::new(),
            shape: Arc::new(shape),
            bounds,
            cache: None,
        }
    }

    fn update_pos(x: &mut Twips, y: &mut Twips, record: &swf::ShapeRecord) {
        use swf::ShapeRecord;
        match record {
            ShapeRecord::StraightEdge { delta } => {
                *x += delta.dx;
                *y += delta.dy;
            }
            ShapeRecord::CurvedEdge {
                control_delta,
                anchor_delta,
            } => {
                *x += control_delta.dx + anchor_delta.dx;
                *y += control_delta.dy + anchor_delta.dy;
            }
            ShapeRecord::StyleChange(style_change) => {
                if let Some(move_to) = &style_change.move_to {
                    *x = move_to.x;
                    *y = move_to.y;
                }
            }
        }
    }
}

/// AQW build: the milliseconds of work since `started`, given the microseconds the renderer
/// waited meanwhile (see `RenderBackend::waited_micros`).
fn work_ms(started: Instant, waited_micros: u64) -> f64 {
    (started.elapsed().as_secs_f64() * 1000.0 - waited_micros as f64 / 1000.0).max(0.0)
}

/// AQW build, for the perf log: how many pixels across (its larger side) a frame with these
/// `bounds` is tessellated for. Detailed tweens drawn small cost the most to tessellate (see
/// `Measure::at_least_min_size`, which tweens don't use).
fn px_across(bounds: &Rectangle<Twips>, scale: Measure) -> f64 {
    bounds.width().to_pixels().max(bounds.height().to_pixels()) * f64::from(scale.scale)
}

// Interpolation functions
// These interpolate between two SWF shape structures.
// a + b should = 1.0

fn lerp_color(start: Color, end: Color, a: f32, b: f32) -> Color {
    // f32 -> u8 cast is defined to saturate for out of bounds values,
    // so we don't have to worry about clamping.
    Color {
        r: (a * f32::from(start.r) + b * f32::from(end.r)) as u8,
        g: (a * f32::from(start.g) + b * f32::from(end.g)) as u8,
        b: (a * f32::from(start.b) + b * f32::from(end.b)) as u8,
        a: (a * f32::from(start.a) + b * f32::from(end.a)) as u8,
    }
}

fn lerp_twips(start: Twips, end: Twips, a: f32, b: f32) -> Twips {
    Twips::new((start.get() as f32 * a + end.get() as f32 * b).round() as i32)
}

fn lerp_point_twips(start: Point<Twips>, end: Point<Twips>, a: f32, b: f32) -> Point<Twips> {
    Point::new(
        lerp_twips(start.x, end.x, a, b),
        lerp_twips(start.y, end.y, a, b),
    )
}

fn lerp_fill(start: &swf::FillStyle, end: &swf::FillStyle, a: f32, b: f32) -> swf::FillStyle {
    use swf::FillStyle;
    match (start, end) {
        // Color-to-color
        (FillStyle::Color(start), FillStyle::Color(end)) => {
            FillStyle::Color(lerp_color(*start, *end, a, b))
        }

        // Bitmap-to-bitmap
        // ID should be the same.
        (
            FillStyle::Bitmap {
                id: start_id,
                matrix: start,
                is_smoothed,
                is_repeating,
            },
            FillStyle::Bitmap { matrix: end, .. },
        ) => FillStyle::Bitmap {
            id: *start_id,
            matrix: lerp_matrix(start, end, a, b),
            is_smoothed: *is_smoothed,
            is_repeating: *is_repeating,
        },

        // Linear-to-linear
        (FillStyle::LinearGradient(start), FillStyle::LinearGradient(end)) => {
            FillStyle::LinearGradient(lerp_gradient(start, end, a, b))
        }

        // Radial-to-radial
        (FillStyle::RadialGradient(start), FillStyle::RadialGradient(end)) => {
            FillStyle::RadialGradient(lerp_gradient(start, end, a, b))
        }

        // Focal gradients also interpolate focal point.
        (
            FillStyle::FocalGradient {
                gradient: start,
                focal_point: start_focal,
            },
            FillStyle::FocalGradient {
                gradient: end,
                focal_point: end_focal,
            },
        ) => FillStyle::FocalGradient {
            gradient: lerp_gradient(start, end, a, b),
            focal_point: *start_focal * Fixed8::from_f32(a) + *end_focal * Fixed8::from_f32(b),
        },

        // All other combinations should not occur, because SWF stores the start/end fill as the same type, always.
        // If you happened to make, say, a solid color-to-radial gradient tween in the IDE, this would get baked down into
        // a radial-to-radial gradient on export.
        _ => {
            tracing::warn!(
                "Unexpected morph shape fill style combination: {:#?}, {:#?}",
                start,
                end
            );
            start.clone()
        }
    }
}

fn lerp_edges(
    start_pen: Point<Twips>,
    end_pen: Point<Twips>,
    start: &swf::ShapeRecord,
    end: &swf::ShapeRecord,
    a: f32,
    b: f32,
) -> swf::ShapeRecord {
    use swf::ShapeRecord;
    let pen = lerp_point_twips(start_pen, end_pen, a, b);
    match (start, end) {
        (
            ShapeRecord::StraightEdge { delta: start_delta },
            ShapeRecord::StraightEdge { delta: end_delta },
        ) => {
            let start_anchor = start_pen + *start_delta;
            let end_anchor = end_pen + *end_delta;

            let anchor = lerp_point_twips(start_anchor, end_anchor, a, b);

            ShapeRecord::StraightEdge {
                delta: anchor - pen,
            }
        }

        (
            ShapeRecord::CurvedEdge {
                control_delta: start_control_delta,
                anchor_delta: start_anchor_delta,
            },
            ShapeRecord::CurvedEdge {
                control_delta: end_control_delta,
                anchor_delta: end_anchor_delta,
            },
        ) => {
            let start_control = start_pen + *start_control_delta;
            let start_anchor = start_control + *start_anchor_delta;

            let end_control = end_pen + *end_control_delta;
            let end_anchor = end_control + *end_anchor_delta;

            let control = lerp_point_twips(start_control, end_control, a, b);
            let anchor = lerp_point_twips(start_anchor, end_anchor, a, b);

            ShapeRecord::CurvedEdge {
                control_delta: control - pen,
                anchor_delta: anchor - control,
            }
        }

        (
            ShapeRecord::StraightEdge { delta: start_delta },
            ShapeRecord::CurvedEdge {
                control_delta: end_control_delta,
                anchor_delta: end_anchor_delta,
            },
        ) => {
            let start_control = start_pen + *start_delta / 2;
            let start_anchor = start_pen + *start_delta;

            let end_control = end_pen + *end_control_delta;
            let end_anchor = end_control + *end_anchor_delta;

            let control = lerp_point_twips(start_control, end_control, a, b);
            let anchor = lerp_point_twips(start_anchor, end_anchor, a, b);

            ShapeRecord::CurvedEdge {
                control_delta: control - pen,
                anchor_delta: anchor - control,
            }
        }

        (
            ShapeRecord::CurvedEdge {
                control_delta: start_control_delta,
                anchor_delta: start_anchor_delta,
            },
            ShapeRecord::StraightEdge { delta: end_delta },
        ) => {
            let start_control = start_pen + *start_control_delta;
            let start_anchor = start_control + *start_anchor_delta;

            let end_control = end_pen + *end_delta / 2;
            let end_anchor = end_pen + *end_delta;

            let control = lerp_point_twips(start_control, end_control, a, b);
            let anchor = lerp_point_twips(start_anchor, end_anchor, a, b);

            ShapeRecord::CurvedEdge {
                control_delta: control - pen,
                anchor_delta: anchor - control,
            }
        }
        _ => unreachable!("{:?} {:?}", start, end),
    }
}

fn lerp_matrix(start: &swf::Matrix, end: &swf::Matrix, a: f32, b: f32) -> swf::Matrix {
    // TODO: Lerping a matrix element-wise is geometrically wrong,
    // but I doubt Flash is decomposing the matrix into scale-rotate-translate?
    let af = Fixed16::from_f32(a);
    let bf = Fixed16::from_f32(b);
    swf::Matrix {
        a: start.a * af + end.a * bf,
        b: start.b * af + end.b * bf,
        c: start.c * af + end.c * bf,
        d: start.d * af + end.d * bf,
        tx: lerp_twips(start.tx, end.tx, a, b),
        ty: lerp_twips(start.ty, end.ty, a, b),
    }
}

fn lerp_gradient(start: &swf::Gradient, end: &swf::Gradient, a: f32, b: f32) -> swf::Gradient {
    use swf::{Gradient, GradientRecord};
    // Morph gradients are guaranteed to have the same number of records in the start/end gradient.
    debug_assert_eq!(start.records.len(), end.records.len());
    let records: Vec<GradientRecord> = start
        .records
        .iter()
        .zip(end.records.iter())
        .map(|(start, end)| swf::GradientRecord {
            ratio: (f32::from(start.ratio) * a + f32::from(end.ratio) * b) as u8,
            color: lerp_color(start.color, end.color, a, b),
        })
        .collect();

    Gradient {
        matrix: lerp_matrix(&start.matrix, &end.matrix, a, b),
        spread: start.spread,
        interpolation: start.interpolation,
        records,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Regression test for #14074
    #[test]
    fn test_lerp_rounding() {
        let ratio: u16 = 17246;
        let b = f32::from(ratio) / 65535.0;
        let a = 1.0 - b;

        assert_eq!(
            lerp_twips(Twips::new(-7), Twips::new(-7), a, b),
            Twips::new(-7)
        );
    }

    /// A 100 px wide rectangle of the given height (px), drawn clockwise from its top left.
    fn wipe(height: f64) -> swf::MorphShape {
        use swf::{PointDelta, ShapeRecord, StyleChangeData};
        let bounds = Rectangle {
            x_min: Twips::ZERO,
            x_max: Twips::from_pixels(100.0),
            y_min: Twips::ZERO,
            y_max: Twips::from_pixels(height),
        };
        let edge = |dx: f64, dy: f64| ShapeRecord::StraightEdge {
            delta: PointDelta::new(Twips::from_pixels(dx), Twips::from_pixels(dy)),
        };
        swf::MorphShape {
            shape_bounds: bounds,
            edge_bounds: bounds,
            fill_styles: vec![swf::FillStyle::Color(Color::BLACK)],
            line_styles: Vec::new(),
            shape: vec![
                ShapeRecord::StyleChange(Box::new(StyleChangeData {
                    move_to: Some(Point::new(Twips::ZERO, Twips::ZERO)),
                    fill_style_0: None,
                    fill_style_1: Some(1),
                    line_style: None,
                    new_styles: None,
                })),
                edge(100.0, 0.0),
                edge(0.0, height),
                edge(-100.0, 0.0),
                edge(0.0, -height),
            ],
        }
    }

    #[test]
    fn a_wipe_travels_as_far_as_its_edge_moves() {
        // (A movie handle needs an arena.)
        gc_arena::arena::rootless_mutate(|mc| {
            let movie = SwfMovie::new(mc, crate::tag_utils::SwfMovieData::empty(10, None));
            // A cooldown wipe: the bottom edge of a rectangle moves down by 58 px over the tween.
            let tag = swf::DefineMorphShape {
                version: 1,
                id: 1,
                flags: swf::DefineMorphShapeFlag::empty(),
                start: wipe(2.0),
                end: wipe(60.0),
            };
            let shared = MorphShapeShared::from_swf_tag(&tag, movie);
            assert_eq!(shared.travel_px(), 58.0);
            assert_eq!(shared.travel_px_if_simple(), 58.0);
            // A detailed tween (here the same wipe, its top edge in 40 pieces) gets no in-betweens
            // for moving slowly.
            let detailed = |height: f64| {
                let mut shape = wipe(height);
                let top = shape.shape.remove(1);
                let piece = swf::ShapeRecord::StraightEdge {
                    delta: swf::PointDelta::new(Twips::from_pixels(2.5), Twips::ZERO),
                };
                assert!(matches!(top, swf::ShapeRecord::StraightEdge { .. }));
                let rest = shape.shape.split_off(1);
                shape.shape.extend(std::iter::repeat_n(piece, 40));
                shape.shape.extend(rest);
                shape
            };
            let detailed_tag = swf::DefineMorphShape {
                version: 1,
                id: 2,
                flags: swf::DefineMorphShapeFlag::empty(),
                start: detailed(2.0),
                end: detailed(60.0),
            };
            let shared = MorphShapeShared::from_swf_tag(&detailed_tag, movie);
            assert_eq!(shared.travel_px(), 58.0);
            assert_eq!(shared.travel_px_if_simple(), 0.0);
            // A tween that only changes colour doesn't move.
            let still = swf::DefineMorphShape {
                start: wipe(60.0),
                ..tag
            };
            let shared = MorphShapeShared::from_swf_tag(&still, movie);
            assert_eq!(shared.travel_px(), 0.0);
        });
    }
}
