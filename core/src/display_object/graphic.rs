use crate::avm1::Object as Avm1Object;
use crate::avm2::{
    Activation as Avm2Activation, Avm2, ClassObject as Avm2ClassObject,
    StageObject as Avm2StageObject,
};
use crate::context::{RenderContext, UpdateContext};
use crate::display_object::{BoundsMode, DisplayObjectBase};
use crate::drawing::Drawing;
use crate::library::MovieLibrarySource;
use crate::prelude::*;
use crate::tag_utils::SwfMovie;
use crate::tessellation_cache::{Measure, TessellationCache};
use crate::vminterface::Instantiator;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::ShapeHandle;
use ruffle_render::commands::CommandHandler;
use ruffle_render::matrix::Matrix;
use ruffle_render::transform::Transform;
use std::cell::{OnceCell, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Graphic<'gc>(Gc<'gc, GraphicData<'gc>>);

impl fmt::Debug for Graphic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graphic")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct GraphicData<'gc> {
    base: DisplayObjectBase<'gc>,
    shared: Lock<Gc<'gc, GraphicShared<'gc>>>,
    class: Lock<Option<Avm2ClassObject<'gc>>>,
    avm2_object: Lock<Option<Avm2StageObject<'gc>>>,
    /// This is lazily allocated on demand, to make `GraphicData` smaller in the common case.
    #[collect(require_static)]
    drawing: OnceCell<Box<RefCell<Drawing>>>,
}

impl<'gc> Graphic<'gc> {
    /// AQW build: the shared data that every instance of this library symbol points to, used
    /// to tell whether the symbol (and so its movie's library) is still in use.
    pub fn aqw_shared(self) -> Gc<'gc, ()> {
        Gc::erase(self.0.shared.get())
    }

    /// Construct a `Graphic` from it's associated `Shape` tag.
    /// AQW build: `swf_shape` may be shared with other graphics (see `shared_shapes`).
    pub fn from_swf_tag(
        context: &mut UpdateContext<'gc>,
        swf_shape: Arc<swf::Shape>,
        movie: SwfMovie<'gc>,
    ) -> Self {
        // AQW build: tessellated when first drawn, at the scale it's drawn at (see
        // `deferred_shapes`).
        let shared = GraphicShared {
            id: swf_shape.id,
            shape_bounds: swf_shape.shape_bounds.clone(),
            edge_bounds: swf_shape.edge_bounds.clone(),
            shape: swf_shape,
            movie,
            scaled_handle: TessellationCache::new_shared(),
            sliced: Default::default(),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    /// Construct an empty `Graphic`.
    pub fn empty(context: &mut UpdateContext<'gc>) -> Self {
        let shared = GraphicShared {
            id: 0,
            shape_bounds: Default::default(),
            edge_bounds: Default::default(),
            shape: Arc::new(swf::Shape {
                version: 32,
                id: 0,
                shape_bounds: Default::default(),
                edge_bounds: Default::default(),
                flags: swf::ShapeFlag::empty(),
                styles: swf::ShapeStyles {
                    fill_styles: Vec::new(),
                    line_styles: Vec::new(),
                },
                shape: Vec::new(),
            }),
            movie: *context.root_swf,
            scaled_handle: TessellationCache::new_shared(),
            sliced: Default::default(),
        };

        Graphic(Gc::new(
            context.gc(),
            GraphicData {
                base: Default::default(),
                shared: Lock::new(Gc::new(context.gc(), shared)),
                class: Lock::new(None),
                avm2_object: Lock::new(None),
                drawing: OnceCell::new(),
            },
        ))
    }

    pub fn instantiate(self, mc: &Mutation<'gc>) -> Self {
        Self(Gc::new(mc, (*self.0).clone()))
    }

    pub fn drawing_mut(&self) -> RefMut<'_, Drawing> {
        // AQW build: its bounds may change (see `display_object::pick_notes`).
        super::pick_notes::mark_from(self.parent());
        // (And what it draws: see `replay::lean_render`.)
        super::replay::mark_dirty(&self.base());
        self.0.drawing.get_or_init(Default::default).borrow_mut()
    }

    /// AQW build: draws its drawing (if it has one), for frame interpolation's replays.
    pub(crate) fn render_drawing(self, context: &mut RenderContext) {
        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().render(context);
        }
    }

    pub fn set_avm2_class(self, mc: &Mutation<'gc>, class: Avm2ClassObject<'gc>) {
        unlock!(Gc::write(mc, self.0), GraphicData, class).set(Some(class));
    }

    fn set_shared(self, mc: &Mutation<'gc>, shared: Gc<'gc, GraphicShared<'gc>>) {
        unlock!(Gc::write(mc, self.0), GraphicData, shared).set(shared);
    }

    /// Returns the best shape handle for the current scale, retessellating if necessary.
    ///
    /// AQW build: there is no scale-1 tessellation any more. Stock Ruffle tessellated every
    /// shape at scale 1 when it was defined, but a graphic's first draw always tessellated it
    /// again at its drawn scale (the cache starts empty), so that work was thrown away. And
    /// during a render, the tessellation is deferred and done for all shapes of the frame at
    /// once on all cores (see `deferred_shapes`). The meshes are the same.
    fn shape_handle(
        self,
        context: &mut RenderContext<'_, 'gc>,
        current_scale: Measure,
    ) -> Option<ShapeHandle> {
        let shared = self.0.shared.get();
        Self::handle_for(
            context,
            shared.movie,
            &shared.shape,
            &shared.scaled_handle,
            current_scale,
        )
    }

    /// The tessellation of `shape` at about `current_scale`, from `cache` or made now (or
    /// queued, see `deferred_shapes`).
    pub(crate) fn handle_for(
        context: &mut RenderContext<'_, 'gc>,
        movie: SwfMovie<'gc>,
        shape: &Arc<swf::Shape>,
        cache: &Rc<RefCell<TessellationCache>>,
        current_scale: Measure,
    ) -> Option<ShapeHandle> {
        let mut current_scale =
            current_scale.at_least_min_size(&shape.shape_bounds, shape.shape.len());
        // Since graphics are created from a shared shape, we may be able to reuse a
        // cached tessellation from another instance at a similar scale.
        let found = cache.borrow_mut().find_near_and_touch(&mut current_scale);
        if let Some(handle) = found {
            // AQW build: a tessellation still being made in the background (see
            // `deferred_shapes`) can only be drawn in a frame, where it's swapped in when done
            // (or left out). Anywhere else (`BitmapData.draw`: a snapshot, kept as it is) it's
            // made now.
            if deferred_shapes::is_placeholder(&handle) && !deferred_shapes::is_active() {
                let library = context.library.library_for_movie(movie)?;
                let made = context.renderer.register_shape_stretched(
                    (&**shape).into(),
                    &MovieLibrarySource { library },
                    current_scale.scale,
                    current_scale.stretch,
                );
                cache.borrow_mut().replace(&handle, Some(made.clone()));
                return Some(made);
            }
            // Found a cached handle at a similar scale; reuse it.
            return Some(handle);
        }

        // Retessellate at the new scale
        let library = context.library.library_for_movie(movie)?;
        let new_handle = if deferred_shapes::is_active() {
            deferred_shapes::queue(movie, shape.clone(), current_scale, cache.clone(), true)
        } else {
            context.renderer.register_shape_stretched(
                (&**shape).into(),
                &MovieLibrarySource { library },
                current_scale.scale,
                current_scale.stretch,
            )
        };

        let mut cache = cache.borrow_mut();
        tracing::debug!(
            "Graphic id={} retessellated: new_scale={:.2}, cache_size={}",
            shape.id,
            current_scale.scale,
            cache.len()
        );
        cache.insert(current_scale, new_handle.clone());
        Some(new_handle)
    }

    /// AQW build: 9-slice scaling. When this shape's parent has a scaling grid
    /// (`scale9Grid`, the SWF's DefineScalingGrid) and is scaled (no rotation or skew), Flash
    /// keeps the parts outside the grid at their size: corners aren't scaled, the side parts
    /// only along the side, the center normally. Returns the sliced shape's handle and the
    /// matrix to draw it with (the parent's scale is in the shape's points instead).
    fn sliced_handle(
        self,
        context: &mut RenderContext<'_, 'gc>,
    ) -> Option<(ShapeHandle, Matrix)> {
        let parent = self.parent()?;
        let grid = parent.scaling_grid();
        if grid.width() <= Twips::ZERO || grid.height() <= Twips::ZERO {
            return None;
        }
        let parent_matrix = parent.base().matrix();
        if parent_matrix.b != 0.0
            || parent_matrix.c != 0.0
            || parent_matrix.a <= 0.0
            || parent_matrix.d <= 0.0
            || (parent_matrix.a == 1.0 && parent_matrix.d == 1.0)
        {
            return None;
        }
        let slicing = scale9::Slicing::new(
            parent.bounds(BoundsMode::Engine),
            grid,
            parent_matrix.a.into(),
            parent_matrix.d.into(),
        )?;
        let child_matrix = self.base().matrix();
        // world = (parent's parent) * parent * child; drawn with (parent's parent) * the
        // parent's translation, as the sliced points already contain the parent's scale.
        let matrix = context.transform_stack.transform().matrix
            * child_matrix.inverse()?
            * Matrix::scale(1.0 / parent_matrix.a, 1.0 / parent_matrix.d);
        let current_scale = Measure::of(&matrix);

        let shared = self.0.shared.get();
        let key = scale9::Key::new(&child_matrix, &slicing);
        let (shape, cache) = {
            let mut sliced = shared.sliced.borrow_mut();
            let entry = match sliced.iter().position(|entry| entry.0 == key) {
                Some(index) => sliced.remove(index),
                None => {
                    if sliced.len() >= scale9::CACHE_SIZE {
                        sliced.remove(0);
                    }
                    (
                        key,
                        Arc::new(scale9::slice_shape(&shared.shape, &child_matrix, &slicing)),
                        TessellationCache::new_shared(),
                    )
                }
            };
            // Most recently used last.
            let found = (entry.1.clone(), entry.2.clone());
            sliced.push(entry);
            found
        };
        let handle = Self::handle_for(context, shared.movie, &shape, &cache, current_scale)?;
        Some((handle, matrix))
    }
}

impl<'gc> TDisplayObject<'gc> for Graphic<'gc> {
    fn base(self) -> Gc<'gc, DisplayObjectBase<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    fn id(self) -> CharacterId {
        self.0.shared.get().id
    }

    fn self_bounds(self, mode: BoundsMode) -> Rectangle<Twips> {
        let include_strokes = mode.includes_strokes();

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().self_bounds(include_strokes)
        } else if include_strokes {
            self.0.shared.get().shape_bounds
        } else {
            self.0.shared.get().edge_bounds
        }
    }

    fn construct_frame(self, context: &mut UpdateContext<'gc>) {
        // AQW build: check `object2` first. `movie()` clones an `Arc` (two atomic ops), and
        // every goto re-runs this for every shape on stage, almost all already constructed.
        if self.object2().is_none() && self.movie().is_action_script_3() {
            let class_object = self
                .0
                .class
                .get()
                .unwrap_or_else(|| context.avm2.classes().shape);

            let mut activation = Avm2Activation::from_nothing(context);

            match Avm2StageObject::for_display_object_childless(
                &mut activation,
                self.into(),
                class_object,
            ) {
                Ok(object) => self.set_object2(activation.context, object),
                Err(err) => {
                    Avm2::uncaught_error(
                        &mut activation,
                        Some(self.into()),
                        err,
                        "Error running AVM2 construction for shape",
                    );
                }
            }

            self.on_construction_complete(context);
        }
    }

    fn replace_with(self, context: &mut UpdateContext<'gc>, id: CharacterId) {
        // Static assets like Graphics can replace themselves via a PlaceObject tag with PlaceObjectAction::Replace.
        // This does not create a new instance, but instead swaps out the underlying static data to point to the new art.
        if let Some(new_graphic) = context
            .library
            .library_for_movie_mut(self.movie())
            .get_graphic(id)
        {
            self.set_shared(context.gc(), new_graphic.0.shared.get());
        } else {
            tracing::warn!("PlaceObject: expected Graphic at character ID {}", id);
        }
        self.invalidate_cached_bitmap();
    }

    fn render_self(self, context: &mut RenderContext<'_, 'gc>) {
        if !context.is_offscreen
            && !self
                .world_bounds(BoundsMode::Engine)
                .intersects(&context.stage.view_bounds())
        {
            // Off-screen; culled
            return;
        }

        if let Some(drawing) = self.0.drawing.get() {
            drawing.borrow().render(context);
            // AQW build: recorded for frame interpolation's next pictures (see `replay`).
            if let Some(recorder) = context.recorder.as_deref_mut() {
                recorder.drawing(&context.commands, self.into());
            }
        } else if !self.0.shared.get().shape.shape.is_empty() {
            let transform = context.transform_stack.transform();

            if let Some((handle, matrix)) = self.sliced_handle(context) {
                context.commands.render_shape(
                    handle,
                    Transform {
                        matrix,
                        ..transform
                    },
                );
                return;
            }

            // Calculate the current scale from the transform, to determine if
            // we can reuse a cached tessellation or need to retessellate.
            let current_scale = Measure::of(&transform.matrix);

            if let Some(handle) = self.shape_handle(context, current_scale) {
                context.commands.render_shape(handle, transform);
                // AQW build: recorded for frame interpolation's next pictures (see `replay`).
                if let Some(recorder) = context.recorder.as_deref_mut()
                    && recorder.recording()
                {
                    let shared = self.0.shared.get();
                    recorder.shape(
                        &context.commands,
                        crate::library::Library::movie_addr(shared.movie),
                        shared.shape.clone(),
                        shared.scaled_handle.clone(),
                    );
                }
            }
        }
    }

    fn hit_test_shape(
        self,
        _context: &mut UpdateContext<'gc>,
        point: Point<Twips>,
        options: HitTestOptions,
    ) -> bool {
        // Transform point to local coordinates and test.
        if (!options.contains(HitTestOptions::SKIP_INVISIBLE) || self.visible())
            && self.world_bounds(BoundsMode::Engine).contains(point)
        {
            let Some(local_matrix) = self.global_to_local_matrix() else {
                return false;
            };
            let point = local_matrix * point;
            if let Some(drawing) = self.0.drawing.get() {
                if drawing.borrow().hit_test(point, &local_matrix) {
                    return true;
                }
            } else {
                let shape = &self.0.shared.get().shape;
                return ruffle_render::shape_utils::shape_hit_test(shape, point, &local_matrix);
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

    fn object1(self) -> Option<Avm1Object<'gc>> {
        None
    }

    fn object2(self) -> Option<Avm2StageObject<'gc>> {
        self.0.avm2_object.get()
    }

    fn set_object2(self, context: &mut UpdateContext<'gc>, to: Avm2StageObject<'gc>) {
        let mc = context.gc();
        unlock!(Gc::write(mc, self.0), GraphicData, avm2_object).set(Some(to));
    }

    fn as_drawing(&self) -> Option<RefMut<'_, Drawing>> {
        Some(self.drawing_mut())
    }
}

/// Data shared between all instances of a Graphic.
#[derive(Collect)]
#[collect(no_drop)]
struct GraphicShared<'gc> {
    movie: SwfMovie<'gc>,
    id: CharacterId,

    #[collect(require_static)]
    shape: Arc<swf::Shape>,

    #[collect(require_static)]
    shape_bounds: Rectangle<Twips>,

    #[collect(require_static)]
    edge_bounds: Rectangle<Twips>,

    /// AQW build: shared with pending deferred tessellations (see `deferred_shapes`).
    #[collect(require_static)]
    scaled_handle: Rc<RefCell<TessellationCache>>,
    /// AQW build: 9-slice scaled versions of the shape, least recently used first (see
    /// `Graphic::sliced_handle`).
    #[collect(require_static)]
    sliced: RefCell<Vec<scale9::Entry>>,
}

/// AQW build: parsed shapes shared between identical shape tags.
///
/// AQW loads a gear file once for every player wearing it (1,763 SWFs loaded for 596
/// distinct URLs in a busy session), and each copy parsed its shapes again: a parsed shape
/// is several times the size of its tag (one record per edge), and those were the biggest
/// part of the heap that wasn't AS3 data (290 MB of 2.2 GB counted). A shape tag whose bytes
/// are the same as one parsed before, still in use, gets that parse (read-only; everything
/// per copy, the movie, its library and its tessellations, stays per graphic). Two hashes of
/// the bytes, their length and the tag's version make the key.
pub mod shared_shapes {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::hash::Hasher;
    use std::sync::{Arc, Weak};

    type Key = (u64, u64, usize, u8);

    thread_local! {
        static SHAPES: RefCell<HashMap<Key, Weak<swf::Shape>>> = RefCell::new(HashMap::new());
        /// Entry count at which entries whose shape is gone are removed.
        static NEXT_PRUNE: Cell<usize> = const { Cell::new(1024) };
        /// (Shape tags parsed, shape tags given an existing parse), for the perf log.
        static COUNTS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    fn key(bytes: &[u8], version: u8) -> Key {
        let mut sip = std::hash::DefaultHasher::new();
        sip.write(bytes);
        let mut fnv = fnv::FnvHasher::default();
        fnv.write(bytes);
        (sip.finish(), fnv.finish(), bytes.len(), version)
    }

    /// The parse of a shape tag (`bytes`, of DefineShape `version`): one made before for the
    /// same bytes if it's still in use, otherwise `parse()`'s.
    pub fn parsed(
        bytes: &[u8],
        version: u8,
        parse: impl FnOnce() -> swf::error::Result<swf::Shape>,
    ) -> swf::error::Result<Arc<swf::Shape>> {
        // `RUFFLE_NO_SHARED_PARSES`: every tag parsed on its own, as before (A/B). (Not
        // `RUFFLE_NO_SHARED_SHAPES`: that is the GPU-side switch, one buffer per shape, and
        // part of safe mode.)
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var_os("RUFFLE_NO_SHARED_PARSES").is_some()) {
            COUNTS.with(|counts| {
                let (parsed, reused) = counts.get();
                counts.set((parsed + 1, reused));
            });
            return Ok(Arc::new(parse()?));
        }
        let key = key(bytes, version);
        if let Some(shape) = SHAPES.with_borrow(|shapes| shapes.get(&key).and_then(Weak::upgrade)) {
            COUNTS.with(|counts| {
                let (parsed, reused) = counts.get();
                counts.set((parsed, reused + 1));
            });
            return Ok(shape);
        }
        let shape = Arc::new(parse()?);
        SHAPES.with_borrow_mut(|shapes| {
            if shapes.len() >= NEXT_PRUNE.with(Cell::get) {
                shapes.retain(|_, shape| shape.strong_count() > 0);
                NEXT_PRUNE.with(|next| next.set((shapes.len() * 2).max(1024)));
            }
            shapes.insert(key, Arc::downgrade(&shape));
        });
        COUNTS.with(|counts| {
            let (parsed, reused) = counts.get();
            counts.set((parsed + 1, reused));
        });
        Ok(shape)
    }

    /// (Shape tags parsed, shape tags given an existing parse) so far, for the perf log.
    pub fn counts() -> (u64, u64) {
        COUNTS.with(Cell::get)
    }
}

/// AQW build: tessellating the shapes a frame needs on all cores, in the background.
///
/// A graphic is tessellated at the scale it's drawn at, the first time it's drawn at about
/// that scale (`Graphic::shape_handle`). When a crowded room loads, that's hundreds of shapes in
/// a single render. While `Player::render` builds the frame's commands (`Scope`), a missing
/// tessellation is queued instead and a placeholder handle goes into the commands and the
/// shape's cache; `resolve` then starts the queue on the thread pool
/// (`RenderBackend::start_shapes`, each shape on its own) and swaps the placeholders for the
/// real handles, before the commands reach the renderer. Same meshes, same caching decisions
/// (later instances at a similar scale reuse the pending entry, as they would reuse the
/// finished one).
///
/// The frame waits for its shapes only so long (`set_wait`; the game's player sets it, tests
/// and other renders wait for all): a single very detailed shape takes one core up to 85 ms
/// (a 26,000-record armor), so a room where several players wear it froze for 70-200 ms on
/// entry. A shape not done by then isn't drawn in this frame (its placeholder stays in its
/// cache, so no other instance queues it again); a later frame swaps it in once it's done.
/// Shapes the frame can't do without are always waited for: a mask (missing, it would hide its
/// content or show it unmasked) and a layer erasing or cutting out (Erase, Alpha blends).
/// A cached object's content (cacheAsBitmap: drawn once and kept) used to be waited for too,
/// which on a laptop froze room entry for 70-150 ms per detailed map shape: now a cached
/// object whose shapes aren't ready keeps its last picture (or, drawn into a new texture
/// that has none, is drawn blank: the texture cleared) and is marked, so it's redrawn in the
/// next picture, and again until its shapes are done (see `take_cache_pending`);
/// `RUFFLE_WAIT_CACHED_SHAPES` waits for them as before. Except in a picture that is mostly
/// new (a room just entered, see `entry_hold`): its cached objects drawn for the first time
/// are the map's pieces, and blank they showed as black sections for a moment, so that
/// picture is held for their shapes again, up to a tenth of a second. Renders
/// outside `Player::render` (e.g. `BitmapData.draw`) still tessellate immediately.
/// `RUFFLE_SERIAL_SHAPES` tessellates each shape immediately, one at a time.
///
/// `RUFFLE_SHAPE_BUDGET` (off unless set): a finished tessellation still has to be uploaded
/// to the GPU, on this thread, and every one found finished was uploaded in the frame that
/// found it: when players walk into a room that is hundreds of shapes in one frame (26-34 ms
/// on a fast PC with DirectX 12, 90-175 ms on a laptop's built-in graphics, i.e. the hitch on
/// entering a room). With the switch a frame uploads for a few milliseconds and leaves the
/// rest for the next frames (`READY`); a shape waiting for its upload counts as still being
/// tessellated, so it is drawn a few frames later, as one that was not done in time. Only in
/// frames that do not wait for all their shapes (the game's), and never for a shape the frame
/// cannot do without.
pub mod deferred_shapes {
    use crate::library::{Library, MovieLibrarySource};
    use crate::tag_utils::SwfMovie;
    use crate::tessellation_cache::{Measure, TessellationCache};
    use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ShapeHandle, ShapeHandleImpl};
    use ruffle_render::bitmap::{BitmapHandle, BitmapHandleImpl, BitmapSource};
    use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, HashSet};
    use std::rc::Rc;
    use std::sync::{Arc, Weak};
    use std::time::{Duration, Instant};
    use swf::BlendMode;

    /// Stands in for a tessellation that isn't done yet. Never reaches a renderer.
    #[derive(Debug)]
    struct Placeholder;

    impl ShapeHandleImpl for Placeholder {}

    struct Job {
        /// The movie by its address (see `Library::movie_addr`): a job can outlive the render
        /// it was queued in, and this is a thread-local.
        movie: usize,
        shape: Arc<swf::Shape>,
        scale: Measure,
        cache: Rc<RefCell<TessellationCache>>,
        placeholder: ShapeHandle,
        /// Whether the frame it's queued in waits for it (as long as it waits for any). Not
        /// for one that's known to take longer than that (a very detailed tween's frame).
        wait: bool,
    }

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        /// Queued in this frame (until `resolve`).
        static JOBS: RefCell<Vec<Job>> = const { RefCell::new(Vec::new()) };
        /// Started and not done yet, by ticket.
        static RUNNING: RefCell<HashMap<u64, Job>> = RefCell::new(HashMap::new());
        static NEXT_TICKET: Cell<u64> = const { Cell::new(0) };
        /// How long a frame waits for its shapes (`None`: until they're done).
        static WAIT: Cell<Option<Duration>> = const { Cell::new(None) };
        /// Cached objects drawn blank, left with their last picture, or drawn with a part
        /// that was, because shapes weren't ready, by their cache texture: the texture (weak)
        /// and the tickets of the tessellations they wait for (see `take_cache_pending`).
        static PENDING_CACHES: RefCell<
            HashMap<*const (), (Weak<dyn BitmapHandleImpl>, Vec<u64>)>,
        > = RefCell::new(HashMap::new());
        /// The cache textures whose contents are being walked right now, outermost first
        /// (see `enter_cache`).
        static CACHE_STACK: RefCell<Vec<BitmapHandle>> = const { RefCell::new(Vec::new()) };
        /// Tessellated and not uploaded yet, oldest first: what a frame's upload budget left
        /// for the next ones (see `upload_budget`). Their jobs are still in `RUNNING`.
        static READY: RefCell<Vec<(u64, ruffle_render::backend::TessellatedShape)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// How long a frame may spend uploading finished tessellations before it leaves the rest
    /// to the next frame: `RUFFLE_SHAPE_BUDGET`, in milliseconds when its value is a number
    /// of 2 or more, 4 ms otherwise; `None` (no limit, as it always was) without it.
    fn upload_budget() -> Option<Duration> {
        static BUDGET: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
        *BUDGET.get_or_init(|| {
            let value = std::env::var("RUFFLE_SHAPE_BUDGET").ok()?;
            let ms = value.trim().parse::<f64>().ok().filter(|ms| *ms >= 2.0).unwrap_or(4.0);
            Some(Duration::from_secs_f64(ms.min(1000.0) / 1000.0))
        })
    }

    /// `RUFFLE_WAIT_CACHED_SHAPES`: frames wait for the shapes of cached objects' contents, as
    /// they did (for A/B comparisons).
    fn wait_for_cached() -> bool {
        static WAIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *WAIT.get_or_init(|| std::env::var_os("RUFFLE_WAIT_CACHED_SHAPES").is_some())
    }

    /// A picture with at least this many new shapes is one of a room just entered (a map is
    /// 300-600, a crowd more; a player walking in is about 80).
    const ENTRY_SHAPES: usize = 200;

    /// How long, from the start of its wait, a room-entry picture (`ENTRY_SHAPES`) is held
    /// for the shapes of the cached objects it draws for the first time, in a frame that
    /// does not wait for all its shapes anyway. Those are the map's pieces: drawn blank
    /// (see `resolve`) they showed as black sections for a moment on entering a room, which
    /// the frame freezing a little longer does not (the room is only just appearing).
    /// Within a room the short wait stays, where a held frame is a stutter and a part of a
    /// player appearing a picture later is not seen. Not for longer than this, so one very
    /// detailed shape cannot freeze the entry. `RUFFLE_ENTRY_HOLD_MS` sets another time
    /// (comparisons), `RUFFLE_NO_ENTRY_HOLD` none (as it was since cached objects stopped
    /// holding the frame).
    fn entry_hold() -> Option<Duration> {
        static HOLD: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
        *HOLD.get_or_init(|| {
            if std::env::var_os("RUFFLE_NO_ENTRY_HOLD").is_some() {
                return None;
            }
            let ms = std::env::var("RUFFLE_ENTRY_HOLD_MS")
                .ok()
                .and_then(|ms| ms.trim().parse::<u64>().ok())
                .unwrap_or(100);
            (ms > 0).then(|| Duration::from_millis(ms.min(2000)))
        })
    }

    /// Whether the cached object drawn into `handle` was drawn blank, kept its last picture,
    /// or was drawn with a part that was (see `resolve`), because shapes weren't ready, AND
    /// those shapes are done now: then it must be drawn again. While they're still being
    /// tessellated the answer is false, so the object isn't redrawn blank every picture (on a
    /// laptop that doubled the cached-object redraws per picture, and each redraw bumps the
    /// texture's version, which made every layer showing the object miss the layer cache
    /// too). Asked once per draw of the object; a true answer clears the entry.
    ///
    /// An object waits for every shape its cached parts wait for, so when it answers true,
    /// the parts it walks again answer true as well, in the same walk, and it samples their
    /// finished pictures (their entries precede its own).
    pub fn take_cache_pending(handle: &BitmapHandle) -> bool {
        let key = cache_key(handle);
        let waiting = PENDING_CACHES.with_borrow(|pending| {
            pending.get(&key).map(|(_, tickets)| {
                RUNNING.with_borrow(|running| {
                    tickets.iter().any(|ticket| running.contains_key(ticket))
                })
            })
        });
        match waiting {
            None => false,
            Some(true) => false,
            Some(false) => {
                PENDING_CACHES.with_borrow_mut(|pending| pending.remove(&key));
                true
            }
        }
    }

    /// Adds `tickets` to what the cached object drawn into `handle` waits for.
    fn mark_cache_pending(handle: &BitmapHandle, tickets: &[u64]) {
        PENDING_CACHES.with_borrow_mut(|pending| {
            // Objects removed before they were redrawn leave their entries behind: drop the
            // ones whose texture is gone, now and then.
            if pending.len() >= 1024 {
                pending.retain(|_, (handle, _)| handle.strong_count() > 0);
            }
            let (_, waiting) = pending
                .entry(cache_key(handle))
                .or_insert_with(|| (Arc::downgrade(&handle.0), Vec::new()));
            for &ticket in tickets {
                if !waiting.contains(&ticket) {
                    waiting.push(ticket);
                }
            }
        });
    }

    /// The contents of the cached object drawn into `handle` are walked from now until
    /// `leave_cache`: an entry pushed meanwhile is sampled by that object's entry (and by
    /// those of the cached objects around it), so it lists them as its ancestors. Nested
    /// like the walk; only the walk tells (a `BitmapData.draw` of part of the stage has no
    /// cached objects above it on the stage that take part).
    pub fn enter_cache(handle: &BitmapHandle) {
        CACHE_STACK.with_borrow_mut(|stack| stack.push(handle.clone()));
    }

    /// Ends the walk `enter_cache` started, returning the ancestors of the entry about to be
    /// pushed: the cached objects still being walked, innermost first.
    pub fn leave_cache() -> Vec<BitmapHandle> {
        CACHE_STACK.with_borrow_mut(|stack| {
            stack.pop();
            stack.iter().rev().cloned().collect()
        })
    }

    /// Lets go of every tessellation still queued or running, and of what waits for one: at
    /// shutdown, so that no graphics object of theirs outlives the renderer (this thread's
    /// storage is freed last of all, when the graphics driver is already gone). Returns how
    /// many were still running.
    pub fn shutdown() -> usize {
        ACTIVE.with(|active| active.set(false));
        for job in JOBS.with_borrow_mut(std::mem::take) {
            job.cache.borrow_mut().replace(&job.placeholder, None);
        }
        let running = RUNNING.with_borrow_mut(std::mem::take);
        let count = running.len();
        for job in running.into_values() {
            job.cache.borrow_mut().replace(&job.placeholder, None);
        }
        PENDING_CACHES.with_borrow_mut(|pending| pending.clear());
        CACHE_STACK.with_borrow_mut(|stack| stack.clear());
        READY.with_borrow_mut(|ready| ready.clear());
        count
    }

    fn cache_key(handle: &BitmapHandle) -> *const () {
        Arc::as_ptr(&handle.0) as *const ()
    }

    /// A tessellation this slow is logged, with where it's from.
    const SLOW_MICROS: u64 = 20_000;

    fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_SERIAL_SHAPES").is_none())
    }

    /// How long a frame waits for its new shapes before it's drawn without those still being
    /// tessellated (`None`, the default: until they're all done).
    pub fn set_wait(wait: Option<Duration>) {
        WAIT.with(|w| w.set(wait));
    }

    /// Whether tessellations are queued or still running (to be swapped in later), of those
    /// a recorded walk can hold (see `replay`). Not a tween's frames (`wait` false): tweens
    /// are drawn again in every picture, never from a recording, and a very detailed one has
    /// a frame on its way for all of its first loop.
    pub fn has_jobs() -> bool {
        JOBS.with_borrow(|jobs| jobs.iter().any(|job| job.wait))
            || RUNNING.with_borrow(|running| running.values().any(|job| job.wait))
    }

    pub fn is_active() -> bool {
        ACTIVE.with(Cell::get)
    }

    /// Whether `handle` stands in for a tessellation not done yet.
    pub fn is_placeholder(handle: &ShapeHandle) -> bool {
        let any: &dyn std::any::Any = &*handle.0;
        any.is::<Placeholder>()
    }

    /// Whether a shape still being tessellated may be drawn as something else for now (a
    /// tween as it was last drawn): in a frame that doesn't wait for all its shapes, and not
    /// inside a cached object, which is drawn once and kept (it's drawn again when its shapes
    /// are done, which takes their placeholders to be in its commands).
    pub fn may_stand_in() -> bool {
        is_active()
            && WAIT.with(Cell::get).is_some()
            && CACHE_STACK.with_borrow(|stack| stack.is_empty())
    }

    /// Queues a tessellation; returns its placeholder handle. `wait`: whether the frame waits
    /// for it (see `Job`).
    pub fn queue(
        movie: SwfMovie<'_>,
        shape: Arc<swf::Shape>,
        scale: Measure,
        cache: Rc<RefCell<TessellationCache>>,
        wait: bool,
    ) -> ShapeHandle {
        let placeholder = ShapeHandle(Arc::new(Placeholder));
        JOBS.with_borrow_mut(|jobs| {
            jobs.push(Job {
                movie: Library::movie_addr(movie),
                shape,
                scale,
                cache,
                placeholder: placeholder.clone(),
                wait,
            })
        });
        placeholder
    }

    /// Defers tessellations while it's alive (until `resolve`).
    pub struct Scope(());

    impl Scope {
        pub fn begin() -> Self {
            ACTIVE.with(|active| active.set(enabled()));
            // Left over from a walk that panicked, if anything.
            CACHE_STACK.with_borrow_mut(|stack| stack.clear());
            Self(())
        }

        /// Starts everything queued, waits for it as long as allowed (see the module), and
        /// puts what's done (from this frame or earlier ones) into `commands`, `cache_draws`
        /// and the shapes' caches; the draws of shapes still being tessellated are left out.
        pub fn resolve(
            self,
            library: &Library<'_>,
            renderer: &mut dyn RenderBackend,
            commands: &mut CommandList,
            cache_draws: &mut Vec<BitmapCacheEntry>,
        ) {
            ACTIVE.with(|active| active.set(false));
            let jobs = JOBS.with_borrow_mut(std::mem::take);
            if jobs.is_empty() && RUNNING.with_borrow(|running| running.is_empty()) {
                return;
            }
            let source_of = |movie: usize| {
                library
                    .library_at(movie)
                    .map(|library| MovieLibrarySource { library })
            };
            let url_of = |movie: usize| {
                library
                    .library_at(movie)
                    .map(|library| library.movie().url().to_owned())
                    .unwrap_or_default()
            };
            let mut results: HashMap<*const (), Option<ShapeHandle>> = HashMap::new();
            let new_count = jobs.len();
            let mut tickets = Vec::with_capacity(jobs.len());
            let mut sources = Vec::with_capacity(jobs.len());
            for job in jobs {
                let Some(source) = source_of(job.movie) else {
                    // Its movie's library is gone: nothing to tessellate it with.
                    job.cache.borrow_mut().replace(&job.placeholder, None);
                    results.insert(key(&job.placeholder), None);
                    continue;
                };
                let ticket = NEXT_TICKET.with(|next| next.replace(next.get() + 1));
                if job.wait || WAIT.with(Cell::get).is_none() {
                    tickets.push(ticket);
                }
                sources.push((ticket, source, job.shape.clone(), job.scale));
                RUNNING.with_borrow_mut(|running| running.insert(ticket, job));
            }

            // The tickets the frame can't do without (see the module), this frame's or not.
            let by_placeholder: HashMap<*const (), u64> = RUNNING.with_borrow(|running| {
                running
                    .iter()
                    .map(|(&ticket, job)| (key(&job.placeholder), ticket))
                    .collect()
            });
            let mut needed = HashSet::new();
            needs(commands, &by_placeholder, false, &mut needed);
            for entry in cache_draws.iter() {
                needs(&entry.commands, &by_placeholder, wait_for_cached(), &mut needed);
            }
            let needed: Vec<u64> = needed.into_iter().collect();

            let work = sources
                .iter()
                .map(|(ticket, source, shape, scale)| {
                    (*ticket, shape.clone(), scale.scale, scale.stretch, source as &dyn BitmapSource)
                })
                .collect();
            // What earlier frames' budgets left over comes first (see `upload_budget`).
            let mut done = READY.with_borrow_mut(std::mem::take);
            done.extend(renderer.start_shapes(work));
            let waiting_since = Instant::now();
            let deadline = WAIT.with(Cell::get).map(|wait| waiting_since + wait);
            done.extend(renderer.collect_shapes(deadline, &tickets, &needed));
            // A room-entry picture is held for its new cached objects' shapes (`entry_hold`).
            if let Some(hold) = entry_hold()
                && deadline.is_some()
                && new_count >= ENTRY_SHAPES
                && !wait_for_cached()
            {
                let mut held = HashSet::new();
                for entry in cache_draws.iter().filter(|entry| entry.fresh) {
                    needs(&entry.commands, &by_placeholder, true, &mut held);
                }
                // (Those done by now are no longer running, and are not waited for. Nor is a
                // job the frame never waits for, a very detailed tween's frame: it is known
                // to take longer than the hold.)
                let held: Vec<u64> = RUNNING.with_borrow(|running| {
                    held.into_iter()
                        .filter(|ticket| running.get(ticket).is_some_and(|job| job.wait))
                        .collect()
                });
                if !held.is_empty() {
                    let before = done.len();
                    let held_since = Instant::now();
                    done.extend(renderer.collect_shapes(Some(waiting_since + hold), &held, &[]));
                    let held_ms = held_since.elapsed().as_secs_f32() * 1000.0;
                    if held_ms >= 1.0 {
                        tracing::info!(
                            target: "ruffle_aqw_perf",
                            "room entry: held {held_ms:.1} ms more for the shapes of new cached objects ({} of {new_count} new shapes came in meanwhile)",
                            done.len() - before
                        );
                    }
                }
            }

            // Only a frame that does not wait for all its shapes may put uploads off.
            let budget = upload_budget().filter(|_| WAIT.with(Cell::get).is_some());
            let uploading_since = Instant::now();
            let mut put_off = 0usize;
            for (ticket, tessellated) in done {
                if let Some(budget) = budget
                    && uploading_since.elapsed() >= budget
                    && !needed.contains(&ticket)
                {
                    READY.with_borrow_mut(|ready| ready.push((ticket, tessellated)));
                    put_off += 1;
                    continue;
                }
                let Some(job) = RUNNING.with_borrow_mut(|running| running.remove(&ticket)) else {
                    continue;
                };
                let (micros, vertices) = (tessellated.micros, tessellated.vertices);
                let handle = source_of(job.movie)
                    .and_then(|source| renderer.upload_tessellated(tessellated, &source));
                // AQW build: a single shape that took long to tessellate, with where it's from
                // and what it was drawn at.
                // (Not a tween's frames: every one of a very detailed tween's is slow, and
                // the tween says so itself, once.)
                if micros >= SLOW_MICROS && job.wait {
                    tracing::info!(
                        target: "ruffle_aqw_perf",
                        "slow shape: {} ms for shape {} of {} ({} records, {:.0}x{:.0} px, drawn at scale {:.2}) -> {} vertices",
                        micros / 1000,
                        job.shape.id,
                        url_of(job.movie),
                        job.shape.shape.len(),
                        job.shape.shape_bounds.width().to_pixels(),
                        job.shape.shape_bounds.height().to_pixels(),
                        job.scale.scale,
                        vertices
                    );
                }
                // AQW build: which movie a shape drawn with some hard edges is from.
                if ruffle_render::tessellator::take_tangled(job.shape.id) {
                    tracing::info!(
                        target: "ruffle_aqw_perf",
                        "Flash-style edges: shape {} of {} has a fill whose outline crosses itself; drawn with hard edges",
                        job.shape.id,
                        url_of(job.movie)
                    );
                }
                job.cache.borrow_mut().replace(&job.placeholder, handle.clone());
                results.insert(key(&job.placeholder), handle);
            }

            // Still being tessellated: left out of this frame.
            let running: HashSet<*const ()> = RUNNING.with_borrow(|running| {
                running.values().map(|job| key(&job.placeholder)).collect()
            });
            let waiting = tickets.iter().filter(|t| RUNNING.with_borrow(|r| r.contains_key(t))).count();
            if waiting > 0 || put_off > 0 {
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "{waiting} of {new_count} new shapes not ready in time; drawn when done{}",
                    if put_off > 0 {
                        format!(" ({put_off} shapes are tessellated and wait for their upload: this frame's upload budget is used up)")
                    } else {
                        String::new()
                    }
                );
            }
            patch(commands, &results, &running);
            let (mut blank, mut kept, mut partial) = (0usize, 0usize, 0usize);
            // The cache textures whose entries wait for a shape in this resolve: the entries
            // that sample them come later in the list (children before their parent).
            let mut waiting_below: HashSet<*const ()> = HashSet::new();
            cache_draws.retain_mut(|entry| {
                // A cached object is drawn once and kept, so it can't be drawn without a shape:
                // if one of its shapes is still being tessellated, it's drawn again once they're
                // done (see `take_cache_pending`), and so is every cached object it's drawn
                // into (its ancestors), which would otherwise keep the incomplete picture of it
                // for good (a cached map clip stayed without the floor's rocks on a laptop).
                // Until then the object, and an ancestor, keeps its last picture, or, in a new
                // texture (which holds nothing, or a pooled texture's old picture), is drawn
                // this time: the object blank (the texture cleared, the filters skipped), an
                // ancestor with the object as it is.
                let waits_for: Vec<u64> = running_in(&entry.commands, &running)
                    .iter()
                    .filter_map(|placeholder| by_placeholder.get(placeholder).copied())
                    .collect();
                let own = !waits_for.is_empty();
                let below = waiting_below.contains(&cache_key(&entry.handle));
                if wait_for_cached() || (!own && !below) {
                    patch(&mut entry.commands, &results, &running);
                    return true;
                }
                if own {
                    mark_cache_pending(&entry.handle, &waits_for);
                    for ancestor in &entry.ancestors {
                        mark_cache_pending(ancestor, &waits_for);
                        waiting_below.insert(cache_key(ancestor));
                    }
                }
                if !entry.fresh {
                    kept += 1;
                    return false;
                }
                if own {
                    entry.commands = CommandList::new();
                    entry.filters.clear();
                    blank += 1;
                    return true;
                }
                patch(&mut entry.commands, &results, &running);
                partial += 1;
                true
            });
            if blank + kept + partial > 0 {
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "{blank} cached objects drawn blank, {kept} kept their last picture, {partial} drawn with a blank part: their shapes aren't ready; redrawn when done"
                );
            }
        }
    }

    /// The placeholders of the shapes `list` draws whose tessellation is still `running`.
    fn running_in(list: &CommandList, running: &HashSet<*const ()>) -> Vec<*const ()> {
        let mut found = Vec::new();
        collect_running(list, running, &mut found);
        found
    }

    fn collect_running(list: &CommandList, running: &HashSet<*const ()>, found: &mut Vec<*const ()>) {
        for command in &list.commands {
            match command {
                Command::RenderShape { shape, .. } => {
                    let k = key(shape);
                    if running.contains(&k) && !found.contains(&k) {
                        found.push(k);
                    }
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    collect_running(maskee_commands, running, found);
                    collect_running(mask_commands, running, found);
                }
                Command::Blend(commands, _) => collect_running(commands, running, found),
                _ => {}
            }
        }
    }

    /// Adds the tickets of the running tessellations drawn in `list` that the frame can't do
    /// without: in a mask, in an Erase or Alpha blend layer, or anywhere in a cached object's
    /// content (`cached`).
    fn needs(
        list: &CommandList,
        by_placeholder: &HashMap<*const (), u64>,
        needed_here: bool,
        needed: &mut HashSet<u64>,
    ) {
        // Between PushMask and ActivateMask (and DeactivateMask and PopMask) the shapes drawn
        // are the mask.
        let mut in_mask = false;
        for command in &list.commands {
            match command {
                Command::PushMask | Command::DeactivateMask => in_mask = true,
                Command::ActivateMask | Command::PopMask => in_mask = false,
                Command::RenderShape { shape, .. } if needed_here || in_mask => {
                    if let Some(&ticket) = by_placeholder.get(&key(shape)) {
                        needed.insert(ticket);
                    }
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    needs(maskee_commands, by_placeholder, needed_here, needed);
                    needs(mask_commands, by_placeholder, true, needed);
                }
                Command::Blend(commands, mode) => {
                    let cuts = matches!(
                        mode,
                        RenderBlendMode::Builtin(BlendMode::Erase | BlendMode::Alpha)
                    );
                    needs(commands, by_placeholder, needed_here || cuts, needed);
                }
                _ => {}
            }
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            // Only matters if `resolve` didn't run (a panic while rendering): no placeholder
            // may stay in a cache, where a later frame would find it.
            ACTIVE.with(|active| active.set(false));
            CACHE_STACK.with_borrow_mut(|stack| stack.clear());
            for job in JOBS.with_borrow_mut(std::mem::take) {
                job.cache.borrow_mut().replace(&job.placeholder, None);
            }
        }
    }

    fn key(handle: &ShapeHandle) -> *const () {
        Arc::as_ptr(&handle.0) as *const ()
    }

    /// Swaps every placeholder in `list` for its tessellation; drops the draw if it has none,
    /// or if it's still `running`.
    fn patch(
        list: &mut CommandList,
        results: &HashMap<*const (), Option<ShapeHandle>>,
        running: &HashSet<*const ()>,
    ) {
        list.commands.retain_mut(|command| match command {
            Command::RenderShape { shape, .. } => match results.get(&key(shape)) {
                Some(Some(handle)) => {
                    *shape = handle.clone();
                    true
                }
                Some(None) => false,
                None => !running.contains(&key(shape)),
            },
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => {
                patch(maskee_commands, results, running);
                patch(mask_commands, results, running);
                true
            }
            Command::Blend(commands, _) => {
                patch(commands, results, running);
                true
            }
            _ => true,
        });
    }
}

/// AQW build: 9-slice scaling of shapes (see `Graphic::sliced_handle`).
///
/// Like Flash, each point of the shape's outlines (and curve control points) is moved on its
/// own, per axis: left of the grid it keeps its distance to the scaled left edge of the
/// parent's bounds, right of it its distance to the scaled right edge, and inside it's spread
/// evenly between the two (same for y). The parent's scale comes from its own matrix, so
/// corners still scale with everything above it (as in Flash, where a tooltip's 7 px corners
/// grow with the window). Gradient and bitmap fills are scaled normally, and line widths keep
/// their size (the parent's scale isn't applied to them).
pub(crate) mod scale9 {
    use ruffle_render::matrix::Matrix;
    use std::sync::Arc;
    use swf::{
        FillStyle, LineStyle, Point, PointDelta, Rectangle, ShapeRecord, ShapeStyles, Twips,
    };

    /// Sliced shapes kept per shape (sizes in use at the same time, e.g. several panels).
    pub const CACHE_SIZE: usize = 8;

    pub type Entry = (
        Key,
        Arc<swf::Shape>,
        std::rc::Rc<std::cell::RefCell<crate::tessellation_cache::TessellationCache>>,
    );

    /// One axis: the parent's bounds, its grid lines and its scale.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Axis {
        min: f64,
        max: f64,
        grid_min: f64,
        grid_max: f64,
        scale: f64,
    }

    impl Axis {
        fn new(
            min: Twips,
            max: Twips,
            grid_min: Twips,
            grid_max: Twips,
            scale: f64,
        ) -> Option<Self> {
            let (min, max) = (min.get() as f64, max.get() as f64);
            if max <= min {
                return None;
            }
            let grid_min = (grid_min.get() as f64).clamp(min, max);
            let grid_max = (grid_max.get() as f64).clamp(min, max);
            (grid_max > grid_min).then_some(Self {
                min,
                max,
                grid_min,
                grid_max,
                scale,
            })
        }

        /// Where `v` (in the parent's coordinates, twips) goes once the parent is scaled.
        fn map(&self, v: f64) -> f64 {
            let start = self.min * self.scale;
            let end = self.max * self.scale;
            let fixed_min = self.grid_min - self.min;
            let fixed_max = self.max - self.grid_max;
            if fixed_min + fixed_max > end - start {
                // Scaled smaller than the parts outside the grid: shrink those to fit.
                let k = (end - start) / (fixed_min + fixed_max);
                return if v <= self.grid_min {
                    start + (v - self.min) * k
                } else if v >= self.grid_max {
                    end - (self.max - v) * k
                } else {
                    start + fixed_min * k
                };
            }
            if v <= self.grid_min {
                start + (v - self.min)
            } else if v >= self.grid_max {
                end - (self.max - v)
            } else {
                let middle_start = start + fixed_min;
                let middle_end = end - fixed_max;
                middle_start
                    + (v - self.grid_min) * (middle_end - middle_start)
                        / (self.grid_max - self.grid_min)
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Slicing {
        x: Axis,
        y: Axis,
    }

    impl Slicing {
        pub fn new(
            bounds: Rectangle<Twips>,
            grid: Rectangle<Twips>,
            scale_x: f64,
            scale_y: f64,
        ) -> Option<Self> {
            Some(Self {
                x: Axis::new(bounds.x_min, bounds.x_max, grid.x_min, grid.x_max, scale_x)?,
                y: Axis::new(bounds.y_min, bounds.y_max, grid.y_min, grid.y_max, scale_y)?,
            })
        }

        fn map(&self, (x, y): (f64, f64)) -> (f64, f64) {
            (self.x.map(x), self.y.map(y))
        }
    }

    /// Identifies a sliced shape: the shape's matrix in the parent, and the slicing.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Key {
        child: [f32; 4],
        child_translation: (Twips, Twips),
        slicing: Slicing,
    }

    impl Key {
        pub fn new(child: &Matrix, slicing: &Slicing) -> Self {
            Self {
                child: [child.a, child.b, child.c, child.d],
                child_translation: (child.tx, child.ty),
                slicing: *slicing,
            }
        }
    }

    /// A point of the shape (twips) in the parent's coordinates.
    fn to_parent(m: &Matrix, x: f64, y: f64) -> (f64, f64) {
        (
            m.a as f64 * x + m.c as f64 * y + m.tx.get() as f64,
            m.b as f64 * x + m.d as f64 * y + m.ty.get() as f64,
        )
    }

    fn round((x, y): (f64, f64)) -> (i32, i32) {
        (x.round() as i32, y.round() as i32)
    }

    fn fill(style: &FillStyle, fill_matrix: &Matrix) -> FillStyle {
        let transformed = |m: &swf::Matrix| swf::Matrix::from(*fill_matrix * Matrix::from(*m));
        match style {
            FillStyle::Color(_) => style.clone(),
            FillStyle::LinearGradient(g) => FillStyle::LinearGradient(swf::Gradient {
                matrix: transformed(&g.matrix),
                ..g.clone()
            }),
            FillStyle::RadialGradient(g) => FillStyle::RadialGradient(swf::Gradient {
                matrix: transformed(&g.matrix),
                ..g.clone()
            }),
            FillStyle::FocalGradient {
                gradient,
                focal_point,
            } => FillStyle::FocalGradient {
                gradient: swf::Gradient {
                    matrix: transformed(&gradient.matrix),
                    ..gradient.clone()
                },
                focal_point: *focal_point,
            },
            FillStyle::Bitmap {
                id,
                matrix,
                is_smoothed,
                is_repeating,
            } => FillStyle::Bitmap {
                id: *id,
                matrix: transformed(matrix),
                is_smoothed: *is_smoothed,
                is_repeating: *is_repeating,
            },
        }
    }

    fn styles(styles: &ShapeStyles, fill_matrix: &Matrix, line_scale: f64) -> ShapeStyles {
        ShapeStyles {
            fill_styles: styles
                .fill_styles
                .iter()
                .map(|style| fill(style, fill_matrix))
                .collect(),
            line_styles: styles
                .line_styles
                .iter()
                .map(|line: &LineStyle| {
                    let width = Twips::new((line.width().get() as f64 * line_scale).round() as i32);
                    line.clone()
                        .with_width(width)
                        .with_fill_style(fill(line.fill_style(), fill_matrix))
                })
                .collect(),
        }
    }

    /// `shape` with its points sliced, in the scaled parent's coordinates.
    pub fn slice_shape(shape: &swf::Shape, child: &Matrix, slicing: &Slicing) -> swf::Shape {
        let map = |x: f64, y: f64| round(slicing.map(to_parent(child, x, y)));
        // Fills are scaled normally: shape -> parent -> parent's scale.
        let fill_matrix = Matrix::scale(slicing.x.scale as f32, slicing.y.scale as f32) * *child;
        // Line widths keep their size in the parent.
        let line_scale = (((child.a * child.a + child.b * child.b) as f64).sqrt()
            + ((child.c * child.c + child.d * child.d) as f64).sqrt())
            / 2.0;

        let mut records = Vec::with_capacity(shape.shape.len() + 1);
        // Drawing starts at the shape's origin, which moves too.
        let mut pen = (0.0, 0.0);
        let mut out = map(0.0, 0.0);
        records.push(ShapeRecord::StyleChange(Box::new(swf::StyleChangeData {
            move_to: Some(Point::new(Twips::new(out.0), Twips::new(out.1))),
            fill_style_0: None,
            fill_style_1: None,
            line_style: None,
            new_styles: None,
        })));
        for record in &shape.shape {
            match record {
                ShapeRecord::StyleChange(change) => {
                    let mut change = (**change).clone();
                    if let Some(to) = change.move_to {
                        pen = (to.x.get() as f64, to.y.get() as f64);
                        out = map(pen.0, pen.1);
                        change.move_to = Some(Point::new(Twips::new(out.0), Twips::new(out.1)));
                    }
                    if let Some(new_styles) = &change.new_styles {
                        change.new_styles = Some(styles(new_styles, &fill_matrix, line_scale));
                    }
                    records.push(ShapeRecord::StyleChange(Box::new(change)));
                }
                ShapeRecord::StraightEdge { delta } => {
                    pen.0 += delta.dx.get() as f64;
                    pen.1 += delta.dy.get() as f64;
                    let to = map(pen.0, pen.1);
                    records.push(ShapeRecord::StraightEdge {
                        delta: PointDelta::new(Twips::new(to.0 - out.0), Twips::new(to.1 - out.1)),
                    });
                    out = to;
                }
                ShapeRecord::CurvedEdge {
                    control_delta,
                    anchor_delta,
                } => {
                    let control_pen = (
                        pen.0 + control_delta.dx.get() as f64,
                        pen.1 + control_delta.dy.get() as f64,
                    );
                    pen = (
                        control_pen.0 + anchor_delta.dx.get() as f64,
                        control_pen.1 + anchor_delta.dy.get() as f64,
                    );
                    let control = map(control_pen.0, control_pen.1);
                    let to = map(pen.0, pen.1);
                    records.push(ShapeRecord::CurvedEdge {
                        control_delta: PointDelta::new(
                            Twips::new(control.0 - out.0),
                            Twips::new(control.1 - out.1),
                        ),
                        anchor_delta: PointDelta::new(
                            Twips::new(to.0 - control.0),
                            Twips::new(to.1 - control.1),
                        ),
                    });
                    out = to;
                }
            }
        }

        // The mapping keeps the order of points per axis, so bounds map corner to corner.
        let map_rect = |r: &Rectangle<Twips>| {
            let corners = [
                to_parent(child, r.x_min.get() as f64, r.y_min.get() as f64),
                to_parent(child, r.x_max.get() as f64, r.y_min.get() as f64),
                to_parent(child, r.x_min.get() as f64, r.y_max.get() as f64),
                to_parent(child, r.x_max.get() as f64, r.y_max.get() as f64),
            ];
            let (mut x_min, mut y_min) = (f64::MAX, f64::MAX);
            let (mut x_max, mut y_max) = (f64::MIN, f64::MIN);
            for corner in corners {
                let (x, y) = slicing.map(corner);
                x_min = x_min.min(x);
                y_min = y_min.min(y);
                x_max = x_max.max(x);
                y_max = y_max.max(y);
            }
            Rectangle {
                x_min: Twips::new(x_min.floor() as i32),
                y_min: Twips::new(y_min.floor() as i32),
                x_max: Twips::new(x_max.ceil() as i32),
                y_max: Twips::new(y_max.ceil() as i32),
            }
        };

        swf::Shape {
            version: shape.version,
            id: shape.id,
            shape_bounds: map_rect(&shape.shape_bounds),
            edge_bounds: map_rect(&shape.edge_bounds),
            flags: shape.flags,
            styles: styles(&shape.styles, &fill_matrix, line_scale),
            shape: records,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn px(v: f64) -> Twips {
            Twips::from_pixels(v)
        }

        fn rect(x0: f64, x1: f64, y0: f64, y1: f64) -> Rectangle<Twips> {
            Rectangle {
                x_min: px(x0),
                x_max: px(x1),
                y_min: px(y0),
                y_max: px(y1),
            }
        }

        #[test]
        fn corners_keep_their_size() {
            // AQW's tooltip background: a 213 x 20 box, grid x 12..200, y 7..12.75,
            // stretched to 265 x 270.
            let slicing = Slicing::new(
                rect(0.0, 213.0, 0.0, 20.0),
                rect(12.0, 200.0, 7.0, 12.75),
                265.0 / 213.0,
                13.5,
            )
            .unwrap();
            let p = |x: f64, y: f64| {
                let (x, y) = slicing.map((x * 20.0, y * 20.0));
                (x / 20.0, y / 20.0)
            };
            // Outer edges scale normally.
            assert!(p(0.0, 0.0).0.abs() < 1e-9);
            assert!((p(213.0, 20.0).0 - 265.0).abs() < 1e-6);
            assert!((p(213.0, 20.0).1 - 270.0).abs() < 1e-6);
            // Corner points keep their distance to the edges.
            let (x, y) = p(7.0, 2.05);
            assert!((x - 7.0).abs() < 1e-6 && (y - 2.05).abs() < 1e-6);
            let (x, y) = p(210.95, 17.95);
            assert!((x - (265.0 - 2.05)).abs() < 1e-6 && (y - (270.0 - 2.05)).abs() < 1e-6);
            // The grid lines land their fixed distances in.
            assert!((p(12.0, 0.0).0 - 12.0).abs() < 1e-6);
            assert!((p(200.0, 0.0).0 - (265.0 - 13.0)).abs() < 1e-6);
        }

        #[test]
        fn shrinking_below_the_corners_scales_them_down() {
            let slicing = Slicing::new(
                rect(0.0, 100.0, 0.0, 100.0),
                rect(20.0, 80.0, 20.0, 80.0),
                0.2,
                1.0,
            )
            .unwrap();
            // 20 px wide: the two 20 px sides shrink to 10 px each.
            let (x, _) = slicing.map((px(20.0).get() as f64, 0.0));
            assert!((x / 20.0 - 10.0).abs() < 1e-6);
            let (x, _) = slicing.map((px(100.0).get() as f64, 0.0));
            assert!((x / 20.0 - 20.0).abs() < 1e-6);
        }

        #[test]
        fn sliced_shape_follows_the_mapping() {
            // A line from (5, 5) to (95, 5), grid 10..90, parent scaled 3x by 1x.
            let shape = swf::Shape {
                version: 4,
                id: 1,
                shape_bounds: rect(0.0, 100.0, 0.0, 100.0),
                edge_bounds: rect(0.0, 100.0, 0.0, 100.0),
                flags: swf::ShapeFlag::empty(),
                styles: ShapeStyles {
                    fill_styles: vec![],
                    line_styles: vec![],
                },
                shape: vec![
                    ShapeRecord::StyleChange(Box::new(swf::StyleChangeData {
                        move_to: Some(Point::new(px(5.0), px(5.0))),
                        fill_style_0: None,
                        fill_style_1: None,
                        line_style: None,
                        new_styles: None,
                    })),
                    ShapeRecord::StraightEdge {
                        delta: PointDelta::new(px(90.0), px(0.0)),
                    },
                ],
            };
            let slicing = Slicing::new(
                rect(0.0, 100.0, 0.0, 100.0),
                rect(10.0, 90.0, 10.0, 90.0),
                3.0,
                1.0,
            )
            .unwrap();
            let sliced = slice_shape(&shape, &Matrix::IDENTITY, &slicing);
            let ShapeRecord::StyleChange(change) = &sliced.shape[1] else {
                panic!("expected a move")
            };
            assert_eq!(change.move_to, Some(Point::new(px(5.0), px(5.0))));
            let ShapeRecord::StraightEdge { delta } = &sliced.shape[2] else {
                panic!("expected an edge")
            };
            // 95 -> 300 - 5 = 295.
            assert_eq!(delta.dx, px(290.0));
            assert_eq!(sliced.shape_bounds.x_max, px(300.0));
        }
    }
}
