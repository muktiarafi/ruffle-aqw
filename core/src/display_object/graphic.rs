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
use crate::tag_utils::{SwfMovie, SwfSlice};
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
use swf::extensions::ReadSwfExt;

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
    shared: Lock<Gc<'gc, GraphicShared>>,
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

    /// Construct a `Graphic` from its DefineShape tag (`tag`, the tag's body, of DefineShape
    /// `version`).
    ///
    /// AQW build: only the tag's header (ID and bounds) is read now; the shape is parsed when
    /// it's first drawn or hit-tested (see `GraphicShared::shape`). Every DefineShape of every
    /// loaded SWF used to be parsed as the SWF loaded, drawn or not: a third of the heap after
    /// some play (most shapes of maps, monsters and other players' gear are never drawn, and
    /// none are without a window).
    pub fn from_swf_tag(
        context: &mut UpdateContext<'gc>,
        tag: SwfSlice,
        version: u8,
        movie: Arc<SwfMovie>,
    ) -> Result<Self, swf::error::Error> {
        let mut header = tag.read_from(0);
        let id = header.read_u16()?;
        let shape_bounds = header.read_rectangle()?;
        let edge_bounds = if version >= 4 {
            header.read_rectangle()?
        } else {
            shape_bounds.clone()
        };
        let shared = GraphicShared {
            id,
            shape: OnceCell::new(),
            tag: Some((tag, version)),
            shape_bounds,
            edge_bounds,
            movie,
            scaled_handle: TessellationCache::new_shared(),
            sliced: Default::default(),
        };
        Ok(Self::with_shared(context, shared))
    }

    /// Construct a `Graphic` from a parsed shape.
    /// AQW build: `swf_shape` may be shared with other graphics (see `shared_shapes`).
    pub fn from_parsed_shape(
        context: &mut UpdateContext<'gc>,
        swf_shape: Arc<swf::Shape>,
        movie: Arc<SwfMovie>,
    ) -> Self {
        // AQW build: tessellated when first drawn, at the scale it's drawn at (see
        // `deferred_shapes`).
        let shared = GraphicShared {
            id: swf_shape.id,
            shape_bounds: swf_shape.shape_bounds.clone(),
            edge_bounds: swf_shape.edge_bounds.clone(),
            shape: OnceCell::from(swf_shape),
            tag: None,
            movie,
            scaled_handle: TessellationCache::new_shared(),
            sliced: Default::default(),
        };
        Self::with_shared(context, shared)
    }

    fn with_shared(context: &mut UpdateContext<'gc>, shared: GraphicShared) -> Self {
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
            shape: OnceCell::from(Arc::new(empty_shape(
                0,
                Default::default(),
                Default::default(),
            ))),
            tag: None,
            movie: context.root_swf.clone(),
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

    fn set_shared(self, mc: &Mutation<'gc>, shared: Gc<'gc, GraphicShared>) {
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
        context: &mut RenderContext,
        current_scale: Measure,
    ) -> Option<ShapeHandle> {
        let shared = self.0.shared.get();
        Self::handle_for(
            context,
            &shared.movie,
            shared.shape(),
            &shared.scaled_handle,
            current_scale,
        )
    }

    /// The tessellation of `shape` at about `current_scale`, from `cache` or made now (or
    /// queued, see `deferred_shapes`).
    pub(crate) fn handle_for(
        context: &mut RenderContext,
        movie: &Arc<SwfMovie>,
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
                let library = context.library.library_for_movie(movie.clone())?;
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
        let library = context.library.library_for_movie(movie.clone())?;
        let new_handle = if deferred_shapes::is_active() {
            deferred_shapes::queue(movie.clone(), shape.clone(), current_scale, cache.clone())
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
    fn sliced_handle(self, context: &mut RenderContext) -> Option<(ShapeHandle, Matrix)> {
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
                        Arc::new(scale9::slice_shape(shared.shape(), &child_matrix, &slicing)),
                        TessellationCache::new_shared(),
                    )
                }
            };
            // Most recently used last.
            let found = (entry.1.clone(), entry.2.clone());
            sliced.push(entry);
            found
        };
        let handle = Self::handle_for(context, &shared.movie, &shape, &cache, current_scale)?;
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
        } else if self.0.shared.get().id != 0 {
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
                        shared.movie.clone(),
                        shared.shape().clone(),
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
                let shared = self.0.shared.get();
                return ruffle_render::shape_utils::shape_hit_test(
                    shared.shape(),
                    point,
                    &local_matrix,
                );
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

    fn movie(self) -> Arc<SwfMovie> {
        self.0.shared.get().movie.clone()
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
#[collect(require_static)]
struct GraphicShared {
    id: CharacterId,
    /// AQW build: parsed from `tag` when first needed (see `shape`).
    shape: OnceCell<Arc<swf::Shape>>,
    /// AQW build: the DefineShape tag's body and version, for `shape`.
    tag: Option<(SwfSlice, u8)>,
    shape_bounds: Rectangle<Twips>,
    edge_bounds: Rectangle<Twips>,
    movie: Arc<SwfMovie>,
    /// AQW build: shared with pending deferred tessellations (see `deferred_shapes`).
    #[collect(require_static)]
    scaled_handle: Rc<RefCell<TessellationCache>>,
    /// AQW build: 9-slice scaled versions of the shape, least recently used first (see
    /// `Graphic::sliced_handle`).
    #[collect(require_static)]
    sliced: RefCell<Vec<scale9::Entry>>,
}

impl GraphicShared {
    /// AQW build: the shape, parsed from its tag the first time it's asked for (an identical
    /// tag's parse if one is still in use, see `shared_shapes`). A tag that doesn't parse gives
    /// an empty shape (it used to leave the character out).
    fn shape(&self) -> &Arc<swf::Shape> {
        self.shape.get_or_init(|| {
            let parsed = self.tag.as_ref().map(|(tag, version)| {
                shared_shapes::parsed(tag.data(), *version, || {
                    tag.read_from(0).read_define_shape(*version)
                })
            });
            if let Some(Err(e)) = &parsed {
                tracing::error!("DefineShape {} didn't parse: {e}", self.id);
            }
            match parsed {
                Some(Ok(shape)) => shape,
                _ => Arc::new(empty_shape(
                    self.id,
                    self.shape_bounds.clone(),
                    self.edge_bounds.clone(),
                )),
            }
        })
    }
}

fn empty_shape(
    id: CharacterId,
    shape_bounds: Rectangle<Twips>,
    edge_bounds: Rectangle<Twips>,
) -> swf::Shape {
    swf::Shape {
        version: 32,
        id,
        shape_bounds,
        edge_bounds,
        flags: swf::ShapeFlag::empty(),
        styles: swf::ShapeStyles {
            fill_styles: Vec::new(),
            line_styles: Vec::new(),
        },
        shape: Vec::new(),
    }
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
/// content or show it unmasked), a cached object's content (cacheAsBitmap: it's drawn once and
/// kept, so it would stay without the shape), and a layer erasing or cutting out (Erase, Alpha
/// blends). Renders outside `Player::render` (e.g. `BitmapData.draw`) still tessellate
/// immediately. `RUFFLE_SERIAL_SHAPES` tessellates each shape immediately, one at a time.
pub mod deferred_shapes {
    use crate::library::{Library, MovieLibrarySource};
    use crate::tag_utils::SwfMovie;
    use crate::tessellation_cache::{Measure, TessellationCache};
    use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ShapeHandle, ShapeHandleImpl};
    use ruffle_render::bitmap::BitmapSource;
    use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, HashSet};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use swf::BlendMode;

    /// Stands in for a tessellation that isn't done yet. Never reaches a renderer.
    #[derive(Debug)]
    struct Placeholder;

    impl ShapeHandleImpl for Placeholder {}

    struct Job {
        movie: Arc<SwfMovie>,
        shape: Arc<swf::Shape>,
        scale: Measure,
        cache: Rc<RefCell<TessellationCache>>,
        placeholder: ShapeHandle,
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
    }

    /// Turu: one player's queued and running tessellations while another player's are the
    /// thread's (see `crate::PlayerThreadState`). Jobs are started on a player's own renderer,
    /// so only that player's frames can finish them.
    #[derive(Default)]
    pub struct State {
        active: bool,
        jobs: Vec<Job>,
        running: HashMap<u64, Job>,
    }

    /// Turu: swaps `state` with the thread's.
    pub(crate) fn swap(state: &mut State) {
        state.active = ACTIVE.replace(state.active);
        JOBS.with_borrow_mut(|jobs| std::mem::swap(jobs, &mut state.jobs));
        RUNNING.with_borrow_mut(|running| std::mem::swap(running, &mut state.running));
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

    /// Whether tessellations are queued or still running (to be swapped in later).
    pub fn has_jobs() -> bool {
        JOBS.with_borrow(|jobs| !jobs.is_empty()) || RUNNING.with_borrow(|running| !running.is_empty())
    }

    pub fn is_active() -> bool {
        ACTIVE.with(Cell::get)
    }

    /// Whether `handle` stands in for a tessellation not done yet.
    pub fn is_placeholder(handle: &ShapeHandle) -> bool {
        let any: &dyn std::any::Any = &*handle.0;
        any.is::<Placeholder>()
    }

    /// Queues a tessellation; returns its placeholder handle.
    pub fn queue(
        movie: Arc<SwfMovie>,
        shape: Arc<swf::Shape>,
        scale: Measure,
        cache: Rc<RefCell<TessellationCache>>,
    ) -> ShapeHandle {
        let placeholder = ShapeHandle(Arc::new(Placeholder));
        JOBS.with_borrow_mut(|jobs| {
            jobs.push(Job {
                movie,
                shape,
                scale,
                cache,
                placeholder: placeholder.clone(),
            })
        });
        placeholder
    }

    /// Defers tessellations while it's alive (until `resolve`).
    pub struct Scope(());

    impl Scope {
        pub fn begin() -> Self {
            ACTIVE.with(|active| active.set(enabled()));
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
            cache_draws: &mut [BitmapCacheEntry],
        ) {
            ACTIVE.with(|active| active.set(false));
            let jobs = JOBS.with_borrow_mut(std::mem::take);
            if jobs.is_empty() && RUNNING.with_borrow(|running| running.is_empty()) {
                return;
            }
            let source_of = |movie: &Arc<SwfMovie>| {
                library
                    .library_for_movie(movie.clone())
                    .map(|library| MovieLibrarySource { library })
            };
            let mut results: HashMap<*const (), Option<ShapeHandle>> = HashMap::new();
            let new_count = jobs.len();
            let mut tickets = Vec::with_capacity(jobs.len());
            let mut sources = Vec::with_capacity(jobs.len());
            for job in jobs {
                let Some(source) = source_of(&job.movie) else {
                    // Its movie's library is gone: nothing to tessellate it with.
                    job.cache.borrow_mut().replace(&job.placeholder, None);
                    results.insert(key(&job.placeholder), None);
                    continue;
                };
                let ticket = NEXT_TICKET.with(|next| next.replace(next.get() + 1));
                tickets.push(ticket);
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
                needs(&entry.commands, &by_placeholder, true, &mut needed);
            }
            let needed: Vec<u64> = needed.into_iter().collect();

            let work = sources
                .iter()
                .map(|(ticket, source, shape, scale)| {
                    (*ticket, shape.clone(), scale.scale, scale.stretch, source as &dyn BitmapSource)
                })
                .collect();
            let mut done = renderer.start_shapes(work);
            let deadline = WAIT.with(Cell::get).map(|wait| Instant::now() + wait);
            done.extend(renderer.collect_shapes(deadline, &tickets, &needed));

            for (ticket, tessellated) in done {
                let Some(job) = RUNNING.with_borrow_mut(|running| running.remove(&ticket)) else {
                    continue;
                };
                let (micros, vertices) = (tessellated.micros, tessellated.vertices);
                let handle = source_of(&job.movie)
                    .and_then(|source| renderer.upload_tessellated(tessellated, &source));
                // AQW build: a single shape that took long to tessellate, with where it's from
                // and what it was drawn at.
                if micros >= SLOW_MICROS {
                    tracing::info!(
                        target: "ruffle_aqw_perf",
                        "slow shape: {} ms for shape {} of {} ({} records, {:.0}x{:.0} px, drawn at scale {:.2}) -> {} vertices",
                        micros / 1000,
                        job.shape.id,
                        job.movie.url(),
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
                        job.movie.url()
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
            if waiting > 0 {
                tracing::info!(
                    target: "ruffle_aqw_perf",
                    "{waiting} of {new_count} new shapes not ready in time; drawn when done"
                );
            }
            patch(commands, &results, &running);
            for entry in cache_draws {
                patch(&mut entry.commands, &results, &running);
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
