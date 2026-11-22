use crate::backend::{NO_STRETCH, Stretch};
use crate::bitmap::BitmapSource;
use crate::shape_utils::{DistilledShape, DrawCommand, DrawPath, GradientType};
use indexmap::IndexSet;
use lyon::path::Path;
use lyon::tessellation::{
    self, FillTessellator, FillVertex, StrokeTessellator, StrokeVertex, StrokeVertexConstructor,
    geometry_builder::{BuffersBuilder, FillVertexConstructor, VertexBuffers},
};
use lyon::tessellation::{FillOptions, StrokeOptions};
use swf::GradientRecord;
use tracing::instrument;

pub struct ShapeTessellator {
    fill_tess: FillTessellator,
    stroke_tess: StrokeTessellator,
    mesh: Vec<Draw>,
    gradients: IndexSet<Gradient>,
    lyon_mesh: VertexBuffers<Vertex, u32>,
    mask_index_count: Option<u32>,
    is_stroke: bool,
}

const TESSELLATION_EPSILON: f32 = 0.0000001;

/// AQW build: Flash-style edge smoothing (see `edge_aa`), set by the stage quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeMode {
    /// No soft edges (MSAA).
    Off = 0,
    /// Low quality: only thin lines (strokes up to 1.5 pixels wide) are soft; without, a
    /// 1-pixel line between two rows of pixels broke up into dots.
    Lines = 1,
    /// High quality: half a pixel each side of the true edge (a box filter's ramp, like
    /// many-sample MSAA).
    Soft = 2,
    /// Medium quality: `Soft` with fewer triangles. With no MSAA at any quality, the fades are
    /// all that High costs a frame over Low: twice the triangles, a tenth of the frame rate
    /// on a built-in graphics chip in a crowd (nothing on a fast card). Here a shape's edges
    /// are soft where it ends against whatever is behind it and hard between its own fills
    /// (where one reaches under the other); a thin stretch's fades are weakened as a whole
    /// instead of getting a third row of vertices; and on smooth curves every other outer
    /// point of a fade is left out. (Tried and dropped: curves flattened half as finely, which
    /// saved few triangles and put outlines a fifth of a pixel off, worse than Low's on one
    /// sheet in nine; and, once, narrower fades, a quarter pixel each side: thinner than the
    /// pixels they're sampled at, so a thin line's brightness depended on where it fell
    /// between pixels.)
    Outlines = 3,
}

static EDGE_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// AQW build: the soft edges of shapes tessellated from now on. A tessellation is made for one
/// mode: callers keep the mode with it and tessellate again when it changes (`edge_mode`).
pub fn set_edge_mode(mode: EdgeMode) {
    EDGE_MODE.store(mode as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn edge_mode() -> EdgeMode {
    match EDGE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => EdgeMode::Lines,
        2 => EdgeMode::Soft,
        3 => EdgeMode::Outlines,
        _ => EdgeMode::Off,
    }
}

/// AQW build: whether shapes get Flash-style soft edges.
pub fn edge_aa() -> bool {
    edge_mode() != EdgeMode::Off
}

/// AQW build: shapes with a fill drawn with hard edges, as its outline crosses itself or
/// another of its loops (see `edge_aa::centered_fill`), for the log: their ids, until taken.
static TANGLED_SHAPES: std::sync::Mutex<Vec<swf::CharacterId>> = std::sync::Mutex::new(Vec::new());

fn note_tangled(id: swf::CharacterId) {
    if let Ok(mut ids) = TANGLED_SHAPES.lock()
        && !ids.contains(&id)
        && ids.len() < 64
    {
        ids.push(id);
    }
}

/// Whether shape `id` had such a fill since last asked (then forgets it).
pub fn take_tangled(id: swf::CharacterId) -> bool {
    let Ok(mut ids) = TANGLED_SHAPES.lock() else {
        return false;
    };
    match ids.iter().position(|&i| i == id) {
        Some(index) => {
            ids.swap_remove(index);
            true
        }
        None => false,
    }
}

/// AQW build: whether Flash-style edge smoothing replaces MSAA: `RUFFLE_FLASH_AA` at start,
/// or `set_flash_aa` (Mugetsu's settings). 0 = not read yet, 1 = off, 2 = on.
static FLASH_AA: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn flash_aa_switch() -> bool {
    match FLASH_AA.load(std::sync::atomic::Ordering::Relaxed) {
        0 => {
            let on = std::env::var_os("RUFFLE_FLASH_AA").is_some();
            set_flash_aa(on);
            on
        }
        1 => false,
        _ => true,
    }
}

/// AQW build: switches Flash-style edge smoothing; it applies once the drawing surface is made
/// again (`RenderBackend::set_quality`), and shapes are tessellated again as they're drawn.
pub fn set_flash_aa(on: bool) {
    FLASH_AA.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

impl ShapeTessellator {
    pub fn new() -> Self {
        Self {
            fill_tess: FillTessellator::new(),
            stroke_tess: StrokeTessellator::new(),
            mesh: Vec::new(),
            gradients: IndexSet::new(),
            lyon_mesh: VertexBuffers::new(),
            mask_index_count: None,
            is_stroke: false,
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub fn tessellate_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Mesh {
        self.tessellate_shape_with_scale(shape, bitmap_source, 1.0)
    }

    pub fn tessellate_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
    ) -> Mesh {
        self.tessellate_shape_stretched(shape, bitmap_source, scale, NO_STRETCH)
    }

    /// AQW build: `tessellate_shape_with_scale` for a shape drawn squashed: Flash-style soft
    /// edges are made half a pixel wide on screen in every direction (see `Stretch`).
    #[instrument(level = "debug", skip_all)]
    pub fn tessellate_shape_stretched(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        scale: f32,
        stretch: Stretch,
    ) -> Mesh {
        let squash = edge_aa::Squash::new(stretch);
        // AQW build: the scale hairlines get their minimum width from: Ruffle's (the average over
        // both axes). A squashed shape's `scale` is its area's, so with that a hairline was a full
        // pixel even across the squashed direction, where it's usually a third of one (a
        // background's red guide lines, stretched 7x wide, showed up as a stripe).
        let line_scale = scale * stretch[3];
        // Curves are flattened as finely as Ruffle's own scale asks (a squashed shape's `scale`
        // is its area's, smaller: curves came out coarse).
        let detail = line_scale;
        self.mesh = Vec::new();
        self.gradients = IndexSet::new();
        self.lyon_mesh = VertexBuffers::new();

        // AQW build: for Flash-style edges, which fill edges are shared between two fills.
        let mode = edge_mode();
        let edge_aa_on = mode != EdgeMode::Off;
        // (Medium: see `EdgeMode::Outlines`.)
        let medium = mode == EdgeMode::Outlines;
        let soft = mode == EdgeMode::Soft || medium;
        let shared_edges = if soft {
            edge_aa::Shared::new(&shape.paths, medium)
        } else {
            Default::default()
        };

        let shape_id = shape.id;
        for (path_index, path) in shape.paths.into_iter().enumerate() {
            // AQW build: whether this path's edges are soft (see `EdgeMode`).
            let edge_aa_on = edge_aa_on
                && match &path {
                    DrawPath::Fill { .. } => soft,
                    DrawPath::Stroke { style, .. } => {
                        soft || style.width().to_pixels() as f32 * scale.abs() <= 1.5
                    }
                };
            let (fill_style, lyon_path, next_is_stroke) = match &path {
                DrawPath::Fill {
                    style,
                    commands,
                    winding_rule: _,
                } => (*style, ruffle_path_to_lyon_path(commands, true), false),
                DrawPath::Stroke {
                    style,
                    commands,
                    is_closed,
                } => (
                    style.fill_style(),
                    ruffle_path_to_lyon_path(commands, *is_closed),
                    true,
                ),
            };

            let (draw, color, needs_flush) = match fill_style {
                swf::FillStyle::Color(color) => (DrawType::Color, *color, false),
                swf::FillStyle::LinearGradient(gradient) => {
                    let uniform =
                        swf_gradient_to_uniforms(GradientType::Linear, gradient, swf::Fixed8::ZERO);
                    let (gradient_index, _) = self.gradients.insert_full(uniform);

                    (
                        DrawType::Gradient {
                            matrix: swf_to_gl_matrix(gradient.matrix.into()),
                            gradient: gradient_index,
                        },
                        swf::Color::WHITE,
                        true,
                    )
                }
                swf::FillStyle::RadialGradient(gradient) => {
                    let uniform =
                        swf_gradient_to_uniforms(GradientType::Radial, gradient, swf::Fixed8::ZERO);
                    let (gradient_index, _) = self.gradients.insert_full(uniform);
                    (
                        DrawType::Gradient {
                            matrix: swf_to_gl_matrix(gradient.matrix.into()),
                            gradient: gradient_index,
                        },
                        swf::Color::WHITE,
                        true,
                    )
                }
                swf::FillStyle::FocalGradient {
                    gradient,
                    focal_point,
                } => {
                    let uniform =
                        swf_gradient_to_uniforms(GradientType::Focal, gradient, *focal_point);
                    let (gradient_index, _) = self.gradients.insert_full(uniform);
                    (
                        DrawType::Gradient {
                            matrix: swf_to_gl_matrix(gradient.matrix.into()),
                            gradient: gradient_index,
                        },
                        swf::Color::WHITE,
                        true,
                    )
                }
                swf::FillStyle::Bitmap {
                    id,
                    matrix,
                    is_smoothed,
                    is_repeating,
                } => {
                    if let Some(bitmap) = bitmap_source.bitmap_size(*id) {
                        (
                            DrawType::Bitmap(Bitmap {
                                matrix: swf_bitmap_to_gl_matrix(
                                    (*matrix).into(),
                                    bitmap.width,
                                    bitmap.height,
                                ),
                                bitmap_id: *id,
                                is_smoothed: *is_smoothed,
                                is_repeating: *is_repeating,
                            }),
                            swf::Color::WHITE,
                            true,
                        )
                    } else {
                        // Missing bitmap -- incorrect character ID in SWF?
                        continue;
                    }
                }
            };

            if needs_flush || (self.is_stroke && !next_is_stroke) {
                // We flush separate draw calls in these cases:
                // * Non-solid color fills which require their own shader.
                // * Strokes followed by fills, because strokes need to be omitted
                //   when using this shape as a mask.
                self.flush_draw(DrawType::Color);
            } else if !self.is_stroke && next_is_stroke {
                // Bake solid color fills followed by strokes into a single draw call, and adjust
                // the index count to omit the strokes when rendering this shape as a mask.
                assert!(self.mask_index_count.is_none());
                self.mask_index_count = Some(self.lyon_mesh.indices.len() as u32);
            }
            self.is_stroke = next_is_stroke;

            let mut buffers_builder =
                BuffersBuilder::new(&mut self.lyon_mesh, RuffleVertexCtor { color });
            // AQW build: Flash-style edges: the fill's outline moved half a pixel in (out where
            // it lies under another fill), plus a fade across the edge (see `edge_aa`).
            let centered = match &path {
                DrawPath::Fill {
                    commands,
                    winding_rule,
                    ..
                } if edge_aa_on => {
                    let centered = edge_aa::centered_fill(
                        commands,
                        &shared_edges,
                        path_index,
                        (*winding_rule).into(),
                        scale,
                        &squash,
                        medium,
                    );
                    if edge_aa::TANGLED.with(|t| t.replace(false)) {
                        note_tangled(shape_id);
                    }
                    centered
                }
                _ => None,
            };
            let result = match path {
                DrawPath::Fill { winding_rule, .. } => {
                    // Larger scales require more precise tessellation to avoid artifacts
                    let tolerance = FillOptions::DEFAULT_TOLERANCE / detail;
                    // A moved outline is wound so its inside is positive (see `centered_fill`).
                    let fill_rule = if centered.is_some() {
                        lyon::path::FillRule::Positive
                    } else {
                        winding_rule.into()
                    };
                    self.fill_tess.tessellate_path(
                        centered.as_ref().map_or(&lyon_path, |c| &c.outline),
                        &FillOptions::default()
                            .with_fill_rule(fill_rule)
                            .with_tolerance(tolerance),
                        &mut buffers_builder,
                    )
                }
                DrawPath::Stroke { style, .. } => {
                    // This calculation ensures that hairline strokes are rendered with
                    // a minimum width of 1 pixel, while still allowing for proper scaling
                    let width = style.width().to_pixels() as f32;
                    let min_screen_width = if f32::abs(line_scale) > TESSELLATION_EPSILON {
                        1.0 / line_scale
                    } else {
                        1.0
                    };
                    let width = width.max(min_screen_width);
                    // AQW build: with Flash-style edges, the body is a pixel narrower and the
                    // fades on both sides make up the rest (see `edge_aa::stroke`).
                    let width = if edge_aa_on {
                        edge_aa::stroke_body_width(width, scale)
                    } else {
                        width
                    };

                    // Larger scales require more precise tessellation to avoid artifacts
                    let tolerance = StrokeOptions::DEFAULT_TOLERANCE / detail;
                    let mut stroke_options = StrokeOptions::default()
                        .with_line_width(width)
                        .with_tolerance(tolerance)
                        .with_start_cap(match style.start_cap() {
                            swf::LineCapStyle::None => tessellation::LineCap::Butt,
                            swf::LineCapStyle::Round => tessellation::LineCap::Round,
                            swf::LineCapStyle::Square => tessellation::LineCap::Square,
                        })
                        .with_end_cap(match style.end_cap() {
                            swf::LineCapStyle::None => tessellation::LineCap::Butt,
                            swf::LineCapStyle::Round => tessellation::LineCap::Round,
                            swf::LineCapStyle::Square => tessellation::LineCap::Square,
                        });

                    let line_join = match style.join_style() {
                        swf::LineJoinStyle::Round => tessellation::LineJoin::Round,
                        swf::LineJoinStyle::Bevel => tessellation::LineJoin::Bevel,
                        swf::LineJoinStyle::Miter(limit) => {
                            // Avoid lyon assert with small miter limits.
                            let limit = limit.to_f32();
                            if limit >= StrokeOptions::MINIMUM_MITER_LIMIT {
                                stroke_options = stroke_options.with_miter_limit(limit);
                                tessellation::LineJoin::MiterClip
                            } else {
                                tessellation::LineJoin::Bevel
                            }
                        }
                    };
                    stroke_options = stroke_options.with_line_join(line_join);
                    if width <= TESSELLATION_EPSILON {
                        // A hairline: only its fades are drawn.
                        Ok(())
                    } else {
                        self.stroke_tess.tessellate_path(
                            &lyon_path,
                            &stroke_options,
                            &mut buffers_builder,
                        )
                    }
                }
            };
            // AQW build: Flash-style soft edges (see `edge_aa`).
            if result.is_ok() && edge_aa_on {
                match &path {
                    DrawPath::Fill { winding_rule, .. } => match &centered {
                        Some(centered) => {
                            for (part, alpha) in &centered.thin {
                                let mut builder = BuffersBuilder::new(
                                    &mut self.lyon_mesh,
                                    RuffleVertexCtor {
                                        color: edge_aa::with_alpha(color, *alpha),
                                    },
                                );
                                let _ = self.fill_tess.tessellate_path(
                                    part,
                                    &FillOptions::default()
                                        .with_tolerance(FillOptions::DEFAULT_TOLERANCE / detail),
                                    &mut builder,
                                );
                            }
                            edge_aa::push_fades(&mut self.lyon_mesh, centered, color)
                        }
                        None => edge_aa::fill(
                            &mut self.lyon_mesh,
                            &lyon_path,
                            (*winding_rule).into(),
                            color,
                            scale,
                            &squash,
                        ),
                    },
                    DrawPath::Stroke { style, .. } => {
                        let width = (style.width().to_pixels() as f32).max(
                            if line_scale.abs() > TESSELLATION_EPSILON {
                                1.0 / line_scale
                            } else {
                                1.0
                            },
                        );
                        edge_aa::stroke(
                            &mut self.lyon_mesh,
                            &lyon_path,
                            width,
                            (style.start_cap(), style.end_cap()),
                            color,
                            scale,
                            &squash,
                        );
                    }
                }
            }
            match result {
                Ok(_) => {
                    if needs_flush {
                        // Non-solid color fills are isolated draw calls; flush immediately.
                        self.flush_draw(draw);
                    }
                }
                Err(e) => {
                    // This may simply be a degenerate path.
                    tracing::error!("Tessellation failure: {:?}", e);
                }
            }
        }

        // Flush the final pending draw.
        self.flush_draw(DrawType::Color);

        self.lyon_mesh = VertexBuffers::new();
        Mesh {
            draws: std::mem::take(&mut self.mesh),
            gradients: std::mem::take(&mut self.gradients).into_iter().collect(),
        }
    }

    fn flush_draw(&mut self, draw: DrawType) {
        if self.lyon_mesh.vertices.is_empty() || self.lyon_mesh.indices.len() < 3 {
            // Ignore degenerate fills
            self.lyon_mesh = VertexBuffers::new();
            self.mask_index_count = None;
            return;
        }
        let draw_mesh = std::mem::replace(&mut self.lyon_mesh, VertexBuffers::new());
        self.mesh.push(Draw {
            draw_type: draw,
            mask_index_count: self
                .mask_index_count
                .unwrap_or(draw_mesh.indices.len() as u32),
            vertices: draw_mesh.vertices,
            indices: draw_mesh.indices,
        });
        self.mask_index_count = None;
    }
}

impl Default for ShapeTessellator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Mesh {
    pub draws: Vec<Draw>,
    pub gradients: Vec<Gradient>,
}

pub struct Draw {
    pub draw_type: DrawType,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub mask_index_count: u32,
}

pub enum DrawType {
    Color,
    Gradient {
        matrix: [[f32; 3]; 3],
        gradient: usize,
    },
    Bitmap(Bitmap),
}

impl DrawType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Color => "Color",
            Self::Gradient { .. } => "Gradient",
            Self::Bitmap { .. } => "Bitmap",
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Gradient {
    pub gradient_type: GradientType,
    pub repeat_mode: swf::GradientSpread,
    pub focal_point: swf::Fixed8,
    pub interpolation: swf::GradientInterpolation,
    pub records: Vec<GradientRecord>,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub color: swf::Color,
}

#[derive(Clone, Debug)]
pub struct Bitmap {
    pub matrix: [[f32; 3]; 3],
    pub bitmap_id: u16,
    pub is_smoothed: bool,
    pub is_repeating: bool,
}

#[expect(clippy::many_single_char_names)]
fn swf_to_gl_matrix(m: crate::matrix::Matrix) -> [[f32; 3]; 3] {
    let tx = m.tx.get() as f32;
    let ty = m.ty.get() as f32;
    let det = m.a * m.d - m.c * m.b;
    let mut a = m.d / det;
    let mut b = -m.c / det;
    let mut c = -(tx * m.d - m.c * ty) / det;
    let mut d = -m.b / det;
    let mut e = m.a / det;
    let mut f = (tx * m.b - m.a * ty) / det;

    a *= 20.0 / 32768.0;
    b *= 20.0 / 32768.0;
    d *= 20.0 / 32768.0;
    e *= 20.0 / 32768.0;

    c /= 32768.0;
    f /= 32768.0;
    c += 0.5;
    f += 0.5;
    [[a, d, 0.0], [b, e, 0.0], [c, f, 1.0]]
}

#[expect(clippy::many_single_char_names)]
fn swf_bitmap_to_gl_matrix(
    m: crate::matrix::Matrix,
    bitmap_width: u32,
    bitmap_height: u32,
) -> [[f32; 3]; 3] {
    let bitmap_width = bitmap_width as f32;
    let bitmap_height = bitmap_height as f32;

    let tx = m.tx.get() as f32;
    let ty = m.ty.get() as f32;
    let det = m.a * m.d - m.c * m.b;
    let mut a = m.d / det;
    let mut b = -m.c / det;
    let mut c = -(tx * m.d - m.c * ty) / det;
    let mut d = -m.b / det;
    let mut e = m.a / det;
    let mut f = (tx * m.b - m.a * ty) / det;

    a *= 20.0 / bitmap_width;
    b *= 20.0 / bitmap_width;
    d *= 20.0 / bitmap_height;
    e *= 20.0 / bitmap_height;

    c /= bitmap_width;
    f /= bitmap_height;

    [[a, d, 0.0], [b, e, 0.0], [c, f, 1.0]]
}

fn ruffle_path_to_lyon_path(commands: &[DrawCommand], is_closed: bool) -> Path {
    fn point(point: swf::Point<swf::Twips>) -> lyon::math::Point {
        lyon::math::Point::new(point.x.to_pixels() as f32, point.y.to_pixels() as f32)
    }

    let mut builder = Path::builder();
    let mut cursor = Some(swf::Point::ZERO);
    for command in commands {
        match command {
            DrawCommand::MoveTo(move_to) => {
                if cursor.is_none() {
                    builder.end(false);
                }
                cursor = Some(*move_to);
            }
            DrawCommand::LineTo(line_to) => {
                if let Some(cursor) = cursor.take() {
                    builder.begin(point(cursor));
                }
                builder.line_to(point(*line_to));
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                if let Some(cursor) = cursor.take() {
                    builder.begin(point(cursor));
                }
                builder.quadratic_bezier_to(point(*control), point(*anchor));
            }
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => {
                if let Some(cursor) = cursor.take() {
                    builder.begin(point(cursor));
                }
                builder.cubic_bezier_to(point(*control_a), point(*control_b), point(*anchor));
            }
        }
    }

    if cursor.is_none() {
        if is_closed {
            builder.close();
        } else {
            builder.end(false);
        }
    }

    builder.build()
}

/// AQW build: Flash-style edge smoothing. Flash didn't use MSAA: it computed how much of
/// each edge pixel a shape covers and used that as its alpha. Here each shape is drawn as usual
/// (pixels whose centres are inside are fully covered) plus a thin strip along its outline,
/// outside the shape, fading from half the shape's alpha at the edge to nothing half a pixel
/// out: the coverage a 1-pixel box filter gives the pixels just outside (the same scheme as
/// NanoVG's antialiasing). It costs a few triangles per edge, instead of 2-4 samples per pixel
/// and a resolve for every layer (MSAA), which is what made Medium and High slow on weak
/// graphics. Solid colors carry the fade in their vertex alpha; gradients and bitmaps in their
/// vertex color too (white, its alpha passed as the coverage of their `uv.z`).
mod edge_aa {
    use super::{FillOptions, Path, Vertex, VertexBuffers};
    use crate::shape_utils::{DrawCommand, DrawPath};
    use lyon::math::{Point, Vector};
    use lyon::path::PathEvent;
    use lyon::path::iterator::PathIterator as _;

    const EPSILON: f32 = 1e-12;

    /// Maps a shape's points to where it isn't squashed (screen space, up to a scale and a
    /// rotation; see `Stretch`), and back. The soft edges are worked out there, so they're
    /// half a pixel wide on screen in every direction (a floor circle drawn flat kept its thin
    /// line's edges too narrow across, and the line came out dashed).
    #[derive(Clone, Copy)]
    pub struct Squash {
        to: [f32; 4],
        from: [f32; 4],
    }

    impl Squash {
        pub fn new(stretch: super::Stretch) -> Self {
            // The symmetric square root of the stretch (its determinant is 1).
            let [xx, xy, yy, _] = stretch;
            let root = (xx + yy + 2.0).max(EPSILON).sqrt();
            let to = [(xx + 1.0) / root, xy / root, xy / root, (yy + 1.0) / root];
            let det = to[0] * to[3] - to[1] * to[2];
            let det = if det.abs() > EPSILON { det } else { 1.0 };
            let from = [to[3] / det, -to[1] / det, -to[2] / det, to[0] / det];
            Self { to, from }
        }

        fn map(m: &[f32; 4], p: Point) -> Point {
            Point::new(m[0] * p.x + m[1] * p.y, m[2] * p.x + m[3] * p.y)
        }

        fn apply(&self, p: Point) -> Point {
            Self::map(&self.to, p)
        }

        fn undo(&self, p: Point) -> Point {
            Self::map(&self.from, p)
        }

        /// How much farther a stroke reaches in direction `v` (a unit vector here) than in
        /// the shape: a stroke is round in the shape, and squashed into an ellipse here.
        fn widen(&self, v: Vector) -> f32 {
            let p = Self::map(&self.to, Point::new(v.x, v.y));
            p.x.hypot(p.y)
        }

        /// Maps the vertices pushed since `start` back to the shape.
        fn undo_vertices(&self, mesh: &mut VertexBuffers<Vertex, u32>, start: usize) {
            for v in &mut mesh.vertices[start..] {
                let p = self.undo(Point::new(v.x, v.y));
                (v.x, v.y) = (p.x, p.y);
            }
        }

        fn polylines(&self, path: &Path, tolerance: f32) -> Vec<(Vec<Point>, bool)> {
            let mut lines = polylines(path, tolerance);
            for (points, _) in &mut lines {
                for p in points.iter_mut() {
                    *p = self.apply(*p);
                }
            }
            lines
        }
    }

    /// The polylines of `path`, flattened; `true` for closed ones.
    fn polylines(path: &Path, tolerance: f32) -> Vec<(Vec<Point>, bool)> {
        let mut lines: Vec<(Vec<Point>, bool)> = Vec::new();
        for event in path.iter().flattened(tolerance) {
            match event {
                PathEvent::Begin { at } => lines.push((vec![at], false)),
                PathEvent::Line { to, .. } => {
                    if let Some((line, _)) = lines.last_mut()
                        && line
                            .last()
                            .is_none_or(|last| (*last - to).square_length() > EPSILON)
                    {
                        line.push(to);
                    }
                }
                PathEvent::End { close, .. } => {
                    if let Some((line, closed)) = lines.last_mut() {
                        if line.len() > 2
                            && (line[0] - line[line.len() - 1]).square_length() <= EPSILON
                        {
                            line.pop();
                            *closed = true;
                        }
                        *closed |= close;
                    }
                }
                _ => {}
            }
        }
        lines
    }

    fn left_normal(a: Point, b: Point) -> Vector {
        let d = (b - a).normalize();
        Vector::new(-d.y, d.x)
    }

    /// Unit left normals averaged at each vertex, lengthened for the miter (at most 2x).
    fn vertex_normals(points: &[Point], closed: bool) -> Vec<Vector> {
        let n = points.len();
        (0..n)
            .map(|i| {
                let prev = if i > 0 {
                    Some(left_normal(points[i - 1], points[i]))
                } else if closed {
                    Some(left_normal(points[n - 1], points[0]))
                } else {
                    None
                };
                let next = if i + 1 < n {
                    Some(left_normal(points[i], points[i + 1]))
                } else if closed {
                    Some(left_normal(points[n - 1], points[0]))
                } else {
                    None
                };
                match (prev, next) {
                    (Some(a), Some(b)) => {
                        let sum = a + b;
                        if sum.square_length() < EPSILON {
                            b
                        } else {
                            let miter = sum.normalize();
                            miter / miter.dot(b).max(0.5)
                        }
                    }
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => Vector::zero(),
                }
            })
            .collect()
    }

    /// A strip from `points` (at half the alpha) out by `offsets` (to none).
    fn push_strip(
        mesh: &mut VertexBuffers<Vertex, u32>,
        points: &[Point],
        offsets: &[Vector],
        closed: bool,
        color: swf::Color,
    ) {
        push_strip_from(
            mesh,
            points,
            offsets,
            closed,
            swf::Color {
                a: color.a / 2,
                ..color
            },
            color,
        );
    }

    /// A strip from `points` (in `edge`) out by `offsets` (to `color` with no alpha).
    fn push_strip_from(
        mesh: &mut VertexBuffers<Vertex, u32>,
        points: &[Point],
        offsets: &[Vector],
        closed: bool,
        edge: swf::Color,
        color: swf::Color,
    ) {
        let n = points.len();
        if n < 2 {
            return;
        }
        let clear = swf::Color { a: 0, ..color };
        let base = mesh.vertices.len() as u32;
        for (point, offset) in points.iter().zip(offsets) {
            let outer = *point + *offset;
            mesh.vertices.push(Vertex {
                x: point.x,
                y: point.y,
                color: edge,
            });
            mesh.vertices.push(Vertex {
                x: outer.x,
                y: outer.y,
                color: clear,
            });
        }
        let segments = if closed { n } else { n - 1 } as u32;
        for i in 0..segments {
            let j = (i + 1) % n as u32;
            let (inner_i, outer_i) = (base + 2 * i, base + 2 * i + 1);
            let (inner_j, outer_j) = (base + 2 * j, base + 2 * j + 1);
            mesh.indices
                .extend_from_slice(&[inner_i, outer_i, outer_j, inner_i, outer_j, inner_j]);
        }
    }

    /// Half a pixel at the scale the shape is tessellated at, and the flattening tolerance.
    fn sizes(scale: f32) -> Option<(f32, f32)> {
        let half = 0.5;
        (scale.abs() > 1e-7).then(|| {
            (
                half / scale.abs(),
                FillOptions::DEFAULT_TOLERANCE / scale.abs(),
            )
        })
    }

    /// An edge as the SWF gives it, the same whichever way it's walked: its end points (and
    /// control points) in twips.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct EdgeKey([i32; 8]);

    fn bytemuck_bytes(key: &[i32; 8]) -> &[u8] {
        // SAFETY: `[i32; 8]` is 32 plain bytes with no padding.
        unsafe { std::slice::from_raw_parts(key.as_ptr().cast::<u8>(), 32) }
    }

    /// Also whether it's walked forward (from its lower end): fills are wound with their inside
    /// on the same side of the way they walk (`shape_utils` flips fill style 0's edges), so two
    /// fills on opposite sides of an edge walk it opposite ways, and a fill lying over another
    /// along it (a later layer of the shape ending on the same outline) walks it the same way.
    fn edge_key(from: swf::Point<swf::Twips>, command: &DrawCommand) -> Option<(EdgeKey, bool)> {
        let p = |q: swf::Point<swf::Twips>| [q.x.get(), q.y.get()];
        let (a, b) = (p(from), p(command.end_point()));
        let forward = a <= b;
        let (lo, hi) = if forward { (a, b) } else { (b, a) };
        let key = EdgeKey(match command {
            DrawCommand::MoveTo(_) => return None,
            DrawCommand::LineTo(_) => [lo[0], lo[1], hi[0], hi[1], 0, 0, i32::MIN, 0],
            DrawCommand::QuadraticCurveTo { control, .. } => {
                let c = p(*control);
                [lo[0], lo[1], hi[0], hi[1], c[0], c[1], i32::MIN + 1, 0]
            }
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                ..
            } => {
                let (c1, c2) = if forward {
                    (p(*control_a), p(*control_b))
                } else {
                    (p(*control_b), p(*control_a))
                };
                [lo[0], lo[1], hi[0], hi[1], c1[0], c1[1], c2[0], c2[1]]
            }
        });
        Some((key, forward))
    }

    /// Which fills each edge of a shape belongs to (see `Side`).
    /// Each fill edge's `Side`, worked out for a whole shape at once (by sorting its edges, so
    /// edges shared between fills end up next to each other).
    #[derive(Default)]
    pub struct Shared {
        /// Per path (by index in `paths`; empty for strokes), per command.
        sides: Vec<Vec<Side>>,
    }

    /// The line a straight edge is on (its direction, reduced, and offset) and where along the
    /// line it starts and ends, in twips; the same whichever way it's walked.
    fn line_of(
        from: swf::Point<swf::Twips>,
        to: swf::Point<swf::Twips>,
    ) -> Option<([i64; 3], i64, i64, bool)> {
        let (x0, y0) = (i64::from(from.x.get()), i64::from(from.y.get()));
        let (x1, y1) = (i64::from(to.x.get()), i64::from(to.y.get()));
        let (mut dx, mut dy) = (x1 - x0, y1 - y0);
        if dx == 0 && dy == 0 {
            return None;
        }
        // Their greatest common divisor, by shifts and subtractions (dividing, as Euclid's way
        // does at every step, was most of the time this took: a seventh of all the time spent
        // preparing shapes of several fills).
        let a = {
            let (mut a, mut b) = (dx.unsigned_abs(), dy.unsigned_abs());
            if a == 0 || b == 0 {
                a | b
            } else {
                let shift = (a | b).trailing_zeros();
                a >>= a.trailing_zeros();
                loop {
                    b >>= b.trailing_zeros();
                    if a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    b -= a;
                    if b == 0 {
                        break a << shift;
                    }
                }
            }
        } as i64;
        (dx, dy) = (dx / a, dy / a);
        if dx < 0 || (dx == 0 && dy < 0) {
            (dx, dy) = (-dx, -dy);
        }
        let (t0, t1) = (dx * x0 + dy * y0, dx * x1 + dy * y1);
        Some(([dx, dy, dy * x0 - dx * y0], t0.min(t1), t0.max(t1), t0 < t1))
    }

    impl Shared {
        /// With `inner_hard` (Medium), a faded edge with a fill reaching under it from across
        /// the edge is `Hard` instead: not faded, ending where the edge is. (The fill under
        /// it still reaches under: left hard too, the background showed through a strand
        /// thinner than a pixel lying over it, whose own body is gone and whose fade isn't
        /// all there is of it: a light slit in dark hair.)
        pub fn new(paths: &[DrawPath], inner_hard: bool) -> Self {
            let mut sides: Vec<Vec<Side>> = paths
                .iter()
                .map(|path| match path {
                    DrawPath::Fill { commands, .. } => vec![Side::Faded; commands.len()],
                    DrawPath::Stroke { .. } => Vec::new(),
                })
                .collect();
            // (edge's hash, edge, fill, command), and straight edges as (line's hash, line, fill,
            // command, start, end): sorted by the hash first (cheaper than the whole key).
            let hash = |bytes: &[u8]| {
                use std::hash::BuildHasher;
                static STATE: std::sync::LazyLock<hashbrown::DefaultHashBuilder> =
                    std::sync::LazyLock::new(Default::default);
                STATE.hash_one(bytes)
            };
            let fills = paths
                .iter()
                .filter(|path| matches!(path, DrawPath::Fill { .. }))
                .count();
            // (Each also with the direction it's walked in, see `edge_key`.)
            let commands: usize = paths
                .iter()
                .map(|path| match path {
                    DrawPath::Fill { commands, .. } => commands.len(),
                    DrawPath::Stroke { .. } => 0,
                })
                .sum();
            let mut edges: Vec<(u64, EdgeKey, usize, usize, bool)> = Vec::with_capacity(commands);
            let mut lines: Vec<(u64, [i64; 3], usize, usize, i64, i64, bool)> =
                Vec::with_capacity(if fills > 1 { commands } else { 0 });
            for (fill, path) in paths.iter().enumerate() {
                let DrawPath::Fill { commands, .. } = path else {
                    continue;
                };
                let mut cursor = swf::Point::ZERO;
                for (command_index, command) in commands.iter().enumerate() {
                    if let Some((key, forward)) = edge_key(cursor, command) {
                        let h = hash(bytemuck_bytes(&key.0));
                        edges.push((h, key, fill, command_index, forward));
                    }
                    if fills > 1
                        && let DrawCommand::LineTo(to) = command
                        && let Some((line, t0, t1, forward)) = line_of(cursor, *to)
                    {
                        let mut bytes = [0u8; 24];
                        for (part, value) in bytes.chunks_exact_mut(8).zip(line) {
                            part.copy_from_slice(&value.to_le_bytes());
                        }
                        let h = hash(&bytes);
                        lines.push((h, line, fill, command_index, t0, t1, forward));
                    }
                    cursor = command.end_point();
                }
            }
            // Split differently (one fill has an edge whole, another in parts, which the exact
            // match below misses; the background then showed through a seam): below another
            // fill where most of it lies along a later fill's straight edges and no earlier's.
            lines.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            for group in lines.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
                if group.len() < 2 || group.len() > 64 {
                    continue;
                }
                for &(_, _, fill, command, t0, t1, forward) in group {
                    // Neighbours walk the line the other way (see `edge_key`). Where a later
                    // fill on the same side lies over this edge all along, it's left as it is.
                    let (mut later, mut earlier, mut overlaid) = (false, false, false);
                    for &(_, _, other, _, s0, s1, other_forward) in group {
                        if other == fill {
                            continue;
                        }
                        if other_forward == forward {
                            overlaid |= other > fill && s0 <= t0 && s1 >= t1;
                        } else if (s1.min(t1) - s0.max(t0)) * 2 >= t1 - t0 {
                            later |= other > fill;
                            earlier |= other < fill;
                        }
                    }
                    if later && !earlier && !overlaid {
                        sides[fill][command] =
                            if group.iter().all(|&(_, _, other, ..)| opaque(&paths[other])) {
                                Side::Under
                            } else {
                                Side::Hard
                            };
                    }
                }
            }
            // The same edge in several fills: below another fill when one drawn later shares it
            // and none drawn earlier does. Between two parts of the same fill there's no edge to
            // see: both extend under.
            edges.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            for group in edges.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
                if group.len() < 2 {
                    continue;
                }
                let all_opaque = group.iter().all(|e| opaque(&paths[e.2]));
                for &(_, _, fill, command, forward) in group {
                    // Only fills on the other side (walking the edge the other way) are
                    // neighbours. A later fill on the same side lies over this one, ending on
                    // the same outline (a dark part drawn over a red glow, both ending at an
                    // eye's rim): extended out under it, this one showed past the edge (a red
                    // line around the eye). Then both just fade, like two of Flash's layers.
                    // (Nor under a neighbour, while a later fill lies over it along the edge.)
                    let overlaid = group.iter().any(|e| e.2 > fill && e.4 == forward);
                    let mut neighbours = group.iter().filter(|e| e.4 != forward).map(|e| e.2);
                    let other = match neighbours.next() {
                        Some(other) if !overlaid => other,
                        _ => {
                            sides[fill][command] =
                                if all_opaque { Side::Faded } else { Side::Hard };
                            continue;
                        }
                    };
                    let (first, last) = neighbours
                        .fold((fill.min(other), fill.max(other)), |(lo, hi), e| {
                            (lo.min(e), hi.max(e))
                        });
                    sides[fill][command] = if !all_opaque {
                        Side::Hard
                    } else if first == last || (last > fill && first >= fill) {
                        Side::Under
                    } else {
                        Side::Faded
                    };
                }
                // Faded, with a fill across the edge that doesn't reach under this one, or
                // with a later fill on this side over it: a seam (see `Side::Seam`).
                for &(_, _, fill, command, forward) in group {
                    if sides[fill][command] != Side::Faded {
                        continue;
                    }
                    let overlaid = group.iter().any(|e| e.2 > fill && e.4 == forward);
                    let mut across = group.iter().filter(|e| e.4 != forward).peekable();
                    if overlaid
                        || (across.peek().is_some()
                            && !across.any(|e| sides[e.2][e.3] == Side::Under))
                    {
                        sides[fill][command] = Side::Seam;
                    } else if inner_hard && group.iter().any(|e| e.4 != forward) {
                        sides[fill][command] = Side::Hard;
                    }
                }
            }
            // Straight edges again: a faded one that lies along another fill's edge across
            // the line, where that fill doesn't reach under it (split differently and mostly
            // elsewhere, see the first pass), is a seam too.
            for group in lines.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
                if group.len() < 2 {
                    continue;
                }
                for &(_, _, fill, command, t0, t1, forward) in group {
                    if sides[fill][command] != Side::Faded {
                        continue;
                    }
                    let seam = if group.len() > 64 {
                        group
                            .iter()
                            .any(|other| other.2 != fill && other.6 != forward)
                    } else {
                        group
                            .iter()
                            .any(|&(_, _, other, other_command, s0, s1, other_forward)| {
                                other != fill
                                    && other_forward != forward
                                    && s1.min(t1) > s0.max(t0)
                                    && sides[other][other_command] != Side::Under
                            })
                    };
                    if seam {
                        sides[fill][command] = Side::Seam;
                    }
                }
            }
            Self { sides }
        }
    }

    /// A fade point's knee (see `Centered::fade_points`) that says it has none and that its
    /// outer point may be left out (Medium, see `push_fades`).
    const SPARE: f32 = 2.0;

    /// A fill's outline for Flash-style edges, and the fades to draw across its edges.
    pub struct Centered {
        pub outline: Path,
        /// Parts thinner than a pixel (see `centered_fill`), each drawn on its own at the
        /// alpha given (times the fill's).
        pub thin: Vec<(Path, f32)>,
        /// The fades: strips through these points, inner (at the run's alpha times the point's
        /// own, times the fill's) and outer (none), with the point's knee (see `Ring::knee`:
        /// 1 for none) ...
        fade_points: Vec<(Point, Point, f32, f32)>,
        /// ... as (first point, points, alpha, closed).
        fade_runs: Vec<(usize, usize, f32, bool)>,
    }

    /// The winding numbers of a fill's outline at `probes`, in one pass: the probes are sorted
    /// by height, so each segment only looks at the probes beside it (testing each contour
    /// against the whole outline made big shapes slow to prepare). They're kept in that order
    /// while counted, each coordinate in a row of its own, and every probe beside a segment
    /// is tested the same way without a branch, so the processor takes several at a time
    /// (looked up one by one through the order, this was the largest part of preparing a
    /// fill's edges).
    fn windings_at(rings: &[&[Point]], probes: &[Point]) -> Vec<i16> {
        let mut order: Vec<(f32, u32)> = probes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.y, i as u32))
            .collect();
        order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let ys: Vec<f32> = order.iter().map(|&(y, _)| y).collect();
        let xs: Vec<f32> = order.iter().map(|&(_, i)| probes[i as usize].x).collect();
        let mut counted = vec![0i16; probes.len()];
        for points in rings {
            for i in 0..points.len() {
                let (a, b) = (points[i], points[(i + 1) % points.len()]);
                if a.y == b.y {
                    continue;
                }
                // The probes with `min.y <= y < max.y` (where the edge crosses their row).
                let (lo, hi) = (a.y.min(b.y), a.y.max(b.y));
                let start = ys.partition_point(|&y| y < lo);
                let end = ys.partition_point(|&y| y < hi);
                let step: i16 = if b.y > a.y { 1 } else { -1 };
                let (run, rise) = (b.x - a.x, b.y - a.y);
                for ((x, y), count) in xs[start..end]
                    .iter()
                    .zip(&ys[start..end])
                    .zip(&mut counted[start..end])
                {
                    // (Left of the edge in its row.)
                    *count += step * i16::from(a.x + (y - a.y) * run / rise > *x);
                }
            }
        }
        let mut windings = vec![0i16; probes.len()];
        for (&(_, k), count) in order.iter().zip(counted) {
            windings[k as usize] = count;
        }
        windings
    }

    /// A look from a point on a fill's outline into the fill: from `from` along `dir` (a unit
    /// vector) as far as `reach`, past the edges `skip` of ring `ring` (the ones it starts on).
    struct Ray {
        from: Point,
        dir: Vector,
        reach: f32,
        ring: usize,
        skip: [usize; 2],
    }

    /// A fill's edges in a tree of boxes, for finding the ones a ray could meet. The edges are
    /// taken in order, ring after ring, eight to a leaf, and each level up has a box around
    /// each of eight of the nodes below: edges next to each other in a ring are next to each
    /// other in space, so the boxes are tight without sorting anything. Each coordinate of a
    /// node's eight boxes (and of a leaf's eight edges) is in a row of its own and all eight
    /// are tested the same way, without a branch, so the processor takes them at once: a ray
    /// a pixel long costs a few such tests. (A grid of the rays, every edge tried against all
    /// the rays in the cells it passed, was 15 tries a ray, a seventh of all the time spent
    /// preparing shapes; a tree of fours tested one by one was no better.)
    struct EdgeTree {
        leaves: Vec<Edges8>,
        /// The ring and edge each edge is.
        owners: Vec<(u32, u32)>,
        /// Level after level of nodes, the leaves' parents first; the last has one node.
        levels: Vec<Vec<Boxes8>>,
    }

    /// Eight edges: where each starts, and how far it goes (none, for a place left empty).
    #[derive(Clone, Copy)]
    struct Edges8 {
        ax: [f32; 8],
        ay: [f32; 8],
        ex: [f32; 8],
        ey: [f32; 8],
    }

    /// Eight boxes (inside out, for a place left empty: nothing meets it).
    #[derive(Clone, Copy)]
    struct Boxes8 {
        lo_x: [f32; 8],
        lo_y: [f32; 8],
        hi_x: [f32; 8],
        hi_y: [f32; 8],
    }

    impl Boxes8 {
        const EMPTY: Self = Self {
            lo_x: [f32::MAX; 8],
            lo_y: [f32::MAX; 8],
            hi_x: [f32::MIN; 8],
            hi_y: [f32::MIN; 8],
        };

        /// The box around all eight.
        fn around(&self) -> [f32; 4] {
            let mut around = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
            for k in 0..8 {
                around = [
                    around[0].min(self.lo_x[k]),
                    around[1].min(self.lo_y[k]),
                    around[2].max(self.hi_x[k]),
                    around[3].max(self.hi_y[k]),
                ];
            }
            around
        }

        fn set(&mut self, k: usize, around: [f32; 4]) {
            (self.lo_x[k], self.lo_y[k], self.hi_x[k], self.hi_y[k]) =
                (around[0], around[1], around[2], around[3]);
        }

        /// Which of the eight a box meets, a bit each.
        fn meeting(&self, reach: &[f32; 4]) -> u32 {
            let mut bits = 0u32;
            for k in 0..8 {
                let meets = (self.lo_x[k] <= reach[2])
                    & (self.hi_x[k] >= reach[0])
                    & (self.lo_y[k] <= reach[3])
                    & (self.hi_y[k] >= reach[1]);
                bits |= u32::from(meets) << k;
            }
            bits
        }
    }

    impl EdgeTree {
        fn new(rings: &[&[Point]]) -> Self {
            let count: usize = rings.iter().map(|points| points.len()).sum();
            let mut owners = Vec::with_capacity(count);
            let mut leaves = vec![
                Edges8 {
                    ax: [0.0; 8],
                    ay: [0.0; 8],
                    ex: [0.0; 8],
                    ey: [0.0; 8]
                };
                count.div_ceil(8).max(1)
            ];
            let mut level = vec![Boxes8::EMPTY; leaves.len().div_ceil(8)];
            for (r, points) in rings.iter().enumerate() {
                let n = points.len();
                for i in 0..n {
                    let (a, b) = (points[i], points[(i + 1) % n]);
                    let (leaf, k) = (owners.len() / 8, owners.len() % 8);
                    owners.push((r as u32, i as u32));
                    let edges = &mut leaves[leaf];
                    (edges.ax[k], edges.ay[k], edges.ex[k], edges.ey[k]) =
                        (a.x, a.y, b.x - a.x, b.y - a.y);
                    // (The leaf's box, in its parent.)
                    let boxes = &mut level[leaf / 8];
                    let at = leaf % 8;
                    boxes.lo_x[at] = boxes.lo_x[at].min(a.x.min(b.x));
                    boxes.lo_y[at] = boxes.lo_y[at].min(a.y.min(b.y));
                    boxes.hi_x[at] = boxes.hi_x[at].max(a.x.max(b.x));
                    boxes.hi_y[at] = boxes.hi_y[at].max(a.y.max(b.y));
                }
            }
            let mut levels = vec![level];
            while levels[levels.len() - 1].len() > 1 {
                let below = &levels[levels.len() - 1];
                let mut above = vec![Boxes8::EMPTY; below.len().div_ceil(8)];
                for (index, node) in below.iter().enumerate() {
                    above[index / 8].set(index % 8, node.around());
                }
                levels.push(above);
            }
            Self {
                leaves,
                owners,
                levels,
            }
        }

        /// Where the ray first meets the outline: (how far along it, the ring, the edge).
        fn hit(&self, ray: &Ray) -> Option<(f32, usize, usize)> {
            let end = ray.from + ray.dir * ray.reach;
            let reach = [
                ray.from.x.min(end.x),
                ray.from.y.min(end.y),
                ray.from.x.max(end.x),
                ray.from.y.max(end.y),
            ];
            let mut best: Option<(f32, usize, usize)> = None;
            // Nodes still to look into, as (level, node): at most seven more a level on the
            // way down.
            let mut pending = [(0u32, 0u32); 48];
            let mut waiting = 1;
            pending[0] = (self.levels.len() as u32 - 1, 0);
            while waiting > 0 {
                waiting -= 1;
                let (level, node) = (pending[waiting].0 as usize, pending[waiting].1 as usize);
                let mut bits = self.levels[level][node].meeting(&reach);
                while bits != 0 {
                    let child = node * 8 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if level > 0 {
                        if waiting < pending.len() {
                            pending[waiting] = (level as u32 - 1, child as u32);
                            waiting += 1;
                        }
                        continue;
                    }
                    // A leaf: where along each of its edges the ray's line crosses it, times
                    // `det` (outside the edge for nearly all of them; a shade over its ends
                    // counts), for all eight at once.
                    let edges = &self.leaves[child];
                    let (mut dets, mut on_edges) = ([0.0f32; 8], [0.0f32; 8]);
                    let mut crossed = 0u32;
                    for k in 0..8 {
                        let det = ray.dir.x * edges.ey[k] - ray.dir.y * edges.ex[k];
                        let (x, y) = (edges.ax[k] - ray.from.x, edges.ay[k] - ray.from.y);
                        let on_edge = x * ray.dir.y - y * ray.dir.x;
                        let (low, high) = (-1e-4 * det, 1.0001 * det);
                        let within = (det.abs() >= 1e-12)
                            & (on_edge >= low.min(high))
                            & (on_edge <= low.max(high));
                        (dets[k], on_edges[k]) = (det, on_edge);
                        crossed |= u32::from(within) << k;
                    }
                    let _ = on_edges;
                    while crossed != 0 {
                        let k = crossed.trailing_zeros() as usize;
                        crossed &= crossed - 1;
                        let (r, i) = self.owners[child * 8 + k];
                        let (r, i) = (r as usize, i as usize);
                        if ray.ring == r && (ray.skip[0] == i || ray.skip[1] == i) {
                            continue;
                        }
                        let (x, y) = (edges.ax[k] - ray.from.x, edges.ay[k] - ray.from.y);
                        let along = (x * edges.ey[k] - y * edges.ex[k]) / dets[k];
                        if along > ray.reach * 1e-3
                            && along <= ray.reach
                            && best.is_none_or(|(nearest, _, _)| along < nearest)
                        {
                            best = Some((along, r, i));
                        }
                    }
                }
            }
            best
        }
    }

    /// How strong the fades along a stretch of fill `t` pixels thick are drawn as a whole, so
    /// that the two (one drawn over the other, each a ramp a pixel wide centred on its edge)
    /// hold t of ink across: at strength f they hold f - f^2 * `overlap`. Medium's way with a
    /// thin stretch: the right weight, a little narrower and darker in the middle than the
    /// knee High gives it, without the third row of vertices.
    fn thin_fade(t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let i = overlap(t);
        if i < 1e-4 {
            return t;
        }
        ((1.0 - (1.0 - 4.0 * i * t).max(0.0).sqrt()) / (2.0 * i)).clamp(0.0, 1.0)
    }

    /// How strong the fade along a stretch of fill `t` pixels thick is drawn (`t` under 1)
    /// where its far edge lies under another fill: the stretch reaches half a pixel under that
    /// fill, whose own fade is drawn over both, and at full strength the stretch showed up to
    /// twice as much ink as it covers. (Its own ramp at strength f, times what the fill above
    /// lets through, holds t^2 / 2 + f * (t^2 / 2 + i) of ink across, with i the overlap
    /// below.)
    fn under_fade(t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        ((t - t * t / 2.0) / (t * t / 2.0 + overlap(t)).max(1e-6)).clamp(0.0, 1.0)
    }

    /// The integral of u * (t + 1 - u) for u from t to 1: two opposite ramps a pixel wide,
    /// t apart, multiplied where they lie over each other.
    fn overlap(t: f32) -> f32 {
        (t + 1.0) * (1.0 - t * t) / 2.0 - (1.0 - t * t * t) / 3.0
    }

    /// Whether `p` is inside the closed polygon `points` (even-odd).
    fn in_polygon(points: &[Point], p: Point) -> bool {
        let mut inside = false;
        for i in 0..points.len() {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            if (a.y <= p.y) != (b.y <= p.y) && a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) > p.x {
                inside = !inside;
            }
        }
        inside
    }

    /// `color` at `alpha` times its alpha.
    pub fn with_alpha(color: swf::Color, alpha: f32) -> swf::Color {
        swf::Color {
            a: (f32::from(color.a) * alpha).round() as u8,
            ..color
        }
    }

    /// Where an edge is: against the background or above another fill (faded across the
    /// edge, the fill moved in half a pixel), or below another fill (the fill extended half a
    /// pixel under it, no fade: the fill above fades over it, so no background shows through),
    /// or shared with a see-through fill (`Hard`: left where it is, not faded; the two fills
    /// meet exactly, as MSAA's samples would split them, since with see-through fills the
    /// extended one showed through the faded one: a bright line along a glow's two halves).
    ///
    /// `Seam` is `Faded` where the fill across the edge doesn't reach under this one: it
    /// fades too (two fills ending on the same outline, like two of Flash's layers), or a
    /// later fill on this side lies over this one. It's drawn like `Faded`; what's done for
    /// thin stretches and tips (`Ring::knee`) leaves it alone, as a weaker fade there has
    /// nothing under it but the background.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Side {
        Faded,
        Seam,
        Under,
        Hard,
    }

    impl Side {
        /// Faded across the edge, the fill moved in.
        fn fades(self) -> bool {
            matches!(self, Side::Faded | Side::Seam)
        }
    }

    /// Whether a fill is opaque everywhere (bitmaps may not be).
    fn opaque(path: &DrawPath) -> bool {
        match path {
            DrawPath::Fill { style, .. } => match style {
                swf::FillStyle::Color(color) => color.a == 255,
                swf::FillStyle::LinearGradient(gradient)
                | swf::FillStyle::RadialGradient(gradient)
                | swf::FillStyle::FocalGradient { gradient, .. } => {
                    gradient.records.iter().all(|record| record.color.a == 255)
                }
                swf::FillStyle::Bitmap { .. } => false,
            },
            DrawPath::Stroke { .. } => false,
        }
    }

    thread_local! {
        /// Set when `centered_fill` finds a fill it can't move (see there).
        pub static TANGLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// The outline of a fill moved for centered soft edges (see `Side`), or `None` where that
    /// can't be done cleanly (a fill whose outline crosses itself): then the outer-only fade
    /// is used.
    pub fn centered_fill(
        commands: &[DrawCommand],
        shared: &Shared,
        index: usize,
        fill_rule: lyon::path::FillRule,
        scale: f32,
        squash: &Squash,
        medium: bool,
    ) -> Option<Centered> {
        let (width, tolerance) = sizes(scale)?;
        // The outline, flattened, each segment with its side.
        let mut contours: Vec<(Vec<Point>, Vec<Side>)> = Vec::new();
        let point = |q: swf::Point<swf::Twips>| {
            squash.apply(Point::new(q.x.to_pixels() as f32, q.y.to_pixels() as f32))
        };
        let mut cursor = swf::Point::ZERO;
        let sides = shared.sides.get(index);
        for (command_index, command) in commands.iter().enumerate() {
            let side = sides
                .and_then(|sides| sides.get(command_index))
                .copied()
                .unwrap_or(Side::Faded);
            let from = point(cursor);
            let push = |to: Point, contours: &mut Vec<(Vec<Point>, Vec<Side>)>| {
                if let Some((points, sides)) = contours.last_mut()
                    && points
                        .last()
                        .is_some_and(|last| (*last - to).square_length() > EPSILON)
                {
                    points.push(to);
                    sides.push(side);
                }
            };
            match command {
                DrawCommand::MoveTo(to) => contours.push((vec![point(*to)], Vec::new())),
                DrawCommand::LineTo(to) => push(point(*to), &mut contours),
                DrawCommand::QuadraticCurveTo { control, anchor } => {
                    let curve = lyon::geom::QuadraticBezierSegment {
                        from,
                        ctrl: point(*control),
                        to: point(*anchor),
                    };
                    curve.for_each_flattened(tolerance, &mut |line| push(line.to, &mut contours));
                }
                DrawCommand::CubicCurveTo {
                    control_a,
                    control_b,
                    anchor,
                } => {
                    let curve = lyon::geom::CubicBezierSegment {
                        from,
                        ctrl1: point(*control_a),
                        ctrl2: point(*control_b),
                        to: point(*anchor),
                    };
                    curve.for_each_flattened(tolerance, &mut |line| push(line.to, &mut contours));
                }
            }
            cursor = command.end_point();
        }

        // Closed rings (the last point isn't the first again; an open one closes with a faded
        // edge), each edge with its length and left normal (worked out once here: they're
        // needed again and again below).
        struct Outline {
            points: Vec<Point>,
            sides: Vec<Side>,
            lengths: Vec<f32>,
            normals: Vec<Vector>,
            perimeter: f32,
            /// Twice the area, positive with the inside on the left.
            area: f32,
        }
        let area = |pts: &[Point]| -> f32 {
            (0..pts.len())
                .map(|i| {
                    pts[i]
                        .to_vector()
                        .cross(pts[(i + 1) % pts.len()].to_vector())
                })
                .sum::<f32>()
        };
        let rings: Vec<Outline> = contours
            .into_iter()
            .filter_map(|(mut points, mut sides)| {
                if points.len() > 1
                    && (points[0] - points[points.len() - 1]).square_length() <= EPSILON
                {
                    points.pop();
                } else {
                    sides.push(Side::Faded);
                }
                if points.len() < 3 || sides.len() != points.len() {
                    return None;
                }
                let n = points.len();
                let mut lengths = Vec::with_capacity(n);
                let mut normals = Vec::with_capacity(n);
                let mut perimeter = 0.0f32;
                for i in 0..n {
                    let edge = points[(i + 1) % n] - points[i];
                    let length = edge.length();
                    let d = edge / length;
                    lengths.push(length);
                    normals.push(Vector::new(-d.y, d.x));
                    perimeter += length;
                }
                let area = area(&points);
                // A ring with no area to speak of (out along a line and back: under a 50th of
                // a pixel thick on average) is left out: there's nothing of it to see, and
                // neither side of it is inside, so its votes below came from whatever lay
                // beside it, half each way: the whole fill was then drawn plain as tangled.
                if area.abs() < 0.04 * width * perimeter {
                    return None;
                }
                Some(Outline {
                    points,
                    sides,
                    lengths,
                    normals,
                    perimeter,
                    area,
                })
            })
            .collect();

        struct Ring {
            points: Vec<Point>,
            sides: Vec<Side>,
            lengths: Vec<f32>,
            normals: Vec<Vector>,
            left_is_inside: bool,
            /// Whether the tests that said so all agreed (see `votes`).
            sure: bool,
            moved: Vec<Point>,
            outer: Vec<Point>,
            /// Vertices whose two moved edges meet too far away (sharp corners), in order.
            sharp: Vec<usize>,
            /// Twice the area it fills (negative for a hole).
            filled: f32,
            /// Thinner than a pixel all over (turned inside out by moving, or a speck) and
            /// drawn as a thin part.
            all_thin: bool,
            /// How strong the fades are at each vertex (a tip, see below, or `under_fade`).
            alpha: Vec<f32>,
            /// Where the fill is thinner than a pixel behind a vertex, how thick (as a share
            /// of the pixel; 1 elsewhere). A fade is a ramp a pixel wide, centred on its edge,
            /// so across a thin stretch the fades of its two edges lie over each other: at
            /// full strength a hairline was as dark as a line of a whole pixel (thin circles
            /// and outlines came out solid black and jagged). There the ramp goes up at its
            /// usual slope only as far as the stretch is thick, then down again to nothing at
            /// the inner point; the two sides together are then what a box filter (or MSAA
            /// with many samples) makes of the stretch: its thickness high over the pixel
            /// around it, with ramps as wide as it's thick.
            knee: Vec<f32>,
            /// More points of the fades along edges: (the edge, how far along it, strength,
            /// knee).
            extra: Vec<(usize, f32, f32, f32)>,
        }
        // Votes for which side of a ring is inside, from tests `beside` its edges (those at
        // least `shortest` long), on both sides of each edge's middle: where the fill is on
        // one side only (not on an edge between two rings of the fill), that side gets the
        // edge's length as a vote, added to `votes` (left, right).
        fn vote(
            rings: &[Outline],
            outlines: &[&[Point]],
            fill_rule: lyon::path::FillRule,
            votes: &mut [(f32, f32)],
            which: &dyn Fn(usize) -> bool,
            beside: f32,
            shortest: f32,
        ) {
            let mut probes = Vec::new();
            let mut edges = Vec::new();
            for (r, ring) in rings.iter().enumerate() {
                if !which(r) {
                    continue;
                }
                let n = ring.points.len();
                for i in 0..n {
                    let length = ring.lengths[i];
                    if length < shortest {
                        continue;
                    }
                    let middle = ring.points[i].lerp(ring.points[(i + 1) % n], 0.5);
                    let beside = ring.normals[i] * beside;
                    probes.push(middle + beside);
                    probes.push(middle - beside);
                    edges.push((r, length));
                }
            }
            if probes.is_empty() {
                return;
            }
            let windings = windings_at(outlines, &probes);
            for (k, &(r, length)) in edges.iter().enumerate() {
                let left = fill_rule.is_in(windings[2 * k]);
                let right = fill_rule.is_in(windings[2 * k + 1]);
                if left && !right {
                    votes[r].0 += length;
                } else if right && !left {
                    votes[r].1 += length;
                }
            }
        }
        // Which side of each ring is inside: tested an eighth of a pixel beside the middle
        // of each edge. (A single test beside the longest edge could land outside a thin
        // part, turning the whole ring the wrong way: it was then left out, a hole in the
        // fill.)
        let outlines: Vec<&[Point]> = rings.iter().map(|ring| ring.points.as_slice()).collect();
        let mut votes = vec![(0.0f32, 0.0f32); rings.len()];
        vote(
            &rings,
            &outlines,
            fill_rule,
            &mut votes,
            &|_| true,
            width * 0.25,
            width * 0.25,
        );
        // A ring with a quarter of its length or more the other way crosses itself or another
        // ring of this fill (overlapping loops, a figure eight): it can't be moved as a whole,
        // as those stretches would be left out. Such a fill is drawn as it is, with plain
        // edges. (Less than that is mostly misjudged tests at sharp corners and thin parts: in
        // test renders against an 8x reference, drawing those fills plain was worse.)
        let tangled = |&(left, right): &(f32, f32)| {
            let (more, less) = (left.max(right), left.min(right));
            less > width * 4.0 && less > more * 0.25
        };
        // Before giving a fill up, a ring that looks tangled is looked at again from much
        // closer (a 60th of a pixel): an eighth of a pixel beside an edge is past the far side
        // of a part thinner than that and may be inside the next part, a vote the wrong way.
        // A fill's edges all have it on the same side as they're walked, so with tests close
        // enough only a real crossing still looks tangled. (One fill in ten of a big set of
        // game art was drawn plain for this: jagged, heavier, and not reaching under the
        // fills over it, so the background showed between them.)
        // A sliver among them (under a tenth of a pixel thick on average: twisted leftovers of
        // the drawing tool, mostly) is left out instead: there's next to nothing of it to
        // see, and moved like a ring of any size it came out as a line.
        let (rings, mut votes) = if votes.iter().any(tangled) {
            drop(outlines);
            let mut rings = rings;
            let sliver: Vec<bool> = rings
                .iter()
                .zip(&votes)
                .map(|(ring, vote)| tangled(vote) && ring.area.abs() < 0.2 * width * ring.perimeter)
                .collect();
            if sliver.contains(&true) {
                let mut keep = sliver.iter().map(|s| !*s);
                rings.retain(|_| keep.next().unwrap_or(true));
                let mut keep = sliver.iter().map(|s| !*s);
                votes.retain(|_| keep.next().unwrap_or(true));
            }
            (rings, votes)
        } else {
            drop(outlines);
            (rings, votes)
        };
        let outlines: Vec<&[Point]> = rings.iter().map(|ring| ring.points.as_slice()).collect();
        if votes.iter().any(tangled) {
            let again: Vec<bool> = votes.iter().map(tangled).collect();
            for (vote, again) in votes.iter_mut().zip(&again) {
                if *again {
                    *vote = (0.0, 0.0);
                }
            }
            vote(
                &rings,
                &outlines,
                fill_rule,
                &mut votes,
                &|r| again[r],
                width / 30.0,
                width / 15.0,
            );
        }
        if votes.iter().any(tangled) {
            TANGLED.with(|t| t.set(true));
            return None;
        }
        // Rings without a vote are looked at again from much closer too: where the band
        // between an outline and the hole in it is thinner than an eighth of a pixel, a test
        // beside any edge of either ring is past the band, so neither got a vote and both
        // were given a side by the left's word below, one of them the wrong one (a hairline
        // ring was drawn as a solid line a pixel wide).
        if votes
            .iter()
            .any(|&(left, right)| left <= 0.0 && right <= 0.0)
        {
            let none: Vec<bool> = votes
                .iter()
                .map(|&(left, right)| left <= 0.0 && right <= 0.0)
                .collect();
            vote(
                &rings,
                &outlines,
                fill_rule,
                &mut votes,
                &|r| none[r],
                width / 30.0,
                width * 0.25,
            );
        }
        // Rings still without votes (too small to test): beside the middle of the longest
        // edge, on both sides, no farther from it than a quarter of the ring's own thickness
        // (twice its area over its perimeter). (An eighth of a pixel, as for the votes, is
        // past the far side of a speck smaller than that: in a fine pattern two specks in
        // three were taken for holes, and a hole's fades were drawn around each, most of a
        // pixel across.)
        let unvoted: Vec<usize> = (0..rings.len())
            .filter(|&r| votes[r].0 <= 0.0 && votes[r].1 <= 0.0)
            .collect();
        let fallback_probes: Vec<Point> = unvoted
            .iter()
            .flat_map(|&r| {
                let ring = &rings[r];
                let (points, n) = (&ring.points, ring.points.len());
                let longest = (0..n)
                    .max_by(|&i, &j| {
                        let li = (points[(i + 1) % n] - points[i]).square_length();
                        let lj = (points[(j + 1) % n] - points[j]).square_length();
                        li.total_cmp(&lj)
                    })
                    .unwrap_or(0);
                let thickness = ring.area.abs() / ring.perimeter.max(EPSILON);
                let (a, b) = (points[longest], points[(longest + 1) % n]);
                let beside = ring.normals[longest] * (width * 0.25).min(thickness * 0.25);
                [a.lerp(b, 0.5) + beside, a.lerp(b, 0.5) - beside]
            })
            .collect();
        let fallback_windings = if fallback_probes.is_empty() {
            Vec::new()
        } else {
            windings_at(&outlines, &fallback_probes)
        };
        let mut left_inside: Vec<bool> = votes.iter().map(|&(left, right)| left >= right).collect();
        for (k, &r) in unvoted.iter().enumerate() {
            // (Where both sides or neither are in the fill, the left's word, as before.)
            left_inside[r] = fill_rule.is_in(fallback_windings[2 * k]);
        }
        drop(outlines);
        let rings: Vec<Ring> = rings
            .into_iter()
            .zip(left_inside)
            .zip(&votes)
            .map(|((ring, left_is_inside), &(left, right))| {
                let Outline {
                    points,
                    sides,
                    lengths,
                    normals,
                    perimeter,
                    area: original_area,
                } = ring;
                let n = points.len();
                let sign = if left_is_inside { -1.0 } else { 1.0 };
                // (Outwards.)
                let normal = |i: usize| normals[i] * sign;
                let along = |i: usize| Vector::new(normals[i].y, -normals[i].x);
                let offset = |i: usize| match sides[i] {
                    Side::Faded | Side::Seam => -width,
                    Side::Under => width,
                    Side::Hard => 0.0,
                };
                // Each moved vertex: where the two moved edges meet.
                let mut sharp = Vec::new();
                let moved: Vec<Point> = (0..n)
                    .map(|i| {
                        let prev = (i + n - 1) % n;
                        let (n0, n1) = (normal(prev), normal(i));
                        let (d0, d1) = (offset(prev), offset(i));
                        // Medium: a corner between an edge that reaches under another fill and
                        // a hard one stays where it is. (Moved, it slid along the hard edge's
                        // line, out into whatever third fill meets the two there: a dot at
                        // every crossing of a plaid.)
                        if medium && ((d0 == 0.0 && d1 > 0.0) || (d1 == 0.0 && d0 > 0.0)) {
                            return points[i];
                        }
                        let a0 = points[prev] + n0 * d0;
                        let a1 = points[i] + n1 * d1;
                        let (u0, u1) = (along(prev), along(i));
                        let cross = u0.cross(u1);
                        let fallback = points[i] + (n0 * d0 + n1 * d1) * 0.5;
                        if cross.abs() < 1e-3 {
                            if u0.dot(u1) < 0.0 {
                                sharp.push(i);
                            }
                            return fallback;
                        }
                        let t = (a1 - a0).cross(u1) / cross;
                        let q = a0 + u0 * t;
                        if (q - points[i]).length() > 4.0 * width {
                            sharp.push(i);
                            fallback
                        } else {
                            q
                        }
                    })
                    .collect();
                // Half a pixel out along each vertex's normal (its edges' normals averaged,
                // lengthened for the miter, at most 2x: `vertex_normals`).
                let outer: Vec<Point> = (0..n)
                    .map(|i| {
                        let (a, b) = (normals[(i + n - 1) % n], normals[i]);
                        let sum = a + b;
                        let v = if sum.square_length() < EPSILON {
                            b
                        } else {
                            let miter = sum.normalize();
                            miter / miter.dot(b).max(0.5)
                        };
                        points[i] + v * (sign * width)
                    })
                    .collect();
                let moved_area = area(&moved);
                let all_thin = original_area * moved_area <= 0.0
                    || moved_area.abs() < 0.05 * original_area.abs();
                // Or most of its edges come out of the move pointing backwards: a speck,
                // smaller than a pixel both ways, which the move turns round and makes BIGGER
                // (a 0.2 px square becomes a 0.8 px one, the same way round, so the area test
                // above misses it: fine patterns were drawn as solid dots most of a pixel
                // across).
                let reversed: f32 = (0..n)
                    .filter(|&i| {
                        let j = (i + 1) % n;
                        (moved[j] - moved[i]).dot(points[j] - points[i]) < 0.0
                    })
                    .map(|i| lengths[i])
                    .sum();
                let all_thin = all_thin || reversed >= 0.5 * perimeter;
                // In `area`'s sense the fill is on the left of a positive outline.
                let filled = if left_is_inside {
                    original_area
                } else {
                    -original_area
                };
                Ring {
                    alpha: vec![1.0; n],
                    knee: vec![1.0; n],
                    extra: Vec::new(),
                    sure: (left > 0.0) != (right > 0.0),
                    points,
                    sides,
                    lengths,
                    normals,
                    left_is_inside,
                    moved,
                    outer,
                    sharp,
                    filled,
                    all_thin,
                }
            })
            .collect();

        // The fades' strength where the fill is thin, and sharp tips.
        let mut rings = rings;
        {
            let mut moved: Vec<Vec<Point>> = rings
                .iter_mut()
                .map(|ring| std::mem::take(&mut ring.moved))
                .collect();
            let mut alphas: Vec<Vec<f32>> = rings
                .iter_mut()
                .map(|ring| std::mem::take(&mut ring.alpha))
                .collect();
            let mut knees: Vec<Vec<f32>> = rings
                .iter_mut()
                .map(|ring| std::mem::take(&mut ring.knee))
                .collect();
            let mut extras: Vec<Vec<(usize, f32, f32, f32)>> = vec![Vec::new(); rings.len()];
            // What each ray is for: a vertex (and whether it's a tip) or a point along an edge
            // (and whether the vertex at the edge's start, 1, or end, 2, takes its result too).
            enum For {
                Vertex(usize, bool),
                Along(usize, f32, u8),
            }
            const CAP: f32 = 4.0;
            let mut rays: Vec<Ray> = Vec::new();
            let mut ray_for: Vec<(usize, For)> = Vec::new();
            for (r, ring) in rings.iter().enumerate() {
                if ring.all_thin {
                    continue;
                }
                let points = &ring.points;
                let n = points.len();
                let sign = if ring.left_is_inside { -1.0 } else { 1.0 };
                let mut sharp = ring.sharp.iter().copied().peekable();
                for i in 0..n {
                    let prev = (i + n - 1) % n;
                    let p = points[i];
                    let is_sharp = sharp.next_if_eq(&i).is_some();
                    // The other side of a sharp tip (see below): the fill around it, drawn
                    // before it, whose two edges there lie under the tip's fill. Its corner
                    // was left at the tip too, so it reached under the tip's sides only (less
                    // and less towards the point); it goes as far under as the tip's inner
                    // point now goes in, or the background showed through the tip's fades
                    // (light dots on tips drawn over another colour).
                    if ring.sides[prev] == Side::Under && ring.sides[i] == Side::Under {
                        if is_sharp {
                            let sum = (ring.normals[prev] + ring.normals[i]) * sign;
                            let turn = (p - points[prev]).cross(points[(i + 1) % n] - p);
                            let sine = sum.length() / 2.0;
                            if turn * sign > 0.0 && sine > 1e-4 && sine < 1.0 / CAP {
                                // (No farther than half as far as its edges on both sides go
                                // on under that fill: a tip shorter than 2 px has something
                                // else behind it, which this fill then showed in as a spur.)
                                let run = |forward: bool| -> f32 {
                                    let (mut total, mut k) = (0.0f32, i);
                                    for _ in 0..n {
                                        let edge = if forward { k } else { (k + n - 1) % n };
                                        if ring.sides[edge] != Side::Under {
                                            break;
                                        }
                                        total += ring.lengths[edge];
                                        if total >= 2.0 * CAP * width {
                                            break;
                                        }
                                        k = if forward { (k + 1) % n } else { edge };
                                    }
                                    total
                                };
                                let reach = (CAP * width).min(0.5 * run(true).min(run(false)));
                                moved[r][i] = p + sum.normalize() * reach;
                            }
                        }
                        continue;
                    }
                    if ring.sides[prev] != Side::Faded || ring.sides[i] != Side::Faded {
                        continue;
                    }
                    // A sharp convex corner (under 29 degrees): the two moved edges meet
                    // farther in than 2 px, somewhere the fill may not even reach. Its inner
                    // point was left at the tip then, at full strength, with the fade around
                    // it: a dark dot on every sharp tip. It goes 2 px in along the bisector
                    // now (no farther than half-way to the far side), and is as strong as the
                    // fill's width there asks for (see `tips` below).
                    let mut tip = false;
                    if is_sharp {
                        let sum = (ring.normals[prev] + ring.normals[i]) * sign;
                        let turn = (p - points[prev]).cross(points[(i + 1) % n] - p);
                        let convex = turn * sign < 0.0;
                        // Half the corner's angle has this sine.
                        let sine = sum.length() / 2.0;
                        if convex && sine > 1e-4 && sine < 1.0 / CAP {
                            moved[r][i] = p - sum.normalize() * (CAP * width);
                            tip = true;
                        }
                    }
                    let inwards = moved[r][i] - p;
                    let reach = inwards.length();
                    if reach < 0.25 * width {
                        continue;
                    }
                    rays.push(Ray {
                        from: p,
                        dir: inwards / reach,
                        reach: 2.0 * reach,
                        ring: r,
                        skip: [prev, i],
                    });
                    ray_for.push((r, For::Vertex(i, tip)));
                }
                // Long edges (over 4 px) are also looked at a pixel from each end and every
                // 4 px or so between: a long straight stretch has no vertices to say where
                // it turns thin. And an edge at the end of a run of faded ones (its vertex
                // there has an edge under another fill on its other side, and isn't looked
                // from above) is looked at beside that vertex, or in its middle when it's
                // short, straight in from the edge; the vertex takes what's found there. (A
                // thin strip with one faded side and one covered kept full-strength fades at
                // its ends, or all along when it's made of few points.)
                for i in 0..n {
                    if ring.sides[i] != Side::Faded {
                        continue;
                    }
                    let j = (i + 1) % n;
                    let (a, b) = (points[i], points[j]);
                    let length = ring.lengths[i];
                    let ends = u8::from(ring.sides[(i + n - 1) % n] != Side::Faded)
                        | (u8::from(ring.sides[j] != Side::Faded) << 1);
                    if length <= 8.0 * width && ends == 0 {
                        continue;
                    }
                    let straight = ring.normals[i] * (-sign * width);
                    let mut look = |t: f32, ends: u8| {
                        let from = a.lerp(b, t);
                        let inwards = if ends == 0 {
                            moved[r][i].lerp(moved[r][j], t) - from
                        } else {
                            straight
                        };
                        let reach = inwards.length();
                        if reach >= 0.25 * width {
                            rays.push(Ray {
                                from,
                                dir: inwards / reach,
                                reach: (2.0 * reach).min(4.0 * CAP * width),
                                ring: r,
                                skip: [i, usize::MAX],
                            });
                            ray_for.push((r, For::Along(i, t, ends)));
                        }
                    };
                    if length <= 8.0 * width {
                        look(0.5, ends);
                        continue;
                    }
                    let near = 2.0 * width / length;
                    let between = ((length / (8.0 * width)).ceil() as usize).clamp(1, 8);
                    look(near, ends & 1);
                    for k in 1..between {
                        let t = k as f32 / between as f32;
                        if t > near * 1.5 && t < 1.0 - near * 1.5 {
                            look(t, 0);
                        }
                    }
                    look(1.0 - near, ends & 2);
                }
            }
            if !rays.is_empty() {
                let outlines: Vec<&[Point]> =
                    rings.iter().map(|ring| ring.points.as_slice()).collect();
                let tree = EdgeTree::new(&outlines);
                let hits: Vec<Option<(f32, usize, usize)>> =
                    rays.iter().map(|ray| tree.hit(ray)).collect();
                // A hit counts where the fill really ends there: inside it half-way, outside
                // just past it (overlapping loops have edges inside the fill). That's only
                // in doubt, and tested, where the tests for a ring's inside didn't all agree,
                // for the ray's ring or the one it met.
                let doubtful = |ray: &Ray, hit: &Option<(f32, usize, usize)>| {
                    hit.is_some_and(|(_, ring, _)| {
                        !rings[ring].sure || rings.get(ray.ring).is_some_and(|own| !own.sure)
                    })
                };
                let mut probes = Vec::new();
                for (ray, hit) in rays.iter().zip(&hits) {
                    if let Some((along, _, _)) = hit
                        && doubtful(ray, hit)
                    {
                        probes.push(ray.from + ray.dir * (along * 0.5));
                        probes.push(ray.from + ray.dir * (along + 0.1 * width));
                    }
                }
                let windings = if probes.is_empty() {
                    Vec::new()
                } else {
                    windings_at(&outlines, &probes)
                };
                let mut probe = 0;
                // (Ring, vertex, the direction it went in.)
                let mut tips: Vec<(usize, usize, Vector)> = Vec::new();
                for (k, (ray, hit)) in rays.iter().zip(&hits).enumerate() {
                    let ends = hit.is_some()
                        && (!doubtful(ray, hit) || {
                            probe += 2;
                            fill_rule.is_in(windings[probe - 2])
                                && !fill_rule.is_in(windings[probe - 1])
                        });
                    let (r, target) = &ray_for[k];
                    // (Strength, knee.)
                    let (strength, knee) = match hit {
                        Some((along, ring, edge)) if ends => match rings[*ring].sides[*edge] {
                            Side::Faded if medium => (thin_fade(along / ray.reach), 1.0),
                            Side::Faded => (1.0, (along / ray.reach).clamp(0.0, 1.0)),
                            Side::Under => (under_fade(along / ray.reach), 1.0),
                            // (Medium's hard inner edges: the fade is all the ink of a stretch
                            // up to half a pixel thick, so twice its thickness strong.)
                            Side::Hard if medium => ((2.0 * along / ray.reach).min(1.0), 1.0),
                            _ => (1.0, 1.0),
                        },
                        _ => (1.0, 1.0),
                    };
                    match (target, hit) {
                        (For::Vertex(i, false), _) => {
                            alphas[*r][*i] *= strength;
                            knees[*r][*i] = knee;
                        }
                        (For::Vertex(i, true), _) => {
                            // A tip's inner point goes no farther than half-way to the far
                            // side.
                            if let Some((along, _, _)) = hit
                                && ends
                            {
                                moved[*r][*i] = ray.from + ray.dir * (along * 0.5);
                            }
                            tips.push((*r, *i, ray.dir));
                        }
                        (For::Along(i, t, ends), _) => {
                            extras[*r].push((*i, *t, strength, knee));
                            let n = alphas[*r].len();
                            for (bit, vertex) in [(1, *i), (2, (*i + 1) % n)] {
                                if ends & bit != 0 {
                                    alphas[*r][vertex] = alphas[*r][vertex].min(strength);
                                    knees[*r][vertex] = knees[*r][vertex].min(knee);
                                }
                            }
                        }
                    }
                }
                // How strong each tip's inner point is: the fades of the tip's two sides meet
                // there, a tent as wide as the fill is across at that point (t, measured, as
                // a tip between two curves is wider than its angle says) plus a pixel; it
                // holds t of ink where it's 2t / (t + 1) high. From a pixel across, full
                // strength.
                if !tips.is_empty() {
                    let across: Vec<Ray> = tips
                        .iter()
                        .flat_map(|&(r, i, dir)| {
                            let side = Vector::new(-dir.y, dir.x);
                            [side, -side].map(|dir| Ray {
                                from: moved[r][i],
                                dir,
                                reach: 2.0 * width,
                                ring: usize::MAX,
                                skip: [usize::MAX; 2],
                            })
                        })
                        .collect();
                    let sides: Vec<Option<(f32, usize, usize)>> =
                        across.iter().map(|ray| tree.hit(ray)).collect();
                    for (k, &(r, i, _)) in tips.iter().enumerate() {
                        if let (Some((left, ..)), Some((right, ..))) =
                            (sides[2 * k], sides[2 * k + 1])
                        {
                            let t = (left + right) / (2.0 * width);
                            alphas[r][i] = (2.0 * t / (t + 1.0)).min(1.0);
                        }
                    }
                }
            }
            for (r, ring) in rings.iter_mut().enumerate() {
                // An edge's extra points only count where something along it is weaker.
                let n = ring.points.len();
                let whole = |i: usize| alphas[r][i] >= 0.98 && knees[r][i] >= 0.98;
                let mut extra = std::mem::take(&mut extras[r]);
                if !extra.is_empty() {
                    let mut weak = vec![false; n];
                    for &(i, _, strength, knee) in &extra {
                        weak[i] |= strength < 0.98 || knee < 0.98;
                    }
                    extra.retain(|&(i, ..)| weak[i] || !whole(i) || !whole((i + 1) % n));
                }
                ring.extra = extra;
                ring.moved = std::mem::take(&mut moved[r]);
                ring.alpha = std::mem::take(&mut alphas[r]);
                ring.knee = std::mem::take(&mut knees[r]);
            }
        }

        // Parts thinner than a pixel all over (with their holes, so a thin ring stays a ring)
        // keep their outline, drawn see-through so they have as much ink as they cover (MSAA's
        // look), with the outer-only fade from there. Each part (a thin outline and the holes
        // in it) has its own alpha k: its body holds k times its area of ink, and its fades
        // (k at the outline, none half a pixel out) k times `fade_ink`, so
        // k = area / (area + fade ink). For a long strip of thickness t that is t / (t + half
        // a pixel on each side's fade); for a speck the ring of fade around it is many times
        // its area, and it is nearly gone, as it is with MSAA. (One alpha for all of a fill's
        // thin parts, from their average thickness, drew a fine pattern's specks two or three
        // times too heavy: ground runes came out as solid blobs.) Edges under another fill
        // get no fade (the fill above fades over them): only the faded edges count.
        let thin_outers: Vec<usize> = (0..rings.len())
            .filter(|&r| rings[r].all_thin && rings[r].filled > 0.0)
            .collect();
        // The thin outline each ring belongs to (itself, or the one around a thin hole).
        let part_of: Vec<Option<usize>> = rings
            .iter()
            .enumerate()
            .map(|(r, ring)| {
                if !ring.all_thin {
                    None
                } else if ring.filled > 0.0 {
                    Some(r)
                } else {
                    thin_outers
                        .iter()
                        .copied()
                        .find(|&o| in_polygon(&rings[o].points, ring.points[0]))
                }
            })
            .collect();
        // The ink of a ring's fades at alpha 1: each faded segment's strip is two triangles,
        // (inner, inner, outer) with two thirds of its area and (inner, outer, outer) with one.
        let fade_ink = |ring: &Ring| -> f32 {
            let n = ring.points.len();
            (0..n)
                .filter(|&i| ring.sides[i].fades())
                .map(|i| {
                    let j = (i + 1) % n;
                    let (a, b) = (ring.points[i], ring.points[j]);
                    let (a_out, b_out) = (ring.outer[i], ring.outer[j]);
                    let first = (b - a).cross(b_out - a).abs() * 0.5;
                    let second = (b_out - a).cross(a_out - a).abs() * 0.5;
                    (2.0 * first + second) / 3.0
                })
                .sum()
        };
        // Per thin outline: twice the area filled, and the fades' ink.
        let mut part_ink = vec![(0.0f32, 0.0f32); rings.len()];
        for (ring, part) in rings.iter().zip(&part_of) {
            if let Some(part) = *part {
                part_ink[part].0 += ring.filled;
                part_ink[part].1 += fade_ink(ring);
            }
        }
        // In `THIN_LEVELS` steps, so a fill with thousands of thin parts is still a few paths;
        // a part that comes out at none isn't drawn at all.
        const THIN_LEVELS: f32 = 32.0;
        let thin_level: Vec<Option<u8>> = part_of
            .iter()
            .map(|part| {
                part.map(|part| {
                    let (twice_area, fades) = part_ink[part];
                    let area = twice_area.max(0.0) * 0.5;
                    let alpha = (area / (area + fades).max(EPSILON)).clamp(0.0, 1.0);
                    (alpha * THIN_LEVELS).round() as u8
                })
            })
            .collect();

        let mut builder = Path::builder();
        let mut thin: std::collections::BTreeMap<u8, lyon::path::path::Builder> =
            std::collections::BTreeMap::new();
        let mut fade_points = Vec::new();
        let mut fade_runs = Vec::new();
        for (ring, level) in rings.into_iter().zip(thin_level) {
            let n = ring.points.len();
            let in_thin = level.is_some();
            if level == Some(0) {
                continue;
            }
            let moved = if in_thin { &ring.points } else { &ring.moved };
            let alpha = level.map_or(1.0, |level| f32::from(level) / THIN_LEVELS);
            // The fades, from the moved outline (inside) to half a pixel outside: strips over
            // runs of faded segments (each point shared by the segments on both sides).
            let point = |i: usize| {
                let (strength, knee) = if in_thin {
                    (1.0, 1.0)
                } else {
                    (ring.alpha[i], ring.knee[i])
                };
                // Medium: the outer point can go where the fade's outer line is nearly
                // straight through it (within a fifth of a pixel of the line between its
                // neighbours), inside a run of faded edges.
                let spare = medium && knee >= 1.0 && {
                    let (before, after) = ((i + n - 1) % n, (i + 1) % n);
                    let (a, b, c) = (ring.outer[before], ring.outer[i], ring.outer[after]);
                    let chord = c - a;
                    ring.sides[before].fades()
                        && ring.sides[i].fades()
                        && (b - a).cross(chord).abs() < 0.4 * width * chord.length()
                };
                (
                    squash.undo(moved[i]),
                    squash.undo(ring.outer[i]),
                    strength,
                    if spare { SPARE } else { knee },
                )
            };
            // The extra points along edge `i` (see `Ring::extra`).
            let along = |i: usize, points: &mut Vec<(Point, Point, f32, f32)>| {
                if in_thin || ring.extra.is_empty() {
                    return;
                }
                let j = (i + 1) % n;
                let first = ring.extra.partition_point(|e| e.0 < i);
                for &(_, t, strength, knee) in ring.extra[first..].iter().take_while(|e| e.0 == i) {
                    points.push((
                        squash.undo(moved[i].lerp(moved[j], t)),
                        squash.undo(ring.outer[i].lerp(ring.outer[j], t)),
                        strength,
                        knee,
                    ));
                }
            };
            match (0..n).find(|&i| !ring.sides[i].fades()) {
                None => {
                    let start = fade_points.len();
                    for i in 0..n {
                        fade_points.push(point(i));
                        along(i, &mut fade_points);
                    }
                    fade_runs.push((start, fade_points.len() - start, alpha, true));
                }
                Some(under) => {
                    // Starting after an edge under another fill, each run of faded ones.
                    let mut run: Option<usize> = None;
                    for k in 1..=n {
                        let i = (under + k) % n;
                        if ring.sides[i].fades() {
                            if run.is_none() {
                                run = Some(fade_points.len());
                                fade_points.push(point(i));
                            }
                            along(i, &mut fade_points);
                            fade_points.push(point((i + 1) % n));
                        } else if let Some(start) = run.take() {
                            fade_runs.push((start, fade_points.len() - start, alpha, false));
                        }
                    }
                    if let Some(start) = run {
                        fade_runs.push((start, fade_points.len() - start, alpha, false));
                    }
                }
            }
            if let Some(level) = level {
                let thin = thin.entry(level).or_insert_with(Path::builder);
                thin.begin(squash.undo(moved[0]));
                for q in &moved[1..] {
                    thin.line_to(squash.undo(*q));
                }
                thin.end(true);
            } else {
                // Wound with the fill on the right, which lyon counts as positive (holes then
                // cancel the outline around them; bits turned inside out wind negative and
                // are left out by the `Positive` fill rule).
                let mut ordered = ring.moved;
                if ring.left_is_inside {
                    ordered.reverse();
                }
                builder.begin(squash.undo(ordered[0]));
                for q in &ordered[1..] {
                    builder.line_to(squash.undo(*q));
                }
                builder.end(true);
            }
        }
        Some(Centered {
            outline: builder.build(),
            thin: thin
                .into_iter()
                .map(|(level, part)| (part.build(), f32::from(level) / THIN_LEVELS))
                .collect(),
            fade_points,
            fade_runs,
        })
    }

    /// The fades of a centered fill.
    pub fn push_fades(
        mesh: &mut VertexBuffers<Vertex, u32>,
        centered: &Centered,
        color: swf::Color,
    ) {
        if color.a == 0 {
            return;
        }
        let clear = swf::Color { a: 0, ..color };
        // Where each point's vertices start: its inner point, its knee where it has one
        // (see `Ring::knee`: the inner point is then at none), its outer point (`outers`:
        // none where it's left out).
        let mut firsts: Vec<u32> = Vec::new();
        let mut outers: Vec<u32> = Vec::new();
        for &(start, count, alpha, closed) in &centered.fade_runs {
            let points = &centered.fade_points[start..start + count];
            let segments = if closed {
                count
            } else {
                count.saturating_sub(1)
            };
            firsts.clear();
            if points.iter().all(|point| point.3 >= 1.0) {
                // No knees. Outer points that can be spared (Medium, `SPARE`) are left out,
                // never the first, the last of an open run, or two in a row: the strip is
                // then a fan round the last outer point kept, three triangles for two
                // segments instead of four.
                outers.clear();
                let mut kept = false;
                for (k, (inner, outer, strength, knee)) in points.iter().enumerate() {
                    firsts.push(mesh.vertices.len() as u32);
                    let edge = with_alpha(color, alpha * strength);
                    mesh.vertices.push(Vertex {
                        x: inner.x,
                        y: inner.y,
                        color: edge,
                    });
                    kept = !(kept && *knee >= SPARE && k > 0 && (closed || k + 1 < count));
                    if kept {
                        outers.push(mesh.vertices.len() as u32);
                        mesh.vertices.push(Vertex {
                            x: outer.x,
                            y: outer.y,
                            color: clear,
                        });
                    } else {
                        outers.push(u32::MAX);
                    }
                }
                let mut last = outers.first().copied().unwrap_or(0);
                for i in 0..segments {
                    let j = (i + 1) % count;
                    let (inner_i, inner_j) = (firsts[i], firsts[j]);
                    if outers[j] == u32::MAX {
                        mesh.indices.extend_from_slice(&[inner_i, inner_j, last]);
                    } else {
                        mesh.indices.extend_from_slice(&[
                            inner_i, inner_j, outers[j], inner_i, outers[j], last,
                        ]);
                        last = outers[j];
                    }
                }
                continue;
            }
            for (inner, outer, strength, knee) in points {
                firsts.push(mesh.vertices.len() as u32);
                let full = alpha * strength;
                if *knee < 1.0 {
                    let knee = knee.max(0.0);
                    let middle = outer.lerp(*inner, knee);
                    for (p, a) in [(*inner, 0.0), (middle, full * knee)] {
                        mesh.vertices.push(Vertex {
                            x: p.x,
                            y: p.y,
                            color: with_alpha(color, a),
                        });
                    }
                } else {
                    let edge = with_alpha(color, full);
                    mesh.vertices.push(Vertex {
                        x: inner.x,
                        y: inner.y,
                        color: edge,
                    });
                }
                mesh.vertices.push(Vertex {
                    x: outer.x,
                    y: outer.y,
                    color: clear,
                });
            }
            for i in 0..segments {
                let j = (i + 1) % count;
                let (kneed_i, kneed_j) = (points[i].3 < 1.0, points[j].3 < 1.0);
                let (inner_i, inner_j) = (firsts[i], firsts[j]);
                // (A point without a knee: its inner point stands in for it.)
                let (knee_i, knee_j) = (inner_i + u32::from(kneed_i), inner_j + u32::from(kneed_j));
                let (outer_i, outer_j) = (knee_i + 1, knee_j + 1);
                // From the inner points to the knees, where there's anything between them.
                if kneed_j {
                    mesh.indices.extend_from_slice(&[inner_i, inner_j, knee_j]);
                }
                if kneed_i {
                    mesh.indices.extend_from_slice(&[inner_i, knee_j, knee_i]);
                }
                mesh.indices
                    .extend_from_slice(&[knee_i, knee_j, outer_j, knee_i, outer_j, outer_i]);
            }
        }
    }

    /// The fade around a fill.
    pub fn fill(
        mesh: &mut VertexBuffers<Vertex, u32>,
        path: &Path,
        fill_rule: lyon::path::FillRule,
        color: swf::Color,
        scale: f32,
        squash: &Squash,
    ) {
        let Some((width, tolerance)) = sizes(scale) else {
            return;
        };
        if color.a == 0 {
            return;
        }
        let start = mesh.vertices.len();
        for (points, _) in squash.polylines(path, tolerance) {
            // Fills are always closed.
            let n = points.len();
            if n < 3 {
                continue;
            }
            // Which side is outside: test just beside the middle of the longest edge.
            let longest = (0..n)
                .max_by(|&i, &j| {
                    let li = (points[(i + 1) % n] - points[i]).square_length();
                    let lj = (points[(j + 1) % n] - points[j]).square_length();
                    li.total_cmp(&lj)
                })
                .unwrap_or(0);
            let (a, b) = (points[longest], points[(longest + 1) % n]);
            let probe = squash.undo(a.lerp(b, 0.5) + left_normal(a, b) * (width * 0.25));
            let left_is_inside = lyon::algorithms::hit_test::hit_test_path(
                &probe,
                path.iter(),
                fill_rule,
                tolerance,
            );
            let outward = if left_is_inside { -width } else { width };
            let offsets: Vec<Vector> = vertex_normals(&points, true)
                .into_iter()
                .map(|v| v * outward)
                .collect();
            push_strip(mesh, &points, &offsets, true, color);
        }
        squash.undo_vertices(mesh, start);
    }

    /// How much narrower a stroke's solid body is drawn with Flash-style edges: a pixel (half
    /// on each side), which the fades on both sides cover again (see `stroke`).
    pub fn stroke_body_width(stroke_width: f32, scale: f32) -> f32 {
        match sizes(scale) {
            Some((width, _)) => (stroke_width - 2.0 * width).max(0.0),
            None => stroke_width,
        }
    }

    /// The fades of a stroke, centered on its true edges: from the narrowed body (full alpha,
    /// see `stroke_body_width`) to half a pixel outside the edge (none), along both sides and
    /// around open ends' caps. A 1-pixel line is then a 2-pixel tent: one pixel of ink, like MSAA.
    pub fn stroke(
        mesh: &mut VertexBuffers<Vertex, u32>,
        path: &Path,
        stroke_width: f32,
        caps: (swf::LineCapStyle, swf::LineCapStyle),
        color: swf::Color,
        scale: f32,
        squash: &Squash,
    ) {
        let Some((width, tolerance)) = sizes(scale) else {
            return;
        };
        if color.a == 0 {
            return;
        }
        // Across the line, in the shape. In a squashed shape the stroke (round in the shape)
        // reaches farther here in some directions than others, and so does its narrowed body:
        // the fades follow it, scaled by `widen` in each direction (else they missed the body:
        // a gap down the middle of the line, or an extra dark band).
        let inner = stroke_body_width(stroke_width, scale) / 2.0;
        let outer = stroke_width / 2.0 + width;
        let fade = outer - inner;
        let clear = swf::Color { a: 0, ..color };
        let start = mesh.vertices.len();
        // A hairline has no body: the inner points of its fades are all on the line itself,
        // one vertex per point of the line for both sides (and for all of a round end, or of
        // the wedge at a sharp turn).
        let hairline = inner <= 0.0;
        type Mesh = VertexBuffers<Vertex, u32>;
        let push = |mesh: &mut Mesh, p: Point, color: swf::Color| -> u32 {
            mesh.vertices.push(Vertex {
                x: p.x,
                y: p.y,
                color,
            });
            mesh.vertices.len() as u32 - 1
        };
        // A quad of fade between two spokes across it, each from its inner vertex (full alpha)
        // to its outer one (none). Neighbouring quads share their spokes' vertices; where the
        // two inner vertices are one, the quad is a triangle.
        let quad = |mesh: &mut Mesh, a: (u32, u32), b: (u32, u32)| {
            if a.0 != b.0 {
                mesh.indices.extend_from_slice(&[a.0, b.0, b.1]);
            }
            mesh.indices.extend_from_slice(&[a.0, b.1, a.1]);
        };
        for (points, closed) in squash.polylines(path, tolerance) {
            let n = points.len();
            if n < 2 {
                continue;
            }
            let segments = if closed { n } else { n - 1 };
            let normal = |i: usize| left_normal(points[i], points[(i + 1) % n]);
            // At gentle turns both segments share the miter direction; at sharp ones (over
            // 60 degrees) each keeps its own normal and a round wedge fills the outside of the
            // turn. (One miter for a sharp turn, e.g. a thin line going down and back up, made
            // its fades wrong all along both segments: dashed balloon strings.)
            let joint = |k: usize| -> Option<Vector> {
                if !closed && (k == 0 || k == n - 1) {
                    return None;
                }
                let (n0, n1) = (normal((k + n - 1) % n), normal(k));
                (n0.dot(n1) >= 0.5).then(|| {
                    let sum = n0 + n1;
                    sum * (2.0 / sum.square_length())
                })
            };
            // Where the fade lines meet at a vertex, as the offset from it per unit of width:
            // at a shared (miter) joint, where the two segments' lines (each at its own
            // distance, see `widen`) cross; else on segment `own`'s normal. And whether it's
            // shared: the same for both segments there.
            let widen_segment = |i: usize| squash.widen(normal(i));
            let corner = |k: usize, own: usize| -> (Vector, bool) {
                let Some(miter) = joint(k) else {
                    return (normal(own) * widen_segment(own), false);
                };
                let prev = (k + n - 1) % n;
                let (k0, k1) = (widen_segment(prev), widen_segment(k));
                if (k0 - k1).abs() < 1e-6 {
                    return (miter * k0, true);
                }
                let (n0, n1) = (normal(prev), normal(k));
                let det = n0.cross(n1);
                if det.abs() < 1e-3 {
                    return (miter * ((k0 + k1) / 2.0), true);
                }
                (
                    Vector::new((k0 * n1.y - k1 * n0.y) / det, (n0.x * k1 - n1.x * k0) / det),
                    true,
                )
            };
            // A hairline's inner vertices: its points, in order.
            let hubs = mesh.vertices.len() as u32;
            if hairline {
                for p in points.iter() {
                    push(mesh, *p, color);
                }
            }
            // The fades' spokes at vertex `k` as segment `own` has them, the left side's then
            // the right side's; and whether the other segment there has the same.
            let corner_at = |mesh: &mut Mesh, k: usize, own: usize| -> ([(u32, u32); 2], bool) {
                let (across, shared) = corner(k, own);
                let p = points[k];
                let (left, right) = if hairline {
                    (hubs + k as u32, hubs + k as u32)
                } else {
                    (
                        push(mesh, p + across * inner, color),
                        push(mesh, p + across * -inner, color),
                    )
                };
                let spokes = [
                    (left, push(mesh, p + across * outer, clear)),
                    (right, push(mesh, p + across * -outer, clear)),
                ];
                (spokes, shared)
            };
            let mut first = None;
            let mut carried = None;
            for i in 0..segments {
                let j = (i + 1) % n;
                let a = match carried.take() {
                    Some(a) => a,
                    None => {
                        let (a, shared) = corner_at(mesh, i, i);
                        if i == 0 {
                            first = Some((a, shared));
                        }
                        a
                    }
                };
                let b = match first {
                    // Around to the start again.
                    Some((b, true)) if closed && j == 0 => b,
                    _ => {
                        let (b, shared) = corner_at(mesh, j, i);
                        if shared {
                            carried = Some(b);
                        }
                        b
                    }
                };
                quad(mesh, a[0], b[0]);
                quad(mesh, a[1], b[1]);
            }
            for k in 0..n {
                if (!closed && (k == 0 || k == n - 1)) || joint(k).is_some() {
                    continue;
                }
                let prev = (k + n - 1) % n;
                let (n0, n1) = (normal(prev), normal(k));
                let forward = (points[k] - points[prev]).normalize();
                for side in [1.0f32, -1.0] {
                    // The outside of the turn: where the next segment heads away from this
                    // side (for a full reversal, one side's wedge covers the tip).
                    if (points[(k + 1) % n] - points[k]).dot(n0 * side) >= 0.0 {
                        continue;
                    }
                    let (from, to) = (n0 * side, n1 * side);
                    let start = from.y.atan2(from.x);
                    let mut delta = to.y.atan2(to.x) - start;
                    while delta > std::f32::consts::PI {
                        delta -= std::f32::consts::TAU;
                    }
                    while delta < -std::f32::consts::PI {
                        delta += std::f32::consts::TAU;
                    }
                    // Nearly reversed: go round the tip (ahead of the line).
                    let mid = start + delta / 2.0;
                    if Vector::new(mid.cos(), mid.sin()).dot(forward) < 0.0 {
                        delta -= std::f32::consts::TAU * delta.signum();
                    }
                    let steps =
                        (delta.abs() / (std::f32::consts::PI / 8.0)).ceil().max(1.0) as usize;
                    let centre = points[k];
                    let spoke = |mesh: &mut Mesh, t: usize| -> (u32, u32) {
                        let angle = start + delta * t as f32 / steps as f32;
                        let d = Vector::new(angle.cos(), angle.sin());
                        let reach = squash.widen(d);
                        let inner_vertex = if hairline {
                            hubs + k as u32
                        } else {
                            push(mesh, centre + d * (inner * reach), color)
                        };
                        (
                            inner_vertex,
                            push(mesh, centre + d * (outer * reach), clear),
                        )
                    };
                    let mut last = spoke(mesh, 0);
                    for t in 0..steps {
                        let next = spoke(mesh, t + 1);
                        quad(mesh, last, next);
                        last = next;
                    }
                }
            }
            if closed {
                continue;
            }
            // Caps: at the start the outside is backwards along the line, at the end forwards.
            for (end, cap) in [(0, caps.0), (n - 1, caps.1)] {
                let round = cap == swf::LineCapStyle::Round;
                if hairline && !round {
                    // Nothing across the end of a line without a body.
                    continue;
                }
                let tangent = if end == 0 {
                    -(points[1] - points[0]).normalize()
                } else {
                    (points[n - 1] - points[n - 2]).normalize()
                };
                let normal = Vector::new(-tangent.y, tangent.x);
                let centre = points[end];
                let (kn, kt) = (squash.widen(normal), squash.widen(tangent));
                // The cap's spokes: where each starts (full alpha), and how far it goes.
                let spoke = |c: usize| -> (Point, Vector) {
                    let side = if c == 0 { -normal } else { normal };
                    let across = side * (inner * kn);
                    match cap {
                        swf::LineCapStyle::None => (centre + across, tangent * (width * kt)),
                        swf::LineCapStyle::Square => {
                            let tip = centre + tangent * (inner * kt);
                            (tip + across, tangent * (fade * kt))
                        }
                        swf::LineCapStyle::Round => {
                            let angle = -std::f32::consts::FRAC_PI_2
                                + std::f32::consts::PI * c as f32 / 8.0;
                            let radial = tangent * angle.cos() + normal * angle.sin();
                            let kr = squash.widen(radial);
                            (centre + radial * (inner * kr), radial * (fade * kr))
                        }
                    }
                };
                let count = if round { 9 } else { 2 };
                let mut last = None;
                for c in 0..count {
                    let (p, offset) = spoke(c);
                    let inner_vertex = if hairline {
                        hubs + end as u32
                    } else {
                        push(mesh, p, color)
                    };
                    let next = (inner_vertex, push(mesh, p + offset, clear));
                    if let Some(last) = last {
                        quad(mesh, last, next);
                    }
                    last = Some(next);
                }
            }
        }
        squash.undo_vertices(mesh, start);
    }
}

/// Converts a gradient to the uniforms used by the shader.
fn swf_gradient_to_uniforms(
    gradient_type: GradientType,
    gradient: &swf::Gradient,
    focal_point: swf::Fixed8,
) -> Gradient {
    Gradient {
        records: gradient.records.clone(),
        gradient_type,
        repeat_mode: gradient.spread,
        focal_point,
        interpolation: gradient.interpolation,
    }
}

struct RuffleVertexCtor {
    color: swf::Color,
}

impl FillVertexConstructor<Vertex> for RuffleVertexCtor {
    fn new_vertex(&mut self, vertex: FillVertex) -> Vertex {
        Vertex {
            x: vertex.position().x,
            y: vertex.position().y,
            color: self.color,
        }
    }
}

impl StrokeVertexConstructor<Vertex> for RuffleVertexCtor {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> Vertex {
        Vertex {
            x: vertex.position().x,
            y: vertex.position().y,
            color: self.color,
        }
    }
}
