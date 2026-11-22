use super::target::PoolOrArcTexture;
use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::target::CommandTarget;
use crate::surface::{ChunkSink, Surface};
use crate::{
    Descriptors, MaskState, Pipelines, PosColorVertex, PosUvVertex, Transforms, as_texture,
};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::cell::Cell;
use std::mem;
use std::sync::Arc;
use swf::{BlendMode, Color, ColorTransform, Twips};
use wgpu::Backend;
use wgpu_profiler::Scope;

/// AQW build: shape vertices start at a multiple of this in the shared shape buffers (and of
/// their own size within a shape), a multiple of both vertex sizes, so every draw from a buffer
/// can use one binding of the whole buffer (see `CommandRenderer::render_shape`).
pub const VERTEX_BASE_MULTIPLE: wgpu::BufferAddress = 60;
const _: () = assert!(
    VERTEX_BASE_MULTIPLE % std::mem::size_of::<PosColorVertex>() as u64 == 0
        && VERTEX_BASE_MULTIPLE % std::mem::size_of::<PosUvVertex>() as u64 == 0
);

pub struct CommandRenderer<'encoder> {
    pipelines: &'encoder Pipelines,
    descriptors: &'encoder Descriptors,
    num_masks: u32,
    mask_state: MaskState,
    needs_stencil: bool,
    dynamic_vertex_buffer: &'encoder wgpu::Buffer,
    /// AQW build: the backdrop and the atlas page of this pass's `DrawCommand::BlendAtlas`.
    page_blend_binds: Option<&'encoder wgpu::BindGroup>,
    /// AQW build: what's set in the render pass, so each draw only sets what changes: wgpu
    /// validates and translates every call, even one repeating the last (thousands of draws a
    /// picture, mostly the same pipeline and shared shape buffers).
    bound: Bound<'encoder>,
}

#[derive(Default)]
struct Bound<'encoder> {
    pipeline: Cell<Option<&'encoder wgpu::RenderPipeline>>,
    group2: Cell<Option<&'encoder wgpu::BindGroup>>,
    vertices: Cell<Option<wgpu::BufferSlice<'encoder>>>,
    /// (Shapes with 16-bit and 32-bit indices share buffers: the format is part of it.)
    indices: Cell<Option<(wgpu::BufferSlice<'encoder>, wgpu::IndexFormat)>>,
    stencil: Cell<Option<u32>>,
    /// A measurement (see `perf_counters::DRAWS_REPEAT`): whether anything was set in the
    /// pass since the last draw, and that draw's first index, base vertex, index count and
    /// instance.
    changed: Cell<bool>,
    last_draw: Cell<Option<(u32, i32, u32, u32)>>,
}

impl<'encoder> CommandRenderer<'encoder> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: &'encoder Pipelines,
        descriptors: &'encoder Descriptors,
        dynamic_vertex_buffer: &'encoder wgpu::Buffer,
        num_masks: u32,
        mask_state: MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            pipelines,
            num_masks,
            mask_state,
            descriptors,
            needs_stencil,
            dynamic_vertex_buffer,
            page_blend_binds: None,
            bound: Bound::default(),
        }
    }

    /// AQW build: the bind group for the pass's `DrawCommand::BlendAtlas`.
    pub fn set_page_blend_binds(&mut self, binds: &'encoder wgpu::BindGroup) {
        self.page_blend_binds = Some(binds);
    }

    fn set_pipeline(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        pipeline: &'encoder wgpu::RenderPipeline,
    ) {
        if self.bound.pipeline.get() != Some(pipeline) {
            render_pass.set_pipeline(pipeline);
            self.bound.pipeline.set(Some(pipeline));
            self.bound.changed.set(true);
            // (Set again after another pipeline: they don't all have a group 2.)
            self.bound.group2.set(None);
        }
    }

    fn set_group2(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.bound.group2.get() != Some(bind_group) {
            render_pass.set_bind_group(2, bind_group, &[]);
            self.bound.group2.set(Some(bind_group));
            self.bound.changed.set(true);
        }
    }

    fn set_stencil(&self, render_pass: &mut wgpu::RenderPass<'encoder>, reference: u32) {
        if self.bound.stencil.get() != Some(reference) {
            render_pass.set_stencil_reference(reference);
            self.bound.stencil.set(Some(reference));
            self.bound.changed.set(true);
        }
    }

    pub fn execute(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        command: &'encoder DrawCommand,
    ) {
        if self.needs_stencil {
            match self.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    self.set_stencil(render_pass, self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    self.set_stencil(render_pass, self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    self.set_stencil(render_pass, self.num_masks);
                }
            }
        }

        match command {
            DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode,
                render_stage3d,
            } => self.render_bitmap(
                render_pass,
                bitmap,
                *instance_index,
                *smoothing,
                *blend_mode,
                *render_stage3d,
                *vertex_offset,
            ),
            DrawCommand::RenderTexture {
                _texture,
                binds,
                instance_index,
                blend_mode,
            } => self.render_texture(render_pass, *instance_index, binds, *blend_mode),
            DrawCommand::RenderShape {
                shape,
                instance_index,
            } => self.render_shape(render_pass, shape, *instance_index),
            DrawCommand::DrawRect { instance_index } => {
                self.draw_rect(render_pass, *instance_index)
            }
            DrawCommand::DrawLine { instance_index } => {
                self.draw_lines::<false>(render_pass, *instance_index)
            }
            DrawCommand::DrawLineRect { instance_index } => {
                self.draw_lines::<true>(render_pass, *instance_index)
            }
            DrawCommand::PushMask => self.push_mask(render_pass),
            DrawCommand::ActivateMask => self.activate_mask(render_pass),
            DrawCommand::DeactivateMask => self.deactivate_mask(render_pass),
            DrawCommand::PopMask => self.pop_mask(render_pass),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, *instance_index),
            DrawCommand::Scissor(rect) => {
                render_pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
                self.bound.changed.set(true);
            }
            DrawCommand::RenderAtlas {
                page,
                instance_index,
                vertex_offset,
                blend_mode,
            } => {
                self.prep_bitmap(render_pass, &page.binds, *blend_mode, false);
                self.draw(
                    render_pass,
                    self.dynamic_vertex_buffer.slice(*vertex_offset..),
                    self.descriptors.quad.indices.slice(..),
                    6,
                    *instance_index,
                );
            }
            DrawCommand::BlendAtlas {
                instance_index,
                vertex_offset,
                blend_mode,
                ..
            } => {
                let pipeline = self
                    .pipelines
                    .complex_blend_page(self.descriptors, *blend_mode);
                self.set_pipeline(
                    render_pass,
                    if self.needs_stencil {
                        pipeline.pipeline_for(self.mask_state)
                    } else {
                        pipeline.stencilless_pipeline()
                    },
                );
                let binds = self
                    .page_blend_binds
                    .expect("a chunk with a blend from a page is drawn with its bind group");
                self.set_group2(render_pass, binds);
                self.draw(
                    render_pass,
                    self.dynamic_vertex_buffer.slice(*vertex_offset..),
                    self.descriptors.quad.indices.slice(..),
                    6,
                    *instance_index,
                );
            }
        }
    }

    pub fn prep_color(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            self.set_pipeline(
                render_pass,
                self.pipelines.color.pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline(render_pass, self.pipelines.color.stencilless_pipeline());
        }
    }

    pub fn prep_lines(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            self.set_pipeline(
                render_pass,
                self.pipelines.lines.pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline(render_pass, self.pipelines.lines.stencilless_pipeline());
        }
    }

    pub fn prep_gradient(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            self.set_pipeline(
                render_pass,
                self.pipelines.gradients.pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline(render_pass, self.pipelines.gradients.stencilless_pipeline());
        }

        self.set_group2(render_pass, bind_group);
    }

    pub fn prep_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state),
                );
            }
            (false, true) => {
                self.set_pipeline(render_pass, &self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                self.set_pipeline(
                    render_pass,
                    self.pipelines.bitmap[blend_mode].stencilless_pipeline(),
                );
            }
        }

        self.set_group2(render_pass, bind_group);
    }

    pub fn prep_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            self.set_pipeline(
                render_pass,
                self.pipelines.alpha_mask.pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline(
                render_pass,
                self.pipelines.alpha_mask.stencilless_pipeline(),
            );
        }

        self.set_group2(render_pass, bind_group);
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: wgpu::BufferSlice<'encoder>,
        num_indices: u32,
        instance_index: u32,
    ) {
        self.draw_at(
            render_pass,
            vertices,
            (indices, wgpu::IndexFormat::Uint32),
            0,
            0,
            num_indices,
            instance_index,
        );
    }

    /// `draw`, from `first_index` in `indices` (of the given format) and with `base_vertex`
    /// added to them.
    #[allow(clippy::too_many_arguments)]
    fn draw_at(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: (wgpu::BufferSlice<'encoder>, wgpu::IndexFormat),
        first_index: u32,
        base_vertex: i32,
        num_indices: u32,
        instance_index: u32,
    ) {
        if self.bound.vertices.get() != Some(vertices) {
            render_pass.set_vertex_buffer(0, vertices);
            self.bound.vertices.set(Some(vertices));
            self.bound.changed.set(true);
        }
        if self.bound.indices.get() != Some(indices) {
            render_pass.set_index_buffer(indices.0, indices.1);
            self.bound.indices.set(Some(indices));
            self.bound.changed.set(true);
        }
        // A measurement: how many draws repeat the one before (same geometry, nothing set in
        // between, the next instance), which could be one instanced draw.
        let repeats = !self.bound.changed.replace(false)
            && self.bound.last_draw.get()
                == Some((
                    first_index,
                    base_vertex,
                    num_indices,
                    instance_index.wrapping_sub(1),
                ));
        self.bound.last_draw.set(Some((
            first_index,
            base_vertex,
            num_indices,
            instance_index,
        )));
        crate::perf_counters::inc(crate::perf_counters::DRAWS);
        if repeats {
            crate::perf_counters::inc(crate::perf_counters::DRAWS_REPEAT);
        }
        render_pass.draw_indexed(
            first_index..first_index + num_indices,
            base_vertex,
            instance_index..(instance_index + 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bitmap: &'encoder BitmapHandle,
        instance_index: u32,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
        vertex_offset: Option<wgpu::BufferAddress>,
    ) {
        let texture = as_texture(bitmap);

        let descriptors = self.descriptors;
        let bind = texture.bind_group(
            false,
            smoothing,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            bitmap.clone(),
            &descriptors.bitmap_samplers,
        );
        self.prep_bitmap(render_pass, &bind.bind_group, blend_mode, render_stage3d);

        let vertex_slice = if let Some(vertex_offset) = vertex_offset {
            self.dynamic_vertex_buffer.slice(vertex_offset..)
        } else {
            self.descriptors.quad.vertices_pos_uv.slice(..)
        };

        self.draw(
            render_pass,
            vertex_slice,
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn render_texture(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
    ) {
        self.prep_bitmap(render_pass, bind_group, blend_mode, false);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_uv.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn render_shape(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        shape: &'encoder ShapeHandle,
        instance_index: u32,
    ) {
        let mesh = as_mesh(shape);
        for draw in &mesh.draws {
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            match &draw.draw_type {
                DrawType::Color => {
                    self.prep_color(render_pass);
                }
                DrawType::Gradient { bind_group, .. } => {
                    self.prep_gradient(render_pass, bind_group);
                }
                DrawType::Bitmap { binds, .. } => {
                    self.prep_bitmap(render_pass, &binds.bind_group, TrivialBlend::Normal, false);
                }
            }

            let (Some(vertices), Some(indices)) = (&mesh.vertices, &mesh.indices) else {
                continue;
            };
            // AQW build: the whole shared buffer bound once for all draws from it, each draw
            // starting at its vertices and indices in it (they start at multiples of the vertex
            // size there, see `VERTEX_BASE_MULTIPLE`; a shape made otherwise binds its own).
            let stride = match draw.draw_type {
                DrawType::Color => std::mem::size_of::<PosColorVertex>(),
                _ => std::mem::size_of::<PosUvVertex>(),
            } as wgpu::BufferAddress;
            let index_size = draw.index_format.byte_size() as wgpu::BufferAddress;
            let base_vertex = draw.vertices.start / stride;
            let first_index = draw.indices.start / index_size;
            // Turu: not on a driver without `base_vertex` (OpenGL ES 3.0 and 3.1).
            if self.descriptors.base_vertex
                && draw.vertices.start % stride == 0
                && draw.indices.start % index_size == 0
                && base_vertex <= i32::MAX as u64
                && first_index + u64::from(num_indices) <= u32::MAX as u64
            {
                self.draw_at(
                    render_pass,
                    vertices.buffer().slice(..),
                    (indices.buffer().slice(..), draw.index_format),
                    first_index as u32,
                    base_vertex as i32,
                    num_indices,
                    instance_index,
                );
            } else {
                self.draw_at(
                    render_pass,
                    vertices.buffer().slice(draw.vertices.clone()),
                    (
                        indices.buffer().slice(draw.indices.clone()),
                        draw.index_format,
                    ),
                    0,
                    0,
                    num_indices,
                    instance_index,
                );
            }
        }
    }

    pub fn render_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder wgpu::BindGroup,
        instance_index: u32,
    ) {
        if cfg!(feature = "render_debug_labels") {
            render_pass.push_debug_group("render_alpha_mask");
        }

        self.prep_alpha_mask(render_pass, bind_group);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );

        if cfg!(feature = "render_debug_labels") {
            render_pass.pop_debug_group();
        }
    }

    pub fn draw_rect(&self, render_pass: &mut wgpu::RenderPass<'encoder>, instance_index: u32) {
        self.prep_color(render_pass);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
            instance_index,
        );
    }

    pub fn draw_lines<const RECT: bool>(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        instance_index: u32,
    ) {
        self.prep_lines(render_pass);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            if RECT {
                self.descriptors.quad.indices_line_rect.slice(..)
            } else {
                self.descriptors.quad.indices_line.slice(..)
            },
            if RECT { 5 } else { 2 },
            instance_index,
        );
    }

    pub fn push_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.set_stencil(render_pass, self.num_masks - 1);
    }

    pub fn activate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.set_stencil(render_pass, self.num_masks);
    }

    pub fn deactivate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.set_stencil(render_pass, self.num_masks);
    }

    pub fn pop_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.set_stencil(render_pass, self.num_masks);
        if self.num_masks == 0 {
            self.mask_state = MaskState::NoMask;
        } else {
            self.mask_state = MaskState::DrawMaskedContent;
        };
    }

    pub fn num_masks(&self) -> u32 {
        self.num_masks
    }

    pub fn mask_state(&self) -> MaskState {
        self.mask_state
    }
}

pub enum Chunk {
    Draw {
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
    },
    Blend {
        texture: PoolOrArcTexture,
        blend_mode: ChunkBlendMode,
        needs_stencil: bool,
        /// AQW build: where the (bounded) layer goes; `None` = full-size.
        rect: Option<LayerRect>,
    },
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderBitmap {
        bitmap: BitmapHandle,
        instance_index: u32,
        vertex_offset: Option<wgpu::BufferAddress>,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
        blend_mode: TrivialBlend,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        instance_index: u32,
    },
    RenderShape {
        shape: ShapeHandle,
        instance_index: u32,
    },
    DrawRect {
        instance_index: u32,
    },
    DrawLine {
        instance_index: u32,
    },
    DrawLineRect {
        instance_index: u32,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
    /// AQW build: clip the following draws to an atlas slot.
    Scissor(LayerRect),
    /// AQW build: composite a layer from its atlas slot (`vertex_offset`: the slot's quad).
    RenderAtlas {
        page: Arc<AtlasPage>,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: TrivialBlend,
    },
    /// AQW build: blend a layer from its atlas slot (`vertex_offset`: its quad, with the
    /// slot's texel coordinates) with what the target holds in `rect`. The first draw of its
    /// chunk: what is under it is drawn, and copied to the target's blend buffer, before the
    /// chunk's pass begins (`ChunkSink::draw`). See `page_blends_enabled`.
    BlendAtlas {
        page: Arc<AtlasPage>,
        instance_index: u32,
        vertex_offset: wgpu::BufferAddress,
        blend_mode: ComplexBlend,
        rect: LayerRect,
    },
}

impl DrawCommand {
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderBitmap { .. } => "render bitmap",
            DrawCommand::RenderShape { .. } => "render shape",
            DrawCommand::RenderTexture { .. } => "render texture",
            DrawCommand::DrawRect { .. } => "draw rect",
            DrawCommand::DrawLine { .. } => "draw line",
            DrawCommand::DrawLineRect { .. } => "draw line rect",
            DrawCommand::PushMask => "push mask",
            DrawCommand::ActivateMask => "activate mask",
            DrawCommand::DeactivateMask => "deactivate mask",
            DrawCommand::PopMask => "pop mask",
            DrawCommand::RenderAlphaMask { .. } => "render alpha mask",
            DrawCommand::Scissor(..) => "scissor",
            DrawCommand::RenderAtlas { .. } => "render atlas",
            DrawCommand::BlendAtlas { .. } => "blend atlas",
        }
    }
}

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together.
/// AQW build: each chunk goes to `sink` (which draws it) as soon as it is complete.
#[expect(clippy::too_many_arguments)]
pub fn draw_chunks<'encoder, 'global: 'encoder>(
    commands: CommandList,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef,
    texture_pool: &'encoder mut TexturePool,
    opaque_target: bool,
    sink: ChunkSink<'encoder>,
) {
    WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        nearest_layer,
        texture_pool,
        opaque_target,
        sink,
    )
    .chunk_blends(commands)
}

/// AQW build: the most blended-child textures one draw chunk holds before it is drawn.
const MAX_TEXTURES_PER_CHUNK: usize = 16;

/// AQW build: where a blend layer sits in its parent target, in whole pixels.
#[derive(Clone, Copy, Debug)]
pub struct LayerRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// AQW build: layer sizes are rounded up to this, so animated content reuses pooled textures.
const LAYER_GRANULARITY: u32 = 64;
/// AQW build: margin around the content bounds (antialiasing, hairlines, bitmap smoothing,
/// pixel snapping, curve flattening).
const LAYER_PADDING: f64 = 4.0;

/// AQW build: `RUFFLE_FULL_LAYERS` keeps every blend layer full-size, for A/B comparisons.
fn bounded_layers_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_FULL_LAYERS").is_none())
}

/// Screen-space bounds in pixels.
struct PixelBounds {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl PixelBounds {
    fn empty() -> Self {
        Self {
            x0: f64::INFINITY,
            y0: f64::INFINITY,
            x1: f64::NEG_INFINITY,
            y1: f64::NEG_INFINITY,
        }
    }

    /// Adds the rectangle (x0, y0)-(x1, y1) in local pixels, transformed by `matrix`.
    fn add(&mut self, matrix: &Matrix, x0: f64, y0: f64, x1: f64, y1: f64) {
        let (a, b, c, d) = (
            matrix.a as f64,
            matrix.b as f64,
            matrix.c as f64,
            matrix.d as f64,
        );
        let (tx, ty) = (matrix.tx.to_pixels(), matrix.ty.to_pixels());
        for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let sx = a * x + c * y + tx;
            let sy = b * x + d * y + ty;
            self.x0 = self.x0.min(sx);
            self.y0 = self.y0.min(sy);
            self.x1 = self.x1.max(sx);
            self.y1 = self.y1.max(sy);
        }
    }
}

/// Adds the screen bounds of everything `commands` draws. Returns false if some bounds are
/// unknown (the layer must then stay full-size).
fn content_bounds(commands: &[Command], bounds: &mut PixelBounds) -> bool {
    for command in commands {
        match command {
            Command::RenderBitmap {
                transform, region, ..
            } => {
                if transform.perspective_projection.is_some() {
                    return false;
                }
                bounds.add(
                    &transform.matrix,
                    0.0,
                    0.0,
                    region.width() as f64,
                    region.height() as f64,
                );
            }
            Command::RenderShape { shape, transform } => {
                if transform.perspective_projection.is_some() {
                    return false;
                }
                let shape_bounds = &as_mesh(shape).bounds;
                if !shape_bounds.is_valid() {
                    continue;
                }
                bounds.add(
                    &transform.matrix,
                    shape_bounds.x_min.to_pixels(),
                    shape_bounds.y_min.to_pixels(),
                    shape_bounds.x_max.to_pixels(),
                    shape_bounds.y_max.to_pixels(),
                );
            }
            Command::DrawRect { matrix, .. }
            | Command::DrawLine { matrix, .. }
            | Command::DrawLineRect { matrix, .. } => {
                bounds.add(matrix, 0.0, 0.0, 1.0, 1.0);
            }
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => {
                if !content_bounds(&maskee_commands.commands, bounds)
                    || !content_bounds(&mask_commands.commands, bounds)
                {
                    return false;
                }
            }
            Command::Blend(commands, _) => {
                if !content_bounds(&commands.commands, bounds) {
                    return false;
                }
            }
            Command::RenderStage3D { .. } => return false,
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => {}
        }
    }
    true
}

/// Moves everything `commands` draws by (-x, -y) pixels.
fn translate_commands(commands: &mut [Command], x: u32, y: u32) {
    let (dx, dy) = (Twips::from_pixels(x as f64), Twips::from_pixels(y as f64));
    let shift = |matrix: &mut Matrix| {
        matrix.tx -= dx;
        matrix.ty -= dy;
    };
    for command in commands {
        match command {
            Command::RenderBitmap { transform, .. }
            | Command::RenderShape { transform, .. }
            | Command::RenderStage3D { transform, .. } => shift(&mut transform.matrix),
            Command::DrawRect { matrix, .. }
            | Command::DrawLine { matrix, .. }
            | Command::DrawLineRect { matrix, .. } => shift(matrix),
            Command::RenderAlphaMask {
                maskee_commands,
                mask_commands,
            } => {
                translate_commands(&mut maskee_commands.commands, x, y);
                translate_commands(&mut mask_commands.commands, x, y);
            }
            Command::Blend(commands, _) => translate_commands(&mut commands.commands, x, y),
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => {}
        }
    }
}

/// AQW build: the part of a `target_width` x `target_height` target that a blend layer needs.
///
/// Stock Ruffle made every blend layer as big as its target: in crowded AQW rooms that is
/// 700+ full-screen multisampled layers per frame (each cleared, drawn, resolved and
/// composited), most of them for one small avatar part, and the GPU becomes the bottleneck.
/// A layer covering just its content's bounds (padded, in whole pixels so rasterization is
/// identical) produces the same image: outside its content a layer is transparent, which
/// every blend mode handled here leaves unchanged.
///
/// `None` keeps the layer full-size: bounds unknown (Stage3D, 3D projection), or the layer
/// would cover most of the target anyway.
///
/// Also returns the padded content bounds before rounding up (what the atlas uses).
fn layer_rect_and_exact_size(
    commands: &CommandList,
    target_width: u32,
    target_height: u32,
) -> Option<(LayerRect, LayerRect)> {
    if !bounded_layers_enabled() {
        return None;
    }
    let mut bounds = PixelBounds::empty();
    if !content_bounds(&commands.commands, &mut bounds) {
        return None;
    }
    let smallest = LayerRect {
        x: 0,
        y: 0,
        width: LAYER_GRANULARITY.min(target_width),
        height: LAYER_GRANULARITY.min(target_height),
    };
    if bounds.x0 > bounds.x1 || bounds.y0 > bounds.y1 {
        // Draws nothing.
        return Some((
            smallest,
            LayerRect {
                width: 0,
                height: 0,
                ..smallest
            },
        ));
    }
    // Far-off coordinates (or NaN) stay full-size, so shifting the commands can't overflow.
    if ![bounds.x0, bounds.y0, bounds.x1, bounds.y1]
        .iter()
        .all(|v| v.is_finite() && v.abs() < 1.0e6)
    {
        return None;
    }
    let x0 = (bounds.x0 - LAYER_PADDING).floor().max(0.0);
    let y0 = (bounds.y0 - LAYER_PADDING).floor().max(0.0);
    let x1 = (bounds.x1 + LAYER_PADDING).ceil().min(target_width as f64);
    let y1 = (bounds.y1 + LAYER_PADDING).ceil().min(target_height as f64);
    if x1 <= x0 || y1 <= y0 {
        // Entirely outside the target.
        return Some((
            smallest,
            LayerRect {
                width: 0,
                height: 0,
                ..smallest
            },
        ));
    }
    let exact = LayerRect {
        x: x0 as u32,
        y: y0 as u32,
        width: (x1 - x0) as u32,
        height: (y1 - y0) as u32,
    };
    let round_up = |v: u32| v.div_ceil(LAYER_GRANULARITY) * LAYER_GRANULARITY;
    let width = round_up((x1 - x0) as u32).min(target_width);
    let height = round_up((y1 - y0) as u32).min(target_height);
    if (width as u64 * height as u64) * 2 > target_width as u64 * target_height as u64 {
        return None;
    }
    // Rounding up may push the layer past the right/bottom edge; slide it back inside.
    Some((
        LayerRect {
            x: (x0 as u32).min(target_width - width),
            y: (y0 as u32).min(target_height - height),
            width,
            height,
        },
        exact,
    ))
}

/// AQW build: the shared layer atlas.
///
/// Every blend layer used to be its own offscreen target: a pooled texture, at least one render
/// pass to draw its content (plus a resolve pass with MSAA), and a bind group to composite it.
/// Crowded AQW rooms have ~400 such layers per frame (avatar parts with Multiply/Add/Overlay),
/// so most of a frame's render passes were per-layer overhead. Now the content of the next
/// batch of layers whose content has no blends of its own ("leaf" layers) is drawn side by side
/// into one shared page texture: a handful of passes and one resolve for the whole batch, then
/// each layer is composited from its slot.
///
/// Same image: a layer's content is moved by whole pixels into its slot (rasterization is
/// identical, with or without MSAA) and clipped to the slot with a scissor rectangle (as the
/// layer's own texture edges clipped it). Trivial blends composite a quad sampling the slot
/// texel for texel; complex blends (Overlay etc.) copy the slot into a texture of the layer's
/// size and blend exactly as before. Layers that aren't leaves, aren't bounded, or don't fit a
/// page use the old path. At most two pages per target level are alive at a time (the chunk
/// holding composites from the previous page is drawn before the next page is rendered).
/// `RUFFLE_NO_ATLAS` turns this off, for A/B comparisons.
const ATLAS_PAGE_WIDTH: u32 = 2048;
const ATLAS_PAGE_MAX_HEIGHT: u32 = 1024;
const ATLAS_MAX_LAYERS_PER_PAGE: usize = 256;

fn atlas_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_NO_ATLAS").is_none())
        && crate::ATLAS_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// AQW build, OFF unless `RUFFLE_PAGE_BLENDS` is set (a `page-blends` file next to the exe):
/// a complex blend (overlay, hard light...) of a layer in the atlas reads the layer straight
/// from its page and the backdrop from the target's one blend buffer, as a draw at the start
/// of the next chunk (`DrawCommand::BlendAtlas`). Was, per blend: the slot copied to a pooled
/// texture, the backdrop copied to another, a bind group made for the two. The same pixels
/// are read and written; a crowded room has 150 to 280 of these in every picture.
pub(crate) fn page_blends_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var_os("RUFFLE_PAGE_BLENDS").is_some();
        if enabled {
            tracing::info!(
                target: "ruffle_aqw_perf",
                "complex blends read their layers from the atlas pages (RUFFLE_PAGE_BLENDS)"
            );
        }
        enabled
    })
}

/// A rendered atlas page (resolved), with its bind group for compositing.
#[derive(Debug)]
pub struct AtlasPage {
    pub(crate) texture: PoolOrArcTexture,
    binds: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl AtlasPage {
    /// AQW build: for the layer cache's budget (see `crate::layer_cache`).
    pub fn bytes(&self) -> u64 {
        self.area() * 4
    }

    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// A leaf blend layer of the current target that can go into the atlas.
#[derive(Clone)]
struct AtlasCandidate {
    /// Where the layer goes in the target: its exact padded bounds for trivial blends, the
    /// rounded `layer_rect` for complex blends (whose composite path pools textures by size).
    rect: LayerRect,
    blend_type: BlendType,
    /// The perf counter `blend` would have counted.
    counter: usize,
}

struct AtlasPlacement {
    page: Arc<AtlasPage>,
    x: u32,
    y: u32,
}

/// The atlas state of one target's command list, indexed like its commands.
#[derive(Default)]
struct AtlasLevel {
    candidates: Vec<Option<AtlasCandidate>>,
    placements: Vec<Option<AtlasPlacement>>,
    /// AQW build: each candidate's layer-cache key (see `crate::layer_cache`), and whether a
    /// miss is rendered into a page that is kept (its key repeated) or a page of this picture.
    keys: Vec<Option<u64>>,
    kept: Vec<bool>,
    /// AQW build, check mode: a sampled hit's cached placement and what the layer draws; the
    /// layer is rendered fresh as well, the two compared, and the cached one shown (see
    /// `crate::layer_cache`).
    checks: Vec<Option<(AtlasPlacement, String)>>,
}

/// AQW build: moves everything `commands` draws by (dx, dy) pixels.
fn offset_commands(commands: &mut [Command], dx: i32, dy: i32) {
    let (dx, dy) = (Twips::from_pixels_i32(dx), Twips::from_pixels_i32(dy));
    let shift = |matrix: &mut Matrix| {
        matrix.tx += dx;
        matrix.ty += dy;
    };
    for command in commands {
        match command {
            Command::RenderBitmap { transform, .. }
            | Command::RenderShape { transform, .. }
            | Command::RenderStage3D { transform, .. } => shift(&mut transform.matrix),
            Command::DrawRect { matrix, .. }
            | Command::DrawLine { matrix, .. }
            | Command::DrawLineRect { matrix, .. } => shift(matrix),
            Command::RenderAlphaMask { .. } | Command::Blend(..) => {
                unreachable!("leaf layers have no blends or alpha masks")
            }
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => {}
        }
    }
}

/// AQW build: draws each command list into its slot of a new `width` x `height` page (cleared
/// transparent), like drawing it into its own texture of the slot's size: moved by whole pixels
/// and clipped to the slot. Only for leaf command lists (see `is_leaf`). Returns the resolved
/// page. Used for batched cacheAsBitmap filters (see `crate::cache_batch`).
#[expect(clippy::too_many_arguments)]
pub(crate) fn draw_into_slots<'global>(
    surface: &Surface,
    descriptors: &Descriptors,
    meshes: &Vec<Mesh>,
    staging_belt: &mut wgpu::util::StagingBelt,
    dynamic_transforms: &DynamicTransforms,
    draw_encoder: &mut Scope<'global, wgpu::CommandEncoder>,
    texture_pool: &mut TexturePool,
    width: u32,
    height: u32,
    slots: Vec<(CommandList, LayerRect)>,
) -> CommandTarget {
    let target = CommandTarget::new(
        descriptors,
        texture_pool,
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        wgpu::TextureFormat::Rgba8Unorm,
        surface.sample_count(),
        RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
        draw_encoder,
    );
    target.set_lazy_resolve();
    {
        let sink = ChunkSink {
            pipelines: &surface.pipelines,
            target: &target,
            blend_parent: LayerRef::None,
            num_masks: 0,
            mask_state: MaskState::NoMask,
            pending_blend: None,
            page_binds: Vec::new(),
        };
        let mut handler = WgpuCommandHandler::new(
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            surface.quality(),
            width,
            height,
            LayerRef::None,
            texture_pool,
            false,
            sink,
        );
        for (mut list, slot) in slots {
            offset_commands(&mut list.commands, slot.x as i32, slot.y as i32);
            // Chunks are drawn as soon as they're complete, so no pass is open here.
            crate::segments::maybe_cut(handler.descriptors, handler.draw_encoder);
            handler.begin_scissor(slot);
            list.execute(&mut handler);
        }
        handler.finish();
    }
    target.ensure_cleared(draw_encoder);
    target.resolve(draw_encoder);
    target
}

/// Whether a layer's content has no blends, alpha masks or Stage3D of its own.
pub(crate) fn is_leaf(commands: &CommandList) -> bool {
    !commands.commands.iter().any(|command| {
        matches!(
            command,
            Command::Blend(..) | Command::RenderAlphaMask { .. } | Command::RenderStage3D { .. }
        )
    })
}

struct WgpuCommandHandler<'encoder, 'global: 'encoder> {
    descriptors: &'encoder Descriptors,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef<'encoder>,
    meshes: &'encoder Vec<Mesh>,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    texture_pool: &'encoder mut TexturePool,
    emulate_lines: bool,
    /// The target is fully opaque (cleared with an opaque color), so every pixel's alpha
    /// stays 1: all trivial blends composite alpha with OVER.
    opaque_target: bool,

    sink: ChunkSink<'encoder>,
    /// Layer textures held by `current` (see `MAX_TEXTURES_PER_CHUNK`).
    current_textures: usize,
    /// AQW build: atlas candidates and placements of the command list being chunked.
    atlas: AtlasLevel,
    /// AQW build: the atlas slot being drawn, re-applied at the start of every chunk.
    layer_scissor: Option<LayerRect>,
    current: Vec<DrawCommand>,
    transforms: BufferBuilder,
    vertices: BufferBuilder,
    needs_stencil: bool,
    num_masks: i32,
}

impl<'encoder, 'global: 'encoder> WgpuCommandHandler<'encoder, 'global> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        descriptors: &'encoder Descriptors,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'encoder Vec<Mesh>,
        quality: StageQuality,
        width: u32,
        height: u32,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
        opaque_target: bool,
        sink: ChunkSink<'encoder>,
    ) -> Self {
        let transforms = Self::new_transforms(descriptors, dynamic_transforms);
        let vertices = Self::new_vertices(descriptors, dynamic_transforms);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        Self {
            descriptors,
            quality,
            width,
            height,
            nearest_layer,
            meshes,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            texture_pool,
            emulate_lines,
            opaque_target,

            sink,
            current_textures: 0,
            atlas: AtlasLevel::default(),
            layer_scissor: None,
            current: vec![],
            transforms,
            vertices,
            needs_stencil: false,
            num_masks: 0,
        }
    }

    fn new_transforms(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut transforms = BufferBuilder::new(&descriptors.limits, 0);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        transforms
    }

    fn new_vertices(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut vertices = BufferBuilder::new(&descriptors.limits, 0);
        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());
        vertices
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(mut self, commands: CommandList) {
        if atlas_enabled()
            && commands
                .commands
                .iter()
                .any(|command| matches!(command, Command::Blend(..)))
        {
            let mut commands: Vec<Option<Command>> =
                commands.commands.into_iter().map(Some).collect();
            self.plan_atlas(&mut commands);
            for i in 0..commands.len() {
                let candidate_size = self.atlas.candidates[i]
                    .as_ref()
                    .map(|candidate| (candidate.rect.width, candidate.rect.height));
                if let Some((width, height)) = candidate_size {
                    // No render pass is open between commands: a safe point to start a new
                    // parallel-encoding segment (as `blend` does for the old path).
                    crate::segments::maybe_cut(self.descriptors, self.draw_encoder);
                    if commands[i].is_some()
                        && self.atlas.placements[i].is_none()
                        && width > 0
                        && height > 0
                    {
                        self.render_atlas_page(i, &mut commands);
                    }
                    if let Some((cached, summary)) = self.atlas.checks[i].take() {
                        self.queue_layer_check(i, cached, summary);
                    }
                    self.atlas_composite(i);
                } else if let Some(command) = commands[i].take() {
                    command.dispatch(&mut self);
                }
            }
        } else {
            commands.execute(&mut self);
        }
        self.finish();
    }

    /// Draws what's left and any complex blend still waiting.
    fn finish(mut self) {
        let current = mem::take(&mut self.current);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );
        let vertices = mem::replace(
            &mut self.vertices,
            Self::new_vertices(self.descriptors, self.dynamic_transforms),
        );

        if !current.is_empty() {
            self.emit(Chunk::Draw {
                chunk: current,
                needs_stencil,
                transforms,
                vertices,
            });
        }
        self.sink.flush(
            self.descriptors,
            self.dynamic_transforms,
            self.draw_encoder,
            self.texture_pool,
        );
    }

    /// The draw chunk collected so far, leaving an empty one.
    fn take_current_chunk(&mut self) -> Chunk {
        Chunk::Draw {
            chunk: mem::take(&mut self.current),
            needs_stencil: self.needs_stencil,
            transforms: mem::replace(
                &mut self.transforms,
                Self::new_transforms(self.descriptors, self.dynamic_transforms),
            ),
            vertices: mem::replace(
                &mut self.vertices,
                Self::new_vertices(self.descriptors, self.dynamic_transforms),
            ),
        }
    }

    /// AQW build: clips what follows to `rect` (an atlas slot), also in later chunks.
    fn begin_scissor(&mut self, rect: LayerRect) {
        self.layer_scissor = Some(rect);
        self.current.push(DrawCommand::Scissor(rect));
    }

    /// AQW build: finds the blend layers of this command list that can go into the atlas, and
    /// which of them were rendered before and can be composited again (see
    /// `crate::layer_cache`): those get their placement here and their command is taken.
    fn plan_atlas(&mut self, commands: &mut [Option<Command>]) {
        let candidates: Vec<Option<AtlasCandidate>> = commands
            .iter()
            .map(|command| match command {
                Some(Command::Blend(list, blend_mode)) => self.atlas_candidate(list, blend_mode),
                _ => None,
            })
            .collect();
        let mut placements: Vec<Option<AtlasPlacement>> = commands.iter().map(|_| None).collect();
        let mut keys: Vec<Option<u64>> = vec![None; commands.len()];
        let mut kept = vec![false; commands.len()];
        let mut checks: Vec<Option<(AtlasPlacement, String)>> =
            commands.iter().map(|_| None).collect();
        if crate::layer_cache::enabled() {
            // (Timed as `layer_keys`: hashing every leaf layer's list, every picture.)
            let started = std::time::Instant::now();
            let mut cache = self.texture_pool.layer_cache();
            for (index, candidate) in candidates.iter().enumerate() {
                let Some(candidate) = candidate else { continue };
                let Some(Command::Blend(list, _)) = &commands[index] else {
                    continue;
                };
                if candidate.rect.width == 0 || candidate.rect.height == 0 {
                    continue;
                }
                let Some(key) = crate::layer_cache::layer_key(
                    list,
                    self.width,
                    self.height,
                    self.quality,
                    self.opaque_target,
                    candidate.rect,
                    matches!(candidate.blend_type, BlendType::Complex(_)),
                ) else {
                    continue;
                };
                keys[index] = Some(key);
                if let Some(cached) = cache.lookup(key, candidate.rect.width, candidate.rect.height)
                {
                    crate::perf_counters::inc(crate::perf_counters::ATLAS_HIT);
                    let placement = AtlasPlacement {
                        page: cached.page,
                        x: cached.x,
                        y: cached.y,
                    };
                    if cache.should_check(key) {
                        // Rendered fresh too (into a page of this picture, not kept), for the
                        // comparison; composited from the cache as usual.
                        checks[index] = Some((placement, crate::layer_cache::describe(list)));
                    } else {
                        placements[index] = Some(placement);
                        commands[index] = None;
                    }
                } else {
                    crate::perf_counters::inc(crate::perf_counters::ATLAS_MISS);
                    kept[index] = cache.admit(key);
                }
            }
            drop(cache);
            crate::perf_counters::timers::add_nanos(
                crate::perf_counters::timers::LAYER_KEYS,
                started.elapsed().as_nanos() as u64,
            );
        }
        self.atlas = AtlasLevel {
            candidates,
            placements,
            keys,
            kept,
            checks,
        };
    }

    /// AQW build, check mode: copies the cached and the fresh slot of candidate `index` into
    /// a buffer for `LayerCache::run_checks`, and shows the cached one.
    fn queue_layer_check(&mut self, index: usize, cached: AtlasPlacement, summary: String) {
        let Some(fresh) = self.atlas.placements[index].take() else {
            self.atlas.placements[index] = Some(cached);
            return;
        };
        let Some(candidate) = &self.atlas.candidates[index] else {
            self.atlas.placements[index] = Some(cached);
            return;
        };
        let (width, height) = (candidate.rect.width, candidate.rect.height);
        let padded_row = (width * 4).div_ceil(256) * 256;
        let buffer = self
            .descriptors
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("layer cache check"),
                size: padded_row as u64 * height as u64 * 2,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        for (half, placement) in [(0u64, &cached), (1u64, &fresh)] {
            self.draw_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: placement.page.texture.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: placement.x,
                        y: placement.y,
                        z: 0,
                    },
                    aspect: Default::default(),
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: half * padded_row as u64 * height as u64,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let mut cache = self.texture_pool.layer_cache();
        let recorded = cache.picture();
        cache.queue_check(crate::layer_cache::PendingCheck {
            key: self.atlas.keys[index].unwrap_or(0),
            summary,
            buffer,
            width,
            height,
            padded_row,
            recorded,
            mapped: Default::default(),
            map_requested: false,
        });
        drop(cache);
        self.atlas.placements[index] = Some(cached);
    }

    /// AQW build: how `blend` would draw this layer, if it can go into the atlas.
    fn atlas_candidate(
        &self,
        commands: &CommandList,
        blend_mode: &RenderBlendMode,
    ) -> Option<AtlasCandidate> {
        use crate::perf_counters as pc;
        let is_layer = matches!(blend_mode, RenderBlendMode::Builtin(BlendMode::Layer));
        // The same choices as `blend`.
        let (blend_type, counter) = match BlendType::from(blend_mode.clone()) {
            BlendType::Complex(ComplexBlend::Alpha | ComplexBlend::Erase)
            | BlendType::Shader(_) => {
                return None;
            }
            BlendType::Complex(ComplexBlend::Multiply)
                if self.opaque_target && fast_multiply_enabled() =>
            {
                (
                    BlendType::Trivial(TrivialBlend::MultiplyOntoOpaque),
                    pc::MULTIPLY_FAST,
                )
            }
            blend_type => {
                let counter = match &blend_type {
                    _ if is_layer => pc::BLEND_LAYER,
                    BlendType::Complex(mode) => {
                        pc::COMPLEX_BASE + enum_map::Enum::into_usize(*mode)
                    }
                    _ => pc::BLEND_TRIVIAL,
                };
                (blend_type, counter)
            }
        };
        let Some((rounded, exact)) = layer_rect_and_exact_size(commands, self.width, self.height)
        else {
            pc::inc(pc::ATLAS_FULL);
            return None;
        };
        if !is_leaf(commands) {
            pc::inc(pc::ATLAS_NESTED);
            return None;
        }
        let rect = if matches!(blend_type, BlendType::Complex(_)) {
            rounded
        } else {
            exact
        };
        if rect.width > ATLAS_PAGE_WIDTH || rect.height > ATLAS_PAGE_MAX_HEIGHT {
            pc::inc(pc::ATLAS_TOO_BIG);
            return None;
        }
        Some(AtlasCandidate {
            rect,
            blend_type,
            counter,
        })
    }

    /// AQW build: renders the content of the atlas candidates from `start` on (as many as fit
    /// one page) into a new page, in rows left to right.
    fn render_atlas_page(&mut self, start: usize, commands: &mut [Option<Command>]) {
        // The current chunk may hold composites from the previous page: draw it first, so that
        // page goes back to the pool before this one is taken.
        if !self.current.is_empty() {
            let chunk = self.take_current_chunk();
            self.emit(chunk);
        }

        // The next candidates in order (as many as one page could hold), packed tightly.
        // AQW build: only those not found in the layer cache, and of the same kind as the
        // first: a page that is kept holds only layers that repeat (see `crate::layer_cache`).
        let keep = self.atlas.kept[start];
        let mut upcoming: Vec<(usize, LayerRect)> = Vec::new();
        for (index, command) in commands.iter().enumerate().skip(start) {
            let Some(candidate) = &self.atlas.candidates[index] else {
                continue;
            };
            if command.is_none()
                || self.atlas.placements[index].is_some()
                || self.atlas.kept[index] != keep
                || candidate.rect.width == 0
                || candidate.rect.height == 0
            {
                continue;
            }
            upcoming.push((index, candidate.rect));
            if upcoming.len() >= ATLAS_MAX_LAYERS_PER_PAGE {
                break;
            }
        }
        if upcoming.is_empty() {
            return;
        }
        let sizes: Vec<(u32, u32)> = upcoming
            .iter()
            .map(|(_, rect)| (rect.width, rect.height))
            .collect();
        let (count, (width, height), positions) =
            crate::packing::pack(&sizes, ATLAS_MAX_LAYERS_PER_PAGE);
        crate::perf_counters::count_page(
            (width, height),
            upcoming[..count]
                .iter()
                .map(|(_, rect)| (rect.width, rect.height)),
        );
        let layers: Vec<(usize, LayerRect, u32, u32)> = upcoming[..count]
            .iter()
            .zip(positions)
            .map(|(&(index, rect), (x, y))| (index, rect, x, y))
            .collect();

        let surface = Surface::new(
            self.descriptors,
            self.quality,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let target = CommandTarget::new(
            self.descriptors,
            self.texture_pool,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            surface.sample_count(),
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.draw_encoder,
        );
        target.set_lazy_resolve();
        {
            let sink = ChunkSink {
                pipelines: &surface.pipelines,
                target: &target,
                blend_parent: LayerRef::None,
                num_masks: 0,
                mask_state: MaskState::NoMask,
                pending_blend: None,
                page_binds: Vec::new(),
            };
            // Like the layer's own target: cleared transparent, so not opaque.
            let mut handler = WgpuCommandHandler::new(
                self.descriptors,
                self.staging_belt,
                self.dynamic_transforms,
                self.draw_encoder,
                self.meshes,
                self.quality,
                width,
                height,
                LayerRef::None,
                self.texture_pool,
                false,
                sink,
            );
            for &(index, rect, slot_x, slot_y) in &layers {
                let Some(Command::Blend(mut list, _)) = commands[index].take() else {
                    unreachable!("atlas candidates are blends");
                };
                offset_commands(
                    &mut list.commands,
                    slot_x as i32 - rect.x as i32,
                    slot_y as i32 - rect.y as i32,
                );
                // Chunks are drawn as soon as they're complete, so no pass is open here.
                crate::segments::maybe_cut(handler.descriptors, handler.draw_encoder);
                handler.begin_scissor(LayerRect {
                    x: slot_x,
                    y: slot_y,
                    width: rect.width,
                    height: rect.height,
                });
                list.execute(&mut handler);
            }
            handler.finish();
        }
        target.ensure_cleared(self.draw_encoder);
        target.resolve(self.draw_encoder);

        let texture = target.take_color_texture();
        let binds = self
            .descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.descriptors.bind_layouts.bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            self.descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });
        let page = Arc::new(AtlasPage {
            texture,
            binds,
            width,
            height,
        });
        crate::perf_counters::inc(crate::perf_counters::ATLAS_PAGES);
        for (index, rect, slot_x, slot_y) in layers {
            // AQW build: a kept page's layers are found again by the next pictures.
            if keep && let Some(key) = self.atlas.keys[index] {
                self.texture_pool.layer_cache().insert(
                    key,
                    page.clone(),
                    slot_x,
                    slot_y,
                    rect.width,
                    rect.height,
                );
            }
            self.atlas.placements[index] = Some(AtlasPlacement {
                page: page.clone(),
                x: slot_x,
                y: slot_y,
            });
        }
    }

    /// AQW build: composites atlas candidate `index` from its page, like `blend` composites a
    /// layer from its own texture.
    fn atlas_composite(&mut self, index: usize) {
        let candidate = self.atlas.candidates[index]
            .clone()
            .expect("an atlas candidate");
        crate::perf_counters::inc(candidate.counter);
        crate::perf_counters::inc(crate::perf_counters::ATLAS_LEAF);
        // A layer that draws nothing changes nothing.
        let Some(placement) = self.atlas.placements[index].take() else {
            return;
        };
        let rect = candidate.rect;
        match candidate.blend_type {
            BlendType::Trivial(blend_mode) => {
                let page = placement.page;
                let (page_width, page_height) = (page.width as f32, page.height as f32);
                let u0 = placement.x as f32 / page_width;
                let v0 = placement.y as f32 / page_height;
                let u1 = (placement.x + rect.width) as f32 / page_width;
                let v1 = (placement.y + rect.height) as f32 / page_height;
                let vertices = [
                    PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                    PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                    PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                    PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
                ];
                let matrix = Matrix::translate(
                    Twips::from_pixels(rect.x as f64),
                    Twips::from_pixels(rect.y as f64),
                ) * Matrix::scale(rect.width as f32, rect.height as f32);
                self.add_to_current_with_vertices(
                    matrix,
                    0.0,
                    Default::default(),
                    Some(&vertices),
                    |instance_index, vertex_offset| DrawCommand::RenderAtlas {
                        page,
                        instance_index,
                        vertex_offset: vertex_offset.expect("atlas quads have vertices"),
                        blend_mode,
                    },
                );
            }
            BlendType::Complex(blend_mode)
                if page_blends_enabled()
                    && !matches!(blend_mode, ComplexBlend::Alpha | ComplexBlend::Erase) =>
            {
                // What it blends with has to be on the target before its pass begins: it
                // starts a chunk, and the draws after it go on in the same pass.
                if !self.current.is_empty() {
                    let chunk = self.take_current_chunk();
                    self.emit(chunk);
                }
                self.needs_stencil = self.num_masks > 0;
                if let Some(scissor) = self.layer_scissor {
                    self.current.push(DrawCommand::Scissor(scissor));
                }
                let page = placement.page;
                // The slot in texels: the shader reads the page texel under each pixel.
                let (u0, v0) = (placement.x as f32, placement.y as f32);
                let (u1, v1) = (u0 + rect.width as f32, v0 + rect.height as f32);
                let vertices = [
                    PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                    PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                    PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                    PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
                ];
                let matrix = Matrix::translate(
                    Twips::from_pixels(rect.x as f64),
                    Twips::from_pixels(rect.y as f64),
                ) * Matrix::scale(rect.width as f32, rect.height as f32);
                self.add_to_current_with_vertices(
                    matrix,
                    0.0,
                    Default::default(),
                    Some(&vertices),
                    |instance_index, vertex_offset| DrawCommand::BlendAtlas {
                        page,
                        instance_index,
                        vertex_offset: vertex_offset.expect("atlas quads have vertices"),
                        blend_mode,
                        rect,
                    },
                );
            }
            BlendType::Complex(blend_mode) => {
                let size = wgpu::Extent3d {
                    width: rect.width,
                    height: rect.height,
                    depth_or_array_layers: 1,
                };
                let texture = self.texture_pool.get_texture(
                    self.descriptors,
                    size,
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC,
                    wgpu::TextureFormat::Rgba8Unorm,
                    1,
                );
                self.draw_encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: placement.page.texture.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: placement.x,
                            y: placement.y,
                            z: 0,
                        },
                        aspect: Default::default(),
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.0,
                        mip_level: 0,
                        origin: Default::default(),
                        aspect: Default::default(),
                    },
                    size,
                );
                if !self.current.is_empty() {
                    let chunk = self.take_current_chunk();
                    self.emit(chunk);
                }
                self.emit(Chunk::Blend {
                    texture: PoolOrArcTexture::Pool(texture),
                    blend_mode: ChunkBlendMode::Complex(blend_mode),
                    needs_stencil: self.num_masks > 0,
                    rect: Some(rect),
                });
                self.needs_stencil = self.num_masks > 0;
            }
            BlendType::Shader(_) => unreachable!("shader blends aren't atlas candidates"),
        }
    }

    /// Draws a completed chunk onto the target.
    fn emit(&mut self, chunk: Chunk) {
        if matches!(chunk, Chunk::Draw { .. }) {
            self.current_textures = 0;
        }
        self.sink.push(
            chunk,
            self.descriptors,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            self.texture_pool,
        );
    }

    fn add_to_current(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        command_builder: impl FnOnce(u32) -> DrawCommand,
    ) {
        self.add_to_current_with_vertices(matrix, tz, color_transform, None, |instance_index, _| {
            command_builder(instance_index)
        })
    }

    fn add_to_current_with_vertices(
        &mut self,
        matrix: Matrix,
        tz: f64,
        color_transform: ColorTransform,
        vertices: Option<&[PosUvVertex]>,
        command_builder: impl FnOnce(u32, Option<wgpu::BufferAddress>) -> DrawCommand,
    ) {
        let transform = Transforms {
            world_matrix: [
                [matrix.a, matrix.b, 0.0, 0.0],
                [matrix.c, matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    matrix.tx.to_pixels() as f32,
                    matrix.ty.to_pixels() as f32,
                    tz as f32,
                    1.0,
                ],
            ],
            mult_color: color_transform.mult_rgba_normalized(),
            add_color: color_transform.add_rgba_normalized(),
        };
        if let (Ok(transform_range), Ok(vertices_range)) = (
            self.transforms.add(&[transform]),
            vertices.map(|v| self.vertices.add(v)).transpose(),
        ) {
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        } else {
            let chunk = Chunk::Draw {
                chunk: mem::take(&mut self.current),
                needs_stencil: self.needs_stencil,
                transforms: mem::replace(
                    &mut self.transforms,
                    Self::new_transforms(self.descriptors, self.dynamic_transforms),
                ),
                vertices: mem::replace(
                    &mut self.vertices,
                    Self::new_vertices(self.descriptors, self.dynamic_transforms),
                ),
            };
            self.emit(chunk);
            if let Some(rect) = self.layer_scissor {
                self.current.push(DrawCommand::Scissor(rect));
            }
            let transform_range = self
                .transforms
                .add(&[transform])
                .expect("Buffer must be able to fit a new thing, it was just emptied");
            let vertices_range = vertices.map(|v| {
                self.vertices
                    .add(v)
                    .expect("Buffer must be able to fit a new thing, it was just emptied")
            });
            self.current.push(command_builder(
                (transform_range.start as usize / size_of::<Transforms>()) as u32,
                vertices_range.map(|v| v.start),
            ));
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_, '_> {
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        // No render pass is open while blends are being chunked, so this is a safe point to
        // start a new parallel-encoding segment.
        crate::segments::maybe_cut(self.descriptors, self.draw_encoder);
        let is_layer = matches!(blend_mode, RenderBlendMode::Builtin(BlendMode::Layer));
        let blend_type = match BlendType::from(blend_mode.clone()) {
            // AQW build: an Alpha or Erase blend with no Layer above it is dropped when the
            // chunks are drawn (see `Surface::draw_commands`), so don't render its contents
            // into an offscreen surface first.
            BlendType::Complex(ComplexBlend::Alpha | ComplexBlend::Erase)
                if matches!(self.nearest_layer, LayerRef::None) =>
            {
                crate::perf_counters::inc(crate::perf_counters::ALPHA_ERASE_SKIPPED);
                return;
            }
            // AQW build: crowded AQW scenes hit hundreds of Multiply blends per frame, each a
            // complex blend (blend-buffer copy plus extra render passes). Onto an opaque
            // target it is exactly expressible with fixed-function blending.
            BlendType::Complex(ComplexBlend::Multiply)
                if self.opaque_target && fast_multiply_enabled() =>
            {
                crate::perf_counters::inc(crate::perf_counters::MULTIPLY_FAST);
                BlendType::Trivial(TrivialBlend::MultiplyOntoOpaque)
            }
            blend_type => {
                crate::perf_counters::inc(match &blend_type {
                    _ if is_layer => crate::perf_counters::BLEND_LAYER,
                    BlendType::Complex(mode) => {
                        crate::perf_counters::COMPLEX_BASE + enum_map::Enum::into_usize(*mode)
                    }
                    _ => crate::perf_counters::BLEND_TRIVIAL,
                });
                blend_type
            }
        };
        // AQW build: size the layer to its content where the blend allows it (see `layer_rect_and_exact_size`).
        // Alpha/Erase change their parent outside their own content, shader blends read the
        // whole parent: those stay full-size.
        let rect_and_size = match &blend_type {
            BlendType::Trivial(_) => layer_rect_and_exact_size(&commands, self.width, self.height),
            BlendType::Complex(ComplexBlend::Alpha | ComplexBlend::Erase) => None,
            BlendType::Complex(_) => layer_rect_and_exact_size(&commands, self.width, self.height),
            BlendType::Shader(_) => None,
        };
        let rect = rect_and_size.map(|(rect, _)| rect);
        let mut commands = commands;
        let (layer_width, layer_height) = match rect {
            Some(rect) => {
                translate_commands(&mut commands.commands, rect.x, rect.y);
                crate::perf_counters::inc(crate::perf_counters::BOUNDED_LAYERS);
                (rect.width, rect.height)
            }
            None => (self.width, self.height),
        };
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            layer_width,
            layer_height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let target_layer = if is_layer {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let clear_color = blend_type.default_color();
        let target = surface.draw_commands(
            RenderTargetMode::FreshWithColor(clear_color),
            self.descriptors,
            self.meshes,
            commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            target_layer,
            self.texture_pool,
        );
        target.ensure_cleared(self.draw_encoder);

        // We currently do not support shader blends in masks. In order not to
        // break other parts of the scene, we just fall back to a normal blend.
        //
        // TODO Add support for shader blends in masks.
        let is_shader_blend_in_mask =
            self.num_masks > 0 && matches!(blend_type, BlendType::Shader(_));
        let blend_type = if is_shader_blend_in_mask {
            BlendType::Trivial(TrivialBlend::Normal)
        } else {
            blend_type
        };

        match blend_type {
            BlendType::Trivial(blend_mode) => {
                let offset = rect.map_or(Matrix::IDENTITY, |rect| {
                    Matrix::translate(
                        Twips::from_pixels(rect.x as f64),
                        Twips::from_pixels(rect.y as f64),
                    )
                });
                let transform = Transform {
                    matrix: offset * Matrix::scale(target.width() as f32, target.height() as f32),
                    tz: 0.0,
                    color_transform: Default::default(),
                    perspective_projection: None,
                };
                let texture = target.take_color_texture();
                let bind_group =
                    self.descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.descriptors.bind_layouts.bitmap,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(texture.view()),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        self.descriptors.bitmap_samplers.get_sampler(false, false),
                                    ),
                                },
                            ],
                            label: None,
                        });
                self.add_to_current(
                    transform.matrix,
                    transform.tz,
                    transform.color_transform,
                    |instance_index| DrawCommand::RenderTexture {
                        _texture: texture,
                        binds: bind_group,
                        instance_index,
                        blend_mode,
                    },
                );
                // AQW build: a chunk keeps its blended children's textures until it is drawn;
                // past a few, draw it now so they return to the pool (costs one extra pass).
                self.current_textures += 1;
                if self.current_textures >= MAX_TEXTURES_PER_CHUNK {
                    let chunk = Chunk::Draw {
                        chunk: mem::take(&mut self.current),
                        needs_stencil: self.needs_stencil,
                        transforms: mem::replace(
                            &mut self.transforms,
                            Self::new_transforms(self.descriptors, self.dynamic_transforms),
                        ),
                        vertices: mem::replace(
                            &mut self.vertices,
                            Self::new_vertices(self.descriptors, self.dynamic_transforms),
                        ),
                    };
                    self.emit(chunk);
                }
            }
            blend_type => {
                if !self.current.is_empty() {
                    let chunk = Chunk::Draw {
                        chunk: mem::take(&mut self.current),
                        needs_stencil: self.needs_stencil,
                        transforms: mem::replace(
                            &mut self.transforms,
                            Self::new_transforms(self.descriptors, self.dynamic_transforms),
                        ),
                        vertices: mem::replace(
                            &mut self.vertices,
                            Self::new_vertices(self.descriptors, self.dynamic_transforms),
                        ),
                    };
                    self.emit(chunk);
                }
                let chunk_blend_mode = match blend_type {
                    BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                    BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                    _ => unreachable!(),
                };
                self.emit(Chunk::Blend {
                    texture: target.take_color_texture(),
                    blend_mode: chunk_blend_mode,
                    needs_stencil: self.num_masks > 0,
                    rect,
                });
                self.needs_stencil = self.num_masks > 0;
            }
        }
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        let texture = as_texture(&bitmap);

        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

        let vertices: &[PosUvVertex] = {
            let (u0, u1, v0, v1) = (
                region.x_min as f32 / texture.texture.width() as f32,
                region.x_max as f32 / texture.texture.width() as f32,
                region.y_min as f32 / texture.texture.height() as f32,
                region.y_max as f32 / texture.texture.height() as f32,
            );
            &[
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ]
        };

        self.add_to_current_with_vertices(
            matrix,
            transform.tz,
            transform.color_transform,
            Some(vertices),
            |instance_index, vertex_offset| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset,
                smoothing,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: false,
            },
        );
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(
            matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderBitmap {
                bitmap,
                instance_index,
                vertex_offset: None,
                smoothing: false,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: true,
            },
        );
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        self.add_to_current(
            transform.matrix,
            transform.tz,
            transform.color_transform,
            |instance_index| DrawCommand::RenderShape {
                shape,
                instance_index,
            },
        );
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.add_to_current(
            matrix,
            0.0,
            ColorTransform::multiply_from(color),
            |instance_index| DrawCommand::DrawRect { instance_index },
        );
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLine { instance_index },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                0.0,
                ColorTransform::multiply_from(color),
                |instance_index| DrawCommand::DrawLineRect { instance_index },
            );
        }
    }

    fn push_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        let surface = Surface::new(
            self.descriptors,
            self.quality,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let maskee = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            maskee_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        maskee.ensure_cleared(self.draw_encoder);
        let matrix = Matrix::scale(maskee.width() as f32, maskee.height() as f32);
        let maskee = maskee.take_color_texture();

        let mask = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            mask_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        mask.ensure_cleared(self.draw_encoder);
        let mask = mask.take_color_texture();

        let binds = self
            .descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.descriptors.bind_layouts.alpha_mask,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(maskee.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mask.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });

        self.add_to_current(matrix, 0.0, Default::default(), |instance_index| {
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                instance_index,
            }
        });
    }
}

/// `RUFFLE_SLOW_MULTIPLY` turns the opaque-target Multiply fast path off, for A/B comparisons.
fn fast_multiply_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_SLOW_MULTIPLY").is_none())
}
