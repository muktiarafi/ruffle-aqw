pub(crate) mod commands;
pub mod target;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{AtlasPage, Chunk, CommandRenderer, DrawCommand, draw_chunks};
use crate::surface::target::PoolOrArcTexture;
use crate::utils::run_copy_pipeline;
use crate::utils::supported_sample_count;
use crate::{Descriptors, MaskState, Pipelines};
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;
use wgpu_profiler::Scope;

pub use crate::surface::commands::LayerRef;
pub(crate) use crate::surface::commands::{LayerRect, draw_into_slots, is_leaf};

use self::commands::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    size: wgpu::Extent3d,
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
}

/// AQW build: the soft edges shapes are tessellated with, from the stage's quality. Turu: only
/// the stage's surface sets it (`WgpuRenderBackend`'s own, made at start, on a resize and on a
/// quality change), not every surface: the mode is one for the whole process, and each
/// tessellation cache puts aside what it holds when the mode changes (`TessellationCache`).
/// When an offscreen draw (`BitmapData.draw` at another quality than the stage's, which AQW
/// does every few seconds) set it, the next frame re-tessellated every shape on screen, about a
/// thousand in Battleon, and did so again when the stage's mode came back; on a phone the
/// shapes not ready in time were left out of those frames, so the skill bar and the Game Menu
/// flickered. Offscreen draws now take the stage's edges.
pub fn set_stage_edge_mode(quality: StageQuality) {
    let flash_aa = ruffle_render::tessellator::flash_aa_switch();
    // (Logged when it changes.)
    {
        static LAST: std::sync::Mutex<Option<(bool, StageQuality)>> = std::sync::Mutex::new(None);
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if *last != Some((flash_aa, quality)) {
            *last = Some((flash_aa, quality));
            tracing::info!(
                target: "ruffle_aqw_perf",
                "Edge smoothing: {} ({quality:?} quality)",
                if flash_aa { "Flash-style" } else { "MSAA" }
            );
        }
    }
    ruffle_render::tessellator::set_edge_mode(match quality {
        _ if !flash_aa => ruffle_render::tessellator::EdgeMode::Off,
        StageQuality::Low => ruffle_render::tessellator::EdgeMode::Lines,
        StageQuality::Medium => ruffle_render::tessellator::EdgeMode::Outlines,
        _ => ruffle_render::tessellator::EdgeMode::Soft,
    });
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // AQW build: with Flash-style edge smoothing, shapes get soft edges of their own (on
        // Medium, with fewer triangles, and up; shapes are tessellated again for the new
        // quality), so nothing is drawn with MSAA.
        let flash_aa = ruffle_render::tessellator::flash_aa_switch();
        let sample_count = supported_sample_count(
            &descriptors.adapter,
            if flash_aa {
                1
            } else {
                // `RUFFLE_TEST_SAMPLES`: another MSAA sample count (reference images).
                std::env::var("RUFFLE_TEST_SAMPLES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(quality.sample_count())
            },
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format);
        Self {
            size,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
        }
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'encoder, 'global: 'encoder>(
        &self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef<'encoder>,
        texture_pool: &'global mut TexturePool,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
        );

        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.globals(),
            1,
            draw_encoder,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'encoder, 'global: 'encoder>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'encoder Descriptors,
        meshes: &'encoder Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> CommandTarget {
        let opaque_target = render_target_mode
            .color()
            .is_some_and(|color| color.a >= 1.0);
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.size,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );
        if std::env::var_os("RUFFLE_EAGER_RESOLVE").is_none() {
            target.set_lazy_resolve();
        }

        // AQW build: chunks are drawn as soon as they're complete (see `ChunkSink`), instead of
        // collecting every chunk of this target first and drawing them afterwards.
        let sink = ChunkSink {
            pipelines: &self.pipelines,
            target: &target,
            blend_parent: nearest_layer,
            num_masks: 0,
            mask_state: MaskState::NoMask,
            pending_blend: None,
            page_binds: Vec::new(),
        };
        draw_chunks(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            target.width(),
            target.height(),
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
            opaque_target,
            sink,
        );

        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);
        target.resolve(draw_encoder);

        target
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}

/// AQW build: draws the chunks of one target as they are produced.
///
/// Stock Ruffle first turned a whole command list into chunks (rendering every blended
/// child into its own offscreen texture along the way) and only then drew the chunks. All of
/// those textures were alive at once: crowded AQW frames hold 500+ full-screen textures
/// (7-8 GB of VRAM), allocated in a burst on room entry. Drawing each chunk as soon as it is
/// complete returns its textures to the pool right away, so later blends reuse them.
///
/// The render passes are the same ones, in the same order relative to the textures they read;
/// only *when* each chunk is recorded moves earlier. A complex blend waits for the next chunk
/// so the following draws can still be merged into its pass.
pub(crate) struct ChunkSink<'a> {
    pipelines: &'a Pipelines,
    target: &'a CommandTarget,
    /// The `nearest_layer` this target was drawn with (Alpha/Erase blends composite onto it).
    blend_parent: LayerRef<'a>,
    num_masks: u32,
    mask_state: MaskState,
    /// A complex blend waiting for the next chunk, which may be merged into its pass.
    pending_blend: Option<(PoolOrArcTexture, ComplexBlend, bool, Option<LayerRect>)>,
    /// The bind groups (this target's blend buffer, an atlas page) of the blends read from
    /// pages so far (`DrawCommand::BlendAtlas`): a handful of pages for hundreds of blends.
    page_binds: Vec<(Arc<AtlasPage>, wgpu::BindGroup)>,
}

impl ChunkSink<'_> {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn push(
        &mut self,
        chunk: Chunk,
        descriptors: &Descriptors,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        match chunk {
            Chunk::Draw {
                chunk,
                needs_stencil,
                transforms,
                vertices,
            } => {
                if let Some((texture, blend_mode, blend_needs_stencil, rect)) =
                    self.pending_blend.take()
                {
                    // (A chunk that starts with a blend from a page needs its backdrop copied
                    // before its pass begins: it can't go on in this blend's pass.)
                    if blend_needs_stencil == needs_stencil
                        && pass_merge_enabled()
                        && page_blend_of(&chunk).is_none()
                    {
                        transforms.copy_to(staging_belt, draw_encoder, &dynamic_transforms.buffer);
                        vertices.copy_to(
                            staging_belt,
                            draw_encoder,
                            &dynamic_transforms.vertex_buffer,
                        );
                        crate::perf_counters::inc(crate::perf_counters::PASS_MERGED);
                        self.complex_blend(
                            texture,
                            blend_mode,
                            blend_needs_stencil,
                            rect,
                            Some(chunk),
                            descriptors,
                            dynamic_transforms,
                            draw_encoder,
                            texture_pool,
                        );
                        return;
                    }
                    self.complex_blend(
                        texture,
                        blend_mode,
                        blend_needs_stencil,
                        rect,
                        None,
                        descriptors,
                        dynamic_transforms,
                        draw_encoder,
                        texture_pool,
                    );
                }
                self.draw(
                    chunk,
                    needs_stencil,
                    transforms,
                    vertices,
                    descriptors,
                    staging_belt,
                    dynamic_transforms,
                    draw_encoder,
                    texture_pool,
                );
            }
            Chunk::Blend {
                texture,
                blend_mode,
                needs_stencil,
                rect,
            } => {
                self.flush(descriptors, dynamic_transforms, draw_encoder, texture_pool);
                match blend_mode {
                    ChunkBlendMode::Shader(shader) => self.shader_blend(
                        texture,
                        shader,
                        needs_stencil,
                        descriptors,
                        draw_encoder,
                        texture_pool,
                    ),
                    ChunkBlendMode::Complex(ComplexBlend::Alpha | ComplexBlend::Erase)
                        if matches!(self.blend_parent, LayerRef::None) =>
                    {
                        // An Alpha or Erase with no Layer above it should be ignored
                    }
                    ChunkBlendMode::Complex(blend_mode) => {
                        self.pending_blend = Some((texture, blend_mode, needs_stencil, rect));
                    }
                }
            }
        }
    }

    /// Draws a complex blend still waiting for a following chunk.
    pub(crate) fn flush(
        &mut self,
        descriptors: &Descriptors,
        dynamic_transforms: &DynamicTransforms,
        draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        if let Some((texture, blend_mode, needs_stencil, rect)) = self.pending_blend.take() {
            self.complex_blend(
                texture,
                blend_mode,
                needs_stencil,
                rect,
                None,
                descriptors,
                dynamic_transforms,
                draw_encoder,
                texture_pool,
            );
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
        descriptors: &Descriptors,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        let target = self.target;
        transforms.copy_to(staging_belt, draw_encoder, &dynamic_transforms.buffer);
        vertices.copy_to(
            staging_belt,
            draw_encoder,
            &dynamic_transforms.vertex_buffer,
        );
        // AQW build: a blend read from an atlas page: what is under it goes to the blend
        // buffer now, before the pass that draws it (and what follows it) begins.
        let page_blend_binds = page_blend_of(&chunk).map(|(page, rect)| {
            let backdrop = target.blend_backdrop(descriptors, texture_pool, draw_encoder, rect);
            if chunk.len() > 1 {
                crate::perf_counters::inc(crate::perf_counters::PASS_MERGED);
            }
            match self
                .page_binds
                .iter()
                .find(|(known, _)| Arc::ptr_eq(known, page))
            {
                Some((_, binds)) => binds.clone(),
                None => {
                    let binds = descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: create_debug_label!("Complex blend binds (page)").as_deref(),
                            layout: &descriptors.bind_layouts.blend,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(backdrop.view()),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        page.texture.view(),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        descriptors.bitmap_samplers.get_sampler(false, false),
                                    ),
                                },
                            ],
                        });
                    self.page_binds.push((page.clone(), binds.clone()));
                    binds
                }
            }
        });
        crate::perf_counters::inc(crate::perf_counters::RENDER_PASS);
        let mut render_pass = draw_encoder.scoped_render_pass(
            format!(
                "Chunked draw calls {}",
                if needs_stencil {
                    "(with stencil)"
                } else {
                    "(Stencilless)"
                }
            ),
            wgpu::RenderPassDescriptor {
                color_attachments: &[target.color_attachments()],
                depth_stencil_attachment: if needs_stencil {
                    target.stencil_attachment(descriptors, texture_pool)
                } else {
                    None
                },
                ..Default::default()
            },
        );
        render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
        render_pass.set_bind_group(1, &dynamic_transforms.bind_group, &[]);
        let mut renderer = CommandRenderer::new(
            self.pipelines,
            descriptors,
            &dynamic_transforms.vertex_buffer,
            self.num_masks,
            self.mask_state,
            needs_stencil,
        );
        if let Some(binds) = &page_blend_binds {
            renderer.set_page_blend_binds(binds);
        }

        crate::perf_counters::add_draws(chunk.len());
        for command in &chunk {
            renderer.execute(&mut render_pass.scope(command.name()), command);
        }

        self.num_masks = renderer.num_masks();
        self.mask_state = renderer.mask_state();
    }

    fn shader_blend(
        &mut self,
        texture: PoolOrArcTexture,
        shader: ruffle_render::pixel_bender::PixelBenderShaderHandle,
        needs_stencil: bool,
        descriptors: &Descriptors,
        draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        let target = self.target;
        assert!(!needs_stencil, "Shader blend mode not implemented in masks");
        let parent_blend_buffer =
            target.update_blend_buffer(descriptors, texture_pool, draw_encoder);
        run_pixelbender_shader_impl(
            descriptors,
            shader,
            ShaderMode::Filter,
            &[
                PixelBenderShaderArgument::ImageInput {
                    index: 0,
                    channels: 0xFF,
                    name: "background".to_string(),
                    texture: Some(ImageInputTexture::TextureRef(parent_blend_buffer.texture())),
                },
                PixelBenderShaderArgument::ImageInput {
                    index: 1,
                    channels: 0xff,
                    name: "foreground".to_string(),
                    texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                },
            ],
            parent_blend_buffer.texture(),
            draw_encoder,
            target.color_attachments(),
            target.sample_count(),
            &FilterSource::for_entire_texture(texture.texture()),
        )
        .expect("Failed to run PixelBender blend mode");
    }

    #[expect(clippy::too_many_arguments)]
    fn complex_blend(
        &mut self,
        texture: PoolOrArcTexture,
        blend_mode: ComplexBlend,
        needs_stencil: bool,
        rect: Option<LayerRect>,
        merged_draw: Option<Vec<DrawCommand>>,
        descriptors: &Descriptors,
        dynamic_transforms: &DynamicTransforms,
        draw_encoder: &mut Scope<'_, wgpu::CommandEncoder>,
        texture_pool: &mut TexturePool,
    ) {
        let target = self.target;
        let parent = match blend_mode {
            ComplexBlend::Alpha | ComplexBlend::Erase => match self.blend_parent {
                LayerRef::None => return,
                LayerRef::Current => target,
                LayerRef::Parent(layer) => layer,
            },
            _ => target,
        };

        // AQW build: a bounded layer blends with just its rectangle of the parent (the same
        // size as the layer, so the shaders' coordinates line up), drawn through a viewport.
        let region;
        let parent_view = match rect {
            Some(rect) => {
                region = parent.blend_region(descriptors, texture_pool, draw_encoder, rect);
                &region.1
            }
            None => parent
                .update_blend_buffer(descriptors, texture_pool, draw_encoder)
                .view(),
        };

        let blend_bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: create_debug_label!(
                    "Complex blend binds {:?} {}",
                    blend_mode,
                    if needs_stencil {
                        "(with stencil)"
                    } else {
                        "(Stencilless)"
                    }
                )
                .as_deref(),
                layout: &descriptors.bind_layouts.blend,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(parent_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
            });

        crate::perf_counters::inc(crate::perf_counters::RENDER_PASS);
        let mut render_pass = draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: create_debug_label!(
                "Complex blend {:?} {}",
                blend_mode,
                if needs_stencil {
                    "(with stencil)"
                } else {
                    "(Stencilless)"
                }
            )
            .as_deref(),
            color_attachments: &[target.color_attachments()],
            depth_stencil_attachment: if needs_stencil {
                target.stencil_attachment(descriptors, texture_pool)
            } else {
                None
            },
            ..Default::default()
        });
        render_pass.set_bind_group(0, target.globals().bind_group(), &[]);

        if needs_stencil {
            match self.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    render_pass.set_stencil_reference(self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    render_pass.set_stencil_reference(self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    render_pass.set_stencil_reference(self.num_masks);
                }
            }
            render_pass.set_pipeline(
                self.pipelines.complex_blends[blend_mode].pipeline_for(self.mask_state),
            );
        } else {
            render_pass
                .set_pipeline(self.pipelines.complex_blends[blend_mode].stencilless_pipeline());
        }

        render_pass.set_bind_group(2, &blend_bind_group, &[]);

        render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos.slice(..));
        render_pass.set_index_buffer(
            descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );

        if let Some(rect) = rect {
            render_pass.set_viewport(
                rect.x as f32,
                rect.y as f32,
                rect.width as f32,
                rect.height as f32,
                0.0,
                1.0,
            );
        }
        render_pass.draw_indexed(0..6, 0, 0..1);
        if rect.is_some() {
            render_pass.set_viewport(
                0.0,
                0.0,
                target.width() as f32,
                target.height() as f32,
                0.0,
                1.0,
            );
        }

        if let Some(chunk) = &merged_draw {
            crate::perf_counters::add_draws(chunk.len());
            render_pass.set_bind_group(1, &dynamic_transforms.bind_group, &[]);
            let mut renderer = CommandRenderer::new(
                self.pipelines,
                descriptors,
                &dynamic_transforms.vertex_buffer,
                self.num_masks,
                self.mask_state,
                needs_stencil,
            );
            for command in chunk {
                renderer.execute(&mut render_pass, command);
            }
            self.num_masks = renderer.num_masks();
            self.mask_state = renderer.mask_state();
        }
    }
}

/// AQW build: the blend read from an atlas page that `chunk` starts with (after the scissor
/// of an atlas slot, if it has one): its page and where it goes on the target.
fn page_blend_of(chunk: &[DrawCommand]) -> Option<(&Arc<AtlasPage>, LayerRect)> {
    chunk.iter().take(2).find_map(|command| match command {
        DrawCommand::BlendAtlas { page, rect, .. } => Some((page, *rect)),
        _ => None,
    })
}

/// `RUFFLE_NO_PASS_MERGE` turns off continuing the next draw chunk inside a complex blend's
/// pass, for A/B comparisons.
fn pass_merge_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUFFLE_NO_PASS_MERGE").is_none())
}
