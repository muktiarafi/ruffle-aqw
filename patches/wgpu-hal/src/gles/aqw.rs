//! AQW build (Turu): what the GL replay (`Queue::submit`) costs, and the calls it sends.
//!
//! On D3D12 and Vulkan the cost of each render pass was the driver's. On GL it's this
//! backend's: a command buffer is a list of [`super::Command`]s that `Queue::submit` replays as
//! GL calls on the submitting thread, holding the context lock. Every state-setting call the
//! replay makes goes through [`Replay`], which counts it by [`Kind`].
//!
//! `RUFFLE_GL_STATS=1` logs the counts every 5 s, on `ruffle_aqw_perf` (Android's log filter
//! lets that target through): submits, command buffers, render passes and draws per second,
//! GL calls per second by kind, buffer uploads, the time spent in `submit`, and the context
//! locks per second by thread (each lock makes the EGL context current and releases it again,
//! which flushes it). See docs/GL_RENDERER_PROPOSAL.md in Turu, G1.

use glow::HasContext;

/// What a GL call is for, for the counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// Framebuffer binds and attachments, draw buffers, invalidation, blits.
    Fbo,
    Program,
    /// Texture units, texture binds and their level parameters.
    Texture,
    Sampler,
    /// Uniform, index and copy buffer binds.
    Buffer,
    /// Vertex attributes and vertex buffer bindings.
    Vertex,
    /// Uniform values (the emulated first instance, immediates).
    Uniform,
    /// Fixed-function state: enables, masks, blend, depth, stencil, viewport, scissor.
    State,
    Draw,
    Clear,
    /// Buffer and texture copies and uploads.
    Copy,
    /// Queries, debug markers, barriers, and everything else.
    Other,
}

const KINDS: usize = 12;
const KIND_NAMES: [&str; KINDS] = [
    "fbo", "program", "texture", "sampler", "buffer", "vertex", "uniform", "state", "draw",
    "clear", "copy", "other",
];

/// One submit's counts.
#[derive(Clone, Default)]
pub(super) struct Counts {
    pub calls: [u64; KINDS],
    pub skipped: [u64; KINDS],
    pub command_buffers: u64,
    pub passes: u64,
    pub draws: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        for i in 0..KINDS {
            self.calls[i] += other.calls[i];
            self.skipped[i] += other.skipped[i];
        }
        self.command_buffers += other.command_buffers;
        self.passes += other.passes;
        self.draws += other.draws;
        self.uploads += other.uploads;
        self.upload_bytes += other.upload_bytes;
    }
}

/// Whether `RUFFLE_GL_STATS` is set (asked once).
pub(super) fn stats_enabled() -> bool {
    #[cfg(gles_with_std)]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("RUFFLE_GL_STATS").is_ok_and(|v| !v.is_empty() && v != "0")
        })
    }
    #[cfg(not(gles_with_std))]
    false
}

/// The replay's view of the GL context during one `Queue::submit`: it makes the state-setting
/// calls and counts them.
pub(super) struct Replay {
    pub counts: Counts,
    stats: bool,
}

impl Replay {
    pub fn new() -> Self {
        Self {
            counts: Counts::default(),
            stats: stats_enabled(),
        }
    }

    /// Counts `n` calls of `kind` made directly on the context.
    #[inline]
    pub fn count(&mut self, kind: Kind, n: u64) {
        if self.stats {
            self.counts.calls[kind as usize] += n;
        }
    }

    #[inline]
    pub fn note_command_buffer(&mut self) {
        self.counts.command_buffers += 1;
    }

    #[inline]
    pub fn note_pass(&mut self) {
        self.counts.passes += 1;
    }

    #[inline]
    pub fn note_draw(&mut self) {
        self.counts.draws += 1;
        self.count(Kind::Draw, 1);
    }

    #[inline]
    pub fn note_upload(&mut self, bytes: usize) {
        self.counts.uploads += 1;
        self.counts.upload_bytes += bytes as u64;
    }

    // State-setting calls, one method per GL call.

    pub unsafe fn enable(&mut self, gl: &glow::Context, cap: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.enable(cap) };
    }

    pub unsafe fn disable(&mut self, gl: &glow::Context, cap: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.disable(cap) };
    }

    pub unsafe fn set_enabled(&mut self, gl: &glow::Context, cap: u32, on: bool) {
        if on {
            unsafe { self.enable(gl, cap) }
        } else {
            unsafe { self.disable(gl, cap) }
        }
    }

    pub unsafe fn use_program(&mut self, gl: &glow::Context, program: Option<glow::Program>) {
        self.count(Kind::Program, 1);
        unsafe { gl.use_program(program) };
    }

    /// `glBindFramebuffer(DRAW_FRAMEBUFFER)`.
    pub unsafe fn bind_draw_framebuffer(
        &mut self,
        gl: &glow::Context,
        fbo: Option<glow::Framebuffer>,
    ) {
        self.count(Kind::Fbo, 1);
        unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, fbo) };
    }

    pub unsafe fn draw_buffers(&mut self, gl: &glow::Context, count: u8) {
        self.count(Kind::Fbo, 1);
        let indices = (0..count as u32)
            .map(|i| glow::COLOR_ATTACHMENT0 + i)
            .collect::<arrayvec::ArrayVec<_, { crate::MAX_COLOR_ATTACHMENTS }>>();
        unsafe { gl.draw_buffers(&indices) };
    }

    pub unsafe fn color_mask(&mut self, gl: &glow::Context, mask: [bool; 4]) {
        self.count(Kind::State, 1);
        unsafe { gl.color_mask(mask[0], mask[1], mask[2], mask[3]) };
    }

    pub unsafe fn depth_mask(&mut self, gl: &glow::Context, mask: bool) {
        self.count(Kind::State, 1);
        unsafe { gl.depth_mask(mask) };
    }

    /// `glStencilMaskSeparate` (`face` FRONT, BACK or FRONT_AND_BACK).
    pub unsafe fn stencil_mask(&mut self, gl: &glow::Context, face: u32, mask: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.stencil_mask_separate(face, mask) };
    }

    pub unsafe fn stencil_func(
        &mut self,
        gl: &glow::Context,
        face: u32,
        function: u32,
        reference: i32,
        read_mask: u32,
    ) {
        self.count(Kind::State, 1);
        unsafe { gl.stencil_func_separate(face, function, reference, read_mask) };
    }

    pub unsafe fn stencil_op(&mut self, gl: &glow::Context, face: u32, ops: [u32; 3]) {
        self.count(Kind::State, 1);
        unsafe { gl.stencil_op_separate(face, ops[0], ops[1], ops[2]) };
    }

    pub unsafe fn viewport(&mut self, gl: &glow::Context, rect: [i32; 4]) {
        self.count(Kind::State, 1);
        unsafe { gl.viewport(rect[0], rect[1], rect[2], rect[3]) };
    }

    pub unsafe fn depth_range(&mut self, gl: &glow::Context, near: f32, far: f32) {
        self.count(Kind::State, 1);
        unsafe { gl.depth_range_f32(near, far) };
    }

    pub unsafe fn scissor(&mut self, gl: &glow::Context, rect: [i32; 4]) {
        self.count(Kind::State, 1);
        unsafe { gl.scissor(rect[0], rect[1], rect[2], rect[3]) };
    }

    pub unsafe fn depth_func(&mut self, gl: &glow::Context, function: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.depth_func(function) };
    }

    pub unsafe fn polygon_offset(&mut self, gl: &glow::Context, factor: f32, units: f32) {
        self.count(Kind::State, 1);
        unsafe { gl.polygon_offset(factor, units) };
    }

    pub unsafe fn front_face(&mut self, gl: &glow::Context, face: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.front_face(face) };
    }

    pub unsafe fn cull_face(&mut self, gl: &glow::Context, face: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.cull_face(face) };
    }

    pub unsafe fn blend_color(&mut self, gl: &glow::Context, c: [f32; 4]) {
        self.count(Kind::State, 1);
        unsafe { gl.blend_color(c[0], c[1], c[2], c[3]) };
    }

    /// `glBlendEquationSeparate` (`glBlendEquation(e)` is `(e, e)`).
    pub unsafe fn blend_equation(&mut self, gl: &glow::Context, color: u32, alpha: u32) {
        self.count(Kind::State, 1);
        unsafe { gl.blend_equation_separate(color, alpha) };
    }

    /// `glBlendFuncSeparate` (`glBlendFunc(s, d)` is `(s, d, s, d)`).
    pub unsafe fn blend_func(&mut self, gl: &glow::Context, f: [u32; 4]) {
        self.count(Kind::State, 1);
        unsafe { gl.blend_func_separate(f[0], f[1], f[2], f[3]) };
    }

    pub unsafe fn active_texture(&mut self, gl: &glow::Context, unit: u32) {
        self.count(Kind::Texture, 1);
        unsafe { gl.active_texture(glow::TEXTURE0 + unit) };
    }

    /// `glBindTexture` on the active unit.
    pub unsafe fn bind_texture(&mut self, gl: &glow::Context, target: u32, texture: glow::Texture) {
        self.count(Kind::Texture, 1);
        unsafe { gl.bind_texture(target, Some(texture)) };
    }

    /// `TEXTURE_BASE_LEVEL` and `TEXTURE_MAX_LEVEL` of the texture bound to `target`.
    pub unsafe fn texture_levels(
        &mut self,
        gl: &glow::Context,
        target: u32,
        _texture: glow::Texture,
        base: i32,
        max: i32,
    ) {
        self.count(Kind::Texture, 2);
        unsafe { gl.tex_parameter_i32(target, glow::TEXTURE_BASE_LEVEL, base) };
        unsafe { gl.tex_parameter_i32(target, glow::TEXTURE_MAX_LEVEL, max) };
    }

    pub unsafe fn bind_sampler(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        sampler: Option<glow::Sampler>,
    ) {
        self.count(Kind::Sampler, 1);
        unsafe { gl.bind_sampler(unit, sampler) };
    }

    pub unsafe fn bind_buffer_range(
        &mut self,
        gl: &glow::Context,
        target: u32,
        slot: u32,
        buffer: glow::Buffer,
        offset: i32,
        size: i32,
    ) {
        self.count(Kind::Buffer, 1);
        unsafe { gl.bind_buffer_range(target, slot, Some(buffer), offset, size) };
    }

    /// `glBindBuffer(ARRAY_BUFFER)`.
    pub unsafe fn bind_array_buffer(&mut self, gl: &glow::Context, buffer: Option<glow::Buffer>) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, buffer) };
    }

    pub unsafe fn enable_attrib(&mut self, gl: &glow::Context, location: u32) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.enable_vertex_attrib_array(location) };
    }

    pub unsafe fn disable_attrib(&mut self, gl: &glow::Context, location: u32) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.disable_vertex_attrib_array(location) };
    }

    pub unsafe fn attrib_divisor(&mut self, gl: &glow::Context, location: u32, divisor: u32) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.vertex_attrib_divisor(location, divisor) };
    }

    pub unsafe fn binding_divisor(&mut self, gl: &glow::Context, index: u32, divisor: u32) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.vertex_binding_divisor(index, divisor) };
    }

    pub unsafe fn bind_vertex_buffer(
        &mut self,
        gl: &glow::Context,
        index: u32,
        buffer: glow::Buffer,
        offset: i32,
        stride: i32,
    ) {
        self.count(Kind::Vertex, 1);
        unsafe { gl.bind_vertex_buffer(index, Some(buffer), offset, stride) };
    }

    /// The emulated first instance (a uniform of the current program), on a driver without
    /// base-instance draws.
    pub unsafe fn first_instance(
        &mut self,
        gl: &glow::Context,
        location: Option<&glow::UniformLocation>,
        value: u32,
    ) {
        self.count(Kind::Uniform, 1);
        unsafe { gl.uniform_1_u32(location, value) };
    }

    /// `glBindFramebuffer(FRAMEBUFFER, None)`: both the draw and the read binding.
    pub unsafe fn unbind_framebuffers(&mut self, gl: &glow::Context) {
        self.count(Kind::Fbo, 1);
        unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
    }

    /// Detaches everything from the draw framebuffer (bound to DRAW_FRAMEBUFFER) at the start
    /// of a render pass: depth-stencil and every color slot.
    pub unsafe fn detach_all(&mut self, gl: &glow::Context, max_color_attachments: u32) {
        self.count(Kind::Fbo, 1 + max_color_attachments as u64);
        unsafe {
            gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                glow::DEPTH_STENCIL_ATTACHMENT,
                glow::TEXTURE_2D,
                None,
                0,
            )
        };
        for i in 0..max_color_attachments {
            unsafe {
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0 + i,
                    glow::TEXTURE_2D,
                    None,
                    0,
                )
            };
        }
    }

    /// A render pass's attachment: whether the caller should attach `view` to `attachment` of
    /// the draw framebuffer.
    pub unsafe fn attach(
        &mut self,
        _gl: &glow::Context,
        _attachment: u32,
        _view: &super::TextureView,
        _depth_slice: Option<u32>,
        _sample_count: u32,
    ) -> bool {
        self.count(Kind::Fbo, 1);
        true
    }

    /// Counts a command whose calls the replay makes without `self`, and forgets what it
    /// knows of the state it may change. The commands `self` sees call by call are left alone.
    pub fn unmodeled(&mut self, command: &super::Command, shader_clear: bool) {
        use super::Command as C;
        let (kind, n, changes_state) = match *command {
            C::Draw { .. }
            | C::DrawIndexed { .. }
            | C::DrawIndirect { .. }
            | C::DrawIndexedIndirect { .. }
            | C::ResetFramebuffer { .. }
            | C::BindAttachment { .. }
            | C::SetDrawColorBuffers(..)
            | C::SetViewport { .. }
            | C::SetScissor(..)
            | C::SetStencilFunc { .. }
            | C::SetStencilOps { .. }
            | C::SetVertexAttribute { .. }
            | C::UnsetVertexAttribute(..)
            | C::SetVertexBuffer { .. }
            | C::SetDepth(..)
            | C::SetDepthBias(..)
            | C::ConfigureDepthStencil(..)
            | C::SetAlphaToCoverage(..)
            | C::SetProgram(..)
            | C::SetPrimitive(..)
            | C::SetBlendConstant(..)
            | C::SetColorTarget { .. }
            | C::BindBuffer { .. }
            | C::BindSampler(..)
            | C::BindTexture { .. } => return,
            // (The sRGB clear workaround draws with a program of its own.)
            C::ClearColorF { .. } => (Kind::Clear, 1, shader_clear),
            C::ClearColorU(..) | C::ClearColorI(..) => (Kind::Clear, 1, false),
            C::ClearDepth(..) | C::ClearStencil(..) => (Kind::Clear, 2, false),
            C::ClearDepthAndStencil(..) => (Kind::Clear, 3, false),
            C::InvalidateAttachments(..) => (Kind::Fbo, 1, false),
            // (The index buffer has a cache of its own, `Queue::current_index_buffer`.)
            C::SetIndexBuffer(..) => (Kind::Buffer, 1, false),
            // (Uniforms of the current program, which `self` doesn't track.)
            C::SetImmediates { .. } => (Kind::Uniform, 1, false),
            C::InsertDebugMarker(..)
            | C::PushDebugGroup(..)
            | C::PopDebugGroup
            | C::BeginQuery(..)
            | C::EndQuery(..)
            | C::TimestampQuery(..) => (Kind::Other, 1, false),
            C::CopyBufferToBuffer { .. }
            | C::CopyBufferToTexture { .. }
            | C::CopyTextureToBuffer { .. }
            | C::CopyTextureToTexture { .. }
            | C::ClearBuffer { .. } => (Kind::Copy, 3, true),
            C::ResolveAttachment { .. } => (Kind::Fbo, 6, true),
            _ => (Kind::Other, 1, true),
        };
        self.count(kind, n);
        if changes_state {
            self.forget();
        }
    }

    /// Forgets what's known of the context's state (G2's shadow).
    #[inline]
    pub fn forget(&mut self) {}
}

#[cfg(gles_with_std)]
mod totals {
    use super::{Counts, KINDS, KIND_NAMES};
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::fmt::Write as _;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    const PERIOD: Duration = Duration::from_secs(5);

    struct Totals {
        since: Option<Instant>,
        counts: Counts,
        submits: u64,
        submit_time: Duration,
        /// Context locks by thread name.
        locks: Vec<(String, u64)>,
    }

    static TOTALS: Mutex<Totals> = Mutex::new(Totals {
        since: None,
        counts: Counts {
            calls: [0; KINDS],
            skipped: [0; KINDS],
            command_buffers: 0,
            passes: 0,
            draws: 0,
            uploads: 0,
            upload_bytes: 0,
        },
        submits: 0,
        submit_time: Duration::ZERO,
        locks: Vec::new(),
    });

    pub fn note_lock() {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("unnamed");
        let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        match totals.locks.iter_mut().find(|(n, _)| n == name) {
            Some((_, count)) => *count += 1,
            None => totals.locks.push((name.into(), 1)),
        }
    }

    pub fn submitted(counts: &Counts, time: Duration) {
        let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let since = *totals.since.get_or_insert(now);
        totals.counts.add(counts);
        totals.submits += 1;
        totals.submit_time += time;
        let elapsed = now - since;
        if elapsed < PERIOD {
            return;
        }
        let s = elapsed.as_secs_f64();
        let per_s = |n: u64| n as f64 / s;
        let c = &totals.counts;
        let mut line = String::new();
        let _ = write!(
            line,
            "GL replay ({s:.1} s): {:.1} submits/s, {:.0} command buffers/s, {:.0} passes/s, {:.0} draws/s; submit {:.1} ms/s; GL calls {:.0}/s (",
            per_s(totals.submits),
            per_s(c.command_buffers),
            per_s(c.passes),
            per_s(c.draws),
            totals.submit_time.as_secs_f64() * 1000.0 / s,
            per_s(c.calls.iter().sum()),
        );
        for (i, name) in KIND_NAMES.iter().enumerate() {
            let _ = write!(
                line,
                "{}{name} {:.0}",
                if i == 0 { "" } else { ", " },
                per_s(c.calls[i])
            );
        }
        let _ = write!(line, "), skipped {:.0}/s (", per_s(c.skipped.iter().sum()));
        for (i, name) in KIND_NAMES.iter().enumerate() {
            let _ = write!(
                line,
                "{}{name} {:.0}",
                if i == 0 { "" } else { ", " },
                per_s(c.skipped[i])
            );
        }
        let _ = write!(
            line,
            "); uploads {:.0}/s, {:.2} MB/s; context locks {:.0}/s (",
            per_s(c.uploads),
            per_s(c.upload_bytes) / (1024.0 * 1024.0),
            per_s(totals.locks.iter().map(|(_, n)| n).sum()),
        );
        totals.locks.sort_by(|a, b| b.1.cmp(&a.1));
        for (i, (name, n)) in totals.locks.iter().enumerate() {
            let _ = write!(
                line,
                "{}{name} {:.0}",
                if i == 0 { "" } else { ", " },
                per_s(*n)
            );
        }
        line.push(')');
        log::info!(target: "ruffle_aqw_perf", "{line}");
        totals.since = Some(now);
        totals.counts = Counts::default();
        totals.submits = 0;
        totals.submit_time = Duration::ZERO;
        totals.locks.clear();
    }
}

/// Counts a context lock on this thread (`RUFFLE_GL_STATS` only).
#[inline]
pub(super) fn note_lock() {
    #[cfg(gles_with_std)]
    if stats_enabled() {
        totals::note_lock();
    }
}

/// Adds a finished submit's counts, and logs the totals every 5 s (`RUFFLE_GL_STATS` only).
#[cfg(gles_with_std)]
pub(super) fn submitted(counts: &Counts, time: std::time::Duration) {
    totals::submitted(counts, time);
}
