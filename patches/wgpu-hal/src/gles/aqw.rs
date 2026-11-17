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

use alloc::vec::Vec;
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

/// The levers of G2 (`RUFFLE_GL_LEVERS`, asked once): a comma-separated list of `fbo` (G2a:
/// detach only stale attachments) and `state` (G2b: skip calls that set what's already set).
/// Unset: both. `none` (or empty): neither, the replay as upstream wgpu makes it.
#[derive(Clone, Copy)]
pub(super) struct Levers {
    pub fbo: bool,
    pub state: bool,
}

pub(super) fn levers() -> Levers {
    #[cfg(gles_with_std)]
    {
        static LEVERS: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
        let &(fbo, state) = LEVERS.get_or_init(|| match std::env::var("RUFFLE_GL_LEVERS") {
            Err(_) => (true, true),
            Ok(list) => {
                let has = |name: &str| list.split(',').any(|l| l.trim() == name);
                (has("fbo"), has("state"))
            }
        });
        Levers { fbo, state }
    }
    #[cfg(not(gles_with_std))]
    Levers {
        fbo: true,
        state: true,
    }
}

/// The capabilities `Replay` keeps track of (`glEnable`/`glDisable`).
const CAPS: [u32; 8] = [
    glow::DEPTH_TEST,
    glow::STENCIL_TEST,
    glow::SCISSOR_TEST,
    glow::BLEND,
    glow::CULL_FACE,
    glow::POLYGON_OFFSET_FILL,
    glow::SAMPLE_ALPHA_TO_COVERAGE,
    glow::DEPTH_CLAMP,
];

/// What `Replay` knows of the context's state: `None` where it doesn't know. All of it is
/// forgotten whenever the replay runs a command it doesn't model (a copy, a resolve, ...).
#[derive(Default)]
struct Shadow {
    program: Option<Option<glow::Program>>,
    /// The emulated first instance of the current program.
    first_instance: Option<u32>,
    draw_fbo: Option<Option<glow::Framebuffer>>,
    read_fbo: Option<Option<glow::Framebuffer>>,
    caps: [Option<bool>; CAPS.len()],
    draw_buffers: Option<u8>,
    color_mask: Option<[bool; 4]>,
    depth_mask: Option<bool>,
    /// Front, back.
    stencil_mask: [Option<u32>; 2],
    stencil_func: [Option<(u32, i32, u32)>; 2],
    stencil_op: [Option<[u32; 3]>; 2],
    viewport: Option<[i32; 4]>,
    depth_range: Option<[u32; 2]>,
    scissor: Option<[i32; 4]>,
    depth_func: Option<u32>,
    polygon_offset: Option<[u32; 2]>,
    front_face: Option<u32>,
    cull_face: Option<u32>,
    blend_color: Option<[u32; 4]>,
    blend_equation: Option<(u32, u32)>,
    blend_func: Option<[u32; 4]>,
    active_texture: Option<u32>,
    /// Per texture unit: the texture bound and its target (the unit's other targets unknown).
    textures: Vec<Option<(u32, glow::Texture)>>,
    samplers: Vec<Option<Option<glow::Sampler>>>,
    buffer_ranges: Vec<((u32, u32), (glow::Buffer, i32, i32))>,
    array_buffer: Option<Option<glow::Buffer>>,
    attrib_enabled: Vec<Option<bool>>,
    attrib_divisor: Vec<Option<u32>>,
    binding_divisor: Vec<Option<u32>>,
    vertex_buffers: Vec<Option<(glow::Buffer, i32, i32)>>,
}

/// Whether `slot` already holds `value` (and `skip` allows skipping); otherwise records it.
#[inline]
fn same<T: PartialEq>(skip: bool, slot: &mut Option<T>, value: T) -> bool {
    if skip && slot.as_ref() == Some(&value) {
        return true;
    }
    *slot = Some(value);
    false
}

/// `v[i]`, grown as needed.
#[inline]
fn at<T: Default>(v: &mut Vec<T>, i: u32) -> &mut T {
    let i = i as usize;
    if v.len() <= i {
        v.resize_with(i + 1, T::default);
    }
    &mut v[i]
}

/// The stencil faces `face` names (front 0, back 1).
fn faces(face: u32) -> &'static [usize] {
    match face {
        glow::FRONT => &[0],
        glow::BACK => &[1],
        _ => &[0, 1],
    }
}

/// What's attached at one point of the draw framebuffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Attached {
    Texture {
        raw: glow::Texture,
        target: u32,
        level: u32,
        layer: u32,
        samples: u32,
    },
    Renderbuffer(glow::Renderbuffer),
}

/// Attachment points of the draw framebuffer: colors 0..32, then depth and stencil.
const POINTS: usize = 34;
const DEPTH: usize = 32;
const STENCIL: usize = 33;

fn points(attachment: u32) -> &'static [usize] {
    const COLORS: [usize; 32] = {
        let mut c = [0; 32];
        let mut i = 0;
        while i < 32 {
            c[i] = i;
            i += 1;
        }
        c
    };
    match attachment {
        glow::DEPTH_STENCIL_ATTACHMENT => &[DEPTH, STENCIL],
        glow::DEPTH_ATTACHMENT => &[DEPTH],
        glow::STENCIL_ATTACHMENT => &[STENCIL],
        a => {
            let i = (a - glow::COLOR_ATTACHMENT0) as usize;
            &COLORS[i..i + 1]
        }
    }
}

fn point_attachment(point: usize) -> u32 {
    match point {
        DEPTH => glow::DEPTH_ATTACHMENT,
        STENCIL => glow::STENCIL_ATTACHMENT,
        i => glow::COLOR_ATTACHMENT0 + i as u32,
    }
}

/// G2a: the queue's draw framebuffer's attachments, which only the replay changes. Known for
/// the rest of a submit once the first pass has detached everything (a texture deleted between
/// submits could otherwise come back under the same name).
struct DrawFramebuffer {
    /// Per point: `None` unknown, `Some(None)` nothing attached.
    attached: [Option<Option<Attached>>; POINTS],
    /// The points the pass being set up attaches, until its first command after them.
    wanted: Option<[bool; POINTS]>,
}

/// The replay's view of the GL context during one `Queue::submit`: it makes the state-setting
/// calls, counts them, and with the levers on (`RUFFLE_GL_LEVERS`) skips those that set what
/// is already set. Within one submit nothing else uses the context (the submit holds its
/// lock), and each submit starts knowing nothing, so what it knows can't go stale: only the
/// commands it doesn't model change state behind it, and it forgets everything at those.
pub(super) struct Replay {
    pub counts: Counts,
    stats: bool,
    levers: Levers,
    s: Shadow,
    fb: DrawFramebuffer,
    /// Texture object state, which only `BindTexture` changes in a submit: base and max level.
    texture_levels: Vec<(glow::Texture, i32, i32)>,
}

impl Replay {
    pub fn new() -> Self {
        Self {
            counts: Counts::default(),
            stats: stats_enabled(),
            levers: levers(),
            s: Shadow::default(),
            fb: DrawFramebuffer {
                attached: [None; POINTS],
                wanted: None,
            },
            texture_levels: Vec::new(),
        }
    }

    /// Counts `n` calls of `kind` made directly on the context.
    #[inline]
    pub fn count(&mut self, kind: Kind, n: u64) {
        if self.stats {
            self.counts.calls[kind as usize] += n;
        }
    }

    /// Counts a call skipped, or made if `skipped` is false; returns `skipped`.
    #[inline]
    fn skip(&mut self, kind: Kind, skipped: bool) -> bool {
        if self.stats {
            if skipped {
                self.counts.skipped[kind as usize] += 1;
            } else {
                self.counts.calls[kind as usize] += 1;
            }
        }
        skipped
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

    /// Forgets what's known of the context's state (not the draw framebuffer's attachments,
    /// nor the textures' levels, which only the replay's own commands change).
    #[inline]
    pub fn forget(&mut self) {
        self.s = Shadow::default();
    }

    // State-setting calls, one method per GL call.

    pub unsafe fn set_enabled(&mut self, gl: &glow::Context, cap: u32, on: bool) {
        let skip = self.levers.state;
        let skipped = match CAPS.iter().position(|&c| c == cap) {
            Some(i) => same(skip, &mut self.s.caps[i], on),
            None => false,
        };
        if !self.skip(Kind::State, skipped) {
            if on {
                unsafe { gl.enable(cap) };
            } else {
                unsafe { gl.disable(cap) };
            }
        }
    }

    pub unsafe fn enable(&mut self, gl: &glow::Context, cap: u32) {
        unsafe { self.set_enabled(gl, cap, true) }
    }

    pub unsafe fn disable(&mut self, gl: &glow::Context, cap: u32) {
        unsafe { self.set_enabled(gl, cap, false) }
    }

    pub unsafe fn use_program(&mut self, gl: &glow::Context, program: Option<glow::Program>) {
        let skipped = same(self.levers.state, &mut self.s.program, program);
        if !self.skip(Kind::Program, skipped) {
            // (A uniform's value belongs to its program.)
            self.s.first_instance = None;
            unsafe { gl.use_program(program) };
        }
    }

    /// `glBindFramebuffer(FRAMEBUFFER, None)`: both the draw and the read binding.
    pub unsafe fn unbind_framebuffers(&mut self, gl: &glow::Context) {
        let skip = self.levers.state;
        let skipped = self.s.draw_fbo == Some(None) && self.s.read_fbo == Some(None) && skip;
        if !self.skip(Kind::Fbo, skipped) {
            self.s.draw_fbo = Some(None);
            self.s.read_fbo = Some(None);
            unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
        }
    }

    /// `glBindFramebuffer(DRAW_FRAMEBUFFER)`.
    pub unsafe fn bind_draw_framebuffer(
        &mut self,
        gl: &glow::Context,
        fbo: Option<glow::Framebuffer>,
    ) {
        // (The default framebuffer's pass sets nothing up.)
        if fbo.is_none() {
            self.fb.wanted = None;
        }
        let skipped = same(self.levers.state, &mut self.s.draw_fbo, fbo);
        if !self.skip(Kind::Fbo, skipped) {
            unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, fbo) };
        }
    }

    /// The start of a render pass on the draw framebuffer (bound to DRAW_FRAMEBUFFER).
    /// Upstream detaches depth-stencil and every color slot here. With G2a, only what the pass
    /// doesn't attach again is detached, once its attachments are known (`finish_setup`).
    pub unsafe fn detach_all(&mut self, gl: &glow::Context, max_color_attachments: u32) {
        let known = self.fb.attached[DEPTH].is_some()
            && self.fb.attached[STENCIL].is_some()
            && self.fb.attached[..max_color_attachments as usize]
                .iter()
                .all(Option::is_some);
        if self.levers.fbo && known {
            self.fb.wanted = Some([false; POINTS]);
            return;
        }
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
        self.fb.attached = [None; POINTS];
        self.fb.attached[DEPTH] = Some(None);
        self.fb.attached[STENCIL] = Some(None);
        for point in &mut self.fb.attached[..max_color_attachments as usize] {
            *point = Some(None);
        }
        // (Points past `max_color_attachments` are never attached.)
        for point in &mut self.fb.attached[max_color_attachments as usize..DEPTH] {
            *point = Some(None);
        }
        self.fb.wanted = self.levers.fbo.then_some([false; POINTS]);
    }

    /// A render pass's attachment: whether the caller should attach `view` to `attachment` of
    /// the draw framebuffer (no when it's attached there already).
    pub unsafe fn attach(
        &mut self,
        _gl: &glow::Context,
        attachment: u32,
        view: &super::TextureView,
        depth_slice: Option<u32>,
        sample_count: u32,
    ) -> bool {
        let attached = match view.inner {
            super::TextureInner::Renderbuffer { raw } => Some(Attached::Renderbuffer(raw)),
            super::TextureInner::Texture { raw, target } => {
                let layers = view.array_layers.end - view.array_layers.start;
                (layers <= 1).then(|| Attached::Texture {
                    raw,
                    target,
                    level: view.mip_levels.start,
                    layer: if target == glow::TEXTURE_3D {
                        depth_slice.unwrap_or(0)
                    } else {
                        view.array_layers.start
                    },
                    samples: sample_count,
                })
            }
            #[allow(unreachable_patterns)]
            _ => None,
        };
        if !self.levers.fbo {
            self.count(Kind::Fbo, 1);
            return true;
        }
        let (Some(attached), Some(wanted)) = (attached, self.fb.wanted.as_mut()) else {
            // (Something it doesn't model: an external framebuffer binds a framebuffer of
            // its own.)
            self.count(Kind::Fbo, 1);
            self.fb.attached = [None; POINTS];
            self.fb.wanted = None;
            self.forget();
            return true;
        };
        let points = points(attachment);
        for &p in points {
            wanted[p] = true;
        }
        let skipped = points
            .iter()
            .all(|&p| self.fb.attached[p] == Some(Some(attached)));
        for &p in points {
            self.fb.attached[p] = Some(Some(attached));
        }
        !self.skip(Kind::Fbo, skipped)
    }

    /// After a pass's attachments (before its first other command): detaches what an earlier
    /// pass attached and this one doesn't.
    unsafe fn finish_setup(&mut self, gl: &glow::Context) {
        let Some(wanted) = self.fb.wanted.take() else {
            return;
        };
        for point in 0..POINTS {
            if wanted[point] || self.fb.attached[point] == Some(None) {
                continue;
            }
            self.count(Kind::Fbo, 1);
            unsafe {
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    point_attachment(point),
                    glow::TEXTURE_2D,
                    None,
                    0,
                )
            };
            self.fb.attached[point] = Some(None);
        }
    }

    /// At the end of a command buffer.
    pub unsafe fn end_command_buffer(&mut self, gl: &glow::Context) {
        unsafe { self.finish_setup(gl) };
    }

    pub unsafe fn draw_buffers(&mut self, gl: &glow::Context, count: u8) {
        let skipped = same(self.levers.state, &mut self.s.draw_buffers, count);
        if !self.skip(Kind::Fbo, skipped) {
            let indices = (0..count as u32)
                .map(|i| glow::COLOR_ATTACHMENT0 + i)
                .collect::<arrayvec::ArrayVec<_, { crate::MAX_COLOR_ATTACHMENTS }>>();
            unsafe { gl.draw_buffers(&indices) };
        }
    }

    pub unsafe fn color_mask(&mut self, gl: &glow::Context, mask: [bool; 4]) {
        let skipped = same(self.levers.state, &mut self.s.color_mask, mask);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.color_mask(mask[0], mask[1], mask[2], mask[3]) };
        }
    }

    pub unsafe fn depth_mask(&mut self, gl: &glow::Context, mask: bool) {
        let skipped = same(self.levers.state, &mut self.s.depth_mask, mask);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.depth_mask(mask) };
        }
    }

    /// Whether every face of `face` already has `value` in `slots`; records it otherwise.
    fn same_faces<T: PartialEq + Copy>(
        skip: bool,
        slots: &mut [Option<T>; 2],
        face: u32,
        value: T,
    ) -> bool {
        let faces = faces(face);
        let all = skip && faces.iter().all(|&f| slots[f] == Some(value));
        if !all {
            for &f in faces {
                slots[f] = Some(value);
            }
        }
        all
    }

    /// `glStencilMaskSeparate` (`face` FRONT, BACK or FRONT_AND_BACK).
    pub unsafe fn stencil_mask(&mut self, gl: &glow::Context, face: u32, mask: u32) {
        let skipped = Self::same_faces(self.levers.state, &mut self.s.stencil_mask, face, mask);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.stencil_mask_separate(face, mask) };
        }
    }

    pub unsafe fn stencil_func(
        &mut self,
        gl: &glow::Context,
        face: u32,
        function: u32,
        reference: i32,
        read_mask: u32,
    ) {
        let value = (function, reference, read_mask);
        let skipped = Self::same_faces(self.levers.state, &mut self.s.stencil_func, face, value);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.stencil_func_separate(face, function, reference, read_mask) };
        }
    }

    pub unsafe fn stencil_op(&mut self, gl: &glow::Context, face: u32, ops: [u32; 3]) {
        let skipped = Self::same_faces(self.levers.state, &mut self.s.stencil_op, face, ops);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.stencil_op_separate(face, ops[0], ops[1], ops[2]) };
        }
    }

    pub unsafe fn viewport(&mut self, gl: &glow::Context, rect: [i32; 4]) {
        let skipped = same(self.levers.state, &mut self.s.viewport, rect);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.viewport(rect[0], rect[1], rect[2], rect[3]) };
        }
    }

    pub unsafe fn depth_range(&mut self, gl: &glow::Context, near: f32, far: f32) {
        let value = [near.to_bits(), far.to_bits()];
        let skipped = same(self.levers.state, &mut self.s.depth_range, value);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.depth_range_f32(near, far) };
        }
    }

    pub unsafe fn scissor(&mut self, gl: &glow::Context, rect: [i32; 4]) {
        let skipped = same(self.levers.state, &mut self.s.scissor, rect);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.scissor(rect[0], rect[1], rect[2], rect[3]) };
        }
    }

    pub unsafe fn depth_func(&mut self, gl: &glow::Context, function: u32) {
        let skipped = same(self.levers.state, &mut self.s.depth_func, function);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.depth_func(function) };
        }
    }

    pub unsafe fn polygon_offset(&mut self, gl: &glow::Context, factor: f32, units: f32) {
        let value = [factor.to_bits(), units.to_bits()];
        let skipped = same(self.levers.state, &mut self.s.polygon_offset, value);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.polygon_offset(factor, units) };
        }
    }

    pub unsafe fn front_face(&mut self, gl: &glow::Context, face: u32) {
        let skipped = same(self.levers.state, &mut self.s.front_face, face);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.front_face(face) };
        }
    }

    pub unsafe fn cull_face(&mut self, gl: &glow::Context, face: u32) {
        let skipped = same(self.levers.state, &mut self.s.cull_face, face);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.cull_face(face) };
        }
    }

    pub unsafe fn blend_color(&mut self, gl: &glow::Context, c: [f32; 4]) {
        let value = c.map(f32::to_bits);
        let skipped = same(self.levers.state, &mut self.s.blend_color, value);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.blend_color(c[0], c[1], c[2], c[3]) };
        }
    }

    /// `glBlendEquationSeparate` (`glBlendEquation(e)` is `(e, e)`).
    pub unsafe fn blend_equation(&mut self, gl: &glow::Context, color: u32, alpha: u32) {
        let skipped = same(
            self.levers.state,
            &mut self.s.blend_equation,
            (color, alpha),
        );
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.blend_equation_separate(color, alpha) };
        }
    }

    /// `glBlendFuncSeparate` (`glBlendFunc(s, d)` is `(s, d, s, d)`).
    pub unsafe fn blend_func(&mut self, gl: &glow::Context, f: [u32; 4]) {
        let skipped = same(self.levers.state, &mut self.s.blend_func, f);
        if !self.skip(Kind::State, skipped) {
            unsafe { gl.blend_func_separate(f[0], f[1], f[2], f[3]) };
        }
    }

    pub unsafe fn active_texture(&mut self, gl: &glow::Context, unit: u32) {
        let skipped = same(self.levers.state, &mut self.s.active_texture, unit);
        if !self.skip(Kind::Texture, skipped) {
            unsafe { gl.active_texture(glow::TEXTURE0 + unit) };
        }
    }

    /// `glBindTexture` on the active unit.
    pub unsafe fn bind_texture(&mut self, gl: &glow::Context, target: u32, texture: glow::Texture) {
        let skipped = match self.s.active_texture {
            Some(unit) => same(
                self.levers.state,
                at(&mut self.s.textures, unit),
                (target, texture),
            ),
            None => false,
        };
        if !self.skip(Kind::Texture, skipped) {
            unsafe { gl.bind_texture(target, Some(texture)) };
        }
    }

    /// `TEXTURE_BASE_LEVEL` and `TEXTURE_MAX_LEVEL` of `texture`, bound to `target`.
    pub unsafe fn texture_levels(
        &mut self,
        gl: &glow::Context,
        target: u32,
        texture: glow::Texture,
        base: i32,
        max: i32,
    ) {
        let known = self
            .texture_levels
            .iter_mut()
            .find(|(t, _, _)| *t == texture);
        let skipped =
            self.levers.state && known.as_ref().is_some_and(|k| (k.1, k.2) == (base, max));
        match known {
            Some(k) => (k.1, k.2) = (base, max),
            None => self.texture_levels.push((texture, base, max)),
        }
        if !self.skip(Kind::Texture, skipped) {
            self.count(Kind::Texture, 1);
            unsafe { gl.tex_parameter_i32(target, glow::TEXTURE_BASE_LEVEL, base) };
            unsafe { gl.tex_parameter_i32(target, glow::TEXTURE_MAX_LEVEL, max) };
        }
    }

    pub unsafe fn bind_sampler(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        sampler: Option<glow::Sampler>,
    ) {
        let skipped = same(self.levers.state, at(&mut self.s.samplers, unit), sampler);
        if !self.skip(Kind::Sampler, skipped) {
            unsafe { gl.bind_sampler(unit, sampler) };
        }
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
        let key = (target, slot);
        let value = (buffer, offset, size);
        let skipped = match self.s.buffer_ranges.iter_mut().find(|(k, _)| *k == key) {
            Some((_, known)) => {
                let same = self.levers.state && *known == value;
                *known = value;
                same
            }
            None => {
                self.s.buffer_ranges.push((key, value));
                false
            }
        };
        if !self.skip(Kind::Buffer, skipped) {
            unsafe { gl.bind_buffer_range(target, slot, Some(buffer), offset, size) };
        }
    }

    /// `glBindBuffer(ARRAY_BUFFER)`.
    pub unsafe fn bind_array_buffer(&mut self, gl: &glow::Context, buffer: Option<glow::Buffer>) {
        let skipped = same(self.levers.state, &mut self.s.array_buffer, buffer);
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, buffer) };
        }
    }

    pub unsafe fn enable_attrib(&mut self, gl: &glow::Context, location: u32) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.attrib_enabled, location),
            true,
        );
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.enable_vertex_attrib_array(location) };
        }
    }

    pub unsafe fn disable_attrib(&mut self, gl: &glow::Context, location: u32) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.attrib_enabled, location),
            false,
        );
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.disable_vertex_attrib_array(location) };
        }
    }

    pub unsafe fn attrib_divisor(&mut self, gl: &glow::Context, location: u32, divisor: u32) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.attrib_divisor, location),
            divisor,
        );
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.vertex_attrib_divisor(location, divisor) };
        }
    }

    pub unsafe fn binding_divisor(&mut self, gl: &glow::Context, index: u32, divisor: u32) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.binding_divisor, index),
            divisor,
        );
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.vertex_binding_divisor(index, divisor) };
        }
    }

    pub unsafe fn bind_vertex_buffer(
        &mut self,
        gl: &glow::Context,
        index: u32,
        buffer: glow::Buffer,
        offset: i32,
        stride: i32,
    ) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.vertex_buffers, index),
            (buffer, offset, stride),
        );
        if !self.skip(Kind::Vertex, skipped) {
            unsafe { gl.bind_vertex_buffer(index, Some(buffer), offset, stride) };
        }
    }

    /// The emulated first instance (a uniform of the current program), on a driver without
    /// base-instance draws.
    pub unsafe fn first_instance(
        &mut self,
        gl: &glow::Context,
        location: Option<&glow::UniformLocation>,
        value: u32,
    ) {
        // (Only while the program is known: the value belongs to it.)
        let skipped =
            self.s.program.is_some() && same(self.levers.state, &mut self.s.first_instance, value);
        if !self.skip(Kind::Uniform, skipped) {
            unsafe { gl.uniform_1_u32(location, value) };
        }
    }

    /// Counts a command whose calls the replay makes without `self`, and forgets what it
    /// knows of the state it may change. The commands `self` sees call by call are left alone.
    pub unsafe fn unmodeled(
        &mut self,
        gl: &glow::Context,
        command: &super::Command,
        shader_clear: bool,
    ) {
        use super::Command as C;
        // A pass's setup ends at its first command after the attachments.
        if !matches!(*command, C::BindAttachment { .. }) {
            unsafe { self.finish_setup(gl) };
        }
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

/// G3: how the driver tells its free video memory, if it does.
#[derive(Clone, Copy, Debug)]
pub(super) enum MemoryInfo {
    None,
    /// `GL_NVX_gpu_memory_info` (NVIDIA): the total and the currently available, in KB.
    Nvx,
    /// `GL_ATI_meminfo` (AMD): only the free texture memory, in KB. The budget is taken as
    /// what was free when the adapter was made, and the usage as how much less is free now.
    Ati {
        free_at_start_kb: u64,
    },
}

const GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX: u32 = 0x9048;
const GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX: u32 = 0x9049;
const TEXTURE_FREE_MEMORY_ATI: u32 = 0x87FC;

impl MemoryInfo {
    pub unsafe fn new(gl: &glow::Context, has: impl Fn(&str) -> bool) -> Self {
        if has("GL_NVX_gpu_memory_info") {
            Self::Nvx
        } else if has("GL_ATI_meminfo") {
            match unsafe { Self::ati_free_kb(gl) } {
                Some(free_at_start_kb) => Self::Ati { free_at_start_kb },
                None => Self::None,
            }
        } else {
            Self::None
        }
    }

    unsafe fn ati_free_kb(gl: &glow::Context) -> Option<u64> {
        let mut values = [0i32; 4];
        unsafe { gl.get_parameter_i32_slice(TEXTURE_FREE_MEMORY_ATI, &mut values) };
        (values[0] > 0).then_some(values[0] as u64)
    }

    /// (Budget, usage) in bytes.
    pub unsafe fn budget(&self, gl: &glow::Context) -> Option<(u64, u64)> {
        const KB: u64 = 1024;
        match *self {
            Self::None => None,
            Self::Nvx => {
                let total =
                    unsafe { gl.get_parameter_i32(GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX) };
                let available =
                    unsafe { gl.get_parameter_i32(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) };
                (total > 0).then(|| {
                    let (total, available) = (total as u64, available.max(0) as u64);
                    (total * KB, total.saturating_sub(available) * KB)
                })
            }
            Self::Ati { free_at_start_kb } => {
                let free = unsafe { Self::ati_free_kb(gl) }?;
                Some((
                    free_at_start_kb * KB,
                    free_at_start_kb.saturating_sub(free) * KB,
                ))
            }
        }
    }
}
