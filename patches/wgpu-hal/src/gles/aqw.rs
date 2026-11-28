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

/// The levers of G2 and G5 (`RUFFLE_GL_LEVERS`, asked once): a comma-separated list of `fbo`
/// (G2a: detach only stale attachments), `state` (G2b: skip calls that set what's already set),
/// `current` (G5b: `AdapterContext::hold` keeps the EGL context current through a frame) and
/// `recycle` (G5c: emptied command buffers are reused, [`SpareBuffers`]). Unset: `fbo` and
/// `state`; `current` and `recycle` gained nothing measurable on a phone (ANGLE over Vulkan,
/// docs/GL_RENDERER_PROPOSAL.md in Turu, G5), so they're only on when named. `none` (or empty):
/// none, the replay and locking as upstream wgpu has them.
///
/// `RUFFLE_GL_AB=<lever>,<seconds>` flips one lever in every other window of `<seconds>`, and
/// logs each switch: an A/B run in one session, in the same room with the same players (a new
/// session lands in another room, so runs across sessions don't compare).
#[derive(Clone, Copy)]
pub(super) struct Levers {
    pub fbo: bool,
    pub state: bool,
    #[cfg_attr(any(windows, webgl), allow(dead_code, reason = "only EGL holds"))]
    pub current: bool,
    pub recycle: bool,
}

pub(super) fn levers() -> Levers {
    #[cfg(gles_with_std)]
    {
        static LEVERS: std::sync::OnceLock<Levers> = std::sync::OnceLock::new();
        let levers = *LEVERS.get_or_init(|| match std::env::var("RUFFLE_GL_LEVERS") {
            Err(_) => Levers::DEFAULT,
            Ok(list) => {
                let has = |name: &str| list.split(',').any(|l| l.trim() == name);
                Levers {
                    fbo: has("fbo"),
                    state: has("state"),
                    current: has("current"),
                    recycle: has("recycle"),
                }
            }
        });
        ab::flipped(levers)
    }
    #[cfg(not(gles_with_std))]
    Levers::DEFAULT
}

impl Levers {
    const DEFAULT: Levers = Levers {
        fbo: true,
        state: true,
        current: false,
        recycle: false,
    };

    fn get_mut(&mut self, name: &str) -> Option<&mut bool> {
        match name {
            "fbo" => Some(&mut self.fbo),
            "state" => Some(&mut self.state),
            "current" => Some(&mut self.current),
            "recycle" => Some(&mut self.recycle),
            _ => None,
        }
    }
}

/// `RUFFLE_GL_AB` (see [`Levers`]).
#[cfg(gles_with_std)]
mod ab {
    use super::Levers;
    use alloc::string::String;
    use core::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    struct Ab {
        lever: String,
        secs: u64,
        since: Instant,
    }

    pub(super) fn flipped(mut levers: Levers) -> Levers {
        static AB: OnceLock<Option<Ab>> = OnceLock::new();
        static WINDOW: AtomicU64 = AtomicU64::new(u64::MAX);
        let Some(ab) = AB.get_or_init(|| {
            let value = std::env::var("RUFFLE_GL_AB").ok()?;
            let (lever, secs) = value.split_once(',')?;
            let secs = secs.trim().parse::<u64>().ok().filter(|&s| s > 0)?;
            let lever = String::from(lever.trim());
            let mut probe = levers;
            probe.get_mut(&lever)?;
            Some(Ab {
                lever,
                secs,
                since: Instant::now(),
            })
        }) else {
            return levers;
        };
        let window = ab.since.elapsed().as_secs() / ab.secs;
        let Some(on) = levers.get_mut(&ab.lever) else {
            return levers;
        };
        if window % 2 == 1 {
            *on = !*on;
        }
        if WINDOW.swap(window, Ordering::Relaxed) != window {
            log::info!(
                target: "ruffle_aqw_perf",
                "GL A/B: {} {} for {} s",
                ab.lever,
                if *on { "on" } else { "off" },
                ab.secs
            );
        }
        levers
    }
}

/// G5c (`recycle`): emptied command buffers, shared by the device's encoders, so a new one
/// starts with room for its commands instead of growing its lists again (~11,000 command
/// buffers a second on a phone in a crowded room). Device-wide with a hard cap, not per
/// encoder: wgpu-core pools its encoders, so anything kept per encoder grows with that pool
/// (Mugetsu's D3D12 recycling grew to gigabytes over a session before it was capped).
#[derive(Default)]
pub(super) struct SpareBuffers(parking_lot::Mutex<Vec<super::CommandBuffer>>);

impl core::fmt::Debug for SpareBuffers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SpareBuffers({})", self.0.lock().len())
    }
}

impl SpareBuffers {
    /// Buffers kept at most.
    const MAX: usize = 128;
    /// Kept only if no list grew past this: a buffer that once held a huge pass isn't kept.
    const MAX_COMMANDS: usize = 1024;
    const MAX_DATA_BYTES: usize = 16 * 1024;
    const MAX_QUERIES: usize = 64;

    /// An empty command buffer, a kept one if there is one.
    pub fn take(&self) -> super::CommandBuffer {
        if !levers().recycle {
            return Default::default();
        }
        let buffer = self.0.lock().pop();
        match buffer {
            Some(buffer) => {
                note_reused();
                buffer
            }
            None => Default::default(),
        }
    }

    /// Keeps what fits of `buffers`, emptied (`CommandEncoder::reset_all`); drops the rest.
    pub fn give_back(&self, buffers: impl Iterator<Item = super::CommandBuffer>) {
        if !levers().recycle {
            return;
        }
        let room = Self::MAX.saturating_sub(self.0.lock().len());
        let kept: Vec<_> = buffers
            .filter(|b| {
                b.commands.capacity() <= Self::MAX_COMMANDS
                    && b.data_bytes.capacity() <= Self::MAX_DATA_BYTES
                    && b.queries.capacity() <= Self::MAX_QUERIES
            })
            .take(room)
            .map(|mut b| {
                // Emptied here, outside the lock: the commands hold buffers and textures.
                b.label = None;
                b.commands.clear();
                b.data_bytes.clear();
                b.queries.clear();
                b
            })
            .collect();
        let mut spare = self.0.lock();
        let room = Self::MAX.saturating_sub(spare.len());
        spare.extend(kept.into_iter().take(room));
    }
}

/// Counts a command buffer reused (G5c; `RUFFLE_GL_STATS` only).
fn note_reused() {
    #[cfg(gles_with_std)]
    if stats_enabled() {
        totals::REUSED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
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
        let Some(attached) = attached else {
            // (Something it doesn't model: an external framebuffer binds a framebuffer of its
            // own, whatever the levers.)
            self.count(Kind::Fbo, 1);
            self.fb.attached = [None; POINTS];
            self.fb.wanted = None;
            self.forget();
            return true;
        };
        let (true, Some(wanted)) = (self.levers.fbo, self.fb.wanted.as_mut()) else {
            self.count(Kind::Fbo, 1);
            self.fb.attached = [None; POINTS];
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

    /// Before `glVertexAttribPointer` or `glVertexAttribFormat` and `glVertexAttribBinding` of
    /// `location`: since OpenGL ES 3.1 these also set which vertex buffer binding the
    /// attribute reads (the pointer: binding `location`, with the buffer, stride and offset),
    /// so what's known of that binding goes. `elsewhere`: bound to a binding of another index
    /// (`glVertexAttribBinding`), so its `glVertexAttribDivisor` no longer holds either.
    pub fn attrib_layout_changed(&mut self, location: u32, elsewhere: bool) {
        *at(&mut self.s.vertex_buffers, location) = None;
        *at(&mut self.s.binding_divisor, location) = None;
        if elsewhere {
            *at(&mut self.s.attrib_divisor, location) = None;
        }
    }

    /// `glVertexAttribDivisor`, which since OpenGL ES 3.1 is `glVertexAttribBinding(i, i)` and
    /// `glVertexBindingDivisor(i, divisor)`.
    pub unsafe fn attrib_divisor(&mut self, gl: &glow::Context, location: u32, divisor: u32) {
        let skipped = same(
            self.levers.state,
            at(&mut self.s.attrib_divisor, location),
            divisor,
        );
        if !self.skip(Kind::Vertex, skipped) {
            *at(&mut self.s.binding_divisor, location) = None;
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
            // (Which attributes read this binding isn't tracked.)
            self.s.attrib_divisor.clear();
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

    /// G5c: command buffers reused (`SpareBuffers::take`).
    pub static REUSED: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

    struct Totals {
        since: Option<Instant>,
        counts: Counts,
        submits: u64,
        submit_time: Duration,
        /// Context locks by thread name.
        locks: Vec<(String, u64)>,
        /// G5b: locks taken while the thread held the context (`AdapterContext::hold`), which
        /// neither made it current nor released it.
        held_locks: u64,
        /// G5b: holds begun (each one makes the context current and releases it once).
        holds: u64,
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
        held_locks: 0,
        holds: 0,
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

    pub fn note_held_lock() {
        TOTALS.lock().unwrap_or_else(|e| e.into_inner()).held_locks += 1;
    }

    pub fn note_hold() {
        TOTALS.lock().unwrap_or_else(|e| e.into_inner()).holds += 1;
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
        let _ = write!(
            line,
            "); held: {:.0} holds/s, {:.0} locks/s inside them; command buffers reused {:.0}/s",
            per_s(totals.holds),
            per_s(totals.held_locks),
            per_s(REUSED.swap(0, core::sync::atomic::Ordering::Relaxed)),
        );
        log::info!(target: "ruffle_aqw_perf", "{line}");
        totals.since = Some(now);
        totals.counts = Counts::default();
        totals.submits = 0;
        totals.submit_time = Duration::ZERO;
        totals.locks.clear();
        totals.held_locks = 0;
        totals.holds = 0;
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

/// Counts a lock taken inside a hold (G5b), which made nothing current (`RUFFLE_GL_STATS`
/// only).
#[inline]
#[cfg_attr(any(windows, webgl), allow(dead_code, reason = "only EGL holds"))]
pub(super) fn note_held_lock() {
    #[cfg(gles_with_std)]
    if stats_enabled() {
        totals::note_held_lock();
    }
}

/// Counts a hold begun (G5b; `RUFFLE_GL_STATS` only). Its own lock is counted by [`note_lock`].
#[inline]
#[cfg_attr(any(windows, webgl), allow(dead_code, reason = "only EGL holds"))]
pub(super) fn note_hold() {
    #[cfg(gles_with_std)]
    if stats_enabled() {
        totals::note_hold();
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

/// Turu (docs/GL_RENDERER_PROPOSAL.md, "Step 1"): GL errors seen since [`take_errors`] last
/// took them, by code. A GL call that fails only sets an error flag nobody reads, and the
/// picture is wrong with nothing in the log (G6's first build: every `glBufferSubData` refused
/// by immutable storage). GL keeps one flag per code until it's read, so reading them once a
/// second ([`check_errors`], at the end of a submit, the context locked) still finds every kind
/// of error that happened, for the cost of a few `glGetError`s a second.
#[cfg(gles_with_std)]
static ERRORS: std::sync::Mutex<Vec<(u32, u64)>> = std::sync::Mutex::new(Vec::new());

/// Reads the GL error flags, at most once a second (see [`ERRORS`]).
///
/// # Safety
///
/// `gl` must be current on this thread (the context locked).
#[cfg(gles_with_std)]
pub(super) unsafe fn check_errors(gl: &glow::Context) {
    use std::time::{Duration, Instant};
    static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
            return;
        }
        *last = Some(Instant::now());
    }
    // `RUFFLE_TEST_GL_ERROR=1`: one invalid call at the first check (GL_INVALID_ENUM), to test
    // that an error reaches the frontend's log.
    static TESTED: std::sync::Once = std::sync::Once::new();
    TESTED.call_once(|| {
        if std::env::var("RUFFLE_TEST_GL_ERROR").is_ok_and(|v| v == "1") {
            unsafe { gl.enable(0) };
        }
    });
    // One flag per code: a handful at most (bounded, in case a lost context keeps answering).
    for _ in 0..8 {
        let code = unsafe { gl.get_error() };
        if code == glow::NO_ERROR {
            break;
        }
        let mut errors = ERRORS.lock().unwrap_or_else(|e| e.into_inner());
        match errors.iter_mut().find(|(c, _)| *c == code) {
            Some((_, count)) => *count += 1,
            None => errors.push((code, 1)),
        }
    }
}

/// The GL errors seen since the last call, by code (with how many of the once-a-second checks
/// found each), for the frontend's log. Empty without GL.
pub fn take_errors() -> Vec<(u32, u64)> {
    #[cfg(gles_with_std)]
    {
        core::mem::take(&mut *ERRORS.lock().unwrap_or_else(|e| e.into_inner()))
    }
    #[cfg(not(gles_with_std))]
    Vec::new()
}

/// A GL error code's name.
pub fn error_name(code: u32) -> &'static str {
    match code {
        glow::INVALID_ENUM => "GL_INVALID_ENUM",
        glow::INVALID_VALUE => "GL_INVALID_VALUE",
        glow::INVALID_OPERATION => "GL_INVALID_OPERATION",
        glow::OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        glow::INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
        glow::CONTEXT_LOST => "GL_CONTEXT_LOST",
        _ => "an unknown GL error",
    }
}
