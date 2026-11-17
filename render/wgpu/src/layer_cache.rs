//! AQW build: reusing rendered blend layers between pictures.
//!
//! Every picture used to render every blend layer from scratch, though most of them (map
//! decoration with overlay and add blends, players standing still, frame-by-frame animations
//! between two game frames) are pixel for pixel what they were in the previous picture. In
//! a crowded room that was 300-500 leaf layers and 26-76 MB of atlas pages per picture, for
//! every picture of frame interpolation and for every game frame.
//!
//! A leaf blend layer's command list decides its pixels completely: the shapes (by mesh id,
//! and the bitmaps their fills sample), the bitmaps (by texture id and version), and their
//! transforms, which are absolute. So a layer is identified by a hash of that list, plus the
//! target it's drawn into (size, quality, opacity). `surface::commands::plan_atlas` looks every
//! leaf layer up here: a hit composites the page slot it was rendered into before; a miss
//! renders it as usual. A miss is only KEPT (rendered into a page that stays alive) when the
//! same key was seen in the previous picture too: layers that move or change every picture
//! never pin a page, so the kept pages hold what repeats. Entries unused for a while are
//! dropped, pages that are mostly dead slots are compacted (their live layers re-rendered
//! into a fresh page), and a byte budget bounds the whole thing; memory pressure and GPU
//! out-of-memory clear it. Design and review: aqw-booster/design-layer-cache.md.
//!
//! `RUFFLE_NO_LAYER_CACHE` (or the `no-layer-cache` file next to the exe) turns it off.
//!
//! CHECK MODE (`RUFFLE_LAYER_CACHE_CHECK=1`): the image tests render single frames, so they
//! can't catch this cache's one failure mode, a layer kept past a change of its contents.
//! In check mode 1 hit in 16 (chosen by hash, a different set every picture) is ALSO rendered
//! fresh into a page of that picture, both slots are copied to a buffer, and a picture or two
//! later (once the GPU is done) the two are compared on the CPU. The screen shows the cached
//! one, as usual. A stale layer differs by a lot over many pixels; anti-aliasing rounding at
//! the other slot offset differs by 1-2/255 at a few edge pixels, so the log line gives both
//! the largest channel difference and how many pixels differ by more than 3/255, once per
//! layer key. Counters: `layer_cache_checked`, `layer_cache_mismatch`.

use crate::surface::commands::{AtlasPage, LayerRect};
use crate::{Texture, as_texture};
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Runtime switch (see `set_enabled`); `RUFFLE_NO_LAYER_CACHE` turns it off for good.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("RUFFLE_NO_LAYER_CACHE").is_none())
        && ENABLED.load(Ordering::Relaxed)
}

/// Whether hits are sampled and compared with fresh renders (see the module): the env var,
/// or `set_check_enabled` (the desktop's `layer-cache-check` file next to the exe).
pub fn check_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("RUFFLE_LAYER_CACHE_CHECK").is_some())
        || CHECK.load(Ordering::Relaxed)
}

static CHECK: AtomicBool = AtomicBool::new(false);

pub fn set_check_enabled(enabled: bool) {
    CHECK.store(enabled, Ordering::Relaxed);
}

/// A sampled hit being compared with a fresh render: both slots copied into `buffer` (the
/// cached one first, then the fresh one, rows padded to `padded_row` bytes), read back once
/// the GPU is done with the picture that recorded the copies.
#[derive(Debug)]
pub struct PendingCheck {
    pub key: u64,
    /// What the layer draws (see `describe`), for the log line if it differs.
    pub summary: String,
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub padded_row: u32,
    /// The picture the copies were recorded in; mapping is asked for two pictures later (the
    /// frame is on the GPU queue by then) and read once the GPU reports it mapped.
    pub recorded: u64,
    pub mapped: Arc<std::sync::atomic::AtomicBool>,
    pub map_requested: bool,
}

/// Over this channel difference a value counts as different (anti-aliasing rounding at the
/// other slot offset stays within it); a layer with more than `STALE_COUNT` values over
/// `STALE_THRESHOLD` is called stale rather than suspect.
const CHECK_THRESHOLD: u8 = 3;
const STALE_THRESHOLD: u8 = 24;
const STALE_COUNT: usize = 16;
/// Checks per picture, about (every check is a fresh render plus a readback; a crowd has ~500
/// hits per picture): the sampling rate is this over the previous picture's hits, so every
/// layer has the same chance wherever it's drawn, and the same number is the hard cap.
const CHECKS_PER_PICTURE: u32 = 4;

/// What a leaf layer draws, for a check's log line: command kinds, mesh ids and the textures'
/// (id, version), the first few of each.
pub fn describe(commands: &CommandList) -> String {
    let mut kinds: HashMap<&'static str, usize> = HashMap::new();
    let mut meshes: Vec<u64> = Vec::new();
    let mut textures: Vec<(u64, u64)> = Vec::new();
    for command in &commands.commands {
        let kind = match command {
            Command::RenderShape { shape, .. } => {
                let mesh = crate::mesh::as_mesh(shape);
                meshes.push(mesh.id);
                for bitmap in &mesh.bitmaps {
                    textures.push(as_texture(bitmap).stamp());
                }
                "shape"
            }
            Command::RenderBitmap { bitmap, .. } => {
                textures.push(as_texture(bitmap).stamp());
                "bitmap"
            }
            Command::RenderStage3D { .. } => "stage3d",
            Command::DrawRect { .. } => "rect",
            Command::DrawLine { .. } | Command::DrawLineRect { .. } => "line",
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => "mask",
            Command::RenderAlphaMask { .. } => "alpha-mask",
            Command::Blend(..) => "blend",
        };
        *kinds.entry(kind).or_default() += 1;
    }
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort();
    meshes.truncate(8);
    textures.truncate(8);
    format!("{kinds:?}, meshes {meshes:?}, textures (id, version) {textures:?}")
}

/// How long an entry is kept after its last use: a few game frames at 30 fps (a layer that
/// blinks every other frame still hits), much more than the time between in-between pictures.
const KEEP: Duration = Duration::from_millis(120);
/// Pages older than this whose live slots cover less than `COMPACT_BELOW` of them are
/// compacted: their entries are dropped, so the layers render again into a fresh page.
const COMPACT_AFTER: Duration = Duration::from_secs(1);
const COMPACT_BELOW: f64 = 0.4;
const BUDGET_DISCRETE_MB: u64 = 256;
const BUDGET_INTEGRATED_MB: u64 = 64;
const BUDGET_PRESSURE_MB: u64 = 32;

#[derive(Debug)]
struct Entry {
    page: Arc<AtlasPage>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    last_used: Instant,
}

#[derive(Default, Debug)]
pub struct LayerCache {
    entries: HashMap<u64, Entry>,
    /// The keys looked up in the previous picture and in this one (a miss is kept only when
    /// its key was seen in the previous picture: see the module).
    seen_last: HashSet<u64>,
    seen_now: HashSet<u64>,
    /// When each page (by address) was made and the slot area packed into it, for compaction.
    page_made: HashMap<usize, (Instant, u64)>,
    /// Pictures so far (for check sampling), checks sampled in this picture, and the hits of
    /// this and the previous picture (the sampling rate aims at `CHECKS_PER_PICTURE`).
    picture: u64,
    checks_this_picture: u32,
    hits_this_picture: u64,
    hits_last_picture: u64,
    /// Check mode: comparisons waiting for the GPU, and the keys already reported.
    checks: Vec<PendingCheck>,
    reported: HashSet<u64>,
}

/// Where a cached layer was rendered.
pub struct Cached {
    pub page: Arc<AtlasPage>,
    pub x: u32,
    pub y: u32,
}

impl LayerCache {
    /// Looks a layer up (`width`, `height`: its slot size, which a hit must have too); also
    /// notes its key as seen in this picture.
    pub fn lookup(&mut self, key: u64, width: u32, height: u32) -> Option<Cached> {
        self.seen_now.insert(key);
        let entry = self.entries.get_mut(&key)?;
        if entry.width != width || entry.height != height {
            return None;
        }
        self.hits_this_picture += 1;
        entry.last_used = Instant::now();
        Some(Cached {
            page: entry.page.clone(),
            x: entry.x,
            y: entry.y,
        })
    }

    /// Whether a layer that missed should be rendered into a page that is kept (its key was
    /// seen in the previous picture, so it's likely to repeat).
    pub fn admit(&self, key: u64) -> bool {
        self.seen_last.contains(&key)
    }

    /// The picture being recorded (counts up in `end_picture`).
    pub fn picture(&self) -> u64 {
        self.picture
    }

    /// Check mode: whether this hit is one of the ~`CHECKS_PER_PICTURE` per picture compared
    /// with a fresh render: chosen by hash (a different set every picture) at a rate of that
    /// many over the previous picture's hits, so a layer drawn late has the same chance as one
    /// drawn first; the same number caps it.
    pub fn should_check(&mut self, key: u64) -> bool {
        if !check_enabled() || self.checks_this_picture >= CHECKS_PER_PICTURE {
            return false;
        }
        let hashed = (key ^ self.picture).wrapping_mul(0x9E3779B97F4A7C15);
        let per_hit = u64::MAX / self.hits_last_picture.max(CHECKS_PER_PICTURE as u64);
        if hashed >= per_hit.saturating_mul(CHECKS_PER_PICTURE as u64) {
            return false;
        }
        self.checks_this_picture += 1;
        true
    }

    pub fn queue_check(&mut self, check: PendingCheck) {
        crate::perf_counters::inc(crate::perf_counters::LAYER_CACHE_CHECKED);
        self.checks.push(check);
    }

    /// Check mode: asks for the buffers of checks old enough to be on the GPU queue, and
    /// compares the ones the GPU has finished.
    fn run_checks(&mut self) {
        let picture = self.picture;
        let mut done = Vec::new();
        for (i, check) in self.checks.iter_mut().enumerate() {
            if !check.map_requested && picture >= check.recorded + 2 {
                check.map_requested = true;
                let mapped = check.mapped.clone();
                check.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped.store(true, Ordering::Release);
                    }
                });
            } else if check.map_requested && picture >= check.recorded + 60 {
                // The GPU never answered (or the mapping failed): give it up.
                done.push(i);
            } else if check.mapped.load(Ordering::Acquire) {
                done.push(i);
            }
        }
        for i in done.into_iter().rev() {
            let check = self.checks.swap_remove(i);
            if !check.mapped.load(Ordering::Acquire) {
                continue;
            }
            let (max, over, big) = {
                let Ok(data) = check.buffer.slice(..).get_mapped_range() else {
                    check.buffer.unmap();
                    continue;
                };
                let half = check.padded_row as usize * check.height as usize;
                let (cached, fresh) = (&data[..half], &data[half..half * 2]);
                let mut max = 0u8;
                let mut over = 0usize;
                let mut big = 0usize;
                for y in 0..check.height as usize {
                    let row = y * check.padded_row as usize;
                    for x in 0..check.width as usize * 4 {
                        let d = cached[row + x].abs_diff(fresh[row + x]);
                        if d > max {
                            max = d;
                        }
                        if d > CHECK_THRESHOLD {
                            over += 1;
                        }
                        if d > STALE_THRESHOLD {
                            big += 1;
                        }
                    }
                }
                (max, over, big)
            };
            check.buffer.unmap();
            if over > 0 {
                crate::perf_counters::inc(crate::perf_counters::LAYER_CACHE_MISMATCH);
                if self.reported.insert(check.key) {
                    let verdict = if big > STALE_COUNT { "STALE" } else { "suspect" };
                    tracing::warn!(
                        target: "ruffle_aqw_perf",
                        "layer cache CHECK {verdict}: key {:016x} differs from a fresh render: {} channel values over {}/255, {} over {}/255, largest {}/255, in a {}x{} slot; layer: {}",
                        check.key,
                        over,
                        CHECK_THRESHOLD,
                        big,
                        STALE_THRESHOLD,
                        max,
                        check.width,
                        check.height,
                        check.summary
                    );
                }
            }
        }
    }

    pub fn insert(&mut self, key: u64, page: Arc<AtlasPage>, x: u32, y: u32, width: u32, height: u32) {
        let now = Instant::now();
        self.page_made
            .entry(Arc::as_ptr(&page) as usize)
            .or_insert((now, 0))
            .1 += width as u64 * height as u64;
        self.entries.insert(
            key,
            Entry {
                page,
                x,
                y,
                width,
                height,
                last_used: now,
            },
        );
    }

    /// Once per picture, after it was recorded: forgets stale entries, compacts sparse pages
    /// and keeps within the byte budget. `integrated`: the GPU shares the system's memory.
    pub fn end_picture(&mut self, integrated: bool) {
        self.picture += 1;
        self.checks_this_picture = 0;
        self.hits_last_picture = std::mem::take(&mut self.hits_this_picture);
        if check_enabled() {
            self.run_checks();
        }
        self.seen_last = std::mem::take(&mut self.seen_now);
        let now = Instant::now();
        self.entries.retain(|_, entry| now.duration_since(entry.last_used) <= KEEP);

        // Per page: entries, their live slot area, and the page's bytes.
        let mut pages: HashMap<usize, (usize, u64, u64)> = HashMap::new();
        for entry in self.entries.values() {
            let page = pages
                .entry(Arc::as_ptr(&entry.page) as usize)
                .or_insert((0, 0, entry.page.bytes()));
            page.0 += 1;
            page.1 += entry.width as u64 * entry.height as u64;
        }
        // Compaction: an old page most of whose slots are dead (against what was packed into
        // it, not its whole area: packing leaves gaps) loses its live ones too; they render
        // again into a fresh page together, and this one goes back to the pool.
        let sparse: HashSet<usize> = pages
            .iter()
            .filter(|(page, (_, live, _))| {
                let (made, packed) = self.page_made.get(page).copied().unwrap_or((now, 0));
                now.duration_since(made) >= COMPACT_AFTER
                    && (*live as f64) < packed as f64 * COMPACT_BELOW
            })
            .map(|(page, _)| *page)
            .collect();
        if !sparse.is_empty() {
            crate::perf_counters::add(
                crate::perf_counters::LAYER_CACHE_COMPACTED,
                sparse.len() as u32,
            );
            self.entries
                .retain(|_, entry| !sparse.contains(&(Arc::as_ptr(&entry.page) as usize)));
            for page in &sparse {
                pages.remove(page);
            }
        }

        let budget_mb = if crate::buffer_pool::MEMORY_PRESSURE.load(Ordering::Relaxed) {
            BUDGET_PRESSURE_MB
        } else if integrated {
            BUDGET_INTEGRATED_MB
        } else {
            BUDGET_DISCRETE_MB
        };
        let mut bytes: u64 = pages.values().map(|(_, _, bytes)| bytes).sum();
        if bytes > budget_mb << 20 {
            // Least recently used first; a page's bytes are freed with its last entry.
            let mut by_age: Vec<(Instant, u64)> = self
                .entries
                .iter()
                .map(|(key, entry)| (entry.last_used, *key))
                .collect();
            by_age.sort_unstable();
            for (_, key) in by_age {
                if bytes <= budget_mb << 20 {
                    break;
                }
                if let Some(entry) = self.entries.remove(&key)
                    && let Some(page) = pages.get_mut(&(Arc::as_ptr(&entry.page) as usize))
                {
                    page.0 -= 1;
                    if page.0 == 0 {
                        bytes -= page.2;
                    }
                }
            }
        }
        self.page_made
            .retain(|page, _| pages.get(page).is_some_and(|(count, _, _)| *count > 0));
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.page_made.clear();
        self.seen_last.clear();
        self.seen_now.clear();
    }

    /// Check mode: comparisons still waiting for the GPU.
    pub fn checks_pending(&self) -> usize {
        self.checks.len()
    }

    /// Distinct pages held, their bytes, and the layers in them.
    pub fn stats(&self) -> (usize, u64, usize) {
        let mut pages: HashMap<usize, u64> = HashMap::new();
        for entry in self.entries.values() {
            pages
                .entry(Arc::as_ptr(&entry.page) as usize)
                .or_insert_with(|| entry.page.bytes());
        }
        (pages.len(), pages.values().sum(), self.entries.len())
    }

}

/// What a leaf layer's pixels and slot depend on, as a 64-bit key, or `None` for a layer that
/// can't be cached (it draws something whose contents aren't tracked: Stage3D, a nested blend
/// or alpha mask, a 3D projection). `rect`: where and how big its slot is, which depends on
/// the blend kind (`complex` blends get a rounded rectangle): the same content with another
/// blend mode is another layer.
pub fn layer_key(
    commands: &CommandList,
    target_width: u32,
    target_height: u32,
    quality: StageQuality,
    opaque_target: bool,
    rect: LayerRect,
    complex: bool,
) -> Option<u64> {
    let mut hasher = DefaultHasher::new();
    hasher.write_u32(target_width);
    hasher.write_u32(target_height);
    hasher.write_u8(quality as u8);
    hasher.write_u8(opaque_target as u8);
    hasher.write_u32(rect.x);
    hasher.write_u32(rect.y);
    hasher.write_u32(rect.width);
    hasher.write_u32(rect.height);
    hasher.write_u8(complex as u8);
    hash_list(commands, &mut hasher)?;
    Some(hasher.finish())
}

fn hash_list(commands: &CommandList, hasher: &mut DefaultHasher) -> Option<()> {
    hasher.write_usize(commands.commands.len());
    for command in &commands.commands {
        match command {
            Command::RenderShape { shape, transform } => {
                hasher.write_u8(1);
                let mesh = crate::mesh::as_mesh(shape);
                hasher.write_u64(mesh.id);
                for bitmap in &mesh.bitmaps {
                    hash_texture(as_texture(bitmap), hasher);
                }
                hash_transform(transform, hasher)?;
            }
            Command::RenderBitmap {
                bitmap,
                transform,
                smoothing,
                pixel_snapping,
                region,
            } => {
                hasher.write_u8(2);
                hash_texture(as_texture(bitmap), hasher);
                hash_transform(transform, hasher)?;
                hasher.write_u8(*smoothing as u8);
                hasher.write_u8(*pixel_snapping as u8);
                hasher.write_u32(region.x_min);
                hasher.write_u32(region.y_min);
                hasher.write_u32(region.x_max);
                hasher.write_u32(region.y_max);
            }
            Command::RenderStage3D { .. } => return None,
            Command::DrawRect { color, matrix }
            | Command::DrawLine { color, matrix }
            | Command::DrawLineRect { color, matrix } => {
                hasher.write_u8(match command {
                    Command::DrawRect { .. } => 3,
                    Command::DrawLine { .. } => 4,
                    _ => 5,
                });
                hasher.write_u8(color.r);
                hasher.write_u8(color.g);
                hasher.write_u8(color.b);
                hasher.write_u8(color.a);
                hash_matrix(matrix, hasher);
            }
            Command::PushMask => hasher.write_u8(6),
            Command::ActivateMask => hasher.write_u8(7),
            Command::DeactivateMask => hasher.write_u8(8),
            Command::PopMask => hasher.write_u8(9),
            Command::RenderAlphaMask { .. } | Command::Blend(..) => return None,
        }
    }
    Some(())
}

fn hash_texture(texture: &Texture, hasher: &mut DefaultHasher) {
    let (id, version) = texture.stamp();
    hasher.write_u64(id);
    hasher.write_u64(version);
}

fn hash_matrix(matrix: &ruffle_render::matrix::Matrix, hasher: &mut DefaultHasher) {
    hasher.write_u32(matrix.a.to_bits());
    hasher.write_u32(matrix.b.to_bits());
    hasher.write_u32(matrix.c.to_bits());
    hasher.write_u32(matrix.d.to_bits());
    hasher.write_i32(matrix.tx.get());
    hasher.write_i32(matrix.ty.get());
}

fn hash_transform(transform: &Transform, hasher: &mut DefaultHasher) -> Option<()> {
    if transform.perspective_projection.is_some() {
        return None;
    }
    hash_matrix(&transform.matrix, hasher);
    hasher.write_u64(transform.tz.to_bits());
    let color = &transform.color_transform;
    hasher.write_u32(color.r_multiply.to_f32().to_bits());
    hasher.write_u32(color.g_multiply.to_f32().to_bits());
    hasher.write_u32(color.b_multiply.to_f32().to_bits());
    hasher.write_u32(color.a_multiply.to_f32().to_bits());
    hasher.write_i16(color.r_add);
    hasher.write_i16(color.g_add);
    hasher.write_i16(color.b_add);
    hasher.write_i16(color.a_add);
    Some(())
}
