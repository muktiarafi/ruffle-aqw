//! AQW build: batched cacheAsBitmap redraws with filters.
//!
//! Every animated object with a glow/blur/drop shadow (glowing AQW gear, skill effects) is
//! redrawn into its cached texture every frame: drawing its content, then each filter step
//! (blur passes, glow) as separate render passes. Crowds and skill bursts redraw 60-130 such
//! objects per frame, each 5-9 passes. Here objects with the same kind of filter chain are
//! drawn side by side into an atlas page instead (`surface::draw_into_slots`), each filter step
//! runs once per page for all of them (`filters::batch`), and each object's slot is then copied
//! into its own cached texture. The texels are the same as the per-object path: a slot is
//! drawn like the object's own texture (moved by whole pixels, clipped to the slot), and the
//! filter passes clamp their samples to the slot as the sampler clamped them to the texture.
//!
//! Only for: a transparent background, contents without blends/alpha masks/Stage3D (leaf), and
//! filters that are all blurs, glows or drop shadows. Everything else, and objects referring to
//! a cached object still waiting in a batch, keeps the per-object path (a batch is drawn first
//! when something refers to one of its objects). `RUFFLE_NO_CACHE_BATCH` turns this off.

use crate::filters::batch::{MAX_SLOTS, Slot, SlotGlow};
use crate::surface::{LayerRect, is_leaf};
use ruffle_render::backend::BitmapCacheEntry;
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::filters::Filter;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const PAGE_WIDTH: u32 = crate::packing::MAX_WIDTH;
pub const PAGE_MAX_HEIGHT: u32 = crate::packing::MAX_HEIGHT;

/// Kinds of filter steps in a chain (pages hold objects with the same chain).
const BLUR: u8 = 0;
const GLOW: u8 = 1;
const SHADOW: u8 = 2;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// AQW build: turns batching on or off at runtime (for A/B comparisons in the same scene).
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    static ENV_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ENABLED.load(Ordering::Relaxed)
        && !*ENV_OFF.get_or_init(|| std::env::var_os("RUFFLE_NO_CACHE_BATCH").is_some())
}

fn key(handle: &ruffle_render::bitmap::BitmapHandle) -> usize {
    Arc::as_ptr(&handle.0) as *const () as usize
}

/// The filter chain of `entry` if it can be batched (see the module docs).
pub fn signature(entry: &BitmapCacheEntry, width: u32, height: u32) -> Option<Vec<u8>> {
    if !enabled()
        || entry.filters.is_empty()
        || entry.clear.a != 0
        || width == 0
        || height == 0
        || width > PAGE_WIDTH
        || height > PAGE_MAX_HEIGHT
        || !is_leaf(&entry.commands)
    {
        return None;
    }
    entry
        .filters
        .iter()
        .map(|filter| match filter {
            Filter::BlurFilter(_) => Some(BLUR),
            Filter::GlowFilter(_) => Some(GLOW),
            Filter::DropShadowFilter(_) => Some(SHADOW),
            _ => None,
        })
        .collect()
}

/// Cached objects waiting to be drawn, grouped by filter chain.
#[derive(Default)]
pub struct CacheBatch {
    groups: Vec<(Vec<u8>, Vec<(BitmapCacheEntry, (u32, u32))>)>,
    pending: HashSet<usize>,
    count: usize,
}

impl CacheBatch {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn add(&mut self, signature: Vec<u8>, entry: BitmapCacheEntry, size: (u32, u32)) {
        self.pending.insert(key(&entry.handle));
        self.count += 1;
        if let Some((_, entries)) = self.groups.iter_mut().find(|(s, _)| *s == signature) {
            entries.push((entry, size));
        } else {
            self.groups.push((signature, vec![(entry, size)]));
        }
    }

    /// Whether `commands` draw a cached object that's still waiting here.
    pub fn references_pending(&self, commands: &CommandList) -> bool {
        fn refers(commands: &[Command], pending: &HashSet<usize>) -> bool {
            commands.iter().any(|command| match command {
                Command::RenderBitmap { bitmap, .. } | Command::RenderStage3D { bitmap, .. } => {
                    pending.contains(&key(bitmap))
                }
                Command::Blend(list, _) => refers(&list.commands, pending),
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    refers(&maskee_commands.commands, pending)
                        || refers(&mask_commands.commands, pending)
                }
                _ => false,
            })
        }
        !self.pending.is_empty() && refers(&commands.commands, &self.pending)
    }

    pub fn take(&mut self) -> Vec<(Vec<u8>, Vec<(BitmapCacheEntry, (u32, u32))>)> {
        self.pending.clear();
        self.count = 0;
        std::mem::take(&mut self.groups)
    }
}

/// Packs the next objects (from the start of `sizes`) into a page (see `crate::packing`).
/// Returns the page size and the slots (one per packed object, in order).
pub fn pack(sizes: &[(u32, u32)]) -> ((u32, u32), Vec<Slot>) {
    let (_, page, positions) = crate::packing::pack(sizes, MAX_SLOTS);
    let slots = positions
        .into_iter()
        .zip(sizes)
        .map(|((x, y), &(width, height))| Slot {
            x,
            y,
            width,
            height,
        })
        .collect();
    (page, slots)
}

pub fn layer_rect(slot: &Slot) -> LayerRect {
    LayerRect {
        x: slot.x,
        y: slot.y,
        width: slot.width,
        height: slot.height,
    }
}

/// The glow step arguments of filter `filter` (a glow or drop shadow) of one slot.
pub fn glow_step(filter: &Filter) -> (swf::GlowFilter, (f32, f32)) {
    match filter {
        Filter::GlowFilter(glow) => (glow.clone(), (0.0, 0.0)),
        Filter::DropShadowFilter(shadow) => {
            // As `DropShadowFilter::apply`.
            let distance = shadow.distance.to_f32();
            let angle = shadow.angle.to_f32();
            let x = angle.cos() * distance;
            let y = angle.sin() * distance;
            (shadow.inner_glow_filter(), (-x, -y))
        }
        _ => unreachable!("only glows and drop shadows have a glow step"),
    }
}

pub fn is_blur_step(kind: u8) -> bool {
    kind == BLUR
}

pub fn slot_glows<'a>(
    slots: &[Slot],
    glows: &'a [(swf::GlowFilter, (f32, f32))],
) -> Vec<SlotGlow<'a>> {
    slots
        .iter()
        .zip(glows)
        .map(|(slot, (filter, blur_offset))| SlotGlow {
            slot: *slot,
            filter,
            blur_offset: *blur_offset,
        })
        .collect()
}
