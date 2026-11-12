//! Turu: how long the GPU takes to run a frame, for the perf log (`RUFFLE_GPU_TIMING=1`).
//!
//! The CPU timers (`perf_counters::timers`) say how long recording and submitting took, not how
//! long the GPU then ran. Here three timestamps per frame (its start, the end of its
//! cacheAsBitmap and filter redraws, its end) split it into those redraws and the stage itself.
//! They're written at the end of tiny passes (a 1x1 clear), so only `TIMESTAMP_QUERY` is needed:
//! Metal has no timestamps between passes (`TIMESTAMP_QUERY_INSIDE_ENCODERS`). Not wgpu-profiler's scopes:
//! Ruffle opens one per pass and per draw, thousands in a crowded frame, which would weigh on
//! what's measured. Results are read back a few frames later, never waited for.
//!
//! A span is from the frame's first command to its last on the GPU, so it also counts the time
//! the GPU waited for the CPU to finish the frame's later command segments (`segments`).

use crate::descriptors::Descriptors;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Frames measured at once: one recorded, one waiting for the next frame to copy its
/// timestamps out, the others read back (up to two frames are in flight).
const SLOTS: usize = 6;
/// Timestamps per frame: start, cache redraws done, end.
const POINTS: u32 = 3;
const BYTES: u64 = POINTS as u64 * 8;

/// `RUFFLE_GPU_TIMING` set (and not `0`).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("RUFFLE_GPU_TIMING").is_ok_and(|v| !v.is_empty() && v != "0"))
}

// Totals since the last `take`, over every player (Grid View's tiles draw too).
static FRAMES: AtomicU64 = AtomicU64::new(0);
static CACHE_NS: AtomicU64 = AtomicU64::new(0);
static STAGE_NS: AtomicU64 = AtomicU64::new(0);
static MAX_NS: AtomicU64 = AtomicU64::new(0);
/// Frames not measured: every slot was still being read back.
static SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Frames whose timestamps came back out of order or zero (a driver that doesn't really write
/// them): left out of the times.
static INVALID: AtomicU64 = AtomicU64::new(0);
/// The size and samples of the last frame measured.
static LAST_SIZE: AtomicU64 = AtomicU64::new(0);
static LAST_SAMPLES: AtomicU64 = AtomicU64::new(0);
/// Whether the device has timestamps at all (set by the first player that asked).
static SUPPORTED: AtomicU64 = AtomicU64::new(0); // 0 unknown, 1 no, 2 yes

/// What [`take`] returns: GPU time per measured frame, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub frames: u64,
    pub skipped: u64,
    pub invalid: u64,
    /// cacheAsBitmap redraws and their filters, per frame.
    pub cache_ms: f64,
    /// The stage, per frame.
    pub stage_ms: f64,
    /// The longest frame (both parts).
    pub max_ms: f64,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl Totals {
    pub fn total_ms(&self) -> f64 {
        self.cache_ms + self.stage_ms
    }
}

/// The totals since the last call, or `None` when timing is off or the device has no
/// timestamps (the reason is logged once at the start).
pub fn take() -> Option<Totals> {
    if !enabled() || SUPPORTED.load(Ordering::Relaxed) != 2 {
        return None;
    }
    let frames = FRAMES.swap(0, Ordering::Relaxed);
    let per = |ns: u64| {
        if frames == 0 {
            0.0
        } else {
            ns as f64 / frames as f64 / 1e6
        }
    };
    let size = LAST_SIZE.load(Ordering::Relaxed);
    Some(Totals {
        frames,
        skipped: SKIPPED.swap(0, Ordering::Relaxed),
        invalid: INVALID.swap(0, Ordering::Relaxed),
        cache_ms: per(CACHE_NS.swap(0, Ordering::Relaxed)),
        stage_ms: per(STAGE_NS.swap(0, Ordering::Relaxed)),
        max_ms: MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1e6,
        width: (size >> 32) as u32,
        height: size as u32,
        samples: LAST_SAMPLES.load(Ordering::Relaxed) as u32,
    })
}

/// Adds one frame's timestamps (in ticks) to the totals.
fn record(ticks: [u64; POINTS as usize], period_ns: f64) {
    let [start, cached, end] = ticks;
    if start == 0 || cached < start || end < cached {
        INVALID.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let ns = |ticks: u64| (ticks as f64 * period_ns) as u64;
    let (cache, stage) = (ns(cached - start), ns(end - cached));
    FRAMES.fetch_add(1, Ordering::Relaxed);
    CACHE_NS.fetch_add(cache, Ordering::Relaxed);
    STAGE_NS.fetch_add(stage, Ordering::Relaxed);
    MAX_NS.fetch_max(cache + stage, Ordering::Relaxed);
}

enum Slot {
    Free,
    Recording,
    /// Ended; its timestamps are copied out at the start of the next frame.
    Ended,
    /// Copied out in the frame being recorded; read back once that frame is submitted.
    Copying,
    /// Submitted; `done` once mapped, `ok` if mapping worked.
    Reading {
        done: Arc<AtomicBool>,
        ok: Arc<AtomicBool>,
    },
}

/// One player's timer (each renderer has its own queries and buffers).
pub struct GpuTimer {
    queries: wgpu::QuerySet,
    /// The 1x1 texture the timestamp passes clear.
    marker: wgpu::TextureView,
    resolve: wgpu::Buffer,
    readback: Vec<wgpu::Buffer>,
    slots: Vec<Slot>,
    current: Option<usize>,
    period_ns: f64,
}

impl GpuTimer {
    /// `None` unless timing is on and the device has timestamps.
    pub fn new(descriptors: &Descriptors) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let supported = descriptors
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        if SUPPORTED.swap(if supported { 2 } else { 1 }, Ordering::Relaxed) == 0 {
            if supported {
                tracing::info!("GPU frame timing on (RUFFLE_GPU_TIMING)");
            } else {
                tracing::warn!("RUFFLE_GPU_TIMING: this graphics device has no timestamp queries");
            }
        }
        if !supported {
            return None;
        }
        let device = &descriptors.device;
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Turu GPU timing"),
            ty: wgpu::QueryType::Timestamp,
            count: POINTS * SLOTS as u32,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Turu GPU timing resolve"),
            size: wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT * SLOTS as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = (0..SLOTS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Turu GPU timing readback"),
                    size: BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let marker = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Turu GPU timing marker"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Some(Self {
            queries,
            marker,
            resolve,
            readback,
            slots: (0..SLOTS).map(|_| Slot::Free).collect(),
            current: None,
            period_ns: descriptors.queue.get_timestamp_period() as f64,
        })
    }

    fn write(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        // A pass that only clears a 1x1 texture: Metal leaves out an empty compute pass and its
        // timestamps (they came back zero), and Ruffle's device can't dispatch compute work.
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Turu GPU timestamp"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.marker,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(index),
            }),
            ..Default::default()
        });
    }

    /// The frame's first timestamp, before any of its commands. `size` and `samples` are the
    /// frame's target, for the log.
    pub fn begin_frame(
        &mut self,
        descriptors: &Descriptors,
        encoder: &mut wgpu::CommandEncoder,
        size: (u32, u32),
        samples: u32,
    ) {
        self.collect(descriptors);
        // The last frame's timestamps, copied out now, in a later submission than the one that
        // wrote them: on Metal, a copy in the same command buffer often ran before the frame's
        // last timestamp was written (it read the slot's previous value, or zero).
        for slot in 0..SLOTS {
            if matches!(self.slots[slot], Slot::Ended) {
                let first = slot as u32 * POINTS;
                let offset = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT * slot as u64;
                encoder.resolve_query_set(
                    &self.queries,
                    first..first + POINTS,
                    &self.resolve,
                    offset,
                );
                encoder.copy_buffer_to_buffer(
                    &self.resolve,
                    offset,
                    &self.readback[slot],
                    0,
                    BYTES,
                );
                self.slots[slot] = Slot::Copying;
            }
        }
        let Some(slot) = self.slots.iter().position(|s| matches!(s, Slot::Free)) else {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            self.current = None;
            return;
        };
        self.slots[slot] = Slot::Recording;
        self.current = Some(slot);
        LAST_SIZE.store(
            (u64::from(size.0) << 32) | u64::from(size.1),
            Ordering::Relaxed,
        );
        LAST_SAMPLES.store(u64::from(samples), Ordering::Relaxed);
        self.write(encoder, slot as u32 * POINTS);
    }

    /// The end of the cacheAsBitmap and filter redraws.
    pub fn mark_cache_done(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(slot) = self.current {
            self.write(encoder, slot as u32 * POINTS + 1);
        }
    }

    /// The frame's last timestamp.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(slot) = self.current.take() else {
            return;
        };
        self.write(encoder, slot as u32 * POINTS + 2);
        self.slots[slot] = Slot::Ended;
    }

    /// After the frame was submitted: asks for the timestamps it copied out (the last frame's).
    pub fn submitted(&mut self) {
        for slot in 0..SLOTS {
            if !matches!(self.slots[slot], Slot::Copying) {
                continue;
            }
            let (done, ok) = (
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(false)),
            );
            let (done2, ok2) = (done.clone(), ok.clone());
            self.readback[slot]
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    ok2.store(result.is_ok(), Ordering::Release);
                    done2.store(true, Ordering::Release);
                });
            self.slots[slot] = Slot::Reading { done, ok };
        }
    }

    /// Reads the frames whose timestamps have arrived, without waiting for the others.
    fn collect(&mut self, descriptors: &Descriptors) {
        if self.slots.iter().any(|s| matches!(s, Slot::Reading { .. })) {
            let _ = descriptors.device.poll(wgpu::PollType::Poll);
        }
        for (slot, state) in self.slots.iter_mut().enumerate() {
            let Slot::Reading { done, ok } = state else {
                continue;
            };
            if !done.load(Ordering::Acquire) {
                continue;
            }
            if ok.load(Ordering::Acquire) {
                let buffer = &self.readback[slot];
                if let Ok(bytes) = buffer.slice(..).get_mapped_range() {
                    let mut ticks = [0u64; POINTS as usize];
                    for (i, tick) in ticks.iter_mut().enumerate() {
                        let mut word = [0u8; 8];
                        word.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                        *tick = u64::from_le_bytes(word);
                    }
                    record(ticks, self.period_ns);
                }
                buffer.unmap();
            }
            *state = Slot::Free;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_the_two_spans_and_rejects_timestamps_out_of_order() {
        // Not run in parallel with anything else touching the totals (only this test does).
        let _ = (
            FRAMES.swap(0, Ordering::Relaxed),
            CACHE_NS.swap(0, Ordering::Relaxed),
            STAGE_NS.swap(0, Ordering::Relaxed),
            MAX_NS.swap(0, Ordering::Relaxed),
            INVALID.swap(0, Ordering::Relaxed),
        );
        record([1000, 1500, 4000], 2.0); // 1 us of redraws, 5 us of stage
        record([1000, 900, 4000], 2.0); // out of order
        record([0, 0, 0], 2.0); // never written
        assert_eq!(FRAMES.load(Ordering::Relaxed), 1);
        assert_eq!(CACHE_NS.load(Ordering::Relaxed), 1000);
        assert_eq!(STAGE_NS.load(Ordering::Relaxed), 5000);
        assert_eq!(MAX_NS.load(Ordering::Relaxed), 6000);
        assert_eq!(INVALID.load(Ordering::Relaxed), 2);
    }
}
