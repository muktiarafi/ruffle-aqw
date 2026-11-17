//! AQW build: shape geometry and uniforms packed into a few large shared GPU buffers.
//!
//! Stock Ruffle gives every shape its own vertex, index and uniform buffer. AQW keeps every
//! gear SWF it has loaded, so a long session reached ~160,000 tiny buffers: each a separate
//! driver object with its own overhead, scattered over GPU memory blocks (7.4 GB reserved for
//! 2.5 GB used). Here shapes get slices of large chunks instead, and give them back when freed.

use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

/// A slice of an arena chunk, returned to the arena when dropped.
#[derive(Debug)]
pub struct ArenaSlice {
    buffer: wgpu::Buffer,
    /// What was allocated (returned when dropped), and where the data starts in it.
    range: Range<wgpu::BufferAddress>,
    data_start: wgpu::BufferAddress,
    chunk: usize,
    arena: Weak<Mutex<ArenaInner>>,
}

impl ArenaSlice {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Where this slice's data starts in `buffer()`.
    pub fn offset(&self) -> wgpu::BufferAddress {
        self.data_start
    }

    pub fn size(&self) -> wgpu::BufferAddress {
        self.range.end - self.data_start
    }
}

impl Drop for ArenaSlice {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.upgrade()
            && let Ok(mut arena) = arena.lock()
        {
            // Commands recorded (or submitted but unfinished) may still read this range, so
            // it is only reused once that work is done (see `BufferArena::end_frame`).
            let frame = arena.frame;
            arena.pending.push((frame, self.chunk, self.range.clone()));
        }
    }
}

/// The GPU had no memory left for a new arena chunk.
#[derive(Debug)]
pub struct ArenaOutOfMemory;

/// Frames a freed range waits before reuse: the frame it was freed in may still be recorded
/// or submitted, plus the frames the GPU may still be working on (`MAX_FRAMES_IN_FLIGHT`).
const REUSE_DELAY_FRAMES: u64 = 4;

#[derive(Debug)]
struct Chunk {
    buffer: wgpu::Buffer,
    /// Free ranges, sorted and coalesced.
    free: Vec<Range<wgpu::BufferAddress>>,
    used: wgpu::BufferAddress,
    /// Made for a single allocation larger than a normal chunk; dropped when that is freed.
    dedicated: bool,
}

#[derive(Debug)]
struct ArenaInner {
    label: &'static str,
    usage: wgpu::BufferUsages,
    alignment: wgpu::BufferAddress,
    chunk_size: wgpu::BufferAddress,
    chunks: Vec<Option<Chunk>>,
    /// Freed ranges waiting for the GPU: (frame freed in, chunk, range).
    pending: Vec<(u64, usize, Range<wgpu::BufferAddress>)>,
    frame: u64,
    /// Debug builds: every allocated range, to catch overlaps.
    #[cfg(debug_assertions)]
    live: std::collections::BTreeMap<(usize, wgpu::BufferAddress), wgpu::BufferAddress>,
}

impl ArenaInner {
    fn release(&mut self, chunk_index: usize, range: Range<wgpu::BufferAddress>) {
        #[cfg(debug_assertions)]
        assert_eq!(
            self.live.remove(&(chunk_index, range.start)),
            Some(range.end),
            "arena: releasing a range that isn't allocated"
        );
        let Some(chunk) = self.chunks.get_mut(chunk_index).and_then(Option::as_mut) else {
            return;
        };
        chunk.used -= range.end - range.start;
        let at = chunk.free.partition_point(|r| r.start < range.start);
        chunk.free.insert(at, range);
        // Merge with the neighbours.
        if at + 1 < chunk.free.len() && chunk.free[at].end == chunk.free[at + 1].start {
            chunk.free[at].end = chunk.free[at + 1].end;
            chunk.free.remove(at + 1);
        }
        if at > 0 && chunk.free[at - 1].end == chunk.free[at].start {
            chunk.free[at - 1].end = chunk.free[at].end;
            chunk.free.remove(at);
        }
        if chunk.used == 0 {
            let other_shared_chunks = self
                .chunks
                .iter()
                .enumerate()
                .filter(|(i, c)| *i != chunk_index && c.as_ref().is_some_and(|c| !c.dedicated))
                .count();
            // Free empty chunks, but keep one shared chunk around for new shapes.
            if self.chunks[chunk_index].as_ref().is_some_and(|c| c.dedicated)
                || other_shared_chunks > 0
            {
                self.chunks[chunk_index] = None;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferArena {
    inner: Arc<Mutex<ArenaInner>>,
}

impl BufferArena {
    pub fn new(
        label: &'static str,
        usage: wgpu::BufferUsages,
        alignment: wgpu::BufferAddress,
        chunk_size: wgpu::BufferAddress,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ArenaInner {
                label,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                alignment: alignment.max(wgpu::COPY_BUFFER_ALIGNMENT),
                chunk_size,
                chunks: Vec::new(),
                pending: Vec::new(),
                frame: 0,
                #[cfg(debug_assertions)]
                live: Default::default(),
            })),
        }
    }

    /// Reserves room for `data` and records its upload into `encoder` (through
    /// `staging_belt`), so it is ordered with the draws recorded around it. (`queue.write_buffer`
    /// runs before the whole next submission, and a shared buffer is also read by commands
    /// recorded earlier: those reads and the write raced.) `None` for empty data; an error
    /// if a new chunk was needed and the GPU is out of memory.
    pub fn alloc(
        &self,
        device: &wgpu::Device,
        staging_belt: &mut wgpu::util::StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
        data: &[u8],
    ) -> Result<Option<ArenaSlice>, ArenaOutOfMemory> {
        self.alloc_at_multiple(device, staging_belt, encoder, data, 1)
    }

    /// `alloc`, with the data starting at a multiple of `multiple` bytes in the buffer (for
    /// vertices: draws can then share one binding of the whole buffer, see
    /// `CommandRenderer::render_shape`).
    pub fn alloc_at_multiple(
        &self,
        device: &wgpu::Device,
        staging_belt: &mut wgpu::util::StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
        data: &[u8],
        multiple: wgpu::BufferAddress,
    ) -> Result<Option<ArenaSlice>, ArenaOutOfMemory> {
        if data.is_empty() {
            return Ok(None);
        }
        let mut inner = self.inner.lock().expect("arena lock");
        let size = (data.len() as wgpu::BufferAddress + multiple - 1)
            .next_multiple_of(inner.alignment);

        // First fit in an existing shared chunk. Every allocation is a multiple of the
        // alignment, so free ranges always start aligned.
        let found = inner.chunks.iter().enumerate().find_map(|(index, chunk)| {
            let chunk = chunk.as_ref()?;
            if chunk.dedicated {
                return None;
            }
            let at = chunk.free.iter().position(|r| r.end - r.start >= size)?;
            Some((index, at))
        });
        let (chunk_index, start) = match found {
            Some((index, at)) => {
                let chunk = inner.chunks[index].as_mut().expect("found above");
                let start = chunk.free[at].start;
                chunk.free[at].start += size;
                if chunk.free[at].start == chunk.free[at].end {
                    chunk.free.remove(at);
                }
                chunk.used += size;
                (index, start)
            }
            None => {
                let dedicated = size > inner.chunk_size / 2;
                let chunk_bytes = if dedicated { size } else { inner.chunk_size };
                let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(inner.label),
                    size: chunk_bytes,
                    usage: inner.usage,
                    mapped_at_creation: false,
                });
                if crate::aqw_out_of_memory(scope) {
                    return Err(ArenaOutOfMemory);
                }
                let chunk = Chunk {
                    buffer,
                    free: if dedicated {
                        vec![]
                    } else {
                        vec![size..chunk_bytes]
                    },
                    used: size,
                    dedicated,
                };
                let index = match inner.chunks.iter().position(Option::is_none) {
                    Some(index) => {
                        inner.chunks[index] = Some(chunk);
                        index
                    }
                    None => {
                        inner.chunks.push(Some(chunk));
                        inner.chunks.len() - 1
                    }
                };
                (index, 0)
            }
        };
        #[cfg(debug_assertions)]
        {
            let overlap = inner
                .live
                .range((chunk_index, 0)..(chunk_index, start + size))
                .any(|(&(_, s), &e)| s < start + size && start < e);
            assert!(!overlap, "arena: overlapping allocation at {chunk_index}:{start}+{size}");
            inner.live.insert((chunk_index, start), start + size);
        }
        let buffer = inner.chunks[chunk_index]
            .as_ref()
            .expect("chunk exists")
            .buffer
            .clone();
        drop(inner);

        let data_start = start.next_multiple_of(multiple);
        let mut view = staging_belt.write_buffer(
            encoder,
            &buffer,
            start,
            wgpu::BufferSize::new(size).expect("data is not empty"),
        );
        if data.len() as wgpu::BufferAddress == size {
            view.copy_from_slice(data);
        } else {
            let mut padded = vec![0; (data_start - start) as usize];
            padded.extend_from_slice(data);
            padded.resize(size as usize, 0);
            view.copy_from_slice(&padded);
        }
        drop(view);
        Ok(Some(ArenaSlice {
            buffer,
            range: start..start + size,
            data_start,
            chunk: chunk_index,
            arena: Arc::downgrade(&self.inner),
        }))
    }

    /// Call once per rendered frame (after submitting it): ranges freed long enough ago
    /// become reusable, and empty chunks are released.
    pub fn end_frame(&self) {
        let mut inner = self.inner.lock().expect("arena lock");
        inner.frame += 1;
        let frame = inner.frame;
        let ready: Vec<_> = {
            let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut inner.pending)
                .into_iter()
                .partition(|(freed, _, _)| freed + REUSE_DELAY_FRAMES <= frame);
            inner.pending = waiting;
            ready
        };
        for (_, chunk, range) in ready {
            inner.release(chunk, range);
        }
    }

    /// (chunks, reserved bytes, used bytes), for the perf log.
    pub fn stats(&self) -> (usize, u64, u64) {
        let inner = self.inner.lock().expect("arena lock");
        let chunks = inner.chunks.iter().flatten();
        let (mut count, mut reserved, mut used) = (0, 0, 0);
        for chunk in chunks {
            count += 1;
            reserved += chunk.buffer.size();
            used += chunk.used;
        }
        (count, reserved, used)
    }
}
