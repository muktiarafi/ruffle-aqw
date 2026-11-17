use bytemuck::{AnyBitPattern, NoUninit};
use std::ops::Range;
use wgpu::BufferAddress;
use wgpu::util::DeviceExt;

pub struct BufferBuilder {
    inner: Vec<u8>,
    align_mask: usize,
    limit: u64,
}

#[derive(Debug, Copy, Clone)]
pub struct BufferFull;

impl BufferBuilder {
    pub fn new(limits: &wgpu::Limits, alignment: u32) -> Self {
        Self {
            inner: Vec::new(),
            align_mask: if alignment > 0 {
                assert!(alignment.is_power_of_two());
                (alignment - 1) as usize
            } else {
                0
            },
            limit: limits.max_buffer_size,
        }
    }

    pub fn set_buffer_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    pub fn add<T: NoUninit + AnyBitPattern>(
        &mut self,
        value: &[T],
    ) -> Result<Range<wgpu::BufferAddress>, BufferFull> {
        let start_pos = if !self.inner.is_empty() {
            if self.align_mask > 0 {
                // Pad the internal buffer to match alignment requirements
                // Pad on creation so that we don't wastefully pad the end of the buffer
                (self.inner.len() + self.align_mask) & !self.align_mask
            } else {
                self.inner.len()
            }
        } else {
            0
        };

        let slice = bytemuck::cast_slice(value);
        if (start_pos + slice.len()) as u64 > self.limit {
            return Err(BufferFull);
        }

        if start_pos > 0 && self.align_mask > 0 {
            self.inner.resize(start_pos, 0);
        }

        self.inner.extend_from_slice(slice);
        Ok((start_pos as wgpu::BufferAddress)..(self.inner.len() as wgpu::BufferAddress))
    }

    /// AQW build: `add`, starting at a multiple of the element size (vertices: see
    /// `CommandRenderer::render_shape`).
    pub fn add_strided<T: NoUninit + AnyBitPattern>(
        &mut self,
        value: &[T],
    ) -> Result<Range<wgpu::BufferAddress>, BufferFull> {
        let start = self.inner.len().next_multiple_of(std::mem::size_of::<T>().max(1));
        if (start + std::mem::size_of_val(value)) as u64 > self.limit {
            return Err(BufferFull);
        }
        self.inner.resize(start, 0);
        self.inner.extend_from_slice(bytemuck::cast_slice(value));
        Ok((start as wgpu::BufferAddress)..(self.inner.len() as wgpu::BufferAddress))
    }

    /// AQW build: the bytes added so far (see `crate::arena`).
    pub fn bytes(&self) -> &[u8] {
        &self.inner
    }

    pub fn finish(
        self,
        device: &wgpu::Device,
        label: Option<String>,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.as_deref(),
            contents: &self.inner,
            usage,
        })
    }

    pub fn copy_to(
        self,
        staging_belt: &mut wgpu::util::StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
    ) {
        if let Some(length) = wgpu::BufferSize::new(self.inner.len() as u64) {
            let mut view =
                staging_belt.write_buffer(encoder, buffer, BufferAddress::default(), length);
            view.copy_from_slice(&self.inner);
        }
    }
}
