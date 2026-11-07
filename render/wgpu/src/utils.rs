use crate::buffer_pool::BufferDescription;
use crate::descriptors::Descriptors;
use crate::globals::Globals;
use std::borrow::Cow;
use wgpu::{CommandEncoder, TextureFormat};

macro_rules! create_debug_label {
    ($($arg:tt)*) => (
        if cfg!(feature = "render_debug_labels") {
            Some(format!($($arg)*))
        } else {
            None
        }
    )
}

pub fn remove_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnorm,
        wgpu::TextureFormat::Bc2RgbaUnormSrgb => wgpu::TextureFormat::Bc2RgbaUnorm,
        wgpu::TextureFormat::Bc3RgbaUnormSrgb => wgpu::TextureFormat::Bc3RgbaUnorm,
        wgpu::TextureFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnorm,
        wgpu::TextureFormat::Etc2Rgb8UnormSrgb => wgpu::TextureFormat::Etc2Rgb8Unorm,
        wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb => wgpu::TextureFormat::Etc2Rgb8A1Unorm,
        wgpu::TextureFormat::Etc2Rgba8UnormSrgb => wgpu::TextureFormat::Etc2Rgba8Unorm,
        wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::UnormSrgb,
        } => wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::Unorm,
        },
        _ => format,
    }
}

pub fn format_list<'a>(values: &[&'a str], connector: &'a str) -> Cow<'a, str> {
    match values.len() {
        0 => Cow::Borrowed(""),
        1 => Cow::Borrowed(values[0]),
        _ => Cow::Owned(
            values[0..values.len() - 1].join(", ")
                + " "
                + connector
                + " "
                + values[values.len() - 1],
        ),
    }
}

pub fn get_backend_names(backends: wgpu::Backends) -> Vec<&'static str> {
    let mut names = Vec::new();

    if backends.contains(wgpu::Backends::VULKAN) {
        names.push("Vulkan");
    }
    if backends.contains(wgpu::Backends::DX12) {
        names.push("DirectX 12");
    }
    if backends.contains(wgpu::Backends::METAL) {
        names.push("Metal");
    }
    if backends.contains(wgpu::Backends::GL) {
        names.push("Open GL");
    }
    if backends.contains(wgpu::Backends::BROWSER_WEBGPU) {
        names.push("Web GPU");
    }

    names
}

pub fn create_buffer_with_data(
    device: &wgpu::Device,
    data: &[u8],
    usage: wgpu::BufferUsages,
    label: Option<String>,
) -> wgpu::Buffer {
    create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            usage,
            label: label.as_deref(),
            contents: data,
        },
    )
}

/// AQW build: `wgpu::util::DeviceExt::create_buffer_init`, without its panic when the GPU is out
/// of memory. wgpu's version maps the new buffer and `expect`s the mapping, but a buffer the GPU
/// couldn't allocate is invalid and can't be mapped, so out-of-memory panicked there ("Failed to
/// get mapped range for buffer created with mapped_at_creation"; phones with Mali GPUs, from
/// `Globals::new`) before `gpu_oom`'s handler could tolerate it. Here an invalid buffer is
/// returned unfilled: like any object that failed to allocate, using it only gives validation
/// errors, which `gpu_oom` tolerates after an out-of-memory error.
pub fn create_buffer_init(
    device: &wgpu::Device,
    descriptor: &wgpu::util::BufferInitDescriptor,
) -> wgpu::Buffer {
    // As wgpu's: a mapped buffer's size must be a multiple of COPY_BUFFER_ALIGNMENT, and above 0.
    let unpadded_size = descriptor.contents.len() as wgpu::BufferAddress;
    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    let padded_size = ((unpadded_size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: descriptor.label,
        size: padded_size,
        usage: descriptor.usage,
        mapped_at_creation: true,
    });
    if let Ok(mut range) = buffer.get_mapped_range_mut(..) {
        range
            .slice(..unpadded_size as usize)
            .copy_from_slice(descriptor.contents);
        drop(range);
        buffer.unmap();
    }
    buffer
}

/// AQW build: `StagingBelt::write_buffer`, without its panic. wgpu's version `expect`s the
/// staging buffer's mapping, but a staging buffer the GPU couldn't allocate is invalid (and one
/// has also come back "destroyed"), so phones out of GPU memory crashed there (Redmi 10 2022,
/// Infinix HOT 11S, from `BufferArena::alloc_at_multiple`). Here the copy into `target` is only
/// recorded once the staging space is mapped; `None` when it couldn't be, as an out-of-memory
/// error (`gpu_oom` then tolerates what follows).
pub fn try_staging_write(
    staging_belt: &mut wgpu::util::StagingBelt,
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    size: wgpu::BufferSize,
) -> Option<wgpu::BufferViewMut> {
    let staged = staging_belt.allocate(
        size,
        wgpu::BufferSize::new(wgpu::COPY_BUFFER_ALIGNMENT).expect("not 0"),
    );
    let Ok(view) = staged.get_mapped_range_mut() else {
        // At most one line a few seconds: a frame out of memory skips hundreds of writes, and
        // on Android each warning is a Sentry breadcrumb (a burst filled the last 100).
        static SKIPPED: std::sync::Mutex<(Option<std::time::Instant>, u32)> =
            std::sync::Mutex::new((None, 0));
        if let Ok(mut skipped) = SKIPPED.lock() {
            skipped.1 += 1;
            if skipped
                .0
                .is_none_or(|last| last.elapsed() >= std::time::Duration::from_secs(5))
            {
                tracing::warn!(
                    target: "ruffle_aqw_perf",
                    "GPU staging buffer unavailable ({} writes skipped since the last report, this one {size} bytes)",
                    skipped.1
                );
                *skipped = (Some(std::time::Instant::now()), 0);
            }
        }
        crate::gpu_oom::note();
        return None;
    };
    encoder.copy_buffer_to_buffer(staged.buffer(), staged.offset(), target, offset, size.get());
    Some(view)
}

/// AQW build: [`try_staging_write`] for data written again every frame (uniforms, instances,
/// the frame's transforms): when the staging space couldn't be had, the write is skipped and
/// that frame may draw wrong, instead of a crash.
pub fn staging_write(
    staging_belt: &mut wgpu::util::StagingBelt,
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    size: wgpu::BufferSize,
) -> StagingWrite {
    StagingWrite(try_staging_write(
        staging_belt,
        encoder,
        target,
        offset,
        size,
    ))
}

/// A write into staging space, or nowhere when there was none (see [`staging_write`]).
pub struct StagingWrite(Option<wgpu::BufferViewMut>);

impl StagingWrite {
    pub fn copy_from_slice(&mut self, data: &[u8]) {
        if let Some(view) = &mut self.0 {
            view.copy_from_slice(data);
        }
    }
}

// Based off wgpu example 'capture'
#[derive(Debug, Clone)]
pub struct BufferDimensions {
    pub width: usize,
    pub height: usize,
    pub unpadded_bytes_per_row: usize,
    pub padded_bytes_per_row: u32,
}

impl BufferDimensions {
    pub fn new(width: usize, height: usize, format: TextureFormat) -> Self {
        let bytes_per_pixel = format.block_copy_size(None).unwrap() as usize;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = (unpadded_bytes_per_row + padded_bytes_per_row_padding) as u32;

        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }

    pub fn size(&self) -> u64 {
        self.padded_bytes_per_row as u64 * self.height as u64
    }
}

impl BufferDescription for BufferDimensions {
    type Cost = u64;

    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost> {
        if self.size() <= other.size() {
            Some(other.size() - self.size())
        } else {
            None
        }
    }
}

pub fn capture_image<R, F: FnOnce(&[u8], u32) -> R>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    dimensions: &BufferDimensions,
    index: Option<wgpu::SubmissionIndex>,
    with_rgba: F,
) -> R {
    // AQW build: an image that can't be read back reads as transparent black instead of
    // panicking: once the GPU dropped the device (`crate::device_lost`), the copy's submission
    // failed, so waiting for it gave `WrongSubmissionIndex`, and its mapping may never finish
    // (waiting for it would hang the player).
    let empty = |with_rgba: F| {
        let empty = vec![0; dimensions.padded_bytes_per_row as usize * dimensions.height];
        with_rgba(&empty, dimensions.padded_bytes_per_row)
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        // Nobody listens any more when the read gave up.
        let _ = sender.send(result);
    });
    let polled = device.poll(wgpu::PollType::Wait {
        submission_index: index,
        timeout: None,
    });
    if polled.is_err() || crate::device_lost().is_some() {
        return empty(with_rgba);
    }
    let mapped = match receiver.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => buffer_slice.get_mapped_range().ok(),
        _ => None,
    };
    let Some(map) = mapped else {
        return empty(with_rgba);
    };
    let result = with_rgba(&map, dimensions.padded_bytes_per_row);
    drop(map);
    buffer.unmap();
    result
}

#[cfg(not(target_family = "wasm"))]
pub fn buffer_to_image(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    dimensions: &BufferDimensions,
    index: Option<wgpu::SubmissionIndex>,
    size: wgpu::Extent3d,
) -> image::RgbaImage {
    capture_image(device, buffer, dimensions, index, |rgba, _buffer_width| {
        let mut bytes = Vec::with_capacity(dimensions.height * dimensions.unpadded_bytes_per_row);

        for chunk in rgba.chunks(dimensions.padded_bytes_per_row as usize) {
            bytes.extend_from_slice(&chunk[..dimensions.unpadded_bytes_per_row]);
        }

        // The image copied from the GPU uses premultiplied alpha, so
        // convert to straight alpha if requested by the user.
        ruffle_render::utils::unmultiply_alpha_rgba(&mut bytes);

        image::RgbaImage::from_raw(size.width, size.height, bytes)
            .expect("Retrieved texture buffer must be a valid RgbaImage")
    })
}

pub fn supported_sample_count(
    adapter: &wgpu::Adapter,
    mut sample_count: u32,
    format: wgpu::TextureFormat,
) -> u32 {
    let features = adapter.get_texture_format_features(format).flags;

    // Keep halving the sample count until we get one that's supported - or 1 (no multisampling)
    // It's not guaranteed that supporting 4x means supporting 2x, so there's no "max" option
    // And it's probably safer to round down than up, given it's a performance setting.
    while sample_count > 1 && !features.sample_count_supported(sample_count) {
        sample_count /= 2;
    }
    sample_count
}

pub fn run_copy_pipeline(
    descriptors: &Descriptors,
    format: wgpu::TextureFormat,
    frame_view: &wgpu::TextureView,
    input: &wgpu::TextureView,
    globals: &Globals,
    sample_count: u32,
    encoder: &mut CommandEncoder,
) {
    let copy_bind_group = descriptors
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &descriptors.bind_layouts.bitmap,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        descriptors.bitmap_samplers.get_sampler(false, false),
                    ),
                },
            ],
            label: create_debug_label!("Copy bind group").as_deref(),
        });

    let pipeline = descriptors.copy_pipeline(format, sample_count);

    // We overwrite the pixels in the target texture (no blending at all),
    // so this doesn't matter.
    let load = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);

    crate::perf_counters::inc(crate::perf_counters::RENDER_PASS);
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: create_debug_label!("Copy back to render target").as_deref(),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: frame_view,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            resolve_target: None,
            depth_slice: None,
        })],
        ..Default::default()
    });

    render_pass.set_pipeline(&pipeline);
    render_pass.set_bind_group(0, globals.bind_group(), &[]);

    render_pass.set_bind_group(2, &copy_bind_group, &[]);

    render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos_uv.slice(..));
    render_pass.set_index_buffer(
        descriptors.quad.indices.slice(..),
        wgpu::IndexFormat::Uint32,
    );

    render_pass.draw_indexed(0..6, 0, 0..1);
    drop(render_pass);
}

#[derive(Debug)]
pub struct SampleCountMap<T> {
    one: T,
    two: T,
    four: T,
    eight: T,
    sixteen: T,
}

impl<T: Default> Default for SampleCountMap<T> {
    fn default() -> Self {
        SampleCountMap {
            one: Default::default(),
            two: Default::default(),
            four: Default::default(),
            eight: Default::default(),
            sixteen: Default::default(),
        }
    }
}

impl<T> SampleCountMap<T> {
    pub fn get(&self, sample_count: u32) -> &T {
        match sample_count {
            1 => &self.one,
            2 => &self.two,
            4 => &self.four,
            8 => &self.eight,
            16 => &self.sixteen,
            _ => unreachable!("Sample counts must be powers of two between 1..=16"),
        }
    }
}

impl<T> SampleCountMap<std::sync::OnceLock<T>> {
    pub fn get_or_init<F>(&self, sample_count: u32, init: F) -> &T
    where
        F: FnOnce() -> T,
    {
        match sample_count {
            1 => self.one.get_or_init(init),
            2 => self.two.get_or_init(init),
            4 => self.four.get_or_init(init),
            8 => self.eight.get_or_init(init),
            16 => self.sixteen.get_or_init(init),
            _ => unreachable!("Sample counts must be powers of two between 1..=16"),
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::create_buffer_init;

    /// A device on this machine's GPU (or software adapter), `None` when there's none.
    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter =
            futures::executor::block_on(instance.request_adapter(&Default::default())).ok()?;
        futures::executor::block_on(adapter.request_device(&Default::default())).ok()
    }

    #[test]
    fn create_buffer_init_fills_the_buffer() {
        let Some((device, _queue)) = device() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        // 5 bytes: padded to 8, as wgpu's own does.
        let buffer = create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[1, 2, 3, 4, 5],
                usage: wgpu::BufferUsages::MAP_READ,
            },
        );
        assert_eq!(buffer.size(), 8);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("mapped");
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polled");
        let data = buffer.get_mapped_range(..).expect("mapped range").to_vec();
        assert_eq!(data, [1, 2, 3, 4, 5, 0, 0, 0]);
    }

    #[test]
    fn try_staging_write_copies_into_the_target() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut belt = wgpu::util::StagingBelt::new(device.clone(), 64);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut view = super::try_staging_write(
            &mut belt,
            &mut encoder,
            &target,
            0,
            wgpu::BufferSize::new(8).unwrap(),
        )
        .expect("staged");
        view.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        drop(view);
        belt.finish();
        queue.submit([encoder.finish()]);
        target.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("mapped");
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("polled");
        let data = target.get_mapped_range(..).expect("mapped range").to_vec();
        assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// A staging buffer the GPU couldn't create (out of memory on a phone; here larger than any
    /// buffer may be) gives `None`, not a panic as with wgpu's `StagingBelt::write_buffer`.
    #[test]
    fn try_staging_write_survives_a_staging_buffer_that_failed() {
        let Some((device, _queue)) = device() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let too_big = device.limits().max_buffer_size.next_multiple_of(4) + 4;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut belt = wgpu::util::StagingBelt::new(device.clone(), 64);
        let mut encoder = device.create_command_encoder(&Default::default());
        let view = super::try_staging_write(
            &mut belt,
            &mut encoder,
            &target,
            0,
            wgpu::BufferSize::new(too_big).unwrap(),
        );
        assert!(view.is_none());
        let error = futures::executor::block_on(scope.pop());
        assert!(error.is_some(), "the staging buffer should have failed");
    }

    /// A buffer the GPU couldn't create (out of memory on a phone; here an invalid usage) is
    /// returned unfilled, not a panic as with wgpu's `create_buffer_init`.
    #[test]
    fn create_buffer_init_survives_a_buffer_that_failed() {
        let Some((device, _queue)) = device() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _buffer = create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0; 64],
                // MAP_READ only goes with COPY_DST (without MAPPABLE_PRIMARY_BUFFERS).
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::UNIFORM,
            },
        );
        let error = futures::executor::block_on(scope.pop());
        assert!(error.is_some(), "the buffer should have failed");
    }

    /// On a lost device, polling doesn't panic (the wgpu-core patch) and reading a texture back
    /// gives transparent black, not a panic.
    #[test]
    fn a_lost_device_neither_polls_nor_reads_back_with_a_panic() {
        let Some((device, _queue)) = device() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        // The work sent to the dead device reports errors; wgpu's default handler panics.
        device.on_uncaptured_error(std::sync::Arc::new(|_| {}));
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        device.destroy();
        assert!(device.poll(wgpu::PollType::Poll).is_ok());
        assert!(device.poll(wgpu::PollType::wait_indefinitely()).is_ok());
        let dimensions = super::BufferDimensions::new(2, 2, wgpu::TextureFormat::Rgba8Unorm);
        let read =
            super::capture_image(&device, &buffer, &dimensions, None, |rgba, _| rgba.to_vec());
        assert_eq!(read.len(), dimensions.padded_bytes_per_row as usize * 2);
        assert!(read.iter().all(|&byte| byte == 0));
    }
}
