//! AQW build: recorded command lists.
//!
//! wgpu-core encodes every render pass into its own command buffer (plus one for the barriers
//! before it), and Ruffle draws thousands of passes per frame in crowded AQW scenes. On D3D12
//! each of those used to be a real `ID3D12GraphicsCommandList`: a `Reset` and a `Close` in the
//! driver per pass (~60% of render pass encoding), and with a few frames in flight ~20k live
//! lists (creating them caused >1 s stutters, and they held gigabytes of driver memory).
//!
//! Instead, the encoder records the calls it would make into a [`Recorder`] (cheap, no driver
//! work), and `CommandEncoder::merge_command_buffers` replays a run of recorded buffers, in
//! order, into one real command list. The GPU gets the same calls in the same order; only
//! redundant `SetDescriptorHeaps` calls between passes are skipped.
//!
//! The method names and argument shapes mirror `ID3D12GraphicsCommandList`, so the encoder's
//! code calls a `Recorder` exactly as it called the list. Resources referenced by raw pointer
//! (barriers, texture copy locations) are kept alive by wgpu-core until the submission is done,
//! as they were for the real list; other objects are cloned (AddRef).
#![allow(non_snake_case)]

use alloc::vec::Vec;
use core::{cell::UnsafeCell, ffi, mem::ManuallyDrop, ptr};

use windows::core::Interface as _;
use windows::Win32::{
    Foundation,
    Graphics::{Direct3D, Direct3D12, Dxgi},
};

pub(super) enum Op {
    BeginEvent(Vec<u8>),
    EndEvent,
    SetMarker(Vec<u8>),
    SetDescriptorHeaps(Vec<Option<Direct3D12::ID3D12DescriptorHeap>>),
    VertexBuffers(u32, Vec<Direct3D12::D3D12_VERTEX_BUFFER_VIEW>),
    GraphicsConstant(u32, u32, u32),
    ComputeConstant(u32, u32, u32),
    GraphicsTable(u32, Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE),
    ComputeTable(u32, Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE),
    GraphicsCbv(u32, u64),
    ComputeCbv(u32, u64),
    /// Bitwise copies: they don't own their resource references.
    Barriers(Vec<ManuallyDrop<Direct3D12::D3D12_RESOURCE_BARRIER>>),
    CopyBufferRegion {
        dst: Direct3D12::ID3D12Resource,
        dst_offset: u64,
        src: Direct3D12::ID3D12Resource,
        src_offset: u64,
        size: u64,
    },
    /// Bitwise copies of the locations: they don't own their resource references.
    CopyTextureRegion {
        dst: ManuallyDrop<Direct3D12::D3D12_TEXTURE_COPY_LOCATION>,
        x: u32,
        y: u32,
        z: u32,
        src: ManuallyDrop<Direct3D12::D3D12_TEXTURE_COPY_LOCATION>,
        src_box: Option<Direct3D12::D3D12_BOX>,
    },
    BeginQuery(Direct3D12::ID3D12QueryHeap, Direct3D12::D3D12_QUERY_TYPE, u32),
    EndQuery(Direct3D12::ID3D12QueryHeap, Direct3D12::D3D12_QUERY_TYPE, u32),
    ResolveQueryData {
        heap: Direct3D12::ID3D12QueryHeap,
        ty: Direct3D12::D3D12_QUERY_TYPE,
        start: u32,
        count: u32,
        dst: Direct3D12::ID3D12Resource,
        offset: u64,
    },
    RenderTargets {
        count: u32,
        rtvs: Vec<Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
        single_range: bool,
        dsv: Option<Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    },
    ClearRtv(
        Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
        [f32; 4],
        Option<Vec<Foundation::RECT>>,
    ),
    ClearDsv(
        Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
        Direct3D12::D3D12_CLEAR_FLAGS,
        f32,
        u8,
        Option<Vec<Foundation::RECT>>,
    ),
    Viewports(Vec<Direct3D12::D3D12_VIEWPORT>),
    Scissors(Vec<Foundation::RECT>),
    ResolveSubresource {
        dst: Direct3D12::ID3D12Resource,
        dst_sub: u32,
        src: Direct3D12::ID3D12Resource,
        src_sub: u32,
        format: Dxgi::Common::DXGI_FORMAT,
    },
    GraphicsRootSignature(Option<Direct3D12::ID3D12RootSignature>),
    ComputeRootSignature(Option<Direct3D12::ID3D12RootSignature>),
    PipelineState(Direct3D12::ID3D12PipelineState),
    Topology(Direct3D::D3D_PRIMITIVE_TOPOLOGY),
    IndexBuffer(Option<Direct3D12::D3D12_INDEX_BUFFER_VIEW>),
    StencilRef(u32),
    BlendFactor(Option<[f32; 4]>),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    Dispatch(u32, u32, u32),
    ExecuteIndirect {
        signature: Direct3D12::ID3D12CommandSignature,
        max_count: u32,
        args: Direct3D12::ID3D12Resource,
        args_offset: u64,
        count: Option<Direct3D12::ID3D12Resource>,
        count_offset: u64,
    },
}

/// Records the calls the encoder makes, in order (see the module docs).
pub(super) struct Recorder {
    ops: UnsafeCell<Vec<Op>>,
}

// SAFETY: a `Recorder` is only used through its `CommandEncoder`, which hal only uses from
// one thread at a time (`&mut self`); the COM objects it holds are free-threaded.
unsafe impl Send for Recorder {}
unsafe impl Sync for Recorder {}
unsafe impl Send for Op {}
unsafe impl Sync for Op {}

fn marker_bytes(data: Option<*const ffi::c_void>, size: u32) -> Vec<u8> {
    match data {
        // SAFETY: the caller passes `size` readable bytes, as for the real call.
        Some(data) if size > 0 => unsafe {
            core::slice::from_raw_parts(data.cast::<u8>(), size as usize).to_vec()
        },
        _ => Vec::new(),
    }
}

impl Recorder {
    pub(super) fn new(ops: Vec<Op>) -> Self {
        Self {
            ops: UnsafeCell::new(ops),
        }
    }

    pub(super) fn into_ops(self) -> Vec<Op> {
        self.ops.into_inner()
    }

    #[inline]
    fn push(&self, op: Op) {
        // SAFETY: no other reference to the vector exists while this runs (see `Send`/`Sync`).
        unsafe { (*self.ops.get()).push(op) }
    }

    pub(super) unsafe fn BeginEvent(&self, _metadata: u32, data: Option<*const ffi::c_void>, size: u32) {
        self.push(Op::BeginEvent(marker_bytes(data, size)));
    }
    pub(super) unsafe fn EndEvent(&self) {
        self.push(Op::EndEvent);
    }
    pub(super) unsafe fn SetMarker(&self, _metadata: u32, data: Option<*const ffi::c_void>, size: u32) {
        self.push(Op::SetMarker(marker_bytes(data, size)));
    }
    pub(super) unsafe fn SetDescriptorHeaps(&self, heaps: &[Option<Direct3D12::ID3D12DescriptorHeap>]) {
        self.push(Op::SetDescriptorHeaps(heaps.to_vec()));
    }
    pub(super) unsafe fn IASetVertexBuffers(
        &self,
        start: u32,
        views: Option<&[Direct3D12::D3D12_VERTEX_BUFFER_VIEW]>,
    ) {
        self.push(Op::VertexBuffers(start, views.unwrap_or_default().to_vec()));
    }
    pub(super) unsafe fn SetGraphicsRoot32BitConstant(&self, index: u32, value: u32, offset: u32) {
        self.push(Op::GraphicsConstant(index, value, offset));
    }
    pub(super) unsafe fn SetComputeRoot32BitConstant(&self, index: u32, value: u32, offset: u32) {
        self.push(Op::ComputeConstant(index, value, offset));
    }
    pub(super) unsafe fn SetGraphicsRootDescriptorTable(
        &self,
        index: u32,
        table: Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        self.push(Op::GraphicsTable(index, table));
    }
    pub(super) unsafe fn SetComputeRootDescriptorTable(
        &self,
        index: u32,
        table: Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        self.push(Op::ComputeTable(index, table));
    }
    pub(super) unsafe fn SetGraphicsRootConstantBufferView(&self, index: u32, address: u64) {
        self.push(Op::GraphicsCbv(index, address));
    }
    pub(super) unsafe fn SetComputeRootConstantBufferView(&self, index: u32, address: u64) {
        self.push(Op::ComputeCbv(index, address));
    }
    pub(super) unsafe fn ResourceBarrier(&self, barriers: &[Direct3D12::D3D12_RESOURCE_BARRIER]) {
        // SAFETY: bitwise copies, never dropped (no reference counts change), see `Op::Barriers`.
        let copies = barriers
            .iter()
            .map(|barrier| ManuallyDrop::new(unsafe { ptr::read(barrier) }))
            .collect();
        self.push(Op::Barriers(copies));
    }
    pub(super) unsafe fn CopyBufferRegion(
        &self,
        dst: &Direct3D12::ID3D12Resource,
        dst_offset: u64,
        src: &Direct3D12::ID3D12Resource,
        src_offset: u64,
        size: u64,
    ) {
        self.push(Op::CopyBufferRegion {
            dst: dst.clone(),
            dst_offset,
            src: src.clone(),
            src_offset,
            size,
        });
    }
    pub(super) unsafe fn CopyTextureRegion(
        &self,
        dst: *const Direct3D12::D3D12_TEXTURE_COPY_LOCATION,
        x: u32,
        y: u32,
        z: u32,
        src: *const Direct3D12::D3D12_TEXTURE_COPY_LOCATION,
        src_box: Option<*const Direct3D12::D3D12_BOX>,
    ) {
        // SAFETY: valid pointers from the caller; bitwise copies, see `Op::CopyTextureRegion`.
        unsafe {
            self.push(Op::CopyTextureRegion {
                dst: ManuallyDrop::new(ptr::read(dst)),
                x,
                y,
                z,
                src: ManuallyDrop::new(ptr::read(src)),
                src_box: src_box.map(|b| *b),
            });
        }
    }
    pub(super) unsafe fn BeginQuery(
        &self,
        heap: &Direct3D12::ID3D12QueryHeap,
        ty: Direct3D12::D3D12_QUERY_TYPE,
        index: u32,
    ) {
        self.push(Op::BeginQuery(heap.clone(), ty, index));
    }
    pub(super) unsafe fn EndQuery(
        &self,
        heap: &Direct3D12::ID3D12QueryHeap,
        ty: Direct3D12::D3D12_QUERY_TYPE,
        index: u32,
    ) {
        self.push(Op::EndQuery(heap.clone(), ty, index));
    }
    pub(super) unsafe fn ResolveQueryData(
        &self,
        heap: &Direct3D12::ID3D12QueryHeap,
        ty: Direct3D12::D3D12_QUERY_TYPE,
        start: u32,
        count: u32,
        dst: &Direct3D12::ID3D12Resource,
        offset: u64,
    ) {
        self.push(Op::ResolveQueryData {
            heap: heap.clone(),
            ty,
            start,
            count,
            dst: dst.clone(),
            offset,
        });
    }
    pub(super) unsafe fn OMSetRenderTargets(
        &self,
        count: u32,
        rtvs: Option<*const Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
        single_range: bool,
        dsv: Option<*const Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE>,
    ) {
        // A single-handle range passes one handle; otherwise `count` handles.
        let len = if single_range { count.min(1) } else { count } as usize;
        // SAFETY: the caller passes `len` readable handles and a valid DSV pointer.
        let rtvs = match rtvs {
            Some(rtvs) if len > 0 => unsafe { core::slice::from_raw_parts(rtvs, len).to_vec() },
            _ => Vec::new(),
        };
        let dsv = dsv.map(|dsv| unsafe { *dsv });
        self.push(Op::RenderTargets {
            count,
            rtvs,
            single_range,
            dsv,
        });
    }
    pub(super) unsafe fn ClearRenderTargetView(
        &self,
        rtv: Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
        color: &[f32; 4],
        rects: Option<&[Foundation::RECT]>,
    ) {
        self.push(Op::ClearRtv(rtv, *color, rects.map(<[_]>::to_vec)));
    }
    pub(super) unsafe fn ClearDepthStencilView(
        &self,
        dsv: Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
        flags: Direct3D12::D3D12_CLEAR_FLAGS,
        depth: f32,
        stencil: u8,
        rects: Option<&[Foundation::RECT]>,
    ) {
        self.push(Op::ClearDsv(dsv, flags, depth, stencil, rects.map(<[_]>::to_vec)));
    }
    pub(super) unsafe fn RSSetViewports(&self, viewports: &[Direct3D12::D3D12_VIEWPORT]) {
        self.push(Op::Viewports(viewports.to_vec()));
    }
    pub(super) unsafe fn RSSetScissorRects(&self, rects: &[Foundation::RECT]) {
        self.push(Op::Scissors(rects.to_vec()));
    }
    pub(super) unsafe fn ResolveSubresource(
        &self,
        dst: &Direct3D12::ID3D12Resource,
        dst_sub: u32,
        src: &Direct3D12::ID3D12Resource,
        src_sub: u32,
        format: Dxgi::Common::DXGI_FORMAT,
    ) {
        self.push(Op::ResolveSubresource {
            dst: dst.clone(),
            dst_sub,
            src: src.clone(),
            src_sub,
            format,
        });
    }
    pub(super) unsafe fn SetGraphicsRootSignature(
        &self,
        signature: Option<&Direct3D12::ID3D12RootSignature>,
    ) {
        self.push(Op::GraphicsRootSignature(signature.cloned()));
    }
    pub(super) unsafe fn SetComputeRootSignature(
        &self,
        signature: Option<&Direct3D12::ID3D12RootSignature>,
    ) {
        self.push(Op::ComputeRootSignature(signature.cloned()));
    }
    pub(super) unsafe fn SetPipelineState(&self, pipeline: &Direct3D12::ID3D12PipelineState) {
        self.push(Op::PipelineState(pipeline.clone()));
    }
    pub(super) unsafe fn IASetPrimitiveTopology(&self, topology: Direct3D::D3D_PRIMITIVE_TOPOLOGY) {
        self.push(Op::Topology(topology));
    }
    pub(super) unsafe fn IASetIndexBuffer(&self, view: Option<*const Direct3D12::D3D12_INDEX_BUFFER_VIEW>) {
        // SAFETY: a valid pointer from the caller.
        self.push(Op::IndexBuffer(view.map(|view| unsafe { *view })));
    }
    pub(super) unsafe fn OMSetStencilRef(&self, value: u32) {
        self.push(Op::StencilRef(value));
    }
    pub(super) unsafe fn OMSetBlendFactor(&self, color: Option<&[f32; 4]>) {
        self.push(Op::BlendFactor(color.copied()));
    }
    pub(super) unsafe fn DrawInstanced(&self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        self.push(Op::Draw(vertices, instances, first_vertex, first_instance));
    }
    pub(super) unsafe fn DrawIndexedInstanced(
        &self,
        indices: u32,
        instances: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.push(Op::DrawIndexed(indices, instances, first_index, base_vertex, first_instance));
    }
    pub(super) unsafe fn Dispatch(&self, x: u32, y: u32, z: u32) {
        self.push(Op::Dispatch(x, y, z));
    }
    pub(super) unsafe fn ExecuteIndirect<'a>(
        &self,
        signature: &Direct3D12::ID3D12CommandSignature,
        max_count: u32,
        args: &Direct3D12::ID3D12Resource,
        args_offset: u64,
        count: impl Into<Option<&'a Direct3D12::ID3D12Resource>>,
        count_offset: u64,
    ) {
        let count = count.into();
        self.push(Op::ExecuteIndirect {
            signature: signature.clone(),
            max_count,
            args: args.clone(),
            args_offset,
            count: count.cloned(),
            count_offset,
        });
    }
}

/// What the list being replayed into currently has bound (see `replay`).
#[derive(Default)]
pub(super) struct ReplayState {
    heaps: Vec<usize>,
}

/// Plays `ops` into `list`, in order. Consecutive passes in one list only need their
/// descriptor heaps set once: setting the same heaps again, or unsetting them at the end of a
/// pass, is skipped (changing heaps can be expensive on some GPUs).
pub(super) unsafe fn replay(
    ops: &mut Vec<Op>,
    list: &Direct3D12::ID3D12GraphicsCommandList,
    state: &mut ReplayState,
) {
    for op in ops.drain(..) {
        // SAFETY: the same calls with the same arguments the encoder would have made directly.
        unsafe {
            match op {
                Op::BeginEvent(bytes) => list.BeginEvent(
                    0,
                    (!bytes.is_empty()).then(|| bytes.as_ptr().cast()),
                    bytes.len() as u32,
                ),
                Op::EndEvent => list.EndEvent(),
                Op::SetMarker(bytes) => list.SetMarker(
                    0,
                    (!bytes.is_empty()).then(|| bytes.as_ptr().cast()),
                    bytes.len() as u32,
                ),
                Op::SetDescriptorHeaps(heaps) => {
                    if heaps.is_empty() {
                        continue;
                    }
                    let raw: Vec<usize> = heaps
                        .iter()
                        .map(|heap| heap.as_ref().map_or(0, |heap| heap.as_raw() as usize))
                        .collect();
                    if raw != state.heaps {
                        list.SetDescriptorHeaps(&heaps);
                        state.heaps = raw;
                    }
                }
                Op::VertexBuffers(start, views) => list.IASetVertexBuffers(start, Some(&views)),
                Op::GraphicsConstant(i, v, o) => list.SetGraphicsRoot32BitConstant(i, v, o),
                Op::ComputeConstant(i, v, o) => list.SetComputeRoot32BitConstant(i, v, o),
                Op::GraphicsTable(i, t) => list.SetGraphicsRootDescriptorTable(i, t),
                Op::ComputeTable(i, t) => list.SetComputeRootDescriptorTable(i, t),
                Op::GraphicsCbv(i, a) => list.SetGraphicsRootConstantBufferView(i, a),
                Op::ComputeCbv(i, a) => list.SetComputeRootConstantBufferView(i, a),
                Op::Barriers(barriers) => {
                    // `ManuallyDrop<T>` has the same layout as `T`.
                    let raw: &[Direct3D12::D3D12_RESOURCE_BARRIER] =
                        core::slice::from_raw_parts(barriers.as_ptr().cast(), barriers.len());
                    list.ResourceBarrier(raw);
                }
                Op::CopyBufferRegion {
                    dst,
                    dst_offset,
                    src,
                    src_offset,
                    size,
                } => list.CopyBufferRegion(&dst, dst_offset, &src, src_offset, size),
                Op::CopyTextureRegion {
                    dst,
                    x,
                    y,
                    z,
                    src,
                    src_box,
                } => list.CopyTextureRegion(
                    &*dst,
                    x,
                    y,
                    z,
                    &*src,
                    src_box.as_ref().map(ptr::from_ref),
                ),
                Op::BeginQuery(heap, ty, index) => list.BeginQuery(&heap, ty, index),
                Op::EndQuery(heap, ty, index) => list.EndQuery(&heap, ty, index),
                Op::ResolveQueryData {
                    heap,
                    ty,
                    start,
                    count,
                    dst,
                    offset,
                } => list.ResolveQueryData(&heap, ty, start, count, &dst, offset),
                Op::RenderTargets {
                    count,
                    rtvs,
                    single_range,
                    dsv,
                } => list.OMSetRenderTargets(
                    count,
                    (!rtvs.is_empty()).then(|| rtvs.as_ptr()),
                    single_range,
                    dsv.as_ref().map(ptr::from_ref),
                ),
                Op::ClearRtv(rtv, color, rects) => {
                    list.ClearRenderTargetView(rtv, &color, rects.as_deref())
                }
                Op::ClearDsv(dsv, flags, depth, stencil, rects) => {
                    list.ClearDepthStencilView(dsv, flags, depth, stencil, rects.as_deref())
                }
                Op::Viewports(viewports) => list.RSSetViewports(&viewports),
                Op::Scissors(rects) => list.RSSetScissorRects(&rects),
                Op::ResolveSubresource {
                    dst,
                    dst_sub,
                    src,
                    src_sub,
                    format,
                } => list.ResolveSubresource(&dst, dst_sub, &src, src_sub, format),
                Op::GraphicsRootSignature(signature) => {
                    list.SetGraphicsRootSignature(signature.as_ref())
                }
                Op::ComputeRootSignature(signature) => {
                    list.SetComputeRootSignature(signature.as_ref())
                }
                Op::PipelineState(pipeline) => list.SetPipelineState(&pipeline),
                Op::Topology(topology) => list.IASetPrimitiveTopology(topology),
                Op::IndexBuffer(view) => {
                    list.IASetIndexBuffer(view.as_ref().map(ptr::from_ref))
                }
                Op::StencilRef(value) => list.OMSetStencilRef(value),
                Op::BlendFactor(color) => list.OMSetBlendFactor(color.as_ref()),
                Op::Draw(a, b, c, d) => list.DrawInstanced(a, b, c, d),
                Op::DrawIndexed(a, b, c, d, e) => list.DrawIndexedInstanced(a, b, c, d, e),
                Op::Dispatch(x, y, z) => list.Dispatch(x, y, z),
                Op::ExecuteIndirect {
                    signature,
                    max_count,
                    args,
                    args_offset,
                    count,
                    count_offset,
                } => list.ExecuteIndirect(
                    &signature,
                    max_count,
                    &args,
                    args_offset,
                    count.as_ref(),
                    count_offset,
                ),
            }
        }
    }
}
