//! Heap inspection (AQW build of Ruffle): enumerate the `Gc` pointers a value holds, so a tool
//! can walk the object graph from the root and find what keeps an object alive.

use crate::{Gc, GcWeak, collect::Trace, types::GcPtr};

/// Receives the edges of a value, as erased value pointers (`Gc::as_ptr`).
pub trait RawEdgeVisitor {
    fn visit_edge(&mut self, value_ptr: *const (), weak: bool);
}

/// A `Trace` implementation that reports every traced pointer to a [`RawEdgeVisitor`].
/// Pass it to `Collect::trace` on a root (or any value) to list what it points to.
pub struct RawEdgeTracer<'a>(pub &'a mut dyn RawEdgeVisitor);

impl<'gc> Trace<'gc> for RawEdgeTracer<'_> {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        self.0.visit_edge(Gc::as_ptr(gc), false);
    }

    fn trace_gc_weak(&mut self, gc: GcWeak<'gc, ()>) {
        self.0.visit_edge(gc.as_ptr(), true);
    }
}

/// Reports every pointer held by the value at `value_ptr`.
///
/// # Safety
///
/// `value_ptr` must come from `Gc::as_ptr` of a live (not dropped, not freed) `Gc` in an arena
/// that is not being collected during the call. Strongly reachable values are always live.
pub unsafe fn visit_edges(value_ptr: *const (), visitor: &mut dyn RawEdgeVisitor) {
    unsafe {
        let ptr = GcPtr::<()>::from_ptr(value_ptr);
        (ptr.header().vtable().trace_edges)(value_ptr, &mut RawEdgeTracer(visitor));
    }
}

/// The Rust type name of the value at `value_ptr`.
///
/// # Safety
///
/// Same as [`visit_edges`], except the value may already be dropped (only the header is read).
pub unsafe fn type_name(value_ptr: *const ()) -> &'static str {
    unsafe { (GcPtr::<()>::from_ptr(value_ptr).header().vtable().type_name)() }
}
