//! AQW build: find what keeps leaked display objects alive. Breadth-first search over strong
//! `Gc` edges from the GC root (weak edges never keep anything alive), then reports the shortest
//! chain of references from the root to each target.

use crate::avm2::object::aqw_inspect::describe;
use gc_arena::Collect;
use gc_arena::inspect::{RawEdgeTracer, RawEdgeVisitor, visit_edges};
use std::collections::{HashMap, VecDeque};

struct StrongEdges<'a>(&'a mut Vec<*const ()>);

impl RawEdgeVisitor for StrongEdges<'_> {
    fn visit_edge(&mut self, value_ptr: *const (), weak: bool) {
        if !weak {
            self.0.push(value_ptr);
        }
    }
}

/// For each target, the chain from the root (first) to the target (last), described.
pub(crate) fn retention_paths<'gc, R: Collect<'gc>>(
    root: &R,
    targets: &[*const ()],
) -> Vec<Vec<String>> {
    let root_marker: *const () = std::ptr::null();
    let mut parent: HashMap<*const (), *const ()> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut edges = Vec::new();
    root.trace(&mut RawEdgeTracer(&mut StrongEdges(&mut edges)));
    for child in edges.drain(..) {
        if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(child) {
            e.insert(root_marker);
            queue.push_back(child);
        }
    }
    let mut remaining = targets.iter().filter(|t| !parent.contains_key(t)).count();
    while remaining > 0 {
        let Some(node) = queue.pop_front() else { break };
        // SAFETY: every queued pointer was reached through strong edges from the root, so it is
        // live, and no collection runs during this search.
        unsafe { visit_edges(node, &mut StrongEdges(&mut edges)) };
        for child in edges.drain(..) {
            if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(child) {
                e.insert(node);
                queue.push_back(child);
                if targets.contains(&child) {
                    remaining -= 1;
                }
            }
        }
    }
    targets
        .iter()
        .map(|&target| {
            let Some(_) = parent.get(&target) else {
                return vec!["not strongly reachable (should be collected soon)".to_string()];
            };
            let mut chain = vec![target];
            let mut node = target;
            while let Some(&p) = parent.get(&node) {
                if p.is_null() {
                    break;
                }
                chain.push(p);
                node = p;
            }
            chain.reverse();
            let mut lines = vec![format!("GcRoot ({} objects searched)", parent.len())];
            for (i, &node) in chain.iter().enumerate() {
                let next = chain.get(i + 1).copied();
                // SAFETY: as above, every node on the chain is strongly reachable.
                lines.push(unsafe { describe(node, next) });
            }
            lines
        })
        .collect()
}

/// AQW build: how many `Gc` values of each kind are strongly reachable (AS3 objects by class),
/// to see what grows in long sessions. Walks the whole heap.
pub(crate) fn census<'gc, R: Collect<'gc>>(root: &R) -> HashMap<String, u64> {
    let mut seen: std::collections::HashSet<*const ()> = std::collections::HashSet::new();
    let mut stack = Vec::new();
    let mut edges = Vec::new();
    root.trace(&mut RawEdgeTracer(&mut StrongEdges(&mut edges)));
    for child in edges.drain(..) {
        if seen.insert(child) {
            stack.push(child);
        }
    }
    let mut counts: HashMap<String, u64> = HashMap::new();
    while let Some(node) = stack.pop() {
        // SAFETY: every pointer here was reached through strong edges from the root, so it is
        // live, and no collection runs during this walk.
        let kind = unsafe { describe(node, None) };
        let kind = kind.strip_prefix("ruffle_core::").unwrap_or(&kind).to_string();
        *counts.entry(kind).or_default() += 1;
        unsafe { visit_edges(node, &mut StrongEdges(&mut edges)) };
        for child in edges.drain(..) {
            if seen.insert(child) {
                stack.push(child);
            }
        }
    }
    counts
}
