//! AQW build: packing rectangles into atlas pages (the blend layer atlas in
//! `surface::commands` and the batched cacheAsBitmap filters in `cache_batch`).
//!
//! A page is cleared, drawn, and (with MSAA) resolved over its whole area, so empty space costs
//! GPU time: on integrated graphics, fixed 2048-wide pages packed in arrival order doubled the
//! GPU work compared with one texture per layer. Here the items are packed tallest first into
//! rows of a width chosen to make the page roughly square, and the page is cut to the area
//! actually used (in steps of `GRANULARITY`, so pages are reused from the pool).

pub const MAX_WIDTH: u32 = 2048;
pub const MAX_HEIGHT: u32 = 1024;
const GRANULARITY: u32 = 64;

fn round_up(value: u32) -> u32 {
    value.div_ceil(GRANULARITY) * GRANULARITY
}

/// Packs the longest prefix of `sizes` (each at most `MAX_WIDTH` x `MAX_HEIGHT`, at most
/// `max_count` of them) that fits one page. Returns how many were taken, the page size, and
/// the position of each taken item. At least one item is always taken.
pub fn pack(sizes: &[(u32, u32)], max_count: usize) -> (usize, (u32, u32), Vec<(u32, u32)>) {
    debug_assert!(!sizes.is_empty());
    // Leave some room for the rows' unused ends.
    let budget = MAX_WIDTH as u64 * MAX_HEIGHT as u64 * 85 / 100;
    let mut count = 0;
    let mut area = 0u64;
    for &(width, height) in sizes {
        let item = width as u64 * height as u64;
        if count > 0 && (count >= max_count || area + item > budget) {
            break;
        }
        area += item;
        count += 1;
    }
    loop {
        if let Some((size, positions)) = try_pack(&sizes[..count], area) {
            return (count, size, positions);
        }
        // Doesn't fit after all: leave the last one for the next page.
        count -= 1;
        let (width, height) = sizes[count];
        area -= width as u64 * height as u64;
    }
}

fn try_pack(items: &[(u32, u32)], area: u64) -> Option<((u32, u32), Vec<(u32, u32)>)> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(items[index].1));
    let widest = items.iter().map(|&(width, _)| width).max().unwrap_or(0);
    let square = round_up(((area as f64 * 1.15).sqrt()) as u32);
    let mut widths = vec![square.clamp(widest, MAX_WIDTH)];
    if widths[0] != MAX_WIDTH {
        widths.push(MAX_WIDTH);
    }
    for shelf_width in widths {
        let mut positions = vec![(0, 0); items.len()];
        let (mut x, mut y, mut row_height, mut used_width) = (0u32, 0u32, 0u32, 0u32);
        for &index in &order {
            let (width, height) = items[index];
            if x + width > shelf_width {
                x = 0;
                y += row_height;
                row_height = 0;
            }
            positions[index] = (x, y);
            x += width;
            used_width = used_width.max(x);
            row_height = row_height.max(height);
        }
        let used_height = y + row_height;
        if used_height <= MAX_HEIGHT {
            let size = (
                round_up(used_width).min(MAX_WIDTH),
                round_up(used_height).min(MAX_HEIGHT),
            );
            return Some((size, positions));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(sizes: &[(u32, u32)], max_count: usize) {
        let (count, (page_width, page_height), positions) = pack(sizes, max_count);
        assert!(count >= 1 && count <= sizes.len().min(max_count));
        assert!(page_width <= MAX_WIDTH && page_height <= MAX_HEIGHT);
        for (i, &(x, y)) in positions.iter().enumerate() {
            let (w, h) = sizes[i];
            assert!(x + w <= page_width && y + h <= page_height, "item {i} outside the page");
            for (j, &(x2, y2)) in positions.iter().enumerate().skip(i + 1) {
                let (w2, h2) = sizes[j];
                let overlap = x < x2 + w2 && x2 < x + w && y < y2 + h2 && y2 < y + h;
                assert!(!overlap, "items {i} and {j} overlap");
            }
        }
    }

    #[test]
    fn packs_without_overlap() {
        check(&[(100, 50)], 256);
        check(&[(2048, 1024)], 256);
        check(&[(2048, 1024), (10, 10)], 256);
        let mixed: Vec<(u32, u32)> = (0..400)
            .map(|i| (20 + (i * 37 % 230) as u32, 15 + (i * 53 % 190) as u32))
            .collect();
        check(&mixed, 256);
        check(&mixed[100..], 256);
        let wide: Vec<(u32, u32)> = (0..50).map(|i| (900 + i * 7, 40 + i)).collect();
        check(&wide, 256);
    }
}
