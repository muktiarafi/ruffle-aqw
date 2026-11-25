//! Turu (charpage-api): a frontend's handles on the display list, by instance-name paths: see
//! it as text, and draw one object of it alone. For tools that take pictures of a movie's parts
//! (a character without its page around it).
//!
//! A path is instance names from the root movie, separated by dots (`mcChar.head`). A `*`
//! segment matches any one child, and `**` any number of levels (`**.mcChar`: an `mcChar`
//! anywhere).

use crate::avm2::object::TObject;
use crate::bitmap::bitmap_data::{BitmapData, IBitmapDrawable};
use crate::display_object::{BoundsMode, DisplayObject, TDisplayObject, TDisplayObjectContainer};
use crate::player::Player;
use ruffle_render::matrix::Matrix;
use ruffle_render::transform::Transform;
use std::fmt::Write;
use swf::{BlendMode, Twips};

/// An object drawn alone (`Player::draw_at`).
pub struct Drawing {
    pub width: u32,
    pub height: u32,
    /// Straight-alpha RGBA, row by row.
    pub rgba: Vec<u8>,
    /// Where the object's own (0, 0) is in the picture, in pixels: a fixed point to place
    /// drawings of different sizes by (an avatar's is at its feet).
    pub origin: (f64, f64),
}

/// The objects under `object` (itself included) that `path` reaches.
fn matching<'gc>(object: DisplayObject<'gc>, path: &[&str], found: &mut Vec<DisplayObject<'gc>>) {
    let Some((first, rest)) = path.split_first() else {
        found.push(object);
        return;
    };
    if *first == "**" {
        // Zero levels here, or one more below (`**` stays).
        matching(object, rest, found);
        for child in children(object) {
            matching(child, path, found);
        }
        return;
    }
    for child in children(object) {
        let named = child.name().is_some_and(|name| name.to_string() == *first);
        if *first == "*" || named {
            matching(child, rest, found);
        }
    }
}

fn children(object: DisplayObject<'_>) -> Vec<DisplayObject<'_>> {
    object
        .as_container()
        .map(|container| container.iter_render_list().collect())
        .unwrap_or_default()
}

fn segments(path: &str) -> Vec<&str> {
    path.split('.').filter(|s| !s.is_empty()).collect()
}

impl Player {
    /// The display list from the root movie, one object a line, indented by depth, to
    /// `max_depth` levels: instance name, AS3 class, `hidden` when it isn't visible, and for a
    /// clip its frame and frame count, and its frame labels.
    pub fn display_tree(&mut self, max_depth: usize) -> String {
        fn describe(object: DisplayObject<'_>, depth: usize, max_depth: usize, out: &mut String) {
            let name = object.name().map(|n| n.to_string()).unwrap_or_default();
            let class = object
                .object2()
                .map(|o| o.instance_class().name().local_name().to_string())
                .unwrap_or_default();
            let _ = write!(out, "{:indent$}{name} [{class}]", "", indent = depth * 2);
            if !object.visible() {
                out.push_str(" hidden");
            }
            if let Some(clip) = object.as_movie_clip() {
                let _ = write!(
                    out,
                    " frame {}/{}",
                    clip.current_frame(),
                    clip.header_frames()
                );
                let labels = clip.labels_in_range(0, u16::MAX);
                if !labels.is_empty() {
                    let labels: Vec<_> = labels
                        .iter()
                        .map(|(label, frame)| format!("{label}@{frame}"))
                        .collect();
                    let _ = write!(out, " labels {}", labels.join(","));
                }
            }
            out.push('\n');
            if depth < max_depth {
                for child in children(object) {
                    describe(child, depth + 1, max_depth, out);
                }
            }
        }
        self.mutate_with_update_context(|context| {
            let mut out = String::new();
            if let Some(root) = context.stage.root_clip() {
                describe(root, 0, max_depth, &mut out);
            }
            out
        })
    }

    /// The first object `path` reaches, drawn alone as AS3's `BitmapData.draw` would draw it
    /// (in its own coordinates, without its parents' transforms, visible or not), at `scale`,
    /// into a transparent picture the size of its bounds plus `pad` pixels around. `None` if
    /// nothing matched or it has no size.
    pub fn draw_at(&mut self, path: &str, scale: f64, pad: u32) -> Option<Drawing> {
        /// Past this, a picture isn't worth the memory (a misplaced part's bounds).
        const MAX_SIDE: f64 = 8192.0;
        let path = segments(path);
        self.mutate_with_update_context(|context| {
            let root = context.stage.root_clip()?;
            let mut found = Vec::new();
            matching(root, &path, &mut found);
            let object = *found.first()?;
            let bounds = object.bounds(BoundsMode::Engine);
            let (x_min, y_min) = (bounds.x_min.to_pixels(), bounds.y_min.to_pixels());
            let width = (bounds.x_max.to_pixels() - x_min) * scale + f64::from(2 * pad);
            let height = (bounds.y_max.to_pixels() - y_min) * scale + f64::from(2 * pad);
            if !(1.0..=MAX_SIDE).contains(&width) || !(1.0..=MAX_SIDE).contains(&height) {
                return None;
            }
            let (width, height) = (width.ceil() as u32, height.ceil() as u32);
            let target = BitmapData::new(context.gc(), width, height, true, 0);
            let matrix = Matrix {
                a: scale as f32,
                b: 0.0,
                c: 0.0,
                d: scale as f32,
                tx: Twips::from_pixels(-x_min * scale + f64::from(pad)),
                ty: Twips::from_pixels(-y_min * scale + f64::from(pad)),
            };
            let transform = Transform {
                matrix,
                ..Default::default()
            };
            let quality = context.stage.quality();
            crate::bitmap::operations::draw(
                context,
                target,
                IBitmapDrawable::DisplayObject(object),
                transform,
                true,
                BlendMode::Normal,
                None,
                quality,
            )
            .ok()?;
            let data = target.sync(context.renderer);
            let data = data.borrow();
            let mut rgba = Vec::with_capacity(data.pixels().len() * 4);
            for pixel in data.pixels() {
                let pixel = pixel.to_un_multiplied_alpha();
                rgba.extend_from_slice(&[pixel.red(), pixel.green(), pixel.blue(), pixel.alpha()]);
            }
            Some(Drawing {
                width,
                height,
                rgba,
                origin: (
                    -x_min * scale + f64::from(pad),
                    -y_min * scale + f64::from(pad),
                ),
            })
        })
    }
}
