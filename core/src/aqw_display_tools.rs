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
    /// The scale it was drawn at: the one asked for, or less to fit `max_side`.
    pub scale: f64,
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
    /// No side is longer than `max_side` pixels: a bigger object is drawn at a smaller scale.
    /// (One huge drawing can take the GPU long enough for the system to reset it.)
    pub fn draw_at(&mut self, path: &str, scale: f64, pad: u32, max_side: u32) -> Option<Drawing> {
        let path = segments(path);
        self.mutate_with_update_context(|context| {
            let root = context.stage.root_clip()?;
            let mut found = Vec::new();
            matching(root, &path, &mut found);
            let object = *found.first()?;
            let bounds = object.bounds(BoundsMode::Engine);
            let (x_min, y_min) = (bounds.x_min.to_pixels(), bounds.y_min.to_pixels());
            let (object_width, object_height) = (
                bounds.x_max.to_pixels() - x_min,
                bounds.y_max.to_pixels() - y_min,
            );
            if object_width <= 0.0 || object_height <= 0.0 {
                return None;
            }
            let room = f64::from(max_side.saturating_sub(2 * pad).max(1));
            let scale = scale.min(room / object_width).min(room / object_height);
            let width = object_width * scale + f64::from(2 * pad);
            let height = object_height * scale + f64::from(2 * pad);
            if width < 1.0 || height < 1.0 {
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
                scale,
            })
        })
    }

    /// Sets the transform of every object `path` reaches, as AS3's `transform.matrix = ...`
    /// would (the timeline no longer moves it): `(a, b, c, d, tx, ty)`, tx and ty in pixels.
    /// How many it reached. Posing an avatar: a game animation's frame copied part by part.
    pub fn set_matrix_at(
        &mut self,
        path: &str,
        (a, b, c, d, tx, ty): (f32, f32, f32, f32, f64, f64),
    ) -> usize {
        let path = segments(path);
        self.mutate_with_update_context(|context| {
            let Some(root) = context.stage.root_clip() else {
                return 0;
            };
            let mut found = Vec::new();
            matching(root, &path, &mut found);
            for object in &found {
                object.set_matrix(Matrix {
                    a,
                    b,
                    c,
                    d,
                    tx: Twips::from_pixels(tx),
                    ty: Twips::from_pixels(ty),
                });
                object.set_transformed_by_script(true);
                if let Some(parent) = object.parent() {
                    parent.invalidate_cached_bitmap();
                }
            }
            found.len()
        })
    }

    /// Restacks the children of the first object `path` reaches: those named in `names` are
    /// put in that order, back to front, in the slots they already hold (other children stay
    /// where they are). How many it moved into place. Posing an avatar: a game animation's
    /// frame stacks the parts its own way (the weapon behind the head in a salute).
    pub fn order_children_at(&mut self, path: &str, names: &[&str]) -> usize {
        let path = segments(path);
        self.mutate_with_update_context(|context| {
            let Some(root) = context.stage.root_clip() else {
                return 0;
            };
            let mut found = Vec::new();
            matching(root, &path, &mut found);
            let Some(mut container) = found.first().and_then(|object| object.as_container()) else {
                return 0;
            };
            let name_of = |object: DisplayObject<'_>| object.name().map(|n| n.to_string());
            // The render-list slots the named children hold, in order, and the order they
            // should be in.
            let children: Vec<_> = container.iter_render_list().collect();
            let slots: Vec<usize> = children
                .iter()
                .enumerate()
                .filter(|(_, child)| name_of(**child).is_some_and(|n| names.contains(&n.as_str())))
                .map(|(index, _)| index)
                .collect();
            let wanted: Vec<String> = names
                .iter()
                .filter(|name| {
                    children
                        .iter()
                        .any(|child| name_of(*child).as_deref() == Some(**name))
                })
                .map(|name| (*name).to_owned())
                .collect();
            // Fill the slots front to back, swapping the wanted child into each.
            for (slot, name) in slots.iter().zip(&wanted) {
                let current: Vec<_> = container.iter_render_list().collect();
                let Some(from) = current
                    .iter()
                    .position(|child| name_of(*child).as_deref() == Some(name))
                else {
                    continue;
                };
                if from != *slot {
                    container.swap_at_index(context, from, *slot);
                }
            }
            let root_object: DisplayObject<'_> = container.into();
            root_object.invalidate_cached_bitmap();
            wanted.len()
        })
    }
}
