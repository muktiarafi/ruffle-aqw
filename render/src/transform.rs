use crate::matrix::Matrix;
use crate::perspective_projection::PerspectiveProjection;
use swf::ColorTransform;

/// Represents the transform for a DisplayObject.
/// This includes both the transformation matrix and the color transform.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transform {
    pub matrix: Matrix,
    pub tz: f64,
    pub color_transform: ColorTransform,
    pub perspective_projection: Option<PerspectiveProjection>,
}

impl Transform {
    /// AQW build: `local` drawn inside this transform (what `TransformStack::push` keeps).
    pub fn then(&self, local: &Transform) -> Transform {
        Transform {
            matrix: self.matrix * local.matrix,
            color_transform: self.color_transform * local.color_transform,
            // TODO: Merge perspective_projections from cur_transform and transform properly
            perspective_projection: Default::default(),
            tz: self.tz + local.tz,
        }
    }

    /// AQW build: whether both are exactly the same.
    pub fn same(&self, other: &Transform) -> bool {
        self.matrix == other.matrix
            && self.color_transform == other.color_transform
            && self.tz.to_bits() == other.tz.to_bits()
            && self.perspective_projection == other.perspective_projection
    }
}

pub struct TransformStack(Vec<Transform>);

impl TransformStack {
    pub fn new() -> Self {
        Self(vec![Transform::default()])
    }

    pub fn push(&mut self, transform: &Transform) {
        let combined = self.0[self.0.len() - 1].then(transform);
        self.0.push(combined);
    }

    /// AQW build: makes `transform` the current one as it is (not inside the current one).
    pub fn push_absolute(&mut self, transform: &Transform) {
        self.0.push(transform.clone());
    }

    pub fn pop(&mut self) {
        assert!(self.0.len() > 1, "Transform stack underflow");
        self.0.pop();
    }

    pub fn transform(&self) -> Transform {
        self.0[self.0.len() - 1].clone()
    }
}

impl Default for TransformStack {
    fn default() -> Self {
        TransformStack::new()
    }
}
