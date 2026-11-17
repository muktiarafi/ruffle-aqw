/// AQW build: one blur pass for many cacheAsBitmap objects at once, each in its own slot of an
/// atlas page (see `filters::batch`). Per slot, the same math as `blur.wgsl` with that object's
/// arguments; samples are clamped to the slot, as the sampler clamped them to the edges of the
/// object's own texture.

struct VertexInput {
    /// The corner of the quad (0,0 to 1,1).
    @location(0) position: vec2<f32>,
    @location(1) unused_uv: vec2<f32>,
};

struct Instance {
    /// The slot: x, y, width, height in texels.
    @location(2) rect: vec4<f32>,
    /// Page width and height in texels, then the blur direction (1,0 or 0,1).
    @location(3) page_dir: vec4<f32>,
    /// full_size, m, m2, first_weight (see `blur.rs`).
    @location(4) params: vec4<f32>,
    /// last_offset, last_weight, identity (1.0: copy the slot unchanged), unused.
    @location(5) params2: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) uv_min: vec2<f32>,
    @location(2) @interpolate(flat) uv_max: vec2<f32>,
    @location(3) @interpolate(flat) direction: vec2<f32>,
    @location(4) @interpolate(flat) params: vec4<f32>,
    @location(5) @interpolate(flat) params2: vec4<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@vertex
fn main_vertex(in: VertexInput, instance: Instance) -> VertexOutput {
    let page = instance.page_dir.xy;
    let texel = instance.rect.xy + in.position * instance.rect.zw;
    let p = texel / page;
    let pos = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    let direction = instance.page_dir.zw / page;
    // Pre-shifting the UV coords to put the center of the first trivially
    // sampled (with 1.0 weight, "in the center") pixel at 0.0 (as in blur.wgsl).
    let uv = p - direction * instance.params.y;
    let uv_min = (instance.rect.xy + vec2<f32>(0.5, 0.5)) / page;
    let uv_max = (instance.rect.xy + instance.rect.zw - vec2<f32>(0.5, 0.5)) / page;
    return VertexOutput(pos, uv, uv_min, uv_max, direction, instance.params, instance.params2);
}

fn sample_clamped(in: VertexOutput, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(texture, texture_sampler, clamp(uv, in.uv_min, in.uv_max), 0.0);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.params2.z > 0.5) {
        // This pass doesn't blur this slot (as `blur.rs` skips a pass of width 1 or less).
        return textureLoad(texture, vec2<i32>(in.position.xy), 0);
    }
    let direction = in.direction;
    let full_size = in.params.x;
    let m2 = in.params.z;
    let first_weight = in.params.w;
    let last_offset = in.params2.x;
    let last_weight = in.params2.y;

    var total = sample_clamped(in, in.uv - direction) * first_weight;

    var center = vec4<f32>();
    for (var i = 0.5; i < m2; i += 2.0) {
        center += sample_clamped(in, in.uv + direction * i);
    }
    total += center * 2.0;

    let last_location = in.uv + direction * (m2 + last_offset);
    total += sample_clamped(in, last_location) * last_weight;

    let result = total / full_size;
    return floor(result * 255.0) / 255.0;
}
