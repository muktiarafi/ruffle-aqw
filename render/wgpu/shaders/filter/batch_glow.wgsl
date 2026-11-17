/// AQW build: the glow (and drop shadow) step for many cacheAsBitmap objects at once, each in its
/// own slot of an atlas page (see `filters::batch`). Per slot, the same math as `glow.wgsl` with
/// that object's arguments.

struct VertexInput {
    /// The corner of the quad (0,0 to 1,1).
    @location(0) position: vec2<f32>,
    @location(1) unused_uv: vec2<f32>,
};

struct Instance {
    /// The slot: x, y, width, height in texels.
    @location(2) rect: vec4<f32>,
    /// Page width and height in texels, then the blur offset in texels (drop shadows).
    @location(3) page_offset: vec4<f32>,
    @location(4) color: vec4<f32>,
    /// strength, inner, knockout, composite_source (flags as 0.0 / 1.0).
    @location(5) params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) source_uv: vec2<f32>,
    @location(1) blur_uv: vec2<f32>,
    /// `blur_uv` relative to the slot (0,0 to 1,1 inside it), as `glow.wgsl` sees it.
    @location(2) blur_local: vec2<f32>,
    @location(3) @interpolate(flat) uv_min: vec2<f32>,
    @location(4) @interpolate(flat) uv_max: vec2<f32>,
    @location(5) @interpolate(flat) color: vec4<f32>,
    @location(6) @interpolate(flat) params: vec4<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var blurred: texture_2d<f32>;

@vertex
fn main_vertex(in: VertexInput, instance: Instance) -> VertexOutput {
    let page = instance.page_offset.xy;
    let offset = instance.page_offset.zw;
    let texel = instance.rect.xy + in.position * instance.rect.zw;
    let p = texel / page;
    let pos = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    let blur_uv = (texel + offset) / page;
    let blur_local = (in.position * instance.rect.zw + offset) / instance.rect.zw;
    let uv_min = (instance.rect.xy + vec2<f32>(0.5, 0.5)) / page;
    let uv_max = (instance.rect.xy + instance.rect.zw - vec2<f32>(0.5, 0.5)) / page;
    return VertexOutput(pos, p, blur_uv, blur_local, uv_min, uv_max, instance.color, instance.params);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let strength = in.params.x;
    let inner = in.params.y > 0.5;
    let knockout = in.params.z > 0.5;
    let composite_source = in.params.w > 0.5;
    var blur = textureSampleLevel(blurred, texture_sampler, clamp(in.blur_uv, in.uv_min, in.uv_max), 0.0).a;
    var dest = textureSampleLevel(texture, texture_sampler, clamp(in.source_uv, in.uv_min, in.uv_max), 0.0);

    if (in.blur_local.x < 0.0 || in.blur_local.x > 1.0 || in.blur_local.y < 0.0 || in.blur_local.y > 1.0) {
        blur = 0.0;
    }

    // Start with 1 alpha because we'll be multiplying the whole thing
    var color = vec4<f32>(in.color.r, in.color.g, in.color.b, 1.0);
    if (inner) {
        let alpha = in.color.a * saturate((1.0 - blur) * strength);
        if (knockout) {
            color = color * alpha * dest.a;
        } else if (composite_source) {
            color = color * alpha * dest.a + dest * (1.0 - alpha);
        } else {
            color = color * alpha * dest.a;
        }
    } else {
        let alpha = in.color.a * saturate(blur * strength);
        if (knockout) {
            color = color * alpha * (1.0 - dest.a);
        } else if (composite_source) {
            color = color * alpha * (1.0 - dest.a) + dest;
        } else {
            color = color * alpha;
        }
    }

    return color;
}
