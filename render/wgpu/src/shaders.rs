use crate::blend::ComplexBlend;
use crate::dynamic_transforms::transforms_per_draw;
use enum_map::EnumMap;
use ruffle_render::shader_source::SHADER_FILTER_COMMON;

#[derive(Debug)]
pub struct Shaders {
    pub color_shader: wgpu::ShaderModule,
    /// This has a pipeline-overridable `bool` constant, `late_saturate`,
    /// with a default of `false`. It switches to performing saturation
    /// after re-multiplying the alpha, rather than before. This is used
    /// for the Stage3D `bitmap_opaque` pipeline, which needs to be able to
    /// avoid changing initially-in-range rgb values (regadless of whether
    /// dividing by the alpha value would produce an out-of-range value).
    pub bitmap_shader: wgpu::ShaderModule,
    pub gradient_shader: wgpu::ShaderModule,
    pub copy_shader: wgpu::ShaderModule,
    pub alpha_mask_shader: wgpu::ShaderModule,
    pub blend_shaders: EnumMap<ComplexBlend, wgpu::ShaderModule>,
    /// AQW build: `blend_shaders` for a layer that is read straight from its atlas page (see
    /// `blend_page_shader`), each made when first used.
    blend_page_shaders: EnumMap<ComplexBlend, std::sync::OnceLock<wgpu::ShaderModule>>,
    max_transforms: u64,
    pub color_matrix_filter: wgpu::ShaderModule,
    pub blur_filter: wgpu::ShaderModule,
    pub glow_filter: wgpu::ShaderModule,
    pub bevel_filter: wgpu::ShaderModule,
    pub displacement_map_filter: wgpu::ShaderModule,
    /// AQW build: see `filters::batch`.
    pub batch_blur_filter: wgpu::ShaderModule,
    pub batch_glow_filter: wgpu::ShaderModule,
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let max_transforms = transforms_per_draw(&device.limits());
        let color_shader = make_shader(
            device,
            "color.wgsl",
            include_str!("../shaders/color.wgsl"),
            max_transforms,
        );
        let bitmap_shader = make_shader(
            device,
            "bitmap.wgsl",
            include_str!("../shaders/bitmap.wgsl"),
            max_transforms,
        );
        let copy_shader = make_shader(
            device,
            "copy.wgsl",
            include_str!("../shaders/copy.wgsl"),
            max_transforms,
        );
        let color_matrix_filter = make_filter_shader(
            device,
            "filter/color_matrix.wgsl",
            include_str!("../shaders/filter/color_matrix.wgsl"),
        );
        let blur_filter = make_filter_shader(
            device,
            "filter/blur.wgsl",
            include_str!("../shaders/filter/blur.wgsl"),
        );
        let glow_filter = make_filter_shader(
            device,
            "filter/glow.wgsl",
            include_str!("../shaders/filter/glow.wgsl"),
        );
        let bevel_filter = make_filter_shader(
            device,
            "filter/bevel.wgsl",
            include_str!("../shaders/filter/bevel.wgsl"),
        );
        let displacement_map_filter = make_filter_shader(
            device,
            "filter/displacement_map.wgsl",
            include_str!("../shaders/filter/displacement_map.wgsl"),
        );
        let batch_blur_filter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: create_debug_label!("Shader filter/batch_blur.wgsl").as_deref(),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/filter/batch_blur.wgsl").into(),
            ),
        });
        let batch_glow_filter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: create_debug_label!("Shader filter/batch_glow.wgsl").as_deref(),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/filter/batch_glow.wgsl").into(),
            ),
        });
        let gradient_shader = make_shader(
            device,
            "gradient.wgsl",
            include_str!("../shaders/gradient.wgsl"),
            max_transforms,
        );
        let alpha_mask_shader = make_shader(
            device,
            "alpha_mask.wgsl",
            include_str!("../shaders/alpha_mask.wgsl"),
            max_transforms,
        );

        let blend_shaders = EnumMap::from_fn(|blend: ComplexBlend| {
            make_shader(
                device,
                &format!("blend/{blend:?}"),
                blend_source(blend),
                max_transforms,
            )
        });

        Self {
            color_shader,
            bitmap_shader,
            gradient_shader,
            copy_shader,
            alpha_mask_shader,
            blend_shaders,
            blend_page_shaders: EnumMap::default(),
            max_transforms,
            color_matrix_filter,
            blur_filter,
            glow_filter,
            bevel_filter,
            displacement_map_filter,
            batch_blur_filter,
            batch_glow_filter,
        }
    }
}

impl Shaders {
    /// AQW build: the shader of a complex blend whose layer is read straight from its atlas
    /// page: the quad is placed by a transform like any draw (not by the viewport), carries
    /// the layer's texel coordinates in the page, and reads the backdrop at its own pixel (the
    /// target's blend buffer, of the target's size). Both are read texel for texel, as the
    /// plain shader does through its two textures of the layer's size.
    pub fn blend_page_shader(
        &self,
        device: &wgpu::Device,
        blend: ComplexBlend,
    ) -> &wgpu::ShaderModule {
        self.blend_page_shaders[blend].get_or_init(|| {
            make_shader(
                device,
                &format!("blend/{blend:?} (page)"),
                &blend_page_source(blend_source(blend)),
                self.max_transforms,
            )
        })
    }
}

fn blend_source(blend: ComplexBlend) -> &'static str {
    match blend {
        ComplexBlend::Multiply => include_str!("../shaders/blend/multiply.wgsl"),
        ComplexBlend::Lighten => include_str!("../shaders/blend/lighten.wgsl"),
        ComplexBlend::Darken => include_str!("../shaders/blend/darken.wgsl"),
        ComplexBlend::Difference => include_str!("../shaders/blend/difference.wgsl"),
        ComplexBlend::Invert => include_str!("../shaders/blend/invert.wgsl"),
        ComplexBlend::Alpha => include_str!("../shaders/blend/alpha.wgsl"),
        ComplexBlend::Erase => include_str!("../shaders/blend/erase.wgsl"),
        ComplexBlend::Overlay => include_str!("../shaders/blend/overlay.wgsl"),
        ComplexBlend::HardLight => include_str!("../shaders/blend/hardlight.wgsl"),
    }
}

/// AQW build: a blend shader's source rewritten for `Shaders::blend_page_shader`. (The nine
/// differ only in their blend function; a change to the lines they share shows up here, and
/// in the `blend_page_shaders_validate` test.)
fn blend_page_source(source: &str) -> String {
    let mut source = source.replace("\r\n", "\n");
    for (plain, page) in [
        (
            "@group(2) @binding(0) var parent_texture: texture_2d<f32>;",
            "@group(1) @binding(0) var<uniform> transforms: array<common__Transforms, max_transforms>;\n@group(2) @binding(0) var parent_texture: texture_2d<f32>;",
        ),
        (
            "fn main_vertex(in: common__VertexInput) -> VertexOutput {",
            "fn main_vertex(in: common__VertexInputUv, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {",
        ),
        (
            "let pos = vec4<f32>((in.position.x * 2.0) - 1.0, -(in.position.y * 2.0) + 1.0, 0.0, 1.0);",
            "let pos = common__globals.global_matrix * (transforms[instanceIndex].world_matrix * vec4<f32>(in.position.x, in.position.y, 0.0, 1.0));",
        ),
        ("let uv = vec2<f32>(in.position.xy);", "let uv = in.uv.xy;"),
        (
            "textureSample(parent_texture, texture_sampler, in.uv)",
            "textureLoad(parent_texture, vec2<i32>(in.position.xy), 0)",
        ),
        (
            "textureSample(current_texture, texture_sampler, in.uv)",
            "textureLoad(current_texture, vec2<i32>(in.uv), 0)",
        ),
    ] {
        assert_eq!(
            source.matches(plain).count(),
            1,
            "blend shader without `{plain}`"
        );
        source = source.replacen(plain, page, 1);
    }
    source
}

fn make_shader(
    device: &wgpu::Device,
    name: &str,
    source: &str,
    max_transforms: u64,
) -> wgpu::ShaderModule {
    let common = include_str!("../shaders/common.wgsl");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(
            format!("{common}\nconst max_transforms = {max_transforms};\n{source}").into(),
        ),
    })
}
fn make_filter_shader(device: &wgpu::Device, name: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: create_debug_label!("Shader {name}").as_deref(),
        source: wgpu::ShaderSource::Wgsl(format!("{SHADER_FILTER_COMMON}\n{source}").into()),
    })
}

#[cfg(test)]
mod tests {
    /// AQW build: the shaders of blends read from an atlas page are made from the text of
    /// the plain ones when first used: the lines must still be found, and the result must
    /// compile and validate.
    #[test]
    fn blend_page_shaders_validate() {
        use crate::blend::ComplexBlend;
        let common = include_str!("../shaders/common.wgsl");
        for blend in [
            ComplexBlend::Multiply,
            ComplexBlend::Lighten,
            ComplexBlend::Darken,
            ComplexBlend::Difference,
            ComplexBlend::Invert,
            ComplexBlend::Alpha,
            ComplexBlend::Erase,
            ComplexBlend::Overlay,
            ComplexBlend::HardLight,
        ] {
            let source = super::blend_page_source(super::blend_source(blend));
            let full = format!("{common}\nconst max_transforms = 64;\n{source}");
            let module = naga::front::wgsl::parse_str(&full)
                .unwrap_or_else(|e| panic!("{blend:?}: {}", e.emit_to_string(&full)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{blend:?}: {e:?}"));
        }
    }

    /// AQW build: the fill shaders compile and validate (they're only compiled at startup).
    #[test]
    fn fill_shaders_validate() {
        let common = include_str!("../shaders/common.wgsl");
        for (name, source) in [
            ("color", include_str!("../shaders/color.wgsl")),
            ("gradient", include_str!("../shaders/gradient.wgsl")),
            ("bitmap", include_str!("../shaders/bitmap.wgsl")),
        ] {
            let full = format!(
                "{common}
const max_transforms = 64;
{source}"
            );
            let module = naga::front::wgsl::parse_str(&full)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&full)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
