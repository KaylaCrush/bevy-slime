@group(0) @binding(0)
var input_tex: texture_storage_2d<rgba32float, read>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var<uniform> uniforms: DiffusionUniforms;

struct DiffusionUniforms {
    decay: f32,               // decay per second (NOT per frame)
    diffusion_strength: f32,  // diffusion per second (NOT per frame)
    delta_time: f32,          // dt in seconds
    padding1: f32,
};


// Instead of linear scaling, use exponential time scaling:
//
//   per_frame_factor = 1 - (1 - per_second_factor)^(dt)
//
// This gives stable behavior at any framerate.
fn per_frame_factor(rate: f32, dt: f32) -> f32 {
    // rate is in 0..1 (e.g. decay = 0.01 means "1% per second")
    // output is frame-corrected factor for this dt
    return 1.0 - pow(1.0 - rate, dt);
}


@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    let x = id.x;
    let y = id.y;

    if (x >= dims.x || y >= dims.y) {
        return;
    }

    let dt = uniforms.delta_time;
    let coord = vec2<i32>(i32(x), i32(y));

    // Neighbor coords (clamped)
    let left   = vec2<i32>(i32(max(1u, x)) - 1, i32(y));
    let right  = vec2<i32>(i32(min(dims.x - 2u, x)) + 1, i32(y));
    let up     = vec2<i32>(i32(x), i32(min(dims.y - 2u, y)) + 1);
    let down   = vec2<i32>(i32(x), i32(max(1u, y)) - 1);

    let c = textureLoad(input_tex, coord);
    let l = textureLoad(input_tex, left);
    let r = textureLoad(input_tex, right);
    let u = textureLoad(input_tex, up);
    let d = textureLoad(input_tex, down);

    // Mild blur
    let diffuse = (c * 4.0 + l + r + u + d) / 8.0;

    // ---- FRAME-RATE CORRECT DIFFUSION ----
    let diffusion_factor = per_frame_factor(uniforms.diffusion_strength, dt);
    let mixed = mix(c, diffuse, diffusion_factor);

    // ---- FRAME-RATE CORRECT DECAY ----
    let decay_factor = per_frame_factor(uniforms.decay, dt);
    let result = mixed * (1.0 - decay_factor);

    textureStore(output_tex, coord, result);
}
