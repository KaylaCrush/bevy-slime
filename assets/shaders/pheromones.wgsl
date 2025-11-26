// Pheromone-specific compute shaders: extract, composite, per-pheromone env passes

struct GlobalUniforms {
    delta_time: f32,
    frame: u32,
    mouse_position: vec2<f32>,
    screen_size: vec2<f32>,
    left_button_pressed: u32,
    right_button_pressed: u32,
};

struct PheromoneParams {
    diffusion: vec4<f32>,
    decay: vec4<f32>,
};

fn per_frame_factor(rate: f32, dt: f32) -> f32 {
    return 1.0 - pow(1.0 - rate, dt);
}

// =========================
// Array-based variants (Step 2)
// These operate on a single R32Float texture2D array where each layer is a pheromone channel.

// Diffuse/decay: prev -> next (per-layer via id.z)
@group(0) @binding(0) var prev_array: texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var next_array: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(2) var rgba_temp_array: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> globals_array: GlobalUniforms;
@group(0) @binding(4) var<storage, read> species_array: array<vec4<f32>>; // unused here; keep layout compatible
@group(0) @binding(5) var<uniform> pheromones_array: PheromoneParams;

@compute @workgroup_size(8, 8, 1)
fn diffuse_phero_array(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(prev_array);
    let x = id.x; let y = id.y; let l = i32(id.z);
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let dt = globals_array.delta_time;
    let left   = vec2<i32>(i32(max(1u, x)) - 1, i32(y));
    let right  = vec2<i32>(i32(min(dims.x - 2u, x)) + 1, i32(y));
    let up     = vec2<i32>(i32(x), i32(min(dims.y - 2u, y)) + 1);
    let down   = vec2<i32>(i32(x), i32(max(1u, y)) - 1);
    let c = textureLoad(prev_array, coord, l).x;
    let lval = textureLoad(prev_array, left, l).x;
    let rval = textureLoad(prev_array, right, l).x;
    let uval = textureLoad(prev_array, up, l).x;
    let dval = textureLoad(prev_array, down, l).x;
    let blurred = (c * 4.0 + lval + rval + uval + dval) / 8.0;
    // Use x component of params for now; can extend to vector per-layer later
    let diff_factor = per_frame_factor(pheromones_array.diffusion.x, dt);
    let dec_factor  = per_frame_factor(pheromones_array.decay.x, dt);
    let mixed = mix(c, blurred, diff_factor);
    let result = mixed * (1.0 - dec_factor);
    textureStore(next_array, coord, l, vec4<f32>(result, 0.0, 0.0, 0.0));
}

// Input (brush) in-place on next_array
@compute @workgroup_size(8, 8, 1)
fn handle_input_phero_array(@builtin(global_invocation_id) id: vec3<u32>) {
    if (globals_array.left_button_pressed == 0u && globals_array.right_button_pressed == 0u) { return; }
    let dims = textureDimensions(next_array);
    let x = id.x; let y = id.y; let l = i32(id.z);
    if (x >= dims.x || y >= dims.y) { return; }
    if (globals_array.mouse_position.x < -9000.0) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let pixel_pos = vec2<f32>(f32(x), f32(y));
    let brush_radius = 80.0;
    let d = distance(pixel_pos, globals_array.mouse_position);
    if (d >= brush_radius) { return; }
    let t = 1.0 - (d / brush_radius);
    let brush_strength = pow(t, 2.0);
    let current = textureLoad(next_array, coord, l).x;
    let brush_val: f32 = select(1.0, 0.0, globals_array.left_button_pressed != 0u);
    let altered = mix(current, brush_val, brush_strength);
    textureStore(next_array, coord, l, vec4<f32>(altered, 0.0, 0.0, 0.0));
}

// Composite array -> RGBA display (layers 0..2 to r,g,b)
@group(0) @binding(0) var p_in_array: texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var rgba_out_array: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn composite_pheromones_array(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rgba_out_array);
    let x = id.x; let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    // Guarded load: if layers < 3, missing layers are treated as 0
    let r = textureLoad(p_in_array, coord, 0).x;
    let g = textureLoad(p_in_array, coord, 1).x;
    let b = textureLoad(p_in_array, coord, 2).x;
    textureStore(rgba_out_array, coord, vec4<f32>(r, g, b, 1.0));
}

// --- Step 1: extract RGBA -> three R32 pheromone textures ---
@group(0) @binding(0) var rgba_in: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var p_out0: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var p_out1: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var p_out2: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn extract_pheromones(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rgba_in);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let c = textureLoad(rgba_in, coord);
    textureStore(p_out0, coord, vec4<f32>(c.r, 0.0, 0.0, 0.0));
    textureStore(p_out1, coord, vec4<f32>(c.g, 0.0, 0.0, 0.0));
    textureStore(p_out2, coord, vec4<f32>(c.b, 0.0, 0.0, 0.0));
}

// --- Step 2: composite R32 pheromones -> RGBA display ---
@group(0) @binding(0) var p_in0: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var p_in1: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var p_in2: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var rgba_out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn composite_pheromones(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rgba_out);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let r = textureLoad(p_in0, coord).x;
    let g = textureLoad(p_in1, coord).x;
    let b = textureLoad(p_in2, coord).x;
    textureStore(rgba_out, coord, vec4<f32>(r, g, b, 1.0));
}

// --- Per-pheromone env passes (copy / input / diffuse) ---
@group(0) @binding(0) var input_phero: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var temp_phero: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var output_phero: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var rgba_temp: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var<storage, read> species: array<vec4<f32>>; // placeholder layout
@group(0) @binding(6) var<uniform> pheromones: PheromoneParams;

@compute @workgroup_size(8, 8, 1)
fn copy_to_temp_phero(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_phero);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let v = textureLoad(input_phero, coord);
    textureStore(temp_phero, coord, v);
}

@compute @workgroup_size(8, 8, 1)
fn handle_input_phero(@builtin(global_invocation_id) id: vec3<u32>) {
    if (globals.left_button_pressed == 0u && globals.right_button_pressed == 0u) {
        return;
    }
    let dims = textureDimensions(temp_phero);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    if (globals.mouse_position.x < -9000.0) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    let pixel_pos = vec2<f32>(f32(x), f32(y));
    let brush_radius = 80.0;
    let d = distance(pixel_pos, globals.mouse_position);
    if (d >= brush_radius) { return; }
    let t = 1.0 - (d / brush_radius);
    let brush_strength = pow(t, 2.0);
    let current = textureLoad(temp_phero, coord).x;
    let brush_val: f32 = select(1.0, 0.0, globals.left_button_pressed != 0u);
    let altered = mix(current, brush_val, brush_strength);
    textureStore(temp_phero, coord, vec4<f32>(altered, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn diffuse_phero(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(temp_phero);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let dt = globals.delta_time;
    let coord = vec2<i32>(i32(x), i32(y));
    let left   = vec2<i32>(i32(max(1u, x)) - 1, i32(y));
    let right  = vec2<i32>(i32(min(dims.x - 2u, x)) + 1, i32(y));
    let up     = vec2<i32>(i32(x), i32(min(dims.y - 2u, y)) + 1);
    let down   = vec2<i32>(i32(x), i32(max(1u, y)) - 1);
    let c = textureLoad(temp_phero, coord).x;
    let l = textureLoad(temp_phero, left).x;
    let r = textureLoad(temp_phero, right).x;
    let u = textureLoad(temp_phero, up).x;
    let d = textureLoad(temp_phero, down).x;
    let blurred = (c * 4.0 + l + r + u + d) / 8.0;
    let diff_factor = per_frame_factor(pheromones.diffusion.x, dt);
    let dec_factor = per_frame_factor(pheromones.decay.x, dt);
    let mixed = mix(c, blurred, diff_factor);
    let result = mixed * (1.0 - dec_factor);
    textureStore(output_phero, coord, vec4<f32>(result, 0.0, 0.0, 0.0));
}
