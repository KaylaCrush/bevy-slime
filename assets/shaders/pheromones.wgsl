// Pheromone-specific compute shaders: extract, composite, per-pheromone env passes

struct GlobalUniforms {
    delta_time: f32,
    frame: u32,
    mouse_position: vec2<f32>,
    screen_size: vec2<f32>,
    left_button_pressed: u32,
    right_button_pressed: u32,
};

struct PheromoneLayerParam {
    diffusion: f32,
    decay: f32,
    _pad0: f32,
    _pad1: f32,
    color: vec4<f32>,
};

// per-frame factors are now precomputed on CPU and uploaded in layer_params

// =========================
// Array-based variants (Step 2)
// These operate on a single R32Float texture2D array where each layer is a pheromone channel.

// Diffuse/decay: prev -> next (per-layer via id.z)
@group(0) @binding(0) var prev_array: texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var next_array: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(2) var<uniform> globals_array: GlobalUniforms;
@group(0) @binding(3) var<storage, read> layer_params_array: array<PheromoneLayerParam>;
struct BrushControl { target_layer: u32, _mode: u32, _pad: vec2<u32> };
@group(0) @binding(4) var<uniform> brush_ctrl: BrushControl;

@compute @workgroup_size(16, 16, 1)
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
    let blurred = (c * 4.0 + lval + rval + uval + dval) * 0.125;
    let layer = layer_params_array[id.z];
    // diffusion and decay fields hold per-frame factors now
    let diff_factor = layer.diffusion;
    let dec_factor  = layer.decay;
    let mixed = mix(c, blurred, diff_factor);
    let result = mixed * (1.0 - dec_factor);
    textureStore(next_array, coord, l, vec4<f32>(result, 0.0, 0.0, 0.0));
}

// Input (brush) in-place on next_array
@compute @workgroup_size(16, 16, 1)
fn handle_input_phero_array(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(next_array);
    let x = id.x; let y = id.y; let l = i32(id.z);
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));

    // Seed "avoid" pheromone on screen edges using a squircle (hate layer = 0)
    let border: f32 = 4.0;
    if (l == 0) {
        // Squircle formula: distance from edge using superellipse with power=4
        let cx = f32(x) - f32(dims.x) * 0.5;
        let cy = f32(y) - f32(dims.y) * 0.5;
        let rx = f32(dims.x) * 0.5 - border;
        let ry = f32(dims.y) * 0.5 - border;
        let nx = cx / rx;
        let ny = cy / ry;
        let squircle_dist = pow(abs(nx), 4.0) + pow(abs(ny), 4.0);
        if (squircle_dist >= 1.0) {
            textureStore(next_array, coord, l, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
    }

    // Mouse brush: gate only this section on button state
    if (globals_array.left_button_pressed == 0u && globals_array.right_button_pressed == 0u) { return; }
    if (globals_array.mouse_position.x < -9000.0) { return; }
    let pixel_pos = vec2<f32>(f32(x), f32(y));
    let brush_radius = 80.0;
    let d = distance(pixel_pos, globals_array.mouse_position);
    if (d >= brush_radius) { return; }
    let t = 1.0 - (d / brush_radius);
    let brush_strength = t * t;
    // Paint only the selected layer via uniform control; left = deposit to 1.0, right = erase to 0.0
    let is_left = globals_array.left_button_pressed != 0u;
    let is_right = globals_array.right_button_pressed != 0u;
    let should_paint = (l == i32(brush_ctrl.target_layer)) && (is_left || is_right);
    if (!should_paint) { return; }
    let current = textureLoad(next_array, coord, l).x;
    let brush_val: f32 = select(0.0, 1.0, is_left); // left deposits, right erases to 0.0
    let altered = mix(current, brush_val, brush_strength);
    textureStore(next_array, coord, l, vec4<f32>(altered, 0.0, 0.0, 0.0));
}

// Composite array -> RGBA display using per-layer colors
@group(0) @binding(0) var p_in_array: texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var rgba_out_array: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read> layer_params_comp: array<PheromoneLayerParam>;

@compute @workgroup_size(16, 16, 1)
fn composite_pheromones_array(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(rgba_out_array);
    let x = id.x; let y = id.y;
    if (x >= dims.x || y >= dims.y) { return; }
    let coord = vec2<i32>(i32(x), i32(y));
    var accum = vec3<f32>(0.0, 0.0, 0.0);
    var total = 0.0;
    let layer_count = arrayLength(&layer_params_comp);
    for (var li: u32 = 0u; li < layer_count; li = li + 1u) {
        let v = textureLoad(p_in_array, coord, i32(li)).x;
        let p = layer_params_comp[li];
        accum += v * p.color.rgb;
        total += v;
    }
    // Normalize color by total intensity to preserve hue
    let color = select(vec3<f32>(0.0, 0.0, 0.0), accum / total, total > 0.0);
    // Cheap tone mapping for brightness from total intensity (avoids exp)
    // approx of 1 - exp(-x): x / (1 + x)
    let brightness = total / (1.0 + total);
    textureStore(rgba_out_array, coord, vec4<f32>(color * brightness, 1.0));
}

// Legacy per-pheromone env passes removed; array variants above are the only active path.
