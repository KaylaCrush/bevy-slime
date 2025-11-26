// Agent + RGBA environment compute shader

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var input_tex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var temp_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var<storage, read> species: array<SpeciesSettings>;
// Array-based pheromone field: one layer per pheromone (read/write for sensing and deposit)
@group(0) @binding(6) var phero_array: texture_storage_2d_array<r32float, read_write>;

struct Agent {
    position: vec2<f32>,
    angle: f32,
    species_index: u32,
};

struct SpeciesSettings {
    move_speed: f32,
    turn_speed: f32,
    sensor_angle_degrees: f32,
    sensor_offset_dst: f32,

    sensor_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,

    color: vec4<f32>,
    weights: vec4<f32>,
    emit: vec4<f32>,
};


struct GlobalUniforms {
    delta_time: f32,
    frame: u32,
    mouse_position: vec2<f32>,
    screen_size: vec2<f32>,
    left_button_pressed: u32,
    right_button_pressed: u32,
};


fn hash_u32(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn hash_f32(value: u32) -> f32 {
    return f32(hash_u32(value)) / 4294967295.0;
}

@compute @workgroup_size(64)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&agents)) { return; }
    var agent = agents[index];
    let s = species[agent.species_index];
    let dt = globals.delta_time;
    let px = bitcast<u32>(agent.position.x);
    let py = bitcast<u32>(agent.position.y);
    let seed = px ^ hash_u32(py) ^ globals.frame;
    let random_val = hash_f32(seed);
    let sensor_angle = radians(s.sensor_angle_degrees);
    let w_forward = sense(agent.position, agent.angle, s, globals);
    let w_left    = sense(agent.position, agent.angle + sensor_angle, s, globals);
    let w_right   = sense(agent.position, agent.angle - sensor_angle, s, globals);
    var dir = agent.angle;
    if (w_forward > w_left && w_forward > w_right) {
    } else if (w_forward < w_left && w_forward < w_right) {
        dir = dir + (random_val - 0.5) * 2.0 * s.turn_speed * dt;
    } else if (w_right > w_left) {
        dir = dir - random_val * s.turn_speed * dt;
    } else if (w_left > w_right) {
        dir = dir + random_val * s.turn_speed * dt;
    }
    agent.angle = dir;
    let fwd = vec2<f32>(cos(agent.angle), sin(agent.angle));
    agent.position = agent.position + fwd * s.move_speed * dt;
    agent.angle = bounce_if_needed(agent.position, agent.angle, globals.screen_size);
    let coord = vec2<i32>(i32(agent.position.x), i32(agent.position.y));
    // Deposit pheromone intensity per layer (color comes from pheromone layer params in composite)
    let e = s.emit.xyz; // per-layer emission strengths
    let cur_r = textureLoad(phero_array, coord, 0).x;
    let cur_g = textureLoad(phero_array, coord, 1).x;
    let cur_b = textureLoad(phero_array, coord, 2).x;
    // Scale by dt for frame-rate independent deposition
    textureStore(phero_array, coord, 0, vec4<f32>(cur_r + e.x * globals.delta_time, 0.0, 0.0, 0.0));
    textureStore(phero_array, coord, 1, vec4<f32>(cur_g + e.y * globals.delta_time, 0.0, 0.0, 0.0));
    textureStore(phero_array, coord, 2, vec4<f32>(cur_b + e.z * globals.delta_time, 0.0, 0.0, 0.0));
    agents[index] = agent;
}

// helpers (sense, sample_signal, bounce, etc.)
fn sample_signal(pos: vec2<i32>, weights: vec3<f32>) -> f32 {
    // Read from array layers 0..2 and combine by weights
    let r = textureLoad(phero_array, pos, 0).x;
    let g = textureLoad(phero_array, pos, 1).x;
    let b = textureLoad(phero_array, pos, 2).x;
    return r * weights.x + g * weights.y + b * weights.z;
}

fn sense(position: vec2<f32>, angle: f32, s: SpeciesSettings, globals: GlobalUniforms) -> f32 {
    let dir = vec2<f32>(cos(angle), sin(angle));
    let sensor_pos = position + dir * s.sensor_offset_dst;
    let cx = i32(sensor_pos.x);
    let cy = i32(sensor_pos.y);
    let r = i32(s.sensor_size);
    var sum = 0.0;
    for (var ox = -r; ox <= r; ox++) {
        for (var oy = -r; oy <= r; oy++) {
            let sx = clamp(cx + ox, 0, i32(globals.screen_size.x) - 1);
            let sy = clamp(cy + oy, 0, i32(globals.screen_size.y) - 1);
            let signal = sample_signal(vec2<i32>(sx, sy), s.weights.xyz);
            sum += signal;
        }
    }
    return sum;
}

fn bounce_if_needed(position: vec2<f32>, direction: f32, size: vec2<f32>) -> f32 {
    var dir = direction;
    if (position.x <= 0.0 || position.x >= size.x - 1.0) {
        dir = 3.14159265 - dir;
    }
    if (position.y <= 0.0 || position.y >= size.y - 1.0) {
        dir = -dir;
    }
    return dir;
}

fn wrap_if_needed(position: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    var pos = position;
    if (pos.x < 0.0) {
        pos.x = pos.x + size.x;
    }
    if (pos.x >= size.x) {
        pos.x = pos.x - size.x;
    }
    if (pos.y < 0.0) {
        pos.y = pos.y + size.y;
    }
    if (pos.y >= size.y) {
        pos.y = pos.y - size.y;
    }
    return pos;
}
