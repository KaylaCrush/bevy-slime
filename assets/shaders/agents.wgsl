// Agent compute shader using layer-based pheromone array

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var<storage, read> species: array<SpeciesSettings>;
// Array-based pheromone field: one layer per pheromone (read/write for sensing and deposit)
@group(0) @binding(6) var phero_array: texture_storage_2d_array<r32float, read_write>;
// Extended pheromone controls and per-species profiles (dense). Not used yet; bound with dummies.
@group(0) @binding(7) var<storage, read> species_weights: array<f32>;
@group(0) @binding(9) var<uniform> phero_ctrl: PheroControl;

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
    emit_layer: u32,
    emit_amount: f32,
    _pad_emit: vec2<u32>,
};


struct GlobalUniforms {
    delta_time: f32,
    frame: u32,
    mouse_position: vec2<f32>,
    screen_size: vec2<f32>,
    left_button_pressed: u32,
    right_button_pressed: u32,
};

struct PheroControl {
    layer_count: u32,
    _pad: vec3<u32>,
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

fn weight_base(species_index: u32, layer_count: u32) -> u32 {
    return species_index * layer_count;
}

fn sample_signal_ext_with_base(pos: vec2<i32>, base: u32, lc: u32) -> f32 {
    var sum = 0.0;
    for (var li: u32 = 0u; li < lc; li = li + 1u) {
        let v = textureLoad(phero_array, pos, i32(li)).x;
        let w = species_weights[base + li];
        sum = sum + v * w;
    }
    return sum;
}

@compute @workgroup_size(256)
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
    // Always use layer-based sensing
    let fwdv = vec2<f32>(cos(agent.angle), sin(agent.angle));
    let leftv = vec2<f32>(cos(agent.angle + sensor_angle), sin(agent.angle + sensor_angle));
    let rightv = vec2<f32>(cos(agent.angle - sensor_angle), sin(agent.angle - sensor_angle));
    let p_f = agent.position + fwdv * s.sensor_offset_dst;
    let p_l = agent.position + leftv * s.sensor_offset_dst;
    let p_r = agent.position + rightv * s.sensor_offset_dst;
    let cx_f = clamp(i32(p_f.x), 0, i32(globals.screen_size.x) - 1);
    let cy_f = clamp(i32(p_f.y), 0, i32(globals.screen_size.y) - 1);
    let cx_l = clamp(i32(p_l.x), 0, i32(globals.screen_size.x) - 1);
    let cy_l = clamp(i32(p_l.y), 0, i32(globals.screen_size.y) - 1);
    let cx_r = clamp(i32(p_r.x), 0, i32(globals.screen_size.x) - 1);
    let cy_r = clamp(i32(p_r.y), 0, i32(globals.screen_size.y) - 1);
    // Use a square sensor mask
    let r = i32(s.sensor_size);
    var w_forward = 0.0;
    var w_left = 0.0;
    var w_right = 0.0;
    let lc = phero_ctrl.layer_count;
    let base = weight_base(agent.species_index, lc);
    if (r == 0) {
        w_forward = sample_signal_ext_with_base(vec2<i32>(cx_f, cy_f), base, lc);
        w_left    = sample_signal_ext_with_base(vec2<i32>(cx_l, cy_l), base, lc);
        w_right   = sample_signal_ext_with_base(vec2<i32>(cx_r, cy_r), base, lc);
    } else {
        for (var ox = -r; ox <= r; ox++) {
            for (var oy = -r; oy <= r; oy++) {
                w_forward += sample_signal_ext_with_base(vec2<i32>(cx_f + ox, cy_f + oy), base, lc);
                w_left    += sample_signal_ext_with_base(vec2<i32>(cx_l + ox, cy_l + oy), base, lc);
                w_right   += sample_signal_ext_with_base(vec2<i32>(cx_r + ox, cy_r + oy), base, lc);
            }
        }
    }
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
    // Wrap around screen edges instead of bouncing
    agent.position = wrap_if_needed(agent.position, globals.screen_size);
    let coord = vec2<i32>(i32(agent.position.x), i32(agent.position.y));
    // Deposit only to the species' configured emit layer
    let el = i32(s.emit_layer);
    if (el >= 0) {
        let cur = textureLoad(phero_array, coord, el).x;
        let add = s.emit_amount * globals.delta_time;
        textureStore(phero_array, coord, el, vec4<f32>(cur + add, 0.0, 0.0, 0.0));
    }
    agents[index] = agent;
}

// helpers (movement utils)

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
