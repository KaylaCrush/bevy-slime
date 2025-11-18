// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for the game of life as each pixel of step N depends on the state of its
// neighbors at step N-1.

@group(0) @binding(0) var<storage, read_write> agents: array<SlimeAgent>;
@group(0) @binding(1) var input_tex: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var temp_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var<storage, read> species: array<SpeciesSettings>;

struct SlimeAgent {
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
    deposit_amount: f32,
    diffusion_strength: f32,
    decay_rate: f32,

    follow_strength: f32,
    avoid_strength: f32,
    _pad0: f32,
    _pad1: f32,

    color: vec4<f32>,
    follow_mask: vec4<f32>,
    avoid_mask: vec4<f32>,
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

// fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
//     let value: vec4<f32> = textureLoad(input_tex, location + vec2<i32>(offset_x, offset_y));
//     return i32(value.a);
// }

// fn count_alive(location: vec2<i32>) -> i32 {
//     return is_alive(location, -1, -1) +
//            is_alive(location, -1,  0) +
//            is_alive(location, -1,  1) +
//            is_alive(location,  0, -1) +
//            is_alive(location,  0,  1) +
//            is_alive(location,  1, -1) +
//            is_alive(location,  1,  0) +
//            is_alive(location,  1,  1);
// }

// ------------------------------------------------------------
// PASS 0: COPY input_tex → temp_tex
// ------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn copy_to_temp(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(i32(x), i32(y));
    let v = textureLoad(input_tex, coord);
    textureStore(temp_tex, coord, v);
}

// ------------------------------------------------------------
// PASS 1: HANDLE INPUT (erase pheromone under mouse when LMB down)
// operates on temp_tex BEFORE diffusion
// ------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn handle_input(@builtin(global_invocation_id) id: vec3<u32>) {
    // no button, nothing to do
    if (globals.left_button_pressed == 0u && globals.right_button_pressed == 0u) {
        return;
    }

    let dims = textureDimensions(temp_tex);
    let x = id.x;
    let y = id.y;
    if (x >= dims.x || y >= dims.y) {
        return;
    }

    // sentinel: mouse off-screen
    if (globals.mouse_position.x < -9000.0) {
        return;
    }

    let coord = vec2<i32>(i32(x), i32(y));
    let pixel_pos = vec2<f32>(f32(x), f32(y));

    let brush_radius = 80.0;
    let d = distance(pixel_pos, globals.mouse_position);
    if (d >= brush_radius) {
        return;
    }

    let t = 1.0 - (d / brush_radius);    // 1 at center, 0 at edge
    let brush_strength = pow(t, 2.0);    // smoother falloff

    let current = textureLoad(temp_tex, coord);
    let brush_color: vec4<f32> =
        select(vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.0, 0.0, 0.0, 0.0),
               globals.left_button_pressed != 0u);

    let altered = mix(current, brush_color, brush_strength);
    textureStore(temp_tex, coord, altered);
}

// ------------------------------------------------------------
// PASS 2: DIFFUSION (blur + decay) temp_tex → output_tex
// ------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(temp_tex);
    let x = id.x;
    let y = id.y;

    if (x >= dims.x || y >= dims.y) {
        return;
    }

    let dt = globals.delta_time;
    let coord = vec2<i32>(i32(x), i32(y));

    // Neighbor coords (clamped)
    let left   = vec2<i32>(i32(max(1u, x)) - 1, i32(y));
    let right  = vec2<i32>(i32(min(dims.x - 2u, x)) + 1, i32(y));
    let up     = vec2<i32>(i32(x), i32(min(dims.y - 2u, y)) + 1);
    let down   = vec2<i32>(i32(x), i32(max(1u, y)) - 1);

    let c = textureLoad(temp_tex, coord);
    let l = textureLoad(temp_tex, left);
    let r = textureLoad(temp_tex, right);
    let u = textureLoad(temp_tex, up);
    let d = textureLoad(temp_tex, down);

    // Mild blur across all channels (all species)
    let blurred = (c * 4.0 + l + r + u + d) / 8.0;

    // frame-rate-correct diffusion (use species[0] as global until you want per-species)
    let diffusion_factor = per_frame_factor(species[0].diffusion_strength, dt);
    let mixed = mix(c, blurred, diffusion_factor);

    // frame-rate-correct decay
    let decay_factor = per_frame_factor(species[0].decay_rate, dt);
    let result = mixed * (1.0 - decay_factor);

    textureStore(output_tex, coord, result);
}
// ------------------------------------------------------------
// PASS 3: AGENTS (sense → steer → move → deposit)
// ------------------------------------------------------------
@compute @workgroup_size(64)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&agents)) {
        return;
    }

    var agent = agents[index];
    let s = species[agent.species_index];
    let dt = globals.delta_time;

    // ------- Randomness -------
    let px = bitcast<u32>(agent.position.x);
    let py = bitcast<u32>(agent.position.y);
    let seed = px ^ hash_u32(py) ^ globals.frame;
    let random_val = hash_f32(seed);

    // ------- Slime Steering (pheromone) -------
    let sensor_angle = radians(s.sensor_angle_degrees);

    let w_forward = sense(agent.position, agent.angle, s, globals);
    let w_left    = sense(agent.position, agent.angle + sensor_angle, s, globals);
    let w_right   = sense(agent.position, agent.angle - sensor_angle, s, globals);

    var dir = agent.angle;

    if (w_forward > w_left && w_forward > w_right) {
        // keep going
    } else if (w_forward < w_left && w_forward < w_right) {
        // random turn
        dir = dir + (random_val - 0.5) * 2.0 * s.turn_speed * dt;
    } else if (w_right > w_left) {
        dir = dir - random_val * s.turn_speed * dt;
    } else if (w_left > w_right) {
        dir = dir + random_val * s.turn_speed * dt;
    }

    agent.angle = dir;

    // ------- Move -------
    let fwd = vec2<f32>(cos(agent.angle), sin(agent.angle));
    agent.position = agent.position + fwd * s.move_speed * dt;

    // ------- Bounce off bounds -------
    agent.angle = bounce_if_needed(agent.position, agent.angle, globals.screen_size);
    // agent.position = wrap_if_needed(agent.position, globals.screen_size);

    // ------- Deposit (species-specific via mask) -------
    let coord = vec2<i32>(i32(agent.position.x), i32(agent.position.y));

    // read current pheromone (already diffused + decayed)
    let old = textureLoad(temp_tex, coord);
    let mask = s.follow_mask; // e.g. (1,0,0,0) or (0,1,0,0), etc.
    let added = mask * s.deposit_amount;

    textureStore(output_tex, coord, old + added);

    // write back updated agent
    agents[index] = agent;
}

// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------
fn sample_signal(pos: vec2<i32>, mask: vec4<f32>) -> f32 {
    let c = textureLoad(temp_tex, pos);
    return dot(c, mask);
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

            let own = sample_signal(vec2<i32>(sx, sy), s.follow_mask);
            let avoid = sample_signal(vec2<i32>(sx, sy), s.avoid_mask);

            sum += own - avoid*100.0; // avoid signal is weighted more heavily
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

fn per_frame_factor(rate: f32, dt: f32) -> f32 {
    // rate is in 0..1 (e.g. decay = 0.01 means "1% per second")
    // output is frame-corrected factor for this dt
    return 1.0 - pow(1.0 - rate, dt);
}
