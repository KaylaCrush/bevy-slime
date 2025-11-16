// ============================================================================
// AGENT COMPUTE SHADER
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> agents: array<SlimeAgent>;

@group(0) @binding(1)
var input_tex: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3)
var<uniform> uniforms: AgentUniforms;


// ============================================================================
// DATA LAYOUTS
// ============================================================================
struct SlimeAgent {
    position: vec2<f32>,
    direction: f32,
    pad: f32,
};

struct AgentUniforms {
    move_speed: f32,
    turn_speed: f32,

    sensor_angle_degrees: f32,
    sensor_offset_dst: f32,

    sensor_size: i32,
    _pad0: i32,

    screen_size: vec2<f32>,

    color: vec4<f32>,

    delta_time: f32,
    frame: u32,
    _pad1: vec2<f32>,
};


// ============================================================================
// HELPERS
// ============================================================================

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn hash_f32(x: u32) -> f32 {
    return f32(hash_u32(x)) / 4294967295.0;
}

fn sense(position: vec2<f32>, angle: f32, u: AgentUniforms) -> f32 {
    let dir = vec2<f32>(cos(angle), sin(angle));
    let sensor_pos = position + dir * u.sensor_offset_dst;

    let cx = i32(sensor_pos.x);
    let cy = i32(sensor_pos.y);

    var sum: f32 = 0.0;

    for (var ox = -u.sensor_size; ox <= u.sensor_size; ox = ox + 1) {
        for (var oy = -u.sensor_size; oy <= u.sensor_size; oy = oy + 1) {
            let sx = clamp(cx + ox, 0, i32(u.screen_size.x) - 1);
            let sy = clamp(cy + oy, 0, i32(u.screen_size.y) - 1);

            let col = textureLoad(input_tex, vec2<i32>(sx, sy)).r;
            sum = sum + col;
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


// ============================================================================
// MAIN UPDATE
// ============================================================================

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&agents)) {
        return;
    }

    var agent = agents[index];
    let dt = uniforms.delta_time;
    // let dt = 0.01; // for more visible trails at low fps

    // ------- Randomness (true hash) -------
    let px = bitcast<u32>(agent.position.x);
    let py = bitcast<u32>(agent.position.y);
    let seed = px ^ hash_u32(py) ^ uniforms.frame;
    let random_val = hash_f32(seed);

    // ------- Slime Steering -------
    let sensor_angle = radians(uniforms.sensor_angle_degrees);

    let w_forward = sense(agent.position, agent.direction, uniforms);
    let w_left    = sense(agent.position, agent.direction + sensor_angle, uniforms);
    let w_right   = sense(agent.position, agent.direction - sensor_angle, uniforms);

    var dir = agent.direction;

    if (w_forward > w_left && w_forward > w_right) {
        // stay straight
    }
    else if (w_forward < w_left && w_forward < w_right) {
        dir = dir + (random_val - 0.5) * 2.0 * uniforms.turn_speed * dt;
    }
    else if (w_right > w_left) {
        dir = dir - random_val * uniforms.turn_speed * dt;
    }
    else if (w_left > w_right) {
        dir = dir + random_val * uniforms.turn_speed * dt;
    }

    agent.direction = dir;

    // ------- Move -------
    let fwd = vec2<f32>(cos(agent.direction), sin(agent.direction));
    agent.position = agent.position + fwd * uniforms.move_speed * dt;

    // ------- Bounce -------
    agent.direction = bounce_if_needed(agent.position, agent.direction, uniforms.screen_size);

    // ------- Deposit -------
    let coord = vec2<i32>(i32(agent.position.x), i32(agent.position.y));
    let old = textureLoad(input_tex, coord).rgba;
    let added = vec4<f32>(uniforms.color.rgb, 1.0);

    textureStore(output_tex, coord, old + added);

    agents[index] = agent;
}
