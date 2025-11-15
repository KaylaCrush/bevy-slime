// shaders/slime_sim.wgsl
// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the simulation.

@group(0) @binding(0) var input: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read_write> agents: array<SlimeAgent>;
@group(0) @binding(3) var<uniform> config: SlimeSimUniforms;

struct SlimeAgent {
    position: vec2<f32>,
    direction: f32,
    buffer: f32,
}

struct SlimeSimUniforms {
    pheromone_color: vec4<f32>,
    agent_count: u32,
    screen_size: vec2<f32>,
};

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init_agents(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= config.agent_count) { return; }

    let theta = randomFloat(index) * 6.2831855;
    let radius = randomFloat(index + 1u) * 50.0;
    let center = config.screen_size * 0.5;

    agents[index].position = center + vec2(cos(theta), sin(theta)) * radius;
    agents[index].direction = randomFloat(index + 2u) * 6.2831855;
    agents[index].buffer = 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&agents)) { return; }

    var agent = agents[id.x];

    // --- Movement ---
    let move_speed: f32 = 50.0;       // units per second
    let delta_time: f32 = 0.016;      // assuming 60fps
    let trail_weight: f32 = 0.05;

    let screen_width: f32 = 800.0;    // hardcoded screen width
    let screen_height: f32 = 600.0;   // hardcoded screen height

    let dir = vec2<f32>(cos(agent.direction), sin(agent.direction));
    agent.position += dir * move_speed * delta_time;

    // Clamp to bounds
    agent.position = clamp(
        agent.position,
        vec2<f32>(0.0, 0.0),
        vec2<f32>(screen_width - 1.0, screen_height - 1.0)
    );

    // --- Trail writing to current output texture ---
    let coord = vec2<i32>(i32(agent.position.x), i32(agent.position.y));
    let oldTrail = textureLoad(output, coord);   // ✅ exactly 2 args
    textureStore(output, coord, oldTrail + vec4<f32>(trail_weight));

    // --- Save updated agent back ---
    agents[id.x] = agent;
}
