use bevy::prelude::*;
use bevy::render::renderer::{RenderQueue, RenderDevice};
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::extract_resource::ExtractResource;
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};

pub const AGENT_WORKGROUP_SIZE: u32 = 64;
pub const NUM_AGENTS: u32 = 100000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct SlimeAgentData {
    pub position: Vec2,
    pub angle: f32,
    pub species_index: u32,
}

#[derive(Resource)]
pub struct AgentsCpu {
    pub list: Vec<SlimeAgentData>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct AgentGpuBuffer {
    pub buffer: Buffer,
}

pub fn sync_agents_to_gpu(
    agents_cpu: Res<AgentsCpu>,
    agents_gpu: Res<AgentGpuBuffer>,
    queue: Res<RenderQueue>,
) {
    if !agents_cpu.is_changed() {
        return;
    }

    queue.write_buffer(&agents_gpu.buffer, 0, bytemuck::cast_slice(&agents_cpu.list));
}

/// Initialize CPU agent list and GPU agent buffer, inserting `AgentsCpu` and `AgentGpuBuffer` resources.
pub fn init_agents(
    commands: &mut Commands,
    render_device: &RenderDevice,
    size: UVec2,
    num_agents: u32,
    species_count: u32,
) {
    let mut agents: Vec<SlimeAgentData> = Vec::with_capacity(num_agents as usize);
    // Use the crate's convenient RNG (renamed API) instead of deprecated `thread_rng`
    let mut rng = rand::rng();

    let center = Vec2::new(size.x as f32 * 0.5, size.y as f32 * 0.5);
    let radius = (size.x.min(size.y) as f32) * 0.4;
    for i in 0..num_agents {
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        let r = radius * rng.random_range(0.0_f32..1.0_f32).sqrt();
        let index = (i % species_count as u32) as u32;
        let pos = center + Vec2::new(angle.cos() * r, angle.sin() * r);
        let dir_vec = (center - pos).normalize_or_zero();
        let dir = dir_vec.y.atan2(dir_vec.x);
        agents.push(SlimeAgentData { position: pos, angle: dir, species_index: index });
    }

    // Keep CPU copy
    commands.insert_resource(AgentsCpu { list: agents.clone() });

    // GPU agent buffer
    let agent_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Slime agents buffer"),
        contents: bytemuck::cast_slice(&agents),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(AgentGpuBuffer { buffer: agent_buffer });
}
