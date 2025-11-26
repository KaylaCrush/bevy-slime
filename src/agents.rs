// Agent CPU/GPU bridge and helpers
//
// This module maintains a CPU-side `Vec` of agents (for easy initialization
// and UI manipulation) and a GPU storage buffer containing the same data in
// tightly-packed shader-friendly form. The compute shader `agents.wgsl` reads
// this storage buffer and performs per-agent updates, sensing pheromones from
// the `R32` texture array and depositing into it.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};
use rand::Rng;

pub const AGENT_WORKGROUP_SIZE: u32 = 64;
pub const NUM_AGENTS: u32 = 100000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct Agent {
    pub position: Vec2,
    pub angle: f32,
    pub species_index: u32,
}

/// Write the CPU `AgentsCpu` list into the GPU `AgentGpuBuffer`.
/// This is invoked each frame from the `setup::update` stage when the CPU-side
/// agents are modified. The GPU buffer is `COPY_DST` so we update it with a
/// `RenderQueue::write_buffer` call.

#[derive(Resource)]
pub struct AgentsCpu {
    pub list: Vec<Agent>,
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

    queue.write_buffer(
        &agents_gpu.buffer,
        0,
        bytemuck::cast_slice(&agents_cpu.list),
    );
}

/// Initialize CPU agent list and GPU agent buffer, inserting `AgentsCpu` and `AgentGpuBuffer` resources.
pub fn init_agents(
    commands: &mut Commands,
    render_device: &RenderDevice,
    size: UVec2,
    num_agents: u32,
    species_count: u32,
) {
    // Create agents using the pure helper so we can test the generation logic
    // independently of GPU buffer creation.
    let agents = generate_agents(size, num_agents, species_count);

    // Keep CPU copy
    commands.insert_resource(AgentsCpu {
        list: agents.clone(),
    });

    // GPU agent buffer
    let buffer_contents = agents_to_gpu_bytes(&agents);
    let agent_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Agent buffer"),
        contents: &buffer_contents,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(AgentGpuBuffer {
        buffer: agent_buffer,
    });
}

/// Convert a list of `Agent` to GPU-ready bytes.
/// This is a pure helper so we can unit-test byte packing independently.
pub fn agents_to_gpu_bytes(agents: &[Agent]) -> Vec<u8> {
    bytemuck::cast_slice(agents).to_vec()
}

/// Generate a deterministic list of `Agent` positioned in a disc
/// around the center of `size`. This is separated from `init_agents` so we
/// can unit-test the generation logic without requiring GPU resources.
pub fn generate_agents(size: UVec2, num_agents: u32, species_count: u32) -> Vec<Agent> {
    let mut agents: Vec<Agent> = Vec::with_capacity(num_agents as usize);
    // Use the crate's convenient RNG (renamed API)
    let mut rng = rand::rng();

    let center = Vec2::new(size.x as f32 * 0.5, size.y as f32 * 0.5);
    let radius = (size.x.min(size.y) as f32) * 0.4;
    for i in 0..num_agents {
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        let r = radius * rng.random_range(0.0_f32..1.0_f32).sqrt();
        let index = i % species_count;
        let pos = center + Vec2::new(angle.cos() * r, angle.sin() * r);
        let dir_vec = (center - pos).normalize_or_zero();
        let dir = dir_vec.y.atan2(dir_vec.x);
        agents.push(Agent {
            position: pos,
            angle: dir,
            species_index: index,
        });
    }
    agents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_agents_basic() {
        let size = UVec2::new(200, 100);
        let agents = generate_agents(size, 1000, 3);
        assert_eq!(agents.len(), 1000);

        // positions should be within bounds and species index in range
        for a in agents.iter() {
            assert!(a.position.x >= 0.0 && a.position.x <= size.x as f32);
            assert!(a.position.y >= 0.0 && a.position.y <= size.y as f32);
            assert!(a.species_index < 3);
            // angle should be finite
            assert!(a.angle.is_finite());
        }
    }

    #[test]
    fn generate_agents_zero() {
        let size = UVec2::new(100, 100);
        let agents = generate_agents(size, 0, 3);
        assert!(agents.is_empty());
    }

    #[test]
    fn generate_agents_species_one() {
        let size = UVec2::new(100, 100);
        let agents = generate_agents(size, 10, 1);
        assert_eq!(agents.len(), 10);
        for a in agents.iter() {
            assert_eq!(a.species_index, 0);
        }
    }

    #[test]
    fn agents_to_gpu_bytes_roundtrip() {
        let agents = vec![
            Agent {
                position: Vec2::new(10.0, 20.0),
                angle: std::f32::consts::FRAC_PI_2,
                species_index: 0,
            },
            Agent {
                position: Vec2::new(30.0, 40.0),
                angle: std::f32::consts::PI,
                species_index: 1,
            },
        ];

        let bytes = agents_to_gpu_bytes(&agents);
        // bytes should be non-empty and equal in size to the raw cast
        assert_eq!(bytes.len(), std::mem::size_of_val(&agents[..]));
        // bytes should match the original bytemuck cast
        assert_eq!(bytes.as_slice(), bytemuck::cast_slice::<Agent, u8>(&agents));
    }
}
