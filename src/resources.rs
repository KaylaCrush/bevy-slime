use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bytemuck::{Pod, Zeroable};

// Species settings (moved from main)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, bevy::render::render_resource::ShaderType)]
pub struct SpeciesSettings {
    pub move_speed: f32,
    pub turn_speed: f32,
    pub sensor_angle_degrees: f32,
    pub sensor_offset_dst: f32,
    pub sensor_size: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub color: Vec4,
    pub weights: Vec4,
    pub emit: Vec4,
}
impl Default for SpeciesSettings {
    fn default() -> Self {
        Self {
            move_speed: 30.0,
            turn_speed: 6.0,
            sensor_angle_degrees: 30.0,
            sensor_offset_dst: 35.0,
            sensor_size: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            weights: Vec4::ZERO,
            emit: Vec4::ZERO,
        }
    }
}
impl SpeciesSettings {
    pub fn red() -> Self {
        Self { color: Vec4::new(1.0, 0.0, 0.0, 1.0), weights: Vec4::new(1.0, -1.0, -1.0, 0.0), emit: Vec4::new(1.0, 0.0, 0.0, 0.0), ..Default::default() }
    }
    pub fn green() -> Self {
        Self { color: Vec4::new(0.0, 1.0, 0.0, 1.0), weights: Vec4::new(-1.0, 1.0, -1.0, 0.0), emit: Vec4::new(0.0, 1.0, 0.0, 0.0), ..Default::default() }
    }
    pub fn blue() -> Self {
        Self { color: Vec4::new(0.0, 0.0, 1.0, 1.0), weights: Vec4::new(-1.0, -1.0, 1.0, 0.0), emit: Vec4::new(0.0, 0.0, 1.0, 0.0), ..Default::default() }
    }
}

// Paths to shader assets
pub const AGENTS_SHADER_PATH: &str = "shaders/agents.wgsl";
pub const PHERO_SHADER_PATH: &str = "shaders/pheromones.wgsl";

pub const DISPLAY_FACTOR: u32 = 1;
pub const SIZE: UVec2 = UVec2::new(1920 / DISPLAY_FACTOR, 1080 / DISPLAY_FACTOR);
pub const WORKGROUP_SIZE: u32 = 8;
pub const NUM_PHEROMONES: usize = 3;

#[derive(Resource, Clone, bevy::render::extract_resource::ExtractResource, bevy::render::render_resource::ShaderType)]
pub struct GlobalUniforms {
    pub delta_time: f32,
    pub frame: u32,
    pub mouse_position: Vec2,
    pub screen_size: Vec2,

    pub left_button_pressed: u32,
    pub right_button_pressed: u32,
}

#[derive(Resource, Clone, bevy::render::extract_resource::ExtractResource, bevy::render::render_resource::ShaderType)]
pub struct PheromoneUniforms {
    pub diffusion: Vec4,
    pub decay: Vec4,
}

// Per-layer pheromone parameters (used by array-based shaders)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, bevy::render::render_resource::ShaderType)]
pub struct PheromoneLayerParam {
    pub diffusion: f32,
    pub decay: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub color: Vec4,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct PheromoneLayerParamsBuffer {
    pub buffer: bevy::render::render_resource::Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SlimeSimImages {
    pub texture_a: Handle<Image>,
    pub texture_b: Handle<Image>,
    pub temp_texture: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SpeciesGpuBuffer {
    pub buffer: bevy::render::render_resource::Buffer,
}

#[derive(Resource)]
pub struct SlimeSimImageBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

// Removed legacy per-channel bind groups (extract/composite/env)

// Array-based pheromone bind groups (prev/next ping)
#[derive(Resource)]
pub struct PheroArrayEnvBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

#[derive(Resource)]
pub struct PheroArrayCompositeBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

#[derive(Resource, Clone, ExtractResource)]
pub struct SlimeSimRunConfig {
    pub run_copy_and_input: bool,
    pub run_diffuse: bool,
    pub run_agents: bool,
}
