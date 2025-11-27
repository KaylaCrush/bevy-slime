// Shared resources and GPU-friendly data types used across the app and render
// sub-app. This file defines the shader parameter structs, global uniforms,
// image handles, and small configuration flags used by the render node.
//
// Keep types declared here in sync with the WGSL shader layouts (names and
// field ordering must match).

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
    // New emission model: single target layer with a scalar amount
    pub emit_layer: u32,
    pub emit_amount: f32,
    pub _pad_emit: UVec2,
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
            emit_layer: 0,
            emit_amount: 0.0,
            _pad_emit: UVec2::ZERO,
        }
    }
}
impl SpeciesSettings {
    pub fn red() -> Self {
        Self {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            weights: Vec4::new(1.0, -1.0, -1.0, 0.0),
            emit_layer: 0,
            emit_amount: 1.0,
            _pad_emit: UVec2::ZERO,
            ..Default::default()
        }
    }
    pub fn green() -> Self {
        Self {
            color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            weights: Vec4::new(-1.0, 1.0, -1.0, 0.0),
            emit_layer: 1,
            emit_amount: 1.0,
            _pad_emit: UVec2::ZERO,
            ..Default::default()
        }
    }
    pub fn blue() -> Self {
        Self {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            weights: Vec4::new(-1.0, -1.0, 1.0, 0.0),
            emit_layer: 2,
            emit_amount: 1.0,
            _pad_emit: UVec2::ZERO,
            ..Default::default()
        }
    }
}

// Paths to shader assets
pub const AGENTS_SHADER_PATH: &str = "shaders/agents.wgsl";
pub const PHERO_SHADER_PATH: &str = "shaders/pheromones.wgsl";

pub const DISPLAY_FACTOR: u32 = 1;
pub const SIZE: UVec2 = UVec2::new(1920 / DISPLAY_FACTOR, 1080 / DISPLAY_FACTOR);
pub const WORKGROUP_SIZE: u32 = 16;
pub const NUM_PHEROMONES: usize = 3;

// Runtime-configurable pheromone system options. Defaults preserve current behavior.
#[derive(Resource, Clone, ExtractResource)]
pub struct PheromoneConfig {
    /// Number of pheromone layers (texture array depth). Default 3 to match legacy RGB.
    pub layer_count: u32,
    /// Optional brush target layer (default 0). Used by input compute to decide which layer to paint.
    pub brush_target_layer: u32,
    /// Layers that are universally attractive (positive weight for all species) and paint-only.
    pub universal_love_layers: Vec<u32>,
    /// Layers that are universally repulsive (negative weight for all species) and paint-only.
    pub universal_hate_layers: Vec<u32>,
    /// Additional paint-only layers (agents do not deposit). Love/hate layers are implicitly paint-only.
    pub paint_only_layers: Vec<u32>,
}

impl Default for PheromoneConfig {
    fn default() -> Self {
        Self {
            layer_count: NUM_PHEROMONES as u32,
            brush_target_layer: 0,
            universal_love_layers: Vec::new(),
            universal_hate_layers: Vec::new(),
            paint_only_layers: Vec::new(),
        }
    }
}

#[derive(
    Resource,
    Clone,
    bevy::render::extract_resource::ExtractResource,
    bevy::render::render_resource::ShaderType,
)]
pub struct GlobalUniforms {
    pub delta_time: f32,
    pub frame: u32,
    pub mouse_position: Vec2,
    pub screen_size: Vec2,

    pub left_button_pressed: u32,
    pub right_button_pressed: u32,
}

// Removed legacy PheromoneUniforms (RGBA-era). Diffusion/decay now live in per-layer params.

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

// Uniform used by agents to control extended pheromone path (layer count and enable flag)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, bevy::render::render_resource::ShaderType)]
pub struct PheroControlUniform {
    pub layer_count: u32,
    pub _pad: bevy::math::UVec3,
}

// Uniform passed to the input/brush compute shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, bevy::render::render_resource::ShaderType)]
pub struct BrushControlUniform {
    pub target_layer: u32,
    pub _mode: u32, // reserved
    pub _pad: bevy::math::UVec2,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct PheromoneLayerParamsBuffer {
    #[allow(dead_code)]
    pub buffer: bevy::render::render_resource::Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct PheromoneImages {
    pub texture_a: Handle<Image>,
    pub texture_b: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SpeciesGpuBuffer {
    pub buffer: bevy::render::render_resource::Buffer,
}

#[derive(Resource)]
pub struct AgentSimImageBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

// Removed legacy per-channel bind groups (extract/composite/env)

// Array-based pheromone bind groups (prev/next ping)
#[derive(Resource)]
pub struct PheroArrayEnvBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

#[derive(Resource)]
pub struct PheroArrayCompositeBindGroups(pub [bevy::render::render_resource::BindGroup; 2]);

// Extended per-species, per-layer weights/emission buffers (dense L floats per species)
#[derive(Resource, Clone, ExtractResource)]
pub struct SpeciesLayerWeights {
    pub weights: bevy::render::render_resource::Buffer,
    pub layer_count: u32,
    pub species_count: u32,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct AgentSimRunConfig {
    // Flags to control which simulation stages run. Useful for debugging or
    // for stepping parts of the pipeline individually:
    // - `run_copy_and_input`: enable the copy/input (brush) pass for pheromones
    // - `run_diffuse`: enable the diffusion/decay pass for pheromones
    // - `run_agents`: enable the agent compute pass
    pub run_copy_and_input: bool,
    pub run_diffuse: bool,
    pub run_agents: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_and_defaults() {
        // Verify NUM_PHEROMONES matches expected RGB channels
        assert_eq!(NUM_PHEROMONES, 3);

        // Verify default species settings match expectations
        let s = SpeciesSettings::default();
        assert_eq!(s.move_speed, 30.0);
        assert_eq!(s.turn_speed, 6.0);
        assert_eq!(s.sensor_angle_degrees, 30.0);
        assert_eq!(s.sensor_offset_dst, 35.0);
        assert_eq!(s.sensor_size, 1.0);
        assert_eq!(s.color.w, 1.0);
        // weights default to zero and emission defaults disabled
        assert_eq!(s.weights, Vec4::ZERO);
        assert_eq!(s.emit_layer, 0);
        assert_eq!(s.emit_amount, 0.0);
    }
}
