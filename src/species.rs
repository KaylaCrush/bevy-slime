use bevy::prelude::*;
use bevy::math::Vec4;
use bevy::render::renderer::RenderDevice;
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};
use crate::resources::SpeciesSettings;

// Authoring helpers used by the app to assemble `SpeciesSettings` that are
// uploaded to the GPU. These helpers are intentionally small and tested below.

// Authoring-side behavior components for Slime-like agents/species.
// These are used to build GPU SpeciesSettings; we don't spawn per-agent ECS entities
// for performance. Instead we author a small set of species/archetypes.

#[derive(Component)]
pub struct SlimeSpecies; // marker entity representing a species/archetype

#[derive(Component, Deref, DerefMut)]
pub struct MoveSpeed(pub f32);

#[derive(Component, Deref, DerefMut)]
pub struct TurnSpeed(pub f32);

#[derive(Component)]
pub struct Sensor {
    pub angle_degrees: f32,
    pub offset_dst: f32,
    pub size: f32,
}

#[derive(Component, Deref, DerefMut)]
pub struct AgentColor(pub Vec4);

#[derive(Component)]
pub struct FollowsPheromone {
    pub channel: u32,
    pub strength: f32,
}

#[derive(Component)]
pub struct AvoidsPheromone {
    pub channel: u32,
    pub strength: f32,
}

#[derive(Component)]
pub struct EmitsPheromone {
    pub channel: u32,
    pub amount: f32,
}

#[allow(dead_code)]
pub struct SpeciesAuthoringPlugin;

impl Plugin for SpeciesAuthoringPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_default_species);
    }
}

// Helper to construct a species entity from authored components directly
pub fn build_species_settings_from_components(
    color: &AgentColor,
    move_speed: &MoveSpeed,
    turn_speed: &TurnSpeed,
    sensor: &Sensor,
    follow: Option<&FollowsPheromone>,
    avoid: Option<&AvoidsPheromone>,
    emit: Option<&EmitsPheromone>,
) -> SpeciesSettings {
    let mut settings = SpeciesSettings::default();
    settings.color = **color;
    settings.move_speed = **move_speed;
    settings.turn_speed = **turn_speed;
    settings.sensor_angle_degrees = sensor.angle_degrees;
    settings.sensor_offset_dst = sensor.offset_dst;
    settings.sensor_size = sensor.size;

    // Build weights: positive = follow, negative = avoid
    let mut weights = Vec4::ZERO;
    if let Some(f) = follow {
        let mask = channel_to_mask(f.channel);
        weights += mask * f.strength;
    }
    if let Some(a) = avoid {
        let mask = channel_to_mask(a.channel);
        weights -= mask * a.strength;
    }
    settings.weights = weights;

    // Build per-channel emission
    let mut emit_v = Vec4::ZERO;
    if let Some(e) = emit {
        let mask = channel_to_mask(e.channel);
        emit_v += mask * e.amount;
    }
    // Do not deposit into alpha channel; it's used only for display visibility
    emit_v.w = 0.0;
    settings.emit = emit_v;
    settings
}

fn channel_to_mask(channel: u32) -> Vec4 {
    // Ensure alpha (w) is 1.0 so the texture remains visible with alpha-blended sprites.
    match channel {
        0 => Vec4::new(1.0, 0.0, 0.0, 1.0),
        1 => Vec4::new(0.0, 1.0, 0.0, 1.0),
        2 => Vec4::new(0.0, 0.0, 1.0, 1.0),
        3 => Vec4::new(0.0, 0.0, 0.0, 1.0),
        _ => Vec4::new(0.0, 0.0, 0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_to_mask_basic() {
        assert_eq!(channel_to_mask(0), Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(channel_to_mask(1), Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(channel_to_mask(2), Vec4::new(0.0, 0.0, 1.0, 1.0));
        // out-of-range channels default to zero rgb + alpha 1
        assert_eq!(channel_to_mask(99), Vec4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn build_species_packs_weights_and_emit() {
        let color = AgentColor(Vec4::new(0.2, 0.3, 0.4, 1.0));
        let move_speed = MoveSpeed(12.0);
        let turn_speed = TurnSpeed(3.0);
        let sensor = Sensor { angle_degrees: 10.0, offset_dst: 5.0, size: 2.0 };
        let follow = FollowsPheromone { channel: 0, strength: 0.5 };
        let avoid = AvoidsPheromone { channel: 1, strength: 0.25 };
        let emit = EmitsPheromone { channel: 2, amount: 0.75 };

        let settings = build_species_settings_from_components(
            &color,
            &move_speed,
            &turn_speed,
            &sensor,
            Some(&follow),
            Some(&avoid),
            Some(&emit),
        );

        // color is copied
        assert_eq!(settings.color, Vec4::new(0.2, 0.3, 0.4, 1.0));
        // weights: follow on channel 0 (positive), avoid on channel 1 (negative)
        assert!(settings.weights.x > 0.0);
        assert!(settings.weights.y < 0.0);
        // emit: channel 2 set, alpha must be zero
        assert!(settings.emit.z > 0.0);
        assert_eq!(settings.emit.w, 0.0);
    }
}

// Spawn three default slime species to match the current shader/channel assumptions (RGB)
pub fn spawn_default_species(mut commands: Commands) {
    // Red species (channel 0)
    commands.spawn((
        SlimeSpecies,
        AgentColor(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        MoveSpeed(30.0),
        TurnSpeed(6.0),
        Sensor { angle_degrees: 30.0, offset_dst: 35.0, size: 1.0 },
        FollowsPheromone { channel: 0, strength: 1.0 },
        AvoidsPheromone { channel: 1, strength: 1.0 },
        EmitsPheromone { channel: 0, amount: 1.0 },
    ));

    // Green species (channel 1)
    commands.spawn((
        SlimeSpecies,
        AgentColor(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        MoveSpeed(30.0),
        TurnSpeed(6.0),
        Sensor { angle_degrees: 30.0, offset_dst: 35.0, size: 1.0 },
        FollowsPheromone { channel: 1, strength: 1.0 },
        AvoidsPheromone { channel: 2, strength: 1.0 },
        EmitsPheromone { channel: 1, amount: 1.0 },
    ));

    // Blue species (channel 2)
    commands.spawn((
        SlimeSpecies,
        AgentColor(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        MoveSpeed(30.0),
        TurnSpeed(6.0),
        Sensor { angle_degrees: 30.0, offset_dst: 35.0, size: 1.0 },
        FollowsPheromone { channel: 2, strength: 1.0 },
        AvoidsPheromone { channel: 0, strength: 1.0 },
        EmitsPheromone { channel: 2, amount: 1.0 },
    ));
}

/// Build a GPU buffer from authored SlimeSpecies entities and upload as SpeciesGpuBuffer resource.
/// If no species are authored, falls back to the default RGB trio.
pub fn upload_species_to_gpu(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    query: Query<
        (
            &AgentColor,
            &MoveSpeed,
            &TurnSpeed,
            &Sensor,
            Option<&FollowsPheromone>,
            Option<&AvoidsPheromone>,
            Option<&EmitsPheromone>,
        ),
        With<SlimeSpecies>,
    >,
)
{
    let mut species: Vec<SpeciesSettings> = query
        .iter()
        .map(|(color, move_speed, turn_speed, sensor, follow, avoid, emit)| {
            build_species_settings_from_components(color, move_speed, turn_speed, sensor, follow, avoid, emit)
        })
        .collect();

    if species.is_empty() {
        species = vec![
            SpeciesSettings::red(),
            SpeciesSettings::green(),
            SpeciesSettings::blue(),
        ];
    }

    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Species settings buffer"),
        contents: bytemuck::cast_slice(&species),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(crate::resources::SpeciesGpuBuffer { buffer });
}
