use bevy::prelude::*;
use bevy::math::Vec4;

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
pub struct AuthoringPlugin;

impl Plugin for AuthoringPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_default_species);
    }
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

// Helper to construct a species entity from authored components directly
pub fn build_species_settings_from_components(
    color: &AgentColor,
    move_speed: &MoveSpeed,
    turn_speed: &TurnSpeed,
    sensor: &Sensor,
    follow: Option<&FollowsPheromone>,
    avoid: Option<&AvoidsPheromone>,
    emit: Option<&EmitsPheromone>,
) -> crate::SpeciesSettings {
    let mut settings = crate::SpeciesSettings::default();
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
