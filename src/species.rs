// Species authoring and GPU upload helpers.
//
// This module provides a compact authoring API (components such as
// `MoveSpeed`, `Sensor`, and `EmitsPheromone`) that are attached to a small
// number of species entities. Those entities are then converted into a tightly
// packed `SpeciesSettings` buffer uploaded to the GPU for use by the agent
// compute shader.

use crate::resources::SpeciesSettings;
use bevy::math::Vec4;
use bevy::prelude::*;
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
use crate::resources::{PheromoneConfig, SpeciesLayerWeights};

// Authoring helpers used by the app to assemble `SpeciesSettings` that are
// uploaded to the GPU. These helpers are intentionally small and tested below.

// Authoring-side behavior components for Agent-like agents/species.
// These are used to build GPU SpeciesSettings; we don't spawn per-agent ECS entities
// for performance. Instead we author a small set of species/archetypes.

#[derive(Component)]
pub struct AgentSpecies; // marker entity representing a species/archetype

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

// Legacy follow/avoid components removed; LayerWeights now encode full sensing biases.

#[derive(Component)]
pub struct EmitsPheromone {
    pub channel: u32,
    pub amount: f32,
}

// Optional per-species arrays for L-layer sensing weights
#[derive(Component, Deref, DerefMut)]
pub struct LayerWeights(pub Vec<f32>);

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
    emit: Option<&EmitsPheromone>,
) -> SpeciesSettings {
    // Build emission: single layer index + amount (weights now handled directly into dense buffer)

    // Build emission: single layer index + amount
    let mut emit_layer = 0u32;
    let mut emit_amount = 0.0f32;
    if let Some(e) = emit {
        emit_layer = e.channel;
        emit_amount = e.amount;
    }

    SpeciesSettings {
        move_speed: **move_speed,
        turn_speed: **turn_speed,
        sensor_angle_degrees: sensor.angle_degrees,
        sensor_offset_dst: sensor.offset_dst,
        sensor_size: sensor.size,
        color: **color,
        emit_layer,
        emit_amount,
        _pad_emit: UVec2::ZERO,
        ..Default::default()
    }
}

// Spawn three default agent species to match the current shader/channel assumptions (RGB)
pub fn spawn_default_species(mut commands: Commands) {
    // With extended layers: 0=hate, 1=love, 2..4 agent-specific
    // Red species (channel 2): sprinter (very fast, low turning, narrow sensing)
    commands.spawn((
        AgentSpecies,
        AgentColor(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        MoveSpeed(90.0),
        TurnSpeed(2.0),
        Sensor {
            angle_degrees: 12.0,
            offset_dst: 25.0,
            size: 0.0,
        },
        EmitsPheromone { channel: 2, amount: 0.6 },
        // Layer weights override: emphasize its own channel strongly, avoid next
        // [L0 hate, L1 love, L2 self, L3 next, L4 other]
        LayerWeights(vec![-1.0, 1.0, 1.5, -1.0, 0.2]),
    ));

    // Green species (channel 3): twitchy scout (medium speed, high turning, wide sensing)
    commands.spawn((
        AgentSpecies,
        AgentColor(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        MoveSpeed(28.0),
        TurnSpeed(12.0),
        Sensor {
            angle_degrees: 60.0,
            offset_dst: 30.0,
            size: 0.0,
        },
        EmitsPheromone { channel: 3, amount: 1.2 },
        // Broader sensing with moderate biases
        LayerWeights(vec![-1.0, 1.0, 0.2, 1.0, -0.6]),
    ));

    // Blue species (channel 4): whirl drifter (mid speed, high turning, wide sensing)
    commands.spawn((
        AgentSpecies,
        AgentColor(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        MoveSpeed(42.0),
        TurnSpeed(10.0),
        Sensor {
            angle_degrees: 75.0,
            offset_dst: 28.0,
            size: 0.0,
        },
        EmitsPheromone { channel: 4, amount: 2.0 },
        // Broader curiosity: attracted to love(1) and self(4), slight avoidance of purple(2)
        // [L0 hate, L1 love, L2 purple, L3 yellow, L4 blue]
        LayerWeights(vec![-1.0, 1.0, -0.6, 0.2, 1.1]),
    ));
}

/// Build a GPU buffer from authored AgentSpecies entities and upload as SpeciesGpuBuffer resource.
/// If no species are authored, falls back to the default RGB trio.
#[allow(clippy::type_complexity)]
pub fn upload_species_to_gpu(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    phero_cfg: Res<PheromoneConfig>,
    query: Query<
        (
            &AgentColor,
            &MoveSpeed,
            &TurnSpeed,
            &Sensor,
            Option<&EmitsPheromone>,
            Option<&LayerWeights>,
        ),
        With<AgentSpecies>,
    >,
) {
    // Collect species settings and optional extended arrays aligned by index
    let mut species: Vec<SpeciesSettings> = Vec::new();
    let mut layer_w: Vec<Option<Vec<f32>>> = Vec::new();
    for (color, move_speed, turn_speed, sensor, emit, wext) in query.iter() {
        species.push(build_species_settings_from_components(
            color, move_speed, turn_speed, sensor, emit,
        ));
        layer_w.push(wext.map(|v| v.0.clone()));
    }

    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Species settings buffer"),
        contents: bytemuck::cast_slice(&species),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(crate::resources::SpeciesGpuBuffer { buffer });

    // Build dense extended arrays (weights) sized species_count * L.
    let layer_count = phero_cfg.layer_count.max(1);
    let species_count = species.len() as u32;
    let mut weights: Vec<f32> = vec![0.0; (layer_count * species_count) as usize];
    for (si, _s) in species.iter().enumerate() {
        let base = (si as u32) * layer_count;
        // Extended overrides if provided
        if let Some(w_override) = &layer_w.get(si).and_then(|o| o.as_ref()) {
            let n = layer_count.min(w_override.len() as u32);
            for li in 0..n { weights[(base + li) as usize] = w_override[li as usize]; }
        }
    }

    // Apply universal and paint-only rules
    let love_set: std::collections::HashSet<u32> = phero_cfg.universal_love_layers.iter().copied().collect();
    let hate_set: std::collections::HashSet<u32> = phero_cfg.universal_hate_layers.iter().copied().collect();

    for si in 0..species_count {
        let base = si * layer_count;
        for li in 0..layer_count {
            // universal loved/hated -> override weight regardless of authored species weight
            if love_set.contains(&li) {
                weights[(base + li) as usize] = 1.0;
            }
            if hate_set.contains(&li) {
                weights[(base + li) as usize] = -1.0;
            }
        }
    }

    let weights_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Species extended weights"),
        contents: bytemuck::cast_slice(&weights),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(SpeciesLayerWeights { weights: weights_buf });
}

#[cfg(test)]
/// Collect a `Vec<SpeciesSettings>` from an iterator of component references.
/// This is a small pure helper so we can unit-test the translation from
/// authoring components to the GPU-friendly `SpeciesSettings` layout.
pub fn collect_species_settings_from_refs<'a, I>(iter: I) -> Vec<SpeciesSettings>
where
    I: IntoIterator<
        Item = (
            &'a AgentColor,
            &'a MoveSpeed,
            &'a TurnSpeed,
            &'a Sensor,
            Option<&'a EmitsPheromone>,
        ),
    >,
{
    iter.into_iter()
        .map(|(color, move_speed, turn_speed, sensor, emit)| {
            build_species_settings_from_components(color, move_speed, turn_speed, sensor, emit)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn build_species_packs_weights_and_emit() {
        let color = AgentColor(Vec4::new(0.2, 0.3, 0.4, 1.0));
        let move_speed = MoveSpeed(12.0);
        let turn_speed = TurnSpeed(3.0);
        let sensor = Sensor {
            angle_degrees: 10.0,
            offset_dst: 5.0,
            size: 2.0,
        };
        let emit = EmitsPheromone {
            channel: 2,
            amount: 0.75,
        };

        let settings = build_species_settings_from_components(
            &color,
            &move_speed,
            &turn_speed,
            &sensor,
            Some(&emit),
        );

        // color is copied
        assert_eq!(settings.color, Vec4::new(0.2, 0.3, 0.4, 1.0));
        // weights are no longer stored on SpeciesSettings; validated via upload path
        // emit: single-layer 2 set with amount
        assert_eq!(settings.emit_layer, 2);
        assert!(settings.emit_amount > 0.0);
    }

    #[test]
    fn build_species_no_follow_avoid_emit() {
        let color = AgentColor(Vec4::new(0.1, 0.2, 0.3, 1.0));
        let move_speed = MoveSpeed(10.0);
        let turn_speed = TurnSpeed(2.0);
        let sensor = Sensor {
            angle_degrees: 15.0,
            offset_dst: 5.0,
            size: 1.0,
        };

        let settings = build_species_settings_from_components(
            &color,
            &move_speed,
            &turn_speed,
            &sensor,
            None,
        );

        // no emission configured
        assert_eq!(settings.emit_amount, 0.0);
    }

    #[test]
    fn collect_species_settings_from_refs_basic() {
        let color = AgentColor(Vec4::new(0.2, 0.3, 0.4, 1.0));
        let move_speed = MoveSpeed(12.0);
        let turn_speed = TurnSpeed(3.0);
        let sensor = Sensor {
            angle_degrees: 10.0,
            offset_dst: 5.0,
            size: 2.0,
        };
        let emit = EmitsPheromone {
            channel: 2,
            amount: 0.75,
        };

        let items = vec![(&color, &move_speed, &turn_speed, &sensor, Some(&emit))];

        let list = collect_species_settings_from_refs(items);
        assert_eq!(list.len(), 1);
        let s = &list[0];
        assert_eq!(s.color, Vec4::new(0.2, 0.3, 0.4, 1.0));
        // weights are applied in dense array during upload; no Vec4 on SpeciesSettings
        assert_eq!(s.emit_layer, 2);
        assert!(s.emit_amount > 0.0);
    }
}
