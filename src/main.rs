// Application entry: assemble Bevy app, register plugins, and wire startup
// systems. The order here matters: we ensure species and buffers are created
// during `Startup` before the render sub-app extracts resources.

use bevy::prelude::*;
use bevy::window::{Window, WindowMode, WindowPlugin};

mod agents;
mod input;
mod pheromones;
mod render;
mod resources;
mod setup;
mod species;

use input::InputPlugin;
use render::AgentSimComputePlugin;
use resources::PheromoneConfig;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        // Insert runtime pheromone config with safe defaults (RGB, legacy behavior)
        // Configure 5 pheromone layers by default with universal hate/love paint-only channels
        .insert_resource(PheromoneConfig {
            layer_count: 5,
            brush_target_layer: 1, // default to painting "love"
            universal_love_layers: vec![1],
            universal_hate_layers: vec![0],
            paint_only_layers: vec![], // universal love/hate are implicitly paint-only
        })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-slime".into(),
                        mode: WindowMode::BorderlessFullscreen(
                            bevy::window::MonitorSelection::Primary,
                        ),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest()),
            AgentSimComputePlugin,
            InputPlugin,
        ))
        // Startup systems: spawn species, upload species buffer, and create
        // textures/agents. The chain ensures species are created before we
        // attempt to upload them to the GPU.
        .add_systems(
            Startup,
            (
                species::spawn_default_species,
                species::upload_species_to_gpu,
                setup::setup,
            )
                .chain(),
        )
        // Update systems: alternate display textures, push CPU agent changes
        // to the GPU, and refresh global uniforms (mouse/frames/time).
        .add_systems(
            Update,
            (
                setup::switch_textures,
                agents::sync_agents_to_gpu,
                setup::update_globals_uniform,
                setup::update_brush_layer_text,
                setup::update_fps_counter,
                setup::update_layer_params_buffer,
            ),
        )
        .run();
}
