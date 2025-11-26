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

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
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
            ),
        )
        .run();
}
