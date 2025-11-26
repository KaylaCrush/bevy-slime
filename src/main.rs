use bevy::prelude::*;
use bevy::window::{Window, WindowMode, WindowPlugin};

mod species;
mod pheromones;
mod agents;
mod input;
mod resources;
mod setup;
mod render;

use input::InputPlugin;
use render::SlimeSimComputePlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window { title: "bevy-slime".into(), mode: WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Primary), ..Default::default() }),
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest()),
            SlimeSimComputePlugin,
            InputPlugin,
        ))
        .add_systems(Startup, (species::spawn_default_species, species::upload_species_to_gpu, setup::setup).chain())
        .add_systems(Update, (setup::switch_textures, agents::sync_agents_to_gpu, setup::update_globals_uniform))
        .run();
}
