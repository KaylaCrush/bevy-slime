// Input handling utilities for mapping OS/window input into the simulation.
//
// - `MouseWorldPos` stores the mouse position in world (texture) coordinates
//   so shaders can read it via the `GlobalUniforms` uniform buffer.
// - `MouseButtonState` tracks left/right button pressed state for the brush.

use bevy::{input::keyboard, prelude::*};
use crate::resources::{PheromoneConfig, GlobalUniforms};
use bevy::input::mouse::MouseWheel;
use bevy::render::renderer::RenderDevice;
use crate::agents;
use crate::pheromones::make_pheromone_array_images;
use crate::pheromones::PheromoneArrayImages;
use crate::{SIZE, NUM_AGENTS, NUM_SPECIES};

pub struct InputPlugin;
impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                update_mouse_position,
                handle_button_input,
                handle_keyboard_input,
                handle_mouse_wheel_layer,
                handle_brush_hotkeys,
                handle_species_cycle_hotkey,
                handle_reset_hotkey,
            ),
        )
        .insert_resource(MouseWorldPos(Vec2::ZERO))
        .insert_resource(MouseButtonState {
            left_pressed: false,
            right_pressed: false,
        });
    }
}

#[derive(Resource)]
pub struct MouseWorldPos(pub Vec2);

fn update_mouse_position(
    mut cursor_moved_events: MessageReader<CursorMoved>,
    mut mouse_pos: ResMut<MouseWorldPos>,
    cameras: Query<(&Camera, &GlobalTransform)>,
) {
    // Read the most recent cursor move event (if any) and convert to world
    // coordinates using the active 2D camera. The shader expects mouse
    // coordinates in the same space as the RGBA/pheromone textures.
    if let (Some(ev), Ok((camera, camera_transform))) =
        (cursor_moved_events.read().last(), cameras.single())
        && let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, ev.position)
    {
        mouse_pos.0 = world_pos;
    }
}

#[derive(Resource)]
pub struct MouseButtonState {
    pub left_pressed: bool,
    pub right_pressed: bool,
}

fn handle_button_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_button_state: ResMut<MouseButtonState>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        mouse_button_state.left_pressed = true;
    }
    if buttons.just_released(MouseButton::Left) {
        mouse_button_state.left_pressed = false;
    }
    if buttons.just_pressed(MouseButton::Right) {
        mouse_button_state.right_pressed = true;
    }
    if buttons.just_released(MouseButton::Right) {
        mouse_button_state.right_pressed = false;
    }
}

fn handle_keyboard_input(keyboard_input: Res<ButtonInput<keyboard::KeyCode>>) {
    if keyboard_input.just_pressed(keyboard::KeyCode::Escape) {
        println!("Escape key pressed. Exiting...");
        std::process::exit(0);
    }
}

// Press 'C' to cycle species mapping (offset used by agent shader)
fn handle_species_cycle_hotkey(
    keyboard_input: Res<ButtonInput<keyboard::KeyCode>>,
    mut globals: ResMut<crate::resources::GlobalUniforms>,
) {
    if keyboard_input.just_pressed(keyboard::KeyCode::KeyC) {
        let count = globals.species_count.max(1);
        globals.species_offset = (globals.species_offset + 1) % count;
    }
}

// Mouse wheel cycles brush target layer (with wrap)
fn handle_mouse_wheel_layer(
    mut wheel: MessageReader<MouseWheel>,
    mut cfg: ResMut<PheromoneConfig>,
) {
    let mut delta: i32 = 0;
    for ev in wheel.read() {
        // Positive y scrolls up; negative scrolls down
        if ev.y > 0.0 {
            delta += 1;
        } else if ev.y < 0.0 {
            delta -= 1;
        }
    }
    if delta != 0 {
        let layers = cfg.layer_count.max(1) as i32;
        let cur = cfg.brush_target_layer as i32;
        let mut next = (cur + delta) % layers;
        if next < 0 { next += layers; }
        cfg.brush_target_layer = next as u32;
    }
}

// Number keys 0-9 set brush target layer directly (clamped to available layers)
fn handle_brush_hotkeys(
    keyboard_input: Res<ButtonInput<keyboard::KeyCode>>,
    mut cfg: ResMut<PheromoneConfig>,
) {
    // Map numeric keys to indices
    let keys = [
        keyboard::KeyCode::Digit0,
        keyboard::KeyCode::Digit1,
        keyboard::KeyCode::Digit2,
        keyboard::KeyCode::Digit3,
        keyboard::KeyCode::Digit4,
        keyboard::KeyCode::Digit5,
        keyboard::KeyCode::Digit6,
        keyboard::KeyCode::Digit7,
        keyboard::KeyCode::Digit8,
        keyboard::KeyCode::Digit9,
    ];
    let mut set: Option<u32> = None;
    for (idx, key) in keys.iter().enumerate() {
        if keyboard_input.just_pressed(*key) {
            set = Some(idx as u32);
            break;
        }
    }
    if let Some(mut v) = set {
        let max_layer = cfg.layer_count.saturating_sub(1);
        if v > max_layer { v = max_layer; }
        cfg.brush_target_layer = v;
    }
}

// Press 'R' to reset simulation: agents reinitialized, pheromones cleared, globals reset
fn handle_reset_hotkey(
    keyboard_input: Res<ButtonInput<keyboard::KeyCode>>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut images: ResMut<Assets<Image>>,
    phero_cfg: Res<PheromoneConfig>,
    mut globals: ResMut<GlobalUniforms>,
    mut mouse_buttons: ResMut<MouseButtonState>,
) {
    if !keyboard_input.just_pressed(keyboard::KeyCode::KeyR) {
        return;
    }
    // Re-init agents (positions, angles, species indices)
    agents::init_agents(
        &mut commands,
        &render_device,
        SIZE,
        NUM_AGENTS,
        NUM_SPECIES,
    );

    // Clear pheromones by allocating fresh array images
    let layers = phero_cfg.layer_count.max(1);
    let array_images = make_pheromone_array_images(&mut images, layers);
    commands.insert_resource::<PheromoneArrayImages>(array_images);

    // Reset globals / input state
    globals.frame = 0;
    globals.species_offset = 0;
    globals.left_button_pressed = 0;
    globals.right_button_pressed = 0;
    globals.mouse_position = Vec2::new(-10000.0, -10000.0);
    mouse_buttons.left_pressed = false;
    mouse_buttons.right_pressed = false;
}
