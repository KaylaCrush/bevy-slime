// Input handling utilities for mapping OS/window input into the simulation.
//
// - `MouseWorldPos` stores the mouse position in world (texture) coordinates
//   so shaders can read it via the `GlobalUniforms` uniform buffer.
// - `MouseButtonState` tracks left/right button pressed state for the brush.

use bevy::{input::keyboard, prelude::*};

pub struct InputPlugin;
impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                update_mouse_position,
                handle_button_input,
                handle_keyboard_input,
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
