use bevy::{input::keyboard, prelude::*};

pub struct InputPlugin;
impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_mouse_position, handle_button_input, handle_keyboard_input))
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
    if let Some(ev) = cursor_moved_events.read().last() {
        if let Ok((camera, camera_transform)) = cameras.single() {
            if let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, ev.position) {
                mouse_pos.0 = world_pos;
            }
        }
    }
}

#[derive(Resource)]
pub struct MouseButtonState{
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

fn handle_keyboard_input(
    keyboard_input: Res<ButtonInput<keyboard::KeyCode>>,
) {
    if keyboard_input.just_pressed(keyboard::KeyCode::Escape) {
        println!("Escape key pressed. Exiting...");
        std::process::exit(0);
    }
}
