// Setup and initialization of images, uniform resources, and agent buffers.
//
// This module is responsible for creating the RGBA display textures (two
// ping-ponged images and a temporary), allocating the pheromone array images
// (prev/next), and inserting the initial `GlobalUniforms` and `PheromoneUniforms`.
// It also creates the per-layer parameter buffer used for future per-channel
// features.

use bevy::prelude::*;
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;

use crate::agents;
use crate::pheromones::{PheromoneArrayImages, make_pheromone_array_images};
use crate::resources::{
    AgentSimRunConfig, DISPLAY_FACTOR, GlobalUniforms, PheromoneImages, PheromoneUniforms, SIZE,
};
use crate::resources::{PheromoneLayerParam, PheromoneLayerParamsBuffer};

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
) {
    // Create two RGBA render targets (texture_a/texture_b) used for display
    // ping-ponging. A third `temp_texture` is used as a working target where
    // necessary.
    // TEXTURES
    let mut image = Image::new_target_texture(
        SIZE.x,
        SIZE.y,
        bevy::render::render_resource::TextureFormat::Rgba32Float,
    );
    image.asset_usage = bevy::asset::RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage = bevy::render::render_resource::TextureUsages::COPY_DST
        | bevy::render::render_resource::TextureUsages::STORAGE_BINDING
        | bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(image.clone());
    let image1 = images.add(image.clone());
    let image2 = images.add(image.clone());

    commands.spawn((
        Sprite {
            image: image0.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(PheromoneImages {
        texture_a: image0,
        texture_b: image1,
        temp_texture: image2,
    });

    // ARRAY PHEROMONE IMAGES (prev/next) - allocated for upcoming refactor
    let phero_array = make_pheromone_array_images(&mut images);
    commands.insert_resource::<PheromoneArrayImages>(phero_array);

    // GLOBAL UNIFORMS
    commands.insert_resource(GlobalUniforms {
        delta_time: 0.01,
        frame: 0,
        mouse_position: Vec2::new(-10000.0, -10000.0),
        screen_size: SIZE.as_vec2(),
        left_button_pressed: 0,
        right_button_pressed: 0,
    });

    commands.insert_resource(PheromoneUniforms {
        diffusion: Vec4::new(0.5, 0.3, 0.7, 0.0),
        decay: Vec4::new(0.8, 0.6, 0.9, 0.0),
    });

    // Per-layer params (diffusion, decay, color)
    let layer_params: Vec<PheromoneLayerParam> = vec![
        PheromoneLayerParam {
            diffusion: 0.5,
            decay: 0.8,
            _pad0: 0.0,
            _pad1: 0.0,
            color: Vec4::new(104.0 / 255.0, 80.0 / 255.0, 120.0 / 255.0, 1.0),
        },
        PheromoneLayerParam {
            diffusion: 0.3,
            decay: 0.6,
            _pad0: 0.0,
            _pad1: 0.0,
            color: Vec4::new(0.7, 0.9, 0.1, 1.0),
        },
        PheromoneLayerParam {
            diffusion: 0.7,
            decay: 0.9,
            _pad0: 0.0,
            _pad1: 0.0,
            color: Vec4::new(0.1, 0.2, 0.8, 1.0),
        },
    ];
    let layer_param_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Pheromone layer params"),
        contents: bytemuck::cast_slice(&layer_params),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(PheromoneLayerParamsBuffer {
        buffer: layer_param_buffer,
    });

    // Run config
    commands.insert_resource(AgentSimRunConfig {
        run_copy_and_input: true,
        run_diffuse: true,
        run_agents: true,
    });

    // Initialize agents (agent module takes care of CPU/GPU agent resources)
    let species_count = 3u32; // kept simple; authoring plugin can update later
    agents::init_agents(
        &mut commands,
        &render_device,
        SIZE,
        agents::NUM_AGENTS,
        species_count,
    );

    // Species GPU buffer is uploaded by species::upload_species_to_gpu during Startup
}

pub fn switch_textures(images: Res<PheromoneImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

pub fn update_globals_uniform(
    mouse_pos: Res<crate::input::MouseWorldPos>,
    mouse_button_state: Res<crate::input::MouseButtonState>,
    mut globals: ResMut<GlobalUniforms>,
    time: Res<Time>,
) {
    let mut tex = mouse_pos.0 + globals.screen_size / 2.0;
    tex.y = globals.screen_size.y - tex.y;
    globals.mouse_position = tex;
    globals.delta_time = time.delta_secs();
    globals.frame += 1;
    globals.left_button_pressed = if mouse_button_state.left_pressed {
        1
    } else {
        0
    };
    globals.right_button_pressed = if mouse_button_state.right_pressed {
        1
    } else {
        0
    };
}
