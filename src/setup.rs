// Setup and initialization of images, uniform resources, and agent buffers.
//
// This module is responsible for creating the RGBA display textures (two
// ping-ponged images and a temporary), allocating the pheromone array images
// (prev/next), and inserting the initial `GlobalUniforms` and `PheromoneUniforms`.
// It also creates the per-layer parameter buffer used for future per-channel
// features.

use bevy::prelude::*;
// Using Text2D-style overlay for the layer indicator
use bevy::render::render_resource::{BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;

use crate::agents;
use crate::pheromones::{PheromoneArrayImages, make_pheromone_array_images};
use crate::resources::{
    AgentSimRunConfig, GlobalUniforms, PheromoneConfig, PheromoneImages,
};
use crate::resources::{PheromoneLayerParam, PheromoneLayerParamsBuffer};
use crate::{DISPLAY_FACTOR, SIZE, NUM_AGENTS};

#[derive(Component)]
pub struct BrushLayerText;

#[derive(Resource, Clone, Copy)]
pub struct FpsCounter {
    pub acc_time: f32,
    pub frames: u32,
    pub fps: f32,
}

#[derive(Resource, Clone)]
pub struct PheromoneLayerParamsCpu {
    pub params: Vec<PheromoneLayerParam>, // diffusion/decay as base rates; color as display
}

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    phero_cfg: Res<PheromoneConfig>,
) {
    // Create two RGBA render targets (texture_a/texture_b) used for display
    // ping-ponging. No separate temp texture is required for the current pipeline.
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
    // No temp texture required

    commands.spawn((
        Sprite {
            image: image0.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    // Minimal on-screen text: show current brush target layer and FPS (top-left-ish)
    commands.insert_resource(FpsCounter { acc_time: 0.0, frames: 0, fps: 0.0 });
    commands.spawn((
        Text::new(format!("Layer: {} | FPS: --", phero_cfg.brush_target_layer)),
        TextFont { font_size: 18.0, ..default() },
        TextColor(Color::WHITE),
        Transform::from_translation(Vec3::new(
            - (SIZE.x as f32) * 0.5 + 16.0,
            (SIZE.y as f32) * 0.5 - 24.0,
            10.0,
        )),
        BrushLayerText,
    ));

    commands.insert_resource(PheromoneImages { texture_a: image0, texture_b: image1 });

    // ARRAY PHEROMONE IMAGES (prev/next)
    let layer_count = phero_cfg.layer_count.max(1);
    info!("Pheromones: layers = {layer_count}");
    let phero_array = make_pheromone_array_images(&mut images, layer_count);
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

    // Legacy PheromoneUniforms removed; using per-layer param buffer below

    // Per-layer params (diffusion, decay, color)
    // Define explicit colors for the first five layers:
    // 0: hate (red), 1: love (green), 2..4: agent-specific (purple, yellow, blue)
    let mut layer_params: Vec<PheromoneLayerParam> = Vec::with_capacity(layer_count as usize);
    let defaults = [
        (0.4, 0.7, Vec4::new(0.0, 0.0, 0.0, 1.0)), // 0 hate
        (0.4, 0.7, Vec4::new(0.2, 0.95, 0.2, 1.0)), // 1 love
        (0.5, 0.8, Vec4::new(0.8, 80.0 / 255.0, 120.0 / 255.0, 1.0)), // 2 purple
        (0.6, 0.85, Vec4::new(0.5, 0.9, 0.2, 1.0)), // 3 yellow
        (0.7, 0.9, Vec4::new(0.1, 0.2, 0.85, 1.0)), // 4 blue
    ];
    for i in 0..layer_count {
        let (diff, dec, col) = if (i as usize) < defaults.len() {
            defaults[i as usize]
        } else {
            (0.5, 0.8, Vec4::new(0.6, 0.6, 0.6, 1.0))
        };
        layer_params.push(PheromoneLayerParam {
            diffusion: diff,
            decay: dec,
            _pad0: 0.0,
            _pad1: 0.0,
            color: col,
        });
    }
    let layer_param_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Pheromone layer params"),
        contents: bytemuck::cast_slice(&layer_params),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(PheromoneLayerParamsBuffer {
        buffer: layer_param_buffer,
    });
    // Keep CPU copy of base rates/colors
    commands.insert_resource(PheromoneLayerParamsCpu { params: layer_params });

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
        NUM_AGENTS,
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
    // Convert world coordinates (affected by sprite display scale) to texture pixel coords
    let mut tex = (mouse_pos.0 / (crate::DISPLAY_FACTOR as f32)) + globals.screen_size / 2.0;
    // Clamp to texture bounds to avoid NaNs in shaders when off-screen
    tex.x = tex.x.clamp(0.0, globals.screen_size.x - 1.0);
    tex.y = tex.y.clamp(0.0, globals.screen_size.y - 1.0);
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

// Keep the on-screen label in sync with the current brush layer
pub fn update_brush_layer_text(
    cfg: Res<crate::resources::PheromoneConfig>,
    fps: Res<FpsCounter>,
    mut q: Query<&mut Text, With<BrushLayerText>>,
) {
    if !cfg.is_changed() { return; }
    for mut t in &mut q {
        let fps_disp = if fps.fps > 0.0 { format!("{:.0}", fps.fps) } else { "--".to_string() };
        *t = Text::new(format!("Layer: {} | FPS: {}", cfg.brush_target_layer, fps_disp));
    }
}

// Update FPS every ~0.25s and refresh the label text
pub fn update_fps_counter(
    time: Res<Time>,
    cfg: Res<crate::resources::PheromoneConfig>,
    mut counter: ResMut<FpsCounter>,
    mut q: Query<&mut Text, With<BrushLayerText>>,
) {
    counter.acc_time += time.delta_secs();
    counter.frames += 1;
    if counter.acc_time >= 0.25 {
        counter.fps = (counter.frames as f32) / counter.acc_time.max(1e-6);
        counter.acc_time = 0.0;
        counter.frames = 0;
        let fps_disp = format!("{:.0}", counter.fps);
        for mut t in &mut q {
            *t = Text::new(format!("Layer: {} | FPS: {}", cfg.brush_target_layer, fps_disp));
        }
    }
}

// Precompute per-frame diffusion/decay factors on CPU and upload to GPU buffer
pub fn update_layer_params_buffer(
    time: Res<Time>,
    cpu: Res<PheromoneLayerParamsCpu>,
    params_buf: Res<PheromoneLayerParamsBuffer>,
    queue: Res<bevy::render::renderer::RenderQueue>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 { return; }
    // Helper: per-frame factor = 1 - (1 - rate)^dt
    fn per_frame_factor(rate: f32, dt: f32) -> f32 {
        let base = 1.0 - rate;
        1.0 - base.powf(dt)
    }
    let mut upload: Vec<PheromoneLayerParam> = Vec::with_capacity(cpu.params.len());
    for p in cpu.params.iter() {
        upload.push(PheromoneLayerParam {
            diffusion: per_frame_factor(p.diffusion, dt),
            decay: per_frame_factor(p.decay, dt),
            _pad0: 0.0,
            _pad1: 0.0,
            color: p.color,
        });
    }
    queue.write_buffer(&params_buf.buffer, 0, bytemuck::cast_slice(&upload));
}
