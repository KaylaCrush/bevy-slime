use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        render_resource::*,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use rand::Rng;

mod diffusion;
use diffusion::DiffusionComputePlugin;
use diffusion::DiffusionUniforms;

mod agent;
use agent::AgentComputePlugin;
use agent::AgentUniforms;


#[derive(Resource, Clone, ExtractResource)]
pub struct DiffusionImages {
    pub texture_a: Handle<Image>,
    pub texture_b: Handle<Image>,
}

const DISPLAY_FACTOR: u32 = 1;
const SIZE: UVec2 = UVec2::new(1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;
const NUM_AGENTS: u32 = 500000;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            ExtractResourcePlugin::<DiffusionImages>::default(),
            DiffusionComputePlugin,
            AgentComputePlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (switch_textures, update_agent_uniforms))
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, render_device: Res<RenderDevice>,) {
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(image.clone());
    let image1 = images.add(image);

    commands.spawn((
        Sprite {
            image: image0.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(DiffusionImages {
        texture_a: image0,
        texture_b: image1,
    });

    commands.insert_resource(DiffusionUniforms {
        decay: 0.8,
        diffusion_strength: 0.1,
        delta_time: 0.001,
        padding1: 0.0,
    });

    commands.insert_resource(AgentUniforms {
        move_speed: 40.0,
        turn_speed: 6.0,
        sensor_angle_degrees: 30.0,
        sensor_offset_dst: 15.0,
        sensor_size: 3,
        _pad0: 0,  // padding
        screen_size: SIZE.as_vec2(),
        color: LinearRgba::WHITE,
        delta_time: 0.001,
        frame: 0,
        _pad1: Vec2::ZERO,
    });
    // ---- Init agents on GPU ----
    let agent_count: u32 = NUM_AGENTS; // or whatever

    let mut rng = rand::thread_rng();
    let center = Vec2::new(SIZE.x as f32 * 0.5, SIZE.y as f32 * 0.5);

    let radius = (SIZE.x.min(SIZE.y) as f32) * 0.4; // spawn inside 40% of screen, tweak as desired

    let mut agents_cpu = Vec::with_capacity(agent_count as usize);

    for _ in 0..agent_count {
        // --- Random position inside circle ---
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        let r = radius * rng.random::<f32>().sqrt();
        let offset = Vec2::new(angle.cos() * r, angle.sin() * r);

        let position = center + offset;

        // --- Direction facing the center ---
        let dir_vec = (center - position).normalize_or_zero();
        let direction = dir_vec.y.atan2(dir_vec.x);

        agents_cpu.push(agent::SlimeAgent {
            position,
            direction,
            _pad: 0.0,
        });
    }

    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Slime Agents Buffer"),
        contents: bytemuck::cast_slice(&agents_cpu),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    commands.insert_resource(agent::AgentGpuBuffer {
        buffer,
    });


}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<DiffusionImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

fn update_agent_uniforms(
    time: Res<Time>,
    mut agent_uniforms: ResMut<AgentUniforms>,
    mut diffusion_uniforms: ResMut<DiffusionUniforms>,
) {
    agent_uniforms.delta_time = time.delta_secs();
    agent_uniforms.frame += 1;
    diffusion_uniforms.delta_time = time.delta_secs();
}
