use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        render_resource::*,
        extract_resource::ExtractResource,
    },
};

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

const DISPLAY_FACTOR: u32 = 4;
const SIZE: UVec2 = UVec2::new(1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;


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
            DiffusionComputePlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
        decay: 0.99,
        diffusion_strength: 0.25,
        padding0: 0.0,
        padding1: 0.0,
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