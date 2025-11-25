//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    asset::{RenderAssetUsages,},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::{ExtractResource, ExtractResourcePlugin}, render_asset::RenderAssets, render_graph::{self, RenderGraph, RenderLabel}, render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer, storage_buffer, storage_buffer_read_only},
            *,
        }, renderer::{RenderContext, RenderDevice, RenderQueue}, texture::GpuImage
    },
    shader::PipelineCacheError,
    window::{WindowMode},
};
use std::borrow::Cow;
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use bevy::math::Vec4;

mod authoring;
use authoring::{
    SlimeSpecies, AgentColor, MoveSpeed, TurnSpeed, Sensor, FollowsPheromone, AvoidsPheromone,
    EmitsPheromone,
};

mod input;
use input::{InputPlugin, MouseWorldPos, MouseButtonState};

/// This example uses a shader source file from the assets subdirectory
const AGENTS_SHADER_PATH: &str = "shaders/agents.wgsl";
const PHERO_SHADER_PATH: &str = "shaders/pheromones.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: UVec2 = UVec2::new(1920 / DISPLAY_FACTOR, 1080 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;
const AGENT_WORKGROUP_SIZE: u32 = 64;
const NUM_AGENTS: u32 = 100000;
const NUM_PHEROMONES: usize = 3;
// Species are authored via ECS and packed at runtime; species count is dynamic now.


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct SlimeAgentData {
    position: Vec2,
    angle: f32,
    species_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct SpeciesSettings {
    // movement / sensing
    pub move_speed: f32,
    pub turn_speed: f32,
    pub sensor_angle_degrees: f32,
    pub sensor_offset_dst: f32,

    // sensor footprint
    pub sensor_size: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,

    // visuals and behavior
    pub color: Vec4,
    // per-channel weights: positive means follow, negative means avoid
    pub weights: Vec4,
    // per-channel emission amounts
    pub emit: Vec4,
}
impl Default for SpeciesSettings {
    fn default() -> Self {
        Self {
            move_speed: 30.0,
            turn_speed: 6.0,
            sensor_angle_degrees: 30.0,
            sensor_offset_dst: 35.0,

            sensor_size: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,

            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            weights: Vec4::ZERO,
            emit: Vec4::ZERO,
        }
    }
}
impl SpeciesSettings {
    fn red() -> Self {
        Self {
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            weights: Vec4::new(1.0, -1.0, -1.0, 0.0),
            emit: Vec4::new(1.0, 0.0, 0.0, 0.0),
            ..Default::default()
        }
    }
    fn green() -> Self {
        Self {
            color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            weights: Vec4::new(-1.0, 1.0, -1.0, 0.0),
            emit: Vec4::new(0.0, 1.0, 0.0, 0.0),
            ..Default::default()
        }
    }
    fn blue() -> Self {
        Self {
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            weights: Vec4::new(-1.0, -1.0, 1.0, 0.0),
            emit: Vec4::new(0.0, 0.0, 1.0, 0.0),
            ..Default::default()
        }
    }
}


#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct GlobalUniforms {
    pub delta_time: f32,
    pub frame: u32,
    pub mouse_position: Vec2,
    pub screen_size: Vec2,

    pub left_button_pressed: u32,
    pub right_button_pressed: u32,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct PheromoneUniforms {
    // Per-channel diffusion and decay for RGBA pheromones
    pub diffusion: Vec4,
    pub decay: Vec4,
}

#[derive(Resource, Clone, ExtractResource)]
struct SlimeSimImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
    temp_texture: Handle<Image>,
}

#[derive(Resource)]
struct AgentsCpu {
    pub list: Vec<SlimeAgentData>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct AgentGpuBuffer {
    pub buffer: Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SpeciesGpuBuffer {
    pub buffer: Buffer,
}

#[derive(Resource)]
struct SlimeSimImageBindGroups([BindGroup; 2]);

#[derive(Resource)]
struct SlimeSimExtractBindGroups([BindGroup; 2]);

#[derive(Resource)]
struct SlimeSimCompositeBindGroups([BindGroup; 2]);

#[derive(Resource)]
struct SlimeSimPheroEnvBindGroups([[BindGroup; 2]; 3]);

#[derive(Resource, Clone, ExtractResource)]
struct PheromoneImages {
    // main pheromone textures (r32float), size = NUM_PHEROMONES
    phero: Vec<Handle<Image>>,
    // temp buffers for future passes (diffuse), not used yet in step 1
    temp: Vec<Handle<Image>>,
    // scratch / intermediate buffer per pheromone (r32float)
    scratch: Vec<Handle<Image>>,
}



fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resizable: false,
                        mode: WindowMode::Fullscreen(MonitorSelection::Primary, VideoModeSelection::Current),
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            SlimeSimComputePlugin,
            InputPlugin,
        ))
        // Ensure default species exist before GPU buffers and agents are initialized
        .add_systems(Startup, (authoring::spawn_default_species, setup).chain())
        .add_systems(Update, (switch_textures, sync_agents_to_gpu, update_globals_uniform))
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    species_query: Query<
        (
            &AgentColor,
            &MoveSpeed,
            &TurnSpeed,
            &Sensor,
            Option<&FollowsPheromone>,
            Option<&AvoidsPheromone>,
            Option<&EmitsPheromone>,
        ),
        With<SlimeSpecies>,
    >,
) {
    // ---- TEXTURES (unchanged) ----
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
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

    commands.insert_resource(SlimeSimImages {
        texture_a: image0,
        texture_b: image1,
        temp_texture: image2,
    });

    // ---- PHEROMONE IMAGES (r32float), for step 1 just allocate storage ----
    let mut phero_handles: Vec<Handle<Image>> = Vec::with_capacity(NUM_PHEROMONES);
    let mut temp_handles: Vec<Handle<Image>> = Vec::with_capacity(NUM_PHEROMONES);
    let mut scratch_handles: Vec<Handle<Image>> = Vec::with_capacity(NUM_PHEROMONES);
    for _ in 0..NUM_PHEROMONES {
        let mut p_img = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Float);
        p_img.asset_usage = RenderAssetUsages::RENDER_WORLD;
        p_img.texture_descriptor.usage =
            TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let t_img = p_img.clone();
        let s_img = p_img.clone();
        let p_h = images.add(p_img);
        let t_h = images.add(t_img);
        let s_h = images.add(s_img);
        phero_handles.push(p_h);
        temp_handles.push(t_h);
        scratch_handles.push(s_h);
    }
    commands.insert_resource(PheromoneImages { phero: phero_handles, temp: temp_handles, scratch: scratch_handles });

    // ---- SPECIES BUFFER (authored species) ----
    let mut species_list: Vec<SpeciesSettings> = Vec::new();
    for (color, move_speed, turn_speed, sensor, follow, avoid, emit) in species_query.iter() {
        let settings = authoring::build_species_settings_from_components(
            color,
            move_speed,
            turn_speed,
            sensor,
            follow,
            avoid,
            emit,
        );
        species_list.push(settings);
    }
    if species_list.is_empty() {
        // Fallback to defaults if no authored species exist
        species_list = vec![
            SpeciesSettings::red(),
            SpeciesSettings::green(),
            SpeciesSettings::blue(),
        ];
    }
    let species_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Slime species buffer"),
        contents: bytemuck::cast_slice(&species_list),
        usage: BufferUsages::STORAGE,
    });
    commands.insert_resource(SpeciesGpuBuffer { buffer: species_buffer });

    // ---- CPU AGENTS ----
    let mut agents = Vec::with_capacity(NUM_AGENTS as usize);
    let mut rand = rand::rng();

    let center = Vec2::new(SIZE.x as f32 * 0.5, SIZE.y as f32 * 0.5);
    let radius = (SIZE.x.min(SIZE.y) as f32) * 0.4;
    let species_count = species_list.len().max(1) as u32;
    for i in 0..NUM_AGENTS {
        let angle = rand.random::<f32>() * std::f32::consts::TAU;
        let r = radius * rand.random::<f32>().sqrt(); // uniform distribution in circle
        let index = (i % species_count as u32) as u32;
        // let dir = angle;
        // let pos: Vec2 = Vec2::new(
            // rand.random::<f32>() * SIZE.x as f32,
            // rand.random::<f32>() * SIZE.y as f32,
        // );

        let pos = center + Vec2::new(angle.cos() * r, angle.sin() * r);

        // face toward center
        let dir_vec = (center - pos).normalize_or_zero();
        let dir = dir_vec.y.atan2(dir_vec.x);

        agents.push(SlimeAgentData {
            position: pos,
            angle: dir,
            species_index: index,
        });
    }

    // Keep CPU copy
    commands.insert_resource(AgentsCpu { list: agents.clone() });

    // ---- GPU AGENT BUFFER ----
    let agent_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Slime agents buffer"),
        contents: bytemuck::cast_slice(&agents),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(AgentGpuBuffer { buffer: agent_buffer });

    // species buffer already created above

    // ---- GLOBAL UNIFORMS ----
    commands.insert_resource(GlobalUniforms {
        delta_time: 0.01,             // small nonzero to avoid first-frame weirdness
        frame: 0,
        mouse_position: Vec2::new(-10000.0, -10000.0),
        screen_size: SIZE.as_vec2(),
        left_button_pressed: 0,
        right_button_pressed: 0,
    });

    // Pheromone per-channel parameters (RGBA). Tweak as desired.
    commands.insert_resource(PheromoneUniforms {
        diffusion: Vec4::new(0.5, 0.3, 0.7, 0.0),
        decay: Vec4::new(0.8, 0.6, 0.9, 0.0),
    });

    // Run configuration: allow running parts of the simulation independently.
    commands.insert_resource(SlimeSimRunConfig {
        run_copy_and_input: true,
        run_diffuse: true,
        run_agents: true,
    });
}


// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<SlimeSimImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

fn sync_agents_to_gpu(
    agents_cpu: Res<AgentsCpu>,
    agents_gpu: Res<AgentGpuBuffer>,
    queue: Res<RenderQueue>,
) {
    if !agents_cpu.is_changed() {
        return;
    }

    queue.write_buffer(
        &agents_gpu.buffer,
        0,
        bytemuck::cast_slice(&agents_cpu.list),
    );
}

fn update_globals_uniform(
    mouse_pos: Res<MouseWorldPos>,
    mouse_button_state: Res<MouseButtonState>,
    mut globals: ResMut<GlobalUniforms>,
    time: Res<Time>,
) {
    // world → texture coords
    let mut tex = mouse_pos.0 + globals.screen_size / 2.0;

    // flip Y for texture space
    tex.y = globals.screen_size.y - tex.y;

    globals.mouse_position = tex;
    globals.delta_time = time.delta_secs();
    globals.frame += 1;
    globals.left_button_pressed = if mouse_button_state.left_pressed { 1 } else { 0 };
    globals.right_button_pressed = if mouse_button_state.right_pressed { 1 } else { 0 };
}


struct SlimeSimComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SlimeSimLabel;

impl Plugin for SlimeSimComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<AgentGpuBuffer>::default(),
            ExtractResourcePlugin::<SlimeSimImages>::default(),
            ExtractResourcePlugin::<SpeciesGpuBuffer>::default(),
            ExtractResourcePlugin::<GlobalUniforms>::default(),
            ExtractResourcePlugin::<SlimeSimRunConfig>::default(),
            ExtractResourcePlugin::<PheromoneUniforms>::default(),
            ExtractResourcePlugin::<PheromoneImages>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_slime_sim_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SlimeSimLabel, SlimeSimNode::default());
        render_graph.add_node_edge(SlimeSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}


fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<SlimeSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    slime_agent_data: Res<AgentGpuBuffer>,
    slime_sim_images: Res<SlimeSimImages>,
    phero_images: Res<PheromoneImages>,
    species_settings: Res<SpeciesGpuBuffer>,
    globals: Res<GlobalUniforms>,
    pheros: Res<PheromoneUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // ---- TEXTURE VIEWS ----
    let Some(image_a) = gpu_images.get(&slime_sim_images.texture_a) else { return; };
    let Some(image_b) = gpu_images.get(&slime_sim_images.texture_b) else { return; };
    let Some(image_temp) = gpu_images.get(&slime_sim_images.temp_texture) else { return; };

    let view_a = &image_a.texture_view;
    let view_b = &image_b.texture_view;
    let view_temp = &image_temp.texture_view;

    // ---- GLOBAL UNIFORMS BUFFER ----
    let mut global_uniform_buffer = UniformBuffer::from(globals.into_inner());
    global_uniform_buffer.write_buffer(&render_device, &queue);

    // ---- PHEROMONE UNIFORMS BUFFER ----
    let mut pheromone_uniform_buffer = UniformBuffer::from(pheros.into_inner());
    pheromone_uniform_buffer.write_buffer(&render_device, &queue);

    // ---- BUILD TWO PING-PONG BIND GROUPS ----
    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            // binding 0: agent array
            BufferBinding {
                buffer: &slime_agent_data.buffer,
                offset: 0,
                size: None,
            },
            // binding 1: read
            view_a,
            // binding 2: write
            view_b,
            // binding 3: temp
            view_temp,
            // binding 4: global uniforms
            &global_uniform_buffer,
            // binding 5: species array
            BufferBinding {
                buffer: &species_settings.buffer,
                offset: 0,
                size: None,
            },
            // binding 6: pheromone params
            &pheromone_uniform_buffer,
        )),
    );

    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            // binding 0: agents
            BufferBinding {
                buffer: &slime_agent_data.buffer,
                offset: 0,
                size: None,
            },
            // binding 1: read
            view_b,
            // binding 2: write
            view_a,
            view_temp,
            // binding 3: global uniforms
            &global_uniform_buffer,
            // binding 4: species array
            BufferBinding {
                buffer: &species_settings.buffer,
                offset: 0,
                size: None,
            },
            // binding 6: pheromone params
            &pheromone_uniform_buffer,
        )),
    );

    commands.insert_resource(SlimeSimImageBindGroups([bind_group_0, bind_group_1]));

    // ---- PHEROMONE EXTRACT & COMPOSITE BIND GROUPS (2 indices) ----
    // get r32 views
        if phero_images.phero.len() >= NUM_PHEROMONES && phero_images.temp.len() >= NUM_PHEROMONES {
        let mut p_views: Vec<&TextureView> = Vec::with_capacity(NUM_PHEROMONES);
        let mut t_views: Vec<&TextureView> = Vec::with_capacity(NUM_PHEROMONES);
        let mut s_views: Vec<&TextureView> = Vec::with_capacity(NUM_PHEROMONES);
        for h in &phero_images.phero {
            if let Some(gi) = gpu_images.get(h) {
                p_views.push(&gi.texture_view);
            } else {
                return; // not ready yet
            }
        }
        for h in &phero_images.temp {
            if let Some(gi) = gpu_images.get(h) {
                t_views.push(&gi.texture_view);
            } else {
                return; // not ready yet
            }
        }
        for h in &phero_images.scratch {
            if let Some(gi) = gpu_images.get(h) {
                s_views.push(&gi.texture_view);
            } else {
                return; // not ready yet
            }
        }

        // index 0: latest RGBA output is view_b; write also goes to view_b for composite
        let extract_bg0 = render_device.create_bind_group(
            None,
            &pipeline.extract_bind_group_layout,
            &BindGroupEntries::sequential((
                view_b,
                p_views[0],
                p_views[1],
                p_views[2],
            )),
        );
        let composite_bg0 = render_device.create_bind_group(
            None,
            &pipeline.composite_bind_group_layout,
            &BindGroupEntries::sequential((
                p_views[0],
                p_views[1],
                p_views[2],
                view_b,
            )),
        );
        // index 1: latest RGBA output is view_a
        let extract_bg1 = render_device.create_bind_group(
            None,
            &pipeline.extract_bind_group_layout,
            &BindGroupEntries::sequential((
                view_a,
                p_views[0],
                p_views[1],
                p_views[2],
            )),
        );
        let composite_bg1 = render_device.create_bind_group(
            None,
            &pipeline.composite_bind_group_layout,
            &BindGroupEntries::sequential((
                p_views[0],
                p_views[1],
                p_views[2],
                view_a,
            )),
        );
        commands.insert_resource(SlimeSimExtractBindGroups([extract_bg0, extract_bg1]));
        commands.insert_resource(SlimeSimCompositeBindGroups([composite_bg0, composite_bg1]));

        // Step 3: pheromone env bind groups (copy/input/diffuse)
        let phero_env_groups: [[BindGroup; 2]; 3] = std::array::from_fn(|p| {
            // bg_copy_input: input=mainA, temp=scratch, output=mainB
            let bg_copy_input = render_device.create_bind_group(
                None,
                &pipeline.phero_env_bind_group_layout,
                &BindGroupEntries::sequential((
                    p_views[p], // input (mainA)
                    s_views[p], // temp/scratch
                    t_views[p], // output (mainB)
                    view_temp,
                    &global_uniform_buffer,
                    BufferBinding {
                        buffer: &species_settings.buffer,
                        offset: 0,
                        size: None,
                    },
                    &pheromone_uniform_buffer,
                )),
            );

            // bg_diffuse: input=scratch, temp=temp, output=mainA
            let bg_diffuse = render_device.create_bind_group(
                None,
                &pipeline.phero_env_bind_group_layout,
                &BindGroupEntries::sequential((
                    s_views[p],
                    t_views[p],
                    p_views[p],
                    view_temp,
                    &global_uniform_buffer,
                    BufferBinding {
                        buffer: &species_settings.buffer,
                        offset: 0,
                        size: None,
                    },
                    &pheromone_uniform_buffer,
                )),
            );

            [bg_copy_input, bg_diffuse]
        });
        commands.insert_resource(SlimeSimPheroEnvBindGroups(phero_env_groups));
    }
}

#[derive(Resource)]
struct SlimeSimPipeline {
    texture_bind_group_layout: BindGroupLayout,
    agent_sim_pipeline: CachedComputePipelineId,
    diffuse_pipeline: CachedComputePipelineId,
    copy_pipeline: CachedComputePipelineId,
    input_pipeline: CachedComputePipelineId,
    // step 1
    extract_bind_group_layout: BindGroupLayout,
    extract_pipeline: CachedComputePipelineId,
    // step 2
    composite_bind_group_layout: BindGroupLayout,
    composite_pipeline: CachedComputePipelineId,
    // step 3
    phero_env_bind_group_layout: BindGroupLayout,
    copy_phero_pipelines: [CachedComputePipelineId; 3],
    _input_phero_pipelines: [CachedComputePipelineId; 3],
    _diffuse_phero_pipelines: [CachedComputePipelineId; 3],
}

#[derive(Resource, Clone, ExtractResource)]
struct SlimeSimRunConfig {
    pub run_copy_and_input: bool,
    pub run_diffuse: bool,
    pub run_agents: bool,
}

fn init_slime_sim_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        "SlimeSimBindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<SlimeAgentData>(false),            // binding 0: agents
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),  // binding 1: input
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly), // binding 2: output
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite), // binding 3: temp
                uniform_buffer::<GlobalUniforms>(false),            // binding 4: globals
                storage_buffer_read_only::<SpeciesSettings>(false),            // binding 5: species array
                uniform_buffer::<PheromoneUniforms>(false),         // binding 6: per-channel diffusion/decay
            ),
        ),
    );

    let agents_shader = asset_server.load(AGENTS_SHADER_PATH);

    let copy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: agents_shader.clone(), // same shader
        entry_point: Some(Cow::from("copy_to_temp")),
        ..default()
    });

    let agent_sim_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: agents_shader.clone(),
        entry_point: Some(Cow::from("update_agents")),
        ..default()
    });

    let diffuse_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: agents_shader.clone(),
        entry_point: Some(Cow::from("diffuse")),
        ..default()
    });

    let input_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: agents_shader.clone(),
        entry_point: Some(Cow::from("handle_input")),
        ..default()
    });

    // Step 1: Layout and pipeline for extract_pheromones (rgba -> r32float x3)
    let extract_bind_group_layout = render_device.create_bind_group_layout(
        "PheromoneExtractBindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),  // binding 0: rgba input
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),     // binding 1: p0 out
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),     // binding 2: p1 out
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),     // binding 3: p2 out
            ),
        ),
    );

    let phero_shader = asset_server.load(PHERO_SHADER_PATH);
    let extract_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![extract_bind_group_layout.clone()],
        shader: phero_shader.clone(),
        entry_point: Some(Cow::from("extract_pheromones")),
        ..default()
    });

    // Step 2: Layout and pipeline for compositing pheromones -> RGBA
    let composite_bind_group_layout = render_device.create_bind_group_layout(
        "PheromoneCompositeBindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),     // binding 0: p0 in
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),     // binding 1: p1 in
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),     // binding 2: p2 in
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly), // binding 3: rgba out
            ),
        ),
    );

    let composite_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![composite_bind_group_layout.clone()],
        shader: phero_shader.clone(),
        entry_point: Some(Cow::from("composite_pheromones")),
        ..default()
    });

    // Step 3: per-pheromone copy/input/diffuse on R32Float buffers
    let phero_env_bind_group_layout = render_device.create_bind_group_layout(
        "PheromoneEnvBindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // binding 0: input pheromone (r32)
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                // binding 1: temp pheromone (r32) read_write
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                // binding 2: output pheromone (r32) write
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                // binding 3: rgba temp (keep same type as main temp texture)
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                // binding 4: globals
                uniform_buffer::<GlobalUniforms>(false),
                // binding 5: species array (placeholder)
                storage_buffer_read_only::<SpeciesSettings>(false),
                // binding 6: pheromone params
                uniform_buffer::<PheromoneUniforms>(false),
            ),
        ),
    );

    let shader = asset_server.load(PHERO_SHADER_PATH);
    let copy_phero_pipelines: [CachedComputePipelineId; 3] = [
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("copy_to_temp_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("copy_to_temp_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("copy_to_temp_phero")),
            ..default()
        }),
    ];
    let input_phero_pipelines: [CachedComputePipelineId; 3] = [
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("handle_input_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("handle_input_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("handle_input_phero")),
            ..default()
        }),
    ];
    let diffuse_phero_pipelines: [CachedComputePipelineId; 3] = [
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("diffuse_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("diffuse_phero")),
            ..default()
        }),
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![phero_env_bind_group_layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("diffuse_phero")),
            ..default()
        }),
    ];

    commands.insert_resource(SlimeSimPipeline {
        texture_bind_group_layout,
        agent_sim_pipeline,
        diffuse_pipeline,
        copy_pipeline,
        input_pipeline,
        extract_bind_group_layout,
        extract_pipeline,
        composite_bind_group_layout,
        composite_pipeline,
        phero_env_bind_group_layout,
        copy_phero_pipelines,
        _input_phero_pipelines: input_phero_pipelines,
        _diffuse_phero_pipelines: diffuse_phero_pipelines,
    });
}

enum SlimeSimState {
    Loading,
    Init,
    Update(usize),
}

struct SlimeSimNode {
    state: SlimeSimState,
}

impl Default for SlimeSimNode {
    fn default() -> Self {
        Self {
            state: SlimeSimState::Loading,
        }
    }
}

impl render_graph::Node for SlimeSimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<SlimeSimPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            SlimeSimState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.agent_sim_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = SlimeSimState::Init;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing shader pipeline: {err}")
                    }
                    _ => {}
                }
            }
            SlimeSimState::Init => {
                let diffuse_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.diffuse_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let copy_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.copy_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let input_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.input_pipeline),
                    CachedPipelineState::Ok(_)
                );
                if diffuse_ok && copy_ok && input_ok {
                    // Start on index 0 so the first write goes to texture_b,
                    // matching the first sprite flip from a -> b.
                    self.state = SlimeSimState::Update(0);
                }
            }
            SlimeSimState::Update(0) => {
                self.state = SlimeSimState::Update(1);
            }
            SlimeSimState::Update(1) => {
                self.state = SlimeSimState::Update(0);
            }
            SlimeSimState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<SlimeSimImageBindGroups>().0;
        let extract_groups = world.get_resource::<SlimeSimExtractBindGroups>();
        let composite_groups = world.get_resource::<SlimeSimCompositeBindGroups>();
        let phero_env_groups = world.get_resource::<SlimeSimPheroEnvBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SlimeSimPipeline>();
        // select the pipeline based on the current state
        match self.state {
            SlimeSimState::Loading => {}
            SlimeSimState::Init => {}
            SlimeSimState::Update(index) => {
                let Some(copy_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.copy_pipeline) else { return Ok(()); };
                let Some(agent_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.agent_sim_pipeline) else { return Ok(()); };
                let Some(diffuse_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.diffuse_pipeline) else { return Ok(()); };
                let Some(input_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.input_pipeline) else { return Ok(()); };

                let groups_x = (SIZE.x + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                let groups_y = (SIZE.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                // Read run configuration (allows running passes independently)
                let run_config = world.resource::<SlimeSimRunConfig>();

                // PASS 1: COPY → INPUT → (optional) DIFFUSE
                if run_config.run_copy_and_input || run_config.run_diffuse {
                    {
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());

                        // Use the ping-pong bind group for this frame
                        pass.set_bind_group(0, &bind_groups[index], &[]);

                        // COPY A -> TEMP
                        if run_config.run_copy_and_input {
                            pass.set_pipeline(copy_pipeline);
                            pass.dispatch_workgroups(groups_x, groups_y, 1);

                            // HANDLE INPUT
                            pass.set_pipeline(input_pipeline);
                            pass.dispatch_workgroups(groups_x, groups_y, 1);
                        }

                        // DIFFUSE (optional)
                        if run_config.run_diffuse {
                            pass.set_pipeline(diffuse_pipeline);
                            pass.dispatch_workgroups(groups_x, groups_y, 1);
                        }

                        // `pass` drops here, ending the compute pass and guaranteeing the GPU
                        // will see writes from the above dispatches before the next pass.
                    }
                }

                // PASS 2: AGENTS (optional)
                if run_config.run_agents {
                    let mut pass2 = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    pass2.set_bind_group(0, &bind_groups[index], &[]);
                    pass2.set_pipeline(agent_pipeline);
                    let agent_groups = (NUM_AGENTS + AGENT_WORKGROUP_SIZE - 1) / AGENT_WORKGROUP_SIZE;
                    pass2.dispatch_workgroups(agent_groups, 1, 1);
                    // `pass2` drops here
                }

                // STEP 1: Extract RGBA (current output) into 3 r32 pheromone textures
                if let Some(extract_groups) = extract_groups {
                    let Some(extract_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.extract_pipeline) else { return Ok(()); };
                    let mut pass3 = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass3.set_bind_group(0, &extract_groups.0[index], &[]);
                    pass3.set_pipeline(extract_pipeline);
                    pass3.dispatch_workgroups(groups_x, groups_y, 1);
                    // pass3 drops here
                }

                // STEP 2: Composite pheromone r32 textures back into RGBA display texture
                if let Some(composite_groups) = composite_groups {
                    let Some(composite_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.composite_pipeline) else { return Ok(()); };
                    let mut pass4 = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass4.set_bind_group(0, &composite_groups.0[index], &[]);
                    pass4.set_pipeline(composite_pipeline);
                    pass4.dispatch_workgroups(groups_x, groups_y, 1);
                    // pass4 drops here
                }

                // STEP 3: Per-pheromone copy/input/diffuse on R32 buffers
                // We have two bind-group variants per pheromone to avoid binding the
                // same r32 texture for read and write in the same pass. The
                // convention is:
                //  - variant 0 (bg_pair[0]) : COPY + INPUT -> input=mainA, temp=mainB, output=mainB
                //  - variant 1 (bg_pair[1]) : DIFFUSE      -> input=mainB, temp=mainB, output=mainA
                if let Some(phero_env_groups) = phero_env_groups {
                    for p in 0..NUM_PHEROMONES {
                        // Start by only running the COPY step to isolate validation issues.
                        let Some(copy_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.copy_phero_pipelines[p]) else { return Ok(()); };
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        pass.set_bind_group(0, &phero_env_groups.0[p][index], &[]);
                        pass.set_pipeline(copy_pipeline);
                        pass.dispatch_workgroups(groups_x, groups_y, 1);
                        // `pass` drops here
                    }
                }
            }
        }

        Ok(())
    }
}
