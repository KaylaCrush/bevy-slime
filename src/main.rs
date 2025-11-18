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

mod input;
use input::{InputPlugin, MouseWorldPos, MouseButtonState};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/compute.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: UVec2 = UVec2::new(1920 / DISPLAY_FACTOR, 1080 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;
const AGENT_WORKGROUP_SIZE: u32 = 64;
const NUM_AGENTS: u32 = 1000000;


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct SlimeAgentData {
    position: Vec2,
    angle: f32,
    species_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
struct SpeciesSettings {
    pub move_speed: f32,
    pub turn_speed: f32,
    pub sensor_angle_degrees: f32,
    pub sensor_offset_dst: f32,
    pub sensor_size: f32,
    pub deposit_amount: f32,
    pub diffusion_strength: f32,
    pub decay_rate: f32,
    pub color: [f32; 4],              // 32–47  (vec4)
    pub pheromone_mask: [f32; 4],              // 48–63  final padding to 64
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
        .add_systems(Startup, setup)
        .add_systems(Update, (switch_textures, sync_agents_to_gpu, update_globals_uniform))
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
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

    // ---- CPU AGENTS ----
    let mut agents = Vec::with_capacity(NUM_AGENTS as usize);

    let center = Vec2::new(SIZE.x as f32 * 0.5, SIZE.y as f32 * 0.5);
    let radius = (SIZE.x.min(SIZE.y) as f32) * 0.4;

    for i in 0..NUM_AGENTS {
        let t = i as f32 / NUM_AGENTS as f32;
        let angle = t * std::f32::consts::TAU;
        let r = radius * 0.5; // you can randomize later

        let pos = center + Vec2::new(angle.cos() * r, angle.sin() * r);

        // face toward center
        let dir_vec = (center - pos).normalize_or_zero();
        let dir = dir_vec.y.atan2(dir_vec.x);

        agents.push(SlimeAgentData {
            position: pos,
            angle: dir,
            species_index: 0,
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

    // ---- SPECIES BUFFER (for now: 1 species) ----
    let species_list = vec![SpeciesSettings {
        move_speed: 40.0,
        turn_speed: 6.0,
        sensor_angle_degrees: 30.0,
        sensor_offset_dst: 15.0,
        sensor_size: 3.0,
        deposit_amount: 1.0,
        diffusion_strength: 1.0,
        decay_rate: 0.8,
        color: [0.0, 0.0, 1.0, 1.0],
        pheromone_mask: [0.0,0.0,1.0,1.0],
    }];

    let species_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Slime species buffer"),
        contents: bytemuck::cast_slice(&species_list),
        usage: BufferUsages::STORAGE,
    });
    commands.insert_resource(SpeciesGpuBuffer { buffer: species_buffer });

    // ---- GLOBAL UNIFORMS ----
    commands.insert_resource(GlobalUniforms {
        delta_time: 0.01,             // small nonzero to avoid first-frame weirdness
        frame: 0,
        mouse_position: Vec2::new(-10000.0, -10000.0),
        screen_size: SIZE.as_vec2(),
        left_button_pressed: 0,
        right_button_pressed: 0,
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
    species_settings: Res<SpeciesGpuBuffer>,
    globals: Res<GlobalUniforms>,
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
        )),
    );

    commands.insert_resource(SlimeSimImageBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct SlimeSimPipeline {
    texture_bind_group_layout: BindGroupLayout,
    agent_sim_pipeline: CachedComputePipelineId,
    diffuse_pipeline: CachedComputePipelineId,
    copy_pipeline: CachedComputePipelineId,
    input_pipeline: CachedComputePipelineId,
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
            ),
        ),
    );

    let shader = asset_server.load(SHADER_ASSET_PATH);

    let copy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(), // same shader
        entry_point: Some(Cow::from("copy_to_temp")),
        ..default()
    });

    let agent_sim_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("update_agents")),
        ..default()
    });

    let diffuse_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("diffuse")),
        ..default()
    });

    let input_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: asset_server.load(SHADER_ASSET_PATH),
        entry_point: Some(Cow::from("handle_input")),
        ..default()
    });

    commands.insert_resource(SlimeSimPipeline {
        texture_bind_group_layout,
        agent_sim_pipeline,
        diffuse_pipeline,
        copy_pipeline,
        input_pipeline,
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
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            SlimeSimState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.diffuse_pipeline)
                {
                    self.state = SlimeSimState::Update(1);
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
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SlimeSimPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            SlimeSimState::Loading => {}
            SlimeSimState::Init => {}
            SlimeSimState::Update(index) => {
                let copy_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.copy_pipeline)
                    .unwrap();

                let agent_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.agent_sim_pipeline)
                    .unwrap();
                let diffuse_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.diffuse_pipeline)
                    .unwrap();

                let input_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.input_pipeline)
                    .unwrap();

                let groups_x = (SIZE.x + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                let groups_y = (SIZE.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                // COPY A → TEMP (or B → TEMP depending on index)
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(copy_pipeline);
                pass.dispatch_workgroups(groups_x, groups_y, 1);

                pass.set_pipeline(input_pipeline);
                pass.dispatch_workgroups(groups_x, groups_y, 1);

                // 1) Diffuse: still use bind_groups[index]
                pass.set_pipeline(diffuse_pipeline);
                pass.dispatch_workgroups(groups_x, groups_y, 1);

                // 2) Agents: use bind_groups[index]
                pass.set_pipeline(agent_pipeline);
                let agent_groups = (NUM_AGENTS + AGENT_WORKGROUP_SIZE - 1) / AGENT_WORKGROUP_SIZE;
                pass.dispatch_workgroups(agent_groups, 1, 1);


                }
           }

        Ok(())
    }
}
