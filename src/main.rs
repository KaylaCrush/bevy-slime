use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::{ExtractResource, ExtractResourcePlugin}, render_asset::RenderAssets, render_graph::{self, RenderGraph, RenderLabel}, render_resource::{
            binding_types::{storage_buffer, texture_storage_2d, uniform_buffer},
            *,
        }, renderer::{RenderContext, RenderDevice, RenderQueue}, texture::GpuImage
    },
    shader::PipelineCacheError,
};
use std::borrow::Cow;


const SHADER_ASSET_PATH: &str = "shaders/slime_sim.wgsl";

const NUM_AGENTS: usize = 100;
const DISPLAY_FACTOR: u32 = 4;
const SIZE: UVec2 = UVec2::new(1280/DISPLAY_FACTOR, 720/DISPLAY_FACTOR);
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
                // Default is linear sampling, for smooth transition. Nearest makes blocky pixel-art style
                .set(ImagePlugin::default_nearest()),
            SlimeSimComputePlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, render_device: Res<RenderDevice>){
    // this makes a transparent texture of a given size to be sent to the gpu
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    // this desegnates that the image is destined to live on the gpu, and will be removed from the asset system after being uploaded
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    // defines which kinds of ways this can be used in wgpu
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    // we copy the thing we just made -  we need two of em, so we can blit and swap
    let image0 = images.add(image.clone());
    // we are also uploading them to the Image asset that we loaded up top
    let image1 = images.add(image);

    let agent_data_size = (std::mem::size_of::<SlimeAgent>() * NUM_AGENTS) as u64;
    // create the GPU buffer descriptor
    let agent_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Slime Agents Buffer"),
        size: agent_data_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // wrap it in a resource
    commands.insert_resource(SlimeSimAgentBuffer(agent_buffer));

    // hmmm not sure what this is doing. maybe spawning the texture im the engine,
    // which we will actually write to after we ship the other two off to the gpu
    // for calculation purposes?
    // confirmed, this is the dealie we actually draw.
    commands.spawn((
        Sprite {
            image: image0.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    // ok, I think this is us passing the images for the gpu
    // into the thing that will put them in the gpu.
    commands.insert_resource(SlimeSimImages {
        texture_a: image0,
        texture_b: image1,
    });

    // uniforms, remember, are just things that don't change.
    // todo: add slime sim constants to this?
    commands.insert_resource(SlimeSimUniforms{
        pheromone_color: LinearRgba::BLUE,
        agent_count: NUM_AGENTS as u32,
        screen_size: SIZE.as_vec2(),
    });
}


// ExtractResource means it gets taken to the renderworld
// during the 'ExtractSchedule'(?) step
#[derive(Resource, Clone, ExtractResource)]
struct SlimeSimImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct SlimeAgent {
    pub position: Vec2,
    pub direction: f32,
    pub buffer: f32,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SlimeSimAgentBuffer(pub Buffer);

// ShaderType: "Base trait for all WGSL host-shareable types"
#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct SlimeSimUniforms {
    pheromone_color: LinearRgba,
    agent_count: u32,
    screen_size:Vec2,
}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<SlimeSimImages>, mut sprite: Single<&mut Sprite>) {
    // heck yeah I was right. This is us copying the SlimeSim Images and cycling them into the
    // sprite every frame.
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

struct SlimeSimComputePlugin;
// Renderlabel is "A strongly-typed class of labels used to identify a Node in a render graph"
// ie, it identifies ComputePlugin as being a shader somehow?
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SlimeSimLabel;

impl Plugin for SlimeSimComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<SlimeSimImages>::default(),
            ExtractResourcePlugin::<SlimeSimAgentBuffer>::default(),
            ExtractResourcePlugin::<SlimeSimUniforms>::default(),
        ));
        // SUB apps? What will they think of next? TODO: look into sub apps
        // it seems kinda obvious what this is - I've built plugins before after all
        // but I'm not sure why it's making its own sub-app and not just doing
        // app.whatever like my other plugins do.
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_slime_sim_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        // a render graph configures the modular render logic
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        // adding the slime sim compute node
        render_graph.add_node(SlimeSimLabel, SlimeSimNode::default());
        render_graph.add_node_edge(SlimeSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}


#[derive(Resource)]
struct SlimeSimImageBindGroups([BindGroup; 2]);

// As far as I can tell, this gets called every render step
fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<SlimeSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    slime_sim_images: Res<SlimeSimImages>,
    slime_sim_uniforms: Res<SlimeSimUniforms>,
    slime_sim_agents: Res<SlimeSimAgentBuffer>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_a = gpu_images.get(&slime_sim_images.texture_a).unwrap();
    let view_b = gpu_images.get(&slime_sim_images.texture_b).unwrap();

    // Uniform buffer is used here to demonstrate how to set up a uniform in a compute shader
    // Alternatives such as storage buffers or push constants may be more suitable for your use case
    let mut uniform_buffer = UniformBuffer::from(slime_sim_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            slime_sim_agents.0.as_entire_binding(),
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            slime_sim_agents.0.as_entire_binding(),
            &uniform_buffer,
        )),
    );
    commands.insert_resource(SlimeSimImageBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct SlimeSimPipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

fn init_slime_sim_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {

    let texture_bind_group_layout = render_device.create_bind_group_layout(
        "SlimeSimBindGroup",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                storage_buffer::<SlimeAgent>(false),
                uniform_buffer::<SlimeSimUniforms>(false),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init")),
        ..default()
    });
    let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..default()
    });

    commands.insert_resource(SlimeSimPipeline {
        texture_bind_group_layout,
        init_pipeline,
        update_pipeline,
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
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
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
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
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
            SlimeSimState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
            SlimeSimState::Update(index) => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
