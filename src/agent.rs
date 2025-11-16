use bevy::{
    math::usize, prelude::*, render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::{ExtractResource, ExtractResourcePlugin}, render_asset::RenderAssets, render_graph::{self, RenderGraph, RenderLabel}, render_resource::{
            binding_types::{storage_buffer, texture_storage_2d, uniform_buffer},
            *,
        }, renderer::{RenderContext, RenderDevice, RenderQueue}, texture::GpuImage
    }
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

// Bring in texture handles from main
use crate::DiffusionImages;
use crate::{SIZE, WORKGROUP_SIZE, NUM_AGENTS};

const SHADER_ASSET_PATH: &str = "shaders/agents.wgsl";


// -------------------------------------------------------------
// CPU-SIDE AGENT DATA (Main World)
// -------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct SlimeAgent {
    pub position: Vec2,
    pub direction: f32,
    pub _pad: f32,
}


// -------------------------------------------------------------
// GPU-SIDE BUFFER (Render World)
// -------------------------------------------------------------

#[derive(Resource, Clone, ExtractResource)]
pub struct AgentGpuBuffer {
    pub buffer: Buffer,
}


// -------------------------------------------------------------
// AGENT UNIFORMS
// -------------------------------------------------------------

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct AgentUniforms {
    pub move_speed: f32,
    pub turn_speed: f32,

    pub sensor_angle_degrees: f32,
    pub sensor_offset_dst: f32,

    pub sensor_size: i32,
    pub _pad0: i32,

    pub screen_size: Vec2,

    pub color: LinearRgba,

    pub delta_time: f32,   // ← NEW
    pub frame: u32,        // ← also super useful
    pub _pad1: Vec2,       // alignment padding (optional)
}


// -------------------------------------------------------------
// PLUGIN
// -------------------------------------------------------------

pub struct AgentComputePlugin;

impl Plugin for AgentComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract CPU → GPU
        app.add_plugins((
            ExtractResourcePlugin::<AgentGpuBuffer>::default(),
            ExtractResourcePlugin::<AgentUniforms>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(RenderStartup, init_agent_pipeline);

        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(AgentLabel, AgentNode::default());

        // Make agents run AFTER diffusion
        graph.add_node_edge(crate::diffusion::DiffusionLabel, AgentLabel);
    }
}


// -------------------------------------------------------------
// PIPELINE SETUP
// -------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct AgentLabel;

#[derive(Resource)]
struct AgentPipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

fn init_agent_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let bind_group_layout = render_device.create_bind_group_layout(
        "agent_bind_group_layout",
        &BindGroupLayoutEntries::sequential(

            ShaderStages::COMPUTE,
            (
                storage_buffer::<SlimeAgent>(false), // read-only agents
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<AgentUniforms>(false),
            ),
        ),
    );

    let shader = asset_server.load(SHADER_ASSET_PATH);

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..Default::default()
    });

    commands.insert_resource(AgentPipeline {
        bind_group_layout,
        pipeline,
    });
}


// -------------------------------------------------------------
// BIND GROUP
// -------------------------------------------------------------

#[derive(Resource)]
struct AgentBindGroup([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<AgentPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    diffusion: Res<DiffusionImages>,
    gpu_agents: Res<AgentGpuBuffer>,
    uniforms: Res<AgentUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_a = &gpu_images.get(&diffusion.texture_a).unwrap().texture_view;
    let view_b = &gpu_images.get(&diffusion.texture_b).unwrap().texture_view;

    // Upload uniforms
    let mut uniform_buffer = UniformBuffer::from(uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            BufferBinding {
                buffer: &gpu_agents.buffer,
                offset: 0,
                size: None,
            },
            view_a,
            view_b,
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            BufferBinding {
                buffer: &gpu_agents.buffer,
                offset: 0,
                size: None,
            },
            view_b,
            view_a,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(AgentBindGroup([bind_group_0, bind_group_1]));
}


// -------------------------------------------------------------
// COMPUTE NODE
// -------------------------------------------------------------

enum AgentState {
    Loading,
    Update(usize),
}

struct AgentNode {
    state: AgentState,
}

impl Default for AgentNode {
    fn default() -> Self {
        Self { state: AgentState::Loading }
    }
}

impl render_graph::Node for AgentNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<AgentPipeline>();
        let cache = world.resource::<PipelineCache>();
        match self.state {
            AgentState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    cache.get_compute_pipeline_state(pipeline.pipeline)
                    {
                        self.state = AgentState::Update(1);
                    }
            },
            AgentState::Update(0) => {
                self.state = AgentState::Update(1);
            }
            AgentState::Update(1) => {
                self.state = AgentState::Update(0);
            }
            AgentState::Update(_) => unreachable!(),
        }

    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        ctx: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<AgentBindGroup>().0;
        let pipeline   = world.resource::<AgentPipeline>();
        let cache      = world.resource::<PipelineCache>();

        // Small helper like you used in diffusion
        fn ceil_div(x: u32, y: u32) -> u32 {
            (x + y - 1) / y
        }

        match self.state {
            AgentState::Loading => {
                // do nothing until the pipeline is ready / resources are set up
            }
            AgentState::Update(index) => {
                if let Some(pipeline_obj) = cache.get_compute_pipeline(pipeline.pipeline) {
                    let mut pass = ctx
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    pass.set_pipeline(pipeline_obj);

                    // Pick the correct ping-pong bind group
                    pass.set_bind_group(0, &bind_groups[index], &[]);

                    // 1D dispatch over agents; use ceil_div in case agent_count
                    // isn’t a multiple of WORKGROUP_SIZE.
                    //
                    // If you have agent_count in uniforms or a resource,
                    // plug that in here instead of SIZE.x.
                    pass.dispatch_workgroups(
                        ceil_div(NUM_AGENTS, WORKGROUP_SIZE),
                        1,
                        1,
                    );
                }
            }
        }
        Ok(())
    }
}
