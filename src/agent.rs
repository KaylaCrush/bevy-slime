use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use std::borrow::Cow;
use bytemuck::{Pod, Zeroable};

use crate::DiffusionImages;
use crate::SIZE;
use crate::WORKGROUP_SIZE;

const SHADER_ASSET_PATH: &str = "shaders/agents.wgsl";


// ----------------------------------------------------------
// Plugin
// ----------------------------------------------------------

pub struct AgentComputePlugin;

impl Plugin for AgentComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the resources we need in the render world
        app.add_plugins((
            ExtractResourcePlugin::<AgentBuffer>::default(),
            ExtractResourcePlugin::<AgentUniforms>::default(),
            ExtractResourcePlugin::<DiffusionImages>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_systems(RenderStartup, init_agent_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(AgentLabel, AgentNode::default());

        // Agents run AFTER diffusion pass
        graph
            .add_node_edge(crate::diffusion::DiffusionLabel, AgentLabel)
            .unwrap();
    }
}


// ----------------------------------------------------------
// Labels
// ----------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct AgentLabel;


// ----------------------------------------------------------
// GPU data types
// ----------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct SlimeAgent {
    pub position: Vec2,
    pub direction: f32,
    pub pad: f32, // alignment padding
}

#[derive(Resource, Clone, ExtractResource)]
pub struct AgentBuffer {
    pub buffer: StorageBuffer<SlimeAgent>,
    pub count: u32,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct AgentUniforms {
    pub move_speed: f32,
    pub turn_speed: f32,

    pub sensor_angle_degrees: f32,
    pub sensor_offset_dst: f32,

    pub sensor_size: i32,
    pub _pad0: i32,
    pub _pad1: i32,
    pub _pad2: i32,

    pub colour: LinearRgba,
}


// ----------------------------------------------------------
// Pipeline
// ----------------------------------------------------------

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
                storage_buffer::<SlimeAgent>(true), // read+write
                texture_storage_2d(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::ReadOnly,
                ),
                texture_storage_2d(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<AgentUniforms>(false),
            ),
        ),
    );

    let shader = asset_server.load(SHADER_ASSET_PATH);

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..default()
    });

    commands.insert_resource(AgentPipeline {
        bind_group_layout,
        pipeline,
    });
}


// ----------------------------------------------------------
// Bind group
// ----------------------------------------------------------

#[derive(Resource)]
struct AgentBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<AgentPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    diffusion: Res<DiffusionImages>,
    agent_buf: Res<AgentBuffer>,
    agent_uniforms: Res<AgentUniforms>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Upload uniforms
    let mut uniform_buffer = UniformBuffer::from(agent_uniforms.into_inner());
    uniform_buffer.write_buffer(&device, &queue);

    let view_a = &gpu_images.get(&diffusion.texture_a).unwrap().texture_view;
    let view_b = &gpu_images.get(&diffusion.texture_b).unwrap().texture_view;

    let bind_group = device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            &agent_buf.buffer, // <-- PERFECT NOW
            view_a,
            view_b,
            &uniform_buffer,
        )),
    );

    commands.insert_resource(AgentBindGroup(bind_group));
}


// ----------------------------------------------------------
// Node
// ----------------------------------------------------------

enum AgentState {
    Loading,
    Update,
}

struct AgentNode {
    state: AgentState,
}

impl Default for AgentNode {
    fn default() -> Self {
        Self {
            state: AgentState::Loading,
        }
    }
}

impl render_graph::Node for AgentNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<AgentPipeline>();
        let cache = world.resource::<PipelineCache>();

        if let AgentState::Loading = self.state {
            if let CachedPipelineState::Ok(_) =
                cache.get_compute_pipeline_state(pipeline.pipeline)
            {
                self.state = AgentState::Update;
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        ctx: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_group = &world.resource::<AgentBindGroup>().0;
        let pipeline = world.resource::<AgentPipeline>();
        let cache = world.resource::<PipelineCache>();

        if let Some(pipeline_obj) = cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(pipeline_obj);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(world.resource::<AgentBuffer>().count, 1, 1);
        }

        Ok(())
    }
}
