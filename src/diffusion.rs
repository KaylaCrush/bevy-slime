use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::PipelineCacheError,
};
use std::borrow::Cow;

use crate::DiffusionImages;

use crate::SIZE;
use crate::WORKGROUP_SIZE;

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/diffusion.wgsl";

pub struct DiffusionComputePlugin;

impl Plugin for DiffusionComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<DiffusionUniforms>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_diffusion_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(DiffusionLabel, DiffusionNode::default());
        render_graph.add_node_edge(DiffusionLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DiffusionLabel;

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct DiffusionUniforms {
    pub decay: f32,
    pub diffusion_strength: f32,
    pub delta_time: f32,
    pub padding1: f32,
}


#[derive(Resource)]
struct DiffusionImageBindGroups([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<DiffusionPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    diffusion_images: Res<DiffusionImages>,
    diffusion_uniforms: Res<DiffusionUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_a = gpu_images.get(&diffusion_images.texture_a).unwrap();
    let view_b = gpu_images.get(&diffusion_images.texture_b).unwrap();

    // Uniform buffer is used here to demonstrate how to set up a uniform in a compute shader
    // Alternatives such as storage buffers or push constants may be more suitable for your use case
    let mut uniform_buffer = UniformBuffer::from(diffusion_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(DiffusionImageBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct DiffusionPipeline {
    texture_bind_group_layout: BindGroupLayout,
    update_pipeline: CachedComputePipelineId,
}

fn init_diffusion_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        "DiffusionImages",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DiffusionUniforms>(false),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);

    let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..default()
    });

    commands.insert_resource(DiffusionPipeline {
        texture_bind_group_layout,
        update_pipeline,
    });
}


enum DiffusionState {
    Loading,
    Update(usize),
}

struct DiffusionNode {
    state: DiffusionState,
}

impl Default for DiffusionNode {
    fn default() -> Self {
        Self {
            state: DiffusionState::Loading,
        }
    }
}

impl render_graph::Node for DiffusionNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<DiffusionPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            DiffusionState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = DiffusionState::Update(0);
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            DiffusionState::Update(0) => {
                self.state = DiffusionState::Update(1);
            }
            DiffusionState::Update(1) => {
                self.state = DiffusionState::Update(0);
            }
            DiffusionState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<DiffusionImageBindGroups>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<DiffusionPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            DiffusionState::Loading => {}
            DiffusionState::Update(index) => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(update_pipeline);
                fn ceil_div(x: u32, y: u32) -> u32 {
                    (x + y - 1) / y
                }

                pass.dispatch_workgroups(
                    ceil_div(SIZE.x, WORKGROUP_SIZE),
                    ceil_div(SIZE.y, WORKGROUP_SIZE),
                    1
                );
            }
        }

        Ok(())
    }
}
