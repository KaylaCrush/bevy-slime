// Render / compute integration for the simulation.
//
// This module is responsible for creating compute pipeline layouts, preparing
// bind groups, and scheduling compute dispatches inside the Bevy render graph.
// Key responsibilities and ordering expectations:
// - Create the compute pipeline for agent simulation (`update_agents`) which
//   reads/writes the pheromone array (storage texture array) and updates the
//   agent storage buffer.
// - Create array-based pheromone pipelines (diffuse/input/composite) that
//   operate on a ping-pong pair of 2D texture arrays (prev/next). The render
//   node alternates an `index` (0/1) to flip which handle is prev/next.
// - The Render node run order must ensure agents deposit into the correct
//   ping (the 'next' array) before the array-based diffuse/composite steps
//   operate on that data for the next frame's visualization.
//
// When reading this file, pay attention to bind-group layout 0: it binds the
// agent buffer, RGBA display targets, uniforms, and a `R32Float` 2D-array
// used by agents for sensing/depositing. The shader `agents.wgsl` expects
// those bindings in a fixed layout; changing bindings here requires updating
// the shader accordingly.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_graph::{self, RenderGraph},
};
//
use bevy::render::render_asset::RenderAssets;
use bevy::render::texture::GpuImage;
use bevy::shader::PipelineCacheError;
use std::borrow::Cow;

// legacy per-pheromone pipelines removed
use crate::pheromones::{create_phero_array_bind_groups, init_pheromone_array_pipelines};
use crate::resources::*;

pub struct AgentSimComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, bevy::render::render_graph::RenderLabel)]
struct AgentSimLabel;

impl Plugin for AgentSimComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<crate::agents::AgentGpuBuffer>::default(),
            ExtractResourcePlugin::<PheromoneImages>::default(),
            ExtractResourcePlugin::<SpeciesGpuBuffer>::default(),
            ExtractResourcePlugin::<SpeciesLayerWeights>::default(),
            ExtractResourcePlugin::<GlobalUniforms>::default(),
            ExtractResourcePlugin::<PheromoneConfig>::default(),
            ExtractResourcePlugin::<AgentSimRunConfig>::default(),
            ExtractResourcePlugin::<crate::pheromones::PheromoneArrayImages>::default(),
            ExtractResourcePlugin::<crate::resources::PheromoneLayerParamsBuffer>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_agent_sim_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(AgentSimLabel, AgentSimNode::default());
        render_graph.add_node_edge(AgentSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}

/// Initialize the compute pipelines and layouts used by the simulation.
///
/// This creates:
/// - a bind group layout used by the agent compute shader (group 0),
/// - a compute pipeline for agent simulation (`update_agents`), and
/// - array-based pheromone pipelines via `init_pheromone_array_pipelines`.
///
/// The returned `AgentSimPipeline` resource stores the cached pipelines and
/// layouts so the render node can query readiness and dispatch them.

#[derive(Resource)]
pub struct AgentSimPipeline {
    pub texture_bind_group_layout: BindGroupLayout,
    pub agent_sim_pipeline: CachedComputePipelineId,
    // Array-based pipelines
    pub phero_array_env_layout: BindGroupLayout,
    pub diffuse_array_pipeline: CachedComputePipelineId,
    pub input_array_pipeline: CachedComputePipelineId,
    pub phero_array_comp_layout: BindGroupLayout,
    pub composite_array_pipeline: CachedComputePipelineId,
}

// No separate agents pheromone bind group resource needed when using fixed bindings

fn init_agent_sim_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // NOTE: binding indices here are mirrored by the agent shader and by code
    // that constructs BindGroupEntries in `prepare_bind_group`. Keep the layout
    // stable when editing shaders.
    // Build bind group layout for group(0) with only the bindings used by agents.wgsl
    let entries: Vec<BindGroupLayoutEntry> = vec![
        // 0: agents storage buffer
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // 4: globals uniform
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // 5: species storage (read-only)
        BindGroupLayoutEntry {
            binding: 5,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // 6: pheromone texture2D array (read_write) for agents (sensing + deposit)
        BindGroupLayoutEntry {
            binding: 6,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::ReadWrite,
                format: TextureFormat::R32Float,
                view_dimension: TextureViewDimension::D2Array,
            },
            count: None,
        },
        // 7: extended species weights (dense f32 array), read-only
        BindGroupLayoutEntry {
            binding: 7,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // 9: PheroControl uniform (layer_count only)
        BindGroupLayoutEntry {
            binding: 9,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        Some("AgentSimBindGroupLayout"),
        &entries,
    );

    // No separate bind group layout needed for agents' pheromones when using fixed bindings

    let agents_shader = asset_server.load(AGENTS_SHADER_PATH);

    let agent_sim_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: agents_shader.clone(),
        entry_point: Some(Cow::from("update_agents")),
        ..default()
    });

    // Array-based pheromone pipelines
    let (
        phero_array_env_layout,
        diffuse_array_pipeline,
        input_array_pipeline,
        phero_array_comp_layout,
        composite_array_pipeline,
    ) = init_pheromone_array_pipelines(&render_device, &asset_server, &pipeline_cache);

    commands.insert_resource(AgentSimPipeline {
        texture_bind_group_layout,
        agent_sim_pipeline,
        phero_array_env_layout,
        diffuse_array_pipeline,
        input_array_pipeline,
        phero_array_comp_layout,
        composite_array_pipeline,
    });
}

fn prepare_bind_group(
    mut commands: Commands,
    world: &World,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Fetch commonly used resources from the world to keep the system
    // function signature small while preserving Bevy's system parameter
    // ergonomics. This avoids a large argument list while keeping the code
    // explicit about which resources are consumed.
    let pipeline = world.resource::<AgentSimPipeline>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let agent_gpu_buffer = world.resource::<crate::agents::AgentGpuBuffer>();
    let pheromone_images = world.resource::<PheromoneImages>();
    let phero_arrays = world.resource::<crate::pheromones::PheromoneArrayImages>();
    let species_settings = world.resource::<SpeciesGpuBuffer>();
    let species_weights_res = world.get_resource::<SpeciesLayerWeights>();
    let globals = world.resource::<GlobalUniforms>();
    let phero_cfg = world.resource::<PheromoneConfig>();
    let layer_params = world.resource::<crate::resources::PheromoneLayerParamsBuffer>();

    // Resolve GPU image handles
    let Some(image_a) = gpu_images.get(&pheromone_images.texture_a) else {
        return;
    };
    let Some(image_b) = gpu_images.get(&pheromone_images.texture_b) else {
        return;
    };

    let view_a = &image_a.texture_view;
    let view_b = &image_b.texture_view;
    // No temp texture used

    // Build uniform buffers from cloned resources and write them to the GPU
    let mut global_uniform_buffer = UniformBuffer::from(globals);
    global_uniform_buffer.write_buffer(&render_device, &queue);

        // let mut pheromone_uniform_buffer = UniformBuffer::from(pheros);
        // pheromone_uniform_buffer.write_buffer(&render_device, &queue);

    // Get pheromone array views for agents: choose "next" per ping
    let Some(phero_prev_view) = gpu_images.get(&phero_arrays.prev).map(|g| &g.texture_view) else {
        return;
    };
    let Some(phero_next_view) = gpu_images.get(&phero_arrays.next).map(|g| &g.texture_view) else {
        return;
    };

    // Extended pheromone dummy buffers and control uniform (use_extended=0 by default)
    let Some(species_layer_weights) = species_weights_res else { return; };

    let layer_count = phero_cfg.layer_count.max(1);
    let weights_buf_ref = &species_layer_weights.weights;

    let phero_ctrl_uniform = crate::resources::PheroControlUniform { layer_count, _pad: UVec3::ZERO };
    let mut phero_ctrl_buffer = UniformBuffer::from(&phero_ctrl_uniform);
    phero_ctrl_buffer.write_buffer(&render_device, &queue);

    // Build bind group entries for group(0)
    let entries0 = vec![
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &agent_gpu_buffer.buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 4,
            resource: global_uniform_buffer.binding().unwrap(),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &species_settings.buffer,
                offset: 0,
                size: None,
            }),
        },
        // For ping index 0, env writes to next, so agents should read/write next
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::TextureView(phero_next_view),
        },
        BindGroupEntry {
            binding: 7,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: weights_buf_ref,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry { binding: 9, resource: phero_ctrl_buffer.binding().unwrap() },
    ];

    let bind_group_0 =
        render_device.create_bind_group(None, &pipeline.texture_bind_group_layout, &entries0);

    let entries1 = vec![
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &agent_gpu_buffer.buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 4,
            resource: global_uniform_buffer.binding().unwrap(),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &species_settings.buffer,
                offset: 0,
                size: None,
            }),
        },
        // For ping index 1, env writes to prev, so agents should read/write prev
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::TextureView(phero_prev_view),
        },
        BindGroupEntry {
            binding: 7,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: weights_buf_ref,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry { binding: 9, resource: phero_ctrl_buffer.binding().unwrap() },
    ];

    let bind_group_1 =
        render_device.create_bind_group(None, &pipeline.texture_bind_group_layout, &entries1);

    commands.insert_resource(AgentSimImageBindGroups([bind_group_0, bind_group_1]));

    // Brush control uniform for input pass
    let brush_uniform = crate::resources::BrushControlUniform {
        target_layer: phero_cfg.brush_target_layer,
        _mode: 0,
        _pad: UVec2::ZERO,
    };
    let mut brush_uniform_buffer = UniformBuffer::from(&brush_uniform);
    brush_uniform_buffer.write_buffer(&render_device, &queue);

    // Create array-based pheromone bind groups targeting the current ping outputs
    if let Some((env_ping, comp_ping)) = create_phero_array_bind_groups(
        &render_device,
        gpu_images,
        phero_arrays,
        &pipeline.phero_array_env_layout,
        &pipeline.phero_array_comp_layout,
        view_a,
        view_b,
        &global_uniform_buffer,
        &layer_params.buffer,
        &brush_uniform_buffer,
    ) {
        commands.insert_resource(crate::resources::PheroArrayEnvBindGroups(env_ping));
        commands.insert_resource(crate::resources::PheroArrayCompositeBindGroups(comp_ping));
    }
}

enum AgentSimState {
    Loading,
    Init,
    Update(usize),
}

struct AgentSimNode {
    state: AgentSimState,
}

impl Default for AgentSimNode {
    fn default() -> Self {
        Self {
            state: AgentSimState::Loading,
        }
    }
}

impl render_graph::Node for AgentSimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<AgentSimPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        match self.state {
            AgentSimState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.agent_sim_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = AgentSimState::Init;
                    }
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => panic!("Initializing shader pipeline: {err}"),
                    _ => {}
                }
            }
            AgentSimState::Init => {
                let diffuse_ok = true; // legacy RGBA env removed
                let copy_ok = true;
                let input_ok = true;
                // Array-based pipelines readiness
                let array_diff_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.diffuse_array_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let array_input_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.input_array_pipeline),
                    CachedPipelineState::Ok(_)
                );
                let array_comp_ok = matches!(
                    pipeline_cache.get_compute_pipeline_state(pipeline.composite_array_pipeline),
                    CachedPipelineState::Ok(_)
                );
                if diffuse_ok
                    && copy_ok
                    && input_ok
                    && array_diff_ok
                    && array_input_ok
                    && array_comp_ok
                {
                    self.state = AgentSimState::Update(0);
                }
            }
            AgentSimState::Update(0) => {
                self.state = AgentSimState::Update(1);
            }
            AgentSimState::Update(1) => {
                self.state = AgentSimState::Update(0);
            }
            AgentSimState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        // This mirrors the previous run() implementation from main.rs.
        let bind_groups = &world.resource::<AgentSimImageBindGroups>().0;
        // Legacy extract/composite groups removed
        // let phero_env_groups = world.get_resource::<SlimeSimPheroEnvBindGroups>();
        let phero_array_env = world.get_resource::<crate::resources::PheroArrayEnvBindGroups>();
        let phero_array_comp =
            world.get_resource::<crate::resources::PheroArrayCompositeBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<AgentSimPipeline>();
        // Using fixed pheromone bindings in group(0); no separate group(1)

        // The render node alternates between update indices 0 and 1 each
        // frame; those indices are used to select which ping is "prev" and
        // which is "next" for the array-based pheromone passes.
        match self.state {
            AgentSimState::Loading | AgentSimState::Init => {}
            AgentSimState::Update(index) => {
                let Some(agent_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.agent_sim_pipeline)
                else {
                    return Ok(());
                };

                let groups_x = SIZE.x.div_ceil(WORKGROUP_SIZE);
                let groups_y = SIZE.y.div_ceil(WORKGROUP_SIZE);
                let layer_count = world
                    .get_resource::<PheromoneConfig>()
                    .map(|c| c.layer_count)
                    .unwrap_or(NUM_PHEROMONES as u32)
                    .max(1);

                let run_config = world.resource::<AgentSimRunConfig>(); // toggles for agents/array passes

                // Array-based pheromone env passes (diffuse then input) with z-dispatch
                if let Some(arr_env) = phero_array_env {
                    let Some(diffuse_array) =
                        pipeline_cache.get_compute_pipeline(pipeline.diffuse_array_pipeline)
                    else {
                        return Ok(());
                    };
                    let Some(input_array) =
                        pipeline_cache.get_compute_pipeline(pipeline.input_array_pipeline)
                    else {
                        return Ok(());
                    };
                    // Only begin a compute pass if at least one of the array passes is enabled
                    if run_config.run_diffuse || run_config.run_copy_and_input {
                        let mut pass_arr = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        pass_arr.set_bind_group(0, &arr_env.0[index], &[]);
                        if run_config.run_diffuse {
                            pass_arr.set_pipeline(diffuse_array);
                            pass_arr.dispatch_workgroups(groups_x, groups_y, layer_count);
                        }
                        if run_config.run_copy_and_input {
                            pass_arr.set_pipeline(input_array);
                            pass_arr.dispatch_workgroups(groups_x, groups_y, layer_count);
                        }
                    }
                }

                if run_config.run_agents {
                    let mut pass2 = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass2.set_bind_group(0, &bind_groups[index], &[]);
                    // No group(1) needed
                    pass2.set_pipeline(agent_pipeline);
                    let agent_groups =
                        crate::agents::NUM_AGENTS.div_ceil(crate::agents::AGENT_WORKGROUP_SIZE);
                    pass2.dispatch_workgroups(agent_groups, 1, 1);
                }

                // Legacy extract/composite removed

                // Array-based composite (array -> RGBA). Note: writes to the bind-target chosen when creating array composite groups.
                if let Some(arr_comp) = phero_array_comp {
                    let Some(comp_array) =
                        pipeline_cache.get_compute_pipeline(pipeline.composite_array_pipeline)
                    else {
                        return Ok(());
                    };
                    let mut pass_comp = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass_comp.set_bind_group(0, &arr_comp.0[index], &[]);
                    pass_comp.set_pipeline(comp_array);
                    pass_comp.dispatch_workgroups(groups_x, groups_y, 1);
                }

                // Legacy per-pheromone copy-only pass remains disabled (array path active now)
            }
        }

        Ok(())
    }
}
