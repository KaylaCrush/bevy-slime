use bevy::prelude::*;
use bevy::render::{RenderApp, Render, RenderStartup, RenderSystems, render_graph::{self, RenderGraph}};
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::renderer::{RenderDevice, RenderQueue, RenderContext};
use bevy::render::render_resource::*;
//
use bevy::render::render_asset::RenderAssets;
use bevy::render::texture::GpuImage;
use bevy::shader::PipelineCacheError;
use std::borrow::Cow;

// legacy per-pheromone pipelines removed
use crate::pheromones::{init_pheromone_array_pipelines, create_phero_array_bind_groups};
use crate::resources::*;

pub struct SlimeSimComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, bevy::render::render_graph::RenderLabel)]
struct SlimeSimLabel;

impl Plugin for SlimeSimComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<crate::agents::AgentGpuBuffer>::default(),
            ExtractResourcePlugin::<SlimeSimImages>::default(),
            ExtractResourcePlugin::<SpeciesGpuBuffer>::default(),
            ExtractResourcePlugin::<GlobalUniforms>::default(),
            ExtractResourcePlugin::<SlimeSimRunConfig>::default(),
            ExtractResourcePlugin::<PheromoneUniforms>::default(),
            // Legacy per-channel images removed
            ExtractResourcePlugin::<crate::pheromones::PheromoneArrayImages>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_slime_sim_pipeline)
            .add_systems(Render, prepare_bind_group.in_set(RenderSystems::PrepareBindGroups));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SlimeSimLabel, SlimeSimNode::default());
        render_graph.add_node_edge(SlimeSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Resource)]
pub struct SlimeSimPipeline {
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

fn init_slime_sim_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Build bind group layout for group(0)
    let mut entries: Vec<BindGroupLayoutEntry> = Vec::new();
    let mut binding_index = 0u32;
    // 0: agents storage buffer
    entries.push(BindGroupLayoutEntry {
        binding: binding_index,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    });
    binding_index += 1;
    // 1: input rgba
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadOnly, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D2 }, count: None });
    binding_index += 1;
    // 2: output rgba
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D2 }, count: None });
    binding_index += 1;
    // 3: temp rgba
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadWrite, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D2 }, count: None });
    binding_index += 1;
    // 4: globals uniform
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None });
    binding_index += 1;
    // 5: species storage (read-only)
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None });
    binding_index += 1;
    // 6: pheromone params uniform
    entries.push(BindGroupLayoutEntry { binding: binding_index, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None });
    binding_index += 1;
    // 7: pheromone texture2D array (read_write) for agents (sensing + deposit)
    entries.push(BindGroupLayoutEntry {
        binding: binding_index,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadWrite, format: TextureFormat::R32Float, view_dimension: TextureViewDimension::D2Array },
        count: None,
    });
    binding_index += 1;
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        Some("SlimeSimBindGroupLayout"),
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

    commands.insert_resource(SlimeSimPipeline {
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
    pipeline: Res<SlimeSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    slime_agent_data: Res<crate::agents::AgentGpuBuffer>,
    slime_sim_images: Res<SlimeSimImages>,
    phero_arrays: Res<crate::pheromones::PheromoneArrayImages>,
    species_settings: Res<SpeciesGpuBuffer>,
    globals: Res<GlobalUniforms>,
    pheros: Res<PheromoneUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Copy the body from previous main.prepare_bind_group, but using resources types
    // (For brevity, this function calls into create_phero_bind_groups for pheromones.)
    let Some(image_a) = gpu_images.get(&slime_sim_images.texture_a) else { return; };
    let Some(image_b) = gpu_images.get(&slime_sim_images.texture_b) else { return; };
    let Some(image_temp) = gpu_images.get(&slime_sim_images.temp_texture) else { return; };

    let view_a = &image_a.texture_view;
    let view_b = &image_b.texture_view;
    let view_temp = &image_temp.texture_view;

    let mut global_uniform_buffer = UniformBuffer::from(globals.into_inner());
    global_uniform_buffer.write_buffer(&render_device, &queue);

    let mut pheromone_uniform_buffer = UniformBuffer::from(pheros.into_inner());
    pheromone_uniform_buffer.write_buffer(&render_device, &queue);

    // Get pheromone array views for agents: choose "next" per ping
    let Some(phero_prev_view) = gpu_images.get(&phero_arrays.prev).map(|g| &g.texture_view) else { return; };
    let Some(phero_next_view) = gpu_images.get(&phero_arrays.next).map(|g| &g.texture_view) else { return; };

    // Build bind group entries for group(0)
    let mut entries0 = vec![
        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(BufferBinding { buffer: &slime_agent_data.buffer, offset: 0, size: None }) },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(view_a) },
        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(view_b) },
        BindGroupEntry { binding: 3, resource: BindingResource::TextureView(view_temp) },
        BindGroupEntry { binding: 4, resource: global_uniform_buffer.binding().unwrap() },
        BindGroupEntry { binding: 5, resource: BindingResource::Buffer(BufferBinding { buffer: &species_settings.buffer, offset: 0, size: None }) },
        BindGroupEntry { binding: 6, resource: pheromone_uniform_buffer.binding().unwrap() },
        // For ping index 0, env writes to next, so agents should read/write next
        BindGroupEntry { binding: 7, resource: BindingResource::TextureView(phero_next_view) },
    ];

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &entries0,
    );

    let mut entries1 = vec![
        BindGroupEntry { binding: 0, resource: BindingResource::Buffer(BufferBinding { buffer: &slime_agent_data.buffer, offset: 0, size: None }) },
        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(view_b) },
        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(view_a) },
        BindGroupEntry { binding: 3, resource: BindingResource::TextureView(view_temp) },
        BindGroupEntry { binding: 4, resource: global_uniform_buffer.binding().unwrap() },
        BindGroupEntry { binding: 5, resource: BindingResource::Buffer(BufferBinding { buffer: &species_settings.buffer, offset: 0, size: None }) },
        BindGroupEntry { binding: 6, resource: pheromone_uniform_buffer.binding().unwrap() },
        // For ping index 1, env writes to prev, so agents should read/write prev
        BindGroupEntry { binding: 7, resource: BindingResource::TextureView(phero_prev_view) },
    ];

    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &entries1,
    );

    commands.insert_resource(SlimeSimImageBindGroups([bind_group_0, bind_group_1]));

    // No agents pheromone group resource needed when using fixed bindings

    // Create array-based pheromone bind groups targeting the current ping outputs
    if let Some((env_ping, comp_ping)) = create_phero_array_bind_groups(
        &render_device,
        &gpu_images,
        &*phero_arrays,
        &pipeline.phero_array_env_layout,
        &pipeline.phero_array_comp_layout,
        view_a,
        view_b,
        &global_uniform_buffer,
        &species_settings.buffer,
        &pheromone_uniform_buffer,
    ) {
        commands.insert_resource(crate::resources::PheroArrayEnvBindGroups(env_ping));
        commands.insert_resource(crate::resources::PheroArrayCompositeBindGroups(comp_ping));
    }
}

enum SlimeSimState { Loading, Init, Update(usize) }

struct SlimeSimNode { state: SlimeSimState }

impl Default for SlimeSimNode { fn default() -> Self { Self { state: SlimeSimState::Loading } } }

impl render_graph::Node for SlimeSimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<SlimeSimPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        match self.state {
            SlimeSimState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.agent_sim_pipeline) {
                    CachedPipelineState::Ok(_) => { self.state = SlimeSimState::Init; }
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => panic!("Initializing shader pipeline: {err}"),
                    _ => {}
                }
            }
            SlimeSimState::Init => {
                let diffuse_ok = true; // legacy RGBA env removed
                let copy_ok = true;
                let input_ok = true;
                // Array-based pipelines readiness
                let array_diff_ok = matches!(pipeline_cache.get_compute_pipeline_state(pipeline.diffuse_array_pipeline), CachedPipelineState::Ok(_));
                let array_input_ok = matches!(pipeline_cache.get_compute_pipeline_state(pipeline.input_array_pipeline), CachedPipelineState::Ok(_));
                let array_comp_ok = matches!(pipeline_cache.get_compute_pipeline_state(pipeline.composite_array_pipeline), CachedPipelineState::Ok(_));
                if diffuse_ok && copy_ok && input_ok && array_diff_ok && array_input_ok && array_comp_ok {
                    self.state = SlimeSimState::Update(0);
                }
            }
            SlimeSimState::Update(0) => { self.state = SlimeSimState::Update(1); }
            SlimeSimState::Update(1) => { self.state = SlimeSimState::Update(0); }
            SlimeSimState::Update(_) => unreachable!(),
        }
    }

    fn run(&self, _graph: &mut render_graph::RenderGraphContext, render_context: &mut RenderContext, world: &World) -> Result<(), render_graph::NodeRunError> {
        // This mirrors the previous run() implementation from main.rs.
        let bind_groups = &world.resource::<SlimeSimImageBindGroups>().0;
        // Legacy extract/composite groups removed
        // let phero_env_groups = world.get_resource::<SlimeSimPheroEnvBindGroups>();
        let phero_array_env = world.get_resource::<crate::resources::PheroArrayEnvBindGroups>();
        let phero_array_comp = world.get_resource::<crate::resources::PheroArrayCompositeBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SlimeSimPipeline>();
        // Using fixed pheromone bindings in group(0); no separate group(1)

        match self.state {
            SlimeSimState::Loading | SlimeSimState::Init => {}
            SlimeSimState::Update(index) => {
                let Some(agent_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.agent_sim_pipeline) else { return Ok(()); };

                let groups_x = (SIZE.x + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                let groups_y = (SIZE.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                let run_config = world.resource::<SlimeSimRunConfig>(); // toggles for agents/array passes

                // Array-based pheromone env passes (diffuse then input) with z-dispatch
                if let Some(arr_env) = phero_array_env {
                    let Some(diffuse_array) = pipeline_cache.get_compute_pipeline(pipeline.diffuse_array_pipeline) else { return Ok(()); };
                    let Some(input_array) = pipeline_cache.get_compute_pipeline(pipeline.input_array_pipeline) else { return Ok(()); };
                    let mut pass_arr = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
                    pass_arr.set_bind_group(0, &arr_env.0[index], &[]);
                    pass_arr.set_pipeline(diffuse_array);
                    pass_arr.dispatch_workgroups(groups_x, groups_y, NUM_PHEROMONES as u32);
                    pass_arr.set_pipeline(input_array);
                    pass_arr.dispatch_workgroups(groups_x, groups_y, NUM_PHEROMONES as u32);
                }

                if run_config.run_agents {
                    let mut pass2 = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
                    pass2.set_bind_group(0, &bind_groups[index], &[]);
                    // No group(1) needed
                    pass2.set_pipeline(agent_pipeline);
                    let agent_groups = (crate::agents::NUM_AGENTS + crate::agents::AGENT_WORKGROUP_SIZE - 1) / crate::agents::AGENT_WORKGROUP_SIZE;
                    pass2.dispatch_workgroups(agent_groups, 1, 1);
                }

                // Legacy extract/composite removed

                // Array-based composite (array -> RGBA). Note: writes to the bind-target chosen when creating array composite groups.
                if let Some(arr_comp) = phero_array_comp {
                    let Some(comp_array) = pipeline_cache.get_compute_pipeline(pipeline.composite_array_pipeline) else { return Ok(()); };
                    let mut pass_comp = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
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
