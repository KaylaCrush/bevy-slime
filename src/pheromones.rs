use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::{
    render_asset::RenderAssets,
    render_resource::*,
    renderer::RenderDevice,
    texture::GpuImage,
};
use bevy::render::extract_resource::ExtractResource;
use std::borrow::Cow;

use crate::resources::{NUM_PHEROMONES, SIZE};

// Array-based pheromone images (Step 1: allocation only, no behavior change yet)
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct PheromoneArrayImages {
    pub prev: Handle<Image>,
    pub next: Handle<Image>,
}

// Removed legacy per-pheromone image allocation

/// Allocate array-based pheromone textures (prev/next), one layer per pheromone.
pub fn make_pheromone_array_images(images: &mut Assets<Image>) -> PheromoneArrayImages {
    let mut make_array = || {
        let mut img = Image::new_target_texture(
            SIZE.x,
            SIZE.y,
            TextureFormat::R32Float,
        );
        img.asset_usage = RenderAssetUsages::RENDER_WORLD;
        img.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
        // make it a 2D array with NUM_PHEROMONES layers
        let layers = NUM_PHEROMONES as u32;
        img.texture_descriptor.size.depth_or_array_layers = layers;
        // ensure data buffer matches expected size to avoid upload panic
        let bytes_per_pixel: u32 = 4; // R32Float
        let byte_len = SIZE.x * SIZE.y * layers * bytes_per_pixel;
        img.data = vec![0u8; byte_len as usize].into();
        img
    };

    let prev = images.add(make_array());
    let next = images.add(make_array());
    PheromoneArrayImages { prev, next }
}

// Removed legacy per-channel pipeline initialization

/// Initialize array-based pheromone pipelines and layouts (prev/next array processing).
/// Returns (env_layout, diffuse_array_pipeline, input_array_pipeline, composite_array_layout, composite_array_pipeline)
pub fn init_pheromone_array_pipelines(
    render_device: &RenderDevice,
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
) -> (
    BindGroupLayout,
    CachedComputePipelineId,
    CachedComputePipelineId,
    BindGroupLayout,
    CachedComputePipelineId,
) {
    // Env layout: prev_array (ro), next_array (rw), rgba_temp (rw), globals, species, pheros
    let env_bind_group_layout = render_device.create_bind_group_layout(
        Some("PheroArrayEnvBindGroupLayout"),
        &[
            // 0: prev array
            BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadOnly, format: TextureFormat::R32Float, view_dimension: TextureViewDimension::D2Array }, count: None },
            // 1: next array
            BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadWrite, format: TextureFormat::R32Float, view_dimension: TextureViewDimension::D2Array }, count: None },
            // 2: rgba temp
            BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadWrite, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D2 }, count: None },
            // 3: globals
            BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            // 4: species (ro)
            BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            // 5: pheromone params
            BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    );

    let shader = asset_server.load(crate::resources::PHERO_SHADER_PATH);
    let diffuse_array_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![env_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("diffuse_phero_array")),
        ..default()
    });
    let input_array_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![env_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("handle_input_phero_array")),
        ..default()
    });

    // Composite layout: array (ro) + rgba (wo)
    let composite_array_layout = render_device.create_bind_group_layout(
        Some("PheroArrayCompositeBindGroupLayout"),
        &[
            BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::ReadOnly, format: TextureFormat::R32Float, view_dimension: TextureViewDimension::D2Array }, count: None },
            BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D2 }, count: None },
        ],
    );

    let composite_array_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![composite_array_layout.clone()],
        shader,
        entry_point: Some(Cow::from("composite_pheromones_array")),
        ..default()
    });

    (
        env_bind_group_layout,
        diffuse_array_pipeline,
        input_array_pipeline,
        composite_array_layout,
        composite_array_pipeline,
    )
}

/// Create bind groups for array-based pheromone processing (two pings prev/next)
pub fn create_phero_array_bind_groups(
    render_device: &RenderDevice,
    gpu_images: &RenderAssets<GpuImage>,
    phero_arrays: &PheromoneArrayImages,
    env_layout: &BindGroupLayout,
    composite_layout: &BindGroupLayout,
    view_out_a: &TextureView,
    view_out_b: &TextureView,
    global_uniform_buffer: &bevy::render::render_resource::UniformBuffer<&crate::resources::GlobalUniforms>,
    species_buffer: &bevy::render::render_resource::Buffer,
    pheromone_uniform_buffer: &bevy::render::render_resource::UniformBuffer<&crate::resources::PheromoneUniforms>,
) -> Option<([BindGroup; 2], [BindGroup; 2])> {
    let prev_view = &gpu_images.get(&phero_arrays.prev)?.texture_view;
    let next_view = &gpu_images.get(&phero_arrays.next)?.texture_view;

    // Ping 0: prev=prev, next=next
    let env_bg0 = render_device.create_bind_group(
        None,
        env_layout,
        &BindGroupEntries::sequential((
            prev_view,
            next_view,
            view_out_b, // rgba temp unused in env; ok if matching layout; we'll pass the actual output here for consistency
            global_uniform_buffer,
            BufferBinding { buffer: species_buffer, offset: 0, size: None },
            pheromone_uniform_buffer,
        )),
    );
    let comp_bg0 = render_device.create_bind_group(
        None,
        composite_layout,
        &BindGroupEntries::sequential((next_view, view_out_b)),
    );

    // Ping 1: prev=next, next=prev
    let env_bg1 = render_device.create_bind_group(
        None,
        env_layout,
        &BindGroupEntries::sequential((
            next_view,
            prev_view,
            view_out_a,
            global_uniform_buffer,
            BufferBinding { buffer: species_buffer, offset: 0, size: None },
            pheromone_uniform_buffer,
        )),
    );
    let comp_bg1 = render_device.create_bind_group(
        None,
        composite_layout,
        &BindGroupEntries::sequential((prev_view, view_out_a)),
    );

    Some(([env_bg0, env_bg1], [comp_bg0, comp_bg1]))
}

// Removed legacy per-channel bind group creation
