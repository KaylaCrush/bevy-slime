// Pheromone allocation and pipeline helpers.
//
// This module centralizes code that creates the GPU textures and compute
// pipelines used to simulate pheromone diffusion and decay. The project uses
// an array-based representation: a single 2D texture array where each layer
// corresponds to one pheromone channel. Using a 2D array allows us to dispatch
// a single compute pass with a Z dimension equal to the number of pheromones.
//
// Important notes:
// - Allocation: `make_pheromone_array_images` creates a pair of array textures
//   (prev/next) which are ping-ponged by the render node.
// - Pipelines: `init_pheromone_array_pipelines` registers compute pipelines for
//   diffusion, input (brush), and composite steps. The WGSL shader implements
//   `diffuse_phero_array`, `handle_input_phero_array`, and `composite_pheromones_array`.
// - Bind groups created by `create_phero_array_bind_groups` must match the
//   layout expected by the WGSL entry points. Keep binding indices in sync.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::{
    render_asset::RenderAssets, render_resource::*, renderer::RenderDevice, texture::GpuImage,
};
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
    let make_array = create_pheromone_array_image;
    let prev = images.add(make_array());
    let next = images.add(make_array());
    PheromoneArrayImages { prev, next }
}

/// Create a single pheromone array texture descriptor/image without allocating in Assets.
/// This is a pure helper so we can unit-test texture allocation independently.
pub fn create_pheromone_array_image() -> Image {
    let mut img = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Float);
    img.asset_usage = RenderAssetUsages::RENDER_WORLD;
    img.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    // make it a 2D array with NUM_PHEROMONES layers
    let layers = NUM_PHEROMONES as u32;
    img.texture_descriptor.size.depth_or_array_layers = layers;
    // ensure data buffer matches expected size to avoid upload panic
    let bytes_per_pixel: u32 = 4; // R32Float
    let byte_len = SIZE.x * SIZE.y * layers * bytes_per_pixel;
    img.data = vec![0u8; byte_len as usize].into();
    img
}

// Initialize GPU pipelines and layouts for array-based pheromone processing.
//
// The returned tuple contains the env bind group layout (prev/next array + uniforms),
// cached pipeline IDs for the diffuse and input passes, and the composite layout/pipeline
// used to convert the array back into an RGBA display texture.

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
    // Env layout: prev_array (ro), next_array (rw), rgba_temp (rw), globals, species, per-layer params
    let env_bind_group_layout = render_device.create_bind_group_layout(
        Some("PheroArrayEnvBindGroupLayout"),
        &[
            // 0: prev array
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2Array,
                },
                count: None,
            },
            // 1: next array
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2Array,
                },
                count: None,
            },
            // 2: rgba temp
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // 3: globals
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: species (ro)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: pheromone layer params buffer (array<PheromoneLayerParam>)
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

    // Composite layout: array (ro) + rgba (wo) + per-layer params
    let composite_array_layout = render_device.create_bind_group_layout(
        Some("PheroArrayCompositeBindGroupLayout"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2Array,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    let composite_array_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
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
#[allow(clippy::too_many_arguments)]
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
    layer_params_buffer: &bevy::render::render_resource::Buffer,
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
            BufferBinding {
                buffer: species_buffer,
                offset: 0,
                size: None,
            },
            BufferBinding {
                buffer: layer_params_buffer,
                offset: 0,
                size: None,
            },
        )),
    );
    let comp_bg0 = render_device.create_bind_group(
        None,
        composite_layout,
        &BindGroupEntries::sequential((
            next_view,
            view_out_b,
            BufferBinding {
                buffer: layer_params_buffer,
                offset: 0,
                size: None,
            },
        )),
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
            BufferBinding {
                buffer: species_buffer,
                offset: 0,
                size: None,
            },
            BufferBinding {
                buffer: layer_params_buffer,
                offset: 0,
                size: None,
            },
        )),
    );
    let comp_bg1 = render_device.create_bind_group(
        None,
        composite_layout,
        &BindGroupEntries::sequential((
            prev_view,
            view_out_a,
            BufferBinding {
                buffer: layer_params_buffer,
                offset: 0,
                size: None,
            },
        )),
    );

    Some(([env_bg0, env_bg1], [comp_bg0, comp_bg1]))
}

// Removed legacy per-channel bind group creation

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Assets;

    #[test]
    fn make_pheromone_array_images_layers_and_size() {
        let mut images: Assets<Image> = Assets::default();
        let phero_imgs = make_pheromone_array_images(&mut images);

        let prev = images.get(&phero_imgs.prev).expect("prev image exists");
        let next = images.get(&phero_imgs.next).expect("next image exists");

        // Verify depth/array layers equal NUM_PHEROMONES
        assert_eq!(
            prev.texture_descriptor.size.depth_or_array_layers,
            NUM_PHEROMONES as u32
        );
        assert_eq!(
            next.texture_descriptor.size.depth_or_array_layers,
            NUM_PHEROMONES as u32
        );

        // basic sanity: texture size and layer count match expectations
        assert_eq!(prev.texture_descriptor.size.width, SIZE.x);
        assert_eq!(prev.texture_descriptor.size.height, SIZE.y);
        assert_eq!(
            prev.texture_descriptor.size.depth_or_array_layers,
            NUM_PHEROMONES as u32
        );
        assert_eq!(next.texture_descriptor.size.width, SIZE.x);
        assert_eq!(next.texture_descriptor.size.height, SIZE.y);
        assert_eq!(
            next.texture_descriptor.size.depth_or_array_layers,
            NUM_PHEROMONES as u32
        );
    }

    #[test]
    fn create_pheromone_array_image_descriptor() {
        let img = create_pheromone_array_image();
        // check dimensions and layer count
        assert_eq!(img.texture_descriptor.size.width, SIZE.x);
        assert_eq!(img.texture_descriptor.size.height, SIZE.y);
        assert_eq!(
            img.texture_descriptor.size.depth_or_array_layers,
            NUM_PHEROMONES as u32
        );
        // format should be R32Float
        assert_eq!(img.texture_descriptor.format, TextureFormat::R32Float);
        // check usage flags
        assert!(
            img.texture_descriptor
                .usage
                .contains(TextureUsages::STORAGE_BINDING)
        );
        assert!(
            img.texture_descriptor
                .usage
                .contains(TextureUsages::COPY_DST)
        );
        assert!(
            img.texture_descriptor
                .usage
                .contains(TextureUsages::TEXTURE_BINDING)
        );
    }
}
