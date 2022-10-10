#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_utils.hpp"

// Copy line segment data from host (Rust end)
__host__ void static_scene_update(size_t line_seg_num);

/**
 * input : point origin (Vec2 array), ray angles: float array
 * output1 : minimum depth (float (single value, since each block represent one single ray) should be converted back to int)
 * output2 : the obj_index (of the nearest hit line seg or NULL_HIT flag)
 * extra info: number of AABB, number of line segs
 * @param depth is GLOBAL memory float array (for fast data copying)
 */
__global__ void ray_trace_cuda_kernel(
    const Vec2* const origins, const float* const ray_dir, 
    float* const min_depths, short* const inds, int block_offset, int mesh_num, int aabb_num
);

// TODO: Diffusive reflection light ray direction sampler
__global__ void diffusive_ref_sampler_kernel();

// TODO: Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel();

// TODO: Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel();

// TODO: Frensel reflection (can be reflected or refracted)
__global__ void frensel_eff_sampler_kernel();