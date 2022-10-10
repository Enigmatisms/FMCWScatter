#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_utils.cuh"

#define MAX_PNUM 1024

// Note that __constant__ is not big （65536 bytes）, total consumption(1024 -> 15360 bytes)： assume MAX_PNUM = 1024
// There fore, MAX_PNUM can be set to 2048 (maximum, we make some space for possible future features)
// TODO: to enable this extern __constant__: set(CUDA_SEPARABLE_COMPILATION ON)
extern __constant__ Vec2 all_points[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
extern __constant__ AABB aabbs[MAX_PNUM >> 2];     // 256 * 4 * 4 = 4096 bytes used (maximum allowed object number 255)
extern __constant__ short obj_inds[MAX_PNUM];      // line segs -> obj (LUT) (material and media & AABB）(2048 bytes used)
extern __constant__ char next_ids[MAX_PNUM];       // 1024 bytes used

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
    const Vec2* const origins, const Vec2* const ray_dir, 
    float* const min_depths, short* const inds, int block_offset, int mesh_num, int aabb_num
);
