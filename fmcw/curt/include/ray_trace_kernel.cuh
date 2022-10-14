#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_utils.cuh"

#define MAX_PNUM 1024

// not constant memory implementation, therefore no need to consider the memory bound
// TODO: to enable this extern __constant__: set(CUDA_SEPARABLE_COMPILATION ON)
extern __device__ Vec2 all_points[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
extern __device__ Vec2 all_normal[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
extern __device__ ObjInfo objects[MAX_PNUM >> 3];     // 128 * 4 * 12 = 6144 bytes used (maximum allowed object number 255)
extern __device__ short obj_inds[MAX_PNUM];      // line segs -> obj (LUT) (material and media & AABB）(2048 bytes used)
extern __device__ char next_ids[MAX_PNUM];       // 1024 bytes used

__global__ void calculate_normal(int line_seg_num);

/**
 * input : point origin (Vec2 array), ray angles: float array
 * output1 : minimum depth (float (single value, since each block represent one single ray) should be converted back to int)
 * output2 : the obj_index (of the nearest hit line seg or NULL_HIT flag)
 * extra info: number of AABB, number of line segs
 * @param depth is GLOBAL memory float array (for fast data copying)
 */
__global__ void ray_trace_cuda_kernel(
    const Vec2* const origins, const Vec2* const ray_dir, Vec2* const intersects,
    RayInfo* const ray_info, short* const inds, int block_offset, int mesh_num, int aabb_num
);

__global__ void copy_ray_poses_kernel(const Vec2* const intersections, Vec2* const ray_os, Vec2* const ray_ds);
