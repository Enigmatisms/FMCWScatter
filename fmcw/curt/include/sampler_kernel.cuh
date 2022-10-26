#pragma once
#include <cmath>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include "cuda_utils.cuh"

__global__ void general_interact_kernel(const short* const mesh_inds, RayInfo* const ray_info, Vec2* ray_d, size_t rand_offset);

// Refraction direction sampler (including surface reflection)
__device__ bool frensel_eff_sampler_kernel(const ObjInfo& object, Vec2& ray_d, size_t rand_offset, int ray_id, short mesh_ind);

__forceinline__ __device__ void general_reflection(const Vec2& normal, const Vec2& ray_dir, curandState& rstate, Vec2& output, float rdist);

__forceinline__ __host__ __device__ Vec2 get_specular_dir(const Vec2& inc_dir, const Vec2& norm_dir) {
    const float proj = inc_dir.dot(norm_dir);
    return inc_dir - norm_dir * 2.f * proj;
}

__forceinline__ __device__ float sgn(float val) {
    const bool pos = val >= 0.;
    return -1. + 2 * pos;
}
