#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>

__global__ void general_interact_kernel(const short* const mesh_inds, Vec2* ray_d, size_t rand_offset);

__forceinline__ __device__ void general_reflection(const Vec2& normal, const Vec2& ray_dir, curandState& rstate, Vec2& output, float rdist);

__forceinline__ __host__ __device__ Vec2 get_specular_dir(const Vec2& inc_dir, const Vec2& norm_dir) {
    const float proj = inc_dir.dot(norm_dir);
    return inc_dir - norm_dir * 2.f * proj;
}

__forceinline__ __device__ float sgn(float val) {
    const bool pos = val >= 0.;
    return -1. + 2 * pos;
}
