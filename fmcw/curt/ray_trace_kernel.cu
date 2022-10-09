#include <cstdio>
#include <cmath>
#include "ray_trace_kernel.hpp"

constexpr float PI = 3.14159265358979f;
constexpr float PI_2 = PI / 2.;

__host__ int get_padded_len(int non_padded) {
    return static_cast<int>(ceilf(static_cast<float>(non_padded) / 4.));
}

__forceinline__ __device__ void range_min(const float* const input, const uint8* const aux, int start, int end, float& out, uint8& out_aux) {
    float min_depth = 1e6;
    uint8 min_obj_ind = NULL_HIT;
    for (int i = start; i < end; i++) {
        float local_depth = input[i];
        if (local_depth < min_depth) {
            min_depth = local_depth;
            min_obj_ind = aux[i];
        }
    }
    out = min_depth; 
    out_aux = min_obj_ind; 
} 

/** 
 * @brief calculate whether a line intersects aabb 
 * input: id of an aabb, ray origin and ray direction
 * detailed derivation of aabb intersection should be deduced
 */
__device__ bool aabb_intersected(const Vec2& const ray_o, float dx, float dy, int aabb_id) {
    const AABB& aabb = aabbs[aabb_id];
    bool result = false, dx_valid = fabs(dx) > 1e-4f, dy_valid = fabs(dy) > 1e-4f;
    bool x_singular_valid = (ray_o.x < aabb.tl.x && ray_o.x > aabb.br.x);     // valid condition when dx is too small
    bool y_singular_valid = (ray_o.y < aabb.tl.y && ray_o.y > aabb.br.y);     // valid condition when dy is too small
    if (dx_valid && dy_valid) {        // there might be warp divergence (hard to resolve)
        const float enter_xt = (aabb.br.x - ray_o.x) / dx, enter_yt = (aabb.br.y - ray_o.y) / dy;
        const float exit_xt = (aabb.tl.x - ray_o.x) / dx, exit_yt = (aabb.tl.y - ray_o.y) / dy;
        const float enter_t = fmax(enter_xt, enter_yt), exit_t = fmin(exit_xt, exit_yt);
        bool back_cull = ((enter_xt < 0.f && exit_xt < 0.f) || (exit_xt < 0.f && exit_yt < 0.f));       // either pair of (in, out) being both neg, culled.
        result = (!back_cull) & (enter_t < exit_t);     // not back-culled and enter_t is smaller
    }
    result |= ((!dx_valid) & x_singular_valid);         // if x is not valid (false, ! -> true), then use x_singular_valid
    result |= ((!dy_valid) & y_singular_valid);         // if y is not valid (false, ! -> true), then use x_singular_valid
    return result;
}

// v_perp is the 90 deg rotated directional vector of current ray
__forceinline__ __device__ float ray_intersect(const Vec2& pos, const Vec2& v_perp, const Vec2& p1, const Vec2& p2) {
    const Vec2 s2e = p2 - p1;
    const Vec2 obs_v = pos - p1;
    const float D = v_perp.dot(s2e);
    int result = 1e6;
    if (D > 0.) {
        float alpha = v_perp.dot(obs_v) / D;
        if (alpha < 1. && alpha > 0.) {
            result = (-s2e.y * obs_v.x + s2e.x * obs_v.y) / D;
        }
    }
    return result;
}

/**
 * input : point origin (Vec2 array), ray angles: float array
 * output1 : minimum depth (float (single value, since each block represent one single ray) should be converted back to int)
 * output2 : the obj_index (of the nearest hit line seg or NULL_HIT flag)
 * @param depth is GLOBAL memory float array (for fast data copying)
 * @note this is a global function, not where the host could call. Also, AABB will not make this algo faster (only lower the power consumption)
 */
__global__ void ray_trace_cuda_kernel(
    const Vec2* const origins, const float* const ray_dir, float* const min_depths, uint8* const inds
) {
    // mem consumption: mesh_num * 4 bytes (all ranges) + (mesh_num * 1 bytes (all obj_inds) + first padding) + 
    // + 32 bytes (8 floats for stratified comp) + 8 bytes (2 floats -> 8 uint8) + (AABB_NUM * 1 bytes + second padding)
    // TODO: there are two paddings to be done: first for non-4-multiplier obj_inds, second for bool
    // TODO: shared memory initialization
    extern __shared__ float shared_banks[];      
    uint8* min_obj_inds = (uint8*) &shared_banks[mesh_num];   
    bool* hit_flags = (bool*) &shared_banks[mesh_num + padded_ind_floats + 10];

    const int mesh_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Vec2& ray_o = origins[blockIdx.x];
    const float ray_angle = ray_dir[blockIdx.x];
    const float dx = cosf(ray_angle), dy = sinf(ray_angle);
    if (mesh_id < aabb_num) {       // first (aabb_num) threads should process AABB calculation, others remains idle
        // Bank conflict unresolvable (haven't found a very effective way)
        hit_flags[mesh_id] = aabb_intersected(ray_o, dx, dy, mesh_id);
    }
    __syncthreads();
    if (mesh_id < mesh_num) {       // for the sake of mesh (line segment division), there might be more threads than needed
        uint8 aabb_index = obj_inds[mesh_id];
        if (hit_flags[aabb_index] == true) {        // there will be no warp divergence, since the 'else' side is NOP
            // store to shared_banks (atomic)
            const Vec2 v_perp(-dy, dx);
            const int next_id = next_ids[mesh_id];
            const int next_id_neg = int(next_ids[mesh_id] < 0);
            const int next_pt_id = next_id * next_id_neg + 1 - next_id_neg;
            shared_banks[mesh_id] = ray_intersect(ray_o, v_perp, all_points[mesh_id], all_points[mesh_id + next_pt_id]);
            min_obj_inds[mesh_id] = aabb_index;
        }
    }
    __syncthreads();
    float* local_min_depths = &shared_banks[mesh_num + padded_ind_floats];
    uint8* local_obj_inds = (uint8*) &shared_banks[mesh_num + padded_ind_floats + 8];
    if (threadIdx.x == 0) {             // 8-thread parallel
        int max_bound = min(mesh_num, blockDim.x * (threadIdx.y + 1));
        range_min(shared_banks, min_obj_inds, blockDim.x * threadIdx.y, max_bound, local_min_depths[mesh_id], local_obj_inds[mesh_id]);
    }
    __syncthreads();
    if (mesh_id == 0) {             // only one thread attend to the final output
        range_min(local_min_depths, local_obj_inds, 0, blockDim.y, min_depths[blockIdx.x], inds[blockIdx.x]);
    }
    __syncthreads();
}
