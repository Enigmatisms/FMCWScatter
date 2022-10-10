#include <cstdio>
#include <cmath>
#include "ray_trace_kernel.hpp"
#include "cuda_err_check.hpp"

#define MAX_PNUM 1024
#define NULL_HIT 255            // if nothing is hit (unbounded scenes), 255 is assumed, therefore, maximum number of obj is 255

constexpr float PI = 3.14159265358979f;
constexpr float PI_2 = PI / 2.;

// Note that __constant__ is not big （65536 bytes）, total consumption(1024 -> 15360 bytes)： assume MAX_PNUM = 1024
// There fore, MAX_PNUM can be set to 2048 (maximum, we make some space for possible future features)
__constant__ Vec2 all_points[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
__constant__ AABB aabbs[MAX_PNUM >> 2];     // 256 * 4 * 4 = 4096 bytes used (maximum allowed object number 255)
__constant__ short obj_inds[MAX_PNUM];      // line segs -> obj (LUT) (material and media & AABB）(2048 bytes used)
__constant__ char next_ids[MAX_PNUM];       // 1024 bytes used

void static_scene_update(
    const Vec2* const meshes, const AABB* const host_aabb, const short* const host_inds, 
    const char* const host_nexts, size_t line_seg_num, size_t aabb_num
) {
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(all_points, meshes, sizeof(Vec2) * line_seg_num, 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(aabbs, host_aabb, sizeof(AABB) * aabb_num, 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(obj_inds, host_inds, sizeof(short) * line_seg_num, 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(next_ids, host_nexts, sizeof(char) * line_seg_num, 0, cudaMemcpyHostToDevice));
}

__forceinline__ __device__ void range_min(const float* const input, int start, int end, float& out, short& out_aux, const short* const aux = nullptr) {
    float min_depth = 9e5f;
    short min_mesh_ind = NULL_HIT;
    for (int i = start; i < end; i++) {
        float local_depth = input[i];
        if (local_depth < min_depth) {
            min_depth = local_depth;
            min_mesh_ind = (aux == nullptr) ? i : aux[i];
        }
    }
    out = min_depth; 
    out_aux = min_mesh_ind; 
} 

/** 
 * @brief calculate whether a line intersects aabb 
 * input: id of an aabb, ray origin and ray direction
 * detailed derivation of aabb intersection should be deduced
 */
__device__ bool aabb_intersected(const Vec2& const ray_o, float dx, float dy, int aabb_id) {
    const AABB& aabb = aabbs[aabb_id];
    bool result = false, dx_valid = fabs(dx) > 2e-5f, dy_valid = fabs(dy) > 2e-5f;
    bool x_singular_valid = (ray_o.x < aabb.tl.x && ray_o.x > aabb.br.x);     // valid condition when dx is too small
    bool y_singular_valid = (ray_o.y < aabb.tl.y && ray_o.y > aabb.br.y);     // valid condition when dy is too small
    if (dx_valid && dy_valid) {        // there might be warp divergence (hard to resolve)
        const float enter_xt = (aabb.br.x - ray_o.x) / dx, enter_yt = (aabb.br.y - ray_o.y) / dy;
        const float exit_xt = (aabb.tl.x - ray_o.x) / dx, exit_yt = (aabb.tl.y - ray_o.y) / dy;
        const float enter_t = fmax(enter_xt, enter_yt), exit_t = fmin(exit_xt, exit_yt);
        bool back_cull = ((enter_xt < 0.f && exit_xt < 0.f) || (enter_yt < 0.f && exit_yt < 0.f));       // either pair of (in, out) being both neg, culled.
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
    float result = 1e6;
    if (fabs(D) > 5e-5) {
        float alpha = v_perp.dot(obs_v) / D;
        if (alpha < 1. && alpha > 0.) {
            float tmp = (-s2e.y * obs_v.x + s2e.x * obs_v.y) / D;
            float flag = float(tmp > 0.);
            result = tmp * flag + 1e6 * (1. - flag);
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
    const Vec2* const origins, const float* const ray_dir, 
    float* const min_depths, short* const inds, int block_offset, int mesh_num, int aabb_num
) {
    // mem consumption: (1) mesh_num * 4 bytes (for all ranges) (2) 8 * float (min ranges, stratified) -> 32 bytes 
    // (3) 8 * short -> 4 floats -> 16 bytes (4) AABB valid bools (1 bytes * num AABB) padding
    extern __shared__ float shared_banks[];      
    bool* hit_flags = (bool*) &shared_banks[mesh_num + 12];
    const int ray_id = blockIdx.x + gridDim.x * block_offset;

    const int mesh_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Vec2& ray_o = origins[ray_id];
    const float ray_angle = ray_dir[ray_id];
    const float dx = cosf(ray_angle), dy = sinf(ray_angle);
    if (mesh_id < aabb_num) {       // first (aabb_num) threads should process AABB calculation, others remains idle
        // Bank conflict unresolvable (haven't found a very effective way)
        hit_flags[mesh_id] = aabb_intersected(ray_o, dx, dy, mesh_id);
    }
    __syncthreads();
    if (mesh_id < mesh_num) {       // for the sake of mesh (line segment division), there might be more threads than needed
        short aabb_index = obj_inds[mesh_id];
        if (hit_flags[aabb_index] == true) {        // there will be no warp divergence, since the 'else' side is NOP
            const Vec2 v_perp(-dy, dx);
            const int next_id = next_ids[mesh_id];
            const int next_id_neg = int(next_ids[mesh_id] < 0);
            const int next_pt_id = next_id * next_id_neg + 1 - next_id_neg;
            shared_banks[mesh_id] = ray_intersect(ray_o, v_perp, all_points[mesh_id], all_points[mesh_id + next_pt_id]);
            // as we use mesh indices, we don't have to store them in shared memory 
        }
    }
    __syncthreads();
    float* local_min_depths = &shared_banks[mesh_num];
    short* local_obj_inds = (short*) &shared_banks[mesh_num + 8];
    if (threadIdx.x == 0) {             // 8-thread parallel
        int max_bound = min(mesh_num, blockDim.x * (threadIdx.y + 1));
        range_min(shared_banks, blockDim.x * threadIdx.y, max_bound, local_min_depths[mesh_id], local_obj_inds[mesh_id]);
    }
    __syncthreads();
    if (mesh_id == 0) {             // only one thread attend to the final output
        range_min(local_min_depths, 0, blockDim.y, min_depths[ray_id], inds[ray_id], local_obj_inds);
    }
    __syncthreads();
}

// Diffusive reflection light ray direction sampler
__global__ void diffusive_ref_sampler_kernel() {
    
}

// Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel() {

}

// Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel() {
    
}

// Frensel reflection (can be reflected or refracted)
__global__ void frensel_eff_sampler_kernel() {

}
