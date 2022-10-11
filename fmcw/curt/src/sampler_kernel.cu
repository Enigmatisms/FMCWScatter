#include <curand.h>
#include <curand_kernel.h>
#include "cuda_utils.cuh"
#include "ray_trace_kernel.cuh"
#include "sampler_kernel.cuh"

__forceinline__ __host__ __device__ Vec2 get_specular_dir(const Vec2& inc_dir, const Vec2& norm_dir) {
    const float proj = inc_dir.dot(norm_dir);
    return inc_dir - norm_dir * 2.f * proj;
}

/**
 * TODO: next steps 
 * 1. Implement sampler (4 kinds of sampler, initially)
 * 2. Test run, build a testing visualization platform
 */

// Diffusive reflection light ray direction sampler
// block separation (to 8 blocks, 2048 rays)
__global__ void diffusive_ref_sampler_kernel(const short* const mesh_inds, Vec2* ray_os, Vec2* ray_d, size_t rand_offset) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];                                               // the line segment currently hit by the ray
    const short obj_ind = obj_inds[mesh_ind];

    if (objects[obj_ind].type == Material::DIFFUSE) {           // only diffusive objects will be processed here
        curandState rand_state;
        curand_init(ray_id, 0, rand_offset, &rand_state);
        const float sampled_angle = curand_uniform(&rand_state) * (PI - 2e-4) - PI_2 + 1e-4;    // we can not have exact pi/2 or -pi/2
        ray_d[ray_id] = rotate_unit_vec(all_normal[mesh_ind], sampled_angle);                   // diffusive (rotate normal from -pi/2 to pi/2)
    }
    __syncthreads();
}

// Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel(const short* const mesh_inds, Vec2* ray_os, Vec2* ray_d, size_t rand_offset) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];
    const short obj_ind = obj_inds[mesh_ind];
    if (objects[obj_ind].type == Material::GLOSSY) {           // only glossy objects will be processed here
        curandState rand_state;
        curand_init(ray_id, 0, rand_offset, &rand_state);
        const Vec2 reflected_dir = get_specular_dir(ray_d[ray_id], all_normal[mesh_ind]);
        float sampled_angle = curand_normal(&rand_state) * 0.5;             // 3 sigma is 1.5, which is little bit smaller than pi/2
        sampled_angle = fmax(fmin(sampled_angle, PI_2), -PI_2);             // clamp to (-pi/2, pi/2)
        ray_d[ray_id] = rotate_unit_vec(reflected_dir, sampled_angle);      // glossy specular
    }
    __syncthreads();
}

// Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel(const short* const mesh_inds, Vec2* ray_os, Vec2* ray_d) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];
    const short obj_ind = obj_inds[mesh_ind];
    if (objects[obj_ind].type == Material::SPECULAR) {           // only specular objects will be processed here
        const Vec2 reflected_dir = get_specular_dir(ray_d[ray_id], all_normal[mesh_ind]);
        ray_d[ray_id] = get_specular_dir(ray_d[ray_id], all_normal[mesh_ind]);   // diffusive
    }
    __syncthreads();
}

// Frensel reflection (can be reflected or refracted)
__global__ void frensel_eff_sampler_kernel() {
    // CU_RAY_INFO should be used, since there is media interaction
}
