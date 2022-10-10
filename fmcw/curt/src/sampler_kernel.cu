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
 * 1. Create object properties (enum class to indicate the material used) (immediately used in sampler)
 * 2. Implement sampler (4 kinds of sampler, initially)
 * 3. Test run, build a testing visualization platform
 */

// Diffusive reflection light ray direction sampler
// block separation (to 8 blocks, 2048 rays)
__global__ void diffusive_ref_sampler_kernel(const short* const mesh_inds, Vec2* ray_os, Vec2* ray_d, size_t rand_offset) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];                                               // the line segment currently hit by the ray
    const short obj_ind = obj_inds[mesh_ind];

    // TODO: IMPORTANT! We should access the properties of the object (find out whether the object is diffusive material)
    if (true) {
        curandState rand_state;
        curand_init(ray_id, 0, rand_offset, &rand_state);
        const float sampled_angle = curand_uniform(&rand_state) * (PI - 2e-4) - PI_2 + 1e-4;    // we can not have exact pi/2 or -pi/2
        ray_d[ray_id] = rotate_unit_vec(ray_d[ray_id], sampled_angle);                          // diffusive
    }
}

// Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel() {

}

// Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel() {
    
}

// Frensel reflection (can be reflected or refracted)
__global__ void frensel_eff_sampler_kernel() {
    // CU_RAY_INFO should be used, since there is media interaction
}
