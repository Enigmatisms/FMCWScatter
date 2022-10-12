#include <curand.h>
#include <curand_kernel.h>
#include "cuda_utils.cuh"
#include "ray_trace_kernel.cuh"
#include "sampler_kernel.cuh"

__forceinline__ __host__ __device__ Vec2 get_specular_dir(const Vec2& inc_dir, const Vec2& norm_dir) {
    const float proj = inc_dir.dot(norm_dir);
    return inc_dir - norm_dir * 2.f * proj;
}

__device__ bool snells_law(const Vec2& inci_dir, const Vec2& norm_dir, float n1_n2_ratio, bool same_dir, float& output) {
    // if inci dir is of the same direction as the normal dir, it means that the ray is transmitting out from the media
    n1_n2_ratio = 1. / n1_n2_ratio * same_dir + n1_n2_ratio * (1. - same_dir);
    float sin_val = ((norm_dir.y * inci_dir.x - norm_dir.x * inci_dir.y) * n1_n2_ratio);
    bool return_flag = abs(sin_val) <= 1.0;
    if (return_flag == true) {
        float result = asinf(sin_val);
        output = (PI - result) * (1. - same_dir) + result * same_dir;
    }
}

// frensel_equation for natural light (no polarization)
__device__ float frensel_equation_natural(float n1, float n2, float cos_inc, float cos_ref) {
    float n1cos_i = n1 * cos_inc;
    float n2cos_i = n2 * cos_inc;
    float n1cos_r = n1 * cos_ref;
    float n2cos_r = n2 * cos_ref;
    float rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r);
    float rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i);
    return 0.5 * (rs * rs + rp * rp);
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
// Random number is needed here, for reflection and transmission can both happen
__global__ void frensel_eff_sampler_kernel(const short* const mesh_inds, Vec2* ray_os, Vec2* ray_d, size_t rand_offset) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];
    const short obj_ind = obj_inds[mesh_ind];
    if (objects[obj_ind].type == Material::REFRACTIVE) {           // only specular objects will be processed here

    }
    __syncthreads();
    // CU_RAY_INFO should be used, since there is media interaction
}
