#include <iostream>
#include <device_launch_parameters.h>
#include "../include/ray_trace_kernel.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/scatter_kernel.cuh"

__device__ bool snells_law(const Vec2& inci_dir, const Vec2& norm_dir, float n1_n2_ratio, bool same_dir, float& output) {
    // if inci dir is of the same direction as the normal dir, it means that the ray is transmitting out from the media
    float sin_val = ((norm_dir.y * inci_dir.x - norm_dir.x * inci_dir.y) * n1_n2_ratio);
    bool return_flag = abs(sin_val) <= 1.0;
    if (return_flag == true) {
        float result = asinf(sin_val);
        output = (PI - result) * (1 - same_dir) + result * same_dir;
    }
    return return_flag;
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

// non-deterministic branch (light interaction with surface is not decided by Material tag)
__forceinline__ __device__ void general_reflection(const Vec2& normal, const Vec2& ray_dir, curandState& rstate, Vec2& output, float rdist) {
    if (rdist >= 0.) {
        Vec2 reflected_dir = get_specular_dir(ray_dir, normal);
        float sampled_angle = curand_normal(&rstate) * fmin(0.5f, fmax(rdist, 0.f));             // 3 sigma is 1.5, which is little bit smaller than pi/2
        sampled_angle = fmaxf(fminf(sampled_angle, PI_2 - 1e-4), -PI_2 + 1e-4);             // clamp to (-pi/2 + ɛ, pi/2 - ɛ)
        Vec2 output_vec = rotate_unit_vec(reflected_dir, sampled_angle);
        const float sign = sgn(normal.dot(ray_dir));
        if (output_vec.dot(normal * sign) >= 0.) {
            output_vec = reflected_dir;
        }
        output = output_vec;      // glossy specular
    } else {
        const float sampled_angle = curand_uniform(&rstate) * (PI - 2e-4) - PI_2 + 1e-4;
        output = rotate_unit_vec(normal, sampled_angle);
    }
}

// Diffusive reflection light ray direction sampler
// block separation (to 8 blocks, 2048 rays).
// TODO: ray_info (rayi) and obj (ObjInfo) are not used current, but will be of use in the future (considering energy decay)
__device__ void diffusive_ref_sampler_kernel(const ObjInfo& obj, const Vec2& normal, RayInfo& rayi, Vec2* ray_d, size_t rand_offset, int ray_id) {
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset + ray_id, &rand_state);
    const float sampled_angle = curand_uniform(&rand_state) * (PI - 2e-4) - PI_2 + 1e-4;    // we can not have exact pi/2 or -pi/2
    ray_d[ray_id] = rotate_unit_vec(normal, sampled_angle);                   // diffusive (rotate normal from -pi/2 to pi/2)
}

// Glossy object (rough specular) reflection light ray direction sampler
__device__ void glossy_ref_sampler_kernel(const ObjInfo& obj, const Vec2& normal, RayInfo& rayi, Vec2* ray_d, size_t rand_offset, int ray_id, short obj_ind) {
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset + ray_id, &rand_state);
    Vec2& ray_dir = ray_d[ray_id];
    general_reflection(normal, ray_dir, rand_state, ray_dir, objects[obj_ind].rdist);       // glossy specular
}

// Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__forceinline__ __device__ void specular_ref_sampler_kernel(const ObjInfo& obj, const Vec2& normal, RayInfo& rayi, Vec2* ray_d, int ray_id) {
    ray_d[ray_id] = get_specular_dir(ray_d[ray_id], normal);   // pure specular
}

// Frensel reflection (can be reflected or refracted) - general reflection (can be diffusive, glossy or specular)
// Random number is needed here, for reflection and transmission can both happen
__device__ bool frensel_eff_sampler_kernel(const ObjInfo& object, RayInfo& rayi, Vec2& ray_d, size_t rand_offset, int ray_id, short mesh_ind) {
    const Vec2& normal = all_normal[mesh_ind];
    const float ref_index = object.ref_index;
    const float rdist = object.rdist, r_gain = object.r_gain;
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset + ray_id, &rand_state);
    Vec2 refracted_dir, reflected_dir;
    general_reflection(normal, ray_d, rand_state, reflected_dir, rdist);

    float angle = 0., reflection_ratio = 1.0;
    const float cos_inc = ray_d.dot(normal), ri_sum = 1. + ref_index;         // TODO: substitude 1. to world RI 
    const bool same_dir = cos_inc > 0.;
    const float n1 = (1. - same_dir) + ref_index * same_dir;        // if same dir (out from media), n1 = ref_index, n2 = 1., else n1 = 1., n2 = ref_index
    // We do not account for transmitting from one media directly into another media
    const bool result_valid = snells_law(ray_d, normal, n1 / (ri_sum - n1), same_dir, angle);
    if (result_valid == true) {
        refracted_dir = rotate_unit_vec(normal, angle);
        reflection_ratio = frensel_equation_natural(n1, ri_sum - n1, fabs(cos_inc), fabs(cosf(angle))) * r_gain;
    }

    const bool is_reflection = curand_uniform(&rand_state) <= reflection_ratio;   // random choise of refracted or reflected
    ray_d = is_reflection ? reflected_dir : refracted_dir;          // warp divergence might be more efficient in this case
    return !(same_dir ^ is_reflection);         // XOR: when same_dir(1), is_ref(1) (penetrate out from medium but reflected -> is in media)
}

/// TODO: Logic failure: what if photon is not inside the object but the world (with scattering medium)?
/// The information need is not stored in objects (it should not be), but in __constant__ world
__global__ void general_interact_kernel(const short* const mesh_inds, RayInfo* const ray_info, Vec2* ray_d, size_t rand_offset) {
    const int ray_id = blockDim.x * blockIdx.x + threadIdx.x;
    const short mesh_ind = mesh_inds[ray_id];
    const short obj_ind = obj_inds[mesh_ind];
    // There is bound to be warp divergence (inevitable, or rather say, preferred)
    const ObjInfo& object = objects[obj_ind];
    const Vec2& normal = all_normal[mesh_ind];
    RayInfo& this_ray = ray_info[ray_id];
    switch (object.type) {
          case Material::DIFFUSE: {
            diffusive_ref_sampler_kernel(object, normal, this_ray, ray_d, rand_offset, ray_id); break;
        } case Material::GLOSSY: {
            glossy_ref_sampler_kernel(object, normal, this_ray, ray_d, rand_offset, ray_id, obj_ind); break;
        } case Material::SPECULAR: {
            specular_ref_sampler_kernel(object, normal, this_ray, ray_d, ray_id); break;
        } case Material::REFRACTIVE: {
            // TODO: more things should be accounted for: world refraction index, transimitting from media 1 to media 2
            frensel_eff_sampler_kernel(object, this_ray, ray_d[ray_id], rand_offset, ray_id, mesh_ind); break;
        } case Material::SCAT_ISO: {
            scattering_interaction(object, this_ray, ray_d[ray_id], isotropic_phase, ray_info[ray_id].is_in_media, ray_id, mesh_ind, rand_offset); break;
        } case Material::SCAT_HG: {
            scattering_interaction(object, this_ray, ray_d[ray_id], henyey_greenstein_phase, ray_info[ray_id].is_in_media, ray_id, mesh_ind, rand_offset); break;
        } case Material::SCAT_RAYLEIGH: {
            scattering_interaction(object, this_ray, ray_d[ray_id], rayleigh_phase, ray_info[ray_id].is_in_media, ray_id, mesh_ind, rand_offset); break;
        } default: {
            break;
        }
    }
    // Massive warp divergence and serialization
    __syncthreads();
}
