#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>
#include "../include/scatter_kernel.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/ray_trace_kernel.cuh"

__device__ void henyey_greenstein_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // H-G scattering
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset, &rand_state);
    const float next_rd = 2.f * curand_uniform(&rand_state) - 1.f;
    const float p_c = obj.p_c;
    const float inner = (1.f - p_c) / (1.0000001f + p_c) * tanf(PI_2 * next_rd);
    const float cos_t = fmin( cosf( 2.f * atanf(inner) ), 1.f );
    const float sin_t = sgn(next_rd) * sqrtf(fmax(0.f, 1.f - cos_t * cos_t));
    const float out_x = output.y * sin_t + output.x * cos_t;
    const float out_y = -output.x * sin_t + output.y * cos_t;
    output.x = out_x;
    output.y = out_y;
}

// about fmax, fmin: https://stackoverflow.com/questions/64156448/how-to-convince-cmake-to-use-the-cuda-fmax-function-instead-of-the-std-cmath-fun
__device__ void rayleigh_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // Rayleign scattering
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset, &rand_state);
    const float next_rd = 2.f * curand_uniform(&rand_state) - 1.f;
    const float u = -cbrtf(2.f * next_rd + sqrtf(4.f * next_rd * next_rd + 1.0f));
    const float cos_t = fmax(-1.f, fmin(1.f, u - 1.f / u));
    const float sin_t = sgn(next_rd) * sqrtf(fmax(0.f, 1.f - cos_t * cos_t));
    const float out_x = output.y * cos_t + output.x * sin_t;
    const float out_y = -output.x * cos_t + output.y * sin_t;
    output.x = out_x;
    output.y = out_y;
}

__device__ void isotropic_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // Isotropic phase function: input photon will be scatter to all directions uniformly
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset, &rand_state);
    float angle = (curand_uniform(&rand_state) - 0.5f) * PI_D;
    output = rotate_unit_vec(output, angle);
}

// TODO: the code should be modified now, we should directly pass the ray
/**
 * @brief if ray_dir is stored along with ray_info, there would be more code considering memcpy (from host to device)
 * I don't want to store a pointer in RayInfo, since there could be two memory accesses (global)
 */
__device__ void scattering_interaction(
    const ObjInfo& obj, RayInfo& rayi, Vec2& ray_dir, ScatFuncType pfunc,
    bool& in_media, int ray_id, short mesh_id, size_t rand_offset
) {
    Vec2 local_dir = ray_dir;
    // After `frensel_eff_sampler_kernel`, energy of a photon might be changed (early extinction)
    const bool original_in_media = in_media;
    bool local_medium_judge = original_in_media;
    bool path1 = false;
    if (local_medium_judge == false) {
        path1 = true;
        local_medium_judge = frensel_eff_sampler_kernel(obj, rayi, local_dir, rand_offset, ray_id, mesh_id);
    } else {
        printf("Born in media\n");
    }
    printf("Ray (%d) Hits: %d (%d), local: %f, %f, ray: %f, %f\n", ray_id, mesh_id, int(path1), local_dir.x, local_dir.y, ray_dir.x, ray_dir.y, local_dir.y);
    if (local_medium_judge) {
        pfunc(obj, local_dir, ray_id, rand_offset);
        // TODO： if scattering caused the photon to scatter out from the medium, what logic should it be? (this is a make-shift logic)
        // we can not set in_media to be false in this function but only when the photon is on the edge of object
        if (!original_in_media && local_dir.dot(all_normal[mesh_id]) > 0.) {
            local_medium_judge = false;
        }
    }
    // In case we access global memory too many times
    ray_dir = local_dir;
    in_media = local_medium_judge;
}
