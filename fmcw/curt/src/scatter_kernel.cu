#include <curand.h>
#include <curand_kernel.h>
#include "../include/scatter_kernel.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/ray_trace_kernel.cuh"

// TODO: setup function (https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/)
__device__ ScatFuncType hg_pfunc, rl_pfunc, iso_pfunc;       // directly pass the function pointter to the function

__device__ void henyey_greenstein_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // H-G scattering

}

__device__ void rayleign_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // Rayleign scattering
}

__device__ void isotropic_phase(const ObjInfo& obj, Vec2& output, int ray_id, size_t rand_offset) {
    // Isotropic phase function: input photon will be scatter to all directions uniformly
    curandState rand_state;
    curand_init(ray_id, 0, rand_offset + ray_id, &rand_state);
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
    Vec2 local_dir;
    // After `frensel_eff_sampler_kernel`, energy of a photon might be changed (early extinction)
    bool local_medium_judge = frensel_eff_sampler_kernel(obj, rayi, local_dir, rand_offset, ray_id, mesh_id);
    if (in_media) {
        pfunc(obj, local_dir, ray_id, rand_offset);
        // TODO： if scattering caused the photon to scatter out from the medium, what logic should it be? (this is a make-shift logic)
        if (local_dir.dot(all_normal[mesh_id]) > 0.) {      // same direction means out-penetration
            local_medium_judge = false;
        }
    }
    // In case we access global memory too many times
    ray_dir = local_dir;
    in_media = local_medium_judge;
}
