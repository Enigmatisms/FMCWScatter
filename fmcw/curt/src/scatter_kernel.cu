#include <curand.h>
#include <curand_kernel.h>
#include "../include/scatter_kernel.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/ray_trace_kernel.cuh"

// TODO: setup function (https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/)
__device__ ScatFuncType hg_pfunc, rl_pfunc, iso_pfunc;       // directly pass the function pointter to the function

__device__ void henyey_greenstein_phase(const ObjInfo& obj, Vec2& output) {
    // H-G scattering
}

__device__ void rayleign_phase(const ObjInfo& obj, Vec2& output) {
    // Rayleign scattering
}

__device__ void isotropic_phase(const ObjInfo& obj, Vec2& output) {
    // Isotropic scattering
}

__device__ void scattering_interaction(
    const ObjInfo& obj, Vec2& ray_dir, ScatFuncType pfunc,
    bool& in_media, short mesh_id, size_t rand_offset
) {
    Vec2 local_dir;
    bool local_medium_judge = frensel_eff_sampler_kernel(obj, local_dir, rand_offset, in_media, mesh_id);
    if (in_media) {
        pfunc(obj, local_dir);
        // TODO： if scattering caused the photon to scatter out from the medium, what logic should it be? (this is a make-shift logic)
        if (local_dir.dot(all_normal[mesh_id]) > 0.) {      // same direction means out-penetration
            local_medium_judge = false;
        }
    }
    // In case we access global memory too many times
    ray_dir = local_dir;
    in_media = local_medium_judge;
}
