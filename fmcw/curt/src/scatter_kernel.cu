#include <curand.h>
#include <curand_kernel.h>
#include "../include/cuda_utils.cuh"
#include "../include/scatter_kernel.cuh"
#include "../include/sampler_kernel.cuh"

__device__ void scattering_interaction(const ObjInfo& obj, const Vec2& normal, const Vec2& ray_dir, curandState& rstate, Vec2& output, bool in_media) {
    //
}