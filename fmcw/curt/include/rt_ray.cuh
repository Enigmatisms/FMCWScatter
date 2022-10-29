#pragma once
#include "cuda_utils.cuh"

struct RayInfo {
    bool is_in_media;
    bool terminated;            // is terminated, this photon is no longer traced.

    // range bound is the maximum range of a ray (due to occluders), if in a scattering media
    // we will sample by mean free path (which is often smaller than range bound)
    float range_bound;    
    float acc_range;  

    // This field is reserved for future use (energy decays during bouncing)
    float energy;

    __host__ __device__ constexpr RayInfo(): is_in_media(false), terminated(false), 
        range_bound(1e4), acc_range(0.), energy(1.0) {}
};
