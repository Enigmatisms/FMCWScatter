#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "cuda_utils.cuh"

using ScatFuncType = void (*) (const ObjInfo&, Vec2&);

extern __device__ ScatFuncType hg_pfunc, rl_pfunc, iso_pfunc;       // directly pass the function pointter to the function

__device__ void henyey_greenstein_phase(const ObjInfo& obj, Vec2& output);

__device__ void rayleign_phase(const ObjInfo& obj, Vec2& output);

__device__ void isotropic_phase(const ObjInfo& obj, Vec2& output);

__device__ void scattering_interaction(
    const ObjInfo& obj, Vec2& ray_dir, ScatFuncType func,
    bool& in_media, short mesh_id, size_t rand_offset
);
