#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>

#include "rt_ray.cuh"
#include "rt_objects.cuh"

using ScatFuncType = void (*) (const ObjInfo&, Vec2&, int, size_t);

extern __device__ ScatFuncType hg_pfunc, rl_pfunc, iso_pfunc;       // directly pass the function pointter to the function

__device__ void henyey_greenstein_phase(const ObjInfo& obj, Vec2& output);

__device__ void rayleign_phase(const ObjInfo& obj, Vec2& output);

__device__ void isotropic_phase(const ObjInfo& obj, Vec2& output);

__device__ void scattering_interaction(
    const ObjInfo& obj, RayInfo& rayi, Vec2& ray_dir, ScatFuncType func,
    bool& in_media, int ray_id, short mesh_id, size_t rand_offset
);
