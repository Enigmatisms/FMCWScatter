#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_utils.cuh"

class World {
public:
    __host__ __device__ World() {};
public:
    // TODO: what about the logic when the world is full of medium (single type)
    float scale;
    ObjInfo w;
};

extern __constant__ World world;