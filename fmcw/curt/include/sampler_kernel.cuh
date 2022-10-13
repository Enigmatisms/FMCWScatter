#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

__global__ void non_scattering_interact_kernel(const short* const mesh_inds, Vec2* ray_d, size_t rand_offset);
