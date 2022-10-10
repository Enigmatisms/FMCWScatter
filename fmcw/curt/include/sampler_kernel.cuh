#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

// TODO: Diffusive reflection light ray direction sampler
__global__ void diffusive_ref_sampler_kernel();

// TODO: Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel();

// TODO: Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel();

// TODO: Frensel reflection (can be reflected or refracted)
__global__ void frensel_eff_sampler_kernel();
