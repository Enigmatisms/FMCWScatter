#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#define MAX_PNUM 1024
#define NULL_HIT 255            // if nothing is hit (unbounded scenes), 255 is assumed, therefore, maximum number of obj is 255

using uint8 = unsigned char;

struct Vec2 {
    float x;
    float y;
    __host__ __device__ constexpr Vec2(float x, float y): x(x), y(y) {}
    __host__ __device__ constexpr Vec2(): x(0.), y(0.) {}

    __host__ __device__ Vec2 operator-(const Vec2& p) const {
        return Vec2(x - p.x, y - p.y);         // Return value optimized?
    }

    __host__ __device__ float dot(const Vec2& p) const {
        return x * p.x + y * p.y;
    }
};

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ constexpr Vec3(): x(0.), y(0.), z(0.) {}
    __host__ __device__ constexpr Vec3(float x, float y, float z): x(x), y(y), z(z) {}
};

// Axis-aligned bounding box for objects
struct AABB {
    Vec2 tl;        // top left point
    Vec2 br;        // bottom right point

    __host__ __device__ constexpr AABB(): tl(1., 1.), br(0., 0.) {}
    __host__ __device__ constexpr AABB(const Vec2& tl, const Vec2& br): tl(tl), br(br) {}
};

// 
// TODO: These should be copied from CPU (memcpyFromSymbol)
// Note that __constant__ is not big （65536 bytes）, total consumption(1024 -> 17408 bytes)： assume MAX_PNUM = 1024
// There fore, MAX_PNUM can be set to 3072 (maximum) (Unfortunately, atomic function for short / char is absent)
__constant__ Vec2 all_points[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
__constant__ AABB aabbs[MAX_PNUM >> 2];     // 256 * 4 * 4 = 4096 bytes used (maximum allowed object number 255)
__constant__ uint8 obj_inds[MAX_PNUM];      // line segs -> obj (LUT) (material and media & AABB）(4096 bytes used)
__constant__ char next_ids[MAX_PNUM];       // 1024 bytes used
int mesh_num;                               // number of line segments (set during const mem copying)
int aabb_num;                               // number of aabb
int padded_ind_floats;                       // number of bytes taken by indices after padding (to 4-byte floats)

/**
 * input : point origin (Vec2 array), ray angles: float array
 * output1 : minimum depth (float (single value, since each block represent one single ray) should be converted back to int)
 * output2 : the obj_index (of the nearest hit line seg or NULL_HIT flag)
 * extra info: number of AABB, number of line segs
 * @param depth is GLOBAL memory float array (for fast data copying)
 */
__global__ void ray_trace_cuda_kernel(
    const Vec2* const origins, const float* const ray_dir, float* const min_depths, uint8* const inds
);

// TODO: Diffusive reflection light ray direction sampler
__global__ void diffusive_ref_sampler_kernel();

// TODO: Glossy object (rough specular) reflection light ray direction sampler
__global__ void glossy_ref_sampler_kernel();

// TODO: Mirror-like object (pure specular - Dirac BRDF) reflection light ray direction sampler
__global__ void specular_ref_sampler_kernel();