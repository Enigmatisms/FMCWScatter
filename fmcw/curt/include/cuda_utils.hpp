#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

struct Vec2 {
    float x;
    float y;
    __host__ __device__ constexpr Vec2(float x, float y): x(x), y(y) {}
    __host__ __device__ constexpr Vec2(): x(0.), y(0.) {}

    __host__ __device__ Vec2 operator-(const Vec2& p) const {
        return Vec2(x - p.x, y - p.y);         // Return value optimized?
    }

    __host__ __device__ Vec2 operator*(float scaler) const {
        return Vec2(x * scaler, y * scaler);         // Return value optimized?
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

__host__ __device__ Vec2 rotate_unit_vec(const Vec2& input, float angle) {
    return input * cosf(angle)  - Vec2(-input.y, input.x) * sinf(angle);
}