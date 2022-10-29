#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>

#define NULL_HIT 255            // if nothing is hit (unbounded scenes), 255 is assumed, therefore, maximum number of obj is 255
#define MESH_NULL -1

struct Vec2 {
    float x;
    float y;
    __host__ __device__ Vec2(float x, float y): x(x), y(y) {}
    __host__ __device__ Vec2() {}

    __host__ __device__ Vec2 operator-(const Vec2& p) const {
        return Vec2(x - p.x, y - p.y);         // Return value optimized?
    }

    __host__ __device__ Vec2 operator+(const Vec2& p) const {
        return Vec2(x + p.x, y + p.y);         // Return value optimized?
    }

    __host__ __device__ void operator-=(const Vec2& p) {
        x -= p.x;
        y -= p.y;
    }

    __host__ __device__ void operator+=(const Vec2& p) {
        x += p.x;
        y += p.y;
    }

    __host__ __device__ void operator*=(float v) {
        x *= v;
        y *= v;
    }

    __host__ __device__ Vec2 operator*(float scaler) const {
        return Vec2(x * scaler, y * scaler);         // Return value optimized?
    }

    __host__ __device__ float dot(const Vec2& p) const {
        return x * p.x + y * p.y;
    }

    __host__ __device__ float dot(Vec2&& p) const {
        return x * p.x + y * p.y;
    }

    __host__ __device__ float norm() const {
        return sqrtf(x * x + y * y);
    }
};

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float x, float y, float z): x(x), y(y), z(z) {}
};

__forceinline__ __host__ __device__ Vec2 rotate_unit_vec(const Vec2& input, float angle) {
    return input * cosf(angle) - Vec2(-input.y, input.x) * sinf(angle);
}

__forceinline__ __host__ __device__ Vec2 rotate_unit_vec(Vec2&& input, float angle) {       // @overload for rvalue input
    return input * cosf(angle) - Vec2(-input.y, input.x) * sinf(angle);
}

__forceinline__ __host__ __device__ int get_padded_len(int non_padded, float k = 4.) {
    return static_cast<int>(ceilf(static_cast<float>(non_padded) / k));
}

__forceinline__ __host__ __device__ int pad_bytes(int non_padded) {
    int padding = non_padded % 4;
    return non_padded + (padding > 0) * (4 - padding);
}

__forceinline__ __device__ float sgn(float val) {
    const bool pos = val >= 0.;
    return -1. + 2 * pos;
}

constexpr float PI = 3.14159265358979f;
constexpr float PI_2 = PI / 2.;
constexpr float PI_D = PI * 2.;