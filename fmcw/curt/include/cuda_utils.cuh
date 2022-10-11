#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define NULL_HIT 255            // if nothing is hit (unbounded scenes), 255 is assumed, therefore, maximum number of obj is 255

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

    __host__ __device__ float norm() const {
        return sqrtf(x * x + y * y);
    }
};

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ constexpr Vec3(): x(0.), y(0.), z(0.) {}
    __host__ __device__ constexpr Vec3(float x, float y, float z): x(x), y(y), z(z) {}
};

struct RayInfo {
    short prev_obj_id;

    __host__ __device__ constexpr RayInfo(): prev_obj_id(NULL_HIT) {}
    __host__ __device__ constexpr RayInfo(short prev_id): prev_obj_id(prev_id) {}
};

// Axis-aligned bounding box for objects
struct AABB {
    Vec2 tl;        // top left point
    Vec2 br;        // bottom right point

    __host__ __device__ constexpr AABB(): tl(1., 1.), br(0., 0.) {}
    __host__ __device__ constexpr AABB(const Vec2& tl, const Vec2& br): tl(tl), br(br) {}
};

// Media and Material for objects
enum class Material: uint8_t {
    DIFFUSE = 0,
    GLOSSY = 1,
    SPECULAR = 2,
    REFRACTIVE = 3
};

// object-managing struct
struct ObjInfo {
    Material type;              // size is uint8
    uint8_t reserved[3];
    float ref_index;
    float u_a;                  // when material is not semi-transparent, u_a is the absorption coeff upon reflection
    float u_s;                  // scattering coeff
    float p_c;                  // when material is not semi-transparent, p_c is the coefficient of phase function
    float f_reserved[3];        // non-AABB part totaling 8 floats
    AABB aabb;
};

__forceinline__ __host__ __device__ Vec2 rotate_unit_vec(const Vec2& input, float angle) {
    return input * cosf(angle)  - Vec2(-input.y, input.x) * sinf(angle);
}

__forceinline__ __host__ __device__ Vec2 rotate_unit_vec(Vec2&& input, float angle) {
    return input * cosf(angle)  - Vec2(-input.y, input.x) * sinf(angle);
}

__forceinline__ __host__ __device__ int get_padded_len(int non_padded, float k = 4.) {
    return static_cast<int>(ceilf(static_cast<float>(non_padded) / k));
}

inline constexpr float PI = 3.14159265358979f;
inline constexpr float PI_2 = PI / 2.;