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

    __host__ __device__ Vec2 operator-=(const Vec2& p) {
        x -= p.x;
        y -= p.y;
    }

    __host__ __device__ Vec2 operator+=(const Vec2& p) {
        x += p.x;
        y += p.y;
    }

    __host__ __device__ Vec2 operator*=(float v) {
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

struct RayInfo {
    bool is_in_media;

    // range bound is the maximum range of a ray (due to occluders), if in a scattering media
    // we will sample by mean free path (which is often smaller than range bound)
    float range_bound;    
    float acc_range;  

    __host__ __device__ constexpr RayInfo(): is_in_media(false), range_bound(1e4), acc_range(0.) {}
    __host__ __device__ constexpr RayInfo(short prev_id): is_in_media(false), range_bound(1e4), acc_range(0.) {}
};

// Axis-aligned bounding box for objects
struct AABB {
    Vec2 tr;        // top right point  (max x, max y)
    Vec2 bl;        // bottom left point (min x, min y)

    __host__ __device__ AABB() {}
    __host__ __device__ AABB(float bx, float by, float ux, float uy): tr(ux, uy), bl(bx, by) {}
};

using uint8 = unsigned char;

// Media and Material for objects
enum Material: uint8 {
    DIFFUSE = 0,
    GLOSSY = 1,
    SPECULAR = 2,
    REFRACTIVE = 3,
    SCAT_ISO = 4,
    SCAT_HG = 5,
    SCAT_RAYLEIGH = 6,
};

// object-managing struct
struct ObjInfo {
    Material type;              // size is uint8
    uint8 reserved[3];
    float ref_index;
    float rdist;                // reflection distribution (negative: diffusive, 0: specular, (0, 0.5] glossy)
    float r_gain;               // relfection gain (in [0, 2])

    float u_a;                  // when material is not semi-transparent, u_a is the absorption coeff upon reflection
    float u_s;                  // scattering coeff, when material is not semi-transparent, it is BRDF reflection concentration [0 means diffusive, >0.5 means specular]
    float extinct;              // extinction = u_a + u_s
    float p_c;                  // when material is not semi-transparent, p_c is the coefficient of phase function
    AABB aabb;

    __host__ __device__ ObjInfo() {}
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

constexpr float PI = 3.14159265358979f;
constexpr float PI_2 = PI / 2.;