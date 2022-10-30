#pragma once
#include "cuda_utils.cuh"

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
    SCAT_RAYLEIGH = 6
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