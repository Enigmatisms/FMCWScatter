#pragma once
#include <vector>
#include <memory>
#include "cuda_err_check.cuh"
#include "ray_trace_kernel.cuh"

template<typename T>
auto get_deletor() {
    return [](T* ptr) {
        CUDA_CHECK_RETURN(cudaFreeHost(ptr));
    };
}

template<typename T>
using host_ptr = std::unique_ptr<T, decltype(get_deletor<T>())>;



class PathTracer {
public:
    PathTracer(size_t ray_num);
    ~PathTracer();
public:
    void static_scene_allocation(size_t ray_num);
    
    /**
     * @brief calculate next intersections given ray origin and ray direction
     * @note I don't want to use pointers since Rust has nothing to do with this function
     * therefore PathTracer can not be linked to Rust program
     */
    void next_intersections(int mesh_num, int aabb_num);

    // TODO: the current ray_d used in GPU is Vec2 (should be converted from angle)
private:
    
public:
    Vec2 *cu_ray_os, *cu_intersects, *cu_ray_d;
    RayInfo *cu_ray_info;
    float *cu_ranges;
    short* cu_mesh_inds;

    // All of these smart ptrs manage the actual mem
    // If there is any need for providing the data to the outside of the class, use these ptrs
    host_ptr<Vec2> ray_os;
    host_ptr<Vec2> intersects;
    host_ptr<Vec2> ray_d;
    host_ptr<float> ranges;
    host_ptr<short> mesh_inds;
private:
    // I don't want the naked ptr to be leaked out there
    Vec2 *ray_os_ptr, *intersect_ptr, *ray_d_ptr;
    float *ranges_ptr;
    short *mesh_inds_ptr;
    size_t ray_num;
};