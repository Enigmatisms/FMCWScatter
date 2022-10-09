#pragma once
#include <vector>
#include <memory>
#include "cuda_err_check.hpp"
#include "ray_trace_kernel.hpp"

template<typename T>
auto get_deletor() {
    return [](T* ptr) {
        CUDA_CHECK_RETURN(cudaFreeHost(ptr));
    };
}

template<typename T>
using host_ptr = std::unique_ptr<T, decltype(get_deletor<T>())>;

/**
 * @brief calculate next intersections given ray origin and ray direction
 * @note I don't want to use pointers since Rust has nothing to do with this function
 * therefore PathTracer can not be linked to Rust program
 */
void next_intersections(
    const std::vector<Vec2>& ray_os, const std::vector<float>& ray_d, 
    std::vector<Vec2>& intersects, std::vector<float>& ranges, std::vector<int>& line_inds
);

class PathTracer {
public:
    PathTracer(size_t ray_num);
    ~PathTracer();
public:
    void static_scene_allocation(size_t ray_num);

    void next_intersections(int mesh_num, int aabb_num);
private:
    
public:
    Vec2 *cu_ray_os, *cu_intersects;
    float *cu_ray_d, *cu_ranges;
    short* cu_mesh_inds;

    // All of these smart ptrs manage the actual mem
    // If there is any need for providing the data to the outside of the class, use these ptrs
    host_ptr<Vec2> ray_os;
    host_ptr<Vec2> intersects;
    host_ptr<float> ray_d;
    host_ptr<float> ranges;
    host_ptr<short> mesh_inds;
private:
    // I don't want the naked ptr to be leaked out there
    Vec2 *ray_os_ptr, *intersect_ptr;
    float *ray_d_ptr, *ranges_ptr;
    short *mesh_inds_ptr;
    size_t ray_num;
};
