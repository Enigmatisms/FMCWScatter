#include "../include/ray_trace_host.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/cuda_err_check.cuh"

#define BLOCK_PER_STREAM 8

PathTracer::PathTracer():
    ray_os(nullptr, get_deletor<Vec2>()), intersects(nullptr, get_deletor<Vec2>()),
    ray_d(nullptr, get_deletor<Vec2>()), ray_num(0) {}

// CPU end: the last commit when range_ptr and mesh_inds are available is c4815846c
PathTracer::PathTracer(size_t ray_num):
    ray_os(nullptr, get_deletor<Vec2>()), intersects(nullptr, get_deletor<Vec2>()),
    ray_d(nullptr, get_deletor<Vec2>()), ray_num(ray_num)
{
    setup(ray_num);
}

void PathTracer::setup(size_t ray_num) {
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_os, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_d, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_intersects, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_info, ray_num * sizeof(RayInfo)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_mesh_inds, ray_num * sizeof(short)));
    // pinned memory allocation, let's do some math: suppose there are 2048 rays
    // 2048 * 4 * (2 + 2 + 1 + 1 + 1) = 56 KB (only 56KB pinned memory is allocated!)
    // ray_os is definitely needed, since we wish to change the position of the FMCW scanner in the future
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &ray_os_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &ray_d_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &intersect_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    ray_os.reset(ray_os_ptr);
    ray_d.reset(ray_d_ptr);
    intersects.reset(intersect_ptr);
    this->ray_num = ray_num;
}

PathTracer::~PathTracer() {
    CUDA_CHECK_RETURN(cudaFree(cu_ray_os));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_d));
    CUDA_CHECK_RETURN(cudaFree(cu_intersects));
    CUDA_CHECK_RETURN(cudaFree(cu_mesh_inds));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));

    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

// How to call the function from the host side? We should be able to access any data we need
void PathTracer::next_intersections(int mesh_num, int aabb_num) {
    cudaStream_t streams[8];
    for (short i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const int cascade_num = std::max(1, int(ray_num) / BLOCK_PER_STREAM);
    size_t shared_mem_size = (ray_num << 2) + 48 + pad_bytes(aabb_num);
    size_t threads_along_x = get_padded_len(mesh_num, 8.);
    for (int i = 0; i < cascade_num; i++) {
        ray_trace_cuda_kernel<<<BLOCK_PER_STREAM, dim3(threads_along_x, 8), shared_mem_size, streams[i % 8]>>>(
            cu_ray_os, cu_ray_d, cu_intersects, cu_ray_info, cu_mesh_inds, i, mesh_num, aabb_num
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void PathTracer::first_intersection(float origin_x, float origin_y, float dir_a, int mesh_num, int aabb_num) {
    // Things which are not going to be overwritten should be reset
    CUDA_CHECK_RETURN(cudaMemset(cu_mesh_inds, 0xff, ray_num * sizeof(short)));
    first_intersect_reset<<< 8, max(ray_num >> 3, 1lu)>>>(cu_ray_info);

    size_t shared_mem_size = (ray_num << 2) + 48 + pad_bytes(aabb_num);
    size_t threads_along_x = get_padded_len(mesh_num, 8.);
    const Vec2 ray_o(origin_x, origin_y);
    const Vec2 ray_d(cosf(dir_a), sinf(dir_a));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_os, &ray_o, sizeof(Vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_d, &ray_d, sizeof(Vec2), cudaMemcpyHostToDevice));
    ray_trace_cuda_kernel<<<1, dim3(threads_along_x, 8), shared_mem_size>>>(
        cu_ray_os, cu_ray_d, cu_intersects, cu_ray_info, cu_mesh_inds, 0, mesh_num, aabb_num
    );
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    copy_ray_poses_kernel<<<8, max(ray_num >> 3, 1lu)>>>(cu_intersects, cu_mesh_inds, cu_ray_info, cu_ray_d);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void PathTracer::sample_outgoing_rays() {
    static size_t random_offset = 0;
    // within this function, there is nothing to be fetched multiple times, therefore shared memory is not needed.
    // update the ray direction, in order to get next intersection
    general_interact_kernel<<< 8, max(ray_num >> 3, 1lu) >>>(cu_mesh_inds, cu_ray_info, cu_ray_d, random_offset);
    // update the intersections (ray origin updates from original starting point to intersection points) 
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_os, cu_intersects, ray_num * sizeof(Vec2), cudaMemcpyDeviceToDevice));  // assume this copy operation won't emit exception
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    random_offset += 1;
}
