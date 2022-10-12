#include "ray_trace_host.cuh"
#include "sampler_kernel.cuh"
#include "cuda_err_check.cuh"

#define BLOCK_PER_STREAM 32

// CPU end: the last commit when range_ptr and mesh_inds are available is c4815846c
PathTracer::PathTracer(size_t ray_num):
    ray_os(nullptr, get_deletor<Vec2>()), intersects(nullptr, get_deletor<Vec2>()),
    ray_d(nullptr, get_deletor<Vec2>()), ray_num(ray_num)
{
    // actually, ranges will be summed (in RayInfo)
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
}

PathTracer::~PathTracer() {
    CUDA_CHECK_RETURN(cudaFree(cu_ray_os));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_d));
    CUDA_CHECK_RETURN(cudaFree(cu_intersects));
    CUDA_CHECK_RETURN(cudaFree(cu_mesh_inds));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void PathTracer::next_intersections(bool host_update, int mesh_num, int aabb_num) {
    if (host_update == true) {          // if not, it means that we are doing path tracing (otherwise it is the first path tracing given a new pose)
        CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_os, ray_os_ptr, ray_num * sizeof(Vec2), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_d, ray_d_ptr, ray_num * sizeof(Vec2), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    
    cudaStream_t streams[8];
    for (short i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const int cascade_num = ray_num / BLOCK_PER_STREAM;
    size_t shared_mem_size = (ray_num << 2) + 48 + get_padded_len(aabb_num);
    size_t threads_along_x = get_padded_len(mesh_num);
    for (int i = 0; i < cascade_num; i++) {
        ray_trace_cuda_kernel<<<BLOCK_PER_STREAM, dim3(threads_along_x, 8), shared_mem_size, streams[i % 8]>>>(
            cu_ray_os, cu_ray_d, cu_ray_info, cu_mesh_inds, i, mesh_num, aabb_num
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaMemcpy(intersect_ptr, cu_intersects, ray_num * sizeof(Vec2), cudaMemcpyDeviceToHost));
    // TODO: Do we really need range output and mesh_inds output (to CPU end?)
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void PathTracer::sample_outgoing_rays() {
    static size_t random_offset = 0;
    // update the intersections (ray origin updates from original starting point to intersection points) 
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_os, cu_intersects, ray_num * sizeof(Vec2), cudaMemcpyDeviceToDevice));  // assume this copy operation won't emit exception
    
    // within this function, there is nothing to be fetched multiple times, therefore shared memory is not needed.
    // update the ray direction, in order to get next intersection
    non_scattering_interact_kernel<<< 8, (ray_num >> 3) >>>(cu_mesh_inds, cu_ray_info, cu_ray_d, random_offset);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    random_offset += 1;
}
