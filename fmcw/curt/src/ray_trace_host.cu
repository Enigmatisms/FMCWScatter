#include "ray_trace_host.cuh"
#include "cuda_err_check.cuh"

int get_padded_len(int non_padded, float k = 4.) {
    return static_cast<int>(ceilf(static_cast<float>(non_padded) / k));
}

PathTracer::PathTracer(size_t ray_num):
    ray_os(nullptr, get_deletor<Vec2>()), intersects(nullptr, get_deletor<Vec2>()),
    ray_d(nullptr, get_deletor<Vec2>()), ranges(nullptr, get_deletor<float>()),
    mesh_inds(nullptr, get_deletor<short>()), ray_num(ray_num)
{
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_os, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_intersects, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_info, ray_num * sizeof(RayInfo)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ray_d, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_ranges, ray_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_mesh_inds, ray_num * sizeof(short)));
    // pinned memory allocation, let's do some math: suppose there are 2048 rays
    // 2048 * 4 * (2 + 2 + 1 + 1 + 1) = 56 KB (only 56KB pinned memory is allocated!)
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &ray_os_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &intersect_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &ray_d_ptr, ray_num * sizeof(Vec2)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &ranges_ptr, ray_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &mesh_inds_ptr, ray_num * sizeof(short)));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    ray_os.reset(ray_os_ptr);
    intersects.reset(intersect_ptr);
    ray_d.reset(ray_d_ptr);
    ranges.reset(ranges_ptr);
    mesh_inds.reset(mesh_inds_ptr);
}

PathTracer::~PathTracer() {
    CUDA_CHECK_RETURN(cudaFree(cu_ray_os));
    CUDA_CHECK_RETURN(cudaFree(cu_intersects));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_d));
    CUDA_CHECK_RETURN(cudaFree(cu_ranges));
    CUDA_CHECK_RETURN(cudaFree(cu_mesh_inds));
    CUDA_CHECK_RETURN(cudaFree(cu_ray_info));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void PathTracer::next_intersections(int mesh_num, int aabb_num) {
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_os, ray_os_ptr, ray_num * sizeof(Vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_ray_d, ray_d_ptr, ray_num * sizeof(Vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaStream_t streams[8];
    for (short i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const int block_per_stream = ray_num >> 3;
    size_t shared_mem_size = (ray_num << 2) + 48 + get_padded_len(aabb_num);
    size_t threads_along_x = get_padded_len(mesh_num);
    for (int i = 0; i < 8; i++) {
        ray_trace_cuda_kernel<<<block_per_stream, dim3(threads_along_x, 8), shared_mem_size, streams[i]>>>(
            cu_ray_os, cu_ray_d, cu_ranges, cu_mesh_inds, i, mesh_num, aabb_num
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    // TODO: is this kind of implementation good?
    CUDA_CHECK_RETURN(cudaMemcpyAsync(intersect_ptr, cu_intersects, ray_num * sizeof(Vec2), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(ranges_ptr, cu_ranges, ray_num * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(mesh_inds_ptr, cu_mesh_inds, ray_num * sizeof(short), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
