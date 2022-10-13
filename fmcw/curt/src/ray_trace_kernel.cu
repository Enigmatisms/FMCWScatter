#include <cstdio>
#include <cmath>
#include "ray_trace_kernel.cuh"
#include "cuda_err_check.cuh"

// TODO: we can use dynamic allocation, but I am lazy
__device__ Vec2 all_points[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
__device__ Vec2 all_normal[MAX_PNUM];     // 1024 * 2 * 4 = 8192 bytes used
__device__ ObjInfo objects[MAX_PNUM >> 2];     // 256 * 4 * 4 = 4096 bytes used (maximum allowed object number 255)
__device__ short obj_inds[MAX_PNUM];      // line segs -> obj (LUT) (material and media & AABB）(2048 bytes used)
__device__ char next_ids[MAX_PNUM];       // 1024 bytes used

void static_scene_update(
    const Vec2* const meshes, const ObjInfo* const host_objs, const short* const host_inds, 
    const char* const host_nexts, size_t line_seg_num, size_t obj_num
) {
    CUDA_CHECK_RETURN(cudaMemcpy(all_points, meshes, sizeof(Vec2) * line_seg_num, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(objects, host_objs, sizeof(ObjInfo) * obj_num, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(obj_inds, host_inds, sizeof(short) * line_seg_num, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(next_ids, host_nexts, sizeof(char) * line_seg_num, cudaMemcpyHostToDevice));
    // TODO: Logical check needed
    calculate_normal<<<4, get_padded_len(line_seg_num)>>>(static_cast<int>(line_seg_num));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

// block 4, thread: ceil(total_num / 4)
__global__ void calculate_normal(int line_seg_num) {
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid < line_seg_num) {
        const int next_id = next_ids[pid];
        const int next_neg = (next_id < 0);
        const Vec2& p1 = all_points[pid];
        const Vec2& p2 = all_points[next_id * next_neg + (1 - next_neg)];
        const Vec2 dir_vec(p1.y - p2.y, p2.x - p1.x);             // perpendicular of (p2 - p1)
        all_normal[pid] = dir_vec * (1. / dir_vec.norm());  // normalized, since I didn't implement operator/
    }
    __syncthreads();
}

__forceinline__ __device__ void range_min(const float* const input, int start, int end, float& out, short& out_aux, const short* const aux = nullptr) {
    float min_depth = 9e5f;
    short min_mesh_ind = NULL_HIT;
    const bool aux_null = (aux == nullptr);
    for (int i = start; i < end; i++) {
        float local_depth = input[i];
        if (local_depth < min_depth) {
            min_depth = local_depth;
            min_mesh_ind = i * aux_null + (1 - aux_null) * aux[i];
        }
    }
    out = min_depth; 
    out_aux = min_mesh_ind; 
} 

/** 
 * @brief calculate whether a line intersects aabb 
 * input: id of an object, ray origin and ray direction
 * detailed derivation of aabb intersection should be deduced
 */
__device__ bool aabb_intersected(const Vec2& const ray_o, float dx, float dy, int obj_id) {
    const AABB& aabb = objects[obj_id].aabb;
    bool result = false, dx_valid = fabs(dx) > 2e-5f, dy_valid = fabs(dy) > 2e-5f;
    bool x_singular_valid = (ray_o.x < aabb.tr.x && ray_o.x > aabb.bl.x);     // valid condition when dx is too small
    bool y_singular_valid = (ray_o.y < aabb.tr.y && ray_o.y > aabb.bl.y);     // valid condition when dy is too small
    if (dx_valid && dy_valid) {        // there might be warp divergence (hard to resolve)
        const bool dx_pos = dx > 0, dy_pos = dy > 0;
        Vec2 act_tr(aabb.tr.x + aabb.bl.x, aabb.tr.y + aabb.bl.y);
        Vec2 act_bl(aabb.bl.x * dx_pos + aabb.tr.x * (1 - dx_pos), aabb.bl.y * dy_pos + aabb.tr.y * dy_pos);
        act_tr -= act_bl;

        const float enter_xt = (act_bl.x - ray_o.x) / dx, enter_yt = (act_bl.y - ray_o.y) / dy;
        const float exit_xt = (act_tr.x - ray_o.x) / dx, exit_yt = (act_tr.y - ray_o.y) / dy;
        const float enter_t = fmax(enter_xt, enter_yt), exit_t = fmin(exit_xt, exit_yt);
        // the following condition: or (maybe inside aabb) - and (must be outside and back-culled)
        bool back_cull = ((enter_xt < 0.f && exit_xt < 0.f) || (enter_yt < 0.f && exit_yt < 0.f));       // either pair of (in, out) being both neg, culled.
        result = (!back_cull) & (enter_t < exit_t);     // not back-culled and enter_t is smaller
    }
    result |= ((!dx_valid) & x_singular_valid);         // if x is not valid (false, ! -> true), then use x_singular_valid
    result |= ((!dy_valid) & y_singular_valid);         // if y is not valid (false, ! -> true), then use x_singular_valid
    return result;
}

// v_perp is the 90 deg rotated directional vector of current ray
__forceinline__ __device__ float ray_intersect(const Vec2& pos, const Vec2& v_perp, const Vec2& p1, const Vec2& p2) {
    const Vec2 s2e = p2 - p1;
    const Vec2 obs_v = pos - p1;
    const float D = v_perp.dot(s2e);
    float result = 1e6;
    if (fabs(D) > 5e-5) {
        float alpha = v_perp.dot(obs_v) / D;
        if (alpha < 1. && alpha > 0.) {
            float tmp = (-s2e.y * obs_v.x + s2e.x * obs_v.y) / D;
            float flag = float(tmp > 0.);
            result = tmp * flag + 1e6 * (1. - flag);
        }
    }
    return result;
}

/**
 * input : point origin (Vec2 array), ray angles: float array
 * output1 : minimum depth (float (single value, since each block represent one single ray) should be converted back to int)
 * output2 : the obj_index (of the nearest hit line seg or NULL_HIT flag)
 * @param depth is GLOBAL memory float array (for fast data copying)
 * @note this is a global function, not where the host could call. Also, AABB will not make this algo faster (only lower the power consumption)
 */
__global__ void ray_trace_cuda_kernel(
    const Vec2* const origins, const Vec2* const ray_dir, 
    RayInfo* const ray_info, short* const inds, int block_offset, int mesh_num, int aabb_num
) {
    // mem consumption: (1) mesh_num * 4 bytes (for all ranges) (2) 8 * float (min ranges, stratified) -> 32 bytes 
    // (3) 8 * short -> 4 floats -> 16 bytes (4) AABB valid bools (1 bytes * num AABB) padding
    extern __shared__ float shared_banks[];      
    bool* hit_flags = (bool*) &shared_banks[mesh_num + 12];
    const int ray_id = blockIdx.x + gridDim.x * block_offset;

    const int mesh_id = threadIdx.x + threadIdx.y * blockDim.x;
    const Vec2& ray_o = origins[ray_id];
    const Vec2& ray_d = ray_dir[ray_id];
    const float dx = ray_d.x, dy = ray_d.y;
    if (mesh_id < aabb_num) {       // first (aabb_num) threads should process AABB calculation, others remains idle
        // Bank conflict unresolvable (haven't found a very effective way)
        hit_flags[mesh_id] = aabb_intersected(ray_o, dx, dy, mesh_id);
    }
    __syncthreads();
    if (mesh_id < mesh_num) {       // for the sake of mesh (line segment division), there might be more threads than needed
        short aabb_index = obj_inds[mesh_id];
        if (hit_flags[aabb_index] == true) {        // there will be no warp divergence, since the 'else' side is NOP
            const Vec2 v_perp(-dy, dx);
            const int next_id = next_ids[mesh_id];
            const int next_id_neg = int(next_ids[mesh_id] < 0);
            const int next_pt_id = next_id * next_id_neg + 1 - next_id_neg;
            shared_banks[mesh_id] = ray_intersect(ray_o, v_perp, all_points[mesh_id], all_points[mesh_id + next_pt_id]);
            // as we use mesh indices, we don't have to store them in shared memory 
        }
    }
    __syncthreads();
    float* local_min_depths = &shared_banks[mesh_num];
    short* local_obj_inds = (short*) &shared_banks[mesh_num + 8];
    if (threadIdx.x == 0) {             // 8-thread parallel
        int max_bound = min(mesh_num, blockDim.x * (threadIdx.y + 1));
        range_min(shared_banks, blockDim.x * threadIdx.y, max_bound, local_min_depths[mesh_id], local_obj_inds[mesh_id]);
    }
    __syncthreads();
    if (mesh_id == 0) {             // only one thread attend to the final output
        RayInfo& ray = ray_info[ray_id];
        range_min(local_min_depths, 0, blockDim.y, ray.range_bound, inds[ray_id], local_obj_inds);
        ray.acc_range += ray.range_bound;
    }
    __syncthreads();
}
