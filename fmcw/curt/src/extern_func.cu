#include <algorithm>
#include "../include/ray_trace_host.cuh"
#include "../include/sampler_kernel.cuh"
#include "../include/cuda_err_check.cuh"

PathTracer path_tracer;
extern "C" {
    void setup_path_tracer(int ray_num) {
        path_tracer.setup(static_cast<size_t>(ray_num));
        printf("Ray num: %lu, memory allocated\n", ray_num);
    }

    // Get intersection of the light rays and update the ray directions and ray origins
    void get_intersections_update(Vec2* const intersections, int mesh_num, int aabb_num) {
        path_tracer.next_intersections(mesh_num, aabb_num);
        std::copy_n(path_tracer.intersects.get(), path_tracer.get_ray_num(), intersections);
        path_tracer.sample_outgoing_rays();
    }

    void first_ray_intersections(Vec2* const intersections, const Vec3& pose, int mesh_num, int aabb_num) {
        path_tracer.first_intersection(pose.x, pose.y, pose.z, mesh_num, aabb_num);
        Vec2* tmp = path_tracer.intersects.get();
        // printf("Pose: %f, %f\n", pose.x, pose.y, pose.z);
        // for (size_t i = 0; i < path_tracer.get_ray_num(); i++) {
        //     printf("(%.4f, %.4f), ", tmp[i].x, tmp[i].y);
        // }
        // printf("\n");
        std::copy_n(path_tracer.intersects.get(), path_tracer.get_ray_num(), intersections);
        path_tracer.sample_outgoing_rays(false);    // do not update ray_os (from intersection)
    }

    // TODO: To be substituted by texture memory in the future
    void static_scene_update(
        const Vec2* const meshes, const ObjInfo* const host_objs, const short* const host_inds, 
        const char* const host_nexts, size_t line_seg_num, size_t obj_num
    ) {
        printf("Seg num: %lu\n", line_seg_num);
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(all_points, meshes, sizeof(Vec2) * line_seg_num, 0, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(objects, host_objs, sizeof(ObjInfo) * obj_num, 0, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(obj_inds, host_inds, sizeof(short) * line_seg_num, 0, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(next_ids, host_nexts, sizeof(char) * line_seg_num, 0, cudaMemcpyHostToDevice));
        // TODO: Logical check needed
        calculate_normal<<<4, get_padded_len(line_seg_num)>>>(static_cast<int>(line_seg_num));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printf("Static scene setup, seg num = %lu, obj_num = %lu\n", line_seg_num, obj_num);
    }
}
