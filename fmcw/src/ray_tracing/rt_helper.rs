use crate::utils::ffi_helper::{Vec2_cpp, Vec3_cpp};
use crate::utils::world_objs::ObjInfo;

#[link(name = "rt_helper", kind = "static")]
extern {
    pub fn setup_path_tracer(ray_num: libc::c_int);

    pub fn get_intersections_update(intersections: *mut Vec2_cpp, mesh_num: libc::c_int, aabb_num: libc::c_int);

    pub fn first_ray_intersections(intersections: *mut Vec2_cpp, pose: &Vec3_cpp, mesh_num: libc::c_int, aabb_num: libc::c_int);

    pub fn static_scene_update(
        meshes: *const Vec2_cpp, host_objs: *const ObjInfo, host_inds: *const libc::c_short,
        host_nexts: *const libc::c_char, line_seg_num: libc::c_ulong, obj_num: libc::c_ulong
    );
}