extern crate libc;
use super::map_io::Chirp_param;

#[repr(C)]
pub struct Vec3_cpp {
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub z: libc::c_float
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec2_cpp {
    pub x: libc::c_float,
    pub y: libc::c_float
}

#[link(name = "fmcw_helper", kind = "static")]
extern {
    
    // void laserRangeFinder(const Vec3& pose, const Vec2* const pts, const char* const ids, int max_num, float& min_range) {
    pub fn laserRangeFinder(pose: &Vec3_cpp, pts: *const Vec2_cpp, ids: *const libc::c_char, max_num: libc::c_int, min_range: &mut libc::c_float);
    
    // void simulateOnce(const ChirpParams& p, float* spect, float& range, float& vel, int& sp_size, float gt_r, float gt_v, float cutoff) {
    pub fn simulateOnce(p: &Chirp_param, spect: *mut libc::c_float, range: &mut libc::c_float,
        vel: &mut libc::c_float, sp_size: &mut libc::c_int, gt_r: libc::c_float, gt_v: libc::c_float, cutoff: libc::c_float);
}