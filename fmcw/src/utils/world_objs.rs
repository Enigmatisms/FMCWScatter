use nannou::prelude::*;
use super::ffi_helper::Vec2_cpp;
use super::map_io::ObjInfoJson;

#[repr(C)]
pub struct AABB {
    pub tr: Vec2_cpp,        // top right point  (max x, max y)
    pub bl: Vec2_cpp,        // bottom left point (min x, min y)
}

impl AABB {
    pub fn new(tl: Point2, br: Point2) -> Self {
        AABB{ tr: Vec2_cpp::new(tl.x, tl.y), bl: Vec2_cpp::new(br.x, br.y)}
    }

    pub fn default() -> Self {
        AABB{ tr: Vec2_cpp::new(0., 0.), bl: Vec2_cpp::new(0., 0.)}
    }
}

#[repr(C)]
pub struct ObjInfo {
    _type: libc::c_uchar,
    reserved: [libc::c_uchar; 3],
    ref_index: libc::c_float,
    u_a: libc::c_float,
    u_s: libc::c_float,
    p_c: libc::c_float,
    f_reserved: [libc::c_float; 3],
    aabb: AABB,
}

impl ObjInfo {
    pub fn new(_type: libc::c_uchar, ri: libc::c_float, ua: libc::c_float, us: libc::c_float, pc: libc::c_float, aabb: AABB) -> Self {
        ObjInfo {
            _type: _type, reserved: [0x00; 3], ref_index: ri, 
            u_a: ua, u_s: us, p_c: pc, f_reserved: [0.; 3], aabb: aabb
        }
    }

    pub fn from_raw(obj: ObjInfoJson, aabb: AABB) -> Self {
        ObjInfo {
            _type: obj._type, reserved: [0x00; 3], ref_index: obj.ref_index,
            u_a: obj.u_a, u_s: obj.u_s, p_c: obj.p_c, f_reserved: [0.; 3], aabb: aabb 
        }
    }

    pub fn default() -> Self {
        ObjInfo {
            _type: 1, reserved: [0x00; 3], ref_index: 1., 
            u_a: 0., u_s: 0., p_c: 0., f_reserved: [0.; 3], aabb: AABB::default()
        }
    }
}