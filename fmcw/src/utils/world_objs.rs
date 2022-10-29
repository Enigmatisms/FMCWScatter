use nannou::prelude::*;
use super::ffi_helper::Vec2_cpp;
use super::map_io::ObjInfoJson;

#[repr(C)]
pub struct AABB {
    pub tr: Vec2_cpp,        // top right point  (max x, max y)
    pub bl: Vec2_cpp,        // bottom left point (min x, min y)
}

impl AABB {
    pub fn new(tr: Point2, bl: Point2) -> Self {
        AABB{ tr: Vec2_cpp::new(tr.x, tr.y), bl: Vec2_cpp::new(bl.x, bl.y)}
    }

    pub fn default() -> Self {
        AABB{ tr: Vec2_cpp::new(0., 0.), bl: Vec2_cpp::new(0., 0.)}
    }
}

#[repr(C)]
pub struct ObjInfo {
    pub _type: libc::c_uchar,
    reserved: [libc::c_uchar; 3],
    ref_index: libc::c_float,
    rdist: libc::c_float,
    r_gain: libc::c_float,

    u_a: libc::c_float,
    u_s: libc::c_float,
    extinct: libc::c_float,
    p_c: libc::c_float,
    aabb: AABB,
}

impl ObjInfo {
    pub fn new(
        _type: libc::c_uchar, ri: libc::c_float, rd: libc::c_float, rg: libc::c_float, 
        ua: libc::c_float, us: libc::c_float, pc: libc::c_float, aabb: AABB
    ) -> Self {
        ObjInfo {
            _type: _type, reserved: [0x00; 3], ref_index: ri, rdist: rd, r_gain: rg,
            u_a: ua, u_s: us, extinct: ua + us, p_c: pc, aabb: aabb
        }
    }

    pub fn from_raw(obj: ObjInfoJson, aabb: AABB) -> Self {
        ObjInfo {
            _type: obj._type, reserved: [0x00; 3], ref_index: obj.ref_index, rdist: obj.rdist, r_gain: obj.r_gain,
            u_a: obj.u_a, u_s: obj.u_s, extinct: obj.u_a + obj.u_s,  p_c: obj.p_c, aabb: aabb 
        }
    }

    pub fn default() -> Self {
        ObjInfo {
            _type: 1, reserved: [0x00; 3], ref_index: 1., rdist: 0., r_gain: 1.0,
            u_a: 0., u_s: 0., extinct: 0.,  p_c: 0., aabb: AABB::default()
        }
    }
}

#[repr(C)]
pub struct World {
    pub scale: libc::c_float,
    pub prop: ObjInfo
}

impl World {
    pub fn from_raw(scale: f32, obj: ObjInfoJson) -> Self {
        World {
            scale: scale, prop: ObjInfo::from_raw(obj, AABB::default())
        }
    }
}