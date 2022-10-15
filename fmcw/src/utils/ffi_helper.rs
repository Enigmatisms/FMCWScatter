extern crate libc;

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

impl Vec2_cpp {
    pub fn new(x: libc::c_float, y: libc::c_float) -> Self {
        Vec2_cpp { x: x, y: y }
    }

    pub fn default() -> Self {
        Vec2_cpp { x: 0., y: 0. }
    }
}

impl Vec3_cpp {
    pub fn new(x: libc::c_float, y: libc::c_float, z: libc::c_float) -> Self {
        Vec3_cpp { x: x, y: y, z: z}
    }
}