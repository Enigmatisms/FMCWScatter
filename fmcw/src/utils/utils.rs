use nannou::glam::{Mat2, Vec2, Vec3};
use super::map_io;
use super::ffi_helper::Vec2_cpp;

const BOUNDARIES: [(f32, f32); 4] = [(-600.0, -450.), (600., -450.), (600., 450.), (-600., 450.)];
const BOUNDARY_IDS: [i8; 4] = [3, 0, 0, -3];

#[inline(always)]
pub fn good_angle(angle: f32) -> f32 {
    if angle > std::f32::consts::PI {
        return angle - std::f32::consts::PI * 2.;
    } else if angle < -std::f32::consts::PI {
        return angle + std::f32::consts::PI * 2.;
    }
    angle
}

#[inline(always)]
pub fn get_rotation(angle: &f32) -> Mat2 {
    let cosa = angle.cos();
    let sina = angle.sin();
    Mat2::from_cols_array(&[cosa, sina, -sina, cosa])
}

pub fn initialize_cpp_end(new_pts: &map_io::Meshes, pts: &mut Vec<Vec2_cpp>, nexts: &mut Vec<i8>) {
    for mesh in new_pts.iter() {
        for pt in mesh.iter() {
            pts.push(Vec2_cpp{x: pt.x, y: pt.y});
        }
        let length = mesh.len();
        let offset: i8 = (length as i8) - 1;
        let mut ids: Vec<i8> = vec![0; length];
        ids[0] = offset;
        ids[length - 1] = -offset;
        nexts.extend(ids.into_iter());
    }
    for i in 0..4 {                                                 // add boundaries
        let (x, y) = BOUNDARIES[i];
        pts.push(Vec2_cpp{x: x, y: y});
        nexts.push(BOUNDARY_IDS[i]);
    }
} 

pub trait ModelBasics {
    fn reload_config(
        config: &map_io::Config, win_w: &mut f32, 
        win_h: &mut f32, pid: &mut Vec3, velo_max: &mut Vec2
    ) {
        pid.x = config.robot.pid_kp;
        pid.y = config.robot.pid_ki;
        pid.z = config.robot.pid_kd;
        velo_max.x = config.robot.t_vel;
        velo_max.y = config.robot.r_vel;
        *win_w = config.screen.width as f32; 
        *win_h = config.screen.height as f32; 
    }
}

