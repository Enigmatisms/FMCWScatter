use nannou::prelude::*;
use nannou_egui::Egui;

use array2d::Array2D;
use super::fmcw_helper;

use super::map_io;
use super::structs::*;
use super::color::EditorColor;
use std::f32::consts::PI;
use std::path::PathBuf;

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub grid_specs: (f32, f32, f32, f32),
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub pose: Point3,
    pub velo: Point3,
    pub pid: Point3,
    pub velo_max: Point2,

    pub range_pred: libc::c_float,
    pub vel_pred: libc::c_float,
    pub initialized: bool,

    pub color: EditorColor,
    pub egui: Egui,
    pub inside_gui: bool,
    pub key_stat: KeyStatus
}

impl Model {
    pub fn new(
        app: &App, window_id:  WindowId, config: &map_io::Config, meshes: map_io::Meshes) 
    -> Model {
        Model {
            map_points: meshes, 
            grid_specs: grid_specs,
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(window_id, config.screen.width as f32, config.screen.height as f32, exit),
            wtrans: WindowTransform::new(),
            pose: pt3(0., 0., 0.),
            velo: pt3(0., 0., 0.),
            pid: pt3(config.robot.pid_kp, config.robot.pid_ki, config.robot.pid_kd),
            velo_max: pt2(config.robot.t_vel, config.robot.r_vel),
            range_pred: 0.0,
            vel_pred: 0.0,
            initialized: false,
            color: EditorColor::new(),
            egui: Egui::from_window(&app.window(window_id).unwrap()),
            key_stat: KeyStatus{ctrl_pressed: false},
            inside_gui: false,
        }
    }

    pub fn reload_config(
        config: &map_io::Config, win_w: &mut f32, win_h: &mut f32, pid: &mut Point3,
        lidar_color: &mut [f32; 4], velo_max: &mut Point2, lidar_noise: &mut f32
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

fn exit(app: &App) {
    app.quit();
}