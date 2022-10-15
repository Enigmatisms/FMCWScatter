use nannou::prelude::*;
use nannou_egui::Egui;
use super::error_plot::ErrorPlotter;

use crate::utils::map_io;
use crate::utils::structs::*;
use crate::utils::color::EditorColor;
use crate::utils::ffi_helper::Vec2_cpp;
use crate::utils::utils::{initialize_cpp_end, ModelBasics};

pub struct ChirpRelated {
    pub flattened_pts: Vec<Vec2_cpp>,
    pub nexts: Vec<i8>,
    pub spect: Vec<libc::c_float>,
    pub gt_r: f32,
    pub pred_r: f32,
    pub pred_v: f32,
    pub map_resolution: f32,
    pub max_len: usize
}

impl ChirpRelated {
    pub fn new(f_pts: Vec<Vec2_cpp>, nexts: Vec<i8>) -> Self {
        ChirpRelated {
            flattened_pts: f_pts, nexts: nexts, spect: Vec::new(), gt_r: 0., pred_r: 0., pred_v: 0., map_resolution: 0.02, max_len: 4096
        }
    }
}

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub chirp: ChirpRelated,
    pub fmcw_p: map_io::Chirp_param,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub err_plot: ErrorPlotter,
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

impl ModelBasics for Model {}

impl Model {
    pub fn new(
        app: &App, window_id: WindowId, sub_id: WindowId, config: map_io::Config, meshes: map_io::Meshes) 
    -> Model {
        let mut flat_pts: Vec<Vec2_cpp> = Vec::new();
        let mut next_ids: Vec<i8> = Vec::new();
        initialize_cpp_end(&meshes, &mut flat_pts, &mut next_ids);
        Model {
            map_points: meshes, 
            chirp: ChirpRelated::new(flat_pts, next_ids),
            fmcw_p: config.fmcw_p,
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(
                window_id, sub_id, config.screen.width as f32, config.screen.height as f32, 
                config.screen.sub_width as f32, config.screen.sub_height as f32, exit
            ),
            wtrans: WindowTransform::new(),
            err_plot: ErrorPlotter::new(6, 100, 100),
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
}

fn exit(app: &App) {
    app.quit();
}