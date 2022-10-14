use nannou::prelude::*;
use nannou_egui::Egui;
use super::world_objs::ObjInfo;

use crate::utils::map_io;
use crate::utils::color::EditorColor;
use crate::utils::ffi_helper::Vec2_cpp;
use crate::utils::utils::{initialize_cpp_end, ModelBasics};
use crate::utils::structs::{WindowCtrl, WindowTransform, PlotConfig, KeyStatus};

pub struct RayTracingCtrl {
    pub flattened_pts: Vec<Vec2_cpp>,
    pub nexts: Vec<i8>,
    pub objects: Vec<ObjInfo>,
    pub obj_inds: Vec<i16>
}

impl RayTracingCtrl {
    pub fn new(f_pts: Vec<Vec2_cpp>, nexts: Vec<i8>, objs: Vec<ObjInfo>, inds: Vec<i16>) -> Self {
        RayTracingCtrl {
            flattened_pts: f_pts, nexts: nexts, objects: objs, obj_inds: inds
        }
    }
}

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub rt_ctrl: RayTracingCtrl,
    pub pose: Point3,
    pub velo: Point3,
    pub pid: Point3,
    pub velo_max: Point2,

    pub initialized: bool,

    pub color: EditorColor,
    pub egui: Egui,
    pub inside_gui: bool,
    pub key_stat: KeyStatus
}

impl ModelBasics for Model {}

impl Model {
    pub fn new(
        app: &App, window_id: WindowId, config: map_io::Config, meshes: map_io::Meshes) 
    -> Self {
        let mut flat_pts: Vec<Vec2_cpp> = Vec::new();
        let mut next_ids: Vec<i8> = Vec::new();
        // TODO: Load from json config file (Object information)
        initialize_cpp_end(&meshes, &mut flat_pts, &mut next_ids);
        Model {
            map_points: meshes, 
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(
                window_id, window_id, config.screen.width as f32, config.screen.height as f32, 
                config.screen.sub_width as f32, config.screen.sub_height as f32, exit
            ),
            wtrans: WindowTransform::new(),
            rt_ctrl: RayTracingCtrl::new(),
            pose: pt3(0., 0., 0.),
            velo: pt3(0., 0., 0.),
            pid: pt3(config.robot.pid_kp, config.robot.pid_ki, config.robot.pid_kd),
            velo_max: pt2(config.robot.t_vel, config.robot.r_vel),
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