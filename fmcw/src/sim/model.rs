use nannou::prelude::*;
use nannou_egui::Egui;
use super::fmcw_helper;

use super::map_io;
use super::structs::*;
use super::color::EditorColor;

const BOUNDARIES: [(f32, f32); 4] = [(-600.0, -450.), (600., -450.), (600., 450.), (-600., 450.)];
const BOUNDARY_IDS: [i8; 4] = [3, 0, 0, -3];

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub chirp: ChirpRelated,
    pub fmcw_p: map_io::Chirp_param,
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
        app: &App, window_id: WindowId, sub_id: WindowId, config: map_io::Config, meshes: map_io::Meshes) 
    -> Model {
        let mut flat_pts: Vec<fmcw_helper::Vec2_cpp> = Vec::new();
        let mut next_ids: Vec<i8> = Vec::new();
        Model::initialize_cpp_end(&meshes, &mut flat_pts, &mut next_ids);
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

    pub fn initialize_cpp_end(new_pts: &map_io::Meshes, pts: &mut Vec<fmcw_helper::Vec2_cpp>, nexts: &mut Vec<i8>) {
        for mesh in new_pts.iter() {
            for pt in mesh.iter() {
                pts.push(fmcw_helper::Vec2_cpp{x: pt.x, y: pt.y});
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
            pts.push(fmcw_helper::Vec2_cpp{x: x, y: y});
            nexts.push(BOUNDARY_IDS[i]);
        }
    } 

    pub fn reload_config(
        config: &map_io::Config, win_w: &mut f32, 
        win_h: &mut f32, pid: &mut Point3, velo_max: &mut Point2
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