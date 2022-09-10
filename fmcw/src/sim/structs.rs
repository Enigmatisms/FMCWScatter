use nannou::prelude::*;
use super::fmcw_helper;

pub struct WindowCtrl {
    pub window_id: WindowId,
    pub win_w: f32,
    pub win_h: f32,
    pub gui_visible: bool,
    pub exit_func: fn(app: &App)
}

pub struct WindowTransform {
    pub t: Point2,
    pub t_start: Point2,
    pub rot: f32,
    pub rot_start: f32,
    pub t_set: bool,
    pub r_set: bool,
    pub scale: f32
}

pub struct ChirpRelated {
    pub flattened_pts: Vec<fmcw_helper::Vec2_cpp>,
    pub nexts: Vec<i8>,
    pub spect: Vec<libc::c_float>,
    pub gt_r: f32,
    pub pred_r: f32,
    pub pred_v: f32,
    pub map_resolution: f32,
    pub max_len: usize
}

pub struct PlotConfig {
    pub draw_grid: bool,
    pub grid_step: f32,
    pub grid_alpha: f32
}

pub struct KeyStatus {
    pub ctrl_pressed: bool,
}

impl WindowCtrl {
    pub fn new(win_id: WindowId, win_w: f32, win_h: f32, exit_f: fn(app: &App)) -> WindowCtrl {
        WindowCtrl {window_id: win_id, win_w: win_w, win_h: win_h, gui_visible: true, exit_func: exit_f}
    }

    pub fn switch_gui_visibility(&mut self) {
        self.gui_visible = !self.gui_visible;
    }
}

impl WindowTransform {
    pub fn new() -> WindowTransform{
        WindowTransform {
            t: pt2(0.0, 0.0), t_start: pt2(0.0, 0.0),
            rot: 0., rot_start: 0., t_set: true, r_set: true, scale: 1.0,
        }
    }
    
    #[inline(always)]
    pub fn clear_offset(&mut self) {
        self.rot = 0.;
        self.t = pt2(0., 0.);
    }
}

impl PlotConfig {
    pub fn new() -> Self {
        PlotConfig {
            draw_grid: false, grid_step: 100.0, grid_alpha: 0.01
        }
    }
}

impl ChirpRelated {
    pub fn new(f_pts: Vec<fmcw_helper::Vec2_cpp>, nexts: Vec<i8>) -> Self {
        ChirpRelated {
            flattened_pts: f_pts, nexts: nexts, spect: Vec::new(), gt_r: 0., pred_r: 0., pred_v: 0., map_resolution: 0.02, max_len: 4096
        }
    }
}
