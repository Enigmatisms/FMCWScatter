use nannou::prelude::*;

pub struct WindowCtrl {
    pub window_id: WindowId,
    pub sub_win_id: WindowId,
    pub win_w: f32,
    pub win_h: f32,
    pub sub_w: f32,
    pub sub_h: f32,
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

pub struct PlotConfig {
    pub draw_grid: bool,
    pub grid_step: f32,
    pub grid_alpha: f32,
    pub reserve: f32
}

pub struct KeyStatus {
    pub ctrl_pressed: bool,
}

impl WindowCtrl {
    pub fn new(win_id: WindowId, sub_id: WindowId, win_w: f32, win_h: f32, sub_w: f32, sub_h: f32, exit_f: fn(app: &App)) -> WindowCtrl {
        WindowCtrl {
            window_id: win_id, sub_win_id: sub_id, win_w: win_w, win_h: win_h, 
            sub_w: sub_w, sub_h: sub_h, gui_visible: true, exit_func: exit_f
        }
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
            draw_grid: false, grid_step: 100.0, grid_alpha: 0.01, reserve: 100.,
        }
    }
}