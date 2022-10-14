use nannou::prelude::*;

use super::gui;
use super::model::Model;

use crate::utils::plot;
use crate::utils::utils;
use crate::utils::map_io;
use crate::utils::ffi_helper::Vec3_cpp;
use crate::sim::ctrl;

const SPECTRUM_STR: &str = "Spectrum";
const VELOCITY_STR: &str = "Velocity: white -- ground truth, yellow -- prediction";
const ERROR_STR: &str = "Average range error (m)";

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

#[inline(always)]
fn zero_padding(vec: &mut Vec<libc::c_float>, sp_int: f32, sp_time: f32) {
    let raw_num: u32 = (sp_time / sp_int).log2().ceil() as u32;
    vec.resize(2_usize.pow(raw_num), 0.);
}

pub fn model(app: &App) -> Model {
    let config: map_io::Config = map_io::read_config("../config/simulator_config.json");

    let window_id = app
        .new_window()
        .event(event)
        .key_pressed(ctrl::key_pressed)
        .key_released(ctrl::key_released)
        .raw_event(raw_window_event)
        .mouse_moved(ctrl::mouse_moved)
        .mouse_pressed(ctrl::mouse_pressed)
        .mouse_released(ctrl::mouse_released)
        .mouse_wheel(ctrl::mouse_wheel)
        .size(config.screen.width, config.screen.height)
        .view(view)
        .build().unwrap();
    
    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();

    Model::new(app, window_id, config, meshes)
}

pub fn update(_app: &App, model: &mut Model, _update: Update) {
    gui::update_gui(_app, model, &_update);
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if model.initialized == false {return;}
    let sina = model.pose.z.sin();
    let cosa = model.pose.z.cos();
    model.pose.x = model.pose.x + model.velo.x * cosa - model.velo.y * sina; 
    model.pose.y = model.pose.y + model.velo.x * sina + model.velo.y * cosa; 

    unsafe {
        if model.inside_gui == false {
            let mouse = plot::local_mouse_position(_app, &model.wtrans);
            let dir = mouse - pt2(model.pose.x, model.pose.y);
            let target_angle = dir.y.atan2(dir.x);
            let diff = utils::good_angle(target_angle - model.pose.z);
            LOCAL_INT += diff;
            let kd_val = diff - LOCAL_DIFF;
            LOCAL_DIFF = diff;
            model.pose.z += model.pid.x * diff + model.pid.y * LOCAL_INT + model.pid.z * kd_val;
            model.pose.z = utils::good_angle(model.pose.z);
        }
        let pose = Vec3_cpp {x:model.pose.x, y:model.pose.y, z:model.pose.z};
        // TODO: 
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = plot::window_transform(app.draw(), &model.wtrans);

    if model.plot_config.draw_grid == true {
        let win = app.main_window().rect();
        plot::draw_grid(&draw, &win, model.plot_config.grid_step, 1.0, &model.color.grid_color, model.plot_config.grid_alpha);
        plot::draw_grid(&draw, &win, model.plot_config.grid_step / 5., 0.5, &model.color.grid_color, model.plot_config.grid_alpha);
    }
    let (bg_r, bg_g, bg_b) = model.color.bg_color;
    draw.background().rgba(bg_r, bg_g, bg_b, 1.0);
    let (r, g, b, a) = model.color.shape_color;
    for mesh in model.map_points.iter() {
        let points = (0..mesh.len()).map(|i| {
            mesh[i]
        });
        draw.polygon()
            .rgba(r, g, b, a)
            .points(points);
    }

    draw.ellipse()
        .w(15.)
        .h(15.)
        .x(model.pose.x)
        .y(model.pose.y)
        .color(STEELBLUE);
    
    let start_pos = pt2(model.pose.x, model.pose.y);
    let dir = plot::local_mouse_position(app, &model.wtrans) - start_pos;
    let norm = (dir.x * dir.x + dir.y * dir.y + 1e-5).sqrt();
    draw.arrow()
        .start(start_pos)
        .end(start_pos + dir * 40. / norm)
        .weight(2.)
        .color(MEDIUMSPRINGGREEN);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
