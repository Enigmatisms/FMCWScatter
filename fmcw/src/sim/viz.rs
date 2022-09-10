use nannou::prelude::*;
use array2d::Array2D;

use super::gui;
use super::ctrl;
use super::fmcw_helper;
use super::model::Model;

use super::plot;
use super::utils;
use super::map_io;

const BOUNDARIES: [(f32, f32); 4] = [(-1170.0, -870), (1170, -870), (1170, 870), (-1170, 870)];
const BOUNDARY_IDS: [i8; 4] = [3, 0, 0, -3];

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
        .build()
        .unwrap();

    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();

    let lidar_param = fmcw_helper::Vec3_cpp{x: config.lidar.amin, y: config.lidar.amax, z:config.lidar.ainc};
    let ray_num = map_io::get_ray_num(&lidar_param);

    let mut total_pt_num = 0;
    initialize_cuda_end(&meshes, ray_num, &mut total_pt_num, false);
    Model::new(app, window_id, &config, meshes)
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

pub fn initialize_cpp_end(new_pts: &map_io::Meshes, ray_num: usize, total_pt_num: &mut usize, initialized: bool) {
    let mut points: Vec<fmcw_helper::Vec2_cpp> = Vec::new();
    let mut next_ids: Vec<i8> = Vec::new();
    for mesh in new_pts.iter() {
        for pt in mesh.iter() {
            points.push(fmcw_helper::Vec2_cpp{x: pt.x, y: pt.y});
        }
        let length = mesh.len();
        let offset: i8 = (length as i8) - 1;
        let mut ids: Vec<i8> = vec![0; length];
        ids[0] = offset;
        ids[length - 1] = -offset;
        next_ids.extend(ids.into_iter());
    }
    for i in 0..4 {                                                 // add boundaries
        let (x, y) = BOUNDARIES[i];
        points.push(fmcw_helper::Vec2_cpp{x: x, y: y});
        next_ids.push(BOUNDARY_IDS[i]);
    }
} 

pub fn update(_app: &App, _model: &mut Model, _update: Update) {
    gui::update_gui(_app, _model, &_update);
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if _model.initialized == false {return;}

    let mouse = plot::local_mouse_position(_app, &_model.wtrans);
    let dir = mouse - pt2(_model.pose.x, _model.pose.y);
    let target_angle = dir.y.atan2(dir.x);
    let diff = utils::good_angle(target_angle - _model.pose.z);
    unsafe {
        LOCAL_INT += diff;
        let kd_val = diff - LOCAL_DIFF;
        LOCAL_DIFF = diff;
        _model.pose.z += _model.pid.x * diff + _model.pid.y * LOCAL_INT + _model.pid.z * kd_val;
        _model.pose.z = utils::good_angle(_model.pose.z);
        let pose = fmcw_helper::Vec3_cpp {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        // fmcw_helper::rayTraceRender(&_model.lidar_param, &pose, _model.ray_num as i32, _model.lidar_noise, _model.ranges.as_mut_ptr());
        // TODO: fmcw_helper此处调用两个函数，range finder以及chirp gen
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
    
    /// TODO: Visualize single ray!
        
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

/// TODO: visualize single ray
// fn visualize_rays(
//     draw: &Draw, ranges: &Vec<libc::c_float>, pose: &Point3, 
//     lidar_param: &fmcw_helper::Vec3_cpp, color: &[f32; 4], ray_num: usize) {
//     let cur_angle_min = pose.z + lidar_param.x + lidar_param.z;
//     for i in 0..ray_num {
//         let r = ranges[i];
//         // if r > 1e5 {continue;}
//         let cur_angle = cur_angle_min + lidar_param.z * 3. * (i as f32);
//         let dir = pt2( cur_angle.cos(), cur_angle.sin());
//         let start_p = pt2(pose.x, pose.y);
//         let end_p = start_p + dir * r;
//         draw.line()
//             .start(start_p)
//             .end(end_p)
//             .weight(1.)
//             .rgba(color[0], color[1], color[2], color[3]);
//     }
// }
