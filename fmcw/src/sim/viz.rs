use nannou::prelude::*;

use super::gui;
use super::ctrl;
use super::fmcw_helper;
use super::model::Model;

use super::plot;
use super::utils;
use super::map_io;

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn dummy_key_pressed(_app: &App, _model: &mut Model, _key: Key) {}
fn dummy_key_released(_app: &App, _model: &mut Model, _key: Key) {}
fn dummy_mouse_moved(_app: &App, _model: &mut Model, _pos: Point2) {}
fn dummy_mouse_pressed(_app: &App, _model: &mut Model, _button: MouseButton) {}
fn dummy_mouse_released(_app: &App, _model: &mut Model, _button: MouseButton) {}
fn dummy_raw_event(_app: &App, _model: &mut Model, _event: &nannou::winit::event::WindowEvent) {}
fn dummy_mouse_wheel(_app: &App, _model: &mut Model, _dt: MouseScrollDelta, _phase: TouchPhase) {}

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

    let sub_win_id = app.new_window()
        .event(event)
        .key_pressed(dummy_key_pressed)
        .key_released(dummy_key_released)
        .raw_event(dummy_raw_event)
        .mouse_moved(dummy_mouse_moved)
        .mouse_pressed(dummy_mouse_pressed)
        .mouse_released(dummy_mouse_released)
        .mouse_wheel(dummy_mouse_wheel)
        .view(view_spectrum)
        .always_on_top(true)
        .size(config.screen.sub_width, config.screen.sub_height)
        .build().unwrap();
    
    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();

    Model::new(app, window_id, sub_win_id, config, meshes)
}

pub fn update(_app: &App, _model: &mut Model, _update: Update) {
    gui::update_gui(_app, _model, &_update);
    static mut LOCAL_INT: f32 = 0.0;
    static mut LOCAL_DIFF: f32 = 0.0;
    if _model.initialized == false {return;}
    let sina = _model.pose.z.sin();
    let cosa = _model.pose.z.cos();
    _model.pose.x = _model.pose.x +_model.velo.x * cosa - _model.velo.y * sina; 
    _model.pose.y = _model.pose.y +_model.velo.x * sina + _model.velo.y * cosa; 

    unsafe {
        if _model.inside_gui == false {
            let mouse = plot::local_mouse_position(_app, &_model.wtrans);
            let dir = mouse - pt2(_model.pose.x, _model.pose.y);
            let target_angle = dir.y.atan2(dir.x);
            let diff = utils::good_angle(target_angle - _model.pose.z);
            LOCAL_INT += diff;
            let kd_val = diff - LOCAL_DIFF;
            LOCAL_DIFF = diff;
            _model.pose.z += _model.pid.x * diff + _model.pid.y * LOCAL_INT + _model.pid.z * kd_val;
            _model.pose.z = utils::good_angle(_model.pose.z);
        }
        let pose = fmcw_helper::Vec3_cpp {x:_model.pose.x, y:_model.pose.y, z:_model.pose.z};
        fmcw_helper::laserRangeFinder(
            &pose, _model.chirp.flattened_pts.as_ptr(), _model.chirp.nexts.as_ptr(), 
            _model.chirp.nexts.len() as i32, &mut _model.chirp.gt_r
        );
        if _model.fmcw_p.reset == true {
            zero_padding(&mut _model.chirp.spect, _model.fmcw_p.sp_int, _model.fmcw_p.edge_len);
        }
        fmcw_helper::simulateOnce(&mut _model.fmcw_p, _model.chirp.spect.as_mut_ptr(), &mut _model.chirp.pred_r,
         &mut _model.chirp.pred_v, _model.chirp.gt_r * _model.chirp.map_resolution, _model.velo.x * _model.chirp.map_resolution);
        println!("Pred range: {}, range: {}, pred vel: {}, vel: {}", _model.chirp.pred_r, _model.chirp.gt_r * _model.chirp.map_resolution, _model.chirp.pred_v, _model.velo.x * _model.chirp.map_resolution);
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
    
    visualize_single_ray(&draw, model.chirp.gt_r, &model.pose, &model.color.lidar_color);
        
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

fn view_spectrum(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    
    let valid_spect: Vec<&f32> = model.chirp.spect.iter().take_while(|&x| *x > 1e-2).collect();
    let total_len = valid_spect.len();
    if total_len == 0 {
        return;
    }
    let (half_w, half_h) = (model.wctrl.sub_w / 2. - 10., model.wctrl.sub_h / 2. - 10.);
    
    let pts = (0..total_len).map(|i| {
        pt2((i as f32 / total_len as f32) * 2. * half_w - half_w, *valid_spect[i] * 2. * half_h - half_h)
    });
    draw.background().rgba(0., 0., 0., 1.0);
    draw.polyline()
        .weight(1.5)
        .points(pts)
        .rgba(1., 0., 0., 1.);
    draw.to_frame(app, &frame).unwrap();
}

fn visualize_single_ray(draw: &Draw, range: f32, pose: &Point3, color: &[f32; 4]) {
    let dir = pt2( pose.z.cos(), pose.z.sin());
    let start_p = pt2(pose.x, pose.y);
    let end_p = start_p + dir * range;
    draw.line()
        .start(start_p)
        .end(end_p)
        .weight(1.)
        .rgba(color[0], color[1], color[2], color[3]);
}
