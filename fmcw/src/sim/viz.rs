use nannou::prelude::*;

use super::gui;
use super::ctrl;
use super::fmcw_helper;
use super::model::Model;

use super::plot;
use super::utils;
use super::map_io;

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

    let sub_win_id = app.new_window()
        .event(event)
        .view(view_spectrum)
        .always_on_top(true)
        .size(config.screen.sub_width, config.screen.sub_height)
        .build().unwrap();
    
    app.set_exit_on_escape(false);
    let meshes: map_io::Meshes = map_io::parse_map_file(config.map_path.as_str()).unwrap();

    Model::new(app, window_id, sub_win_id, config, meshes)
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
        let pose = fmcw_helper::Vec3_cpp {x:model.pose.x, y:model.pose.y, z:model.pose.z};
        fmcw_helper::laserRangeFinder(
            &pose, model.chirp.flattened_pts.as_ptr(), model.chirp.nexts.as_ptr(), 
            model.chirp.nexts.len() as i32, &mut model.chirp.gt_r
        );
        if model.fmcw_p.reset == true {
            zero_padding(&mut model.chirp.spect, model.fmcw_p.sp_int, model.fmcw_p.edge_len);
        }
        fmcw_helper::simulateOnce(&mut model.fmcw_p, model.chirp.spect.as_mut_ptr(), &mut model.chirp.pred_r,
         &mut model.chirp.pred_v, model.chirp.gt_r * model.chirp.map_resolution, model.velo.x * model.chirp.map_resolution);
        model.err_plot.push((model.chirp.pred_r - model.chirp.gt_r * model.chirp.map_resolution).abs());
        model.err_plot.push_vel(model.chirp.pred_v, model.velo.x * model.chirp.map_resolution);
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
    let valid_spect: Vec<&f32> = model.chirp.spect.iter().take_while(|&x| *x > 1e-2).collect();
    
    let draw = app.draw();
    let total_len = valid_spect.len();
    let (half_w, h_span, half_h) = (model.wctrl.sub_w / 2. - 10., model.wctrl.sub_h / 3., model.wctrl.sub_h / 2.);
    let separation_line_h = h_span / 2.;
    let c = &model.color;
    let (lc_r, lc_g, lc_b, lc_a) = c.sepline_color;

    draw.background().rgb(c.bg_color.0, c.bg_color.1, c.bg_color.2);
    draw.line().weight(4.)
        .start(pt2(-half_w, separation_line_h))
        .end(pt2(half_w, separation_line_h))
        .rgba(lc_r, lc_g, lc_b, lc_a);
    draw.line().weight(4.)
        .start(pt2(-half_w, -separation_line_h))
        .end(pt2(half_w, -separation_line_h))
        .rgba(lc_r, lc_g, lc_b, lc_a);
    draw.text(SPECTRUM_STR)
        .xy(pt2(half_w - 35., half_h - 5.))
        .font_size(14)
        .rgba(lc_r, lc_g, lc_b, lc_a);
    draw.text(VELOCITY_STR)
        .xy(pt2(half_w - 100., separation_line_h - 15.))
        .font_size(14).right_justify()
        .rgba(lc_r, lc_g, lc_b, lc_a);
    draw.text(ERROR_STR)
        .xy(pt2(half_w - 80., -separation_line_h - 8.))
        .font_size(14)
        .rgba(lc_r, lc_g, lc_b, lc_a);

    if total_len > 0 {
        let spectrum_pts = (0..total_len).map(|i| {
            pt2((i as f32 / total_len as f32) * half_w * 2. - half_w, *valid_spect[i] * h_span + separation_line_h)
        });
        
        draw.polyline()
            .weight(1.5)
            .points(spectrum_pts)
            .rgba(c.spec_color.0, c.spec_color.1, c.spec_color.2, c.spec_color.3);
    }
    if let Some((err_pts, avg_y, err)) = model.err_plot.get_err_pts(half_w, h_span, model.wctrl.sub_h / 2.) {
        draw.polyline()
            .weight(1.5)
            .points(err_pts)
            .rgba(c.truth_color.0, c.truth_color.1, c.truth_color.2, c.truth_color.3);
        draw.line()
            .start(pt2(-half_w, avg_y))
            .end(pt2(half_w, avg_y))
            .rgba(c.pred_color.0, c.pred_color.1, c.pred_color.2, c.pred_color.3);
        let msg = format!("Average error: {:.3}", err);
        draw.text(&msg)
            .xy(pt2(-half_w + 100., avg_y - 15.))
            .font_size(14)
            .rgba(lc_r, lc_g, lc_b, lc_a);
    }
    if let Some((gtv_pts, pred_pts)) = model.err_plot.get_vel_pts(half_w, h_span) {
        draw.polyline()
            .weight(1.5)
            .points(gtv_pts)
            .rgba(c.truth_color.0, c.truth_color.1, c.truth_color.2, c.truth_color.3);
        draw.polyline()
            .weight(1.5)
            .points(pred_pts)
            .rgba(c.pred_color.0, c.pred_color.1, c.pred_color.2, c.pred_color.3);
    }
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
