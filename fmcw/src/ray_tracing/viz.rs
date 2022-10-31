use nannou::prelude::*;

use super::gui;
use super::ctrl;
use super::rt_helper;
use super::model::Model;

use crate::utils::plot;
use crate::utils::utils;
use crate::utils::map_io;
use crate::utils::ffi_helper::{Vec2_cpp, Vec3_cpp};

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn copy_ray_path_points(points: &mut Vec<Vec<Point2>>, raw: &Vec<Vec2_cpp>, index: usize) {
    let iter = std::iter::zip(raw.iter(), points.iter_mut());
    for (pt, dst_vec) in iter {
        dst_vec[index] = pt2(pt.x, pt.y);
    }
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
    let map_path = format!("{}.txt", config.map_path);
    let meshes: map_io::Meshes = map_io::parse_map_file(map_path).unwrap();
    unsafe {
        rt_helper::setup_path_tracer(config.tracer.ray_num as libc::c_int);
    }
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
        let mut intersections: Vec<Vec2_cpp> = vec![Vec2_cpp::default(); model.rt_ctrl.ray_num];
        // Total 8 bounces, 9 end points (including the starting point)
        for ray_path in model.ray_paths.iter_mut() {
            ray_path[0] = pt2(pose.x, pose.y);
        }
        rt_helper::first_ray_intersections(
            intersections.as_mut_ptr(), &pose, 
            model.get_mesh_num() as libc::c_int, model.get_obj_num() as libc::c_int
        );
        copy_ray_path_points(&mut model.ray_paths, &intersections, 1);
        for i in 2..=model.rt_ctrl.bounces {
            rt_helper::get_intersections_update(
                intersections.as_mut_ptr(), model.get_mesh_num() as libc::c_int, model.get_obj_num() as libc::c_int
            );
            copy_ray_path_points(&mut model.ray_paths, &intersections, i);
        }
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
    let (r, g, b, _) = model.color.shape_color;
    for (id, mesh) in model.map_points.iter().enumerate() {
        let points = (0..mesh.len()).map(|i| {
            mesh[i]
        });
        let (r, g, b) = match model.rt_ctrl.objects[id]._type {
            0 => {model.color.scene.diffusive},
            1 => {model.color.scene.glossy},
            2 => {model.color.scene.specular},
            3 => {model.color.scene.refractive},
            _ => {(r, g, b)},
        };
        draw.polygon()
            .rgb(r, g, b)
            .points(points);
    }

    draw.ellipse()
        .w(15.)
        .h(15.)
        .x(model.pose.x)
        .y(model.pose.y)
        .color(STEELBLUE);

    draw_ray_path(&draw, &model.ray_paths, (1., 0., 0., 0.01));
    
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

// The strength of ray (concentration of rays) is determined by alpha
fn draw_ray_path(draw: &Draw, ray_paths: &Vec<Vec<Point2>>, base_color: (f32, f32, f32, f32)) {
    for path in ray_paths.iter() {
        let max_len = path.len();
        let path_pts = (0..max_len).map(|i| {
            path[i]
        });
        draw.polyline()
            .weight(3.0)
            .points(path_pts)
            .rgba(base_color.0, base_color.1, base_color.2, base_color.3);
    }
}
