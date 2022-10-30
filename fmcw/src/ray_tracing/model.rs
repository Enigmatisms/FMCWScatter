use chrono::format;
use nannou::prelude::*;
use nannou_egui::Egui;

use super::rt_helper::static_scene_update; 

use crate::utils::map_io;
use crate::utils::color::EditorColor;
use crate::utils::ffi_helper::Vec2_cpp;
use crate::utils::world_objs::{ObjInfo, AABB, World};
use crate::utils::utils::{initialize_cpp_end, ModelBasics, BOUNDARIES};
use crate::utils::structs::{WindowCtrl, WindowTransform, PlotConfig, KeyStatus};

pub fn get_aabb(meshes: &Vec<Vec<Point2>>) -> Vec<AABB> {
    let mut results: Vec<AABB> = Vec::new();
    for mesh in meshes.iter() {
        let mut pt_max = pt2(-1e5, -1e5);
        let mut pt_min = pt2(1e5, 1e5);
        for pt in mesh.iter() {
            pt_max = pt_max.max(*pt);
            pt_min = pt_min.min(*pt);
        }
        results.push(AABB::new(pt_max, pt_min));
    }
    let mut pt_max = pt2(-1e5, -1e5);
    let mut pt_min = pt2(1e5, 1e5);
    for pt in BOUNDARIES.iter() {
        let _pt2 = pt2(pt.0, pt.1);
        pt_max = pt_max.max(_pt2);
        pt_min = pt_min.min(_pt2);
    }
    results.push(AABB::new(pt_max, pt_min));
    results
}

pub fn get_mesh_obj_indices(meshes: &Vec<Vec<Point2>>) -> Vec<i16> {
    let mut results: Vec<i16> = Vec::new();
    for (i, mesh) in meshes.iter().enumerate() {
        results.extend(vec![i as i16; mesh.len()]);
    }
    // Boundaries (which are not inside meshes)
    results.extend(vec![meshes.len() as i16; 4]);
    results
}

pub struct RayTracingCtrl {
    pub flattened_pts: Vec<Vec2_cpp>,
    pub nexts: Vec<i8>,
    pub objects: Vec<ObjInfo>,
    pub world: World,
    pub obj_inds: Vec<i16>,
    pub ray_num: usize,
    pub bounces: usize
}

impl RayTracingCtrl {
    pub fn new(
        f_pts: Vec<Vec2_cpp>, nexts: Vec<i8>, raw_objs: map_io::ObjVecJson, aabbs: Vec<AABB>, 
        w: World, inds: Vec<i16>, rnum: usize, bnum: usize
    ) -> Self {
        RayTracingCtrl {
            flattened_pts: f_pts, nexts: nexts, objects: RayTracingCtrl::load_from_raw(raw_objs, aabbs), 
            world: w, obj_inds: inds, ray_num: rnum, bounces: bnum
        }
    }

    pub fn load_from_raw(raw_objs: map_io::ObjVecJson, aabbs: Vec<AABB>) -> Vec<ObjInfo> {
        let mut objs: Vec<ObjInfo> = Vec::new();
        let iter = std::iter::zip(raw_objs.items.into_iter(), aabbs.into_iter());
        for (item, aabb) in iter {
            objs.push(ObjInfo::from_raw(item, aabb));
        }
        objs
    }
}

pub struct Model {
    pub map_points: Vec<Vec<Point2>>,
    pub plot_config: PlotConfig,
    pub wctrl: WindowCtrl,
    pub wtrans: WindowTransform,
    pub rt_ctrl: RayTracingCtrl,
    pub ray_paths: Vec<Vec<Point2>>,
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
        initialize_cpp_end(&meshes, &mut flat_pts, &mut next_ids);
        let aabbs = get_aabb(&meshes);
        // Load object file
        let path = format!("{}.json", config.map_path);
        let mut raw_objs = map_io::read_config::<map_io::ObjVecJson, _>(path);
        let world_obj = raw_objs.items.pop().unwrap();
        let mesh_inds = get_mesh_obj_indices(&meshes);
        let model = Model {
            map_points: meshes, 
            plot_config: PlotConfig::new(),
            wctrl: WindowCtrl::new(
                window_id, window_id, config.screen.width as f32, config.screen.height as f32, 
                config.screen.sub_width as f32, config.screen.sub_height as f32, exit
            ),
            wtrans: WindowTransform::new(),
            rt_ctrl: RayTracingCtrl::new(
                flat_pts, next_ids, raw_objs, aabbs, World::from_raw(config.tracer.pixel_scale, world_obj), 
                mesh_inds, config.tracer.ray_num, config.tracer.bounces
            ),
            ray_paths: vec![vec![pt2(0., 0.); config.tracer.bounces + 1]; config.tracer.ray_num],
            pose: pt3(0., 0., 0.),
            velo: pt3(0., 0., 0.),
            pid: pt3(config.robot.pid_kp, config.robot.pid_ki, config.robot.pid_kd),
            velo_max: pt2(config.robot.t_vel, config.robot.r_vel),
            initialized: false,
            color: EditorColor::new(),
            egui: Egui::from_window(&app.window(window_id).unwrap()),
            key_stat: KeyStatus{ctrl_pressed: false},
            inside_gui: false,
        };
        unsafe {
            static_scene_update(
                model.rt_ctrl.flattened_pts.as_ptr(), model.rt_ctrl.objects.as_ptr(),
                &model.rt_ctrl.world, model.rt_ctrl.obj_inds.as_ptr(), model.rt_ctrl.nexts.as_ptr(), 
                model.rt_ctrl.flattened_pts.len() as u64, model.rt_ctrl.objects.len() as u64
            );
        }
        model
    }

    pub fn get_mesh_num(&self) -> usize {
        self.rt_ctrl.flattened_pts.len()
    }

    pub fn get_obj_num(&self) -> usize {
        self.rt_ctrl.objects.len()
    }
}

fn exit(app: &App) {
    app.quit();
}