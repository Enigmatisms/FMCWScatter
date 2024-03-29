use std::fs;
use serde_derive::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use nannou::prelude::*;
use std::io::{prelude::*, BufReader};
use super::ffi_helper::Vec2_cpp;

pub type Mesh = Vec<Point2>;
pub type Meshes = Vec<Mesh>;

// For loading and dumping from/to files, I don't want to load AABB from files currently
#[derive(Deserialize, Serialize, Clone)]
pub struct ObjInfoJson {
    pub _type: u8,
    pub ref_index: f32,
    pub rdist: f32,
    pub r_gain: f32,
    pub u_a: f32,
    pub u_s: f32,
    pub p_c: f32
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ObjVecJson {
    pub items: Vec<ObjInfoJson>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ScannerConfig {
    pub t_vel: f32,
    pub r_vel: f32,
    pub pid_kp: f32,
    pub pid_ki: f32,
    pub pid_kd: f32,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ScreenConfig {
    pub width: u32,
    pub height: u32,
    pub sub_width: u32,
    pub sub_height: u32
}

#[derive(Deserialize, Serialize, Clone)]
pub struct TracerConfig {
    pub bounces: usize,
    pub ray_num: usize,
    pub pixel_scale: f32,
}

#[repr(C)]
#[derive(Deserialize, Serialize, Clone)]
pub struct Chirp_param {
    pub base_f: libc::c_float,
    pub edge_len: libc::c_float,
    pub band_w: libc::c_float,
    pub sp_int: libc::c_float,
    pub tof_std: libc::c_float,
    pub doppler_std: libc::c_float,
    pub sample_std: libc::c_float,
    pub cut_off: libc::c_float,
    pub nl_c2: libc::c_float,
    pub nl_c3: libc::c_float,
    pub reset: bool
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Config {
    pub robot: ScannerConfig,
    pub screen: ScreenConfig,
    pub fmcw_p: Chirp_param,
    pub tracer: TracerConfig,
    pub map_path: String
}

/// TODO: offset 600 and 450 needs to be removed
pub fn parse_map_file<T>(filepath: T) -> Option<Meshes> where T: AsRef<std::path::Path> {
    if let Some(all_lines) = read_lines(filepath) {
        let mut result: Meshes = Vec::new();
        for line in all_lines.iter() {
            let str_vec: Vec<&str> = line.split(" ").collect();
            let point_num =  str_vec[0].parse::<usize>().unwrap() + 1;
            let mut mesh: Mesh = Vec::new();
            for i in 1..point_num {
                let str1 = str_vec[(i << 1) - 1];
                let str2 = str_vec[i << 1];
                if str1.is_empty() == true {
                    break;
                } else {
                    mesh.push(pt2(
                        str1.parse::<f32>().unwrap(),
                        str2.parse::<f32>().unwrap()
                    ));
                }
            }
            result.push(mesh);
        }
        return Some(result);
    } else {
        return None;
    }
}

pub fn meshes_to_segments(meshes: &Meshes, segments: &mut Vec<Vec2_cpp>) -> usize {
    let mut ptr: usize = 0;
    for mesh in meshes.iter() {
        let first = &mesh[0];
        segments.push(Vec2_cpp {x: first.x, y: first.y});
        for i in 1..(mesh.len()) {
            let current = &mesh[i];
            segments.push(Vec2_cpp {x: current.x, y: current.y});
            segments.push(Vec2_cpp {x: current.x, y: current.y});
        }
        segments.push(Vec2_cpp {x: first.x, y: first.y});
        ptr += mesh.len();
    }
    ptr
}

pub fn read_config_rdf() -> Option<Config> {
    let path = rfd::FileDialog::new()
        .set_file_name("../config/simulator_config.json")
        .set_directory(".")
        .pick_file();
    if let Some(path_res) = path {
        return Some(read_config::<Config, _>(path_res));
    }
    None
}

pub fn read_config<RType: DeserializeOwned, PathType: AsRef<std::path::Path>>(file_path: PathType) -> RType {
    let file: fs::File = fs::File::open(file_path).ok().unwrap();
    let reader = BufReader::new(file);
    let config: RType = serde_json::from_reader(reader).ok().unwrap();
    config
}

pub fn load_map_file(map_points: &mut Meshes) -> String {
    let path = rfd::FileDialog::new()
        .set_file_name("../maps/reflect_only.txt")
        .set_directory(".")
        .pick_file();
    let mut result = String::new();
    if let Some(path_res) = path {
        result = String::from(path_res.as_os_str().to_str().unwrap());
        *map_points = parse_map_file(path_res).unwrap();
    } else {
        map_points.clear();
    }
    result
}

// ========== privates ==========
fn read_lines<T>(filepath: T) -> Option<Vec<String>> where T: AsRef<std::path::Path> {
    if let Ok(file) = fs::File::open(filepath) {
        let reader = BufReader::new(file);
        let mut result_vec: Vec<String> = Vec::new();
        for line in reader.lines() {
            if let Ok(line_inner) = line {
                result_vec.push(line_inner);
            } else {
                return None;
            }
        }
        return Some(result_vec);
    }
    println!("Unable to open file.");
    return None;
}
