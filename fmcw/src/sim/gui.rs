use nannou::prelude::*;
use nannou_egui::{self, egui};

use super::model::Model;
use super::toggle::toggle;
use super::plot::take_snapshot;
use super::map_io::{read_config_rdf, load_map_file};

pub fn update_gui(app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut map_points,
        ref mut chirp,
        ref mut fmcw_p,
        ref mut wctrl,
        ref mut plot_config,
        ref mut wtrans,
        ref mut egui,
        ref mut inside_gui,
        ref mut color,
        ref mut velo_max,
        ref mut pid,
        ref mut initialized,
        ..
    } = model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Editor configuration").default_width(270.);
    let window = window.open(&mut wctrl.gui_visible);
    window.show(&ctx, |ui| {
        egui::Grid::new("switch_grid")
            .striped(true)
        .show(ui, |ui| {
            ui.label("Draw grid");
            ui.add(toggle(&mut plot_config.draw_grid));
            ui.label("Night mode");
            if ui.add(toggle(&mut color.night_mode)).changed() {
                color.switch_mode();
            }
            ui.end_row();
        });

        egui::Grid::new("slide_bars")
            .striped(true)
        .show(ui, |ui| {
            ui.label("Grid size");
            ui.add(egui::Slider::new(&mut plot_config.grid_step, 20.0..=200.0));
            ui.end_row();

            ui.label("Grid alpha");
            ui.add(egui::Slider::new(&mut plot_config.grid_alpha, 0.001..=0.1));
            ui.end_row();

            ui.label("Canvas zoom scale");
            ui.add(egui::Slider::new(&mut wtrans.scale, 0.5..=2.0));
            ui.end_row();
            
            ui.label("Max linear vel:");
            ui.add(egui::Slider::new(&mut velo_max.x, 0.2..=2.0));
            ui.end_row();
            
            ui.label("Angular K(p): ");
            ui.add(egui::Slider::new(&mut pid.x, 0.01..=0.4));
            ui.end_row();

            ui.label("Angular K(i): ");
            ui.add(egui::Slider::new(&mut pid.y, 0.00..=0.01));
            ui.end_row();

            ui.label("Angular K(d): ");
            ui.add(egui::Slider::new(&mut pid.z, 0.00..=0.1));
            ui.end_row();

            ui.label("Base freq: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.base_f, 3e9..=5e9)).clicked();
            ui.end_row();
            ui.label("Chirp time: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.edge_len, 2e-6..=4e-6)).clicked();
            ui.end_row();
            ui.label("Band width: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.band_w, 5e9..=1e10)).clicked();
            ui.end_row();
            ui.label("Sample time: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.sp_int, 3.33e-11..=5e-11)).clicked();
            ui.end_row();
            ui.label("ToF std: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.tof_std, 2e-11..=3e-10)).clicked();
            ui.end_row();
            ui.label("Doppler std: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.doppler_std, 1e-5..=1e-3)).clicked();
            ui.end_row();

            ui.label("Sample std: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.sample_std, 0.001..=0.1)).clicked();
            ui.end_row();
            ui.label("LPF cutoff: ");
            fmcw_p.reset |= ui.add(egui::Slider::new(&mut fmcw_p.cut_off, 2e8..=2e10)).clicked();
            ui.end_row();

            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Centering view").clicked() {
                    wtrans.clear_offset();
                }
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Take screenshot").clicked() {
                    take_snapshot(&app.main_window());
                }
            });
            ui.end_row();

            // this implementation is so fucking ugly
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Load map file").clicked() {
                    let mut raw_points: Vec<Vec<Point2>> = Vec::new();
                    load_map_file(&mut raw_points);
                    Model::initialize_cpp_end(&raw_points, &mut chirp.flattened_pts, &mut chirp.nexts);      // re-intialize CUDA (ray tracer)
                    *map_points = raw_points;           // replacing map points
                    *initialized = false;               // should reset starting point
                }
            });

            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button("Load config").clicked() {
                    if let Some(new_config) = read_config_rdf() {
                        Model::reload_config(&new_config, &mut wctrl.win_w, &mut wctrl.win_h, pid, velo_max);
                    }
                }
            });
            ui.end_row();
        });
        *inside_gui = ui.ctx().is_pointer_over_area();
    });
}
