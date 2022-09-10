use nannou::prelude::*;
use nannou_egui::{self, egui};

use super::model::Model;
use super::toggle::toggle;
use super::plot::take_snapshot;
use super::map_io::read_config_rdf;

pub fn update_gui(app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut map_points,
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
                    let raw_points: Vec<Vec<Point2>> = Vec::new();
                    // let mut total_pt_num = 0;
                    // TODO: initialize cpp end
                    // initialize_cpp_end(&raw_points, *ray_num, &mut total_pt_num, true);      // re-intialize CUDA (ray tracer)
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
