use nannou::prelude::*;
use nannou_egui::Egui;
use nannou_egui::{self, egui};

pub struct Model {
    pub ref_index: f32,
    pub incident: Point2,
    pub refract: Point2,
    pub reflect: Point2,
    pub norm_angle: f32,
    pub reflect_ratio: f32,
    pub length: f32,
    pub egui: Egui,
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn sphere_direction(input: Point2, angle: f32) -> Point2 {
    input * angle.cos() - pt2(-input.y, input.x) * angle.sin()
}

fn reflect_dir(inci_dir: Point2, norm_dir: Point2) -> Point2 {
    let proj = inci_dir.dot(norm_dir);
    inci_dir - 2. * norm_dir * proj
}

fn snell_law(inci_dir: Point2, norm_dir: Point2, mut n1_n2_ratio: f32, same_dir: bool) -> f32 {
    // n1_n2_ratio = (RI media outside) / (RI media after incidence)
    if same_dir {
        n1_n2_ratio = 1. / n1_n2_ratio;
    }
    let result = ((norm_dir.y * inci_dir.x - norm_dir.x * inci_dir.y) * n1_n2_ratio).asin();
    if same_dir == false {
        return std::f32::consts::PI - result;
    } else {
        return result;
    }
}

// cos_inc should be positive, cos_ref can be calc by cos(output of snell law) then abs
// how much light is reflected
fn frensel_equation(n1: f32, n2: f32, cos_inc: f32, cos_ref: f32) -> f32 {
    let n1cos_i = n1 * cos_inc;
    let n2cos_i = n2 * cos_inc;
    let n1cos_r = n1 * cos_ref;
    let n2cos_r = n2 * cos_ref;
    let rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r);
    let rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i);
    return 0.5 * (rs.pow(2.) + rp.pow(2.));
}

pub fn model(app: &App) -> Model {
    let win_id = app
        .new_window()
        .event(event)
        .raw_event(raw_window_event)
        .size(800, 800)
        .view(view)
        .build().unwrap();
    Model {
        ref_index: 2.0, incident: pt2(0., 0.), refract: pt2(0., 0.), reflect: pt2(0., 0.),
        norm_angle: std::f32::consts::PI / 4., reflect_ratio: 0.5, length: 256., egui: Egui::from_window(&app.window(win_id).unwrap()),
    }
}

pub fn update(_app: &App, model: &mut Model, _update: Update) {
    update_gui(_app, model, &_update);
    let norm_vec = pt2(model.norm_angle.cos(), model.norm_angle.sin());
    model.incident = -_app.mouse.position().normalize_or_zero();
    model.reflect = reflect_dir(model.incident, norm_vec);
    let cos_inc = model.incident.dot(norm_vec);
    let same_direction = cos_inc > 0.;
    let angle = snell_law(model.incident, norm_vec, 1. / model.ref_index, same_direction);
    if angle.is_nan() {
        model.refract = pt2(0., 0.);
        model.reflect_ratio = 1.;
    } else {
        model.refract = sphere_direction(norm_vec, angle);
        if same_direction {
            model.reflect_ratio = frensel_equation(model.ref_index, 1., cos_inc.abs(), angle.cos().abs());
        } else {
            model.reflect_ratio = frensel_equation(1., model.ref_index, cos_inc.abs(), angle.cos().abs());
        }
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().rgba(0., 0., 0., 1.0);
    draw.arrow()
        .start(-model.incident * model.length)
        .end(pt2(0., 0.))
        .weight(4.)
        .color(MEDIUMSPRINGGREEN);
    
    draw.arrow()
        .start(pt2(0., 0.))
        .end(model.refract * model.length)
        .weight(4.)
        .rgba(1., 0., 0., 1. - model.reflect_ratio);

    draw.arrow()
        .start(pt2(0., 0.))
        .end(model.reflect * model.length)
        .weight(4.)
        .rgba(0., 0.5, 1., model.reflect_ratio);
    
    let norm_vec = pt2(model.norm_angle.cos(), model.norm_angle.sin());
    let perp_norm_vec = pt2(-norm_vec.y, norm_vec.x);
    draw.line()
        .start(perp_norm_vec * model.length + norm_vec * 3.)
        .end(-perp_norm_vec * model.length + norm_vec * 3.)
        .weight(6.)
        .color(WHITE);
    draw.line()
        .start(perp_norm_vec * model.length - norm_vec * 3.)
        .end(-perp_norm_vec * model.length - norm_vec * 3.)
        .weight(6.)
        .color(GRAY);

    let points = (0..=360).map(|i| {
        let radian = deg_to_rad(i as f32);
        let x = radian.sin() * model.length;
        let y = radian.cos() * model.length;
        pt2(x,y)
    });
    draw.polyline()
        .weight(3.0)
        .points(points)
        .rgba(0., 0.3, 1.0, 0.1);

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

pub fn update_gui(_app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut norm_angle,
        ref mut ref_index,
        ref mut egui,
        ..
    } = model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Configuration").default_width(200.);
    window.show(&ctx, |ui| {
        egui::Grid::new("slide_bars")
            .striped(true)
        .show(ui, |ui| {
            ui.label("Refractive I: ");
            ui.add(egui::Slider::new(ref_index, 1.0..=3.0));
            ui.end_row();
            ui.label("Normal Angle: ");
            ui.add(egui::Slider::new(norm_angle, -std::f32::consts::PI..=std::f32::consts::PI));
            ui.end_row();
        });
    });
}
