
type Color3 = (f32, f32, f32);
type Color4 = (f32, f32, f32, f32);
type LColor4 = [f32; 4];

pub struct ComplexColor {
    pub diffusive: Color3,
    pub specular: Color3,
    pub glossy: Color3,
    pub refractive: Color3
}

pub struct EditorColor {
    pub bg_color: Color3,
    pub traj_color: Color3,
    pub grid_color: Color3,
    pub selected_pt: Color4,
    pub unfinished_pt: Color4,
    pub finished_pt: Color4,
    pub line_color: Color4,
    pub shape_color: Color4,
    pub select_box: Color4,

    pub spec_color: Color4,
    pub sepline_color: Color4,
    pub pred_color: Color4,
    pub truth_color: Color4,
    pub scene: ComplexColor,
    
    pub lidar_color: LColor4,
    pub night_mode: bool
}

impl EditorColor {
    pub fn new() -> EditorColor{
        EditorColor {
            traj_color: (0., 1., 0.),
            bg_color: (0., 0., 0.),
            grid_color: (1., 1., 1.),
            selected_pt: (1.000000, 0.094118, 0.094118, 1.0),
            unfinished_pt: (0.301961, 0.298039, 0.490196, 0.8),
            finished_pt: (0.301961, 0.298039, 0.490196, 0.8),
            line_color: (0.913725, 0.835294, 0.792157, 0.9),
            shape_color: (0.803922, 0.760784, 0.682353, 1.0),
            select_box: (0.129412, 0.333333, 0.803922, 0.1),

            spec_color: (1., 0., 0., 1.),
            sepline_color: (1., 1., 1., 0.8),
            pred_color: (0.8, 0.8, 0., 0.7),
            truth_color: (1., 1., 1., 0.7),

            scene: ComplexColor {
                diffusive: (0.73333, 0.73333, 0.73333),
                specular: (0.9921568, 0.980392, 0.964705),
                glossy: (0.89803921, 0.862745, 0.764705),
                refractive: (0.7843137, 0.890196, 0.831372)
            },

            lidar_color: [1., 0., 0., 1.],
            night_mode: true
        }
    }

    pub fn switch_mode(&mut self){
        if self.night_mode == true {
            self.traj_color = (0., 1., 0.);
            self.bg_color = (0., 0., 0.);
            self.grid_color = (1., 1., 1.);

            self.selected_pt = (1.000000, 0.094118, 0.094118, 1.0);
            self.unfinished_pt = (0.301961, 0.298039, 0.490196, 0.8);
            self.finished_pt = (0.301961, 0.298039, 0.490196, 0.8);
            self.line_color = (0.913725, 0.835294, 0.792157, 0.9);
            self.shape_color = (0.803922, 0.760784, 0.682353, 1.0);
            self.select_box = (0.129412, 0.333333, 0.803922, 0.1);

            self.spec_color = (1., 0., 0., 1.);
            self.sepline_color = (1., 1., 1., 0.8);
            self.pred_color = (0.8, 0.8, 0., 0.7);
            self.truth_color = (1., 1., 1., 0.7);

            self.scene.diffusive = (0.73333, 0.73333, 0.73333);
            self.scene.specular = (0.9921568, 0.980392, 0.964705);
            self.scene.glossy = (0.89803921, 0.862745, 0.764705);
            self.scene.refractive = (0.7843137, 0.890196, 0.831372);
        } else {
            self.traj_color = (0., 0.5, 0.);
            self.bg_color = (1., 1., 1.);
            self.grid_color = (0., 0., 0.);

            self.selected_pt = (0.700000, 0.000000, 0.000000, 1.0);
            self.unfinished_pt = (0.160784, 0.203922, 0.384314, 0.8);
            self.finished_pt = (0.566667, 0.543137, 0.472549, 0.9);
            self.line_color = (0.082353, 0.074510, 0.235294, 0.9);
            self.shape_color = (0.058824, 0.054902, 0.054902, 1.0);
            self.select_box = (0.129412, 0.333333, 0.803922, 0.3);

            self.spec_color = (0., 0., 0.7, 1.);
            self.sepline_color = (0.1, 0.1, 0.1, 1.);
            self.pred_color = (0.7, 0.7, 0., 0.8);
            self.truth_color = (0., 0., 0., 0.8);

            self.scene.diffusive = (0.262745, 0.239215, 0.23529);
            self.scene.specular = (0.117647, 0.129411, 0.176470);
            self.scene.glossy = (0.6039215, 0.58039, 0.51372);
            self.scene.refractive = (0.2352941, 0.13725, 0.09019);
        }
    }
}