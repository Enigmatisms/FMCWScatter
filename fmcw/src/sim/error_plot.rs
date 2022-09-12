use std::collections::VecDeque;
use nannou::prelude::*;

pub struct ErrorPlotter {
    instants: VecDeque<f32>,
    averages: VecDeque<f32>,
    pred_vel: VecDeque<f32>,
    true_vel: VecDeque<f32>,
    max_ints: usize,
    max_avgs: usize,
    max_vels: usize,
    ints_sum: f32,
    avgs_sum: f32
}

impl ErrorPlotter {
    pub fn new(max_ints: usize, max_avgs: usize, max_vels: usize) -> Self {
        ErrorPlotter { 
            instants: VecDeque::new(), averages: VecDeque::new(), pred_vel: VecDeque::new(), true_vel: VecDeque::new(),
            max_ints: max_ints, max_avgs: max_avgs, max_vels: max_vels, ints_sum: 0., avgs_sum: 0. 
        }
    }

    pub fn push(&mut self, val: f32) {
        if self.instants.len() >= self.max_ints {
            let front = self.instants.pop_front().unwrap();
            self.ints_sum -= front;
        }
        self.instants.push_back(val);
        self.ints_sum += val;
        let avg = self.ints_sum / self.instants.len() as f32;
        if self.averages.len() >= self.max_avgs {
            let front = self.averages.pop_front().unwrap();
            self.avgs_sum -= front;
        }
        self.averages.push_back(avg);
        self.avgs_sum += avg;
    }

    pub fn push_vel(&mut self, pred: f32, gt: f32) {
        if self.true_vel.len() >= self.max_vels {
            self.true_vel.pop_front();
            self.pred_vel.pop_front();
        }
        self.pred_vel.push_back(pred);
        self.true_vel.push_back(gt);
    }

    pub fn get_err_pts(&self, half_w: f32, h_span: f32, h_offset: f32) -> Option<(Vec<Vec2>, f32, f32)> {
        if self.averages.len() < 5 {
            return None;
        }
        let max_val = self.averages.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        if *max_val < 1e-5 {
            return None;
        }
        let res = ErrorPlotter::get_pts(&self.averages, 
            |x, a|{x / a}, *max_val, half_w, h_span, h_offset);
        let avg_err = self.get_avg();
        let avg_ratio = (avg_err / *max_val) * h_span - h_offset - 5.;
        Some((res, avg_ratio, avg_err))
    }

    pub fn get_vel_pts(&self, half_w: f32, height: f32) -> Option<(Vec<Vec2>, Vec<Vec2>)>{
        if self.true_vel.len() < 5 {
            return None;
        }
        let gt_pts = ErrorPlotter::get_pts(&self.true_vel, 
            |x, a|{(x * a).clamp(-0.5, 0.5)}, 2.5, half_w, height, 0.);
        let pd_pts = ErrorPlotter::get_pts(&self.pred_vel, 
            |x, a|{(x * a).clamp(-0.5, 0.5)}, 2.5, half_w, height, 0.);
        Some((gt_pts, pd_pts))
    }

    #[inline]
    fn get_pts(array: &VecDeque<f32>, func: fn(f32, f32) -> f32, f_param: f32, half_w: f32, h_span: f32, h_offset: f32) -> Vec<Vec2> {
        array.iter().enumerate().map(|(i, val)|{
            let x_ratio = (i as f32) / array.len() as f32;
            let y_ratio = func(*val, f_param);
            pt2(x_ratio * half_w * 2. - half_w, y_ratio * h_span - h_offset)
        }).collect()
    }

    fn get_avg(&self) -> f32 {
        if self.averages.is_empty() {
            return 0.;
        }
        return self.avgs_sum / self.averages.len() as f32;
    }
}
