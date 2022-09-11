use std::collections::VecDeque;
use nannou::prelude::*;

pub struct ErrorPlotter {
    instants: VecDeque<f32>,
    averages: VecDeque<f32>,
    max_ints: usize,
    max_avgs: usize,
    ints_sum: f32,
    avgs_sum: f32
}

impl ErrorPlotter {
    pub fn new(max_ints: usize, max_avgs: usize) -> Self {
        ErrorPlotter { instants: VecDeque::new(), averages: VecDeque::new(), max_ints: max_ints, max_avgs: max_avgs, ints_sum: 0., avgs_sum: 0. }
    }

    pub fn push(&mut self, val: f32) {
        if self.instants.len() >= self.max_ints {
            let front = self.instants.pop_front().unwrap();
            self.ints_sum -= front;
        }
        self.instants.push_back(val);
        self.ints_sum += val.abs();
        let avg = self.ints_sum / self.instants.len() as f32;
        if self.averages.len() >= self.max_avgs {
            let front = self.averages.pop_front().unwrap();
            self.avgs_sum -= front;
        }
        self.averages.push_back(avg);
        self.avgs_sum += avg;
    }

    pub fn get_avg(&self) -> f32 {
        if self.averages.is_empty() {
            return 0.;
        }
        return self.avgs_sum / self.averages.len() as f32;
    }

    pub fn get_pts(&self, win_h: f32, win_w: f32) -> Vec<Vec2> {
        self.averages.iter().map(|i| {
            pt2(*i, *i)
        }).collect()
    }

}
