#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" {

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec2 {
    float x = 0.f;
    float y = 0.f;
    Vec2(): x(0), y(0) {}
    Vec2(float x, float y): x(x), y(y) {}
    Vec2(const Vec2& vec): x(vec.x), y(vec.y) {}

    Vec2 operator-(const Vec2& p) const {
        return Vec2(x - p.x, y - p.y);         // Return value optimized?
    }

    float dot(const Vec2& p) const {
        return x * p.x + y * p.y;
    }
};

// p1 is starting point, p2 is ending point
// 7 additions, 6 mults, 2 divs, 4 bool ops
inline float getRange(const Vec2& pos, const Vec2& v_perp, const Vec2& p1, const Vec2& p2) {
    const Vec2& s2e = p2 - p1;
    const Vec2& obs_v = pos - p1;
    float D = v_perp.dot(s2e);
    if (D < 0.) {
        return -1e-3;
    }
    float alpha = v_perp.dot(obs_v) / D;
    if (alpha >= 1. || alpha <= 0.) {
        return -1e-3;
    }
    return (-s2e.y * obs_v.x + s2e.x * obs_v.y) / D;
}

// Rust API, therefore we can not use std::vector but pointer
void laserRangeFinder(const Vec3& pose, const Vec2* const pts, const char* const ids, int max_num, float& min_range) {
    std::vector<float> min_ranges;
    min_ranges.resize(8, 1e5);
    const Vec2 pos(pose.x, pose.y);
    const Vec2 v_perp(-sinf(pose.z), cosf(pose.z));         // direction vector (rotated 90 deg)
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < max_num; i++) {
        const int offset = ids[i];
        const int this_pt_id = i;
        const int next_pt_id = i + (offset < 0 ? offset : 1);
        float range = getRange(pos, v_perp, pts[this_pt_id], pts[next_pt_id]);
        int tid = omp_get_thread_num();
        if (range > 0. && range < min_ranges[tid]) {
            min_ranges[tid] = range; 
        }
    }
    min_range = *std::min_element(min_ranges.begin(), min_ranges.end());
}

}