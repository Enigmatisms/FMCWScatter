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

    float norm() const {
        return sqrtf(x * x + y * y);
    }
};

// p1 is starting point, p2 is ending point
inline float getRange(const Vec2& pos, const Vec2& v, const Vec2& p1, const Vec2& p2) {
    const Vec2 neg_s2e = p1 - p2; // s2e vec (reversed order)
    if (-neg_s2e.x * v.y + neg_s2e.y * v.x >= 0.) {
        return -1e3;
    }
    const Vec2 p2s = p1 - pos;
    return (-neg_s2e.y * p2s.x + p2s.y * neg_s2e.x) / (neg_s2e.x * v.y - neg_s2e.y * v.x);
}

// Rust API, therefore we can not use std::vector but pointer
void laserRangeFinder(const Vec3& pose, const Vec2* const pts, const char* const ids, int max_num, float& min_range) {
    std::vector<float> min_ranges;
    min_ranges.resize(8, 1e9);
    const Vec2 pos(pose.x, pose.y);
    const Vec2 dir(cosf(pose.z), sinf(pose.z));
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < max_num; i++) {
        int tid = omp_get_thread_num();
        const int offset = ids[i];
        const int this_pt_id = i;
        const int next_pt_id = i + (offset < 0 ? offset : 1);
        float range = getRange(pos, dir, pts[this_pt_id], pts[next_pt_id]);
        if (range > 0. && min_ranges[tid] > range) {
            min_ranges[tid] = range; 
        }
    }
    min_range = *std::min_element(min_ranges.cbegin(), min_ranges.cend());
}

}