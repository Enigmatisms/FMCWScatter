#include <chrono>
#include <fstream>
#include "chirp_generator.hpp"

int main(int argc, char** argv) {
    // void simulateOnce(const ChirpParams& p, float* spect, float& range, float& vel, size_t& sp_size, float gt_r, float gt_v, float cutoff) {
    ChirpParams params;
    params.band_w = 600e6;
    params.base_f = 300e6;
    params.doppler_std = 1e-4;
    params.edge_len = 2e-6;        // 10us
    params.reset = true;
    params.sample_std = 0.02;
    params.sp_int = 1. / (2e9);  // 采样间隔
    params.tof_std = 2e-10;         // (25m时，飞行时间为 2 * 25 / c = 1.7e-7)
    size_t max_num = 4096;          // 1e-6 * 1.5e9 = 1500 -> 2048 补零

    std::vector<float> spect(4096);
    size_t result_size = 0;
    float range_gt = 24., range_pred = 0., vel_gt = -4.0, vel_pred = 0.;
    auto start_t = std::chrono::system_clock::now();
    simulateOnce(params, spect.data(), range_pred, vel_pred, result_size, range_gt, vel_gt, float(2 * M_PI * 50e6));
    auto interval = std::chrono::system_clock::now() - start_t;
    printf("Result: gt_range = %lf, pred range = %lf, gt_vel = %lf, pred_vel = %lf\n", range_gt, range_pred, vel_gt, vel_pred);
    printf("Time consumption: %lf ms\n", static_cast<float>(interval.count()) / 1e6);
    std::ofstream output_file("../test.txt");
    for (float val: spect) {
        output_file << val << std::endl;
    }
    output_file.close();
    return 0;
}