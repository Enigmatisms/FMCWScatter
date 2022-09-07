#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Dense>

class ChirpGenerator {
public:
    ChirpGenerator(float edge_len, float band_w, float sp_int, float tof_std, float sample_std):
        edge_length(edge_len), band_width(band_w), sample_int(sp_int), tof_noise_std(tof_std), sample_noise_std(sample_std)
    {
        int total_len = static_cast<int>(edge_length / sample_int) + 1;
        pos_chirp.resize(total_len, 0.);
        neg_chirp.resize(total_len, 0.);
        // TODO: 对于发射信号而言，是否在构造时进行计算比较好？构造时计算正弦，在速度为0的时候也可以直接使用此正弦
        // 只在有速度（多普勒效应）时重新进行计算
    }
public:
    float set_delay_time(float d_time) {
        // non-fixed random seed
        static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
        static std::normal_distribution<float> tof_noise(0.0, tof_noise_std);
        delay_time = d_time + tof_noise(engine);
    }

    // 计算signal points，如果doppler_k大于0.0 则需要改变频率重新采样
    void generateSignalPoints(float doppler_k = -1.);

    // 在进行modulation时，由于Rx信号滞后于Tx信号，Tx信号的部分已经丢失，不参与计算，故Rx，Tx都需要剪切
    void signalCropping(std::vector<float>& output, size_t number) const;

    // (1) tx，rx信号相乘（调制）（2）进行低通滤波（3）FFT得到频谱（4）频谱进行有意义的变化（FFT变为频率）
    // 输入 tx 信号，rx 信号，输出频谱(spect)以及拍频（beat_f）
    void modulateThenTransform(const std::vector<float>& tx, const std::vector<float>& rx, std::vector<float>& spect, float& beat_f) const;

    void solve(float beat_pos, float beat_neg) const;
private:
    float edge_length;      // (Tc) 上升、下降沿的长度
    float band_width;       // B
    float sample_int;       // sampling interval
    float delay_time;       // ToF
    float tof_noise_std;    // noise of ToF (should be small)
    float sample_noise_std; // noise add to each time sample (gaussian std) (之后需要给此值赋值)

    std::vector<float> pos_chirp;
    std::vector<float> neg_chirp;
};