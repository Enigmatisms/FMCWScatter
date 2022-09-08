#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Dense>

template <typename T>
class ChirpGenerator {
using complex_t = std::complex<T>;
public:
    ChirpGenerator(T base_f, T edge_len, T band_w, T sp_int, T tof_std, T doppler_std, T sample_std):
        base_freq(base_f), edge_length(edge_len), band_width(band_w), 
        sample_int(sp_int), tof_noise_std(tof_std), doppler_noise_std(doppler_std), sample_noise_std(sample_std)
    {
        total_len = static_cast<size_t>(edge_length / sample_int) + 1;
        pos_chirp.resize(total_len, 0.);
        neg_chirp.resize(total_len, 0.);
        // TODO: 对于发射信号而言，是否在构造时进行计算比较好？构造时计算正弦，在速度为0的时候也可以直接使用此正弦
        // 只在有速度（多普勒效应）时重新进行计算
    }
public:
    void sendOneFrame(T gt_depth);      // simulation
private:
    T set_delay_time(T d_time) {        // should be called every evaluation
        // non-fixed random seed
        static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
        static std::normal_distribution<T> tof_noise(0.0, tof_noise_std);
        delay_time = d_time + tof_noise(engine);
    }

    // 计算signal points，如果doppler_k大于0.0 则需要改变频率重新采样
    void generateSignalPoints(std::vector<T>& output, T doppler_mv = 0.0, bool perturb = false) const;

    // (1) tx，rx信号相乘（调制）（2）进行低通滤波（3）FFT得到频谱（4）频谱进行有意义的变化（FFT变为频率）
    // 输入 tx 信号，rx 信号，输出频谱(spect)以及拍频（beat_f）
    void modulateThenTransform(const std::vector<T>& tx, const std::vector<T>& rx, std::vector<T>& spect, T& beat_f, T cutoff_f) const;

    void solve(T beat_pos, T beat_neg, T& range, T& speed) const {
        range = c_vel * edge_length / (8. * band_width) * (beat_pos + beat_neg);
        speed = wave_length / 4. * (beat_neg - beat_pos);
    }

    // 在进行modulation时，由于Rx信号滞后于Tx信号，Tx信号的部分已经丢失，不参与计算，故Rx，Tx都需要剪切
    static void signalCropping(const std::vector<T>& input, std::vector<T>& output, size_t number, bool front = true);

    static void zeroPadding(std::vector<T>& inout) {
        size_t len = inout.size();
        if ((len & (len - 1)) == 0) {           // 如果长度已经是2的幂次，那么不需要进行padding
            return;
        }
        size_t ub = static_cast<size_t>(std::ceil(log2(inout.size())));
        inout.resize(pow(2, ub));
    }

    static complex_t freqButterworth4(int k, T cutoff_f, T base_f);
private:
    T base_freq;            // Base frequency
    T edge_length;          // (Tc) 上升、下降沿的长度
    T band_width;           // B
    T sample_int;           // sampling interval
    T delay_time;           // ToF
    T tof_noise_std;        // noise of ToF (should be small)
    T doppler_noise_std;    // Doppler effect noise std
    T sample_noise_std;     // noise add to each time sample (gaussian std) (之后需要给此值赋值)
    T wave_length = 1550e-9;// wave length of LiDAR (1550 nm)

    size_t total_len;
    std::vector<T> pos_chirp;
    std::vector<T> neg_chirp;
};