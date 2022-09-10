#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <complex>
#include <cstring>

constexpr float PI = 3.141592653589793;
constexpr float C_VEL = 299792458;  // speed of light

struct ChirpParams {
    float base_f;
    float edge_len;
    float band_w;
    float sp_int;
    float tof_std;
    float doppler_std;
    float sample_std;
    float cut_off;
    bool reset;
};

template <typename T>
class ChirpGenerator {
using complex_t = std::complex<T>;
public:
    ChirpGenerator(const ChirpParams& p)
    {
        reset(p);
        
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
public:
    void reset(const ChirpParams& p) {
        base_freq = p.base_f;
        edge_length = p.edge_len;
        band_width = p.band_w; 
        sample_int = p.sp_int; 
        tof_noise_std = p.tof_std;
        doppler_noise_std = p.doppler_std; 
        sample_noise_std = p.sample_std;

        total_len = static_cast<size_t>(edge_length / sample_int) + 1;
        pos_chirp.resize(total_len, 0.);
        neg_chirp.resize(total_len, 0.);
        generateSignalPoints(pos_chirp, 1.0);
        generateSignalPoints(neg_chirp, -1.0, band_width);
    }

    void sendOneFrame(std::vector<T>& spectrum, T& f_pos, T& f_neg, T gt_depth, T gt_vel, T cut_off);      // simulation

    void solve(T beat_pos, T beat_neg, T& range, T& speed) const {
        range = C_VEL * edge_length / (8. * band_width) * (beat_pos + beat_neg);
        speed = wave_length / 4. * (beat_neg - beat_pos);
    }
private:
    // default cut off frequency is 1MHz
    void processOneEdge(std::vector<T>& out_spectrum, T& beat_f, T delay, T sign, T doppler_mv, T cut_off = 1e7) const;

    void set_delay_time(T d_time) {        // should be called every evaluation
        // non-fixed random seed
        std::normal_distribution<T> tof_noise(0.0, tof_noise_std);
        delay_time = d_time + tof_noise(engine);
    }

    // 计算signal points，如果doppler_k大于0.0 则需要改变频率重新采样
    void generateSignalPoints(std::vector<T>& output, T sign, T doppler_mv = 0.0, bool perturb = false) const;

    // (1) tx，rx信号相乘（调制）（2）进行低通滤波（3）FFT得到频谱（4）频谱进行有意义的变化（FFT变为频率）
    // 输入 tx 信号，rx 信号，输出频谱(spect)以及拍频（beat_f）
    void modulateThenTransform(const std::vector<T>& tx, const std::vector<T>& rx, std::vector<T>& spect, T& beat_f, T cutoff_f) const;

    static void signalCropping(const std::vector<T>& input, std::vector<T>& output, size_t number) {
        output.assign(input.begin() + number, input.end());
    }

    static void signalCropping(std::vector<T>& inout, size_t number) {
        size_t new_size = inout.size() - number;
        inout.resize(new_size);
    }

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
    std::default_random_engine engine;
};
// Rust API


// Rust API

