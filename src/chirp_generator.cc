#include <algorithm>
#include "simple_fft/fft.h"
#include "chirp_generator.hpp"

constexpr float PI = 3.141592653589793;
constexpr float c_vel = 299792458;  // speed of light

template <typename T>
size_t arg_max(const std::vector<T>& vec) {
    return static_cast<size_t>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

// 计算signal points，如果doppler_k大于0.0 则需要改变频率重新采样
template <typename T>
void ChirpGenerator<T>::generateSignalPoints(std::vector<T>& output, T doppler_mv, bool perturb) const {
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<T> doppler_noise(0.0, tof_noise_std);
    doppler_mv += doppler_noise(engine);
    auto sinusoid = [b_f = base_freq, B = band_width, t = edge_length, dt = sample_int, doppler_mv](size_t sp_i) {
        T now_t = dt * static_cast<T>(sp_i);
        return sin((b_f + B / T * now_t + doppler_mv) * now_t);
    };
    output.resize(total_len);
    std::generate(output.begin(), output.end(), [n = 0] () mutable {return n++;});
    std::transform(output.begin(), output.end(), output.begin(), sinusoid);
    if (perturb == true) {
        std::normal_distribution<T> sample_noise(0.0, sample_noise_std);
        std::transform(output.begin(), output.end(), output.begin(), [&sample_noise, &engine](T val) {return val + sample_noise(engine);});
    }
}

// 在进行modulation时，由于Rx信号滞后于Tx信号，Tx信号的部分已经丢失，不参与计算，故Rx，Tx都需要剪切
template <typename T>
void ChirpGenerator<T>::signalCropping(const std::vector<T>& input, std::vector<T>& output, size_t number, bool front) {
    if (front == true) {
        output.assign(input.begin() + number, input.end());
    } else {
        output.assign(input.begin(), input.end() - number);
    }
}

// (1) tx，rx信号相乘（调制）（2）进行低通滤波（3）FFT得到频谱（4）频谱进行有意义的变化（FFT变为频率）
// 输入 tx 信号，rx 信号，输出频谱(spect)以及拍频（beat_f）
template <typename T>
void ChirpGenerator<T>::modulateThenTransform(const std::vector<T>& tx, const std::vector<T>& rx, std::vector<T>& spect, T& beat_f, T cutoff_f) const {
    const char* error_description = nullptr;
    const T base_f = 2 * PI / static_cast<T>(tx.size());
    std::vector<T> modulated(tx.size());
    std::vector<complex_t> outputs;

    std::transform(tx.begin(), tx.end(), rx.begin(), modulated.begin(), [](T v1, T v2){return v1 * v2;});
    zeroPadding(modulated);
    size_t signal_len = modulated.size();
    simple_fft::FFT(modulated, outputs, modulated.size(), error_description);

    if (cutoff_f > 1e-9) {
        std::vector<complex_t> butterworth_lp(modulated.size());
        std::generate(butterworth_lp.begin(), butterworth_lp.end(), 
            [k = 0, cutoff_f, base_f]() mutable {
                return freqButterworth4(k++, cutoff_f, base_f);
            }
        );
        std::transform(outputs.begin(), outputs.end(), butterworth_lp.begin(), outputs.begin(), [](const auto& v1, const auto& v2){return v1 * v2;});
    }
    outputs.resize(1 + (signal_len >> 1));

    auto get_amp_func = [](const complex_t& c) {return sqrt(pow(c.real(), 2) + pow(c.imag(), 2));};
    spect.resize(signal_len);
    std::transform(outputs.begin(), outputs.end(), spect.begin(), get_amp_func);
    size_t k_num = arg_max(spect);
    beat_f = base_f * static_cast<T>(k_num);
    // TODO: 有意义的变换
}

template <typename T>
void ChirpGenerator<T>::zeroPadding(std::vector<T>& inout) {
    size_t ub = static_cast<size_t>(std::ceil(log2(inout.size())));
    inout.resize(pow(2, ub));
}

template <typename T>
ChirpGenerator<T>::complex_t ChirpGenerator<T>::freqButterworth4(int k, T cutoff_f, T base_f) {
    const complex_t s(0., base_f * static_cast<T>(k) / cutoff_f);
    const complex_t s2 = s * s;   
    return 1. / (s2 + 0.765367 * s + 1) / (s2 + 1.847759 * s + 1);
}
