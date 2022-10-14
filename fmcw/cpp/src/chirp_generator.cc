#include <algorithm>
#include "../thirdparty/simple_fft/fft.hpp"
#include "../include/chirp_generator.hpp"

template <typename T>
void ChirpGenerator<T>::sendOneFrame(std::vector<T>& spectrum, T& f_pos, T& f_neg, T gt_depth, T gt_vel, T cut_off) {
    set_delay_time(gt_depth * 2. / C_VEL);
    T doppler_mv = 2 * gt_vel / wave_length;        // doppler frequency
    std::vector<T> pos_sp, neg_sp;
    processOneEdge(pos_sp, f_pos, delay_time, 1.0, doppler_mv, cut_off);
    processOneEdge(neg_sp, f_neg, delay_time, -1.0, doppler_mv, cut_off);
    spectrum.resize(pos_sp.size());
    std::transform(pos_sp.begin(), pos_sp.end(), neg_sp.begin(), spectrum.begin(), [](T v1, T v2) { return 0.5 * (v1 + v2);});
}

template <typename T>
void ChirpGenerator<T>::processOneEdge(std::vector<T>& out_spectrum, T& beat_f, T delay, T sign, T doppler_mv, T cut_off) const {
    bool is_neg = (sign < 0.0);
    std::vector<T> chirp(total_len);
    generateSignalPoints(chirp, sign, doppler_mv + (is_neg ? band_width : 0.), true);
    size_t crop_num = static_cast<size_t>(std::ceil(delay / sample_int));
    std::vector<T> cropped_tx(total_len - crop_num), cropped_rx(total_len - crop_num);
    signalCropping(chirp, crop_num);
    if (is_neg == true) {
        signalCropping(neg_chirp, cropped_tx, crop_num);
    } else {
        signalCropping(pos_chirp, cropped_tx, crop_num);
    }
    modulateThenTransform(cropped_tx, chirp, out_spectrum, beat_f, cut_off);
}

template <typename T>
size_t arg_max(const std::vector<T>& vec) {
    return static_cast<size_t>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

// 计算signal points，如果doppler_k大于0.0 则需要改变频率重新采样
template <typename T>
void ChirpGenerator<T>::generateSignalPoints(std::vector<T>& output, T sign, T doppler_mv, bool perturb) const {
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<T> doppler_noise(0.0, doppler_noise_std);
    doppler_mv += doppler_noise(engine) * band_width;
    auto sinusoid = [
        b_f = base_freq, B = band_width, t = edge_length,
        dt = sample_int, doppler_mv, s = sign, 
        c2 = this->non_lin_coeffs[0],
        c3 = this->non_lin_coeffs[1]
    ](size_t sp_i) {
        T sp_i_T = static_cast<T>(sp_i);
        T now_t = dt * sp_i_T;
        T t2 = sp_i_T * sp_i_T, t3 = sp_i_T * t2;
        T non_lin_distort = t2 * c2 + t3 * c3;
        return sin((b_f + s * B / t * now_t + doppler_mv + non_lin_distort) * now_t);
    };
    output.resize(total_len);
    std::generate(output.begin(), output.end(), [n = 0] () mutable {return n++;});
    std::transform(output.begin(), output.end(), output.begin(), sinusoid);
    if (perturb == true) {
        std::normal_distribution<T> sample_noise(0.0, sample_noise_std);
        std::transform(output.begin(), output.end(), output.begin(), [&sample_noise](T val) {return val + sample_noise(engine);});
    }
}

// 输入 tx 信号，rx 信号，输出频谱(spect)以及拍频（beat_f）
template <typename T>
void ChirpGenerator<T>::modulateThenTransform(const std::vector<T>& tx, const std::vector<T>& rx, std::vector<T>& spect, T& beat_f, T cutoff_f) const {
    const char* error_description = nullptr;
    std::vector<T> modulated(tx.size());
    std::vector<complex_t> outputs;
    std::transform(tx.begin(), tx.end(), rx.begin(), modulated.begin(), [](T v1, T v2){return v1 * v2;});

    zeroPadding(modulated);
    const T base_f = 2. * PI / static_cast<T>(modulated.size()) / sample_int;
    outputs.resize(modulated.size());
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
}

template <typename T>
typename ChirpGenerator<T>::complex_t ChirpGenerator<T>::freqButterworth4(int k, T cutoff_f, T base_f) {
    const complex_t s(0., base_f * static_cast<T>(k) / cutoff_f);
    const complex_t s2 = s * s;   
    return 1.f / (s2 + 0.765367f * s + 1.f) / (s2 + 1.847759f * s + 1.f);
}

template class ChirpGenerator<float>;
extern  "C" {
void simulateOnce(ChirpParams& p, float* spect, float& range, float& vel, float gt_r, float gt_v) {
    static ChirpGenerator<float> cg(p);
    std::vector<float> spectrum;
    float f_pos = 0., f_neg = 0.;
    if (p.reset) {
        cg.reset(p);
        p.reset = false;
    }
    cg.sendOneFrame(spectrum, f_pos, f_neg, gt_r, gt_v, p.cut_off);
    cg.solve(f_pos, f_neg, range, vel);
    float max_elem = *std::max_element(spectrum.begin(), spectrum.end());
    std::transform(spectrum.begin(), spectrum.end(), spectrum.begin(), [max_elem](float v) {return sqrtf(v / max_elem);});
    std::copy_n(spectrum.data(), spectrum.size(), spect);
}
}
