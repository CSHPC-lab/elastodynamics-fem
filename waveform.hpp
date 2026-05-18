#pragma once
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"

// ============================================================
// ファイル時系列（file タイプ用）
// ============================================================
struct TimeSeries
{
    std::vector<double> t, v;

    void load(const std::string &path)
    {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("waveform file not found: " + path);
        double ti, vi;
        while (f >> ti >> vi)
        {
            t.push_back(ti);
            v.push_back(vi);
        }
        if (t.empty())
            throw std::runtime_error("empty waveform file: " + path);
    }

    double eval(double time) const
    {
        if (time <= t.front()) return v.front();
        if (time >= t.back())  return v.back();
        int lo = 0, hi = static_cast<int>(t.size()) - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            (time < t[mid] ? hi : lo) = mid;
        }
        double s = (time - t[lo]) / (t[hi] - t[lo]);
        return v[lo] + s * (v[hi] - v[lo]);
    }
};

// ============================================================
// 波形タイプと波形構造体
//
// config.txt キー（prefix = "bc_" または "force_"）:
//   {prefix}waveform      : sin / ricker / step / linear / file
//
//   sin    : {prefix}amp, {prefix}omega [rad/s], {prefix}cycles
//   ricker : {prefix}amp, {prefix}omega [rad/s]  (ピーク位置 t = 2π/omega)
//   step   : {prefix}amp
//   linear : {prefix}start [m or N], {prefix}end [m or N]
//              t=0 の値から duration 終端の値まで線形補間、以降は end で一定
//              終端時刻は config.txt の duration キーから自動取得
//   file   : {prefix}waveform_file  (2列テキスト: t [s]  value)
// ============================================================
enum class WaveformType { SIN, RICKER, STEP, LINEAR, FILE_ };

struct Waveform
{
    WaveformType type   = WaveformType::STEP;
    double amp          = 0.0;   // sin / ricker / step
    double omega        = 1.0;   // sin / ricker: 角周波数 [rad/s]
    double cycles       = 1.0;   // sin: ゼロになるまでのサイクル数
    double start_val    = 0.0;   // linear: t=0 の値
    double end_val      = 0.0;   // linear: t=duration の値
    double total_dur    = 1.0;   // linear: シミュレーション総時間（config duration から取得）
    TimeSeries ts;               // file タイプ用

    static Waveform from_config(const Config &cfg, const std::string &prefix)
    {
        Waveform w;
        std::string s = cfg.get_string(prefix + "waveform");
        if      (s == "sin")    w.type = WaveformType::SIN;
        else if (s == "ricker") w.type = WaveformType::RICKER;
        else if (s == "step")   w.type = WaveformType::STEP;
        else if (s == "linear") w.type = WaveformType::LINEAR;
        else if (s == "file")   w.type = WaveformType::FILE_;
        else
            throw std::runtime_error("unknown waveform type: " + s);

        switch (w.type)
        {
        case WaveformType::SIN:
            w.amp    = cfg.get_double(prefix + "amp");
            w.omega  = cfg.get_double(prefix + "omega");
            w.cycles = cfg.get_double(prefix + "cycles");
            break;
        case WaveformType::RICKER:
            w.amp   = cfg.get_double(prefix + "amp");
            w.omega = cfg.get_double(prefix + "omega");
            break;
        case WaveformType::STEP:
            w.amp = cfg.get_double(prefix + "amp");
            break;
        case WaveformType::LINEAR:
            w.start_val = cfg.get_double(prefix + "start");
            w.end_val   = cfg.get_double(prefix + "end");
            w.total_dur = cfg.get_double("duration");
            break;
        case WaveformType::FILE_:
            w.ts.load(cfg.get_string(prefix + "waveform_file"));
            break;
        }
        return w;
    }

    // 絶対値 [m or N]
    double eval(double t) const
    {
        switch (type)
        {
        case WaveformType::SIN:
            if (t > cycles * 2.0 * M_PI / omega) return 0.0;
            return amp * std::sin(omega * t);
        case WaveformType::RICKER:
        {
            double tau = t - 2.0 * M_PI / omega;
            double x   = omega * tau;
            return amp * (1.0 - 0.5 * x * x) * std::exp(-0.25 * x * x);
        }
        case WaveformType::STEP:
            return amp;
        case WaveformType::LINEAR:
            return start_val + (end_val - start_val) * std::min(t / total_dur, 1.0);
        case WaveformType::FILE_:
            return ts.eval(t);
        }
        return 0.0;
    }

    // 速度 df/dt [m/s or N/s]（main_cuda.cu の Newmark BC 用）
    double eval_vel(double t) const
    {
        switch (type)
        {
        case WaveformType::SIN:
            if (t > cycles * 2.0 * M_PI / omega) return 0.0;
            return amp * omega * std::cos(omega * t);
        case WaveformType::RICKER:
        {
            double tau = t - 2.0 * M_PI / omega;
            double x   = omega * tau;
            double e   = std::exp(-0.25 * x * x);
            // d/dt[(1 - x²/2)exp(-x²/4)] = ω·x·(x²/4 - 3/2)·exp(-x²/4)
            return amp * omega * x * (0.25 * x * x - 1.5) * e;
        }
        case WaveformType::STEP:
            return 0.0;
        case WaveformType::LINEAR:
            return (t < total_dur) ? (end_val - start_val) / total_dur : 0.0;
        case WaveformType::FILE_:
        {
            const double h = 1e-7;
            return (ts.eval(t + h) - ts.eval(t - h)) / (2.0 * h);
        }
        }
        return 0.0;
    }

    // 加速度 d²f/dt² [m/s²]（main_cuda.cu の Newmark BC 用）
    double eval_acc(double t) const
    {
        switch (type)
        {
        case WaveformType::SIN:
            if (t > cycles * 2.0 * M_PI / omega) return 0.0;
            return -amp * omega * omega * std::sin(omega * t);
        case WaveformType::RICKER:
        {
            double tau = t - 2.0 * M_PI / omega;
            double x   = omega * tau;
            double e   = std::exp(-0.25 * x * x);
            // d²/dt²[(1 - x²/2)exp(-x²/4)] = ω²·(-x⁴/8 + 3x²/2 - 3/2)·exp(-x²/4)
            return amp * omega * omega * (-0.125 * x * x * x * x + 1.5 * x * x - 1.5) * e;
        }
        case WaveformType::STEP:
            return 0.0;
        case WaveformType::LINEAR:
            return 0.0;
        case WaveformType::FILE_:
        {
            const double h = 1e-7;
            return (ts.eval(t + h) - 2.0 * ts.eval(t) + ts.eval(t - h)) / (h * h);
        }
        }
        return 0.0;
    }
};
