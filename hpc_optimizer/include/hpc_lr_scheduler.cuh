// =============================================================================
// HPC CUDA Optimizer Library
// hpc_lr_scheduler.cuh  –  LR schedulers for HPC training
//
// All schedulers are host-side, zero-device-overhead.
// They hold a float& to the optimizer config's lr field and update it in-place.
//
// Schedulers:
//   WarmupCosine      –  linear warmup → cosine decay (standard for large LM)
//   WarmupLinear      –  linear warmup → linear decay (BERT pre-training)
//   WarmupConstant    –  linear warmup → constant (fine-tuning)
//   CyclicLR          –  triangular / triangular2 / exp_range (Smith 2017)
//   OneCycleLR        –  1-cycle policy (fast convergence, Smith & Topin 2019)
//   PolynomialLR      –  polynomial decay with optional warmup
//   ReduceLROnPlateau –  adaptive: reduce on metric stagnation
// =============================================================================
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace hpc_opt {

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float step() = 0;
    virtual float get_lr() const = 0;
};

// ---------------------------------------------------------------------------
// WarmupCosine  (linear warmup + cosine annealing)
//   Steps 0..W:    lr = base * t/W
//   Steps W..T:    lr = eta_min + 0.5*(base-eta_min)*(1+cos(π*(t-W)/(T-W)))
// ---------------------------------------------------------------------------
class WarmupCosineLR : public LRScheduler {
public:
    WarmupCosineLR(float& lr, int warmup, int total, float eta_min = 0.0f)
        : lr_(lr), base_(lr), W_(warmup), T_(total), eta_(eta_min) {}

    float step() override {
        ++t_;
        if (t_ <= W_) {
            lr_ = base_ * static_cast<float>(t_) / static_cast<float>(W_);
        } else {
            float p = static_cast<float>(t_ - W_) / static_cast<float>(T_ - W_);
            lr_ = eta_ + 0.5f * (base_ - eta_) *
                  (1.0f + cosf(static_cast<float>(M_PI) * p));
        }
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_;
    float  base_, eta_;
    int    W_, T_, t_ = 0;
};

// ---------------------------------------------------------------------------
// WarmupLinear  –  linear warmup then linear decay to eta_min
// ---------------------------------------------------------------------------
class WarmupLinearLR : public LRScheduler {
public:
    WarmupLinearLR(float& lr, int warmup, int total, float eta_min = 0.0f)
        : lr_(lr), base_(lr), W_(warmup), T_(total), eta_(eta_min) {}

    float step() override {
        ++t_;
        if (t_ <= W_) {
            lr_ = base_ * static_cast<float>(t_) / W_;
        } else {
            float p = static_cast<float>(t_ - W_) / (T_ - W_);
            lr_ = base_ - (base_ - eta_) * p;
            lr_ = std::max(lr_, eta_);
        }
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_;
    float  base_, eta_;
    int    W_, T_, t_ = 0;
};

// ---------------------------------------------------------------------------
// WarmupConstant  –  warmup then hold base_lr
// ---------------------------------------------------------------------------
class WarmupConstantLR : public LRScheduler {
public:
    WarmupConstantLR(float& lr, int warmup)
        : lr_(lr), base_(lr), W_(warmup) {}

    float step() override {
        ++t_;
        if (t_ <= W_) lr_ = base_ * static_cast<float>(t_) / W_;
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_; float base_; int W_, t_ = 0;
};

// ---------------------------------------------------------------------------
// PolynomialLR
//   lr_t = (base - eta_min) * ((1 - t/T) ^ power) + eta_min
//   Optional warmup prepended.
// ---------------------------------------------------------------------------
class PolynomialLR : public LRScheduler {
public:
    PolynomialLR(float& lr, int total, float power = 1.0f,
                 float eta_min = 0.0f, int warmup = 0)
        : lr_(lr), base_(lr), T_(total), pow_(power),
          eta_(eta_min), W_(warmup) {}

    float step() override {
        ++t_;
        if (t_ <= W_) {
            lr_ = base_ * static_cast<float>(t_) / W_;
        } else {
            int t_adj = t_ - W_;
            int T_adj = T_  - W_;
            float p = static_cast<float>(t_adj) / static_cast<float>(T_adj);
            p = std::min(p, 1.0f);
            lr_ = (base_ - eta_) * powf(1.0f - p, pow_) + eta_;
        }
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_; float base_, eta_, pow_; int T_, W_, t_ = 0;
};

// ---------------------------------------------------------------------------
// OneCycleLR  (Smith & Topin 2019)
//   Phase 1 [0..pct*T]:  cosine from base/div_factor → base
//   Phase 2 [pct*T..T]:  cosine from base → base/final_div_factor
// ---------------------------------------------------------------------------
class OneCycleLR : public LRScheduler {
public:
    OneCycleLR(float& lr,
               float max_lr,
               int   total_steps,
               float pct_start       = 0.3f,
               float div_factor      = 25.0f,
               float final_div_factor= 1e4f)
        : lr_(lr),
          max_lr_(max_lr),
          base_lr_(max_lr / div_factor),
          min_lr_ (max_lr / final_div_factor),
          T_(total_steps),
          T1_(static_cast<int>(total_steps * pct_start))
    {
        lr_ = base_lr_;
    }

    float step() override {
        ++t_;
        if (t_ <= T1_) {
            float p = static_cast<float>(t_) / T1_;
            lr_ = base_lr_ + 0.5f * (max_lr_ - base_lr_) * (1.0f - cosf(static_cast<float>(M_PI) * p));
        } else {
            int t2 = t_ - T1_, T2 = T_ - T1_;
            float p = static_cast<float>(t2) / T2;
            lr_ = min_lr_ + 0.5f * (max_lr_ - min_lr_) * (1.0f + cosf(static_cast<float>(M_PI) * p));
        }
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_;
    float  max_lr_, base_lr_, min_lr_;
    int    T_, T1_, t_ = 0;
};

// ---------------------------------------------------------------------------
// CyclicLR  –  triangular / triangular2 / exp_range
// ---------------------------------------------------------------------------
class CyclicLR : public LRScheduler {
public:
    enum class Mode { Triangular, Triangular2, ExpRange };

    CyclicLR(float& lr,
             float base_lr, float max_lr,
             int   step_size_up   = 2000,
             int   step_size_down = -1,
             Mode  mode = Mode::Triangular,
             float gamma = 1.0f)
        : lr_(lr), base_(base_lr), max_(max_lr),
          up_(step_size_up),
          down_(step_size_down < 0 ? step_size_up : step_size_down),
          mode_(mode), gamma_(gamma)
    {
        lr_ = base_lr;
        cycle_size_ = up_ + down_;
    }

    float step() override {
        int pos_in_cycle = t_ % cycle_size_;
        float x;
        if (pos_in_cycle < up_) {
            x = static_cast<float>(pos_in_cycle) / up_;
        } else {
            x = 1.0f - static_cast<float>(pos_in_cycle - up_) / down_;
        }

        float scale = 1.0f;
        int cycle = t_ / cycle_size_;
        switch (mode_) {
            case Mode::Triangular:  scale = 1.0f; break;
            case Mode::Triangular2: scale = 1.0f / powf(2.0f, static_cast<float>(cycle)); break;
            case Mode::ExpRange:    scale = powf(gamma_, static_cast<float>(t_)); break;
        }

        lr_ = base_ + (max_ - base_) * x * scale;
        ++t_;
        return lr_;
    }

    float get_lr() const override { return lr_; }

private:
    float& lr_; float base_, max_, gamma_;
    int    up_, down_, cycle_size_, t_ = 0;
    Mode   mode_;
};

// ---------------------------------------------------------------------------
// ReduceLROnPlateau
// ---------------------------------------------------------------------------
class ReduceLROnPlateau : public LRScheduler {
public:
    ReduceLROnPlateau(float& lr,
                      float  factor    = 0.5f,
                      int    patience  = 5,
                      float  min_lr    = 1e-7f,
                      float  threshold = 1e-4f,
                      bool   maximize  = false)
        : lr_(lr), factor_(factor), patience_(patience),
          min_lr_(min_lr), threshold_(threshold), maximize_(maximize),
          best_(maximize ? -1e30f : 1e30f) {}

    // Call with the monitored metric (val_loss or val_acc)
    float step(float metric) {
        bool improved = maximize_
            ? (metric > best_ + threshold_)
            : (metric < best_ - threshold_);

        if (improved) { best_ = metric; bad_ = 0; }
        else if (++bad_ >= patience_) {
            lr_ = std::max(lr_ * factor_, min_lr_);
            bad_ = 0;
        }
        return lr_;
    }

    float step() override { return lr_; }   // base no-op
    float get_lr() const override { return lr_; }

private:
    float& lr_;
    float  factor_, min_lr_, threshold_, best_;
    int    patience_, bad_ = 0;
    bool   maximize_;
};

} // namespace hpc_opt
