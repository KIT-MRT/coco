#pragma once

#include <ceres/autodiff_cost_function.h>

namespace coco {


template <int BLOCK_SIZE>
struct Regularization {
    Regularization(double weight, double* data) {
        std::copy(data, data + BLOCK_SIZE, data_);
        weight_ = weight;
    }

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            residuals[i] = T(weight_) * (T(data_[i]) - params[i]);
        }
        return true;
    }

    double data_[BLOCK_SIZE];
    double weight_;
};


struct SphericalProjection {
    SphericalProjection(double weight) {
        weight_ = weight;
    }

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        *residuals = weight_ * (params[0] - params[2]);
        return true;
    }

    double weight_;
};


} // namespace coco
