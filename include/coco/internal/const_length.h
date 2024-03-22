#pragma once

namespace coco {

struct ConstantLength {
    template <typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
        const T norm_x = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (int i = 0; i < 3; i++) {
            x_plus_delta[i] = x[i] + delta[i];
        }
        const T norm_x_plus_delta = sqrt(x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] +
                                         x_plus_delta[2] * x_plus_delta[2]);
        const T scale = norm_x / norm_x_plus_delta;
        x_plus_delta[0] = x_plus_delta[0] * scale;
        x_plus_delta[1] = x_plus_delta[1] * scale;
        x_plus_delta[2] = x_plus_delta[2] * scale;
        return true;
    }
};

} // namespace coco
