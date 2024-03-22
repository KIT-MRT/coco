#pragma once

#include <memory>

#include <block_optimization/block.h>

#include <Eigen/Core>

namespace coco {

class MotionModel {
public:
    virtual void getTransformMatrix(Eigen::Matrix<double, 4, 4>& T) = 0;
    virtual std::vector<block_optimization::BlockPtr<double>> getTransformChain() = 0;
    virtual std::vector<std::shared_ptr<double>> getParameterBlocksShared() = 0;
    virtual void resetParams() = 0;


    virtual std::vector<double*> getParameterBlocksRaw() {
        std::vector<std::shared_ptr<double>> params = this->getParameterBlocksShared();
        std::vector<double*> rawPtrs;
        for (std::shared_ptr<double> ptr : params) {
            rawPtrs.push_back(ptr.get());
        }
        return rawPtrs;
    }
};

using MotionModelPtr = std::shared_ptr<MotionModel>;

} // namespace coco
