#pragma once

#include <memory>

#include <block_optimization/block.h>

#include <feature_tracking/tracklet.h>

namespace coco {

class StereoCameraModel {
public:
    virtual void computeWorldPoint(feature_tracking::StereoTracklet& tracklet) = 0;
    virtual std::vector<block_optimization::BlockPtr<double>> getProjectionChain(bool c1 = true) = 0;
    virtual std::vector<std::shared_ptr<double>> getParameterBlocksShared(bool c1 = true) = 0;

    virtual std::vector<double*> getParameterBlocksRaw(bool c1 = true) {
        std::vector<std::shared_ptr<double>> params = this->getParameterBlocksShared(c1);
        std::vector<double*> rawPtrs;
        for (std::shared_ptr<double> ptr : params) {
            rawPtrs.push_back(ptr.get());
        }
        return rawPtrs;
    }

    virtual void enableOptimization() = 0;
    virtual void disableOptimization() = 0;
    virtual bool isOptimizationEnabled() {
        return optimizationEnabled_;
    }


    virtual std::shared_ptr<double> getDistortionParams(bool c1 = true) {
        return nullptr;
    }
    virtual std::shared_ptr<double> getProjectionParams(bool c1 = true) {
        return nullptr;
    }
    virtual std::shared_ptr<double> getRotationParams(bool c1 = true) {
        return nullptr;
    }
    virtual std::shared_ptr<double> getTranslationParams(bool c1 = true) {
        return nullptr;
    }

    virtual block_optimization::BlockPtr<double> getDistortionBlock(bool c1 = true) {
        return nullptr;
    }
    virtual block_optimization::BlockPtr<double> getProjectionBlock(bool c1 = true) {
        return nullptr;
    }
    virtual block_optimization::BlockPtr<double> getRotationBlock(bool c1 = true) {
        return nullptr;
    }
    virtual block_optimization::BlockPtr<double> getTranslationBlock(bool c1 = true) {
        return nullptr;
    }


protected:
    bool optimizationEnabled_;
};

using StereoCameraModelPtr = std::shared_ptr<StereoCameraModel>;

} // namespace coco
