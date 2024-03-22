#pragma once

#include "stereo_camera_model.h"

#include <block_optimization/vision_blocks.h>

namespace coco {

class UndistortedStereoCamera : public StereoCameraModel {
public:
    UndistortedStereoCamera();

    virtual void computeWorldPoint(feature_tracking::StereoTracklet& tracklet) override;
    virtual std::vector<block_optimization::BlockPtr<double>> getProjectionChain(bool c1 = true) override;
    virtual std::vector<std::shared_ptr<double>> getParameterBlocksShared(bool c1 = true) override;
    virtual void enableOptimization() override;
    virtual void disableOptimization() override;

    virtual std::shared_ptr<double> getProjectionParams(bool c1 = true) override;
    virtual std::shared_ptr<double> getRotationParams(bool c1 = true) override;
    virtual std::shared_ptr<double> getTranslationParams(bool c1 = true) override;

    virtual block_optimization::BlockPtr<double> getProjectionBlock(bool c1 = true) override;
    virtual block_optimization::BlockPtr<double> getRotationBlock(bool c1 = true) override;
    virtual block_optimization::BlockPtr<double> getTranslationBlock(bool c1 = true) override;

protected:
    std::shared_ptr<block_optimization::PinholeProjection<double>>
        c1_; ///< rectification makes sure the projection is the same for both cameras
    std::shared_ptr<block_optimization::PinholeProjection<double>>
        c2_; ///< rectification makes sure the projection is the same for both cameras
    std::shared_ptr<block_optimization::Transform6DOF<double>>
        baseLine_; ///< rectification compensates relative orientation, extrinsics is only the baseline
    std::shared_ptr<block_optimization::VectorHomogenization<double>> homogenization_;
    std::vector<block_optimization::BlockPtr<double>> projectionC1_;
    std::vector<block_optimization::BlockPtr<double>> projectionC2_;

    std::vector<std::shared_ptr<double>> parametersC1_;
    std::vector<std::shared_ptr<double>> parametersC2_;
};

} // namespace coco
