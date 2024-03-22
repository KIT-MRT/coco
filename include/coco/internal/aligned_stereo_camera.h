#pragma once

#include "stereo_camera_model.h"

namespace coco {

class AlignedStereoCamera : public StereoCameraModel {
public:
    virtual void computeWorldPoint(feature_tracking::StereoTracklet& tracklet) override;
    virtual std::vector<block_optimization::BlockPtr<double>> getProjectionChain(bool c1 = true) override;
    virtual std::vector<std::shared_ptr<double>> getParameterBlocksShared(bool c1 = true) override;
};

} // namespace coco
