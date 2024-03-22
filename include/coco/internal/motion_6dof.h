#pragma once

#include "motion_model.h"

#include <block_optimization/standard_blocks.h>

namespace coco {

class Motion6DOF : public MotionModel {
public:
    Motion6DOF();

    virtual void getTransformMatrix(Eigen::Matrix<double, 4, 4>& T) override;
    virtual std::vector<block_optimization::BlockPtr<double>> getTransformChain() override;
    virtual std::vector<std::shared_ptr<double>> getParameterBlocksShared() override;
    virtual void resetParams() override;


protected:
    std::shared_ptr<block_optimization::Transform6DOF<double>> trafo_;
};

} // namespace coco
