#include "internal/motion_6dof.h"

namespace coco {

Motion6DOF::Motion6DOF() {
    trafo_ = std::make_shared<block_optimization::Transform6DOF<double>>("motion");
    trafo_->setInternalMemory();
}

void Motion6DOF::getTransformMatrix(Eigen::Matrix<double, 4, 4>& T) {
    T.setZero();
    Eigen::Matrix<double, 3, 3> R;
    trafo_->getRotationMatrix(R);
    T.block<3, 3>(0, 0) = R;
    Eigen::Vector3d t;
    trafo_->getTranslation(t);
    T.block<3, 1>(0, 3) = t;
    T(3, 3) = 1.;
}

std::vector<block_optimization::BlockPtr<double>> Motion6DOF::getTransformChain() {
    return std::vector<block_optimization::BlockPtr<double>>{trafo_};
}

std::vector<std::shared_ptr<double>> Motion6DOF::getParameterBlocksShared() {
    return trafo_->getInternalMemory();
}

void Motion6DOF::resetParams() {
    // reset rotation
    double* dat = trafo_->getInternalMemory(0).get();
    std::fill(dat, dat + 3, 0.);
    // reset translation
    dat = trafo_->getInternalMemory(1).get();
    std::fill(dat, dat + 3, 0.);
}

} // namespace coco
