#include "internal/undistorted_stereo_camera.h"

namespace coco {

UndistortedStereoCamera::UndistortedStereoCamera() {
    // init camera projection block
    c1_ = std::make_shared<block_optimization::PinholeProjection<double>>("projection");
    c1_->setInternalMemory();
    // TODO: set default parameters
    c1_->getInternalMemory(0).get()[0] = 625.;
    c1_->getInternalMemory(0).get()[1] = 512.;
    c1_->getInternalMemory(0).get()[2] = 625.;
    c1_->getInternalMemory(0).get()[3] = 256.;

    c2_ = std::make_shared<block_optimization::PinholeProjection<double>>("projection");
    c2_->setInternalMemory();
    // TODO: set default parameters
    c2_->getInternalMemory(0).get()[0] = 625.;
    c2_->getInternalMemory(0).get()[1] = 512.;
    c2_->getInternalMemory(0).get()[2] = 625.;
    c2_->getInternalMemory(0).get()[3] = 256.;

    // init baseline vector add block
    baseLine_ = std::make_shared<block_optimization::Transform6DOF<double>>("baseline");
    baseLine_->setInternalMemory();
    // TODO: set default parameters
    baseLine_->getInternalMemory(1).get()[0] = -0.97;
    baseLine_->getInternalMemory(1).get()[1] = 0.;
    baseLine_->getInternalMemory(1).get()[2] = 0;

    homogenization_ = std::make_shared<block_optimization::VectorHomogenization<double>>("homogenization");

    // c1 projection chain is homogenization and projection
    projectionC1_.push_back(homogenization_);
    projectionC1_.push_back(c1_);
    parametersC1_.push_back(c1_->getInternalMemory(0));

    // c2 projection chain is added baseline, homogenization and projection

    projectionC2_.push_back(baseLine_);
    projectionC2_.push_back(homogenization_);
    projectionC2_.push_back(c2_);

    parametersC2_.push_back(baseLine_->getInternalMemory(0));
    parametersC2_.push_back(baseLine_->getInternalMemory(1));
    parametersC2_.push_back(c2_->getInternalMemory(0));
}

void UndistortedStereoCamera::computeWorldPoint(feature_tracking::StereoTracklet& tracklet) {
    if (tracklet.empty()) {
        // don't do anything if tracklet is empty
        return;
    }
    double& fx1 = c1_->getInternalMemory(0).get()[0];
    double& cx1 = c1_->getInternalMemory(0).get()[1];
    double& fy1 = c1_->getInternalMemory(0).get()[2];
    double& cy1 = c1_->getInternalMemory(0).get()[3];
    double& fx2 = c2_->getInternalMemory(0).get()[0];
    double& cx2 = c2_->getInternalMemory(0).get()[1];
    double& fy2 = c2_->getInternalMemory(0).get()[2];
    double& cy2 = c2_->getInternalMemory(0).get()[3];

    feature_tracking::StereoMatch& match = tracklet.front();
    match.x_ = std::make_shared<feature_tracking::WorldPoint>();
    feature_tracking::WorldPoint& wp = *match.x_;
    //  wp[0] = (match.p1_.u_ - cx)/fx;
    //  wp[1] = (match.p1_.v_ - cy)/fy;

    Eigen::Vector3d y;
    y(0) = (match.p1_.u_ - cx1) / fx1;
    y(1) = (match.p1_.v_ - cy1) / fy1;
    y(2) = 1.;

    Eigen::Vector3d yPrime;
    yPrime(0) = (match.p2_.u_ - cx2) / fx2;
    yPrime(1) = (match.p2_.v_ - cy2) / fy2;
    yPrime(2) = 1.;


    Eigen::Matrix<double, 3, 3> R;
    baseLine_->getRotationMatrix(R);
    Eigen::Vector3d r1 = R.block<1, 3>(0, 0);
    Eigen::Vector3d r3 = R.block<1, 3>(2, 0);


    Eigen::Vector3d t;
    baseLine_->getTranslation(t);

    Eigen::Matrix<double, 4, 4> T = Eigen::Matrix<double, 4, 4>::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    T = T.inverse();
    R = T.block<3, 3>(0, 0);
    t = T.block<3, 1>(0, 3);


    double z = (r1 - yPrime(0) * r3).dot(t) / (r1 - yPrime(0) * r3).dot(y);

    wp[0] = y(0) * z;
    wp[1] = y(1) * z;
    wp[2] = z;
}

std::vector<block_optimization::BlockPtr<double>> UndistortedStereoCamera::getProjectionChain(bool c1) {
    if (c1) {
        return projectionC1_;
    }
    return projectionC2_;
}

std::vector<std::shared_ptr<double>> UndistortedStereoCamera::getParameterBlocksShared(bool c1) {
    if (c1) {
        return parametersC1_;
    }
    return parametersC2_;
}

void UndistortedStereoCamera::enableOptimization() {
    this->optimizationEnabled_ = true;
    c1_->enableOptimization();
    c2_->enableOptimization();
    baseLine_->enableOptimization();
}

void UndistortedStereoCamera::disableOptimization() {
    this->optimizationEnabled_ = false;
    c1_->disableOptimization();
    c2_->disableOptimization();
    baseLine_->disableOptimization();
}

std::shared_ptr<double> UndistortedStereoCamera::getProjectionParams(bool c1) {
    if (c1) {
        return c1_->getInternalMemory(0);
    }
    return c2_->getInternalMemory(0);
}
std::shared_ptr<double> UndistortedStereoCamera::getRotationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(0);
}
std::shared_ptr<double> UndistortedStereoCamera::getTranslationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(1);
}


block_optimization::BlockPtr<double> UndistortedStereoCamera::getProjectionBlock(bool c1) {
    if (c1) {
        return c1_;
    }
    return c2_;
}
block_optimization::BlockPtr<double> UndistortedStereoCamera::getRotationBlock(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->rotation_;
}
block_optimization::BlockPtr<double> UndistortedStereoCamera::getTranslationBlock(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->translation_;
}

} // namespace coco
