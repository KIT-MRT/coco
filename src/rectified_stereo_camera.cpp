#include "internal/rectified_stereo_camera.h"

namespace coco {

RectifiedStereoCamera::RectifiedStereoCamera() {
    // init camera projection block
    c_ = std::make_shared<block_optimization::PinholeProjection<double>>("projection");
    c_->setInternalMemory();
    // TODO: set default parameters
    c_->getInternalMemory(0).get()[0] = 625.;
    c_->getInternalMemory(0).get()[1] = 512.;
    c_->getInternalMemory(0).get()[2] = 625.;
    c_->getInternalMemory(0).get()[3] = 256.;


    // init baseline vector add block
    baseLine_ = std::make_shared<block_optimization::VectorAdd<double, 3>>("baseline");
    baseLine_->setInternalMemory();
    // TODO: set default parameters
    baseLine_->getInternalMemory(0).get()[0] = -0.97;
    baseLine_->getInternalMemory(0).get()[1] = 0.;
    baseLine_->getInternalMemory(0).get()[2] = 0;


    homogenization_ = std::make_shared<block_optimization::VectorHomogenization<double>>("homogenization");

    // c1 projection chain is homogenization and projection
    projectionC1_.push_back(homogenization_);
    projectionC1_.push_back(c_);
    parametersC1_.push_back(c_->getInternalMemory(0));

    // c2 projection chain is added baseline, homogenization and projection
    projectionC2_.push_back(baseLine_);
    projectionC2_.push_back(homogenization_);
    projectionC2_.push_back(c_);
    parametersC2_.push_back(baseLine_->getInternalMemory(0));
    parametersC2_.push_back(c_->getInternalMemory(0));
}

void RectifiedStereoCamera::computeWorldPoint(feature_tracking::StereoTracklet& tracklet) {
    if (tracklet.empty()) {
        // don't do anything if tracklet is empty
        return;
    }

    // read parameters
    double& f = c_->getInternalMemory(0).get()[0];
    double& cx = c_->getInternalMemory(0).get()[1];
    double& cy = c_->getInternalMemory(0).get()[3];

    double b = std::sqrt(std::pow(baseLine_->getInternalMemory(0).get()[0], 2) +
                         std::pow(baseLine_->getInternalMemory(0).get()[1], 2) +
                         std::pow(baseLine_->getInternalMemory(0).get()[2], 2));

    // for sake of legibility
    feature_tracking::StereoMatch& match = tracklet.front();
    double d = match.p1_.u_ - match.p2_.u_;

    match.x_ = std::make_shared<feature_tracking::WorldPoint>();
    feature_tracking::WorldPoint& wp = *match.x_;

    double z = b / std::max(d, std::numeric_limits<double>::epsilon());
    // NOTE: z is scaled by 1/f right now
    // x = (p1_x - cx)/f*z
    wp[0] = (match.p1_.u_ - cx) * z;
    // y = (p1_y - cy)/f*z
    wp[1] = (match.p1_.v_ - cy) * z;
    wp[2] = z * f;
}

std::vector<block_optimization::BlockPtr<double>> RectifiedStereoCamera::getProjectionChain(bool c1) {
    if (c1) {
        return projectionC1_;
    }
    return projectionC2_;
}

std::vector<std::shared_ptr<double>> RectifiedStereoCamera::getParameterBlocksShared(bool c1) {
    if (c1) {
        return parametersC1_;
    }
    return parametersC2_;
}

void RectifiedStereoCamera::enableOptimization() {
    this->optimizationEnabled_ = true;
    c_->enableOptimization();
    baseLine_->enableOptimization();
}

void RectifiedStereoCamera::disableOptimization() {
    this->optimizationEnabled_ = false;
    c_->disableOptimization();
    baseLine_->disableOptimization();
}

std::shared_ptr<double> RectifiedStereoCamera::getProjectionParams(bool c1) {
    return c_->getInternalMemory(0);
}
std::shared_ptr<double> RectifiedStereoCamera::getRotationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(0);
}
std::shared_ptr<double> RectifiedStereoCamera::getTranslationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(1);
}

block_optimization::BlockPtr<double> RectifiedStereoCamera::getProjectionBlock(bool c1) {
    return c_;
}
block_optimization::BlockPtr<double> RectifiedStereoCamera::getRotationBlock(bool c1) {
    return nullptr;
}
block_optimization::BlockPtr<double> RectifiedStereoCamera::getTranslationBlock(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_;
}

} // namespace coco
