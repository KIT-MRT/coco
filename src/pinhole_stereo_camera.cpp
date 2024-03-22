#include "internal/pinhole_stereo_camera.h"

namespace coco {

PinholeStereoCamera::PinholeStereoCamera() {
    // init camera projection block
    c1_ = std::make_shared<block_optimization::PinholeProjection<double>>("projection1");
    c1_->setInternalMemory();
    // TODO: set default parameters
    c1_->getInternalMemory(0).get()[0] = 300.;
    c1_->getInternalMemory(0).get()[1] = 512.;
    c1_->getInternalMemory(0).get()[2] = 300.;
    c1_->getInternalMemory(0).get()[3] = 256.;

    c2_ = std::make_shared<block_optimization::PinholeProjection<double>>("projection2");
    c2_->setInternalMemory();
    // TODO: set default parameters
    c2_->getInternalMemory(0).get()[0] = 300.;
    c2_->getInternalMemory(0).get()[1] = 512.;
    c2_->getInternalMemory(0).get()[2] = 300.;
    c2_->getInternalMemory(0).get()[3] = 256.;

    d1_ = std::make_shared<block_optimization::PolyDistortion<double>>("distortion1");
    d1_->setInternalMemory();
    d2_ = std::make_shared<block_optimization::PolyDistortion<double>>("distortion2");
    d2_->setInternalMemory();


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
    projectionC1_.push_back(d1_);
    projectionC1_.push_back(c1_);
    parametersC1_.push_back(d1_->getInternalMemory(0));
    parametersC1_.push_back(c1_->getInternalMemory(0));

    // c2 projection chain is added baseline, homogenization and projection

    projectionC2_.push_back(baseLine_);
    projectionC2_.push_back(homogenization_);
    projectionC2_.push_back(d2_);
    projectionC2_.push_back(c2_);

    parametersC2_.push_back(baseLine_->getInternalMemory(0));
    parametersC2_.push_back(baseLine_->getInternalMemory(1));
    parametersC2_.push_back(d2_->getInternalMemory(0));
    parametersC2_.push_back(c2_->getInternalMemory(0));
}

void PinholeStereoCamera::computeWorldPoint(feature_tracking::StereoTracklet& tracklet) {
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

    if (!inverseDistParams1_ || !inverseDistParams2_) {
        this->makeInvDistParams();
    }

    double rSq = std::pow(y(0), 2) + std::pow(y(1), 2);
    double dist = 1. + rSq * inverseDistParams1_.get()[0] + std::pow(rSq, 2) * inverseDistParams1_.get()[1] +
                  std::pow(rSq, 3) * inverseDistParams1_.get()[4];

    y(0) *= dist;
    y(1) *= dist;

    rSq = std::pow(yPrime(0), 2) + std::pow(yPrime(1), 2);
    dist = 1. + rSq * inverseDistParams2_.get()[0] + std::pow(rSq, 2) * inverseDistParams2_.get()[1] +
           std::pow(rSq, 3) * inverseDistParams2_.get()[4];

    yPrime(0) *= dist;
    yPrime(1) *= dist;


    double z = (r1 - yPrime(0) * r3).dot(t) / (r1 - yPrime(0) * r3).dot(y);

    wp[0] = y(0) * z;
    wp[1] = y(1) * z;
    wp[2] = z;

    // TODO: optimize, given the distortion
}

void PinholeStereoCamera::resetInvDistParams() {
    this->inverseDistParams1_.reset();
    this->inverseDistParams2_.reset();
}

void PinholeStereoCamera::makeInvDistParams() {
    inverseDistParams1_.reset(new double[5], std::default_delete<double[]>());
    inverseDistParams2_.reset(new double[5], std::default_delete<double[]>());

    this->makeInvDistParams(this->getDistortionParams(), inverseDistParams1_);
    this->makeInvDistParams(this->getDistortionParams(false), inverseDistParams2_);

    // foward distortion for the vector of points:
}

void PinholeStereoCamera::makeInvDistParams(std::shared_ptr<double> input, std::shared_ptr<double> output) {
    const int pointCount = 5;
    const int paramCount = 3;
    Eigen::Matrix<double, 2 * pointCount, paramCount> H;
    Eigen::Matrix<double, 2 * pointCount, 1> yDist;
    Eigen::Matrix<double, 2 * pointCount, 1> yUndist;

    yUndist(0, 0) = 0.5;
    yUndist(1, 0) = 0;

    yUndist(2, 0) = 0;
    yUndist(3, 0) = 1;

    yUndist(4, 0) = 0.5;
    yUndist(5, 0) = 0.5;

    yUndist(6, 0) = -0.5;
    yUndist(7, 0) = 0.5;

    yUndist(8, 0) = 0.5;
    yUndist(9, 0) = -0.5;


    const double& k1 = input.get()[0];
    const double& k2 = input.get()[1];
    const double& p1 = input.get()[2];
    const double& p2 = input.get()[3];
    const double& k3 = input.get()[4];


    for (int i = 0; i < pointCount; i++) {
        const double& x = yUndist(i * 2, 0);
        const double& y = yUndist(i * 2 + 1, 0);

        double rSq = std::pow(x, 2) + std::pow(y, 2);
        double dist = 1. + rSq * k1 + std::pow(rSq, 2) * k2;

        double& xNew = yDist(i * 2, 0);
        double& yNew = yDist(i * 2 + 1, 0);
        xNew = x * dist;
        yNew = y * dist;

        rSq = std::pow(xNew, 2) + std::pow(yNew, 2);
        H(2 * i, 0) = xNew * rSq;
        H(2 * i + 1, 0) = yNew * rSq;
        H(2 * i, 1) = xNew * std::pow(rSq, 2);
        H(2 * i + 1, 1) = yNew * std::pow(rSq, 2);
        H(2 * i, 2) = xNew * std::pow(rSq, 3);
        H(2 * i + 1, 2) = yNew * std::pow(rSq, 3);
    }

    Eigen::Matrix<double, paramCount, 2 * pointCount> Ht = H.transpose();
    Eigen::Matrix<double, paramCount, paramCount> HtH = Ht * H;

    Eigen::Matrix<double, paramCount, 1> params = HtH.inverse() * (Ht * (yUndist - yDist));
    output.get()[0] = params(0);
    output.get()[1] = params(1);
    output.get()[2] = 0;
    output.get()[3] = 0;
    output.get()[4] = params(2);
    std::cout << params << std::endl;

    // sanity check:
    for (int i = 0; i < pointCount; i++) {
        const double& x = yDist(i * 2, 0);
        const double& y = yDist(i * 2 + 1, 0);

        double rSq = std::pow(x, 2) + std::pow(y, 2);
        double dist = 1. + rSq * params(0) + std::pow(rSq, 2) * params(1) + std::pow(rSq, 3) * params(2);

        double xNew = x * dist;
        double yNew = y * dist;

        std::cout << "mapped " << yUndist(i * 2, 0) << "," << yUndist(i * 2 + 1, 0) << " --> " << yDist(i * 2, 0) << ","
                  << yDist(i * 2 + 1, 0) << " --> " << xNew << "," << yNew << std::endl;
    }
}


std::vector<block_optimization::BlockPtr<double>> PinholeStereoCamera::getProjectionChain(bool c1) {
    if (c1) {
        return projectionC1_;
    }
    return projectionC2_;
}

std::vector<std::shared_ptr<double>> PinholeStereoCamera::getParameterBlocksShared(bool c1) {
    if (c1) {
        return parametersC1_;
    }
    return parametersC2_;
}

void PinholeStereoCamera::enableOptimization() {
    this->optimizationEnabled_ = true;
    c1_->enableOptimization();
    c2_->enableOptimization();
    d1_->enableOptimization();
    d2_->enableOptimization();
    baseLine_->enableOptimization();
}

void PinholeStereoCamera::disableOptimization() {
    this->optimizationEnabled_ = false;
    c1_->disableOptimization();
    c2_->disableOptimization();
    d1_->disableOptimization();
    d2_->disableOptimization();
    baseLine_->disableOptimization();
}


std::shared_ptr<double> PinholeStereoCamera::getDistortionParams(bool c1) {
    if (c1) {
        return d1_->getInternalMemory(0);
    }
    return d2_->getInternalMemory(0);
}
std::shared_ptr<double> PinholeStereoCamera::getProjectionParams(bool c1) {
    if (c1) {
        return c1_->getInternalMemory(0);
    }
    return c2_->getInternalMemory(0);
}
std::shared_ptr<double> PinholeStereoCamera::getRotationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(0);
}
std::shared_ptr<double> PinholeStereoCamera::getTranslationParams(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->getInternalMemory(1);
}

block_optimization::BlockPtr<double> PinholeStereoCamera::getDistortionBlock(bool c1) {
    if (c1) {
        return d1_;
    }
    return d2_;
}
block_optimization::BlockPtr<double> PinholeStereoCamera::getProjectionBlock(bool c1) {
    if (c1) {
        return c1_;
    }
    return c2_;
}
block_optimization::BlockPtr<double> PinholeStereoCamera::getRotationBlock(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->rotation_;
}
block_optimization::BlockPtr<double> PinholeStereoCamera::getTranslationBlock(bool c1) {
    if (c1) {
        return nullptr;
    }
    return baseLine_->translation_;
}


} // namespace coco
