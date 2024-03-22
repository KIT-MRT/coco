#include "coco_stereo.h"

#include "internal/const_length.h"
#include "internal/motion_6dof.h"
#include "internal/pinhole_stereo_camera.h"
#include "internal/rectified_stereo_camera.h"
#include "internal/regularizations.h"
#include "internal/tracks_from_file.h"
#include "internal/undistorted_stereo_camera.h"

#include <fstream>

#include <block_optimization/ceres_solver.h>
#include <block_optimization/sgd_solver.h>
#include <block_optimization/standard_blocks.h>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <ceres/ceres.h>

#include <Eigen/Dense>

#include <feature_tracking/stereo_tracker_libviso.h>
#include <feature_tracking/utilities.h>
#include <feature_tracking/visualization.h>

#include <opencv2/core/eigen.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//#define DEBUG_MODE
//#define DEBUG_MODE_SPAM

#if defined(DEBUG_MODE_SPAM) && !defined(DEBUG_MODE)
#define DEBUG_MODE
#endif

#ifdef DEBUG_MODE
#include <iostream>
#endif

namespace coco {

namespace fs = boost::filesystem;

CocoStereo::Parameters::Parameters(const YAML::Node cfg) : Parameters() {
    if (cfg["preproc"]) {
        preproc = PreprocType(cfg["preproc"].as<int>());
    }
    if (cfg["bucketH"]) {
        bucketH = cfg["bucketH"].as<int>();
    }
    if (cfg["bucketW"]) {
        bucketW = cfg["bucketW"].as<int>();
    }
    if (cfg["motionRansacIters"]) {
        motionRansacIters = cfg["motionRansacIters"].as<int>();
    }
    if (cfg["motionRansacBaseThresh"]) {
        motionRansacBaseThresh = cfg["motionRansacBaseThresh"].as<double>();
    }
    if (cfg["motionRansacStartThresh"]) {
        motionRansacStartThresh = cfg["motionRansacStartThresh"].as<double>();
    }
    if (cfg["motionRansacTimeConst"]) {
        motionRansacTimeConst = cfg["motionRansacTimeConst"].as<double>();
    }
    if (cfg["maxTrackLength"]) {
        maxTrackLength = cfg["maxTrackLength"].as<unsigned int>();
    }
    if (cfg["minRotationForCalib"]) {
        minRotationForCalib = cfg["minRotationForCalib"].as<double>();
    }
    if (cfg["minTranslationForCalib"]) {
        minTranslationForCalib = cfg["minTranslationForCalib"].as<double>();
    }
    if (cfg["visualizeMatches"]) {
        visualizeMatches = cfg["visualizeMatches"].as<bool>();
    }
    if (cfg["reestimateProjection"]) {
        reestimateProjection = cfg["reestimateProjection"].as<bool>();
    }
    if (cfg["bundleAdjustMotion"]) {
        bundleAdjustMotion = cfg["bundleAdjustMotion"].as<bool>();
    }
    if (cfg["bundleAdjustDistorted"]) {
        bundleAdjustDistorted = cfg["bundleAdjustDistorted"].as<bool>();
    }
    if (cfg["refineAll"]) {
        refineAll = cfg["refineAll"].as<bool>();
    }
    if (cfg["refinementSteps"]) {
        refinementSteps = cfg["refinementSteps"].as<int>();
    }
    if (cfg["refinementBatchSize"]) {
        refinementBatchSize = cfg["refinementBatchSize"].as<int>();
    }
    if (cfg["refinementLearningRate"]) {
        refinementLearningRate = cfg["refinementLearningRate"].as<double>();
    }
    if (cfg["projectionRegularization"]) {
        projectionRegularization = cfg["projectionRegularization"].as<double>();
    }
    if (cfg["extrinsicsRegularization"]) {
        extrinsicsRegularization = cfg["extrinsicsRegularization"].as<double>();
    }
    if (cfg["sphericalProjectionRegularization"]) {
        sphericalProjectionRegularization = cfg["sphericalProjectionRegularization"].as<double>();
    }
    if (cfg["distortionRegularization"]) {
        distortionRegularization = cfg["distortionRegularization"].as<double>();
    }
    if (cfg["maxTime"]) {
        maxTime = cfg["maxTime"].as<double>();
    }
    if (cfg["calibrate"]) {
        calibrate = cfg["calibrate"].as<bool>();
    }
    if (cfg["cameraFile"]) {
        cameraFile = cfg["cameraFile"].as<std::string>();
    }
    if (cfg["exportOnShutdown"]) {
        exportOnShutdown = cfg["exportOnShutdown"].as<bool>();
    }
    if (cfg["exportRate"]) {
        exportRate = cfg["exportRate"].as<int>();
    }
    if (cfg["readLatest"]) {
        readLatest = cfg["readLatest"].as<bool>();
    }
    if (cfg["cameraFilePrefix"]) {
        cameraFilePrefix = cfg["cameraFilePrefix"].as<std::string>();
    }
    if (cfg["baseLine"]) {
        baseLine = cfg["baseLine"].as<double>();
    }
    if (cfg["focalLength"]) {
        focalLength = cfg["focalLength"].as<double>();
    }
    if (cfg["centerX"]) {
        centerX = cfg["centerX"].as<double>();
    }
    if (cfg["centerY"]) {
        centerY = cfg["centerY"].as<double>();
    }
    if (cfg["printParams"]) {
        printParams = cfg["printParams"].as<bool>();
    }
    if (cfg["estimateTangentialDistortion"]) {
        estimateTangentialDistortion = cfg["estimateTangentialDistortion"].as<bool>();
    }
    if (cfg["estimateK3"]) {
        estimateK3 = cfg["estimateK3"].as<bool>();
    }
}


CocoStereo::CocoStereo(Parameters params) {
    // make sure the cameras are reset to nullptr
    currentRawCamera_.reset();
    currentProcessedCamera_.reset();
    currentUndistCamera_.reset();
    processCount_ = 0;

    // TODO: everything within this block should be configurable
    params_ = params;

    // setup for feature tracking

    // TODO: THIS IS THE REAL TRACKING METHOD
    feature_tracking::StereoTrackerLibViso::Parameters trackerParams;
    trackerParams.match_radius = 300;
    trackerParams.match_disp_tolerance = 300;
    trackerParams.outlier_disp_tolerance = 300;
    trackerParams.outlier_flow_tolerance = 300;
    // trackerParams.method = 2;
    trackerParams.maxTracklength = params_.maxTrackLength;
    featureTracking_ = feature_tracking::StereoTrackerPtr(new feature_tracking::StereoTrackerLibViso(trackerParams));

    // TODO: FOR EVALUATION ONLY
    //  feature_tracking::TracksFromFile::Parameters trackerParams;
    //  trackerParams.maxTracklength = params_.maxTrackLength;
    //  //trackerParams.folder = "/data/points_k1-02/";
    //  trackerParams.folder = "/home/rehder/papers/stereo_simulation/points/";
    //    featureTracking_ = feature_tracking::StereoTrackerPtr(new feature_tracking::TracksFromFile(trackerParams));

    // init the circular buffer for the motion models with one instance less than track length
    // b/c of reference frame
    this->motionSteps_ = boost::circular_buffer<MotionModelPtr>(params_.maxTrackLength - 1);

    this->initCameraModels();
}

CocoStereo::~CocoStereo() {
    if (params_.exportOnShutdown) {
        this->exportCameraParameters();
    }
}

/////////////////////////////////////////////
/// processing
/////////////////////////////////////////////
void CocoStereo::process(const cv::Mat& im1, const cv::Mat& im2) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered process()" << std::endl;
#endif
    processingStart_ = boost::chrono::high_resolution_clock::now();
    boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();
    ++processCount_;
    if (params_.exportRate > 0 && processCount_ % params_.exportRate == 0) {
        this->exportCameraParameters();
    }

    // if no camera model was made yet, create one
    if (!currentRawCamera_ || !currentProcessedCamera_ || !currentUndistCamera_) {
        if (params_.cameraFile.empty()) {
            this->makeCameraModels(im1);
        } else {
            this->makeCameraModels(params_.cameraFile);
        }
    }

    // image preparation:
    // nothing, undistortion, rectification, rectification with no undistortion
    this->prepareImages(im1, im2);

    boost::chrono::duration<double> diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### Img prep    " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    // feature tracklet preparation with matching, outlier removal, remapping
    this->computeMatches();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### matching    " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    this->prepareTracklets();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### track prep  " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    // initial motion estimation
    this->estimateMotion();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### motion init " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    if (params_.printParams) {
        print(currentRawCamera_);
        print(this->motionSteps_);
    }

    if (!this->checkTime()) {
#ifdef DEBUG_MODE
        std::cout << "\033[35;1m"
                  << "No time left after initial motion estimation, returning"
                  << "\033[0m" << std::endl;
#endif
        return;
    }
    this->reestimateMotionChain();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### motion bundle " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    if (!params_.calibrate) {
#ifdef DEBUG_MODE
        std::cout << "Coco: calibration inactive, leaving process()" << std::endl;
#endif
        return;
    }

    if (!this->checkTime()) {
#ifdef DEBUG_MODE
        std::cout << "\033[35;1m"
                  << "No time left after motion chain estimation, returning"
                  << "\033[0m" << std::endl;
#endif
        return;
    }
    // check if all the further processing steps should be done
    if (!this->isCalibrationPossible()) {
        // if not, the processing should end here
#ifdef DEBUG_MODE
        std::cout << "Coco: calibration seems not possible, leaving process()" << std::endl;
#endif
        return;
    }

    // reestimate projection matrix from current information
    this->reestimateProjection();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### projection " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

    if (!this->checkTime()) {
#ifdef DEBUG_MODE
        std::cout << "\033[35;1m"
                  << "No time left after projection estimation, returning"
                  << "\033[0m" << std::endl;
#endif
        return;
    }
    // camera calibration refinement
    this->refineComplete();
    diff = boost::chrono::high_resolution_clock::now() - start;
    std::cout << "###### PROFILIG ###### refine " << diff.count() << std::endl;
    start = boost::chrono::high_resolution_clock::now();

#ifdef DEBUG_MODE
    std::cout << "Coco: leaving process()" << std::endl;
#endif
}

/////////////////////////////////////////////
/// data access
/////////////////////////////////////////////
void CocoStereo::getCameraParams(cv::Mat& c1, cv::Mat& d1, cv::Mat& c2, cv::Mat& d2, cv::Mat& R, cv::Mat& t) {
    // TODO: this could be in the camera interface

    // allocate memory
    c1 = cv::Mat::eye(3, 3, CV_64FC1);
    d1 = cv::Mat::zeros(1, 5, CV_64FC1);
    c2 = cv::Mat::eye(3, 3, CV_64FC1);
    d2 = cv::Mat::zeros(1, 5, CV_64FC1);
    R = cv::Mat::eye(3, 3, CV_64FC1);
    t = cv::Mat::zeros(3, 1, CV_64FC1);

    // left camera
    std::vector<double*> params = currentRawCamera_->getParameterBlocksRaw();
    d1.at<double>(0) = params[0][0];
    d1.at<double>(1) = params[0][1];
    d1.at<double>(2) = params[0][2];
    d1.at<double>(3) = params[0][3];
    d1.at<double>(4) = params[0][4];
    c1.at<double>(0, 0) = params[1][0];
    c1.at<double>(0, 2) = params[1][1];
    c1.at<double>(1, 1) = params[1][2];
    c1.at<double>(1, 2) = params[1][3];

    // right camera
    params = currentRawCamera_->getParameterBlocksRaw(false);
    d2.at<double>(0) = params[2][0];
    d2.at<double>(1) = params[2][1];
    d2.at<double>(2) = params[2][2];
    d2.at<double>(3) = params[2][3];
    d2.at<double>(4) = params[2][4];
    c2.at<double>(0, 0) = params[3][0];
    c2.at<double>(0, 2) = params[3][1];
    c2.at<double>(1, 1) = params[3][2];
    c2.at<double>(1, 2) = params[3][3];

    // extrinsic transform
    // TODO: block readout
    std::shared_ptr<block_optimization::Transform6DOF<double>> trafo =
        std::dynamic_pointer_cast<block_optimization::Transform6DOF<double>>(
            currentRawCamera_->getProjectionChain(false)[0]);
    Eigen::Matrix<double, 3, 3> RTrafo;
    trafo->getRotationMatrix(RTrafo);
    Eigen::Matrix<double, 3, 1> tTrafo;
    trafo->getTranslation(tTrafo);
    // to openCV
    cv::eigen2cv(RTrafo, R);
    cv::eigen2cv(tTrafo, t);
}

/////////////////////////////////////////////
/// image processing
/////////////////////////////////////////////
void CocoStereo::prepareImages(const cv::Mat& im1, const cv::Mat& im2) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered prepareImages()" << std::endl;
#endif
    // do image perparation/remapping,
    // depending on what is set as method for preprocessing
    if (params_.preproc == PREPROC_NONE) {
        im1.copyTo(im1Remapped_);
        im2.copyTo(im2Remapped_);
    } else {
        cv::remap(im1, im1Remapped_, im1MapU_, im1MapV_, cv::INTER_LINEAR);
        cv::remap(im2, im2Remapped_, im2MapU_, im2MapV_, cv::INTER_LINEAR);
    }
}
void CocoStereo::makeMapping() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeMaps()" << std::endl;
#endif
    // this calls the preparation/remapping,
    // depending on what is set as method for preprocessing
    switch (params_.preproc) {
    case PREPROC_NONE:
        this->clearMapping();
        break;
    case PREPROC_UNDISTORT:
        this->makeUndistortionMapping();
        break;
    case PREPROC_RECTIFY:
        this->makeRectificationMapping();
        break;
    case PREPROC_ALIGN:
        this->makeAlignmentMapping();
        break;
    default:
        throw std::runtime_error("Unknown preprocessing specified");
        break;
    }
}
void CocoStereo::clearMapping() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered clearMapping()" << std::endl;
#endif
    im1MapU_ = cv::Mat();
    im1MapV_ = cv::Mat();
    im2MapU_ = cv::Mat();
    im2MapV_ = cv::Mat();

    // all processed cameras are the same as the raw camera
    currentProcessedCamera_ = currentRawCamera_;
    currentUndistCamera_ = currentRawCamera_;
}
void CocoStereo::makeUndistortionMapping() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeUndistortionMapping()" << std::endl;
#endif
    // TODO
    throw std::runtime_error("Coco: preprocessing undistortion not implemented yet.");
    // TODO: this has to set the right camera models
}
void CocoStereo::makeRectificationMapping() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeRectificationMapping()" << std::endl;
#endif
    // TODO
    throw std::runtime_error("Coco: preprocessing rectification not implemented yet.");
    // TODO: this has to set the right camera models
}
void CocoStereo::makeAlignmentMapping() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeAlignmentMapping()" << std::endl;
#endif
    // TODO
    throw std::runtime_error("Coco: preprocessing alignment not implemented yet.");
    // TODO: this has to set the right camera models
}
void CocoStereo::computeMatches() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered computeMatches()" << std::endl;
#endif
    // compute the matches
    featureTracking_->pushBack(im1Remapped_, im2Remapped_);
    // get matches
    if (!tracklets_) {
        tracklets_ = std::make_shared<feature_tracking::StereoTrackletList>();
    }
    featureTracking_->getTracklets(*tracklets_);
#ifdef DEBUG_MODE
    {
        std::cout << "Coco: Im 1 size: " << im1Remapped_.size() << std::endl;
        std::cout << "Coco: Im 2 size: " << im2Remapped_.size() << std::endl;
        std::cout << "Coco: tracklet count is " << tracklets_->size() << std::endl;
    }
#endif
    // visualize if desired
    if (params_.visualizeMatches) {
        visualizationIms_.insert(visualizationIms_.begin(), std::make_pair(cv::Mat(), cv::Mat()));
        visualizationIms_[0].first = im1Remapped_.clone();
        visualizationIms_[0].second = im2Remapped_.clone();
        while (visualizationIms_.size() > params_.maxTrackLength) {
            visualizationIms_.pop_back();
        }

        cv::Mat output;
        feature_tracking::visualization::drawMatches(*tracklets_, visualizationIms_, output);

        cv::namedWindow("matches", 0);
        cv::imshow("matches", output);
        cv::waitKey(1);
    }
}
void CocoStereo::prepareTracklets() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered prepareTracklets()" << std::endl;
#endif
    if (tracklets_->empty()) {
        return;
    }

    if (!currentProcessedCamera_) {
        throw std::runtime_error("Coco: Requested camera in tracklet preparation, none defined.");
    }
#ifdef DEBUG_MODE_SPAM
    std::cout << "Coco: initial guess for 3D world points:" << std::endl;
#endif
    // compute the 3D position of a tracklet with reference to current frame
    for (feature_tracking::StereoTracklet& tracklet : *tracklets_) {
        currentProcessedCamera_->computeWorldPoint(tracklet);
#ifdef DEBUG_MODE_SPAM
        if (!tracklet.empty() && tracklet.front().x_) {
            std::cout << "[" << tracklet.front().x_->data[0] << "," << tracklet.front().x_->data[1] << ","
                      << tracklet.front().x_->data[2] << "]" << std::endl;
        }
#endif
    }


    if (std::dynamic_pointer_cast<PinholeStereoCamera>(currentProcessedCamera_)) {
        std::dynamic_pointer_cast<PinholeStereoCamera>(currentProcessedCamera_)->resetInvDistParams();
    }

    //  block_optimization::BlockPtr<double>  dataBlock
    //      = block_optimization::BlockPtr<double> ( new block_optimization::DataBlock<double, 3>("data"));
    //  block_optimization::BlockPtr<double> reprojection
    //      = block_optimization::BlockPtr<double> (new block_optimization::LinearError<double, 2>("err"));
    //  // error and projection should be constant
    //  dataBlock->enableOptimization();
    //  reprojection->disableOptimization();
    //  currentProcessedCamera_->disableOptimization();
    //  block_optimization::ProcessingChainPtr<double> leftChain;
    //  std::vector<double*> leftParams;
    //  this->makeOptimizationChain(dataBlock,
    //                              std::vector<MotionModelPtr>{},
    //                              currentProcessedCamera_->getProjectionChain(true),
    //                              currentProcessedCamera_->getParameterBlocksRaw(true),
    //                              reprojection,
    //                              leftChain,
    //                              leftParams);
    //  block_optimization::ProcessingChainPtr<double> rightChain;
    //  std::vector<double*> rightParams;
    //  this->makeOptimizationChain(dataBlock,
    //                              std::vector<MotionModelPtr>{},
    //                              currentProcessedCamera_->getProjectionChain(false),
    //                              currentProcessedCamera_->getParameterBlocksRaw(false),
    //                              reprojection,
    //                              rightChain,
    //                              rightParams);

    //  std::vector<double*> worldPoints(1);
    //  std::vector<double*> leftMatches(1);
    //  std::vector<double*> rightMatches(1);

    //  int i = 0;
    //  for(feature_tracking::StereoTracklet& tracklet : *tracklets_) {
    //    if(!tracklet[0].x_) {
    ////      // tracklet world point invalid
    ////      worldPoints[i] = NULL;
    ////      leftMatches[i] = NULL;
    //    } else {
    ////      // data
    ////      worldPoints[i] = tracklet[0].x_->data;
    ////      // left observation
    ////      leftMatches[i] = new double[2];
    ////      leftMatches[i][0] = tracklet[0].p1_.u_;
    ////      leftMatches[i][1] = tracklet[0].p1_.v_;
    ////      rightMatches[i] = new double[2];
    ////      rightMatches[i][0] = tracklet[0].p2_.u_;
    ////      rightMatches[i][1] = tracklet[0].p2_.v_;

    //      //std::cout << worldPoints[i] << " " << leftMatches[i] << " " << rightMatches[i] <<std::endl;
    //      double leftMatch[2];
    //      double rightMatch[2];
    //      leftMatch[0] = tracklet[0].p1_.u_;
    //      leftMatch[1] = tracklet[0].p1_.v_;
    //      rightMatch[0] = tracklet[0].p2_.u_;
    //      rightMatch[1] = tracklet[0].p2_.v_;

    //      block_optimization::CeresSolver blockCeres;

    //      worldPoints[0] = tracklet[0].x_->data;
    //      leftMatches[0] = leftMatch;
    //      rightMatches[0] = rightMatch;
    //      blockCeres.addChain(leftChain, leftParams, worldPoints, leftMatches);
    //      blockCeres.addChain(rightChain, rightParams, worldPoints, rightMatches);

    //      ceres::Solver::Options options;
    //      options.max_num_iterations = 2;
    //      options.function_tolerance = 1e-4;
    //      options.linear_solver_type = ceres::DENSE_QR;
    //    #ifdef DEBUG_MODE_SPAM
    //      options.minimizer_progress_to_stdout = true;
    //    #else
    //      options.minimizer_progress_to_stdout = false;
    //    #endif
    //      ceres::Solver::Summary summary;
    //      Solve(options, blockCeres.problem_.get(), &summary);
    //    #ifdef DEBUG_MODE_SPAM
    //      std::cout << "Coco: " << summary.BriefReport() << std::endl;
    //    #endif
    //    }
    //    ++i;
    //  }

    //  block_optimization::CeresSolver blockCeres;
    //  blockCeres.addChain(leftChain, leftParams, worldPoints, leftMatches);
    //  blockCeres.addChain(rightChain, rightParams, worldPoints, rightMatches);
    //  // setup the solver
    //  ceres::Solver::Options options;
    //  options.max_num_iterations = 2;
    //  options.function_tolerance = 1e-4;
    //  options.linear_solver_type = ceres::DENSE_QR;
    //#ifdef DEBUG_MODE_SPAM
    //  options.minimizer_progress_to_stdout = true;
    //#else
    //  options.minimizer_progress_to_stdout = false;
    //#endif
    //  ceres::Solver::Summary summary;
    //  Solve(options, blockCeres.problem_.get(), &summary);

    ////#ifdef DEBUG_MODE
    //  std::cout << "Coco: " << summary.BriefReport() << std::endl;
    //  CocoStereo::print(this->motionSteps_);
    //  std::cout << "Coco: freeing memory" <<  std::endl;
    //#endif
    // done: free memory
    //  for(double* data : leftMatches) {
    //    delete[] data;
    //    data = NULL;
    //  }
    //  for(double* data : rightMatches) {
    //    delete[] data;
    //    data = NULL;
    //  }

    // TODO: left-right RANSAC

    // TODO: remap all tracklets to the current camera model assumption
}
void CocoStereo::remapTracklets() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered remapTracklets()" << std::endl;
#endif
    // TODO: this should map a tracklet from one camera configuration to another
}
void CocoStereo::computeReprojectionError(std::vector<double*>& worldPoints,
                                          std::vector<double*>& observ,
                                          block_optimization::ProcessingChainPtr<double>& chain,
                                          std::vector<double*>& paramBlocks,
                                          std::vector<double>& reprojectionErr) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered computeReprojectionError()" << std::endl;
#endif
    if (worldPoints.size() != observ.size()) {
        throw std::runtime_error("Worldpoint and observation cout has to be the same in reprojection error");
    }
    reprojectionErr.resize(worldPoints.size());
    double output[2];
    for (unsigned int i = 0; i < worldPoints.size(); i++) {
        if (!worldPoints[i] || !observ[i]) {
            reprojectionErr[i] = 0.;
            continue;
        }
        paramBlocks.front() = worldPoints[i];
        paramBlocks.back() = observ[i];
        chain->forwardChain(paramBlocks.data(), NULL, output, false);
        reprojectionErr[i] = std::hypot(output[0], output[1]);
    }
    // reset param blocks for data and target
    paramBlocks.front() = NULL;
    paramBlocks.back() = NULL;
}

/////////////////////////////////////////////
/// motion estimation
/////////////////////////////////////////////
void CocoStereo::estimateMotion() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered estimateMotion()" << std::endl;
#endif
    if (!tracklets_ || tracklets_->empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: no tracklets available" << std::endl;
#endif
        return;
    }

    // this estimates the newest motion step
#ifdef DEBUG_MODE
    std::cout << "Coco: creating motion model" << std::endl;
#endif
    // create new motion model instance in cicular buffer
    this->motionSteps_.push_front(MotionModelPtr(new Motion6DOF()));
    // if possible, init parameters from previous motion
    bool previousMotionExisted = false;
    if (this->motionSteps_.size() > 1) {
        previousMotionExisted = true;
        // deep copy values
        std::vector<block_optimization::BlockPtr<double>> source = this->motionSteps_[1]->getTransformChain();
        std::vector<block_optimization::BlockPtr<double>> target = this->motionSteps_[0]->getTransformChain();
        for (unsigned int blockIdx = 0; blockIdx < source.size(); blockIdx++) {
            std::vector<int> paramSize = source[blockIdx]->getParamCount();
            for (unsigned int paramBlock = 0; paramBlock < paramSize.size(); paramBlock++) {
                double* srcPtr = source[blockIdx]->getInternalMemory(paramBlock).get();
                double* dstPtr = target[blockIdx]->getInternalMemory(paramBlock).get();
                std::copy(srcPtr, srcPtr + paramSize[paramBlock], dstPtr);
            }
        }
    }


    block_optimization::BlockPtr<double> dataBlock =
        block_optimization::BlockPtr<double>(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> reprojection =
        block_optimization::BlockPtr<double>(new block_optimization::LinearError<double, 2>("err"));
    // error and projection should be constant
    dataBlock->disableOptimization();
    reprojection->disableOptimization();
    currentProcessedCamera_->disableOptimization();

#ifdef DEBUG_MODE
    std::cout << "Coco: making left chain" << std::endl;
#endif
    // create left reprojection error chain
    block_optimization::ProcessingChainPtr<double> leftChain;
    std::vector<double*> leftParams;
    this->makeOptimizationChain(dataBlock,
                                std::vector<MotionModelPtr>{this->motionSteps_.front()},
                                currentProcessedCamera_->getProjectionChain(true),
                                currentProcessedCamera_->getParameterBlocksRaw(true),
                                reprojection,
                                leftChain,
                                leftParams);

    // NOTE: right camera projection is not used at this point, because it's simply
    // not necessary. Scale comes from the 3D-structure of matches

#ifdef DEBUG_MODE
    std::cout << "Coco: making data" << std::endl;
#endif
    // copy data to vectors of pointers
    std::vector<double*> worldPoints(tracklets_->size());
    std::vector<double*> leftMatches(tracklets_->size());
    std::vector<double*> rightMatches(tracklets_->size());

    int i = 0;
    for (feature_tracking::StereoTracklet& tracklet : *tracklets_) {
        if (!tracklet[0].x_) {
            // tracklet world point invalid
            worldPoints[i] = NULL;
            leftMatches[i] = NULL;
        } else {
            // data
            worldPoints[i] = tracklet[0].x_->data;
            // left observation
            leftMatches[i] = new double[2];
            leftMatches[i][0] = tracklet[1].p1_.u_;
            leftMatches[i][1] = tracklet[1].p1_.v_;
            rightMatches[i] = new double[2];
            rightMatches[i][0] = tracklet[1].p2_.u_;
            rightMatches[i][1] = tracklet[1].p2_.v_;
        }
        ++i;
    }

    // find inliers in matches!
    std::vector<bool> inliers;
    int inlierCount = this->findInliers(leftChain, leftParams, worldPoints, leftMatches, inliers);

    double inlierRatio = double(inlierCount) / double(inliers.size());
    if (inlierRatio < 0.7) {
        // ransac motion
        inlierCount =
            this->ransacTransform(leftChain, leftParams, this->motionSteps_.front(), worldPoints, leftMatches, inliers);
        this->motionSteps_.front()->resetParams();
    }

#ifdef DEBUG_MODE
    std::cout << "Coco: motion inliers with current values " << inlierCount << " of " << inliers.size() << std::endl;
#endif

    // remove all invalid tracklets
    std::vector<double*>::iterator matchIt = leftMatches.begin();
    std::vector<double*>::iterator rightMatchIt = rightMatches.begin();
    std::vector<double*>::iterator wpIt = worldPoints.begin();
    feature_tracking::StereoTrackletList::iterator trackletIt = tracklets_->begin();
    for (unsigned int i = 0; i < inliers.size(); i++) {
        if (!inliers[i]) {
            delete[] * matchIt;
            matchIt = leftMatches.erase(matchIt);
            wpIt = worldPoints.erase(wpIt);
            delete[] * rightMatchIt;
            rightMatchIt = rightMatches.erase(rightMatchIt);
            trackletIt = tracklets_->erase(trackletIt);
        } else {
            ++matchIt;
            ++rightMatchIt;
            ++wpIt;
            ++trackletIt;
        }
    }

    this->featureTracking_->getInternalTracklets() = *tracklets_;
    if (tracklets_->empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: in estimate motion, there are no tracklets left" << std::endl;
#endif
        return;
    }
#ifdef DEBUG_MODE
    std::cout << "Coco: making ceres problem" << std::endl;
#endif
    // make ceres problem from the chains and corresponding data

    block_optimization::ProcessingChainPtr<double> rightChain;
    std::vector<double*> rightParams;
    this->makeOptimizationChain(dataBlock,
                                std::vector<MotionModelPtr>{this->motionSteps_.front()},
                                currentProcessedCamera_->getProjectionChain(false),
                                currentProcessedCamera_->getParameterBlocksRaw(false),
                                reprojection,
                                rightChain,
                                rightParams);

    block_optimization::CeresSolver blockCeres;
    blockCeres.addChain(leftChain, leftParams, worldPoints, leftMatches);
    blockCeres.addChain(rightChain, rightParams, worldPoints, rightMatches);
    // setup the solver
    ceres::Solver::Options options;
    options.max_num_iterations = 2;
    options.function_tolerance = 1e-4;
    options.linear_solver_type = ceres::DENSE_QR;
#ifdef DEBUG_MODE_SPAM
    options.minimizer_progress_to_stdout = true;
#else
    options.minimizer_progress_to_stdout = false;
#endif
    ceres::Solver::Summary summary;
    Solve(options, blockCeres.problem_.get(), &summary);

#ifdef DEBUG_MODE
    std::cout << "Coco: " << summary.BriefReport() << std::endl;
    CocoStereo::print(this->motionSteps_);
    std::cout << "Coco: freeing memory" << std::endl;
#endif
    // done: free memory
    for (double* data : leftMatches) {
        delete[] data;
        data = NULL;
    }
    for (double* data : rightMatches) {
        delete[] data;
        data = NULL;
    }
}
int CocoStereo::ransacTransform(block_optimization::ProcessingChainPtr<double>& chain,
                                std::vector<double*> params,
                                MotionModelPtr motionModel,
                                std::vector<double*>& worldPoints,
                                std::vector<double*>& observations,
                                std::vector<bool>& inliers) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered ransacTransform()" << std::endl;
#endif
    // for less or equal 3 points, no ransac is necessary
    if (worldPoints.size() <= 3) {
        inliers.resize(worldPoints.size());
        std::fill(inliers.begin(), inliers.end(), true);
        return inliers.size();
    }

    // prepare all data storage
    std::vector<double*> sampledWp(3);
    std::vector<double*> sampledObserv(3);
    int inlierCount = 0;
    std::vector<bool> thisInliers;
    std::uniform_int_distribution<int> uniform(0, worldPoints.size() - 1);

    // perform ransac iterations
    for (int i = 0; i < params_.motionRansacIters; i++) {
#ifdef DEBUG_MODE
        std::cout << "Coco: motion ransac iter " << i << std::endl;
#endif
        // reset motion parameters to optimize from same starting value
        motionModel->resetParams();
        // make list of the three points that are sampled
        int index0 = uniform(rng_);
        int index1 = index0;
        while (index0 == index1) {
            index1 = uniform(rng_);
        }
        int index2 = index0;
        while (index0 == index2 || index1 == index2) {
            index2 = uniform(rng_);
        }

        // get the three points for model estimation
        sampledWp[0] = worldPoints[index0];
        sampledObserv[0] = observations[index0];
        sampledWp[1] = worldPoints[index1];
        sampledObserv[1] = observations[index1];
        sampledWp[2] = worldPoints[index2];
        sampledObserv[2] = observations[index2];

        // optimize
        block_optimization::CeresSolver blockCeres;
        blockCeres.addChain(chain, params, sampledWp, sampledObserv);

        // setup the solver
        ceres::Solver::Options options;
        options.max_num_iterations = 2;
        options.function_tolerance = 1e-1;
        options.linear_solver_type = ceres::DENSE_QR;
#ifdef DEBUG_MODE_SPAM
        options.minimizer_progress_to_stdout = true;
#else
        options.minimizer_progress_to_stdout = false;
#endif
        ceres::Solver::Summary summary;
        Solve(options, blockCeres.problem_.get(), &summary);
#ifdef DEBUG_MODE
        std::cout << "Coco: " << summary.BriefReport() << std::endl;
#endif
        // assess inliers
        int thisInlierCount = this->findInliers(chain, params, worldPoints, observations, thisInliers);
        if (inlierCount < thisInlierCount) {
            inlierCount = thisInlierCount;
            inliers = thisInliers;
        }
#ifdef DEBUG_MODE
        std::cout << "Coco: motion ransac inlier count " << thisInlierCount << std::endl;
#endif
    }
    return inlierCount;
}
int CocoStereo::findInliers(block_optimization::ProcessingChainPtr<double>& chain,
                            std::vector<double*>& paramBlocks,
                            std::vector<double*>& worldPoints,
                            std::vector<double*>& observations,
                            std::vector<bool>& inliers) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered findInliers()" << std::endl;
#endif
    inliers.resize(worldPoints.size());
    std::vector<double> reprojectionErr;
    double motionRansacThresh =
        params_.motionRansacBaseThresh + (params_.motionRansacStartThresh - params_.motionRansacBaseThresh) *
                                             std::exp(-params_.motionRansacTimeConst * processCount_);
#ifdef DEBUG_MODE
    std::cout << "Coco: motion ransac threshold is " << motionRansacThresh << std::endl;
#endif
    this->computeReprojectionError(worldPoints, observations, chain, paramBlocks, reprojectionErr);
    int inlierCount = 0;
    double avgErr = 0.;
    for (unsigned int i = 0; i < reprojectionErr.size(); i++) {
        inliers[i] = reprojectionErr[i] < motionRansacThresh;
        avgErr += inliers[i] ? reprojectionErr[i] : 0.;
        inlierCount += int(inliers[i]);
    }
#ifdef DEBUG_MODE
    std::cout << "Coco: average reprojection error of inliers is " << avgErr / double(inlierCount) << std::endl;
#endif
    return inlierCount;
}
void CocoStereo::bundleAdjust(StereoCameraModelPtr camera,
                              std::shared_ptr<feature_tracking::StereoTrackletList> tracklets) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered bundleAdjustConstCamera()" << std::endl;
#endif
    // this runs the bundle adjustment

    // make the entire chain, run optimization
    block_optimization::BlockPtr<double> dataBlock =
        block_optimization::BlockPtr<double>(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> reprojection =
        block_optimization::BlockPtr<double>(new block_optimization::LinearError<double, 2>("err"));

    // error should be constant
    reprojection->disableOptimization();

    block_optimization::CeresSolver blockCeres;

    double avgError = 0;

    // for memory mgmt
    std::list<std::shared_ptr<double>> allLeftMatches;
    std::list<std::shared_ptr<double>> allRightMatches;
    for (unsigned int i = 0; i < this->motionSteps_.size() + 1; i++) {

        // make motion vector
        std::vector<MotionModelPtr> motionChain;
        for (unsigned int k = 0; k < i; k++) {
            motionChain.push_back(this->motionSteps_[i - k - 1]);
        }

        // make processing chains
        block_optimization::ProcessingChainPtr<double> leftChain;
        std::vector<double*> leftParams;
        this->makeOptimizationChain(dataBlock,
                                    motionChain,
                                    camera->getProjectionChain(true),
                                    camera->getParameterBlocksRaw(true),
                                    reprojection,
                                    leftChain,
                                    leftParams);

        block_optimization::ProcessingChainPtr<double> rightChain;
        std::vector<double*> rightParams;
        this->makeOptimizationChain(dataBlock,
                                    motionChain,
                                    camera->getProjectionChain(false),
                                    camera->getParameterBlocksRaw(false),
                                    reprojection,
                                    rightChain,
                                    rightParams);


        // make data
        std::vector<double*> worldPoints;
        std::vector<double*> leftMatches;
        std::vector<double*> rightMatches;
#ifdef DEBUG_MODE
        std::cout << "Tracklet count after bucketing " << tracklets->size() << std::endl;
#endif
        for (feature_tracking::StereoTracklet& tracklet : *tracklets) {
            if (tracklet.size() < this->motionSteps_.size() + 1 || !tracklet[0].x_) {
                continue;
            }

            worldPoints.push_back(tracklet[0].x_->data);
            allLeftMatches.push_back(std::shared_ptr<double>(new double[2]));
            allRightMatches.push_back(std::shared_ptr<double>(new double[2]));
            double* lPtr = allLeftMatches.back().get();
            double* rPtr = allRightMatches.back().get();
            lPtr[0] = tracklet[i].p1_.u_;
            lPtr[1] = tracklet[i].p1_.v_;
            rPtr[0] = tracklet[i].p2_.u_;
            rPtr[1] = tracklet[i].p2_.v_;

            leftMatches.push_back(lPtr);
            rightMatches.push_back(rPtr);
        }
        if (leftMatches.empty()) {
            continue;
        }
#ifdef DEBUG_MODE
        std::cout << "Number of valid matches: " << leftMatches.size() << std::endl;
#endif
        // add the residue to problem
        blockCeres.addChain(leftChain, leftParams, worldPoints, leftMatches, new ceres::CauchyLoss(0.5));
        blockCeres.addChain(rightChain, rightParams, worldPoints, rightMatches, new ceres::CauchyLoss(0.5));

        std::vector<double> reprojectionErr;
        this->computeReprojectionError(worldPoints, leftMatches, leftChain, leftParams, reprojectionErr);
        for (const double& err : reprojectionErr) {
            avgError += 0.5 * err / double(reprojectionErr.size());
        }
        reprojectionErr.clear();
        this->computeReprojectionError(worldPoints, rightMatches, rightChain, rightParams, reprojectionErr);
        for (const double& err : reprojectionErr) {
            avgError += 0.5 * err / double(reprojectionErr.size());
        }
    }
    avgError /= double(this->motionSteps_.size() + 1);
    std::cout << "----------------------------------------------------------------------" << avgError << std::endl;

    if (allLeftMatches.empty() || allRightMatches.empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: no valid matches found in reestimateProjection, returning" << std::endl;
#endif
        return;
    }
    avgError = 1.;
    // set the baseline constant for now
    if (camera->isOptimizationEnabled()) {
        // set camera baseline parameterized as constant length
        ceres::LocalParameterization* local_parameterization =
            new ceres::AutoDiffLocalParameterization<ConstantLength, 3, 3>();
        blockCeres.problem_->SetParameterization(camera->getParameterBlocksRaw(false)[1], local_parameterization);

        // regularization
        if (params_.projectionRegularization > std::numeric_limits<double>::epsilon()) {
            ceres::CostFunction* regularizeLeft = new ceres::AutoDiffCostFunction<Regularization<4>, 4, 4>(
                new Regularization<4>(avgError * params_.projectionRegularization,
                                      camera->getProjectionBlock()->getInternalMemory(0).get()));
            blockCeres.problem_->AddResidualBlock(
                regularizeLeft, NULL, camera->getProjectionBlock()->getInternalMemory(0).get());
            ceres::CostFunction* regularizeRight = new ceres::AutoDiffCostFunction<Regularization<4>, 4, 4>(
                new Regularization<4>(avgError * params_.projectionRegularization,
                                      camera->getProjectionBlock(false)->getInternalMemory(0).get()));
            blockCeres.problem_->AddResidualBlock(
                regularizeRight, NULL, camera->getProjectionBlock(false)->getInternalMemory(0).get());
        }
        if (params_.extrinsicsRegularization > std::numeric_limits<double>::epsilon()) {
            ceres::CostFunction* regularizeRotation = new ceres::AutoDiffCostFunction<Regularization<3>, 3, 3>(
                new Regularization<3>(avgError * params_.extrinsicsRegularization,
                                      camera->getRotationBlock(false)->getInternalMemory(0).get()));
            blockCeres.problem_->AddResidualBlock(
                regularizeRotation, NULL, camera->getRotationBlock(false)->getInternalMemory(0).get());
            ceres::CostFunction* regularizeTranslation = new ceres::AutoDiffCostFunction<Regularization<3>, 3, 3>(
                new Regularization<3>(avgError * params_.extrinsicsRegularization,
                                      camera->getTranslationBlock(false)->getInternalMemory(0).get()));
            blockCeres.problem_->AddResidualBlock(
                regularizeTranslation, NULL, camera->getTranslationBlock(false)->getInternalMemory(0).get());
        }
        if (params_.sphericalProjectionRegularization > std::numeric_limits<double>::epsilon()) {
            ceres::CostFunction* regularizeSphericalL = new ceres::AutoDiffCostFunction<SphericalProjection, 1, 4>(
                new SphericalProjection(params_.sphericalProjectionRegularization));
            blockCeres.problem_->AddResidualBlock(
                regularizeSphericalL, NULL, camera->getProjectionBlock(true)->getInternalMemory(0).get());
            ceres::CostFunction* regularizeSphericalR = new ceres::AutoDiffCostFunction<SphericalProjection, 1, 4>(
                new SphericalProjection(params_.sphericalProjectionRegularization));
            blockCeres.problem_->AddResidualBlock(
                regularizeSphericalR, NULL, camera->getProjectionBlock(false)->getInternalMemory(0).get());
        }
        if (params_.distortionRegularization > std::numeric_limits<double>::epsilon()) {
            ceres::CostFunction* regularizeLeftDist =
                new ceres::AutoDiffCostFunction<Regularization<5>, 5, 5>(new Regularization<5>(
                    avgError * params_.distortionRegularization, camera->getDistortionParams(true).get()));
            blockCeres.problem_->AddResidualBlock(regularizeLeftDist, NULL, camera->getDistortionParams(true).get());
            ceres::CostFunction* regularizeRightDist =
                new ceres::AutoDiffCostFunction<Regularization<5>, 5, 5>(new Regularization<5>(
                    avgError * params_.distortionRegularization, camera->getDistortionParams(false).get()));
            blockCeres.problem_->AddResidualBlock(regularizeRightDist, NULL, camera->getDistortionParams(false).get());
        }
        if (!params_.estimateTangentialDistortion) {
            if (!params_.estimateK3) {
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(true).get(),
                                                         new ceres::SubsetParameterization(5, {2, 3, 4}));
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(false).get(),
                                                         new ceres::SubsetParameterization(5, {2, 3, 4}));
            } else {
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(true).get(),
                                                         new ceres::SubsetParameterization(5, {2, 3}));
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(false).get(),
                                                         new ceres::SubsetParameterization(5, {2, 3}));
            }
        } else {
            if (!params_.estimateK3) {
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(true).get(),
                                                         new ceres::SubsetParameterization(5, {4}));
                blockCeres.problem_->SetParameterization(currentRawCamera_->getDistortionParams(false).get(),
                                                         new ceres::SubsetParameterization(5, {4}));
            }
        }
    }

    // setup the solver
    ceres::Solver::Options options;
    options.max_num_iterations = 2;
    options.function_tolerance = 1e-3;
    options.linear_solver_type = ceres::DENSE_QR;
#ifdef DEBUG_MODE
    options.minimizer_progress_to_stdout = true;
#else
    options.minimizer_progress_to_stdout = false;
#endif
    ceres::Solver::Summary summary;
    Solve(options, blockCeres.problem_.get(), &summary);

#ifdef DEBUG_MODE
    std::cout << "Coco: " << summary.BriefReport() << std::endl;
    CocoStereo::print(this->motionSteps_);
    CocoStereo::print(camera);
#endif
}
bool CocoStereo::isCalibrationPossible() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered isCalibrationPossible()" << std::endl;
#endif
    // no calibration is possible if there is no or little motion
    if (motionSteps_.empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: calibration not possible: no motion yet" << std::endl;
#endif
        return false;
    }

    // compute mean motion absolute
    double meanRotation = 0.;
    double meanTranslation = 0.;
    int validMotionSteps = 0;
    for (MotionModelPtr& model : motionSteps_) {
        if (!model) {
            continue;
        }
        Eigen::Matrix<double, 4, 4> T;
        model->getTransformMatrix(T);
        // extract motion information
        Eigen::AngleAxis<double> axis(T.block<3, 3>(0, 0));
        Eigen::Vector3d translation = T.block<3, 1>(0, 3);

        meanRotation += axis.angle();
        meanTranslation += translation.norm();
        ++validMotionSteps;
    }

    if (!validMotionSteps) {
        // if there's no valid motion, can't calibrate
#ifdef DEBUG_MODE
        std::cout << "Coco: calibration not possible: no valid motion" << std::endl;
#endif
        return false;
    }
    // divide by zero checked before
    meanRotation /= double(validMotionSteps);
    meanTranslation /= double(validMotionSteps);

    if (meanRotation >= params_.minRotationForCalib || meanTranslation >= params_.minTranslationForCalib) {
#ifdef DEBUG_MODE
        std::cout << "Coco: calibration seems possible: " << meanRotation << " " << meanTranslation << std::endl;
#endif
        return true;
    }
#ifdef DEBUG_MODE
    std::cout << "Coco: calibration not possible: not enough motion: " << meanRotation << " " << meanTranslation
              << std::endl;
#endif
    return false;
}
void CocoStereo::makeOptimizationChain(block_optimization::BlockPtr<double>& dataBlock,
                                       std::vector<MotionModelPtr> motion,
                                       std::vector<block_optimization::BlockPtr<double>> camera,
                                       std::vector<double*> cameraParams,
                                       block_optimization::BlockPtr<double>& targetBlock,
                                       block_optimization::ProcessingChainPtr<double>& chain,
                                       std::vector<double*>& paramBlocks) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeOptimizationChain()" << std::endl;
#endif
    if (!chain) {
        chain.reset(new block_optimization::ProcessingChain<double>());
    }

    // build up chain:
    // data
    chain->appendBlock(dataBlock);
    // reserve one spot in front for data pointer
    paramBlocks.push_back(NULL);

    // motion chain
    for (MotionModelPtr& motionModel : motion) {
        std::vector<block_optimization::BlockPtr<double>> thisMotionBlocks = motionModel->getTransformChain();
        for (block_optimization::BlockPtr<double>& thisBlock : thisMotionBlocks) {
            chain->appendBlock(thisBlock);
        }
        std::vector<double*> params = motionModel->getParameterBlocksRaw();
        paramBlocks.insert(paramBlocks.end(), params.begin(), params.end());
    }

    // camera
    for (block_optimization::BlockPtr<double>& blockPtr : camera) {
        chain->appendBlock(blockPtr);
    }
    paramBlocks.insert(paramBlocks.end(), cameraParams.begin(), cameraParams.end());

    // reprojection error
    chain->appendBlock(targetBlock);
    // reserve one spot in back for target pointer
    paramBlocks.push_back(NULL);
}
void CocoStereo::reestimateMotionChain() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered reestimateMotionChain()" << std::endl;
#endif

    if (!params_.bundleAdjustMotion) {
#ifdef DEBUG_MODE
        std::cout << "Coco: reestimateMotionChain() set inactive, returning" << std::endl;
#endif
        return;
    }

    std::shared_ptr<feature_tracking::StereoTrackletList> bucketTracklets;
#ifdef DEBUG_MODE
    std::cout << "Tracklet count before bucketing " << tracklets_->size() << std::endl;
#endif
    if (params_.bucketW > 1 && params_.bucketH > 1) {
        bucketTracklets = std::make_shared<feature_tracking::StereoTrackletList>();
        feature_tracking::utils::bucketing(*tracklets_, params_.bucketW, params_.bucketH, *bucketTracklets);
    } else {
        bucketTracklets = tracklets_;
    }

    currentProcessedCamera_->disableOptimization();
    this->bundleAdjust(currentProcessedCamera_, bucketTracklets);
}


/////////////////////////////////////////////
/// calibration
/////////////////////////////////////////////
void CocoStereo::reestimateProjection() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered reestimateProjection()" << std::endl;
#endif

    if (!params_.reestimateProjection) {
#ifdef DEBUG_MODE
        std::cout << "Coco: reestimateProjection() set inactive, returning" << std::endl;
#endif
        return;
    }

    if (!tracklets_ || tracklets_->empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: tracklets empty in reestimateProjection(), returning " << std::endl;
#endif
        return;
    }
    // TODO: this should perform a remapping of the tracklets without distortion

    // reestimate the projection, i.e. rotation, translation and camera matrices
    // together with the motion
    std::shared_ptr<feature_tracking::StereoTrackletList> bucketTracklets;
#ifdef DEBUG_MODE
    std::cout << "Tracklet count before bucketing " << tracklets_->size() << std::endl;
#endif
    if (params_.bucketW > 1 && params_.bucketH > 1) {
        bucketTracklets = std::make_shared<feature_tracking::StereoTrackletList>();
        feature_tracking::utils::bucketing(*tracklets_, params_.bucketW, params_.bucketH, *bucketTracklets);
    } else {
        bucketTracklets = tracklets_;
    }

    //  currentUndistCamera_->disableOptimization();
    //  this->bundleAdjust(currentUndistCamera_, bucketTracklets);
    currentUndistCamera_->enableOptimization();
    //  if(currentUndistCamera_->getDistortionBlock(true)) {
    //    currentUndistCamera_->getDistortionBlock(true)->disableOptimization();
    //  }
    //  if(currentUndistCamera_->getDistortionBlock(false)) {
    //    currentUndistCamera_->getDistortionBlock(false)->disableOptimization();
    //  }
    this->bundleAdjust(currentUndistCamera_, bucketTracklets);
}
void CocoStereo::refineComplete() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered refineComplete()" << std::endl;
#endif

    if (!params_.refineAll && !params_.bundleAdjustDistorted) {
#ifdef DEBUG_MODE
        std::cout << "Coco: refineComplete() set inactive, returning" << std::endl;
#endif
        return;
    }


    std::shared_ptr<feature_tracking::StereoTrackletList> bucketTracklets;
#ifdef DEBUG_MODE
    std::cout << "Tracklet count before bucketing " << tracklets_->size() << std::endl;
#endif
    if (params_.bucketW > 1 && params_.bucketH > 1) {
        bucketTracklets = std::make_shared<feature_tracking::StereoTrackletList>();
        feature_tracking::utils::bucketing(*tracklets_, params_.bucketW, params_.bucketH, *bucketTracklets);
    } else {
        bucketTracklets = tracklets_;
    }

    // remap tracklets to current raw camera
    this->remapTracklets();

    // reestimate complete motion chain and structure

    if (params_.bundleAdjustDistorted) {
#ifdef DEBUG_MODE
        std::cout << "Coco: bundleAdjustDistorted set active, running" << std::endl;
#endif
        currentRawCamera_->disableOptimization();
        this->bundleAdjust(currentRawCamera_, bucketTracklets);
    }

    if (!params_.refineAll) {
#ifdef DEBUG_MODE
        std::cout << "Coco: refineComplete() set inactive, returning" << std::endl;
#endif
        return;
    }

    // copy current camera parameters from after reestimation of projection
    this->copyParams(currentUndistCamera_, currentRawCamera_);


    // SGD
    block_optimization::SGDSolver<double> solver;
    // make the entire chain, run optimization
    block_optimization::BlockPtr<double> dataBlock =
        block_optimization::BlockPtr<double>(new block_optimization::DataBlock<double, 3>("data"));
    block_optimization::BlockPtr<double> reprojection =
        block_optimization::BlockPtr<double>(new block_optimization::LinearError<double, 2>("err"));

    for (unsigned int i = 0; i < this->motionSteps_.size() + 1; i++) {

        // make motion vector
        std::vector<MotionModelPtr> motionChain;
        for (unsigned int k = 0; k < i; k++) {
            motionChain.push_back(this->motionSteps_[i - k - 1]);
        }

        // make processing chains
        block_optimization::ProcessingChainPtr<double> leftChain;
        std::vector<double*> leftParams;
        this->makeOptimizationChain(dataBlock,
                                    motionChain,
                                    currentRawCamera_->getProjectionChain(true),
                                    currentRawCamera_->getParameterBlocksRaw(true),
                                    reprojection,
                                    leftChain,
                                    leftParams);

        block_optimization::ProcessingChainPtr<double> rightChain;
        std::vector<double*> rightParams;
        this->makeOptimizationChain(dataBlock,
                                    motionChain,
                                    currentRawCamera_->getProjectionChain(false),
                                    currentRawCamera_->getParameterBlocksRaw(false),
                                    reprojection,
                                    rightChain,
                                    rightParams);

        solver.addChain(leftChain, leftParams);
        solver.addChain(rightChain, rightParams);
    }

    int measCount = bucketTracklets->size();
    std::vector<std::vector<double*>> points(measCount);
    std::vector<std::vector<double*>> measurements(measCount);
    feature_tracking::StereoTrackletList::iterator tracklet = bucketTracklets->begin();
    for (int i = 0; i < measCount; i++) {
        feature_tracking::StereoTracklet& thisTracklet = *tracklet;
        double* point = thisTracklet[0].x_->data;
        points[i].resize(2 * (this->motionSteps_.size() + 1));
        measurements[i].resize(2 * (this->motionSteps_.size() + 1));
        for (unsigned int k = 0; k < this->motionSteps_.size() + 1; k++) {
            // check if tracklet supports this track length
            bool pointIsNormal = (std::isnormal(point[0]) || point[0] == 0.0) &&
                                 (std::isnormal(point[1]) || point[1] == 0.0) &&
                                 std::isnormal(point[2]); // I don't allow the z-component to be zero
            if (thisTracklet.size() - 1 < this->motionSteps_.size() || !pointIsNormal) {
                // both, left and right camera
                points[i][2 * k + 0] = NULL;
                points[i][2 * k + 1] = NULL;
                measurements[i][2 * k + 0] = NULL;
                measurements[i][2 * k + 1] = NULL;
            } else {
                // both, left and right camera
                points[i][2 * k + 0] = point;
                points[i][2 * k + 1] = point;
                // add measurements
                measurements[i][2 * k + 0] = new double[2];
                measurements[i][2 * k + 0][0] = thisTracklet[k].p1_.u_;
                measurements[i][2 * k + 0][1] = thisTracklet[k].p1_.v_;
                measurements[i][2 * k + 1] = new double[2];
                measurements[i][2 * k + 1][0] = thisTracklet[k].p2_.u_;
                measurements[i][2 * k + 1][1] = thisTracklet[k].p2_.v_;
            }
        }
        ++tracklet;
    }

    // subparametrization of baseline optimization: set x-component of translation const
    solver.setSubParametrization(currentRawCamera_->getTranslationBlock(false)->getInternalMemory(0).get(), {0});

    int batchSize =
        params_.refinementBatchSize > 0 ? std::min(params_.refinementBatchSize, int(points.size())) : points.size();
    solver.setBatchSize(batchSize);
    solver.setLearningRate(1. / double(batchSize) * params_.refinementLearningRate);

    solver.setParameterBlockConstant(currentRawCamera_->getRotationBlock(false)->getInternalMemory(0).get());
    solver.setParameterBlockConstant(currentRawCamera_->getTranslationBlock(false)->getInternalMemory(0).get());
    if (!params_.estimateTangentialDistortion) {
        if (!params_.estimateK3) {
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(true)->getInternalMemory(0).get(),
                                         {2, 3, 4});
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(false)->getInternalMemory(0).get(),
                                         {2, 3, 4});
        } else {
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(true)->getInternalMemory(0).get(),
                                         {2, 3});
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(false)->getInternalMemory(0).get(),
                                         {2, 3});
        }
    } else {
        if (!params_.estimateK3) {
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(true)->getInternalMemory(0).get(), {4});
            solver.setSubParametrization(currentRawCamera_->getDistortionBlock(false)->getInternalMemory(0).get(), {4});
        }
    }
    for (int i = 0; i < params_.refinementSteps; i++) {
        double thisError = solver.step(points, measurements);
#ifdef DEBUG_MODE
        std::cout << "Coco: SGD step error = " << thisError << std::endl;
#endif
    }

#ifdef DEBUG_MODE
    CocoStereo::print(currentRawCamera_);
#endif

    for (std::vector<double*>& thisMeas : measurements) {
        for (double* data : thisMeas) {
            delete[] data;
            data = NULL;
        }
    }
}
/////////////////////////////////////////////
/// helper functions
/////////////////////////////////////////////
void CocoStereo::copyParams(StereoCameraModelPtr src, StereoCameraModelPtr dst) {
    // distortion
    this->copyParams(src->getDistortionParams(true), dst->getDistortionParams(true), 5);
    this->copyParams(src->getDistortionParams(false), dst->getDistortionParams(false), 5);
    // projection
    this->copyParams(src->getProjectionParams(true), dst->getProjectionParams(true), 4);
    this->copyParams(src->getProjectionParams(false), dst->getProjectionParams(false), 4);
    // rotation
    this->copyParams(src->getRotationParams(true), dst->getRotationParams(true), 3);
    this->copyParams(src->getRotationParams(false), dst->getRotationParams(false), 3);
    // translation
    this->copyParams(src->getTranslationParams(true), dst->getTranslationParams(true), 3);
    this->copyParams(src->getTranslationParams(false), dst->getTranslationParams(false), 3);
}
void CocoStereo::copyParams(std::shared_ptr<double> src, std::shared_ptr<double> dst, unsigned int size) {
    if (!src || !dst || !size) {
        // check data pointers, if not valid, don't copy
        return;
    }
    std::copy(src.get(), src.get() + size, dst.get());
}
bool CocoStereo::checkTime() {
    if (params_.maxTime <= 0.) {
        return true;
    }
    boost::chrono::duration<double> diff = boost::chrono::high_resolution_clock::now() - processingStart_;
#ifdef DEBUG_MODE
    std::cout << "Processing time so far is " << diff.count() << std::endl;
#endif
    return diff.count() < params_.maxTime;
}
void CocoStereo::makeCameraModels(std::string file) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeCameraModels() from yaml" << std::endl;
#endif
    // setup camera output
    currentRawCamera_ = std::shared_ptr<StereoCameraModel>(new PinholeStereoCamera());

    if (!fs::exists(file)) {
        throw std::runtime_error("Camera cfg file doesn't exist: " + file);
    }
    YAML::Node camCfg = YAML::LoadFile(file);

    // read the parameter vectors from yaml file
    std::vector<double> R1 = camCfg["R1"].as<std::vector<double>>();
    std::vector<double> R2 = camCfg["R2"].as<std::vector<double>>();
    std::vector<double> t1 = camCfg["T1"].as<std::vector<double>>();
    std::vector<double> t2 = camCfg["T2"].as<std::vector<double>>();
    std::vector<double> K1 = camCfg["K1"].as<std::vector<double>>();
    std::vector<double> K2 = camCfg["K2"].as<std::vector<double>>();
    std::vector<double> d1 = camCfg["D1"].as<std::vector<double>>();
    std::vector<double> d2 = camCfg["D2"].as<std::vector<double>>();

    // parse projection parameters from matrix
    currentRawCamera_->getProjectionParams(true).get()[0] = K1[0];
    currentRawCamera_->getProjectionParams(true).get()[1] = K1[2];
    currentRawCamera_->getProjectionParams(true).get()[2] = K1[4];
    currentRawCamera_->getProjectionParams(true).get()[3] = K1[5];
    // copy distortion parameter vector
    std::copy(d1.begin(), d1.end(), currentRawCamera_->getDistortionParams(true).get());
    // parse projection parameters from matrix
    currentRawCamera_->getProjectionParams(false).get()[0] = K2[0];
    currentRawCamera_->getProjectionParams(false).get()[1] = K2[2];
    currentRawCamera_->getProjectionParams(false).get()[2] = K2[4];
    currentRawCamera_->getProjectionParams(false).get()[3] = K2[5];
    // copy distortion parameter vector
    std::copy(d2.begin(), d2.end(), currentRawCamera_->getDistortionParams(false).get());
    // make rodriguez rotation from rotation matrix
    Eigen::Matrix<double, 3, 3> R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = R2[i * 3 + j];
        }
    }
    Eigen::AngleAxisd angleAx;
    angleAx.fromRotationMatrix(R);
    Eigen::Vector3d axis = angleAx.axis();
    axis *= angleAx.angle();
    currentRawCamera_->getRotationParams(false).get()[0] = axis(0);
    currentRawCamera_->getRotationParams(false).get()[1] = axis(1);
    currentRawCamera_->getRotationParams(false).get()[2] = axis(2);
    // copy the baseline vector
    std::copy(t2.begin(), t2.end(), currentRawCamera_->getTranslationParams(false).get());
    // make the final mapping functions for the other camera models from the raw camera
    this->makeMapping();
}
void CocoStereo::makeCameraModels(const cv::Mat& im) {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered makeCameraModels() from image" << std::endl;
#endif

    // create a new camera
    currentRawCamera_ = std::shared_ptr<StereoCameraModel>(new PinholeStereoCamera());
    // get some initial guesses:
    // center point is in the middle of the image
    double cx = double(im.cols) / 2.;
    double cy = double(im.rows) / 2.;
    // focal length is half the image width: 90° FOV
    double f = cx;

    // default baseline
    double b = -1.;

    // yet, the user may have specified values: if so, take that
    if (params_.baseLine) {
        b = params_.baseLine.get();
    }
    if (params_.focalLength) {
        f = params_.focalLength.get();
    }
    if (params_.centerX) {
        cx = params_.centerX.get();
    }
    if (params_.centerY) {
        cy = params_.centerY.get();
    }

    // set projection left
    currentRawCamera_->getProjectionParams(true).get()[0] = f;
    currentRawCamera_->getProjectionParams(true).get()[1] = cx;
    currentRawCamera_->getProjectionParams(true).get()[2] = f;
    currentRawCamera_->getProjectionParams(true).get()[3] = cy;
    // set projection right
    currentRawCamera_->getProjectionParams(false).get()[0] = f;
    currentRawCamera_->getProjectionParams(false).get()[1] = cx;
    currentRawCamera_->getProjectionParams(false).get()[2] = f;
    currentRawCamera_->getProjectionParams(false).get()[3] = cy;

    // set baseline
    currentRawCamera_->getTranslationParams(false).get()[0] = b;

    this->makeMapping();
}
void CocoStereo::initCameraModels() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered initCameraModels()" << std::endl;
#endif

    if (params_.readLatest && !params_.cameraFilePrefix.empty()) {
#ifdef DEBUG_MODE
        std::cout << "Coco: trying to init camera models from latest file" << std::endl;
#endif
        // identify folder first:
        std::string folder;
        if (fs::exists(params_.cameraFilePrefix) && fs::is_directory(params_.cameraFilePrefix)) {
            folder = params_.cameraFilePrefix;
        } else {
            std::size_t found = params_.cameraFilePrefix.find_last_of("/\\");
            folder = params_.cameraFilePrefix.substr(0, found);
        }
#ifdef DEBUG_MODE
        std::cout << "Coco: searching for file in folder" << folder << std::endl;
#endif
        // try to find newest file with given prefix
        fs::directory_iterator file(folder);
        fs::path newestPath;
        time_t newestTime = 0;
        while (file != fs::directory_iterator()) {
            std::string path = file->path().string();
#ifdef DEBUG_MODE
            std::cout << "Coco: checking file " << path << std::endl;
#endif
            if (fs::is_regular_file(file->path()) &&
                !path.compare(0, params_.cameraFilePrefix.size(), params_.cameraFilePrefix) &&
                file->path().extension() == ".yaml" && fs::last_write_time(file->path()) > newestTime) {
                newestPath = file->path();
                newestTime = fs::last_write_time(file->path());
#ifdef DEBUG_MODE
                std::cout << "Coco: newest file " << newestPath.string() << std::endl;
#endif
            }
            ++file;
        }

        // if found, set this as file, if not, don't do anything
        if (newestTime > 0 && !newestPath.empty()) {
            params_.cameraFile = newestPath.string();
        }
    }
    if (!params_.cameraFile.empty() && fs::exists(params_.cameraFile)) {
#ifdef DEBUG_MODE
        std::cout << "Coco: init camera models from " << params_.cameraFile << std::endl;
#endif
        // if exists, parse
        this->makeCameraModels(params_.cameraFile);
    } else {
#ifdef DEBUG_MODE
        std::cout << "Coco: nothing specified to init camera models from, waiting for first set of images "
                  << std::endl;
#endif
        // not successful: reset camera to make sure that everything is as expected
        currentRawCamera_.reset();
    }
}
void CocoStereo::exportCameraParameters() {
#ifdef DEBUG_MODE
    std::cout << "Coco: entered exportCameraParameters()" << std::endl;
#endif
    // make file name
    std::string fileName = (boost::format("%s_%d.yaml") % params_.cameraFilePrefix % processCount_).str();
#ifdef DEBUG_MODE
    std::cout << "Coco: writing to " << fileName << std::endl;
#endif
    // make Yaml emitter
    YAML::Emitter emitter;
    // left camera extrinsics
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "R1";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << 1. << 0. << 0. << 0. << 1. << 0. << 0. << 0. << 1.
            << YAML::EndSeq;
    emitter << YAML::Key << "T1";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << 0. << 0. << 0. << YAML::EndSeq;
    // left camera intrinsics
    emitter << YAML::Key << "K1";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << currentRawCamera_->getProjectionParams(true).get()[0]
            << 0. << currentRawCamera_->getProjectionParams(true).get()[1] << 0.
            << currentRawCamera_->getProjectionParams(true).get()[2]
            << currentRawCamera_->getProjectionParams(true).get()[3] << 0. << 0. << 1. << YAML::EndSeq;
    emitter << YAML::Key << "D1";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << currentRawCamera_->getDistortionParams(true).get()[0]
            << currentRawCamera_->getDistortionParams(true).get()[1]
            << currentRawCamera_->getDistortionParams(true).get()[2]
            << currentRawCamera_->getDistortionParams(true).get()[3]
            << currentRawCamera_->getDistortionParams(true).get()[4] << YAML::EndSeq;
    // right camera extrinsics
    // TODO:
    Eigen::Matrix<double, 3, 3> R;
    // extrinsic transform
    std::shared_ptr<block_optimization::Rotation<double>> rotation =
        std::dynamic_pointer_cast<block_optimization::Rotation<double>>(currentRawCamera_->getRotationBlock(false));
    rotation->getRotationMatrix(currentRawCamera_->getRotationParams(false).get(), R);
    emitter << YAML::Key << "R2";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << R(0, 0) << R(0, 1) << R(0, 2) << R(1, 0) << R(1, 1)
            << R(1, 2) << R(2, 0) << R(2, 1) << R(2, 2) << YAML::EndSeq;
    emitter << YAML::Key << "T2";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << currentRawCamera_->getTranslationParams(false).get()[0]
            << currentRawCamera_->getTranslationParams(false).get()[1]
            << currentRawCamera_->getTranslationParams(false).get()[2] << YAML::EndSeq;
    // right camera intrinsics
    emitter << YAML::Key << "K2";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << currentRawCamera_->getProjectionParams(false).get()[0]
            << 0. << currentRawCamera_->getProjectionParams(false).get()[1] << 0.
            << currentRawCamera_->getProjectionParams(false).get()[2]
            << currentRawCamera_->getProjectionParams(false).get()[3] << 0. << 0. << 1. << YAML::EndSeq;
    emitter << YAML::Key << "D2";
    emitter << YAML::Value << YAML::Flow << YAML::BeginSeq << currentRawCamera_->getDistortionParams(true).get()[0]
            << currentRawCamera_->getDistortionParams(false).get()[1]
            << currentRawCamera_->getDistortionParams(false).get()[2]
            << currentRawCamera_->getDistortionParams(false).get()[3]
            << currentRawCamera_->getDistortionParams(false).get()[4] << YAML::EndSeq;
    emitter << YAML::EndMap;

    std::fstream outputFile(fileName, std::fstream::out);
    outputFile << emitter.c_str();
    outputFile.close();
}

/////////////////////////////////////////////
/// debug functions
/////////////////////////////////////////////
void CocoStereo::print(StereoCameraModelPtr camera) {
    std::vector<block_optimization::BlockPtr<double>> c = camera->getProjectionChain(true);
    CocoStereo::print(c);
    c = camera->getProjectionChain(false);
    CocoStereo::print(c);
}
void CocoStereo::print(MotionModelPtr motion) {
    std::vector<block_optimization::BlockPtr<double>> c = motion->getTransformChain();
    CocoStereo::print(c);
}
void CocoStereo::print(boost::circular_buffer<MotionModelPtr>& motionChain) {
    for (MotionModelPtr m : motionChain) {
        CocoStereo::print(m);
    }
}
void CocoStereo::print(std::vector<block_optimization::BlockPtr<double>>& blockChain) {
    for (block_optimization::BlockPtr<double>& b : blockChain) {
        if (!b) {
            continue;
        }
        std::vector<int> paramCount = b->getParamCount();
        if (paramCount.empty()) {
            continue;
        }
        std::vector<std::shared_ptr<double>> memory = b->getInternalMemory();
        std::cout << b->getName() << ": ";
        for (unsigned int i = 0; i < paramCount.size(); i++) {
            if (!memory[i] || !paramCount[i]) {
                continue;
            }
            double* dat = memory[i].get();
            for (int k = 0; k < paramCount[i]; k++) {
                std::cout << " " << dat[k];
            }
            std::cout << std::endl;
        }
    }
}


} // namespace coco
