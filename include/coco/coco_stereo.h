#pragma once

#include <list>
#include <memory>
#include <random>

#include <block_optimization/processing_chain.h>

#include <boost/chrono.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>

#include <feature_tracking/stereo_tracker.h>

#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include "motion_model.h"
#include "stereo_camera_model.h"


namespace coco {

class CocoStereo {
public:
    /////////////////////////////////////////////
    /// Data types for internal processing
    /////////////////////////////////////////////
    enum PreprocType { PREPROC_NONE, PREPROC_UNDISTORT, PREPROC_RECTIFY, PREPROC_ALIGN };

    struct Parameters {
        PreprocType preproc;
        unsigned int maxTrackLength;
        int bucketH;
        int bucketW;
        int motionRansacIters;
        double motionRansacBaseThresh;
        double motionRansacStartThresh;
        double motionRansacTimeConst;
        double minRotationForCalib;
        double minTranslationForCalib;
        bool reestimateProjection;
        bool bundleAdjustMotion;
        bool bundleAdjustDistorted;
        bool refineAll;
        int refinementSteps;
        int refinementBatchSize;
        double refinementLearningRate;
        double projectionRegularization;
        double extrinsicsRegularization;
        double sphericalProjectionRegularization;
        double distortionRegularization;
        bool visualizeMatches;
        bool estimateTangentialDistortion;
        bool estimateK3;
        double maxTime;
        bool calibrate;
        std::string cameraFile;
        int exportRate;
        bool exportOnShutdown;
        bool readLatest;
        std::string cameraFilePrefix;
        boost::optional<double> baseLine;
        boost::optional<double> focalLength;
        boost::optional<double> centerX;
        boost::optional<double> centerY;
        bool printParams;

        Parameters() {
            preproc = PREPROC_NONE;
            bucketH = 50;
            bucketW = 100;
            motionRansacIters = 10;
            motionRansacBaseThresh = 5;
            motionRansacStartThresh = 20;
            motionRansacTimeConst = 0.01;
            maxTrackLength = 2;
            minRotationForCalib = 0.04;
            minTranslationForCalib = 1.;
            visualizeMatches = true;
            reestimateProjection = true;
            bundleAdjustMotion = true;
            bundleAdjustDistorted = true;
            refineAll = true;
            refinementSteps = 10;
            refinementBatchSize = -1;
            refinementLearningRate = 1e-8;
            projectionRegularization = 0.;
            extrinsicsRegularization = 0.;
            sphericalProjectionRegularization = 0.;
            maxTime = -1.;
            calibrate = true;
            cameraFile = "";
            exportOnShutdown = false;
            exportRate = -1;
            readLatest = false;
            cameraFilePrefix = "";
            printParams = true;
        }

        Parameters(const YAML::Node cfg);
    };


    /////////////////////////////////////////////
    /// Construction, destruction,
    /// parameter parsing
    /////////////////////////////////////////////
    CocoStereo(Parameters params = Parameters());
    ~CocoStereo();

    /////////////////////////////////////////////
    /// processing
    /////////////////////////////////////////////
    virtual void process(const cv::Mat& im1, const cv::Mat& im2);

    /////////////////////////////////////////////
    /// data access
    /////////////////////////////////////////////
    virtual void getCameraParams(cv::Mat& c1, cv::Mat& d1, cv::Mat& c2, cv::Mat& d2, cv::Mat& R, cv::Mat& t);


protected:
    /////////////////////////////////////////////
    /// image processing
    /////////////////////////////////////////////
    virtual void prepareImages(const cv::Mat& im1, const cv::Mat& im2);
    virtual void makeMapping();
    virtual void clearMapping();
    virtual void makeUndistortionMapping();
    virtual void makeRectificationMapping();
    virtual void makeAlignmentMapping();
    virtual void computeMatches();
    virtual void prepareTracklets();
    virtual void remapTracklets();
    virtual void computeReprojectionError(std::vector<double*>& worldPoints,
                                          std::vector<double*>& observ,
                                          block_optimization::ProcessingChainPtr<double>& chain,
                                          std::vector<double*>& paramBlocks,
                                          std::vector<double>& reprojectionErr);


    /////////////////////////////////////////////
    /// motion estimation
    /////////////////////////////////////////////
    virtual void estimateMotion();
    virtual int ransacTransform(block_optimization::ProcessingChainPtr<double>& chain,
                                std::vector<double*> params,
                                MotionModelPtr motionModel,
                                std::vector<double*>& worldPoints,
                                std::vector<double*>& observations,
                                std::vector<bool>& inliers);
    virtual int findInliers(block_optimization::ProcessingChainPtr<double>& chain,
                            std::vector<double*>& paramBlocks,
                            std::vector<double*>& worldPoints,
                            std::vector<double*>& observations,
                            std::vector<bool>& inliers);
    virtual bool isCalibrationPossible();
    virtual void makeOptimizationChain(block_optimization::BlockPtr<double>& dataBlock,
                                       std::vector<MotionModelPtr> motion,
                                       std::vector<block_optimization::BlockPtr<double>> camera,
                                       std::vector<double*> cameraParams,
                                       block_optimization::BlockPtr<double>& targetBlock,
                                       block_optimization::ProcessingChainPtr<double>& chain,
                                       std::vector<double*>& paramBlocks);
    virtual void reestimateMotionChain();


    /////////////////////////////////////////////
    /// calibration
    /////////////////////////////////////////////
    virtual void reestimateProjection();
    virtual void refineComplete();
    virtual void bundleAdjust(StereoCameraModelPtr camera,
                              std::shared_ptr<feature_tracking::StereoTrackletList> tracklets);

    /////////////////////////////////////////////
    /// helper functions
    /////////////////////////////////////////////
    virtual void copyParams(StereoCameraModelPtr src, StereoCameraModelPtr dst);
    virtual void copyParams(std::shared_ptr<double> src, std::shared_ptr<double> dst, unsigned int size);
    virtual bool checkTime();
    virtual void makeCameraModels(const std::string file);
    virtual void makeCameraModels(const cv::Mat& im1);
    virtual void initCameraModels();
    virtual void exportCameraParameters();

    /////////////////////////////////////////////
    /// debug functions
    /////////////////////////////////////////////
    static void print(StereoCameraModelPtr camera);
    static void print(MotionModelPtr motion);
    static void print(boost::circular_buffer<MotionModelPtr>& motionChain);
    static void print(std::vector<block_optimization::BlockPtr<double>>& blockChain);

    /////////////////////////////////////////////
    /// member variables
    /////////////////////////////////////////////
    unsigned int processCount_;
    cv::Mat im1orig_, im2orig_;                     ///< internal images, as in original
    cv::Mat im1Remapped_, im2Remapped_;             ///< internal images, remapped, whatever that may mean
    cv::Mat im1MapU_, im1MapV_, im2MapU_, im2MapV_; ///< mapping for both images

    Parameters params_;                                  ///< settings
    feature_tracking::StereoTrackerPtr featureTracking_; ///< tracking of features
    boost::circular_buffer<MotionModelPtr> motionSteps_;
    StereoCameraModelPtr currentRawCamera_;
    StereoCameraModelPtr currentProcessedCamera_;
    StereoCameraModelPtr currentUndistCamera_;
    boost::chrono::high_resolution_clock::time_point processingStart_;

    std::shared_ptr<feature_tracking::StereoTrackletList> tracklets_;
    std::vector<std::pair<cv::Mat, cv::Mat>> visualizationIms_;
    std::mt19937_64 rng_;
};

} // namespace coco
