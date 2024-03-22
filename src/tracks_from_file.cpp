#include "internal/tracks_from_file.h"
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;

namespace feature_tracking {


TracksFromFile::TracksFromFile(TracksFromFile::Parameters params) {
    params_ = params;

    fs::directory_iterator fileIt(params_.folder);
    fs::directory_iterator endIt;
    fileList_.clear();

    while (fileIt != endIt) {
        if (fileIt->path().extension().string() != ".txt") {
            ++fileIt;
            continue;
        }
        fileList_.push_back(fileIt->path().string());
        ++fileIt;
    }

    std::sort(fileList_.begin(), fileList_.end());
    fileIndex_ = 0;
}

void TracksFromFile::pushBack(cv::Mat& im1, cv::Mat& im2) {
    // std::cout << "Pushback" << std::endl;
    std::list<StereoMatch> points;
    // std::cout << "Read file" << std::endl;
    this->getPointsInFile(fileList_[fileIndex_], points);
    // std::cout << "Associate tracks" << std::endl;
    this->associateToTracks(points);
    // std::cout << "Associate matches" << std::endl;
    this->associateToMatches(points);
    // // std::cout<< "in list: " << std::endl;
    for (const StereoTracklet& tracklet : tracklets_) {
        // // std::cout<< tracklet.front().p1_.index_ << " ";
    }
    // // std::cout<< std::endl;

    lastPoints_ = points;
    ++fileIndex_;
    if (fileIndex_ >= fileList_.size()) {
        fileIndex_ = 0;
        lastPoints_.clear();
        tracklets_.clear();
    }
}

void TracksFromFile::getPointsInFile(const std::string& file, std::list<StereoMatch>& points) {
    // // std::cout<< "Reading file " << file << std::endl;
    std::fstream pointFile(file, std::fstream::in);
    points.clear();

    while (!pointFile.eof()) {
        StereoMatch thisMatch;
        pointFile >> thisMatch.p1_.index_ >> thisMatch.p1_.u_ >> thisMatch.p1_.v_ >> thisMatch.p2_.u_ >>
            thisMatch.p2_.v_;
        thisMatch.p2_.index_ = thisMatch.p1_.index_;
        if (pointFile.good()) {
            points.push_back(thisMatch);
            //      // // std::cout<< thisMatch.p1_.index_ << " " <<  thisMatch.p1_.u_ << " " <<  thisMatch.p1_.v_ << "
            //      " <<  thisMatch.p2_.u_ << " " <<  thisMatch.p2_.v_ << std::endl;
        }
    }
}

void TracksFromFile::associateToTracks(std::list<StereoMatch>& points) {
    if (tracklets_.empty() || points.empty()) {
        return;
    }
    // sort both lists
    tracklets_.sort(
        [](const StereoTracklet& l, const StereoTracklet& r) { return l.front().p1_.index_ < r.front().p1_.index_; });
    // // std::cout<< "in list: " << std::endl;
    for (const StereoTracklet& tracklet : tracklets_) {
        // // std::cout<< tracklet.front().p1_.index_ << " ";
    }
    // // std::cout<< std::endl;


    points.sort([](const StereoMatch& l, const StereoMatch& r) { return l.p1_.index_ < r.p1_.index_; });

    // iterate through both
    std::list<StereoMatch>::iterator pointsIt = points.begin();
    std::list<StereoTracklet>::iterator trackletsIt = tracklets_.begin();
    while (pointsIt != points.end()) {
        // std::cout<< "Point id " << pointsIt->p1_.index_ << std::endl;
        while (trackletsIt != tracklets_.end() && trackletsIt->front().p1_.index_ < pointsIt->p1_.index_) {
            // std::cout<< "Erasing Tracklet id " << trackletsIt->front().p1_.index_ << std::endl;
            trackletsIt = tracklets_.erase(trackletsIt);
        }
        if (trackletsIt != tracklets_.end()) {
            // std::cout<< "Tracklet id " << trackletsIt->front().p1_.index_ << std::endl;
        }
        if (trackletsIt != tracklets_.end() && trackletsIt->front().p1_.index_ == pointsIt->p1_.index_) {
            // std::cout<< "Associated with " << trackletsIt->front().p1_.index_ << std::endl;
            trackletsIt->push_front(*pointsIt);
            ++trackletsIt->age_;
            if (params_.maxTracklength > 0 && trackletsIt->size() > params_.maxTracklength) {
                // limit track length
                trackletsIt->pop_back();
            }
            pointsIt = points.erase(pointsIt);
            ++trackletsIt;
        } else {
            ++pointsIt;
        }
        //++trackletsIt;
    }
    if (trackletsIt != tracklets_.end()) {
        tracklets_.erase(trackletsIt, tracklets_.end());
    }
}

void TracksFromFile::associateToMatches(std::list<StereoMatch>& points) {
    if (points.empty() || lastPoints_.empty()) {
        return;
    }
    // sort both lists
    lastPoints_.sort([](const StereoMatch& l, const StereoMatch& r) { return l.p1_.index_ < r.p1_.index_; });
    points.sort([](const StereoMatch& l, const StereoMatch& r) { return l.p1_.index_ < r.p1_.index_; });

    std::list<StereoMatch>::iterator newPointsIt = points.begin();
    std::list<StereoMatch>::iterator oldPointsIt = lastPoints_.begin();

    while (newPointsIt != points.end()) {
        // // std::cout<< "Point id " << newPointsIt->p1_.index_ << std::endl;
        while (oldPointsIt != lastPoints_.end() && oldPointsIt->p1_.index_ < newPointsIt->p1_.index_) {
            // // std::cout<< "Old point id " << oldPointsIt->p1_.index_ << std::endl;
            ++oldPointsIt;
        }
        // // std::cout<< "Old point id " << oldPointsIt->p1_.index_ << std::endl;
        if (oldPointsIt != lastPoints_.end() && oldPointsIt->p1_.index_ == newPointsIt->p1_.index_) {
            // // std::cout<< "Associated" << std::endl;
            tracklets_.push_back(StereoTracklet());
            tracklets_.back().push_front(*oldPointsIt);
            tracklets_.back().push_front(*newPointsIt);
            newPointsIt = points.erase(newPointsIt);
            ++oldPointsIt;
        } else {
            ++newPointsIt;
        }
    }
}


} // namespace feature_tracking
