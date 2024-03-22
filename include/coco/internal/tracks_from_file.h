#include <string>
#include <feature_tracking/stereo_tracker.h>

namespace feature_tracking {

///////////////////////////////////
/// \brief The StereoTrackerLibViso class
/// alows feature tracking of LibViso
/// stereo feature dectection.
///
/// The structure is essentially that of
/// the LibViso matcher. Images are pushed
/// into internal memory. For those, features are
/// computed. Macthing between features of subsequent
/// frame pairs happens upon push back of a second image pair.
/// Tracklets can be obtained by using the getTracklet
/// method.
class TracksFromFile : public StereoTracker {
public:
    struct Parameters {
        std::string folder;
        int maxTracklength;
        Parameters() {
            maxTracklength = 3;
        }
    };

    TracksFromFile(Parameters params = Parameters());

    virtual void pushBack(cv::Mat& im1, cv::Mat& im2);

protected:
    void getPointsInFile(const std::string& file, std::list<StereoMatch>& points);
    void associateToTracks(std::list<StereoMatch>& points);
    void associateToMatches(std::list<StereoMatch>& points);

    Parameters params_;
    int fileIndex_;
    std::vector<std::string> fileList_;
    std::list<StereoMatch> lastPoints_;
};

} // namespace feature_tracking
