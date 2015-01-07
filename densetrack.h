#ifndef DENSETRACK_H
#define DENSETRACK_H

/* The implementation of the desne trajectories is adapted from Heng Wang's Dense 
   Trajectory Video Description 
   H. Wang, A. Klaser, C. Schmid and C-L. Liu. Action Recognition by Dense Trajectories
   CVPR 2011. (http://lear.inrialpes.fr/people/wang/dense_trajectories)
   
*/

#include "track.h"
#include "imagepyramid.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

class DenseTrack
{
public:
    DenseTrack(int showtrack = 1, std::string path = "");
    ~DenseTrack() { free(fscales); }

public:
    // generic operations
    cv::Rect getRect(const cv::Point2f& point, const cv::Size& size, const DescInfo& descInfo);
    std::vector<float> getDesc(const DescMat& descMat, const cv::Rect& rect, const DescInfo& descInfo);

    // compute descriptor integral histogram
    void HogComp (const cv::Mat& img, DescMat& descMat);
    void HofComp (const cv::Mat& flow, DescMat& descMat);
    void MbhComp (const cv::Mat& flow, DescMat& descMatX, DescMat& descMatY);

    // dense sampling
    void sampleDense(const cv::Mat &grey, const double quanlity, const double min_dist, std::vector<cv::Point2f>& points_out);
    void sampleDense(const cv::Mat &grey, const std::vector<cv::Point2f>& points_in, const double quality, const double min_dist, std::vector<cv::Point2f>& points_out);
    void trackOpticalFlow(const cv::Mat& flow, const std::vector<cv::Point2f>& points_in, std::vector<cv::Point2f>& points_out, std::vector<int>& status);

    // extract dense trajectories
    void operator()(std::string strVideo, std::vector<std::list<Track> >& xyScaleTracks);
    void saveTrackFeatures(Track &track, int lastFrame, int id);
    void saveTrackFeaturesCSV(Track &track, int lastFrame, int id);

    // setter&getter
    inline std::string getFeaturePath() const {return pathFeatures;}

private:
    void buildDescIntegralHist(const cv::Mat& xComp, const cv::Mat& yComp, const DescInfo& descInfo, DescMat &descMat);
    void drawtrajectory(std::list<PointDesc>& descs, int level, cv::Mat& image);
    void detectFeaturePoints(const cv::Mat &grey, const std::vector<cv::Point2f>& points_in, int level, std::list<Track> &tracks);
    void calcVelocityHist(const std::vector<cv::Point2f>& points_in, const std::vector<cv::Point2f>& points_out, int level, std::vector<float>& orienthist, std::vector<float>& maghist);
    bool isValidTrack(std::vector<cv::Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
    void normalizeTrack(const std::vector<cv::Point2f>& track, bool normShape, bool normLength, std::vector<cv::Point2f>& trackUptd);
    float orientPoint(const cv::Point2f& p1, const cv::Point2f& p2, int level=0);
    float distPoint(const cv::Point2f& p1, const cv::Point2f& p2, int level=0);
    float normVec(const std::vector<float> &vec);

protected:
    int show_track;     // if visualize the trajectories
    float* fscales;     // float scale values;
    std::string pathFeatures; // file to save the resulting features

    TrackerInfo tracker;// info about tracker
    DescInfo hogInfo;   // parameters about HoG descriptor
    DescInfo hofInfo;   // parameters about HoF descriptor
    DescInfo mbhInfo;   // parameters about MBH descriptor
};

#endif // DENSETRACK_H
