#ifndef TRACK_H
#define TRACK_H

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <vector>
#include <list>
#include <string>

// adopted from Wang: Action recognition by dense trajectories

struct Config {
    // parameters for descriptors
    static int patch_size;
    static int nxy_cell;
    static int nt_cell;
    static bool fullOrientation;
    static float epsilon;
    static float min_flow;

    // parameters for tracking
    static int start_frame;
    static int end_frame;
    static double quality;
    static double min_distance;
    static int init_gap;
    static int track_length;

    // parameters for the trajectory descriptor
    static float min_var;
    static float max_var;
    static float max_dis;

    // parameters for multi-scale
    static int scale_num;
    static int start_scale;
    static float scale_stride;

    // parameters for background motion removal
    static float bgorient_stride;
    static float bgmag_stride_ratio;
};

typedef struct TrackerInfo
{
    TrackerInfo(int trackLength_, int initGap_)
        :trackLength(trackLength_), initGap(initGap_){}
    int trackLength; // length of the trajectory
    int initGap; // initial gap for feature detection
}TrackerInfo;

class DescInfo
{
public:
    DescInfo(int nBins_, int flag_, int orientation_, int size_, int nxy_cell_, int nt_cell_)
        :nBins(nBins_), fullOrientation(orientation_), norm(2), threshold(Config::min_flow), flagThre(flag_),
          nxCells(nxy_cell_), nyCells(nxy_cell_), ntCells(nt_cell_),
          dim(nBins*nxCells*nyCells), blockHeight(size_), blockWidth(size_)
    {}

public:
    int nBins; // number of bins for vector quantization
    int fullOrientation; // 0: 180 degree; 1: 360 degree
    int norm; // 1: L1 normalization; 2: L2 normalization
    float threshold; //threshold for normalization
    int flagThre; // whether thresholding or not
    int nxCells; // number of cells in x direction
    int nyCells;
    int ntCells;
    int dim; // dimension of the descriptor
    int blockHeight; // size of the block for computing the descriptor
    int blockWidth;

public:

};

class DescMat
{
public:
    int height;
    int width;
    int nBins;
    cv::Mat* desc;

public:
    DescMat(int height_, int width_, int nBins_)
        :height(height_), width(width_), nBins(nBins_)
    {
        desc = new cv::Mat(height_, width_, CV_32FC(nBins_));
        //desc->setTo(cv::Vec<float, nBins_>::all(0.0f));
    }
    ~DescMat()
    {
        desc->release();
        desc = 0;
    }
};

class PointDesc
{
public:
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    cv::Point2f point;

public:
    PointDesc(const DescInfo& hogInfo, const DescInfo& hofInfo, const DescInfo& mbhInfo, const cv::Point2f& point_)
        : hog(hogInfo.nxCells * hogInfo.nyCells * hogInfo.nBins),
        hof(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
        mbhX(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        mbhY(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        point(point_)
    {}
};

class Track
{
public:
    std::list<PointDesc> pointDescs;  // descriptors for all the points
    int maxNPoints;  // max number of points
    int lastFrame;   // frame no. of last point
    int scale;       // pyramid scale the track is with

    Track(int maxNPoints_)
        : maxNPoints(maxNPoints_)
    {}

    void addPointDesc(const PointDesc& point)
    {
        pointDescs.push_back(point);
        if ((int)(pointDescs.size()) > maxNPoints + 2) {
            pointDescs.pop_front();
        }
    }
};


#endif // TRACK_H
