#ifndef IMAGEPYRAMID_H
#define IMAGEPYRAMID_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImagePyramid
{
public:
    ImagePyramid();
    ImagePyramid(const ImagePyramid& pyramid);
    ImagePyramid(const cv::Mat& image, double scaleFactor);
    ImagePyramid(cv::Size initSize, int depth, int nChannels, double scaleFactor);
    ~ImagePyramid();

    ImagePyramid& operator=(const ImagePyramid& pyramid);
    operator const bool() const;
    operator bool();
    /**
     * round == 0  =>  take the closest level (i.e., rounding)
     * round < 0   =>  take the next level with a smaller factor (i.e., flooring)
     * round > 0   =>  take the next level with a bigger factor (i.e., ceiling)
     */
    std::size_t getIndex(double scaleFactor, int round = 0) const;

    std::size_t numOfLevels() const;
    double getScaleFactor() const;
    double getScaleFactor(std::size_t index) const;
    double getScaleFactorInv(std::size_t index) const;
    double getXScaleFactor(std::size_t index) const;
    double getXScaleFactorInv(std::size_t index) const;
    double getYScaleFactor(std::size_t index) const;
    double getYScaleFactorInv(std::size_t index) const;

    cv::Mat& getImage(std::size_t index);
    const cv::Mat& getImage(std::size_t index) const;

    /**
     * @para round see getIndex()
     */
    cv::Mat& getImage(double scaleFactor, int round = 0);
    const cv::Mat& getImage(double scaleFactor, int round = 0) const;

    /**
     * rebuilds the pyramid (re-using the already allocated space) with the given
     * image
     * NOTE: this image needs to have the exact sames size as the initial scale
     */
    void rebuild(cv::Mat image);

private:
    void init(cv::Mat image, double scaleFactor);
    /**
     * build an empty pyramid (pixel values are set to zeros).
     */
    void init(cv::Size initSize, int depth, int nChannels, double scaleFactor);

protected:
    std::vector<cv::Mat> imagePyramid;
    std::vector<double> scaleFactors;
    std::vector<double> scaleFactorsInv;
    std::vector<double> xScaleFactors;     //? why different scales for x, y and whole
    std::vector<double> xScaleFactorsInv;
    std::vector<double> yScaleFactors;
    std::vector<double> yScaleFactorsInv;
    double scaleFactor;
    double epsilon;
};

#include "imagepyramid.hpp"

#endif // IMAGEPYRAMID_H
