#ifndef IMAGEPYRAMID_HPP
#define IMAGEPYRAMID_HPP

#include "imagepyramid.h"

#include <cmath>
#include <cassert>

inline ImagePyramid::ImagePyramid()
    :imagePyramid(), scaleFactors(), scaleFactorsInv(), xScaleFactors(), xScaleFactorsInv(),
      yScaleFactors(), yScaleFactorsInv(), scaleFactor(0), epsilon(0)
{}

inline ImagePyramid::ImagePyramid(const ImagePyramid &pyramid)
    :imagePyramid(pyramid.imagePyramid.begin(), pyramid.imagePyramid.end()),
      scaleFactors(pyramid.scaleFactors.begin(), pyramid.scaleFactors.end()),
      scaleFactorsInv(pyramid.scaleFactorsInv.begin(), pyramid.scaleFactorsInv.end()),
      xScaleFactors(pyramid.xScaleFactors.begin(), pyramid.xScaleFactors.end()),
      xScaleFactorsInv(pyramid.xScaleFactorsInv.begin(), pyramid.xScaleFactorsInv.end()),
      yScaleFactors(pyramid.yScaleFactors.begin(), pyramid.yScaleFactors.end()),
      yScaleFactorsInv(pyramid.yScaleFactorsInv.begin(), pyramid.yScaleFactorsInv.end()),
      scaleFactor(pyramid.scaleFactor), epsilon(pyramid.epsilon)
{}

inline ImagePyramid::ImagePyramid(const cv::Mat &image, double scaleFactor)
{
    assert(!image.empty());
    init(image, scaleFactor);
}

inline ImagePyramid::ImagePyramid(cv::Size initSize, int depth, int nChannels, double scaleFactor)
{
    init(initSize, depth, nChannels, scaleFactor);
}

inline ImagePyramid::~ImagePyramid()
{}

inline ImagePyramid& ImagePyramid::operator =(const ImagePyramid& pyramid)
{
    this->imagePyramid.clear();
    this->imagePyramid = pyramid.imagePyramid;
    this->scaleFactor = pyramid.scaleFactor;
    this->epsilon = pyramid.epsilon;
    this->scaleFactors = pyramid.scaleFactors;
    this->scaleFactorsInv = pyramid.scaleFactorsInv;
    this->xScaleFactors = pyramid.xScaleFactors;
    this->xScaleFactorsInv = pyramid.xScaleFactorsInv;
    this->yScaleFactors = pyramid.yScaleFactors;
    this->yScaleFactorsInv = pyramid.yScaleFactorsInv;

    return *this;
}

inline ImagePyramid::operator const bool() const
{
    return this->imagePyramid.size() > 0;
}

inline ImagePyramid::operator bool()
{
    return this->imagePyramid.size() > 0;
}

inline std::size_t ImagePyramid::numOfLevels() const
{
    return this->imagePyramid.size();
}

inline double ImagePyramid::getScaleFactor() const
{
    return this->scaleFactor;
}

inline double ImagePyramid::getScaleFactor(std::size_t index) const
{
    assert (index < this->scaleFactors.size());
    return this->scaleFactors[index];
}

inline double ImagePyramid::getScaleFactorInv(std::size_t index) const
{
    assert (index < this->scaleFactorsInv.size());
    return this->scaleFactorsInv[index];
}

inline double ImagePyramid::getXScaleFactor(std::size_t index) const
{
    assert(index < this->xScaleFactors.size());
    return this->xScaleFactors[index];
}

inline double ImagePyramid::getXScaleFactorInv(std::size_t index) const
{
    assert(index < this->xScaleFactorsInv.size());
    return this->xScaleFactorsInv[index];
}

inline double ImagePyramid::getYScaleFactor(std::size_t index) const
{
    assert(index < this->yScaleFactors.size());
    return this->yScaleFactors[index];
}

inline double ImagePyramid::getYScaleFactorInv(std::size_t index) const
{
    assert(index < this->yScaleFactorsInv.size());
    return this->yScaleFactorsInv[index];
}

inline cv::Mat& ImagePyramid::getImage(std::size_t index)
{
    assert(index < this->imagePyramid.size());
    return this->imagePyramid[index];
}

inline const cv::Mat& ImagePyramid::getImage(std::size_t index) const
{
    assert(index < this->imagePyramid.size());
    return this->imagePyramid[index];
}

inline cv::Mat& ImagePyramid::getImage(double scaleFactor, int round)
{
    return getImage(getIndex(scaleFactor, round));
}

inline const cv::Mat& ImagePyramid::getImage(double scaleFactor, int round) const
{
    return getImage(getIndex(scaleFactor, round));
}


#endif // IMAGEPYRAMID_HPP
