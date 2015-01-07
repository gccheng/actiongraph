#include "imagepyramid.h"

#include <cmath>
#include <cassert>
#include <stdexcept>

// construct the image pyramid using an image
void ImagePyramid::init(cv::Mat image, double scaleFactor)
{
    this->epsilon = scaleFactor * 0.05;

    // get the maximum number of levels given the scale factor
    std::size_t nLevels = 1;
    if (scaleFactor > 1.01)
    {
        nLevels = static_cast<std::size_t>(std::floor( log(std::min(image.cols, image.rows))/log(scaleFactor) ) );
    }
    assert( 1 <= nLevels);

    // build up all levels
    std::vector<cv::Mat> imgPyramid(nLevels);
    std::vector<double> correctScaleFactors(nLevels);
    std::vector<double> correctScaleFactorsInv(nLevels);
    std::vector<double> correctXScaleFactors(nLevels);
    std::vector<double> correctXScaleFactorsInv(nLevels);
    std::vector<double> correctYScaleFactors(nLevels);
    std::vector<double> correctYScaleFactorsInv(nLevels);
    imgPyramid[0] = image;
    correctScaleFactors[0] = 1;
    correctScaleFactorsInv[0] = 1;
    correctXScaleFactors[0] = 1;
    correctXScaleFactorsInv[0] = 1;
    correctYScaleFactors[0] = 1;
    correctYScaleFactorsInv[0] = 1;
    for (std::size_t i=1; i<nLevels; ++i)
    {
        // get the image from the last and the current scale level
        cv::Mat oldImag = image;
        if (i > 1)
        {
            oldImag = imgPyramid[i-2];
        }
        cv::Mat& newImg = imgPyramid[i];

        // scale the image from the last level to the wishes size
        double newScaleFactor = std::pow(scaleFactor, i);
        cv::Size newSize = cv::Size(static_cast<int>(image.cols/newScaleFactor+0.5),
                                    static_cast<int>(image.rows/newScaleFactor+0.5));
        cv::resize(image, newImg, newSize);

        // get the real scale factors
        double xScaleFactor = double(image.cols) / double(newImg.cols);
        double yScaleFactor = double(image.rows) / double(newImg.rows);
        correctXScaleFactors[i] = xScaleFactor;
        correctXScaleFactorsInv[i] = 1. / xScaleFactor;
        correctYScaleFactors[i] = yScaleFactor;
        correctYScaleFactorsInv[i] = 1. / yScaleFactor;
        correctScaleFactors[i] = 0.5 * (xScaleFactor + yScaleFactor);
        correctScaleFactorsInv[i] = 1. / correctScaleFactors[i];

        // scale the ROI mask as well, if it exists
    }

    this->imagePyramid = imgPyramid;
    this->scaleFactors = correctScaleFactors;
    this->scaleFactorsInv = correctScaleFactorsInv;
    this->xScaleFactors = correctXScaleFactors;
    this->xScaleFactorsInv = correctXScaleFactorsInv;
    this->yScaleFactors = correctYScaleFactors;
    this->yScaleFactorsInv = correctYScaleFactorsInv;
    if (1 == nLevels)
    {
        this->scaleFactor = 0;
    }
    else
    {
        this->scaleFactor = scaleFactor;
    }

}

// initialize empty image pyramid
void ImagePyramid::init(cv::Size initSize, int depth, int nChannels, double scaleFactor)
{
    cv::Mat img = cv::Mat::zeros(initSize, CV_MAKE_TYPE(depth, nChannels));
    init(img, scaleFactor);
}


std::size_t ImagePyramid::getIndex(double scaleFactor, int round) const
{
    // extreme cases
    if (scaleFactor >= this->scaleFactors[this->scaleFactors.size()-1])
    {
        return this->scaleFactors.size()-1;
    }
    if (scaleFactor <= 1)
    {
        return 0;
    }

    // find the correct answer based on the given rounding type
    std::size_t i = 0;
    if (round < 0)
    {
        // find the next level with a smaller factor
        scaleFactor += this->epsilon;
        for (i=0; scaleFactor>=this->scaleFactors[i] && i<this->scaleFactors.size()-1; ++i);
        i--;
    }
    else if (round > 0)
    {
        // find the nexe level with a bigger factor
        scaleFactor += this->epsilon;
        for (i=0; scaleFactor>=this->scaleFactors[i] && i<this->scaleFactors.size()-1; ++i);
    }
    else
    {
        // find the closest level with respect to the scale factor
        double bestDist = fabs(this->scaleFactors[0]-scaleFactor);
        std::size_t iBest = 0;
        for(i=1; i<this->scaleFactors.size(); ++i)
        {
            double dist = fabs(this->scaleFactors[i]-scaleFactor);
            if (dist<bestDist)
            {
                iBest = i;
                bestDist = dist;
            }
            else
            {
                break;
            }
        }
        i = iBest;
    }

    return i;
}

void ImagePyramid::rebuild(cv::Mat image)
{
    if (image.cols != this->imagePyramid[0].cols
            || image.rows != this->imagePyramid[0].rows
            || image.channels() != this->imagePyramid[0].channels())
    {
        throw std::runtime_error("ImagePyramid::rebuild(): given image dimensions and original image dimensions differ!");
    }
    this->imagePyramid[0] = image;
    for (std::size_t i=1; i<this->imagePyramid.size(); ++i)
    {
        cv::resize(image, this->imagePyramid[i], this->imagePyramid[i].size());
    }
}
