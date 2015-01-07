#include "densetrack.h"
#include "initialize.h"

#include <fstream>
#include <sstream>

#include <ctime>
#include <cstdio>

#include <opencv2/videoio.hpp>

DenseTrack::DenseTrack(int showtrack, std::string path)
    :show_track(showtrack),
     fscales(0),
     tracker(Config::track_length, Config::init_gap),
     hogInfo(8, 0, 1, Config::patch_size, Config::nxy_cell, Config::nt_cell),
     hofInfo(9, 1, 1, Config::patch_size, Config::nxy_cell, Config::nt_cell),
     mbhInfo(8, 0, 1, Config::patch_size, Config::nxy_cell, Config::nt_cell)
{
    if (path=="")
    {
        pathFeatures = "./features.xml";
    }
    else
    {
        pathFeatures = path;
    }

    std::ifstream file(pathFeatures.c_str());
    if (file.good())
    {
        if (remove(pathFeatures.c_str())!=0)
        {
            std::cerr << "Error deleting existing feature file." << std::endl;
        }
    }
    file.close();
}


// get the rectangle for computing the descriptor
cv::Rect DenseTrack::getRect(const cv::Point2f& point,  // the interest point position
                 const cv::Size& size,                  // the size of the image
                 const DescInfo& descInfo) // parameters about the descriptor
{
    int x_min = descInfo.blockWidth/2;
    int y_min = descInfo.blockHeight/2;
    int x_max = size.width - descInfo.blockWidth;
    int y_max = size.height - descInfo.blockHeight;

    // return the rectangle
    cv::Rect rect;

    float temp = point.x - x_min;
    rect.x = std::min<float>(std::max<float>(temp, 0.), x_max);

    temp = point.y - y_min;
    rect.y = std::min<float>(std::max<float>(temp, 0.), y_max);


    rect.height = descInfo.blockHeight;
    rect.width = descInfo.blockWidth;

    return rect;
}


// computer integral histograms for the whole image
void DenseTrack::buildDescIntegralHist(const cv::Mat& xComp, // x gradient component
                           const cv::Mat& yComp,     // y gradient component
                           const DescInfo& descInfo, // parameters about the descriptor
                           DescMat& descMat)         // output integral histogram
{
    // whether use full orientation or not
    float fullAngle = descInfo.fullOrientation ? 360 : 180;
    // one addintional bin for HoF
    int nBins = descInfo.flagThre ? descInfo.nBins-1: descInfo.nBins;
    // angle stride for quantization
    float angleBase = fullAngle/nBins;
    int width = descMat.width;
    int height = descMat.height;
    int histDim = descMat.nBins;
    int index = 0;

    for (int i=0; i<height; i++)
    {
        const float *xcompRow = xComp.ptr<float>(i);
        const float *ycompRow = yComp.ptr<float>(i);

        // the histogram accumulated in the current line
        std::vector<float> sum(histDim);
        for (int j=0; j<width; j++, index++)
        {
            float shiftX = xcompRow[j];
            float shiftY = ycompRow[j];
            float magnitude0 = sqrt(shiftX*shiftX + shiftY*shiftY);
            float magnitude1 = magnitude0;
            int bin0, bin1;

            // for the zero bin of HoF
            if (descInfo.flagThre==1 && magnitude0 <= descInfo.threshold)
            {
                bin0 = nBins;  // the zero bin is the last one
                magnitude0 = 1.0;
                bin1 = 0;
                magnitude1 = 0;
            }
            else
            {
                float orientation = cv::fastAtan2(shiftY, shiftX);
                if (orientation > fullAngle)
                {
                    orientation = fullAngle;
                }

                // split the magnitude to two adjacent bins
                float fbin = orientation/angleBase;
                bin0 = cvFloor(fbin);
                float weight0 = 1 - (fbin-bin0);
                float weight1 = 1 - weight0;
                bin0 %= nBins; //? nBins != fullAngle/angleBase???
                bin1 = (bin0+1)%nBins;

                magnitude0 *= weight0;
                magnitude1 *= weight1;
            }

            sum[bin0] += magnitude0;
            sum[bin1] += magnitude1;

            if (i == 0)
            {
                for (int m=0; m<descMat.nBins; m++)
                {
                    descMat.desc->ptr<float>(i,j)[m] = sum[m];
                }
            }
            else
            {
                for (int m=0; m<descMat.nBins; m++)
                {
                    descMat.desc->ptr<float>(i,j)[m] = descMat.desc->ptr<float>(i-1,j)[m] + sum[m];
                }
            }
        }
    }
}


// get a descriptor from the integral histogram
std::vector<float> DenseTrack::getDesc(const DescMat& descMat, // input integral histogram
                                       const cv::Rect& rect,   // rectangle area for the descriptor
                                       const DescInfo& descInfo) // parameters about the descriptor
{
    int descDim = descInfo.dim;
    int height = descMat.height;
    int width = descMat.width;

    std::vector<float> vec(descDim);
    int xOffset = rect.x;
    int yOffset = rect.y;
    int xStride = rect.width/descInfo.nxCells;
    int yStride = rect.height/descInfo.nyCells;

    // iterate over different cells
    int iDesc = 0;
    for (int iX=0; iX<descInfo.nxCells; iX++)
    {
        for (int iY=0; iY<descInfo.nyCells; iY++)
        {
            // get the positions of the rectangle
            int left = xOffset + iX*xStride;
            int right = std::min<int>(left+xStride, width-1);
            int top = yOffset + iY*yStride;
            int bottom = std::min<int>(top+yStride, height-1);

            // get the integral histograms at four corners
            const float* sumTopLeft = descMat.desc->ptr<float>(top, left);
            const float* sumTopRight = descMat.desc->ptr<float>(top, right);
            const float* sumBottomLeft = descMat.desc->ptr<float>(bottom, left);
            const float* sumBottomRight = descMat.desc->ptr<float>(bottom, right);

            // calculate each channel within the rectangle
            for (int i=0; i<descInfo.nBins; i++, iDesc++)
            {
                float temp = sumBottomRight[i] + sumTopLeft[i]
                        - sumBottomLeft[i] - sumTopRight[i];
                vec[iDesc] = std::max<float>(temp, 0.) + Config::epsilon;
            }
        }
    }

    // normalization
    float normFactor = 0.0f;
    if(descInfo.norm == 1)
    {
        normFactor = cv::norm(vec, cv::NORM_L1);
    }
    else
    {
        normFactor = cv::norm(vec, cv::NORM_L2);
    }
    std::transform(vec.begin(), vec.end(), vec.begin(), std::bind2nd(std::divides<float>(), normFactor));

    return vec;
}

// build integral histogram of HoG
void DenseTrack::HogComp(const cv::Mat& img,      // input image
                         DescMat& descMat) // output integral histogram
{
    int width = descMat.width;
    int height = descMat.height;
    cv::Mat imgX(cv::Size(width, height), CV_32FC1);
    cv::Mat imgY(cv::Size(width, height), CV_32FC1);

    cv::Sobel(img, imgX, CV_32F, 1, 0);
    cv::Sobel(img, imgY, CV_32F, 0, 1);

    buildDescIntegralHist(imgX, imgY, hogInfo, descMat);
}

// build integral histogram of HoF
void DenseTrack::HofComp(const cv::Mat& flow,     // input flow image (2 channels)
                         DescMat& descMat) // output integral histogram
{
    int width = descMat.width;
    int height = descMat.height;
    cv::Mat xComp(cv::Size(width, height), CV_32FC1);
    cv::Mat yComp(cv::Size(width, height), CV_32FC1);

    if (flow.channels()==2)
    {
        std::vector<cv::Mat> xyChannel;
        cv::split(flow, xyChannel);
        xComp = xyChannel[0];
        yComp = xyChannel[1];

        buildDescIntegralHist(xComp, yComp, hofInfo, descMat);
    }
}

void DenseTrack::MbhComp (const cv::Mat& flow,  // optical flow
                          DescMat& descMatX,    // output MBHx
                          DescMat& descMatY)    // output MBHy
{
    int width = descMatX.width;
    int height = descMatX.height;

    cv::Mat flowX(flow.size(), CV_32FC1);     // flow in X direction
    cv::Mat flowY(flow.size(), CV_32FC1);     // flow in Y direction
    cv::Mat flowXdX(flow.size(), CV_32FC1);   // gradients over X in X flow
    cv::Mat flowXdY(flow.size(), CV_32FC1);   // gradients over Y in X flow
    cv::Mat flowYdX(flow.size(), CV_32FC1);   // gradients over X in Y flow
    cv::Mat flowYdY(flow.size(), CV_32FC1);   // gradients over Y in Y flow

    // extract the X and Y components of the flow
    for (int y=0; y<height; y++)
    {
        for (int x=0; x<width; x++)
        {
            flowX.at<float>(y,x) = flow.ptr<float>(y,x)[0];
            flowY.at<float>(y,x) = flow.ptr<float>(y,x)[1];
        }
    }

    cv::Sobel(flowX, flowXdX, -1, 1, 0);
    cv::Sobel(flowX, flowXdY, -1, 0, 1);
    cv::Sobel(flowY, flowYdX, -1, 1, 0);
    cv::Sobel(flowY, flowYdY, -1, 0, 1);

    buildDescIntegralHist(flowXdX, flowXdY, mbhInfo, descMatX);
    buildDescIntegralHist(flowYdX, flowYdY, mbhInfo, descMatY);
}

// detect new feature points in the whole image
void DenseTrack::sampleDense(const cv::Mat& grey,     // grey-scale image
                             const double quality,    // quanlity level
                             const double min_dist,   // minimal distance between points
                             std::vector<cv::Point2f>& points_out) // output points
{
    cv::Mat eig;    // image of eigen-values
    int width = cvFloor(grey.cols/min_dist);
    int height = cvFloor(grey.rows/min_dist);
    double maxVal = 0.0;
    cv::cornerMinEigenVal(grey, eig, 3);
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, cv::noArray());
    const double threshold = maxVal*quality;

    int offset = cvFloor(min_dist/2);
    for (int i=0; i<height; ++i)
    {
        for (int j=0; j<width; ++j)
        {
            int x = cvFloor(j*min_dist+offset);
            int y = cvFloor(i*min_dist+offset);
            if(eig.at<float>(y,x) > threshold)
            {
                points_out.push_back(cv::Point2f(x, y));
            }
        }
    }
}


// detect new feature points in an image without overlapping to previous points
void DenseTrack::sampleDense(const cv::Mat &grey,         // grey-scale image
                             const std::vector<cv::Point2f> &points_in,// image of eigen-values
                             const double quality,
                             const double min_dist,       // quanlity level
                             std::vector<cv::Point2f> &points_out) // output points
{
    cv::Mat eig;    // image of eigen-values
    int width = cvFloor(grey.cols/min_dist);
    int height = cvFloor(grey.rows/min_dist);
    double maxVal = 0.0;
    cv::cornerMinEigenVal(grey, eig, 3);
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, cv::noArray());
    const double threshold = maxVal*quality;

    std::vector<int> counters(width*height);
    for (int i=0; i<points_in.size(); ++i)
    {
        cv::Point2f point = points_in[i];
        if (point.x >= min_dist*width || point.y >= min_dist*height)
        {
            continue;
        }
        int x = cvFloor(point.x/min_dist);
        int y = cvFloor(point.y/min_dist);
        counters[y*width+x]++;
    }

    int index = 0;
    int offset = cvFloor(min_dist/2);
    for (int i=0; i<height; ++i)
    {
        for (int j=0; j<width; ++j, index++)
        {
            if (counters[index]==0)
            {
                int x = cvFloor(j*min_dist+offset);
                int y = cvFloor(i*min_dist+offset);
                if(eig.at<float>(y, x) > threshold)
                {
                    points_out.push_back(cv::Point2f(x, y));
                }
            }
        }
    }
}

// tracking interest points by median filtering in the optical field
void DenseTrack::trackOpticalFlow(const cv::Mat& flow,  // optical flow
                                  const std::vector<cv::Point2f>& points_in, // points at frame i
                                  std::vector<cv::Point2f>& points_out,   // out points at frame i+1
                                  std::vector<int>& status)  // tracking status
{
    if (points_in.size() != points_out.size())
    {
        std::cerr << "the numbers of the points don't match!" << std::endl;
    }
    if (points_in.size() != status.size())
    {
        std::cerr << "the number os status doesn't match!" << std::endl;
    }

    int width = flow.cols;
    int height = flow.rows;

    for (int i=0; i<points_in.size(); ++i)
    {
        cv::Point2f point_in = points_in[i];
        std::list<float> xs, ys;
        int x = cvFloor(point_in.x);
        int y = cvFloor(point_in.y);
        for (int m=x-1; m<=x+1; ++m)
        {
            for (int n=y-1; n<=y+1; ++n)
            {
                int p = std::min<int>(std::max<int>(m,0), width-1);  // current col
                int q = std::min<int>(std::max<int>(n,0), height-1); // current row
                const float* f = flow.ptr<float>(q, p);
                xs.push_back(*f);
                ys.push_back(*(f+1));
            }
        }

        // find the median
        xs.sort();
        ys.sort();
        for (int m=0; m<xs.size()/2; ++m)
        {
            xs.pop_back();
            ys.pop_back();
        }

        cv::Point2f offset;
        offset.x = xs.back();
        offset.y = ys.back();
        cv::Point2f point_out;
        point_out.x = point_in.x + offset.x;
        point_out.y = point_in.y + offset.y;
        points_out[i] = point_out;
        if (point_out.x >0 && point_out.x<width && point_out.y>0 && point_out.y<height)
        {
            status[i] = 1;
        } else
        {
            status[i] = -1;
        }

    }
}


// extract track features from a video file
void DenseTrack::operator()(std::string strVideo, std::vector<std::list<Track> >& xyScaleTracks)
{
    int frameNum = 0;     // current frame NO.
    int trackNum = 0;     // total number of active trajectories
    int id = 1;           // track id saved in file
    int start_scale = Config::start_scale;  // starting level, must be less than Config:scale_num

    cv::Mat image, prev_image, grey, prev_grey;
    ImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;

    cv::VideoCapture capture(strVideo);

    if (show_track == 1)
    {
        cv::namedWindow("DenseTrack", 0);
    }

    int init_counter = 0; // indicate when to detect new feature points
    while (true)
    {
        clock_t loop_start = clock();
        cv::Mat frame;
        frameNum = capture.get(cv::CAP_PROP_POS_FRAMES);
        capture >> frame;
        if (frame.empty())
        {
            std::cout << "break";
            break;
        }
        if (frameNum>=Config::start_frame && frameNum<Config::end_frame)
        {
            // initialize the structure of pyramids, getting feature points
            if (image.empty())
            {
                frame.copyTo(image);
                frame.copyTo(prev_image);
                grey.create(frame.size(), CV_MAKE_TYPE(frame.depth(), 1));
                grey_pyramid = ImagePyramid(frame.size(), 8, 1, Config::scale_stride);
                prev_grey.create(frame.size(), CV_MAKE_TYPE(frame.depth(), 1));
                prev_grey_pyramid = ImagePyramid(frame.size(), 8, 1, Config::scale_stride);
                //eig_pyramid = ImagePyramid(frame.size(), 8, 1, Config::scale_stride);

                cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
                grey_pyramid.rebuild(grey);

                // how many scale we can have
                Config::scale_num = std::min<std::size_t>(Config::scale_num, grey_pyramid.numOfLevels());
                fscales = (float*)malloc(Config::scale_num * sizeof(float));
                xyScaleTracks.resize(Config::scale_num);

                // find good features at each scale separately
                for (int ixyScale=start_scale; ixyScale<Config::scale_num; ++ixyScale)
                {
                    std::list<Track> &tracks = xyScaleTracks[ixyScale];
                    fscales[ixyScale] = pow(Config::scale_stride, ixyScale);
                    std::size_t temp_level = (std::size_t)ixyScale;
                    cv::Mat grey_temp(grey_pyramid.getImage(temp_level).clone());
                    detectFeaturePoints(grey_temp, std::vector<cv::Point2f>(), ixyScale, tracks); // detect feature points from grey_temp, and add them to tracks
                    trackNum += tracks.size();
                    grey_temp.release();
                }
            }

            // build the image pyramid for the current frame
            frame.copyTo(image);
            cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
            grey_pyramid.rebuild(grey);

            // tracking and generating trajectories
            if (frameNum > 0)
            {
                init_counter++;
                std::vector<float> orienthist((int)(360/Config::bgorient_stride)+1, 0.0f);
                std::vector<float> maghist((int)(360/Config::bgorient_stride)+1, 0.0f);
                for (int ixyScale=start_scale; ixyScale<Config::scale_num; ++ixyScale)
                {
                    std::vector<cv::Point2f> points_in(0);
                    std::list<Track> &tracks = xyScaleTracks[ixyScale];
                    for (std::list<Track>::iterator iTrack=tracks.begin(); iTrack!=tracks.end(); ++iTrack)
                    {
                        cv::Point2f point = iTrack->pointDescs.back().point;
                        points_in.push_back(point);
                    }
                    int count = points_in.size();
                    cv::Mat prev_grey_temp, grey_temp;
                    std::size_t temp_level = ixyScale;
                    prev_grey_temp = prev_grey_pyramid.getImage(temp_level);
                    grey_temp = grey_pyramid.getImage(temp_level);

                    std::vector<int> status(count);
                    std::vector<cv::Point2f> points_out(count);

                    // compute the optical flow
                    cv::Mat flow(grey_temp.size(), CV_32FC2);
                    cv::calcOpticalFlowFarneback(prev_grey_temp, grey_temp, flow,
                                                 sqrt(2.0)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
                    // track feature points by median filtering
                    trackOpticalFlow(flow, points_in, points_out, status);

                    // obtain the velocities (orientation+magnitude)
                    calcVelocityHist(points_in, points_out, ixyScale, orienthist, maghist);

                    int width = grey_temp.cols;
                    int height = grey_temp.rows;
                    // compute the integral histograms
                    DescMat hogMat(height, width, hogInfo.nBins);
                    HogComp(prev_grey_temp, hogMat);

                    DescMat hofMat(height, width, hofInfo.nBins);
                    HofComp(flow, hofMat);

                    DescMat mbhMatX(height, width, mbhInfo.nBins);
                    DescMat mbhMatY(height, width, mbhInfo.nBins);
                    MbhComp(flow, mbhMatX, mbhMatY);

                    int i = 0;
                    for (std::list<Track>::iterator iTrack=tracks.begin(); iTrack!=tracks.end(); ++i)
                    {
                        if (status[i] == 1) // if the feature point is successfully tracked
                        {
                            PointDesc& pointDesc = iTrack->pointDescs.back();
                            cv::Point2f prev_point = points_in[i];
                            cv::Rect rect = getRect(prev_point, cv::Size(width, height), hogInfo);
                            pointDesc.hog = getDesc(hogMat, rect, hogInfo);   // get the descrioptors for the feature point
                            pointDesc.hof = getDesc(hofMat, rect, hofInfo);
                            pointDesc.mbhX = getDesc(mbhMatX, rect, mbhInfo);
                            pointDesc.mbhY = getDesc(mbhMatY, rect, mbhInfo);

                            PointDesc point(hogInfo, hofInfo, mbhInfo, points_out[i]);
                            iTrack->addPointDesc(point);
                            ++iTrack;
                        }
                        else
                        {
                            iTrack = tracks.erase(iTrack);
                            trackNum--;
                        }
                    }
                } // for (int ixyScale=start_scale; ixyScale<Config::scale_num; ++ixyScale)

                // filter the tracks at each scale using median filters
                for (int i=0; i<maghist.size(); i++)
                {
                    maghist[i] /= (orienthist[i]+Config::epsilon);
                }
                int maxBin = std::distance(orienthist.begin(), std::max_element(orienthist.begin(), orienthist.end()));
                float freqOrient = Config::bgorient_stride/2 + Config::bgorient_stride * maxBin;
                float freqMag = maghist[maxBin];
                for (int ixyScale=start_scale; ixyScale<Config::scale_num; ++ixyScale)
                {
                    std::list<Track>& tracks = xyScaleTracks[ixyScale];
                    for (std::list<Track>::iterator iTrack=tracks.begin(); iTrack!=tracks.end(); )
                    {
                        std::list<PointDesc> descs = iTrack->pointDescs;  // hard-copy
                        cv::Point2f curr_point = descs.back().point;
                        descs.pop_back();
                        cv::Point2f prev_point = descs.back().point;
                        float orient = orientPoint(prev_point, curr_point, ixyScale);
                        float mag = distPoint(prev_point, curr_point, ixyScale);
                        // check if it's background motion
                        if (true && (fabs(orient-freqOrient)<Config::bgorient_stride || fabs(orient-(freqOrient+360))<Config::bgorient_stride)
                                && (fabs(mag-freqMag)<Config::bgmag_stride_ratio*freqMag))
                        {
                            iTrack = tracks.erase(iTrack);
                            trackNum--;
                            continue;
                        }
                        // achieve specified length
                        if (iTrack->pointDescs.size() >= tracker.trackLength+1)
                        {
                            //saveTrackFeatures(*iTrack, frameNum, id++);
                            saveTrackFeaturesCSV(*iTrack, frameNum, id++);
                            if (show_track == 1) // draw this track
                            {
                                drawtrajectory(iTrack->pointDescs, ixyScale, image);
                            }
                            iTrack = tracks.erase(iTrack);
                            trackNum--;
                        }
                        else
                        {
                            iTrack++;
                        }
                    }
                }

                // detect new feature points every initGap frames
                if (init_counter == tracker.initGap)
                {
                    init_counter = 0;
                    for (int ixyScale=start_scale; ixyScale<Config::scale_num; ++ixyScale)
                    {
                        std::list<Track>& tracks = xyScaleTracks[ixyScale];
                        std::vector<cv::Point2f> points_in(0);
                        for (std::list<Track>::iterator iTrack=tracks.begin(); iTrack!=tracks.end(); iTrack++)
                        {
                            std::list<PointDesc>& descs = iTrack->pointDescs;
                            cv::Point2f point = descs.back().point; // the last point in the track
                            points_in.push_back(point);
                        }

                        std::size_t temp_level = (std::size_t)ixyScale;
                        cv::Mat grey_temp(grey_pyramid.getImage(temp_level).clone());
                        detectFeaturePoints(grey_temp, points_in, ixyScale, tracks); // detect feature points from grey_temp, and add them to tracks
                        trackNum += tracks.size()-points_in.size();
                        grey_temp.release();
                    }
                }
            } // if (frameNum > 0)

            // making current frame as previous frame
            frame.copyTo(prev_image);
            cv::cvtColor(prev_image, prev_grey, cv::COLOR_BGR2GRAY);
            prev_grey_pyramid.rebuild(prev_grey);
        } // if (frameNum>=Config::start_frame && frameNum<Config::end_frame)

        clock_t loop_end = clock();
        std::cout << "Loop running time: " << (double)(loop_end-loop_start)/CLOCKS_PER_SEC << std::endl;

        if (show_track == 1)
        {
            cv::imshow("DenseTrack", image);
            if ((char)(cv::waitKey(3))==27) break;
        }
        frameNum++;
    } // while (true)

    if (show_track == 1)
    {
        cv::destroyWindow("DenseTrack");
    }
}

// save one track features to file
// -level: zero-based level in the pyramid
void DenseTrack::saveTrackFeatures(Track& track, int lastFrame, int id)
{
    std::vector<cv::Point2f> trajectory(tracker.trackLength+1);
    std::list<PointDesc>& descs = track.pointDescs;
    std::list<PointDesc>::iterator iDesc = descs.begin();
    int level = track.scale;

    for (int i=0; i<=tracker.trackLength; ++iDesc, ++i)
    {
        trajectory[i].x = iDesc->point.x*fscales[level];
        trajectory[i].y = iDesc->point.y*fscales[level];
    }
    float basex = trajectory[0].x;
    float basey = trajectory[0].y;

    float mean_x(0.0), mean_y(0.0), var_x(0.0), var_y(0.0), length(0.0);
    if (isValidTrack(trajectory, mean_x, mean_y, var_x, var_y, length) == 1)
    {
        cv::FileStorage ofs(pathFeatures, cv::FileStorage::APPEND);
        std::ostringstream oss(std::ios_base::out | std::ios_base::ate);
        oss << "Track_" << id;
        ofs << oss.str() << "{";
        // statistical infomation
        ofs << "statistics" << "{";
        ofs << "frame" << lastFrame;
        ofs << "mean_x" << mean_x;
        ofs << "mean_y" << mean_y;
        ofs << "var_x" << var_x;
        ofs << "var_y" << var_y;
        ofs << "length" << length;
        ofs << "scale" << fscales[level];
        ofs << "base" << "[" << basex << basey << "]";
        ofs << "}";

        // track shape
        ofs << "shape" << trajectory;

        // HoG features
        // HoF features

        // MBH features
        iDesc = descs.begin();
        int t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
        for (int n=0; n<mbhInfo.ntCells; n++)
        {
            std::vector<float> vec(mbhInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<mbhInfo.dim; m++)
                {
                    vec[m] += iDesc->mbhX[m];
                }
            }
            for (int m=0; m<mbhInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            std::ostringstream oss(std::ios_base::out | std::ios_base::ate);
            oss << "mbhx_" << n+1;
            ofs << oss.str() << vec;
        }

        iDesc = descs.begin();
        t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
        for (int n=0; n<mbhInfo.ntCells; n++)
        {
            std::vector<float> vec(mbhInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<mbhInfo.dim; m++)
                {
                    vec[m] += iDesc->mbhY[m];
                }
            }
            for (int m=0; m<mbhInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            std::ostringstream oss(std::ios_base::out | std::ios_base::ate);
            oss << "mbhy_" << n+1;
            ofs << oss.str() << vec;
        }

        ofs << "}";

        ofs.release();
    }
}

// save one track features to file
// -level: zero-based level in the pyramid
void DenseTrack::saveTrackFeaturesCSV(Track& track, int lastFrame, int id)
{
    std::vector<cv::Point2f> trajectory(tracker.trackLength+1);
    std::list<PointDesc>& descs = track.pointDescs;
    std::list<PointDesc>::iterator iDesc = descs.begin();
    int level = track.scale;

    for (int i=0; i<=tracker.trackLength; ++iDesc, ++i)
    {
        trajectory[i].x = iDesc->point.x*fscales[level];
        trajectory[i].y = iDesc->point.y*fscales[level];
    }

    float mean_x(0.0), mean_y(0.0), var_x(0.0), var_y(0.0), length(0.0);
    if (isValidTrack(trajectory, mean_x, mean_y, var_x, var_y, length) == 1)
    {
        //cv::FileStorage ofs(pathFeatures, cv::FileStorage::APPEND);
        std::ofstream ofs(pathFeatures.c_str(), std::ios_base::out | std::ios_base::app);

        ofs << lastFrame << " ";       // frame #
        ofs << mean_x << " ";          // mean_x
        ofs << mean_y << " ";          // mean_y
        ofs << var_x << " ";           // var_x
        ofs << var_y << " ";           // var_y
        ofs << length << " ";          // length
        ofs << fscales[level] << " ";  // scale

        int cnt = 7;

        // track shape
        for (int i=0; i<tracker.trackLength; ++iDesc, ++i)
        {
            ofs << trajectory[i].x << " ";
            ofs << trajectory[i].y << " ";
            cnt += 2;
        }

        // HoG features
        iDesc = descs.begin();
        int t_stride = cvFloor(tracker.trackLength/hogInfo.ntCells);
        for (int n=0; n<hogInfo.ntCells; n++)
        {
            std::vector<float> vec(hogInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<hogInfo.dim; m++)
                {
                    vec[m] += iDesc->hog[m];
                }
            }
            for (int m=0; m<hogInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            for (int v=0; v<vec.size(); v++)
            {
                ofs << vec[v] << " ";
            }
            cnt += vec.size();
        }

        // HoF features
        iDesc = descs.begin();
        t_stride = cvFloor(tracker.trackLength/hofInfo.ntCells);
        for (int n=0; n<hofInfo.ntCells; n++)
        {
            std::vector<float> vec(hofInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<hofInfo.dim; m++)
                {
                    vec[m] += iDesc->hof[m];
                }
            }
            for (int m=0; m<hofInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            for (int v=0; v<vec.size(); v++)
            {
                ofs << vec[v] << " ";
            }
            cnt += vec.size();
        }

        // MBH features
        iDesc = descs.begin();
        t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
        for (int n=0; n<mbhInfo.ntCells; n++)
        {
            std::vector<float> vec(mbhInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<mbhInfo.dim; m++)
                {
                    vec[m] += iDesc->mbhX[m];
                }
            }
            for (int m=0; m<mbhInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            for (int v=0; v<vec.size(); v++)
            {
                ofs << vec[v] << " ";
            }
            cnt += vec.size();
        }

        iDesc = descs.begin();
        t_stride = cvFloor(tracker.trackLength/mbhInfo.ntCells);
        for (int n=0; n<mbhInfo.ntCells; n++)
        {
            std::vector<float> vec(mbhInfo.dim);
            for (int t=0; t<t_stride; t++, iDesc++) // summarize all the points in one cell
            {
                for ( int m=0; m<mbhInfo.dim; m++)
                {
                    vec[m] += iDesc->mbhY[m];
                }
            }
            for (int m=0; m<mbhInfo.dim; m++)
            {
                vec[m] /= t_stride;
            }
            for (int v=0; v<vec.size(); v++)
            {
                ofs << vec[v] << " ";
            }
            cnt += vec.size();
        }

        ofs << std::endl;

        ofs.close();
    }
}

// calculate the 2-norm of vetor {vec}
float DenseTrack::normVec(const std::vector<float> &vec)
{
    float norm = 0.0f;
    for (std::vector<float>::const_iterator it = vec.begin(); it != vec.end(); it++)
    {
        norm += (*it)*(*it);
    }
    return std::sqrt(norm);
}

void DenseTrack::calcVelocityHist(const std::vector<cv::Point2f>& points_in,
                                const std::vector<cv::Point2f>& points_out,
                                int level,
                                std::vector<float>& orienthist,
                                std::vector<float>& maghist)
{
    assert(points_in.size() == points_out.size());

    int nBins = 360.0f/Config::bgorient_stride;

    for (int i=0; i<points_in.size(); ++i)
    {    
        float xdisp = (points_out[i].x-points_in[i].x)*fscales[level];
        float ydisp = (points_out[i].y-points_in[i].y)*fscales[level];
        float mag = std::sqrt(xdisp*xdisp+ydisp*ydisp);

        if (mag < Config::min_flow*2)
        {
            orienthist[nBins] += 1.0f;
            maghist[nBins] += mag;
            continue;
        }
        float degrees = cv::fastAtan2(ydisp, xdisp);
        float fbin = degrees/Config::bgorient_stride;
        int bin0 = cvFloor(fbin);
        float weight0 = 1 - (fbin-bin0);
        float weight1 = 1 - weight0;
        bin0 %= nBins; //? nBins != fullAngle/angleBase??? ==> in case vector (1, 0) gives 360
        int bin1 = (bin0+1)%nBins;

        orienthist[bin0] += weight0;
        orienthist[bin1] += weight1;
        maghist[bin0] += weight0*mag;
        maghist[bin1] += weight1*mag;
    }
}

// check if a track is valid
bool DenseTrack::isValidTrack(std::vector<cv::Point2f>& track,
                              float& mean_x, float& mean_y,
                              float& var_x, float& var_y,
                              float& length)
{
    int size = track.size();
    for(int i = 0; i < size; i++) {
        mean_x += track[i].x;
        mean_y += track[i].y;
    }
    mean_x /= size;
    mean_y /= size;

    for(int i = 0; i < size; i++) {
        track[i].x -= mean_x;
        var_x += track[i].x*track[i].x;
        track[i].y -= mean_y;
        var_y += track[i].y*track[i].y;
    }
    var_x /= size;
    var_y /= size;
    var_x = sqrt(var_x);
    var_y = sqrt(var_y);
    // remove static trajectory
    if(var_x < Config::min_var && var_y < Config::min_var)
        return false;
    // remove random trajectory
    if( var_x > Config::max_var || var_y > Config::max_var )
        return false;

    for(int i = 1; i < size; i++) {
        float temp_x = track[i].x - track[i-1].x;
        float temp_y = track[i].y - track[i-1].y;
        length += sqrt(temp_x*temp_x+temp_y*temp_y);
        track[i-1].x = temp_x;
        track[i-1].y = temp_y;
    }

    float len_thre = length*0.7;
    for( int i = 0; i < size-1; i++ ) {
        float temp_x = track[i].x;
        float temp_y = track[i].y;
        float temp_dis = sqrt(temp_x*temp_x + temp_y*temp_y);
        if( temp_dis > Config::max_dis && temp_dis > len_thre )
            return false;
    }

    track.pop_back();
    // normalize the trajectory
    for(int i = 0; i < size-1; i++) {
        track[i].x /= length;
        track[i].y /= length;
    }
    return true;
}

void DenseTrack::normalizeTrack(const std::vector<cv::Point2f>& track, // input point-based track
                                bool normShape,     // if normalize to be position invariant (only shape)
                                bool normLength,    // if normalize to be length invariant
                                std::vector<cv::Point2f>& trackUptd)  // updated track
{
    float length = 0.0f;
    int size = track.size();
    trackUptd.resize(size-1);

    for(int i = 1; i < size; i++)
    {
        float temp_x = track[i].x - track[i-1].x;
        float temp_y = track[i].y - track[i-1].y;
        length += sqrt(temp_x*temp_x+temp_y*temp_y);
    }

    for(int i = 0; i < size-1; i++)
    {
        if (normShape)
        {
            trackUptd[i].x = track[i+1].x - track[i].x;
            trackUptd[i].y = track[i+1].y - track[i].y;
            if (normLength)  // normailize the length
            {
                trackUptd[i].x /= length;
                trackUptd[i].y /= length;
            }
        }
        else // not well implemented. normShape should be set to true
        {
            trackUptd[i].x = track[i].x;
            trackUptd[i].y = track[i].y;
        }
    }
}

float DenseTrack::distPoint(const cv::Point2f& p1, const cv::Point2f& p2, int level)
{
    float xdisp = (p2.x-p1.x)*fscales[level];
    float ydisp = (p2.y-p1.y)*fscales[level];
    return std::sqrt(xdisp*xdisp+ydisp*ydisp);
}

float DenseTrack::orientPoint(const cv::Point2f& p1, const cv::Point2f& p2, int level)
{
    float xdisp = (p2.x-p1.x)*fscales[level];
    float ydisp = (p2.y-p1.y)*fscales[level];
    if (std::sqrt(xdisp*xdisp+ydisp*ydisp) < Config::min_flow*2)
    {
        return 360.0f+Config::bgorient_stride/2;
    }
    else
    {
        return cv::fastAtan2(ydisp, xdisp);
    }
}

// draw trajectory {desc}, which is in {level}th pyramid, in {image}
void DenseTrack::drawtrajectory(std::list<PointDesc>& descs, int level, cv::Mat& image)
{
    std::list<PointDesc>::iterator iDesc = descs.begin();
    float length = descs.size();
    cv::Point2f point0 = iDesc->point;
    point0.x *= fscales[level]; // map the point to first scale
    point0.y *= fscales[level];

    float j = 0;
    for(iDesc++; iDesc!=descs.end() ; ++iDesc, ++j)
    {
        cv::Point2f point1 = iDesc->point;
        point1.x *= fscales[level];
        point1.y *= fscales[level];

        cv::line(image, point0, point1, cv::Scalar(0, cv::saturate_cast<uchar>(255.0*(j+1.0)/length), 0), 2, 8, 0);
        point0 = point1;
    }
    cv::circle(image, point0, 2, cv::Scalar(255, 0, 0), -1, 8, 0);
}

// find "new" feature points different from {points_in} in image {grey}
void DenseTrack::detectFeaturePoints(const cv::Mat& grey,      // input pyramid of current image
                                     const std::vector<cv::Point2f>& points_in, // existing points
                                     int level,                // pyramid level the track is with
                                     std::list<Track>& tracks) // output updated tracks
{
    std::vector<cv::Point2f> points(0);
    if (points_in.empty())
    {
        sampleDense(grey,Config::quality, Config::min_distance, points);
    }
    else
    {
        sampleDense(grey, points_in, Config::quality, Config::min_distance, points);
    }

    // save the feature points
    for (int i=0; i<points.size(); ++i)
    {
        Track track(tracker.trackLength);
        PointDesc point(hogInfo, hofInfo, mbhInfo, points[i]);
        track.addPointDesc(point);
        track.scale = level;
        tracks.push_back(track);
    }
}
