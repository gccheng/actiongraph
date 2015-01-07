#include "actgraph.h"
#include "utility.h"


#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>


cv::Scalar colorTab[] =
{
    cv::Scalar(0, 0, 255),
    cv::Scalar(0,255,0),
    cv::Scalar(255,0,0),
    cv::Scalar(255,0,255),
    cv::Scalar(0,255,255),
    cv::Scalar(255,255,0),
    cv::Scalar(255,255,255),
    cv::Scalar(255, 100, 100),
    cv::Scalar(100, 100, 255),
    cv::Scalar(100, 255, 100)
};


ActGraph::ActGraph()
    :fileModelParamShape("em_shape.yaml"), fileModelParamMbhx("em_mbhx.yaml"), fileModelParamMbhy("em_mbhy.yaml")
{
}


void ActGraph::trainModel(const std::string featureFile)
{
    // features
    cv::Mat stat;        // mean_x, mean_y, var_x, var_y
    cv::Mat shapeXY;     // track shapes
    cv::Mat mbhX;        // MBHx
    cv::Mat mbhY;        // MBHy

    // import features from file
    readFeatures(featureFile, stat, shapeXY, mbhX, mbhY);

    // GMM-clustering
    trainModel(stat, shapeXY, mbhX, mbhY);
}

// EM algorithm for GMM clustering
void ActGraph::trainModel(const cv::Mat& stat,       // statistics info about each and every trajectory
                          const cv::Mat& shapeXY,    // trajectory shape
                          const cv::Mat& mbhX,       // MBHx descriptor
                          const cv::Mat& mbhY)       // MBHy descriptor
{
    std::cout << "Training models using EM..." << std::endl;
    // clear existing files
    std::vector<std::string> files;
    files.push_back("em_shape.yaml");
    files.push_back("em_mbhx.yaml");
    files.push_back("em_mbhy.yaml");
    Utility::delFiles(files);

    // shape
    int nclusters = 10;
    int covMatType = cv::ml::EM::COV_MAT_DIAGONAL;
    cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, cv::ml::EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    //cv::ml::EM em(nclusters, covMatType, tc); // Qt 4.x
    cv::Ptr<cv::ml::EM> em = cv::ml::EM::train(shapeXY, cv::noArray(), cv::noArray(), cv::noArray(),
                                               cv::ml::EM::Params(nclusters, covMatType, tc));
    if (!em.empty())
    {
        //const cv::Mat& weights = em.getMat("weights");
        //const cv::Mat& means = em.getMat("means");
        //const std::vector<cv::Mat>& covs = em.getMatVector("covs");

        cv::FileStorage ofs(fileModelParamShape, cv::FileStorage::WRITE);
        em->write(ofs);
        ofs.release();
        std::cout << "  >>EM model for shape is generated." << std::endl;
    }
    // MBHx
    nclusters = 10;
    covMatType = cv::ml::EM::COV_MAT_DIAGONAL;
    tc = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, cv::ml::EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    //em = cv::ml::EM(nclusters, covMatType, tc);
    em = cv::ml::EM::train(mbhX, cv::noArray(), cv::noArray(), cv::noArray(),
                                               cv::ml::EM::Params(nclusters, covMatType, tc));
    if (false && !em.empty())
    {
        //const cv::Mat& weights = em.getMat("weights");
        //const cv::Mat& means = em.getMat("means");
        //const std::vector<cv::Mat>& covs = em.getMatVector("covs");

        cv::FileStorage ofs(fileModelParamMbhx, cv::FileStorage::WRITE);
        em->write(ofs);
        ofs.release();
        std::cout << "  >>EM model for MBHx is generated." << std::endl;
    }
    // MBHy
    nclusters = 10;
    covMatType = cv::ml::EM::COV_MAT_DIAGONAL;
    tc = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, cv::ml::EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    //em = cv::ml::EM(nclusters, covMatType, tc);
    em = cv::ml::EM::train(mbhY, cv::noArray(), cv::noArray(), cv::noArray(),
                                               cv::ml::EM::Params(nclusters, covMatType, tc));
    if (false && !em.empty())
    {
        //const cv::Mat& weights = em.getMat("weights");
        //const cv::Mat& means = em.getMat("means");
        //const std::vector<cv::Mat>& covs = em.getMatVector("covs");

        cv::FileStorage ofs(fileModelParamMbhy, cv::FileStorage::WRITE);
        em->write(ofs);
        ofs.release();
        std::cout << "  >>EM model for MBHy is generated." << std::endl;
    }
}

void ActGraph::trainModel(const std::vector<std::list<Track> >& scaleTracks)
{

}

void ActGraph::buildGraph()
{

}

// for each descriptor (shapeXY, mbhX and mbhY), each row represents info for one trajectory
void ActGraph::readFeatures(const std::string featureFile, // input file storing features
                            cv::Mat& stat,       // statistics info about each and every trajectory
                            cv::Mat& shapeXY,    // trajectory shape
                            cv::Mat& mbhX,       // MBHx descriptor
                            cv::Mat& mbhY)       // MBHy descriptor
{
    std::cout << "Reading features from file..." << std::endl;

    cv::FileStorage ifs(featureFile, cv::FileStorage::READ);

    cv::FileNode fileNode = ifs.root();
    cv::FileNodeIterator iTrack = fileNode.begin();
    for (; iTrack!=fileNode.end(); iTrack++)   // for each and every track
    {
        // statistical infomation
        cv::FileNode statNode = (*iTrack)["statistics"];
        std::vector<float> trackStat(7);
        trackStat[0] = (float)statNode["mean_x"];
        trackStat[1] = (float)statNode["mean_y"];
        trackStat[2] = (float)statNode["var_x"];
        trackStat[3] = (float)statNode["var_y"];
        trackStat[4] = (float)statNode["frame"];
        trackStat[5] = (float)statNode["length"];
        trackStat[6] = (float)statNode["scale"];

        std::vector<float> trackShape;
        std::vector<float> trackMbhX;
        std::vector<float> trackMbhY;

        // shape descriptor
        cv::FileNode shapeNode = (*iTrack)["shape"];
        if (shapeNode.type() == cv::FileNode::SEQ)
        {
            shapeNode >> trackShape;
        }

        for (int i=1; i<=3; i++)
        {
            cv::FileNode mbhNode;
            std::vector<float> mbhxi;
            std::vector<float> mbhyi;

            // MBHx
            std::ostringstream ossx(std::ios_base::out | std::ios_base::ate);
            ossx << "mbhx_" << i;
            mbhNode = (*iTrack)[ossx.str()];
            if (mbhNode.type() == cv::FileNode::SEQ)
            {
                mbhNode >> mbhxi;
                trackMbhX.insert(trackMbhX.end(), mbhxi.begin(), mbhxi.end());
            }

            // MBHy
            std::ostringstream ossy(std::ios_base::out | std::ios_base::ate);
            ossy << "mbhy_" << i;
            mbhNode = (*iTrack)[ossy.str()];
            if (mbhNode.type() == cv::FileNode::SEQ)
            {
                mbhNode >> mbhyi;
                trackMbhY.insert(trackMbhY.end(), mbhyi.begin(), mbhyi.end());
            }
        }

        //trackMbhX.insert(trackMbhX.end(), trackStat[0]);
        //trackMbhX.insert(trackMbhX.end(), trackStat[1]);

        stat.push_back<float>(cv::Mat(trackStat).t());
        shapeXY.push_back<float>(cv::Mat(trackShape).t());
        mbhX.push_back<float>(cv::Mat(trackMbhX).t());
        mbhY.push_back<float>(cv::Mat(trackMbhY).t());
    }

//    cv::Mat meanXs = mbhX.col(96);
//    cv::Mat meanYs = mbhX.col(97);

//    float maxX = *std::max_element(meanXs.begin<float>(), meanXs.end<float>())/2;
//    float maxY = *std::max_element(meanYs.begin<float>(), meanYs.end<float>())/2;

//    mbhX.col(96) /= maxX;
//    mbhX.col(97) /= maxY;

    ifs.release();
}


void ActGraph::clusterFeatures(const std::string featureFile)
{
    // features
    cv::Mat stat;        // mean_x, mean_y, var_x, var_y
    cv::Mat shapeXY;     // track shapes
    cv::Mat mbhX;        // MBHx
    cv::Mat mbhY;        // MBHy

    // import features from file
    readFeatures(featureFile, stat, shapeXY, mbhX, mbhY);

    // K-means clustering
    int K = 10;
    cv::Mat bestLabels;
    cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, cv::ml::EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
    int attempts = 6;
    cv::Mat centers;
    std::cout << mbhX.rows << std::endl;
    cv::kmeans(mbhX, K, bestLabels, tc, attempts, cv::KMEANS_PP_CENTERS, centers);

    cv::Ptr<cv::ml::EM> emmodel;

    cv::Mat img = cv::Mat::zeros(420, 760, CV_8UC3);
    //cv::namedWindow("clusters");
    int rows = mbhX.rows;
    for (int i=1; i<rows; ++i)
    {
        int clusterIdx = bestLabels.at<int>(i);
        cv::Point ipt = cv::Point2f(stat.at<float>(i, 0), stat.at<float>(i, 1));
        cv::circle(img, ipt, 2, colorTab[clusterIdx%10]);
        if (i%300==0)
        {
            std::ostringstream oss(std::ios_base::out | std::ios_base::ate);
            oss << "cluster" << i/300;
            cv::imshow(oss.str(), img);
            char key = (char)cv::waitKey(5);
            if (key==27)
            {
                break;
            }
            img = cv::Mat::zeros(420, 760, CV_8UC3);
        }
        cv::waitKey(3);
    }
}


/*
        const FileStorage fs(filename, FileStorage::READ);
        EM model;
        if (fs.isOpened()) {
            const FileNode& fn = fs["StatModel.EM"];
            model.read(fn);
        }

  */
