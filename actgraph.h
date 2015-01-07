#ifndef ACTGRAPH_H
#define ACTGRAPH_H

#include <string>
#include <vector>
#include <list>

#include "track.h"

class ActGraph
{
public:
    ActGraph();

public:
    void trainModel(const std::string featureFile);
    void trainModel(const cv::Mat &stat, const cv::Mat &shapeXY, const cv::Mat &mbhX, const cv::Mat &mbhY);
    void trainModel(const std::vector<std::list<Track> >& scaleTracks);
    void clusterFeatures(const std::string featureFile);

    void buildGraph();

private:
    void readFeatures(const std::string featureFile, cv::Mat &stat, cv::Mat &shapeXY, cv::Mat &mbhX, cv::Mat &mbhY);

protected:
    std::string fileModelParamShape;
    std::string fileModelParamMbhx;
    std::string fileModelParamMbhy;
};

#endif // ACTGRAPH_H
