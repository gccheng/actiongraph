//#include <QApplication>

#include <opencv2/opencv.hpp>

#include <vector>
#include <list>

#include "actanalysis.h"
#include "densetrack.h"
#include "actgraph.h"

int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    char* video = argv[1];
    char* dest_file = argv[2];

    // all of the tracks
    std::vector<std::list<Track> > xyScaleTracks;

    // extract features
    DenseTrack dt(1, dest_file);
    dt(video, xyScaleTracks);

    // cluster features
    //ActGraph ag;
    //ag.trainModel("./features.xml");  // train a GMM model using EM /*dt.getFeaturePath()*/
    //ag.clusterFeatures("./features.xml");
    
    //return a.exec();
    return 0;
}
