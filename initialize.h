#ifndef INITIALIZE_H
#define INITIALIZE_H

#include "densetrack.h"
#include "track.h"


// parameters for descriptors
int Config::patch_size = 32;
int Config::nxy_cell = 2;
int Config::nt_cell = 3;
bool Config::fullOrientation = true;
float Config::epsilon = 0.05;
float Config::min_flow = 0.4*0.4;
//const float PI = 3.14159;

// parameters for tracking
int Config::start_frame = 0;
int Config::end_frame = 1000000;
double Config::quality = 0.001;
double Config::min_distance = 5;
int Config::init_gap = 1;
int Config::track_length = 15;

// parameters for the trajectory descriptor
float Config::min_var = sqrt(3);
float Config::max_var = 50;
float Config::max_dis = 20;

// parameters for multi-scale
int Config::scale_num = 8;     // <scale_num
int Config::start_scale = 0;
float Config::scale_stride = sqrt(2);

float Config::bgorient_stride = 15.0f;
float Config::bgmag_stride_ratio = 0.3f;


void initTrackerInfo(TrackerInfo* tracker, int track_length, int init_gap)
{
    tracker->trackLength = track_length;
    tracker->initGap = init_gap;
}


#endif // INITIALIZE_H
