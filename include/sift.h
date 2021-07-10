#pragma once
#include <opencv2/opencv.hpp>

namespace sift 
{
	void detect_extreme_point(cv::Mat *** dog_pyramid, int octvs, int intervals, double contr_thresh, int curve_thresh) ;

	// remove low contrast points and edge points£¨Harris corner£©.
	void remove_points();
}

