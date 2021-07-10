#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

namespace sift 
{
	__global__ void kernel_detect_extreme(float *** dog_pyramid_gpu, int *** mask_gpu, int **row_col_gpu,
		int total_interval);

	// detect extreme point on the DoG pyramid
	void detect_extreme_point(cv::Mat **** dog_pyramid, int **** musk, int octvs, int intervals) ;

	// remove low contrast points and edge points£¨Harris corner£©.
	void remove_points();
}

