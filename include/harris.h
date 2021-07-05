#pragma once
#include "cuda_runtime.h"


namespace harris 
{
	void opencv_harris(const char * address);
	
	__global__ void harris_kernal(float * sobel_x_vec, float * sobel_y_vec, float * result_vec,
		const int &img_row, const int &img_col, const int & block_size);

	void cuda_harris(cv::Mat & src, cv::Mat & dst, const int & block_size, const float & threshold, const int &aperture_size = 3);
}

