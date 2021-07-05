#pragma once
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>


namespace harris 
{
	void opencv_harris(const char * address);
	void cuda_harris(cv::Mat & src, cv::Mat & dst, const int & block_size, 
		const float & prop, const int &aperture_size);
	
	// __global__ 函数不能用&引用传入变量
	__global__ void harris_kernel(float * sobel_x_vec, float * sobel_y_vec, float * result_vec,
		const int img_row, const int img_col, const int block_size, 
		const float prop);

}

