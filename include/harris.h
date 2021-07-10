#pragma once
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>


namespace harris 
{
	void opencv_harris(const char * address);
	
	// cuda harris
	void cuda_harris(cv::Mat & src, cv::Mat & dst, const int & block_size, 
		const float & prop, const int &aperture_size);

	// overload use float *
	void cuda_harris(float * src, float * dst, const int & img_rows, const int & img_cols, const int & block_size,
		const float & prop, const int &aperture_size);

	
	// __global__ ����������&���ô������
	__global__ void harris_kernel(float * sobel_x_vec, float * sobel_y_vec, float * result_vec,
		const int img_row, const int img_col, const int block_size, 
		const float prop);

}

