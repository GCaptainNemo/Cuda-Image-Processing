#pragma once
#include "cuda_runtime.h"


namespace harris 
{
	void opencv_harris(const char * address);
	/*__global__ void harris_kernal(float * gpu_img, float * kernel, 
		const int img_row, const int img_col);*/
	void cuda_harris(const char * address, const int & block_size, const float & threshold, const int &aperture_size = 3);
}

