#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

namespace conv 
{
	__global__ void conv_kernel(float *gpu_img, float * gpu_kernel, float * gpu_result,
		const int img_cols, const int img_rows, const int kernel_dim);

	void cuda_conv(cv::Mat & src, cv::Mat & dst, float * kernel, int kernel_dim);

	void opencv_conv(const char * address);
}