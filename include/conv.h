#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

namespace conv 
{
	// conv on gpu
	__global__ void conv_kernel(float *gpu_img, float * gpu_kernel, float * gpu_result,
		const int img_cols, const int img_rows, const int kernel_dim);
	
	// down sample on gpu(delete even row, col)
	__global__ void down_sample_kernel(float * gpu_src, float * gpu_dst, const int src_img_rows, const int src_img_cols,
		const int dst_img_rows, const int dst_img_cols);

	void cuda_conv(cv::Mat & src, cv::Mat & dst, float * kernel, int kernel_dim);

	void opencv_conv(const char * address);

	// calculate gaussian kernel, mod(kernel_size, 2) = 1 
	void get_gaussian_blur_kernel(float & sigma, const int & kernel_size, float * gaussian_kernel);

	// pyramid down sampling(gaussian pyramid downsampling delete even row, col)
	void down_sampling(cv::Mat & src, cv::Mat & dst);

}