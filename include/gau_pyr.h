#pragma once
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"


namespace gau_pyr 
{
	__global__ void gaussian_pyramid_down_kernel(float * gpu_src, float * gpu_dst, float * kernel,
		const int src_img_rows, const int src_img_cols,
		const int dst_img_rows, const int dst_img_cols, const int kernel_dim);

	// down sample on gpu(delete even row, col)
	__global__ void down_sample_kernel(float * gpu_src, float * gpu_dst, const int src_img_rows, const int src_img_cols,
		const int dst_img_rows, const int dst_img_cols);


	// calculate gaussian kernel, mod(kernel_size, 2) = 1 
	void get_gaussian_blur_kernel(float &sigma, int &kernel_size, float ** gaussian_kernel);

	// down sampling(delete even row, col)
	void down_sampling(cv::Mat & src, cv::Mat & dst);

	// pyramid downsample
	void cuda_pyramid_down(cv::Mat & src, cv::Mat & dst, int &size, float & sigma);

	// constuct gaussian pyramid
	void build_gauss_pry(cv::Mat src, float *** dst, int octave, int intervals, float sigma);
	
}