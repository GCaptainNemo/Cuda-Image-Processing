#pragma once
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include <vector>


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
	void cuda_down_sampling(cv::Mat & src, cv::Mat & dst);

	// down sampling(delete even row, col)
	void cuda_down_sampling(float * src, float ** dst, const int & src_rows, const int & src_cols);

	// pyramid downsample
	void cuda_pyramid_down(cv::Mat & src, cv::Mat & dst, int &size, float & sigma);

	// constuct gaussian pyramid
	void cuda_build_gauss_pyramid(cv::Mat src, std::vector<std::vector<cv::Mat *>> &dst, int octave, int intervals, float sigma);
	
	// use pointer
	void cuda_build_gauss_pyramid(cv::Mat src, cv::Mat **** dst, int octave, int intervals, float sigma);

	// use float
	void cuda_build_gauss_pyramid(float * src, float **** dst, const int & origin_rows, const int & origin_cols,
		const int &octave, const int &intervals, float sigma);

	// build DoG pyramid
	void build_dog_pyr(cv::Mat *** gaussian_pyramid, cv::Mat **** dog_pyramid, int octave, int intervals);

	// build DoG pyramid
	void build_dog_pyr(float *** gaussian_pyramid, float **** dog_pyramid, int ** row_col_lst, int octave, int intervals);

}