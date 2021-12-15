#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"


namespace  morphology
{
	// imerode on gpu
	__global__ void imerode_kernel(float *gpu_img, bool *gpu_structre, float * gpu_result,
		const int img_cols, const int img_rows, const int structure_col, const int structure_row, 
		const int anchor_col, const int anchor_row);

	// imdilate on gpu
	__global__ void imdilate_kernel(float *gpu_img, bool * gpu_structre, float * gpu_result,
		const int img_cols, const int img_rows, const int structure_col, const int structure_row,
		const int anchor_col, const int anchor_row);

	// 
	void launch_imerode(cv::Mat & src, cv::Mat & dst, bool * structure, const int structure_col, const int structure_row,
		const int anchor_col, const int anchor_row);

	void launch_imdilate(cv::Mat & src, cv::Mat & dst, bool * structure, const int structure_col, const int structure_row,
		const int anchor_col, const int anchor_row);
}