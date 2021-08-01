#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

namespace transform
{
	// conv on gpu
	__global__ void transform_kernel(int * gpu_img, float * gpu_dst_grid_pos, float * gpu_src_grid_pos,
		float * gpu_homography_dst_to_src, int * gpu_result,
		const int img_cols, const int img_rows, const int grid_cols, const int grid_rows);

	// float replace cv::Mat
	void cuda_transform(float * src, float * dst, int img_row, int img_col, float * kernel, int kernel_dim);

	void cuda_transform(cv::Mat & src, cv::Mat & dst, float * gpu_dst_grid_pos, float * gpu_src_grid_pos,
		float * gpu_homography_dst_to_src)

}