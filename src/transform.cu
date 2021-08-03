#include <stdio.h>
#include "../include/transform.h"
#include "../include/utils.h"
#include "pybind11/pybind11.h"

#include <opencv2/opencv.hpp>
#include <math.h>

// pytho API


#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

namespace transform {
	__global__ void transform_kernel(int * gpu_img, int * gpu_result, float * gpu_dst_grid_pos,
	float * gpu_homography_dst_to_src, const int img_cols, const int img_rows, const int grid_cols, const int grid_rows)
	{
		// rgb img as input(row dst_x col dst_x 3)
		// grid_pos((grid_row + 1) dst_x (grid_col + 1) dst_x 2)
		// homography(grid_row dst_x grid_col dst_x 9)
		int thread_id = threadIdx.x;
		int block_id = blockIdx.x;
		

		int pixel_id = block_id * blockDim.x + thread_id;
		if (pixel_id >= img_rows * img_cols)
		{
			return;
		}
		
		float dst_y = (float)(pixel_id / img_cols);  // row
		float dst_x = (float)(pixel_id% img_cols);  // col
		if (pixel_id == 12743)
		{	
			printf("pixel_id = %d	, dst_y = %f,	dst_x =%f", pixel_id, dst_y, dst_x);
			printf("img_cols = %d, img_rows = %d, grid_cols = %d, grid_rows = %d", img_cols, img_rows, grid_cols, grid_rows);
		}	
		for (int i = 0; i < 3; ++i)
		{
			gpu_result[pixel_id * 3 + i] = 0;
		}
		
		for (int i = 0; i < grid_rows; ++i)
		{
			for (int j = 0; j < grid_cols; ++j)
			{
				// grid points id(顶点并网格数多1)
				int id = i * (grid_cols + 1) + j;
				float left_up_delta_x = gpu_dst_grid_pos[id * 2] - dst_x;
				float left_up_delta_y = gpu_dst_grid_pos[id * 2 + 1] - dst_y;
				float left_down_delta_x = gpu_dst_grid_pos[(id + grid_cols + 1) * 2] - dst_x;
				float left_down_delta_y = gpu_dst_grid_pos[(id + grid_cols + 1) * 2 + 1] - dst_y;
				float right_down_delta_x = gpu_dst_grid_pos[(id + grid_cols + 2) * 2] - dst_x;
				float right_down_delta_y = gpu_dst_grid_pos[(id + grid_cols + 2) * 2 + 1] - dst_y;
				float right_up_delta_x = gpu_dst_grid_pos[(id + 1) * 2] - dst_x;
				float right_up_delta_y = gpu_dst_grid_pos[(id + 1) * 2 + 1] - dst_y;
				// ///////////////////////////////////////////////////////////////////////
				// cross product
				// ////////////////////////////////////////////////////////////////////////
				float prod_1 = left_up_delta_x * left_down_delta_y - left_up_delta_y * left_down_delta_x;
				if (prod_1 > 0) {
					continue;
				}
				float prod_2 = left_down_delta_x * right_down_delta_y - left_down_delta_y * right_down_delta_x;
				if (prod_2 >= 0) {
					continue;
				}
				float prod_3 = right_down_delta_x * right_up_delta_y - right_down_delta_y * right_up_delta_x;
				if (prod_3 >= 0) {
					continue;
				}
				float prod_4 = right_up_delta_x * left_up_delta_y - right_up_delta_y * left_up_delta_x;
				if (prod_4 > 0) {
					continue;
				}
				// //////////////////////////////////////////////////////////////////////////////
				// dst_x, dst_y inside grid
				// //////////////////////////////////////////////////////////////////////////////
				int homography_id = i * grid_cols + j;
				float src_x = gpu_homography_dst_to_src[homography_id * 9 + 0] * dst_x + 
							  gpu_homography_dst_to_src[homography_id * 9 + 1] * dst_y + 
							  gpu_homography_dst_to_src[homography_id * 9 + 2];
				float src_y = gpu_homography_dst_to_src[homography_id * 9 + 3] * dst_x + 
							  gpu_homography_dst_to_src[homography_id * 9 + 4] * dst_y + 
							  gpu_homography_dst_to_src[homography_id * 9 + 5];
				float nor_f = gpu_homography_dst_to_src[homography_id * 9 + 6] * dst_x + 
							  gpu_homography_dst_to_src[homography_id * 9 + 7] * dst_y + 
							  gpu_homography_dst_to_src[homography_id * 9 + 8];
				src_x = src_x / nor_f;
				src_y = src_y / nor_f;
				
				if (src_x >= 0 && src_x < img_cols && src_y >= 0 && src_y < img_rows)
				{
					// printf("src_x = %f, src_y = %f", src_x, src_y);
					int left_up_x = (int)src_x;
					int left_up_y = (int)src_y;
					// printf("left_up_x = %d, left_up_y = %d", left_up_x, left_up_y);
									
					int left_up_idx = (left_up_x + left_up_y * img_cols) * 3;
					int left_down_idx = (left_up_x + (left_up_y + 1)* img_cols) * 3;
					int right_down_idx = (left_up_x + 1 + (left_up_y + 1)* img_cols) * 3;
					int right_up_idx = (left_up_x + 1 + left_up_y * img_cols) * 3;

					float proportion_x = src_x - (float)left_up_x;
					float proportion_y = src_y - (float)left_up_y;
					for (int offset = 0; offset < 3; ++offset)
					{

						float res = (1 - proportion_x) * (1 - proportion_y) * gpu_img[left_up_idx + offset] +
							(1 - proportion_x) * proportion_y * gpu_img[left_down_idx + offset] +
							proportion_x * (1 - proportion_y) * gpu_img[right_up_idx + offset] +
							proportion_x * proportion_y * gpu_img[right_down_idx + offset];
						if (res > 255) {
							gpu_result[pixel_id * 3 + offset] = 255;
						}
						else {
							gpu_result[pixel_id * 3 + offset] = (int)res;
						}
					}
				}
				return;

			}
		}
	}
	
	

	void cuda_transform(cv::Mat & src, cv::Mat & dst, float * cpu_dst_grid_pos, float * cpu_src_grid_pos,
		float * cpu_homography_dst_to_src, int grid_cols, int grid_rows)
	{
		// read linshi and convert to gray image
		cudaSetDevice(0);

		cv::Mat linshi;
		if (src.type() != CV_8UC3) {
			return;
		}
		else {
			linshi = src.clone();
		}
		int img_cols = linshi.cols;
		int img_rows = linshi.rows;


		int * gpu_img = NULL;
		int * gpu_result = NULL;
		float * gpu_dst_grid_pos = NULL;
		float * gpu_src_grid_pos = NULL;
		float * gpu_homography_dst_to_src = NULL;

		size_t img_size = img_cols * img_rows * sizeof(uint3);
		size_t grid_pos_size = (grid_cols + 1)* (grid_rows + 1) * 2 * sizeof(float);
		size_t homography_size = grid_cols * grid_rows * 9 * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_dst_grid_pos, grid_pos_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_src_grid_pos, grid_pos_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_homography_dst_to_src, homography_size));

		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, linshi.data, img_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_dst_grid_pos, cpu_dst_grid_pos, grid_pos_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_src_grid_pos, cpu_src_grid_pos, grid_pos_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_homography_dst_to_src, cpu_homography_dst_to_src, homography_size, cudaMemcpyHostToDevice));

		// //////////////////////////////////////////////////////////////////////////////////////////////
		// resident thread; every pixel of result correspond to a thread;
		// //////////////////////////////////////////////////////////////////////////////////////////////

		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		transform::transform_kernel << < grid_size, block_size >> > (gpu_img, gpu_dst_grid_pos, gpu_src_grid_pos,
			gpu_homography_dst_to_src, gpu_result,
			img_cols, img_rows, grid_cols, grid_rows);

		float * cpu_result = new float[img_cols * img_rows * 3];
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_result, img_size, cudaMemcpyDeviceToHost));

		dst = cv::Mat(img_rows, img_cols, CV_8UC3, cpu_result).clone();
		printf("row = 0, col=0, val = %f", dst.at<int>(0, 0));

		cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX);

		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_src_grid_pos));
		HANDLE_ERROR(cudaFree(gpu_dst_grid_pos));
		HANDLE_ERROR(cudaFree(gpu_homography_dst_to_src));
		HANDLE_ERROR(cudaFree(gpu_result));
		delete[] cpu_result;
		cudaDeviceReset();
	}

	void cuda_transform(int * src, int * dst, float * cpu_dst_grid_pos, float * cpu_src_grid_pos,
		float * cpu_homography_dst_to_src, int grid_cols, int grid_rows, int img_cols, int img_rows)
	{
		// read linshi and convert to gray image
		// read linshi and convert to gray image
		cudaSetDevice(0);

		int * gpu_img = NULL;
		int * gpu_result = NULL;
		float * gpu_dst_grid_pos = NULL;
		float * gpu_src_grid_pos = NULL;
		float * gpu_homography_dst_to_src = NULL;

		size_t img_size = img_cols * img_rows * sizeof(uint3);
		size_t grid_pos_size = (grid_cols + 1)* (grid_rows + 1) * 2 * sizeof(float);
		size_t homography_size = grid_cols * grid_rows * 9 * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_dst_grid_pos, grid_pos_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_src_grid_pos, grid_pos_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_homography_dst_to_src, homography_size));

		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, src, img_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_dst_grid_pos, cpu_dst_grid_pos, grid_pos_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_src_grid_pos, cpu_src_grid_pos, grid_pos_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_homography_dst_to_src, cpu_homography_dst_to_src, homography_size, cudaMemcpyHostToDevice));

		// //////////////////////////////////////////////////////////////////////////////////////////////
		// resident thread; every pixel of result correspond to a thread;
		// //////////////////////////////////////////////////////////////////////////////////////////////

		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		transform::transform_kernel << < grid_size, block_size >> > (gpu_img, gpu_dst_grid_pos, gpu_src_grid_pos,
			gpu_homography_dst_to_src, gpu_result,
			img_cols, img_rows, grid_cols, grid_rows);


		HANDLE_ERROR(cudaMemcpy(dst, gpu_result, img_size, cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_src_grid_pos));
		HANDLE_ERROR(cudaFree(gpu_dst_grid_pos));
		HANDLE_ERROR(cudaFree(gpu_homography_dst_to_src));
		HANDLE_ERROR(cudaFree(gpu_result));
		cudaDeviceReset();
	}


}