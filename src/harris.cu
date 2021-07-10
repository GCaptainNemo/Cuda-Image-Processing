#pragma once
#include "../include/harris.h"
#include "opencv2/opencv.hpp"
#include "../include/conv.h"
#include "../include/utils.h"
#include <math.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));


// cornerHarris函数对于每一个像素（x,y)在blockSize x blockSize 邻域内，
// 计算2x2梯度的协方差矩阵M(x,y)。就可以找出输出图中的局部最大值，即找出了角点。

namespace harris 
{
	void opencv_harris(const char * address)
	{
		cv::Mat src_img = cv::imread(address);
		cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);
		cv::Mat harris_img;
		cornerHarris(src_img, harris_img, 2, 3, 0.04, cv::BORDER_DEFAULT);
		// harris_img type = CV_32F
		printf("type = %d", harris_img.type());
		cv::Mat harris_bw_img;
		cv::threshold(harris_img, harris_bw_img, 0.00001, 255, cv::THRESH_BINARY);
		cv::namedWindow("bw", cv::WINDOW_NORMAL);
		cv::imshow("bw", harris_bw_img);

		cv::normalize(harris_img, harris_img, 0, 1, cv::NORM_MINMAX, CV_32FC1);

		cv::namedWindow("harris_img", cv::WINDOW_NORMAL);
		cv::imshow("harris_img", harris_img);
		cv::waitKey(0);
	}

	void cuda_harris(cv::Mat & src, cv::Mat & dst, const int & block_size, const float & prop, 
		const int &aperture_size)
	{
		
		cudaSetDevice(0);
		// 3 x 3 Sobel operator
		//float * sobel_y_kernel = new float[aperture_size * aperture_size];
		float * sobel_y = (float *)malloc(aperture_size * aperture_size * sizeof(float));
		sobel_y[0] = -1.; sobel_y[1] = -2.; sobel_y[2] = -1.;
		sobel_y[3] =  0.; sobel_y[4] = 0.; sobel_y[5] = 0.;
		sobel_y[6] =  1.; sobel_y[7] = 2.; sobel_y[8] = 1.;
		//float * sobel_x_kernel = new float[aperture_size * aperture_size];
		float * sobel_x = (float *)malloc(aperture_size * aperture_size * sizeof(float));
		sobel_x[0] = -1.; sobel_x[1] = 0.; sobel_x[2] = 1.;
		sobel_x[3] = -2.; sobel_x[4] = 0.; sobel_x[5] = 2.;
		sobel_x[6] = -1.; sobel_x[7] = 0.; sobel_x[8] = 1.;
		
		// conv get gradient map.
		cv::Mat sobel_x_img;
		conv::cuda_conv(src, sobel_x_img, sobel_x, aperture_size);
		cv::Mat sobel_y_img;
		conv::cuda_conv(src, sobel_y_img, sobel_y, aperture_size);
		printf("conv get gradient map finish \n");
		printf("sobel_x_img.type = %d", sobel_x_img.type());

		// /////////////////////////////////////////////////////////////////////////////////////
		cv::namedWindow("sobel_x", cv::WINDOW_NORMAL);
		cv::imshow("sobel_x", sobel_x_img);
		cv::waitKey(0);



		int img_rows = src.rows;
		int img_cols = src.cols;
		printf("\n src_img size = [%d, %d]\n", img_rows, img_cols);
		printf("\n gradient_img size = [%d, %d]\n", sobel_x_img.rows, sobel_x_img.cols);


		size_t img_size_float = img_cols * img_rows * sizeof(float);
		float * sobel_x_img_vec = NULL;
		float * sobel_y_img_vec = NULL;
		float * gpu_result_vec = NULL;

		// memory allocate
		HANDLE_ERROR(cudaMalloc((void **)& sobel_x_img_vec, img_size_float));
		HANDLE_ERROR(cudaMalloc((void **)& sobel_y_img_vec, img_size_float));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result_vec, img_size_float));

		// memory copy 
		HANDLE_ERROR(cudaMemcpy(sobel_x_img_vec, (float *)sobel_x_img.data, img_size_float, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(sobel_y_img_vec, (float *)sobel_y_img.data, img_size_float, cudaMemcpyHostToDevice));


		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		printf("block_num = %d, thread_num = %d !\n", block_num, thread_num);
		//dim3 thread_grid_size(block_num, 1, 1);
		//dim3 thread_block_size(thread_num, 1, 1);
		
		harris::harris_kernel<<< block_num, thread_num >>>
			(sobel_x_img_vec, sobel_y_img_vec, gpu_result_vec, img_rows, img_cols, block_size, prop);
		
		
		float * cpu_result_vec = (float *)malloc(img_size_float);
		HANDLE_ERROR(cudaMemcpy(cpu_result_vec, gpu_result_vec, img_size_float, cudaMemcpyDeviceToHost));

		dst = cv::Mat(img_rows, img_cols, CV_32FC1, cpu_result_vec).clone();
		
		HANDLE_ERROR(cudaFree(sobel_x_img_vec));
		HANDLE_ERROR(cudaFree(sobel_y_img_vec));
		HANDLE_ERROR(cudaFree(gpu_result_vec));
		free(cpu_result_vec);
		free(sobel_y);
		free(sobel_x);
		cudaDeviceReset();
	}

	void cuda_harris(float * src, float * dst, const int & img_rows, const int & img_cols, const int & block_size,
		const float & prop, const int &aperture_size) 
	{
		cudaSetDevice(0);
		// 3 x 3 Sobel operator
		//float * sobel_y_kernel = new float[aperture_size * aperture_size];
		float * sobel_y_kernel = (float *)malloc(aperture_size * aperture_size * sizeof(float));
		sobel_y_kernel[0] = -1.; sobel_y_kernel[1] = -2.; sobel_y_kernel[2] = -1.;
		sobel_y_kernel[3] = 0.; sobel_y_kernel[4] = 0.; sobel_y_kernel[5] = 0.;
		sobel_y_kernel[6] = 1.; sobel_y_kernel[7] = 2.; sobel_y_kernel[8] = 1.;
		//float * sobel_x_kernel = new float[aperture_size * aperture_size];
		float * sobel_x_kernel = (float *)malloc(aperture_size * aperture_size * sizeof(float));
		sobel_x_kernel[0] = -1.; sobel_x_kernel[1] = 0.; sobel_x_kernel[2] = 1.;
		sobel_x_kernel[3] = -2.; sobel_x_kernel[4] = 0.; sobel_x_kernel[5] = 2.;
		sobel_x_kernel[6] = -1.; sobel_x_kernel[7] = 0.; sobel_x_kernel[8] = 1.;


		// conv get gradient map.
		size_t img_size_float = img_cols * img_rows * sizeof(float);

		float * sobel_x_img = (float *)malloc(img_size_float);

		conv::cuda_conv(src, sobel_x_img, img_rows, img_cols, sobel_x_kernel, aperture_size);

		float * sobel_y_img = (float *)malloc(img_size_float);
		conv::cuda_conv(src, sobel_y_img, img_rows, img_cols, sobel_y_kernel, aperture_size);

	
		float * sobel_x_img_vec = NULL;
		float * sobel_y_img_vec = NULL;
		float * gpu_result_vec = NULL;

		// memory allocate
		HANDLE_ERROR(cudaMalloc((void **)& sobel_x_img_vec, img_size_float));
		HANDLE_ERROR(cudaMalloc((void **)& sobel_y_img_vec, img_size_float));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result_vec, img_size_float));

		// memory copy 
		HANDLE_ERROR(cudaMemcpy(sobel_x_img_vec, sobel_x_img, img_size_float, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(sobel_y_img_vec, sobel_y_img, img_size_float, cudaMemcpyHostToDevice));


		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		printf("block_num = %d, thread_num = %d !\n", block_num, thread_num);
		//dim3 thread_grid_size(block_num, 1, 1);
		//dim3 thread_block_size(thread_num, 1, 1);

		harris::harris_kernel << < block_num, thread_num >> >
			(sobel_x_img_vec, sobel_y_img_vec, gpu_result_vec, img_rows, img_cols, block_size, prop);


		HANDLE_ERROR(cudaMemcpy(dst, gpu_result_vec, img_size_float, cudaMemcpyDeviceToHost));

		// //////////////////////////////////////////////////////////////////////
		// release memory
		// /////////////////////////////////////////////////////////////////////
		HANDLE_ERROR(cudaFree(sobel_x_img_vec));
		HANDLE_ERROR(cudaFree(sobel_y_img_vec));
		HANDLE_ERROR(cudaFree(gpu_result_vec));

		free(sobel_y_kernel);
		free(sobel_x_kernel);
		free(sobel_x_img);
		free(sobel_y_img);
		
		cudaDeviceReset();
	};


	__global__ void harris_kernel(float * sobel_x_vec, float * sobel_y_vec, float * gpu_result_vec, 
		const int img_row, const int img_col, const int block_size, 
		const float prop) 
	{
		
		int thread_id = threadIdx.x;
		int block_id = blockIdx.x;
		int index = block_id * blockDim.x + thread_id;
		
		int pixel_col = index % img_col;
		int pixel_row = index / img_col;
		// calculate covariance matrix
		float gradient_x_mean = 0;
		float gradient_y_mean = 0;
		float gradient_xy_mean = 0;
		float gradient_xx_mean = 0;
		float gradient_yy_mean = 0;
		float num = block_size * block_size;

		for (int i = 0; i < block_size; ++i)
		{
			for (int j = 0; j < block_size; ++j) 
			{
				int cur_row = pixel_row - block_size / 2 + i;
				int cur_col = pixel_col - block_size / 2 + j;
				//float 
				float gradient_x = 0.;
				float gradient_y = 0.;
				if (cur_row < 0 || cur_row >= img_row || cur_col < 0 || cur_col >= img_col) 
				{
				}
				else {
					gradient_x = sobel_x_vec[cur_row * img_col + cur_col];
					gradient_y = sobel_y_vec[cur_row * img_col + cur_col];
				}
				gradient_x_mean += gradient_x ;
				gradient_y_mean += gradient_y ;
				gradient_xx_mean += gradient_x * gradient_x ;
				gradient_yy_mean += gradient_y * gradient_y ;
				gradient_xy_mean += gradient_x * gradient_y ;
			}
		}
		float cov_mat_00 = gradient_xx_mean;
		float cov_mat_01 = gradient_xy_mean;
		float cov_mat_11 = gradient_yy_mean;
		// det(H) - k * trace(H) ^ 2, H represpent harris matrix
		float discri_cond = cov_mat_00 * cov_mat_11 - cov_mat_01 * cov_mat_01 - 
			prop * (cov_mat_00 + cov_mat_11) * (cov_mat_00 + cov_mat_11);
		gpu_result_vec[index] = discri_cond;
		if (index == 0) 
		{
			printf("result_vec[0] = %f", gpu_result_vec[index]);
		}
	}
}

