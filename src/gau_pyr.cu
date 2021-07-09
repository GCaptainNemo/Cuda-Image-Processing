#pragma once
#include "../include/gau_pyr.h"
#include "opencv2/opencv.hpp"
#include "../include/utils.h"
#include "../include/conv.h"
#include <math.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));


namespace gau_pyr
{
	__global__ void gaussian_pyramid_down_kernel(float * gpu_src, float * gpu_dst, float * kernel,
		const int src_img_rows, const int src_img_cols,
		const int dst_img_rows, const int dst_img_cols, const int kernel_dim)
	{
		int thread_id = threadIdx.x;
		int block_id = blockIdx.x;

		int src_pixel_id = block_id * blockDim.x + thread_id;
		int src_pixel_row = src_pixel_id / src_img_cols;
		int src_pixel_col = src_pixel_id % src_img_cols;

		if (src_pixel_id >= src_img_cols * src_img_rows || (src_pixel_row % 2 == 1 && src_pixel_col % 2 == 1))
		{
			// out src img and even rows and cols delete
			return;
		}
		int dst_pixel_row = src_pixel_row / 2;
		int dst_pixel_col = src_pixel_col / 2;
		int dst_pixel_id = dst_pixel_row * dst_img_cols + dst_pixel_col;
		gpu_dst[dst_pixel_id] = 0;
		for (int i = 0; i < kernel_dim; ++i)
		{
			for (int j = 0; j < kernel_dim; ++j)
			{
				float img_val = 0;
				int cur_rows = src_pixel_row - kernel_dim / 2 + i;
				int cur_cols = src_pixel_col - kernel_dim / 2 + j;
				if (cur_cols < 0 || cur_rows < 0 || cur_cols >= src_img_cols || cur_rows >= src_img_rows)
				{
				}
				else
				{
					img_val = gpu_src[cur_cols + cur_rows * src_img_cols];
				}
				gpu_dst[dst_pixel_id] += (kernel[i * kernel_dim + j]) * img_val;

			}
		}
	}

	__global__ void down_sample_kernel(float * gpu_src, float * gpu_dst, const int src_img_rows, const int src_img_cols,
		const int dst_img_rows, const int dst_img_cols)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dst_img_rows * dst_img_cols)
		{
			return;
		}
		int row = index / dst_img_cols;
		int col = index % dst_img_cols;
		gpu_dst[index] = gpu_src[row * 2 * src_img_cols + col * 2];
	}


	void get_gaussian_blur_kernel(float &sigma, int &kernel_size, float ** gaussian_kernel)
	{
		float sum = 0;
		if (sigma <= 0 && kernel_size > 0)
		{
			int center = (kernel_size - 1) / 2;
			sigma = 0.3 * (center - 1) + 0.8;
		}
		else if (sigma > 0 && kernel_size <= 0) 
		{
			// according to opencv createGaussianFilter API： atleast size = 1
			kernel_size = ((int)round(sigma * 8 + 1)) | 1; 
		}
		int center = (kernel_size - 1) / 2;
		printf("kernel_size = %d", kernel_size);
		printf("sigma = %f", sigma);

		*gaussian_kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));
		for (int row = 0; row < kernel_size; ++row)
		{
			for (int col = 0; col < kernel_size; ++col)
			{
				int index = row * kernel_size + col;
				float linshi = exp(-(pow(row - center, 2) + pow(col - center, 2)) / 2 / pow(sigma, 2));
				(*gaussian_kernel)[index] = linshi;
				sum += linshi;
			}
		}
		for (int index = 0; index < kernel_size * kernel_size; ++index)
		{
			(*gaussian_kernel)[index] /= sum;
		}
	}

	void down_sampling(cv::Mat & src, cv::Mat & dst)
	{
		HANDLE_ERROR(cudaSetDevice(0));
		float * gpu_src_img = NULL;
		float * gpu_res_img = NULL;
		int src_rows = src.rows;
		int src_cols = src.cols;
		int dst_rows = (src_rows + 1) / 2;
		int dst_cols = (src_cols + 1) / 2;
		size_t src_size = src_rows * src_cols * sizeof(float);
		size_t dst_size = dst_rows * dst_cols * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_src_img, src_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_res_img, dst_size));
		HANDLE_ERROR(cudaMemcpy(gpu_src_img, src.data, src_size, cudaMemcpyHostToDevice));

		int thread_num = getThreadNum();
		int block_num = (dst_cols * dst_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		gau_pyr::down_sample_kernel << < grid_size, block_size >> > (gpu_src_img, gpu_res_img, src_rows, src_cols, dst_rows, dst_cols);
		float * cpu_result = (float *)malloc(dst_size);
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_res_img, dst_size, cudaMemcpyDeviceToHost));

		dst = cv::Mat(dst_rows, dst_cols, CV_32FC1, cpu_result).clone();
		cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX);

		// free the memory
		cudaFree(gpu_res_img);
		cudaFree(gpu_src_img);
		free(cpu_result);
		cudaDeviceReset();
	};

	void cuda_pyramid_down(cv::Mat & src, cv::Mat & dst, int &kernel_dim, float & sigma)
	{
		HANDLE_ERROR(cudaSetDevice(0));
		float * kernel;
		// if kernel_dim < 0 or sigma < 0 will modify to > 0
		gau_pyr::get_gaussian_blur_kernel(sigma, kernel_dim, &kernel);
		printf("out kernel_dim = %d\n", kernel_dim);
		size_t kernel_size_t = kernel_dim * kernel_dim * sizeof(float);
		// ///////////////////////////////////////////////////////
		//
		// /////////////////////////////////////////////////////////
		for (int i = 0; i < kernel_dim; ++i)
		{
			for (int j = 0; j < kernel_dim; ++j)
			{
				printf("%f ", kernel[i * kernel_dim + j]);
			}
			printf("\n");
		}
		float * gpu_src_img = NULL;
		float * gpu_dst_img = NULL;
		float * gpu_kernel = NULL;
		int src_rows = src.rows;
		int src_cols = src.cols;
		int dst_rows = (src_rows + 1) / 2;
		int dst_cols = (src_cols + 1) / 2;
		size_t src_size = src_rows * src_cols * sizeof(float);
		size_t dst_size_t = dst_rows * dst_cols * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_src_img, src_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_dst_img, dst_size_t));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_kernel, kernel_size_t));

		HANDLE_ERROR(cudaMemcpy(gpu_src_img, src.data, src_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_kernel, kernel, kernel_size_t, cudaMemcpyHostToDevice));

		int thread_num = getThreadNum();
		int block_num = (src_cols * src_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		gau_pyr::gaussian_pyramid_down_kernel << < grid_size, block_size >> >
			(gpu_src_img, gpu_dst_img, gpu_kernel, src_rows, src_cols, dst_rows, dst_cols, kernel_dim);

		float * cpu_result = (float *)malloc(dst_size_t);
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_dst_img, dst_size_t, cudaMemcpyDeviceToHost));

		// return and normalize
		dst = cv::Mat(dst_rows, dst_cols, CV_32FC1, cpu_result).clone();
		cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX); // idempotent operator

		// free the memory
		cudaFree(gpu_dst_img);
		cudaFree(gpu_src_img);
		cudaFree(gpu_kernel);
		free(cpu_result);
		free(kernel);
		cudaDeviceReset();
	};

	// return gaussian pyramid (octave x (intervals + 3) imgs
	void build_gauss_pry(cv::Mat * src, cv::Mat *** dst, int octave, int intervals, float sigma) 
	{
		// every octave has intervals + 3 image(default 6)
		double * sigma_diff_array = (double *)calloc(intervals + 3, sizeof(float));
		dst = (cv::Mat ***)calloc(octave, sizeof(**dst)); // octave image
		for (int i = 0; i < octave; ++i) 
		{
			dst[i] = (cv::Mat **)calloc(intervals + 3, sizeof(*dst));
		}
		// init sigma (1.6 default)
		sigma_diff_array[0] = sigma;
		// 相邻层Sigma的比值
		float k = pow(2.0, 1.0 / intervals);
		for (int i = 1; i < intervals + 3; ++i)
		{
			float sig_prev = pow(k, i - 1) * sigma; 
			float sig_total = sig_prev * k; 
			sigma_diff_array[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev); 
		}
		for (int o = 0; o < octave; ++o) 
		{
			for (int i = 0; i < intervals + 3; ++i) 
			{
				// bottom
				if (o == 0 && i == 0) 
				{
					dst[o][i] = src;
				}
				else if (i == 0) 
				{
					// first interval of each octave
					cv::Mat * down_sample_img;
					gau_pyr::down_sampling(*src, *down_sample_img);
					dst[o][i] = down_sample_img;
				}
				else 
				{
					// 在上一张图像上继续做gaussian blur(由于gaussian卷积的封闭性)
					float blur_sigma = sigma_diff_array[i];
					cv::Mat * blur_img;
					float * kernel;
					int size = -1;
					gau_pyr::get_gaussian_blur_kernel(blur_sigma, size, &kernel);

				}
			}
		}



	};

}
