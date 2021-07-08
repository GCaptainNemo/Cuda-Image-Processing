#include <stdio.h>
#include "../include/conv.h"
#include "opencv2/opencv.hpp"
#include "../include/utils.h"
#include <math.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

namespace conv {
	__global__ void conv_kernel(float *gpu_img, float * gpu_kernel, float * gpu_result,
		const int img_cols, const int img_rows, const int kernel_dim)
	{
		int thread_id = threadIdx.x;
		int block_id = blockIdx.x;

		int pixel_id = block_id * blockDim.x + thread_id;
		if (pixel_id >= img_rows * img_cols)
		{
			return;
		}

		int row = pixel_id / img_cols;
		int col = pixel_id % img_cols;

		for (int i = 0; i < kernel_dim; ++i)
		{
			for (int j = 0; j < kernel_dim; ++j)
			{
				float img_val = 0;
				int cur_rows = row - kernel_dim / 2 + i;
				int cur_cols = col - kernel_dim / 2 + j;
				if (cur_cols < 0 || cur_rows < 0 || cur_cols >= img_cols || cur_rows >= img_rows)
				{
				}
				else
				{
					img_val = gpu_img[cur_cols + cur_rows * img_cols];
				}
				gpu_result[pixel_id] += (gpu_kernel[i * kernel_dim + j]) * img_val;
			}
		}
	}

	__global__ void down_sample_kernel(float * gpu_src, float * gpu_dst, const int src_img_rows, const int src_img_cols, 
		const int dst_img_rows, const int dst_img_cols)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= dst_img_rows * dst_img_cols)
			{return;}
		int row = index / dst_img_cols;
		int col = index % dst_img_cols;
		gpu_dst[index] = gpu_src[row * 2 * src_img_cols + col * 2];
	}


	void cuda_conv(cv::Mat & src, cv::Mat & dst, float * kernel, int kernel_dim)
	{
		// read linshi and convert to gray image
		cudaSetDevice(0);
		cv::Mat linshi;
		cv::cvtColor(src, linshi, cv::COLOR_BGR2GRAY);
		printf("origin gray img\n");
		// uchar to float
		linshi.convertTo(linshi, CV_32FC1);
		int img_cols = linshi.cols;
		int img_rows = linshi.rows;


		float * gpu_img;
		float * gpu_result;
		float * gpu_kernel;


		size_t img_size = img_cols * img_rows * sizeof(float);
		size_t kernel_size = kernel_dim * kernel_dim * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_kernel, kernel_size));
		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, linshi.data, img_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_kernel, kernel, kernel_size, cudaMemcpyHostToDevice));

		// //////////////////////////////////////////////////////////////////////////////////////////////
		// resident thread; every pixel of result correspond to a thread;
		// //////////////////////////////////////////////////////////////////////////////////////////////

		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		conv::conv_kernel <<< grid_size, block_size >> > (gpu_img, gpu_kernel, gpu_result, img_cols, img_rows, kernel_dim);
		
		float * cpu_result = new float[img_cols * img_rows];
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_result, img_size, cudaMemcpyDeviceToHost));
		
		dst = cv::Mat(img_rows, img_cols, CV_32FC1, cpu_result).clone();
		printf("row = 0, col=0, val = %f", dst.at<float>(0, 0));

		cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX);

		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_kernel));
		HANDLE_ERROR(cudaFree(gpu_result));
		delete [] cpu_result;
		cudaDeviceReset();
	}

	void opencv_conv(const char * address)
	{
		cv::Mat kernel_ = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
		cv::Mat src_img = cv::imread(address);
		cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);
		src_img.convertTo(src_img, CV_32FC1);
		cv::Mat dst_img;
		cv::filter2D(src_img, dst_img, src_img.depth(), kernel_);
		cv::normalize(dst_img, dst_img, 1.0, 0.0, cv::NORM_MINMAX);
		cv::namedWindow("dst_img", cv::WINDOW_NORMAL);
		cv::imshow("dst_img", dst_img);
		cv::waitKey(0);
	}

	void get_gaussian_blur_kernel(float & sigma, const int & kernel_size, float * gaussian_kernel) 
	{
		float sum = 0;
		int center = (kernel_size - 1) / 2;
		if (sigma <= 0) 
		{
			sigma = 0.3 * (center - 1) + 0.8;
		}
		for (int row = 0; row < kernel_size; ++row) 
		{
			for (int col = 0; col < kernel_size; ++col) 
			{
				int index = row * kernel_size + col;
				float linshi = exp(-(pow(row - center, 2) + pow(col - center, 2)) / 2 / pow(sigma, 2));
				gaussian_kernel[index] = linshi;
				sum += linshi;
			}
		}
		for (int index = 0; index < kernel_size *kernel_size; ++index) 
		{
			gaussian_kernel[index] /= sum;
		}
	}

	void down_sampling(cv::Mat & src, cv::Mat & dst)
	{
		HANDLE_ERROR(cudaSetDevice(0));
		float * gpu_src_img;
		float * gpu_res_img;
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
		conv::down_sample_kernel<< < grid_size, block_size >> > (gpu_src_img, gpu_res_img, src_rows, src_cols, dst_rows, dst_cols);
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

}