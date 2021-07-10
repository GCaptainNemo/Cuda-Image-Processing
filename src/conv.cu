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
		// the kernel anchor is at the center.
		int thread_id = threadIdx.x;
		int block_id = blockIdx.x;

		int pixel_id = block_id * blockDim.x + thread_id;
		if (pixel_id >= img_rows * img_cols)
		{
			return;
		}

		int row = pixel_id / img_cols;
		int col = pixel_id % img_cols;
		gpu_result[pixel_id] = 0;
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

	void cuda_conv(cv::Mat & src, cv::Mat & dst, float * kernel, int kernel_dim)
	{
		// read linshi and convert to gray image
		cudaSetDevice(0);
		
		cv::Mat linshi;
 		if (src.type() == CV_8UC3) {
			cv::cvtColor(src, linshi, cv::COLOR_BGR2GRAY);
			printf("origin gray img\n");
			// uchar to float
			linshi.convertTo(linshi, CV_32FC1);
		}
		else {
			linshi = src.clone();
		}
		int img_cols = linshi.cols;
		int img_rows = linshi.rows;


		float * gpu_img = NULL;
		float * gpu_result = NULL;
		float * gpu_kernel = NULL;


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

	void cuda_conv(float * src, float * dst, int img_rows, int img_cols, float * kernel, int kernel_dim) 
	{
		// read linshi and convert to gray image
		cudaSetDevice(0);

		float * gpu_img = NULL;
		float * gpu_result = NULL;
		float * gpu_kernel = NULL;


		size_t img_size_t = img_cols * img_rows * sizeof(float);
		size_t kernel_size_t = kernel_dim * kernel_dim * sizeof(float);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size_t));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size_t));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_kernel, kernel_size_t));
		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, (float *)src, img_size_t, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_kernel, kernel, kernel_size_t, cudaMemcpyHostToDevice));

		// //////////////////////////////////////////////////////////////////////////////////////////////
		// resident thread; every pixel of result correspond to a thread;
		// //////////////////////////////////////////////////////////////////////////////////////////////

		int thread_num = getThreadNum();
		int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
		dim3 grid_size(block_num, 1, 1);
		dim3 block_size(thread_num, 1, 1);
		conv::conv_kernel << < grid_size, block_size >> > (gpu_img, gpu_kernel, gpu_result, img_cols, img_rows, kernel_dim);

		HANDLE_ERROR(cudaMemcpy(dst, gpu_result, img_size_t, cudaMemcpyDeviceToHost));

		// release gpu memory
		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_kernel));
		HANDLE_ERROR(cudaFree(gpu_result));
		cudaDeviceReset();
	};


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
}