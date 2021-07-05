#pragma once
#include "../include/harris.h"
#include "opencv2/opencv.hpp"
#include "../include/conv.h"
#include "../include/utils.h"

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

	//void cuda_harris(const char * address, const int & block_size, const float & threshold, const int &aperture_size = 3)
	//{
	//	cv::Mat src_img = cv::imread(address);
	//	cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);
	//	cv::Mat dst_img;
	//	src_img.convertTo(dst_img, CV_32FC1);
	//	// 3 x 3 Sobel operator
	//	float * sobel_y = new float[aperture_size * aperture_size];
	//	sobel_y[0] = -1.; sobel_y[1] = -2.; sobel_y[2] = -1.;
	//	sobel_y[3] =  0.; sobel_y[4] = 0.; sobel_y[5] = 0.;
	//	sobel_y[6] =  1.; sobel_y[7] = 2.; sobel_y[8] = 1.;
	//	float * sobel_x = new float[aperture_size * aperture_size];
	//	sobel_x[0] = -1.; sobel_x[1] = 0.; sobel_x[2] = 1.;
	//	sobel_x[3] = -2.; sobel_x[4] = 0.; sobel_x[5] = 2.;
	//	sobel_x[6] = -1.; sobel_x[7] = 0.; sobel_x[8] = 1.;
	//	
	//	int img_rows = dst_img.rows;
	//	int img_cols = dst_img.cols;

	//	int thread_num = getThreadNum();
	//	int block_num = (img_cols * img_rows - 0.5) / thread_num + 1;
	//	dim3 grid_size(block_num, 1, 1);
	//	dim3 block_size(thread_num, 1, 1);

	//	/*conv::conv_kernel << <grid_size, block_size >> > 
	//		(gpu_img, gpu_kernel, gpu_result, img_cols, img_rows, aperture_size);*/

	//	//conv::conv_kernel << < >> > ()
	//}

	/*__global__ void harris_kernal(float * gpu_img, float * kernel,
		const int img_row, const int img_col, ) 
	{
			
	}*/


}

