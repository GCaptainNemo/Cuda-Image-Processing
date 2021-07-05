#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "../include/harris.h"
#include "opencv2/opencv.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));


int main() 
{
	const char * address = "../data/img1.png";
	harris::opencv_harris(address);

	// ///////////////////////////////////////
	cv::Mat src = cv::imread(address);
	cv::Mat dst;
	float prop = 0.03;
	harris::cuda_harris(src, dst, 4, prop, 3);

	cv::Mat harris_bw_img;
	cv::threshold(dst, harris_bw_img, 0.00001, 255, cv::THRESH_BINARY);
	cv::namedWindow("bw", cv::WINDOW_NORMAL);
	cv::imshow("bw", harris_bw_img);

	cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
	cv::namedWindow("harris-image", cv::WINDOW_NORMAL);
	cv::imshow("harris-image", dst);
	cv::waitKey(0);

	
	/*int kernel_size = 3;
	float * kernel = new float[kernel_size * kernel_size];
	for (int i = 0; i < kernel_size * kernel_size; ++i)
	{
		kernel[i] = i % kernel_size - 1;
	}
	printf("kernel\n");
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::Mat dst;
	conv::cuda_conv(src, dst, kernel, kernel_size);
	cv::namedWindow("sobel-image", cv::WINDOW_NORMAL);
	cv::imshow("sobel-image", dst);
	cv::waitKey();*/
	/*
	conv::opencv_conv(address);*/
	return 0;
}


