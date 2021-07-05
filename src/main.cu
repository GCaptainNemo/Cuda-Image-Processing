#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "../include/harris.h"
#include "opencv2/opencv.hpp"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));


int main() 
{
	/*const char * address = "../data/img1.png";
	harris::opencv_harris(address);*/

	int kernel_size = 3;
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
	cv::waitKey();
	/*
	conv::opencv_conv(address);*/
	return 0;
}


