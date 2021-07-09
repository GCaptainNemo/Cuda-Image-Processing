#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "../include/harris.h"
#include "../include/gau_pyr.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

void rgb2gray_01(cv::Mat &src, cv::Mat & dst, bool is_normalize) 
{
	// rgb uint8 img ---> gray float[0-1] img
	cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	dst.convertTo(dst, CV_32FC1);
	if (is_normalize)
	{
		dst /= 255.0;
	}
};


int main() 
{
	//const char * address = "../data/img1.png";
	//harris::opencv_harris(address);

	// ///////////////////////////////////////
	//cv::Mat src = cv::imread(address);
	//cv::Mat dst;
	//float prop = 0.03;
	//harris::cuda_harris(src, dst, 4, prop, 3);

	//cv::Mat harris_bw_img;
	//cv::threshold(dst, harris_bw_img, 0.00001, 255, cv::THRESH_BINARY);
	//cv::namedWindow("bw", cv::WINDOW_NORMAL);
	//cv::imshow("bw", harris_bw_img);

	//cv::rgb2gray_01(dst, dst, 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
	//cv::namedWindow("harris-image", cv::WINDOW_NORMAL);
	//cv::imshow("harris-image", dst);
	//cv::waitKey(0);

	
	printf("kernel\n");
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	rgb2gray_01(src, src, true);
	cv::Mat dst;
	int size = 15;
	float sigma = 100;
	gau_pyr::cuda_pyramid_down(src, dst, size, sigma);
	cv::namedWindow("sobel-image", cv::WINDOW_NORMAL);
	cv::imshow("sobel-image", dst);
	cv::waitKey();
	/*
	conv::opencv_conv(address);*/
	return 0;
}


