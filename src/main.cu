#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "../include/harris.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

void normalize(cv::Mat &src, cv::Mat & dst) 
{
	// rgb uint8 img ---> gray float[0-1] img
	cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	dst.convertTo(dst, CV_32FC1);
	dst /= 255.0;
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

	//cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
	//cv::namedWindow("harris-image", cv::WINDOW_NORMAL);
	//cv::imshow("harris-image", dst);
	//cv::waitKey(0);

	
	int kernel_size = 5;
	float * kernel = new float[kernel_size * kernel_size];
	float sigma = 30;
	conv::get_gaussian_blur_kernel(sigma, kernel_size, kernel);
	for (int i = 0; i < kernel_size ; ++i)
	{
		for (int j = 0; j < kernel_size; ++j) 
		{
			printf("%f ", kernel[i * kernel_size + j]) ;
		}
		printf("\n");
	}
	/*for (int i = 0; i < kernel_size * kernel_size; ++i)
	{
		kernel[i] = i % kernel_size - 1;
	}*/
	printf("kernel\n");
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::Mat dst;
	conv::cuda_conv(src, dst, kernel, kernel_size);
	
	printf("src.type = %d\n", src.type());
	printf("dst.type = %d\n", dst.type());
	printf("src.shape = %d\n", dst.size[0]);

	cv::Mat result;
	std::vector<cv::Mat> mat_vec;
	normalize(src, src);
	mat_vec.push_back(src);
	mat_vec.push_back(dst);
	
	cv::hconcat(mat_vec, result);
	cv::namedWindow("sobel-image", cv::WINDOW_NORMAL);
	cv::imshow("sobel-image", result);
	cv::waitKey();
	/*
	conv::opencv_conv(address);*/
	return 0;
}


