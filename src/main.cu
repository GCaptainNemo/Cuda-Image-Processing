#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "../include/harris.h"
#include "../include/gau_pyr.h"
#include "../include/sift.h"
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
}; // rgb2gray_01

void test_cuda_conv(const char *option) 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	rgb2gray_01(src, src, true);
	float sigma = 100;
	int kernel_size = 20;
	float * gaussian_kernel;
	gau_pyr::get_gaussian_blur_kernel(sigma, kernel_size, &gaussian_kernel);
	cv::Mat dst;
	if (strcmp(option, "mat") == 0)
	{
		
		conv::cuda_conv(src, dst, gaussian_kernel, kernel_size);
	}
	else {
		float * res = new float[src.rows * src.cols];
		conv::cuda_conv((float *)src.data, res, src.rows, src.cols, gaussian_kernel, kernel_size);
		dst = cv::Mat(src.rows, src.cols, CV_32FC1, res).clone();
		delete[] res;
	}
	cv::namedWindow("dst", cv::WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	
}; // test_cuda_conv()

void test_pyr_down() 
{
	int size = -1;
	float sigma = 5;
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::Mat dst;
	gau_pyr::cuda_pyramid_down(src, dst, size, sigma);
	cv::namedWindow("sobel-image", cv::WINDOW_NORMAL);
	cv::imshow("sobel-image", dst);
	cv::waitKey();
	dst *= 255;
	dst.convertTo(dst, CV_8UC1);
	cv::imwrite("dst.png", dst); 
}// test_pyr_down

void test_harris(const char * option) 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::Mat dst;
	float prop = 0.03;
	cv::Mat harris_bw_img;
	if (strcmp(option, "mat") == 0) {
		harris::cuda_harris(src, dst, 4, prop, 3);
	}
	else {
		int rows = src.rows;
		int cols = src.cols;
		float * res = new float[rows * cols];
		harris::cuda_harris((float *)src.data, res, rows, cols, 4, prop, 3);
		harris_bw_img = cv::Mat(rows, cols, CV_32FC1, res);
	}
	cv::threshold(dst, harris_bw_img, 0.00001, 255, cv::THRESH_BINARY);
	cv::namedWindow("bw", cv::WINDOW_NORMAL);
	cv::imshow("bw", harris_bw_img);

	cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
	cv::namedWindow("harris-image", cv::WINDOW_NORMAL);
	cv::imshow("harris-image", dst);
	cv::waitKey(0);

}// test_harris

void test_build_gau_py() 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);

	printf("src.type = %d\n", src.type());
	rgb2gray_01(src, src, true);
	printf("src.type = %d\n", src.type());

	//float *** gaussian_pyramid = NULL;
	int octave_num = 3;
	int interval_num = 3;
	//std::vector<std::vector<cv::Mat *>> gaussian_pyramid(octave_num, std::vector<cv::Mat *>(interval_num + 3));
	cv::Mat *** gaussian_pyramid = NULL; 
	gau_pyr::build_gauss_pry(src, &gaussian_pyramid, octave_num, interval_num, 1.6);
	/*for (int o = 0; o < octave_num; ++o) 
	{
		for (int i = 0; i < interval_num + 3; ++i) 
		{
			cv::Mat output(*(gaussian_pyramid[o][i]));
			output *= 255;
			output.convertTo(output, CV_8UC1);
			std::string file = std::to_string(o) + "-" + std::to_string(i) + ".png";
			cv::imwrite(file, output);
		}
	}*/
	cv::Mat *** dog = NULL;
	gau_pyr::build_dog_pyr(gaussian_pyramid, &dog, octave_num, interval_num);
	/*for (int o = 0; o < octave_num; ++o)
	{
		for (int i = 0; i < interval_num + 2; ++i)
		{
			cv::Mat output(*(dog[o][i]));
			output *= 255;
			output.convertTo(output, CV_8UC1);
			std::string file = std::to_string(o) + "-" + std::to_string(i) + "dog" + ".png";
			cv::imwrite(file, output);
		}
	}*/
	int *** musk = NULL;
	sift::detect_extreme_point(&dog, &musk, octave_num, interval_num);
}

int main() 
{
	//const char * address = "../data/img1.png";
	//harris::opencv_harris(address);

	// ///////////////////////////////////////
	
	
	printf("kernel\n");
	//test_cuda_conv("mat");
	test_harris("float");
	//test_build_gau_py();

	// conv::opencv_conv(address);*/
	return 0;
}


