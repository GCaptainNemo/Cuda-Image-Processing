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

void test_down_sample() 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	src.convertTo(src, CV_32FC1);

	float * res = NULL;
	gau_pyr::cuda_down_sampling((float *)src.data, &res, src.rows, src.cols);
	cv::Mat dst((src.rows + 1) / 2, (src.cols + 1) / 2, CV_32FC1, res);
	cv::normalize(dst, dst, 0.0, 1.0, cv::NORM_MINMAX);
	cv::namedWindow("dst", cv::WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::waitKey(0);

}

void test_harris(const char * option) 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	src.convertTo(src, CV_32FC1);
	
	cv::Mat dst;
	float prop = 0.03;
	if (strcmp(option, "mat") == 0) {
		harris::cuda_harris(src, dst, 4, prop, 3);
	}
	else {
		int rows = src.rows;
		int cols = src.cols;
		float * res = new float[rows * cols];
		harris::cuda_harris((float *)src.data, res, rows, cols, 4, prop, 3);
		dst = cv::Mat(rows, cols, CV_32FC1, res).clone();
	}
	cv::Mat harris_bw_img;
	cv::threshold(dst, harris_bw_img, 0.00001, 255, cv::THRESH_BINARY);
	cv::namedWindow("bw", cv::WINDOW_NORMAL);
	cv::imshow("bw", harris_bw_img);

	cv::normalize(dst, dst, 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
	cv::namedWindow("harris-image", cv::WINDOW_NORMAL);
	cv::imshow("harris-image", dst);
	cv::waitKey(0);

}// test_harris

void test_build_gau_py(const char *option) 
{
	const char * address = "../data/img1.png";
	cv::Mat src = cv::imread(address);
	rgb2gray_01(src, src, true);

	//float *** gaussian_pyramid = NULL;
	int octave_num = 3;
	int interval_num = 3;
	//std::vector<std::vector<cv::Mat *>> gaussian_pyramid(octave_num, std::vector<cv::Mat *>(interval_num + 3));
	if (strcmp(option, "mat") == 0) {
		cv::Mat *** gaussian_pyramid = NULL;
		gau_pyr::cuda_build_gauss_pyramid(src, &gaussian_pyramid, octave_num, interval_num, 1.6);
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
		sift::detect_extreme_point(dog, &musk, octave_num, interval_num);
	}
	else 
	{
		printf("in float!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		// ////////////////////////////////////////////////////
		// build gaussian pyramid
		// /////////////////////////////////////////////////////
		float *** gaussian_pyramid = NULL;
		gau_pyr::cuda_build_gauss_pyramid((float *)src.data, &gaussian_pyramid, src.rows, src.cols, octave_num, interval_num, 1.6);
		int ** row_col_lst = (int **)malloc(octave_num * sizeof(int*));
		int origin_row = src.rows * 2;
		int origin_col = src.cols * 2;
		for (int o = 0; o < octave_num; ++o)
		{
			origin_row = (origin_row + 1) / 2;
			origin_col = (origin_col + 1) / 2;
			row_col_lst[o] = (int *)malloc(2 * sizeof(int));
			row_col_lst[o][0] = origin_row;
			row_col_lst[o][1] = origin_col;
		}
		//for (int o = 0; o < octave_num; ++o)
		//{
		//	for (int i = 0; i < interval_num + 3; ++i)
		//	{
		//		cv::Mat output = cv::Mat(row_col_lst[o][0], row_col_lst[o][1], CV_32FC1, gaussian_pyramid[o][i]).clone();
		//		output *= 255;
		//		output.convertTo(output, CV_8UC1);
		//		std::string file = std::to_string(o) + "-" + std::to_string(i) + ".png";
		//		cv::imwrite(file, output);
		//	}
		//}

		// ////////////////////////////////////////////////////
		// build DoG pyramid
		// /////////////////////////////////////////////////////
		printf("build Dog Pyramid!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		float *** dog_pyramid;
		gau_pyr::build_dog_pyr(gaussian_pyramid, &dog_pyramid, row_col_lst, octave_num, interval_num);
		//for (int o = 0; o < octave_num; ++o)
		//{
		//	for (int i = 0; i < interval_num + 2; ++i)
		//	{
		//		cv::Mat output(row_col_lst[o][0], row_col_lst[o][1], CV_32FC1,  dog_pyramid[o][i]);
		//		cv::normalize(output, output, 0.0, 1.0, cv::NORM_MINMAX);
		//		cv::namedWindow("output", cv::WINDOW_NORMAL);
		//		cv::imshow("output", output);
		//		cv::waitKey(0);
		//		//output *= 255;
		//		//output.convertTo(output, CV_8UC1);
		//		//std::string file = std::to_string(o) + "-" + std::to_string(i) + "dog" + ".png";
		//		//cv::imwrite(file, output);
		//	}
		//}


		// /////////////////////////////////////////////////////////
		// extract extreme points
		// /////////////////////////////////////////////////////////
		printf("extract extreme points!!!!!!!!!!!!!!!!!!!!!!\n");
		int *** mask = NULL;
		sift::detect_extreme_point(dog_pyramid, &mask, row_col_lst, octave_num, interval_num);
		printf("done!!!!!!!\n");
		for (int o = 0; o < octave_num; ++o)
		{
			for (int i = 0; i < interval_num; ++i) 
			{
				for (int index = 0; index < row_col_lst[o][0] * row_col_lst[o][1]; ++index) {
					if (mask[o][i][index] == 1) 
					{
						printf("row = %d, col = %d\n", index / row_col_lst[o][1], index % row_col_lst[o][1]);
						dog_pyramid[o][0][index] = 1.0;		
					}
				}
			}
			cv::Mat output(row_col_lst[o][0], row_col_lst[o][1], CV_32FC1, dog_pyramid[o][0]);
			//cv::normalize(output, output, 0.0, 1.0, cv::NORM_MINMAX);
			cv::namedWindow("output", cv::WINDOW_NORMAL);
			cv::imshow("output", output);
			cv::waitKey(0);

		}

		for (int o = 0; o < octave_num; ++o)
		{
			free(row_col_lst[o]);
		}
		free(row_col_lst);
	
	}
}

	// detect extreme point;
	

int main() 
{
	//const char * address = "../data/img1.png";
	//harris::opencv_harris(address);

	// ///////////////////////////////////////
	
	printf("kernel\n");
	//test_cuda_conv("mat");
	//test_down_sample();
	//test_harris("float");
	test_build_gau_py("float");

	// conv::opencv_conv(address);*/
	return 0;
}


