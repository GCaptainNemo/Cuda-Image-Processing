#pragma once
#include "../include/harris.h"
#include "opencv2/opencv.hpp"

// cornerHarris��������ÿһ�����أ�x,y)��blockSize x blockSize �����ڣ�
// ����2x2�ݶȵ�Э�������M(x,y)���Ϳ����ҳ����ͼ�еľֲ����ֵ�����ҳ��˽ǵ㡣
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
}

