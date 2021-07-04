#include <stdio.h>
#include "../include/conv.h"
#include "../include/utils.h"
#include "opencv2/opencv.hpp"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));


int main() 
{
	const char * address = "../data/img1.png";
	int kernel_size = 3;
	float * kernel = new float[kernel_size * kernel_size];
	for (int i = 0; i < kernel_size * kernel_size; ++i)
	{
		kernel[i] = i % kernel_size - 1;
	}
	printf("kernel\n");
	conv::cuda_conv(address, kernel, kernel_size);
	//opencv_conv(address);
	return 0;
}


