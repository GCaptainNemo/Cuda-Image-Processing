#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace conv 
{
	__global__ void conv(float *gpu_img, float * gpu_kernel, float * gpu_result,
		const int img_cols, const int img_rows, const int kernel_dim);

	void cuda_conv(const char * address, float * kernel, int kernel_dim);

	void opencv_conv(const char * address);
}