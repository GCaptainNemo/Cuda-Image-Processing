#include "../include/morphology.h"
#include "../include/utils.h"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

namespace morphology 
{
	__global__ void imdilate_kernel(float *gpu_img, bool * gpu_structure, float * gpu_result,
		const int img_cols, const int img_rows, const int structure_col, const int structure_row,
		const int anchor_col, const int anchor_row) 
	{
		const int img_col_id = blockIdx.x * blockDim.x + threadIdx.x;
		const int img_row_id = blockIdx.y * blockDim.y + threadIdx.y;
		const int pixel_id = img_col_id + img_row_id * img_cols;
		if (img_col_id >= img_cols || img_row_id >= img_rows) 
		{
			return;
		}
		float max_val = -1.0f;
		for (int delta_row = 0; delta_row < structure_row; ++delta_row) 
		{
			for (int delta_col = 0; delta_col < structure_col; ++delta_col)
			{
				const int cur_img_col_id = img_col_id + delta_col - anchor_col;
				const int cur_img_row_id = img_row_id + delta_row - anchor_row;
				if (cur_img_col_id >= 0 && cur_img_col_id < img_cols 
					&& cur_img_row_id >= 0 && cur_img_row_id < img_rows && 
					gpu_structure[delta_row * structure_col + delta_col])
				{
					if (gpu_img[cur_img_row_id *img_cols + cur_img_col_id] > max_val) 
					{
						max_val = gpu_img[cur_img_row_id *img_cols + cur_img_col_id];
					}
				}
			}
		}
		gpu_result[pixel_id] = max_val;
		if (pixel_id == 0) 
		{
			printf("max_val = %f", max_val);
		}
	};

	__global__ void imerode_kernel(float *gpu_img, bool * gpu_structure, float * gpu_result,
		const int img_cols, const int img_rows, const int structure_col, const int structure_row,
		const int anchor_col, const int anchor_row)
	{
		const int img_col_id = blockIdx.x * blockDim.x + threadIdx.x;
		const int img_row_id = blockIdx.y * blockDim.y + threadIdx.y;
		if (img_col_id >= img_cols || img_row_id >= img_rows)
		{
			return;
		}
		float min_val = 100.0f;
		for (int delta_row = 0; delta_row < structure_row; ++delta_row)
		{
			for (int delta_col = 0; delta_col < structure_col; ++delta_col)
			{
				const int cur_img_col_id = img_col_id + delta_col - anchor_col;
				const int cur_img_row_id = img_row_id + delta_row - anchor_row;
				if (cur_img_col_id >= 0 && cur_img_col_id < img_cols
					&& cur_img_row_id >= 0 && cur_img_row_id < img_rows &&
					gpu_structure[delta_row * structure_col + delta_col])
				{
					if (gpu_img[cur_img_row_id *img_cols + cur_img_col_id] < min_val)
					{
						min_val = gpu_img[cur_img_row_id *img_cols + cur_img_col_id];
					}
				}
			}
		}
		gpu_result[img_row_id * img_cols + img_col_id] = min_val;


	};

	// 
	void launch_imerode(cv::Mat & src, cv::Mat & dst, bool * structure, const int structure_col, const int structure_row,
		const int anchor_col_id, const int anchor_row_id) 
	{
		
		const int img_rows = src.rows;
		const int img_cols = src.cols;
		printf("img_rows = %d, img_cols = %d", img_rows, img_cols);
		float * gpu_img = NULL;
		float * gpu_result = NULL;
		bool * gpu_structure = NULL;
		size_t img_size = img_cols * img_rows * sizeof(float);
		size_t structure_size = structure_col * structure_row * sizeof(bool);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_structure, structure_size));
		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, src.data, img_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_structure, structure, structure_size, cudaMemcpyHostToDevice));

		
		const int grid_size_x = (img_cols - 1) / 32 + 1;
		const int grid_size_y = (img_rows - 1) / 32 + 1;
		printf("grid_size_x = %d, grid_size_y = %d", grid_size_x, grid_size_y);
		dim3 grid_size(grid_size_x, grid_size_y);
		dim3 block_size(32, 32);
		morphology::imerode_kernel << <grid_size, block_size >> > (gpu_img, gpu_structure, gpu_result,
			img_cols, img_rows, structure_col, structure_row,
			anchor_col_id, anchor_row_id);
		float * cpu_result = new float[img_cols * img_rows];
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_result, img_size, cudaMemcpyDeviceToHost));

		dst = cv::Mat(img_rows, img_cols, CV_32FC1, cpu_result).clone();
		printf("row = 0, col=0, val = %f", dst.at<float>(0, 0));


		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_structure));
		HANDLE_ERROR(cudaFree(gpu_result));
		delete[] cpu_result;
		cudaDeviceReset();
	};

	void launch_imdilate(cv::Mat & src, cv::Mat & dst, bool * structure, const int structure_col, const int structure_row,
		const int anchor_col_id, const int anchor_row_id) 
	{
		const int img_rows = src.rows;
		const int img_cols = src.cols;
		printf("img_rows = %d, img_cols = %d", img_rows, img_cols);

		float * gpu_img = NULL;
		float * gpu_result = NULL;
		bool * gpu_structure = NULL;
		size_t img_size = img_cols * img_rows * sizeof(float);
		size_t structure_size = structure_col * structure_row * sizeof(bool);

		HANDLE_ERROR(cudaMalloc((void **)& gpu_img, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_result, img_size));
		HANDLE_ERROR(cudaMalloc((void **)& gpu_structure, structure_size));
		// memory copy kernel and linshi from host to device
		HANDLE_ERROR(cudaMemcpy(gpu_img, src.data, img_size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(gpu_structure, structure, structure_size, cudaMemcpyHostToDevice));

		
		const int grid_size_x = (img_cols - 1) / 32 + 1;
		const int grid_size_y = (img_rows - 1) / 32 + 1;
		printf("grid_size_x = %d, grid_size_y = %d", grid_size_x, grid_size_y);

		dim3 grid_size(grid_size_x, grid_size_y);
		dim3 block_size(32, 32);
		morphology::imdilate_kernel << <grid_size, block_size >> > (gpu_img, gpu_structure, gpu_result,
			img_cols, img_rows, structure_col, structure_row,
			anchor_col_id, anchor_row_id);


		float * cpu_result = new float[img_cols * img_rows];
		HANDLE_ERROR(cudaMemcpy(cpu_result, gpu_result, img_size, cudaMemcpyDeviceToHost));

		dst = cv::Mat(img_rows, img_cols, CV_32FC1, cpu_result).clone();


		HANDLE_ERROR(cudaFree(gpu_img));
		HANDLE_ERROR(cudaFree(gpu_structure));
		HANDLE_ERROR(cudaFree(gpu_result));
		delete[] cpu_result;
		cudaDeviceReset();
	};



}