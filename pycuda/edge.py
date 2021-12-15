import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer
import cv2

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
# mod = SourceModule()

# func = mod.get_function("func")


def edge_kernel():
    kernel = """
    __global__ void edge_func(int *gpu_img, int *edge_img)
    {
      const int img_col_id = blockIdx.x * blockDim.x + threadIdx.x;
      const int img_row_id = blockIdx.y * blockDim.y + threadIdx.y;
      if (img_row_id >= %(ROW)s || img_col_id >= %(COL)s)
      {
        return;
      }
      int max_val = 0;
      int center_val = gpu_img[img_row_id * %(COL)s + img_col_id];
      for(int delta_row = -1; delta_row < 2; delta_row ++)
      {
        for(int delta_col = -1; delta_col < 2; delta_col ++)
        {
            int cur_col_id = img_col_id + delta_col;
            int cur_row_id = img_row_id + delta_row;
            if (cur_col_id >= 0 && cur_row_id >= 0 && 
                cur_col_id < %(COL)s && cur_row_id < %(ROW)s)
            {
                float diff = gpu_img[cur_row_id * %(COL)s + cur_col_id] - center_val;
                if (diff < 0){diff = -diff;}
                if (diff > max_val){max_val = diff;} 
            }        
        }
      }
      edge_img[img_row_id * %(COL)s + img_col_id] = max_val;
      
      // float temp_a = a[i];
      // float temp_b = b[i];
      // a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
      // a[i] = a[i] + b[i];
    }
    """
    return kernel


def idt_kernel():
    kernel = """
    __global__ void idt_func(int *gpu_img, int *edge_img)
    {
      const int img_row_id = blockIdx.x * blockDim.x + threadIdx.x;
      const int img_col_id = blockIdx.y * blockDim.y + threadIdx.y;
      if (img_row_id >= %(ROW)s || img_col_id >= %(COL)s)
      {
        return;
      }
      int max_val = 0;
      int center_val = gpu_img[img_row_id * %(COL)s + img_col_id];
      for(int delta_row = -1; delta_row < 2; delta_row ++)
      {
        for(int delta_col = -1; delta_col < 2; delta_col ++)
        {
            int cur_col_id = img_col_id + delta_col;
            int cur_row_id = img_row_id + delta_row;
            if (cur_col_id >= 0 && cur_row_id >= 0 && 
                cur_col_id < %(COL)s && cur_row_id < %(ROW)s)
            {
                float diff = gpu_img[cur_row_id * %(COL)s + cur_col_id] - center_val;
                if (diff < 0){diff = -diff;}
                if (diff > max_val){max_val = diff;} 
            }        
        }
      }
      edge_img[img_row_id * %(COL)s + img_col_id] = max_val;

      // float temp_a = a[i];
      // float temp_b = b[i];
      // a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
      // a[i] = a[i] + b[i];
    }
    """
    return kernel


def test(dir):
    dtype = np.int32
    # N = 1024 * 1024 * 90   # float: 4M = 1024 * 1024
    img = cv2.imread(dir, -1)
    print("max = ", np.max(img))
    edge_kernel_code = edge_kernel()
    template_code = edge_kernel_code % {
        'COL': img.shape[1],
        'ROW': img.shape[0]
    }
    module = SourceModule(template_code)
    func = module.get_function("edge_func")

    img.astype(dtype)
    result_img = np.zeros(img.shape, dtype)
    img_gpu = gpuarray.to_gpu(img.astype(dtype))
    result_img_gpu = gpuarray.to_gpu(result_img.astype(dtype))
    block_dim = (32, 32, 1)
    grid_row_num = int((img.shape[0] - 1) / 32 + 1)
    grid_col_num = int((img.shape[1] - 1) / 32 + 1)
    gdim = (grid_col_num, grid_row_num, 1)
    start = timer()
    func(img_gpu, result_img_gpu, block=block_dim, grid=gdim)
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)
    result_img = result_img_gpu.get().astype(np.uint8)
    print(result_img.shape)
    print(result_img.dtype)
    print(np.max(result_img))
    res = np.concatenate([result_img, img], axis=0)
    cv2.namedWindow("edge img", cv2.WINDOW_NORMAL)
    cv2.imshow("edge img", res)
    # cv2.imshow("src img", img)

    cv2.waitKey(0)



def main():
    addr = "E:/dataset/registration_dataset/rgb_ir/rgb_2.bmp"
    test(addr)


if __name__ == '__main__':
    main()