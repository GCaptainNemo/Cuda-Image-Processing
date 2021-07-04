# CUDA-CONV

## 一、介绍

本仓库用CUDA编程实现图像的2D卷积操作，相比于串行的方式，使用GPU调用多线程完成卷积速度要快得多。

## 二、安装方法 

```
cd cuda-conv
mkdir build
cd build
cmake ..
```

main.cu中给了一个Prewitt算子的demo，使用其它卷积核的方法类似。

