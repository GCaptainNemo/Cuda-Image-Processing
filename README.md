# CUDA-Harris

## 一、介绍

本仓库用CUDA编程实现图像的卷积和Harris角点提取操作，相比于串行的方式，使用GPU调用多线程完成速度要快得多。(目前提取效果与OpenCV自带的Harris角点API有一定差异)

## 二、安装方法 

```
cd cuda-conv
mkdir build
cd build
cmake ..
```

main.cu中给了一个Prewitt算子和提取Harris角点的demo。

