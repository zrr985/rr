// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _RKNN_DEMO_COMMON_H_  
#define _RKNN_DEMO_COMMON_H_  
#include <stdint.h>    // 标准整数类型定义，提供int8_t、int16_t、int32_t、int64_t等
#include <stdio.h>     // 标准输入输出库，提供printf、scanf等函数
#include <stdlib.h>    // 标准库，提供malloc、free、exit等函数
#include <string.h>    // 字符串处理库，提供strcpy、strlen、memcpy等函数
#include <sys/time.h>  // 系统时间库，提供gettimeofday、timeval等时间相关函数
#include "rknn_api.h"  // RKNN API头文件，提供RKNN推理引擎的接口定义

// 图像格式定义 - 定义常用的图像像素格式
typedef enum {
    IMAGE_FORMAT_RGB888 = 0,    // RGB格式，每个像素3字节（红、绿、蓝）
    IMAGE_FORMAT_BGR888,        // BGR格式，每个像素3字节（蓝、绿、红），OpenCV默认格式
    IMAGE_FORMAT_RGBA8888,      // RGBA格式，每个像素4字节（红、绿、蓝、透明度）
    IMAGE_FORMAT_BGRA8888       // BGRA格式，每个像素4字节（蓝、绿、红、透明度）
} image_format_t;

// 图像缓冲区结构 - 用于描述图像数据的内存布局和属性
typedef struct {
    int width;                    // 图像宽度（像素）
    int height;                   // 图像高度（像素）
    int channel;                  // 图像通道数（如RGB=3，RGBA=4）
    image_format_t format;        // 图像格式（RGB、BGR等）
    void *virt_addr;              // 虚拟内存地址，指向图像数据
    int fd;                       // 文件描述符，用于DMA内存映射
    int size;                     // 图像数据总大小（字节）
} image_buffer_t;

// 图像矩形结构 - 用于描述图像中的矩形区域（如检测框）
typedef struct {
    int left;                     // 矩形左边界坐标
    int top;                      // 矩形上边界坐标
    int right;                    // 矩形右边界坐标
    int bottom;                   // 矩形下边界坐标
} image_rect_t;

// Letterbox结构 - 用于图像预处理中的letterbox变换
typedef struct {
    int x_pad;                    // X方向填充像素数（左右填充）
    int y_pad;                    // Y方向填充像素数（上下填充）
    float scale;                  // 缩放比例，用于将图像缩放到目标尺寸
} letterbox_t;

// 颜色定义 - 预定义的常用颜色值（RGB格式，24位）
#define COLOR_RED     0xFF0000    // 红色：RGB(255,0,0)
#define COLOR_GREEN   0x00FF00    // 绿色：RGB(0,255,0)
#define COLOR_BLUE    0x0000FF    // 蓝色：RGB(0,0,255)
#define COLOR_YELLOW  0xFFFF00    // 黄色：RGB(255,255,0)
#define COLOR_WHITE   0xFFFFFF    // 白色：RGB(255,255,255)
#define COLOR_BLACK   0x000000    // 黑色：RGB(0,0,0)

// 注意：RKNN API已经定义了这些函数，不需要重复定义
// 说明：此文件只定义数据结构，具体的RKNN推理函数在rknn_api.h中定义

#endif 
