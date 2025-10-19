// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _RKNN_DEMO_IMAGE_UTILS_H_
#define _RKNN_DEMO_IMAGE_UTILS_H_

#include "common.h"  // 包含公共定义，使用image_buffer_t、letterbox_t等数据结构

// 图像读取函数 - 从文件路径读取图像数据到image_buffer_t结构
// 参数：image_path-图像文件路径，image-输出参数，存储图像数据
// 返回值：成功返回0，失败返回-1
// 功能：支持常见图像格式（JPEG、PNG、BMP等），自动分配内存
int read_image(const char *image_path, image_buffer_t *image);

// 图像写入函数 - 将image_buffer_t结构中的图像数据写入文件
// 参数：image_path-输出文件路径，image-包含图像数据的结构体
// 返回值：成功返回0，失败返回-1
// 功能：根据文件扩展名自动选择图像格式进行保存
int write_image(const char *image_path, image_buffer_t *image);

// 图像预处理函数 - 调整图像尺寸到指定大小
// 参数：src_image-源图像，dst_image-目标图像，dst_width-目标宽度，dst_height-目标高度
// 返回值：成功返回0，失败返回-1
// 功能：使用双线性插值进行图像缩放，保持图像质量
int resize_image(image_buffer_t *src_image, image_buffer_t *dst_image, int dst_width, int dst_height);

// 图像释放函数 - 释放image_buffer_t结构中的内存资源
// 参数：image-要释放的图像结构体
// 功能：释放virt_addr指向的内存，重置结构体字段
// 注意：调用后image结构体不再有效，需要重新初始化
void release_image(image_buffer_t *image);

// 获取图像大小 - 计算图像数据的总字节数
// 参数：image-图像结构体
// 返回值：图像数据大小（字节），失败返回-1
// 功能：根据width、height、channel计算总大小
int get_image_size(image_buffer_t *image);

// 图像格式转换和letterbox处理 - 将图像转换为模型输入格式
// 参数：src-源图像，dst-目标图像，letter_box-letterbox变换参数，bg_color-背景颜色
// 返回值：成功返回0，失败返回-1
// 功能：进行格式转换、尺寸调整、letterbox填充，为RKNN推理做准备
int convert_image_with_letterbox(image_buffer_t *src, image_buffer_t *dst, letterbox_t *letter_box, int bg_color);

#endif 
