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


#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

#include "rknn_api.h"
#include "common.h"

#if defined(RV1106_1103)  // 针对RV1106/1103芯片的特殊定义
    typedef struct {
        char *dma_buf_virt_addr;  // DMA缓冲区的虚拟内存地址
        int dma_buf_fd;           // DMA缓冲区的文件描述符
        int size;                 // DMA缓冲区的大小（字节）
    }rknn_dma_buf;  // RKNN DMA缓冲区结构体，用于零拷贝内存管理
#endif

typedef struct {
    rknn_context rknn_ctx;              // RKNN上下文句柄，用于与RKNN运行时通信
    rknn_input_output_num io_num;       // 输入输出张量数量信息
    rknn_tensor_attr* input_attrs;      // 输入张量属性数组指针
    rknn_tensor_attr* output_attrs;     // 输出张量属性数组指针
#if defined(RV1106_1103)  // RV1106/1103芯片专用内存管理
    rknn_tensor_mem* input_mems[1];     // 输入张量内存数组（1个输入）
    rknn_tensor_mem* output_mems[9];    // 输出张量内存数组（9个输出）
    rknn_dma_buf img_dma_buf;           // 图像DMA缓冲区，用于零拷贝传输
#endif
#if defined(ZERO_COPY)  // 零拷贝模式专用定义
    rknn_tensor_mem* input_mems[1];     // 输入张量内存数组（1个输入）
    rknn_tensor_mem* output_mems[9];    // 输出张量内存数组（9个输出）
    rknn_tensor_attr* input_native_attrs;   // 输入张量原生属性（零拷贝模式）
    rknn_tensor_attr* output_native_attrs;  // 输出张量原生属性（零拷贝模式）
#endif
    int model_channel;    // 模型输入通道数（如RGB=3，RGBA=4）
    int model_width;      // 模型输入宽度（像素）
    int model_height;     // 模型输入高度（像素）
    bool is_quant;        // 是否为量化模型（true=量化，false=浮点）
} rknn_app_context_t;  // RKNN应用上下文结构体，包含模型运行所需的所有信息

// 初始化YOLOv8模型
// 参数：model_path-模型文件路径，app_ctx-应用上下文结构体指针
// 返回值：成功返回0，失败返回-1
// 功能：加载RKNN模型文件，初始化RKNN上下文，配置输入输出张量属性
int init_yolov8_model(const char* model_path, rknn_app_context_t* app_ctx);

// 释放YOLOv8模型资源
// 参数：app_ctx-应用上下文结构体指针
// 返回值：成功返回0，失败返回-1
// 功能：释放RKNN上下文，清理内存，释放所有相关资源
int release_yolov8_model(rknn_app_context_t* app_ctx);

// 执行YOLOv8模型推理
// 参数：app_ctx-应用上下文结构体指针，img-输入图像缓冲区，od_results-输出检测结果
// 返回值：成功返回0，失败返回-1
// 功能：将图像数据输入模型，执行推理，获取目标检测结果
int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* img, void* od_results);

#endif //_RKNN_DEMO_YOLOV8_H_