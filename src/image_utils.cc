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

#include "image_utils.h"
#include "file_utils.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

int read_image(const char *image_path, image_buffer_t *image)
{
    if (image_path == NULL || image == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    // 使用OpenCV读取图像
    cv::Mat cv_image = cv::imread(image_path);
    if (cv_image.empty()) {
        printf("Failed to read image: %s\n", image_path);
        return -1;
    }

    // 转换为RGB格式
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);

    // 分配内存
    int size = cv_image.rows * cv_image.cols * cv_image.channels();
    unsigned char *data = (unsigned char *)malloc(size);
    if (data == NULL) {
        printf("Failed to allocate memory\n");
        return -1;
    }

    // 复制数据
    memcpy(data, cv_image.data, size);

    // 填充结构体
    image->width = cv_image.cols;
    image->height = cv_image.rows;
    image->channel = cv_image.channels();
    image->format = IMAGE_FORMAT_RGB888;
    image->virt_addr = data;
    image->fd = -1;
    image->size = size;

    return 0;
}

int write_image(const char *image_path, image_buffer_t *image)
{
    if (image_path == NULL || image == NULL || image->virt_addr == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    // 创建OpenCV Mat对象
    cv::Mat cv_image(image->height, image->width, CV_8UC3, image->virt_addr);
    
    // 转换回BGR格式用于保存
    cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);

    // 保存图像
    bool success = cv::imwrite(image_path, cv_image);
    if (!success) {
        printf("Failed to write image: %s\n", image_path);
        return -1;
    }

    return 0;
}

int resize_image(image_buffer_t *src_image, image_buffer_t *dst_image, int dst_width, int dst_height)
{
    if (src_image == NULL || dst_image == NULL || src_image->virt_addr == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    // 创建源图像Mat
    cv::Mat src_cv(src_image->height, src_image->width, CV_8UC3, src_image->virt_addr);
    
    // 创建目标图像Mat
    cv::Mat dst_cv;
    cv::resize(src_cv, dst_cv, cv::Size(dst_width, dst_height));

    // 分配目标图像内存
    int size = dst_cv.rows * dst_cv.cols * dst_cv.channels();
    unsigned char *data = (unsigned char *)malloc(size);
    if (data == NULL) {
        printf("Failed to allocate memory\n");
        return -1;
    }

    // 复制数据
    memcpy(data, dst_cv.data, size);

    // 填充目标结构体
    dst_image->width = dst_cv.cols;
    dst_image->height = dst_cv.rows;
    dst_image->channel = dst_cv.channels();
    dst_image->virt_addr = data;
    dst_image->fd = -1;
    dst_image->size = size;

    return 0;
}

void release_image(image_buffer_t *image)
{
    if (image != NULL && image->virt_addr != NULL) {
        free(image->virt_addr);
        image->virt_addr = NULL;
        image->size = 0;
    }
}

int get_image_size(image_buffer_t *image)
{
    if (image == NULL) {
        return -1;
    }
    
    int channels = 3; // 默认RGB
    if (image->format == IMAGE_FORMAT_RGBA8888 || image->format == IMAGE_FORMAT_BGRA8888) {
        channels = 4;
    }
    
    return image->width * image->height * channels;
}

int convert_image_with_letterbox(image_buffer_t *src, image_buffer_t *dst, letterbox_t *letter_box, int bg_color)
{
    if (src == NULL || dst == NULL || letter_box == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    // 创建源图像Mat
    cv::Mat src_cv(src->height, src->width, CV_8UC3, src->virt_addr);
    
    // 计算缩放比例
    float scale = std::min((float)dst->width / src->width, (float)dst->height / src->height);
    letter_box->scale = scale;
    
    // 计算目标尺寸
    int new_width = (int)(src->width * scale);
    int new_height = (int)(src->height * scale);
    
    // 计算padding
    letter_box->x_pad = (dst->width - new_width) / 2;
    letter_box->y_pad = (dst->height - new_height) / 2;
    
    // 创建目标图像Mat
    cv::Mat dst_cv = cv::Mat::zeros(dst->height, dst->width, CV_8UC3);
    
    // 设置背景色
    int b = (bg_color >> 16) & 0xFF;
    int g = (bg_color >> 8) & 0xFF;
    int r = bg_color & 0xFF;
    dst_cv.setTo(cv::Scalar(b, g, r));
    
    // 缩放源图像
    cv::Mat resized;
    cv::resize(src_cv, resized, cv::Size(new_width, new_height));
    
    // 将缩放后的图像复制到目标图像中心
    cv::Rect roi(letter_box->x_pad, letter_box->y_pad, new_width, new_height);
    resized.copyTo(dst_cv(roi));
    
    // 复制数据到目标缓冲区
    memcpy(dst->virt_addr, dst_cv.data, dst->size);
    
    return 0;
}
