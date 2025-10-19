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

#include "image_drawing.h"
#include <opencv2/opencv.hpp>

int draw_rectangle(image_buffer_t *image, int x, int y, int width, int height, int color, int thickness)
{
    if (image == NULL || image->virt_addr == NULL) {
        printf("Invalid image buffer\n");
        return -1;
    }

    // 创建OpenCV Mat对象
    cv::Mat cv_image(image->height, image->width, CV_8UC3, image->virt_addr);
    
    // 转换颜色格式 (RGB -> BGR)
    int b = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int r = color & 0xFF;
    cv::Scalar cv_color(b, g, r);

    // 绘制矩形
    cv::Point pt1(x, y);
    cv::Point pt2(x + width, y + height);
    cv::rectangle(cv_image, pt1, pt2, cv_color, thickness);

    return 0;
}

int draw_text(image_buffer_t *image, const char *text, int x, int y, int color, int font_size)
{
    if (image == NULL || image->virt_addr == NULL || text == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    // 创建OpenCV Mat对象
    cv::Mat cv_image(image->height, image->width, CV_8UC3, image->virt_addr);
    
    // 转换颜色格式 (RGB -> BGR)
    int b = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int r = color & 0xFF;
    cv::Scalar cv_color(b, g, r);

    // 设置字体
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = font_size / 10.0;
    int thickness = 2;

    // 绘制文本
    cv::Point pt(x, y);
    cv::putText(cv_image, text, pt, font, scale, cv_color, thickness);

    return 0;
}

int draw_point(image_buffer_t *image, int x, int y, int color, int radius)
{
    if (image == NULL || image->virt_addr == NULL) {
        printf("Invalid image buffer\n");
        return -1;
    }

    // 创建OpenCV Mat对象
    cv::Mat cv_image(image->height, image->width, CV_8UC3, image->virt_addr);
    
    // 转换颜色格式 (RGB -> BGR)
    int b = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int r = color & 0xFF;
    cv::Scalar cv_color(b, g, r);

    // 绘制点
    cv::Point center(x, y);
    cv::circle(cv_image, center, radius, cv_color, -1);

    return 0;
}

int draw_line(image_buffer_t *image, int x1, int y1, int x2, int y2, int color, int thickness)
{
    if (image == NULL || image->virt_addr == NULL) {
        printf("Invalid image buffer\n");
        return -1;
    }

    // 创建OpenCV Mat对象
    cv::Mat cv_image(image->height, image->width, CV_8UC3, image->virt_addr);
    
    // 转换颜色格式 (RGB -> BGR)
    int b = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int r = color & 0xFF;
    cv::Scalar cv_color(b, g, r);

    // 绘制线条
    cv::Point pt1(x1, y1);
    cv::Point pt2(x2, y2);
    cv::line(cv_image, pt1, pt2, cv_color, thickness);

    return 0;
}
