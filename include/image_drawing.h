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

#ifndef _RKNN_DEMO_IMAGE_DRAWING_H_
#define _RKNN_DEMO_IMAGE_DRAWING_H_

#include "common.h"

// 绘制矩形
int draw_rectangle(image_buffer_t *image, int x, int y, int width, int height, int color, int thickness);

// 绘制文本
int draw_text(image_buffer_t *image, const char *text, int x, int y, int color, int font_size);

// 绘制点
int draw_point(image_buffer_t *image, int x, int y, int color, int radius);

// 绘制线条
int draw_line(image_buffer_t *image, int x1, int y1, int x2, int y2, int color, int thickness);

#endif //_RKNN_DEMO_IMAGE_DRAWING_H_
