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

#ifndef _RKNN_DEMO_FILE_UTILS_H_  
#define _RKNN_DEMO_FILE_UTILS_H_  

#include <stdint.h>  // 标准整数类型定义，提供int8_t、int16_t、int32_t、int64_t等

// 从文件读取数据 - 将整个文件内容读取到内存中
// 参数：filename-文件路径，data-输出参数，指向分配的内存缓冲区
// 返回值：成功返回读取的字节数，失败返回-1
// 注意：调用者需要负责释放*data指向的内存
int read_data_from_file(const char *filename, char **data);

// 写入数据到文件 - 将数据写入指定文件
// 参数：filename-文件路径，data-要写入的数据，size-数据大小（字节）
// 返回值：成功返回写入的字节数，失败返回-1
int write_data_to_file(const char *filename, const char *data, int size);

// 检查文件是否存在 - 判断指定路径的文件是否存在
// 参数：filename-文件路径
// 返回值：文件存在返回1，不存在返回0，错误返回-1
int file_exists(const char *filename);

// 获取文件大小 - 获取指定文件的大小（字节数）
// 参数：filename-文件路径
// 返回值：成功返回文件大小（字节），失败返回-1
long get_file_size(const char *filename);

#endif 
