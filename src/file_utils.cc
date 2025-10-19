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

#include "file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int read_data_from_file(const char *filename, char **data)
{
    if (filename == NULL || data == NULL) {
        printf("Invalid parameters\n");
        return -1;
    }

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0) {
        printf("Invalid file size: %ld\n", file_size);
        fclose(file);
        return -1;
    }

    // 分配内存
    *data = (char *)malloc(file_size);
    if (*data == NULL) {
        printf("Failed to allocate memory\n");
        fclose(file);
        return -1;
    }

    // 读取文件内容
    size_t bytes_read = fread(*data, 1, file_size, file);
    fclose(file);

    if (bytes_read != (size_t)file_size) {
        printf("Failed to read complete file\n");
        free(*data);
        *data = NULL;
        return -1;
    }

    return file_size;
}

int write_data_to_file(const char *filename, const char *data, int size)
{
    if (filename == NULL || data == NULL || size <= 0) {
        printf("Invalid parameters\n");
        return -1;
    }

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Failed to open file for writing: %s\n", filename);
        return -1;
    }

    size_t bytes_written = fwrite(data, 1, size, file);
    fclose(file);

    if (bytes_written != (size_t)size) {
        printf("Failed to write complete data\n");
        return -1;
    }

    return 0;
}

int file_exists(const char *filename)
{
    if (filename == NULL) {
        return 0;
    }

    FILE *file = fopen(filename, "r");
    if (file != NULL) {
        fclose(file);
        return 1;
    }
    return 0;
}

long get_file_size(const char *filename)
{
    if (filename == NULL) {
        return -1;
    }

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);

    return size;
}
