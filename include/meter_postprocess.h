#ifndef _METER_POSTPROCESS_H_
#define _METER_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov8.h"

// 仪表检测配置
#define METER_INPUT_SIZE 640
#define METER_OBJ_THRESH 0.25
#define METER_NMS_THRESH 0.45
#define METER_MAX_DETECTIONS 128

// 仪表类别定义
#define METER_CLASS_BACKGROUND 0
#define METER_CLASS_POINTER 1
#define METER_CLASS_SCALE 2
#define METER_CLASS_NUM 3

// 仪表检测结果结构体
typedef struct {
    image_rect_t box;
    float confidence;
    int class_id;
    float* mask;  // 分割掩码
    int mask_width;
    int mask_height;
} meter_detection_t;

typedef struct {
    int count;
    meter_detection_t detections[METER_MAX_DETECTIONS];
} meter_result_list_t;

// 仪表读数结果结构体
typedef struct {
    image_buffer_t image;
    int id;
    int box_x;
    int box_y;
    int box_w;
    int box_h;
    float score;
    char name[64];
    int item_id;
    int cls_id;
    int mask_offset;
    float reading_value;  // 仪表读数值
    char meter_type[32];  // 仪表类型
} meter_reading_result;

typedef struct {
    int count;
    meter_reading_result results[METER_MAX_DETECTIONS];
} meter_reading_result_list;

// 仪表后处理函数
int init_meter_post_process();
void deinit_meter_post_process();
int meter_post_process(rknn_app_context_t *app_ctx, void *outputs,
                      letterbox_t *letter_box, float conf_threshold,
                      float nms_threshold, meter_result_list_t *results);

// 仪表读数计算函数
int init_meter_reader();
void deinit_meter_reader();
int meter_reading_process(meter_result_list_t *detections,
                         meter_reading_result_list *results);

// 工具函数
float calculate_iou_meter(float x1, float y1, float x2, float y2,
                         float x3, float y3, float x4, float y4);
int non_max_suppression_meter(meter_detection_t *detections, int count, float nms_threshold);
const char* meter_cls_to_name(int cls_id);

#endif //_METER_POSTPROCESS_H_
