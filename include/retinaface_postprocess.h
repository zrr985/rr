#ifndef _RETINAFACE_POSTPROCESS_H_
#define _RETINAFACE_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov8.h"

// RetinaFace专用配置
#define RETINAFACE_INPUT_SIZE 640
#define RETINAFACE_ANCHOR_SIZES {{16, 32}, {64, 128}, {256, 512}}
#define RETINAFACE_STEPS {8, 16, 32}
#define RETINAFACE_VARIANCE {0.1, 0.2}
#define RETINAFACE_NMS_THRESH 0.4
#define RETINAFACE_CONF_THRESH 0.5

// RetinaFace锚框结构体
typedef struct {
    float cx, cy, s_kx, s_ky;
} retinaface_anchor_t;

// RetinaFace检测结果结构体
typedef struct {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    float landmarks[10];   // 5个关键点坐标 (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)
    int class_id;          // 类别ID
} retinaface_detection_t;

// RetinaFace检测结果列表
typedef struct {
    int count;
    retinaface_detection_t detections[128];
} retinaface_result_list_t;

// RetinaFace人脸识别结果结构体（包含身份信息）
typedef struct {
    image_buffer_t image;
    int     id;
    int     box_x;
    int     box_y;
    int     box_w;
    int     box_h;
    float   score;
    char    name[64];
    int     item_id;
    int     cls_id;
    int     mask_offset;
    float   landmarks[10];  // 关键点坐标
    char    identity[64];   // 身份信息
} retinaface_face_result;

typedef struct {
    int             id;
    int             count;
    retinaface_face_result results[128];
} retinaface_face_result_list;

// RetinaFace后处理函数
int init_retinaface_post_process();
void deinit_retinaface_post_process();
int retinaface_post_process(rknn_app_context_t *app_ctx, void *outputs, 
                           letterbox_t *letter_box, float conf_threshold, 
                           float nms_threshold, retinaface_result_list_t *results);

// 人脸识别函数
int init_face_recognition();
void deinit_face_recognition();
int face_recognition_process(retinaface_result_list_t *detections, 
                            retinaface_face_result_list *results);

// 工具函数
void generate_retinaface_anchors(int input_size, std::vector<retinaface_anchor_t> &anchors);
float calculate_iou(float x1, float y1, float x2, float y2, 
                   float x3, float y3, float x4, float y4);
int non_max_suppression(std::vector<retinaface_detection_t> &detections, float nms_threshold);

#endif //_RETINAFACE_POSTPROCESS_H_
