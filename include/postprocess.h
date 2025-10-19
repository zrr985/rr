#ifndef _RKNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <stdint.h>    // 标准整数类型定义，提供int8_t、int16_t、int32_t、int64_t等
#include <vector>      // C++标准库向量容器，用于动态数组操作
#include "rknn_api.h"  // RKNN API头文件，提供RKNN推理引擎的接口定义
#include "common.h"    // 公共定义头文件，包含image_rect_t等数据结构
#include "image_utils.h"  // 图像处理工具头文件，提供图像操作函数
#include "yolov8.h"    // YOLOv8模型相关定义

// 通用目标检测宏定义
#define OBJ_NAME_MAX_SIZE 64      // 目标名称最大长度（字符数）
#define OBJ_NUMB_MAX_SIZE 128     // 单帧最大检测目标数量
#define OBJ_CLASS_NUM 2           // 安全帽检测类别数：0-无安全帽，1-有安全帽
#define NMS_THRESH 0.45           // 非极大值抑制阈值，用于去除重复检测框
#define BOX_THRESH 0.25           // 检测框置信度阈值，低于此值的检测结果被过滤

// 火焰检测专用宏定义
#define FLAME_OBJ_NAME_MAX_SIZE 64      // 火焰目标名称最大长度
#define FLAME_OBJ_NUMB_MAX_SIZE 128     // 单帧最大火焰检测数量
#define FLAME_OBJ_CLASS_NUM 1           // 火焰检测类别数：0-火焰
#define FLAME_NMS_THRESH 0.45           // 火焰检测NMS阈值
#define FLAME_BOX_THRESH 0.25           // 火焰检测置信度阈值

// 吸烟检测专用宏定义
#define SMOKING_OBJ_NAME_MAX_SIZE 64    // 吸烟目标名称最大长度
#define SMOKING_OBJ_NUMB_MAX_SIZE 128   // 单帧最大吸烟检测数量
#define SMOKING_OBJ_CLASS_NUM 3         // 吸烟检测类别数：0-cigarette, 1-face, 2-smoking
#define SMOKING_NMS_THRESH 0.45         // 吸烟检测NMS阈值
#define SMOKING_BOX_THRESH 0.5          // 吸烟检测置信度阈值

// 人脸检测专用宏定义
#define FACE_OBJ_NAME_MAX_SIZE 64       // 人脸目标名称最大长度
#define FACE_OBJ_NUMB_MAX_SIZE 128      // 单帧最大人脸检测数量
#define FACE_OBJ_CLASS_NUM 3            // 人脸识别类别数：0-范喆洋, 1-陈俊杰, 2-张蕊蕊
#define FACE_NMS_THRESH 0.5             // 人脸检测NMS阈值
#define FACE_BOX_THRESH 0.5             // 人脸检测置信度阈值

// class rknn_app_context_t;  // 前向声明，避免循环包含

// 通用目标检测结果结构体 - 存储单个检测目标的详细信息
typedef struct {
    image_rect_t box;    // 检测框坐标（left, top, right, bottom）
    float prop;          // 检测置信度（0.0-1.0）
    int cls_id;          // 类别ID（0, 1, 2...）
} object_detect_result;

// 通用目标检测结果列表结构体 - 存储一帧图像的所有检测结果
typedef struct {
    int id;                                    // 帧ID或检测任务ID
    int count;                                 // 检测到的目标数量
    object_detect_result results[OBJ_NUMB_MAX_SIZE];  // 检测结果数组，最大128个目标
} object_detect_result_list;

// 火焰检测专用结构体 - 针对火焰检测任务优化的数据结构
typedef struct {
    image_rect_t box;    // 火焰检测框坐标
    float prop;          // 火焰检测置信度
    int cls_id;          // 火焰类别ID（通常为0）
} flame_detect_result;

typedef struct {
    int id;                                    // 帧ID或检测任务ID
    int count;                                 // 检测到的火焰数量
    flame_detect_result results[FLAME_OBJ_NUMB_MAX_SIZE];  // 火焰检测结果数组
} flame_detect_result_list;

// 吸烟检测专用结构体 - 针对吸烟检测任务优化的数据结构
typedef struct {
    image_rect_t box;    // 吸烟相关目标检测框坐标
    float prop;          // 吸烟检测置信度
    int cls_id;          // 类别ID（0-cigarette, 1-face, 2-smoking）
} smoking_detect_result;

typedef struct {
    int id;                                    // 帧ID或检测任务ID
    int count;                                 // 检测到的吸烟相关目标数量
    smoking_detect_result results[SMOKING_OBJ_NUMB_MAX_SIZE];  // 吸烟检测结果数组
} smoking_detect_result_list;

// 人脸检测专用结构体 - 针对人脸识别任务优化的数据结构
typedef struct {
    image_rect_t box;    // 人脸检测框坐标
    float prop;          // 人脸识别置信度
    int cls_id;          // 人员ID（0-范喆洋, 1-陈俊杰, 2-张蕊蕊）
} face_detect_result;

typedef struct {
    int id;                                    // 帧ID或检测任务ID
    int count;                                 // 检测到的人脸数量
    face_detect_result results[FACE_OBJ_NUMB_MAX_SIZE];  // 人脸检测结果数组
} face_detect_result_list;

// 通用目标检测后处理函数
int init_post_process();  // 初始化通用后处理模块，加载标签文件等
void deinit_post_process();  // 清理通用后处理模块资源
const char *coco_cls_to_name(int cls_id);  // 将类别ID转换为类别名称
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);  // 通用后处理主函数

// 火焰检测专用函数
int init_flame_post_process();  // 初始化火焰检测后处理模块
void deinit_flame_post_process();  // 清理火焰检测后处理模块资源
const char *flame_cls_to_name(int cls_id);  // 将火焰类别ID转换为名称
int flame_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, flame_detect_result_list *od_results);  // 火焰检测后处理主函数

// 吸烟检测专用函数
int init_smoking_post_process();  // 初始化吸烟检测后处理模块
void deinit_smoking_post_process();  // 清理吸烟检测后处理模块资源
const char *smoking_cls_to_name(int cls_id);  // 将吸烟类别ID转换为名称
int smoking_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, smoking_detect_result_list *od_results);  // 吸烟检测后处理主函数

// 人脸检测专用函数
int init_face_post_process();  // 初始化人脸检测后处理模块
void deinit_face_post_process();  // 清理人脸检测后处理模块资源
const char *face_cls_to_name(int cls_id);  // 将人脸类别ID转换为人员姓名
int face_post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, face_detect_result_list *od_results);  // 人脸检测后处理主函数

void deinitPostProcess();  // 兼容性函数，清理所有后处理模块
#endif //_RKNN_YOLOV8_DEMO_POSTPROCESS_H_  // 头文件保护宏结束，防止重复包含
