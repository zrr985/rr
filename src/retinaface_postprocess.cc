#include "retinaface_postprocess.h"
#include "yolov8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

// 全局变量
static std::vector<retinaface_anchor_t> g_anchors;
static bool g_anchors_initialized = false;

// 生成RetinaFace锚框
void generate_retinaface_anchors(int input_size, std::vector<retinaface_anchor_t> &anchors) {
    anchors.clear();
    
    // RetinaFace配置
    std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
    std::vector<int> steps = {8, 16, 32};
    
    for (int k = 0; k < 3; k++) {
        int step = steps[k];
        int grid_h = (input_size + step - 1) / step;
        int grid_w = (input_size + step - 1) / step;
        
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                for (int min_size : min_sizes[k]) {
                    retinaface_anchor_t anchor;
                    anchor.s_kx = (float)min_size / input_size;
                    anchor.s_ky = (float)min_size / input_size;
                    anchor.cx = (j + 0.5) * step / input_size;
                    anchor.cy = (i + 0.5) * step / input_size;
                    anchors.push_back(anchor);
                }
            }
        }
    }
}

// 计算IoU
float calculate_iou(float x1, float y1, float x2, float y2, 
                   float x3, float y3, float x4, float y4) {
    float inter_x1 = std::max(x1, x3);
    float inter_y1 = std::max(y1, y3);
    float inter_x2 = std::min(x2, x4);
    float inter_y2 = std::min(y2, y4);
    
    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
        return 0.0f;
    }
    
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x4 - x3) * (y4 - y3);
    float union_area = area1 + area2 - inter_area;
    
    return inter_area / union_area;
}

// 非极大值抑制
int non_max_suppression(std::vector<retinaface_detection_t> &detections, float nms_threshold) {
    if (detections.empty()) {
        return 0;
    }
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(), 
              [](const retinaface_detection_t &a, const retinaface_detection_t &b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(detections.size(), false);
    int count = 0;
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        count++;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            float iou = calculate_iou(detections[i].x1, detections[i].y1, 
                                    detections[i].x2, detections[i].y2,
                                    detections[j].x1, detections[j].y1, 
                                    detections[j].x2, detections[j].y2);
            
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    // 移除被抑制的检测结果
    size_t write_idx = 0;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            if (write_idx != i) {
                detections[write_idx] = detections[i];
            }
            write_idx++;
        }
    }
    detections.resize(write_idx);
    
    return count;
}

// 解码边界框
void decode_box(float *loc, const retinaface_anchor_t &anchor, float *box, float *variance) {
    float cx = anchor.cx + loc[0] * variance[0] * anchor.s_kx;
    float cy = anchor.cy + loc[1] * variance[0] * anchor.s_ky;
    float w = anchor.s_kx * expf(loc[2] * variance[1]);
    float h = anchor.s_ky * expf(loc[3] * variance[1]);
    
    box[0] = cx - w / 2.0f;  // x1
    box[1] = cy - h / 2.0f;  // y1
    box[2] = cx + w / 2.0f;  // x2
    box[3] = cy + h / 2.0f;  // y2
}

// 解码关键点
void decode_landmarks(float *landm, const retinaface_anchor_t &anchor, float *landmarks, float *variance) {
    for (int i = 0; i < 5; i++) {
        landmarks[i * 2] = anchor.cx + landm[i * 2] * variance[0] * anchor.s_kx;
        landmarks[i * 2 + 1] = anchor.cy + landm[i * 2 + 1] * variance[0] * anchor.s_ky;
    }
}

// RetinaFace后处理主函数
int retinaface_post_process(rknn_app_context_t *app_ctx, void *outputs, 
                           letterbox_t *letter_box, float conf_threshold, 
                           float nms_threshold, retinaface_result_list_t *results) {
    
    if (!g_anchors_initialized) {
        generate_retinaface_anchors(RETINAFACE_INPUT_SIZE, g_anchors);
        g_anchors_initialized = true;
        printf("Generated %zu RetinaFace anchors\n", g_anchors.size());
    }
    
    memset(results, 0, sizeof(retinaface_result_list_t));
    
    // 获取输出数据
    rknn_output *_outputs = (rknn_output *)outputs;
    
    // RetinaFace输出格式: [loc, conf, landm]
    float *loc_data = (float *)_outputs[0].buf;
    float *conf_data = (float *)_outputs[1].buf;
    float *landm_data = (float *)_outputs[2].buf;
    
    float variance[2] = {0.1f, 0.2f};
    std::vector<retinaface_detection_t> detections;
    
    // 处理每个锚框
    for (size_t i = 0; i < g_anchors.size(); i++) {
        const retinaface_anchor_t &anchor = g_anchors[i];
        
        // 获取置信度（正样本）
        float confidence = conf_data[i * 2 + 1];  // 正样本置信度
        
        if (confidence < conf_threshold) {
            continue;
        }
        
        // 解码边界框
        float box[4];
        decode_box(&loc_data[i * 4], anchor, box, variance);
        
        // 解码关键点
        float landmarks[10];
        decode_landmarks(&landm_data[i * 10], anchor, landmarks, variance);
        
        // 创建检测结果
        retinaface_detection_t detection;
        detection.x1 = box[0];
        detection.y1 = box[1];
        detection.x2 = box[2];
        detection.y2 = box[3];
        detection.confidence = confidence;
        detection.class_id = 0;  // 人脸类别
        memcpy(detection.landmarks, landmarks, sizeof(landmarks));
        
        detections.push_back(detection);
    }
    
    // 非极大值抑制
    int valid_count = non_max_suppression(detections, nms_threshold);
    
    // 坐标转换到原图
    float scale = letter_box->scale;
    
    // 复制结果
    int count = std::min(valid_count, 128);
    for (int i = 0; i < count; i++) {
        retinaface_detection_t &det = detections[i];
        
        // 转换坐标（与postprocess.cc中的逻辑一致）
        float x1 = det.x1 - letter_box->x_pad;
        float y1 = det.y1 - letter_box->y_pad;
        float x2 = det.x2 - letter_box->x_pad;
        float y2 = det.y2 - letter_box->y_pad;
        
        det.x1 = x1 / scale;
        det.y1 = y1 / scale;
        det.x2 = x2 / scale;
        det.y2 = y2 / scale;
        
        // 转换关键点坐标
        for (int j = 0; j < 10; j += 2) {
            float lx = det.landmarks[j] - letter_box->x_pad;
            float ly = det.landmarks[j + 1] - letter_box->y_pad;
            det.landmarks[j] = lx / scale;
            det.landmarks[j + 1] = ly / scale;
        }
        
        results->detections[results->count] = det;
        results->count++;
    }
    
    return 0;
}

// 初始化RetinaFace后处理
int init_retinaface_post_process() {
    printf("初始化RetinaFace后处理\n");
    return 0;
}

// 释放RetinaFace后处理
void deinit_retinaface_post_process() {
    printf("释放RetinaFace后处理\n");
    g_anchors.clear();
    g_anchors_initialized = false;
}

// 初始化人脸识别
int init_face_recognition() {
    printf("初始化人脸识别\n");
    // TODO: 加载人脸特征数据库
    return 0;
}

// 释放人脸识别
void deinit_face_recognition() {
    printf("释放人脸识别\n");
}

// 人脸识别处理
int face_recognition_process(retinaface_result_list_t *detections, 
                            retinaface_face_result_list *results) {
    // TODO: 实现人脸识别逻辑
    // 1. 提取人脸特征
    // 2. 与数据库比较
    // 3. 返回识别结果
    
    memset(results, 0, sizeof(retinaface_face_result_list));
    
    for (int i = 0; i < detections->count; i++) {
        retinaface_detection_t &det = detections->detections[i];
        
        retinaface_face_result &result = results->results[results->count];
        result.box_x = (int)det.x1;
        result.box_y = (int)det.y1;
        result.box_w = (int)(det.x2 - det.x1);
        result.box_h = (int)(det.y2 - det.y1);
        result.score = det.confidence;
        result.cls_id = det.class_id;
        memcpy(result.landmarks, det.landmarks, sizeof(det.landmarks));
        
        // 默认身份为Unknown
        strcpy(result.identity, "Unknown");
        strcpy(result.name, "Unknown");
        
        results->count++;
    }
    
    return 0;
}
