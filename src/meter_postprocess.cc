#include "meter_postprocess.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <exception>

// 仪表类别名称
static const char* meter_class_names[METER_CLASS_NUM] = {
    "background",
    "pointer", 
    "scale"
};

// 全局变量
static bool g_meter_post_process_initialized = false;
static bool g_meter_reader_initialized = false;

// 初始化仪表后处理
int init_meter_post_process() {
    if (g_meter_post_process_initialized) {
        return 0;
    }
    
    printf("初始化仪表后处理...\n");
    g_meter_post_process_initialized = true;
    return 0;
}

// 释放仪表后处理
void deinit_meter_post_process() {
    if (!g_meter_post_process_initialized) {
        return;
    }
    
    printf("释放仪表后处理...\n");
    g_meter_post_process_initialized = false;
}

// 初始化仪表读数器
int init_meter_reader() {
    if (g_meter_reader_initialized) {
        return 0;
    }
    
    printf("初始化仪表读数器...\n");
    g_meter_reader_initialized = true;
    return 0;
}

// 释放仪表读数器
void deinit_meter_reader() {
    if (!g_meter_reader_initialized) {
        return;
    }
    
    printf("释放仪表读数器...\n");
    g_meter_reader_initialized = false;
}

// 计算IoU
float calculate_iou_meter(float x1, float y1, float x2, float y2,
                         float x3, float y3, float x4, float y4) {
    float intersection_x1 = std::max(x1, x3);
    float intersection_y1 = std::max(y1, y3);
    float intersection_x2 = std::min(x2, x4);
    float intersection_y2 = std::min(y2, y4);
    
    if (intersection_x1 >= intersection_x2 || intersection_y1 >= intersection_y2) {
        return 0.0f;
    }
    
    float intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1);
    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x4 - x3) * (y4 - y3);
    float union_area = area1 + area2 - intersection_area;
    
    return intersection_area / union_area;
}

// 非极大值抑制
int non_max_suppression_meter(meter_detection_t *detections, int count, float nms_threshold) {
    if (count <= 1) {
        return count;
    }
    
    // 按置信度排序
    std::vector<int> indices(count);
    for (int i = 0; i < count; i++) {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<bool> suppressed(count, false);
    int valid_count = 0;
    
    for (int i = 0; i < count; i++) {
        if (suppressed[indices[i]]) {
            continue;
        }
        
        valid_count++;
        
        for (int j = i + 1; j < count; j++) {
            if (suppressed[indices[j]]) {
                continue;
            }
            
            // 只对相同类别的检测进行NMS
            if (detections[indices[i]].class_id != detections[indices[j]].class_id) {
                continue;
            }
            
            float iou = calculate_iou_meter(
                detections[indices[i]].box.left, detections[indices[i]].box.top,
                detections[indices[i]].box.right, detections[indices[i]].box.bottom,
                detections[indices[j]].box.left, detections[indices[j]].box.top,
                detections[indices[j]].box.right, detections[indices[j]].box.bottom
            );
            
            if (iou > nms_threshold) {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    // 重新排列数组
    std::vector<meter_detection_t> temp_detections;
    for (int i = 0; i < count; i++) {
        if (!suppressed[indices[i]]) {
            temp_detections.push_back(detections[indices[i]]);
        }
    }
    
    for (int i = 0; i < valid_count; i++) {
        detections[i] = temp_detections[i];
    }
    
    return valid_count;
}

// 获取类别名称
const char* meter_cls_to_name(int cls_id) {
    if (cls_id >= 0 && cls_id < METER_CLASS_NUM) {
        return meter_class_names[cls_id];
    }
    return "unknown";
}

// sigmoid函数
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// sp_flatten函数（对应Python的sp_flatten）
static void sp_flatten(float* input, int n, int c, int h, int w, float* output) {
    // 对应Python: _in.transpose(0,2,3,1).reshape(-1, ch)
    // 从 [n, c, h, w] 转换为 [n*h*w, c]
    for (int i = 0; i < n; i++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    int src_idx = i * c * h * w + ch * h * w + y * w + x;
                    int dst_idx = (i * h * w + y * w + x) * c + ch;
                    output[dst_idx] = input[src_idx];
                }
            }
        }
    }
}

// 过滤boxes函数（对应Python的filter_boxes）
static int filter_boxes(std::vector<float>& boxes, 
                       std::vector<float>& box_confidences,
                       std::vector<float>& box_class_probs,
                       std::vector<float>& seg_part,
                       float obj_thresh,
                       std::vector<float>& filtered_boxes,
                       std::vector<int>& filtered_classes,
                       std::vector<float>& filtered_scores,
                       std::vector<float>& filtered_seg_parts) {
    
    int candidate = box_class_probs.size() / METER_CLASS_NUM;
    int valid_count = 0;
    
    for (int i = 0; i < candidate; i++) {
        // 找到最大类别分数
        float max_score = 0.0f;
        int max_class = 0;
        for (int j = 0; j < METER_CLASS_NUM; j++) {
            float score = box_class_probs[i * METER_CLASS_NUM + j];
            if (score > max_score) {
                max_score = score;
                max_class = j;
            }
        }
        
        // 计算最终分数
        float final_score = max_score * box_confidences[i];
        
        // 应用阈值过滤
        if (final_score >= obj_thresh) {
            // 添加box坐标
            for (int k = 0; k < 4; k++) {
                filtered_boxes.push_back(boxes[i * 4 + k]);
            }
            
            // 添加类别和分数
            filtered_classes.push_back(max_class);
            filtered_scores.push_back(final_score);
            
            // 添加分割部分
            for (int k = 0; k < (int)seg_part.size() / candidate; k++) {
                filtered_seg_parts.push_back(seg_part[i * (seg_part.size() / candidate) + k]);
            }
            
            valid_count++;
        }
    }
    
    return valid_count;
}

// DFL函数（Distribution Focal Loss）
static void dfl(float* position, int n, int c, int h, int w) {
    int p_num = 4;
    int mc = c / p_num;
    
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < p_num; p++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float* src = position + i * c * h * w + p * mc * h * w + y * w + x;
                    
                    // 找到最大值（数值稳定性）
                    float max_val = src[0];
                    for (int k = 1; k < mc; k++) {
                        if (src[k * h * w] > max_val) {
                            max_val = src[k * h * w];
                        }
                    }
                    
                    // 计算softmax
                    float sum = 0.0f;
                    for (int k = 0; k < mc; k++) {
                        src[k * h * w] = expf(src[k * h * w] - max_val);
                        sum += src[k * h * w];
                    }
                    
                    // 归一化并计算加权和
                    float result = 0.0f;
                    for (int k = 0; k < mc; k++) {
                        src[k * h * w] /= sum;
                        result += src[k * h * w] * k;
                    }
                    
                    // 将结果存储到第一个位置
                    src[0] = result;
                }
            }
        }
    }
}

// box处理函数（对应Python的box_process）
static void box_process(float* position, int n, int c, int h, int w, float* output) {
    // 应用DFL
    dfl(position, n, c, h, w);
    
    // 创建网格坐标
    float stride_x = (float)METER_INPUT_SIZE / w;
    float stride_y = (float)METER_INPUT_SIZE / h;
    
    for (int i = 0; i < n; i++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float* src = position + i * c * h * w + y * w + x;
                float* dst = output + i * 4 * h * w + y * w + x;
                
                // 计算网格坐标
                float grid_x = x + 0.5f;
                float grid_y = y + 0.5f;
                
                // 计算box坐标（对应Python版本）
                dst[0] = (grid_x - src[0]) * stride_x;  // x1
                dst[1] = (grid_y - src[1]) * stride_y;  // y1
                dst[2] = (grid_x + src[2]) * stride_x;  // x2
                dst[3] = (grid_y + src[3]) * stride_y;  // y2
            }
        }
    }
}

// 仪表后处理主函数（对应Python的yolov8_seg_post_process）
int meter_post_process(rknn_app_context_t *app_ctx, void *outputs,
                      letterbox_t *letter_box, float conf_threshold,
                      float nms_threshold, meter_result_list_t *results) {
    if (!g_meter_post_process_initialized) {
        printf("仪表后处理未初始化\n");
        return -1;
    }
    
    results->count = 0;
    
    // 获取模型输出
    rknn_output* output_list = (rknn_output*)outputs;
    int output_count = app_ctx->io_num.n_output;
    
    if (output_count < 4) {
        printf("仪表模型输出数量不足: %d\n", output_count);
        return -1;
    }
    
    printf("仪表后处理: 检测到 %d 个输出\n", output_count);
    
    // 打印输出信息用于调试
    for (int i = 0; i < output_count; i++) {
        rknn_output& output = output_list[i];
        printf("输出 %d: 大小=%d字节, 索引=%d\n", i, output.size, output.index);
    }
    
    try {
        // 1. 获取proto（最后一个输出）
        rknn_output& proto_output = output_list[output_count - 1];
        float* proto_data = (float*)proto_output.buf;
        
        // 2. 处理多个分支（默认3个分支）
        int default_branch = 3;
        int pair_per_branch = (output_count - 1) / default_branch;
        
        std::vector<float> all_boxes;
        std::vector<float> all_classes;
        std::vector<float> all_scores;
        std::vector<float> all_seg_parts;
        
        // 处理每个分支（对应Python的yolov8_seg_post_process）
        for (int i = 0; i < default_branch; i++) {
            // 处理box输出 (pair_per_branch*i)
            if (i * pair_per_branch < output_count - 1) {
                rknn_output& box_output = output_list[i * pair_per_branch];
                float* box_data = (float*)box_output.buf;
                
                // 根据yolov8_seg_newer.rknn的实际输出格式
                // 假设输出格式为 [1, 64, 80, 80] 或类似
                int n = 1; // batch size
                int c = 64; // 通道数（需要根据实际模型调整）
                int h = 80; // 高度（需要根据实际模型调整）
                int w = 80; // 宽度（需要根据实际模型调整）
                
                // 验证输出尺寸
                int expected_size = n * c * h * w * sizeof(float);
                if (box_output.size != expected_size) {
                    // 动态计算尺寸
                    int total_elements = box_output.size / sizeof(float);
                    // 尝试不同的尺寸组合
                    for (int test_h = 20; test_h <= 160; test_h += 20) {
                        for (int test_w = 20; test_w <= 160; test_w += 20) {
                            if (total_elements == n * c * test_h * test_w) {
                                h = test_h;
                                w = test_w;
                                break;
                            }
                        }
                        if (h != 80) break;
                    }
                }
                
                // 应用box_process
                std::vector<float> processed_boxes(n * 4 * h * w);
                box_process(box_data, n, c, h, w, processed_boxes.data());
                
                // 使用sp_flatten展平
                std::vector<float> flattened_boxes(n * h * w * 4);
                sp_flatten(processed_boxes.data(), n, 4, h, w, flattened_boxes.data());
                
                // 添加到总列表
                for (float val : flattened_boxes) {
                    all_boxes.push_back(val);
                }
            }
            
            // 处理class输出 (pair_per_branch*i+1)
            if (i * pair_per_branch + 1 < output_count - 1) {
                rknn_output& class_output = output_list[i * pair_per_branch + 1];
                float* class_data = (float*)class_output.buf;
                
                // 假设输出格式为 [1, 3, h, w]
                int n = 1;
                int c = METER_CLASS_NUM; // 3个类别
                int h = 80; // 需要根据实际模型调整
                int w = 80; // 需要根据实际模型调整
                
                // 验证并调整尺寸
                int total_elements = class_output.size / sizeof(float);
                if (total_elements != n * c * h * w) {
                    for (int test_h = 20; test_h <= 160; test_h += 20) {
                        for (int test_w = 20; test_w <= 160; test_w += 20) {
                            if (total_elements == n * c * test_h * test_w) {
                                h = test_h;
                                w = test_w;
                                break;
                            }
                        }
                        if (h != 80) break;
                    }
                }
                
                // 使用sp_flatten展平
                std::vector<float> flattened_classes(n * h * w * c);
                sp_flatten(class_data, n, c, h, w, flattened_classes.data());
                
                // 添加到总列表
                for (float val : flattened_classes) {
                    all_classes.push_back(val);
                }
                
                // 创建分数（全1，对应Python的np.ones_like）
                for (int j = 0; j < n * h * w; j++) {
                    all_scores.push_back(1.0f);
                }
            }
            
            // 处理seg输出 (pair_per_branch*i+3)
            if (i * pair_per_branch + 3 < output_count - 1) {
                rknn_output& seg_output = output_list[i * pair_per_branch + 3];
                float* seg_data = (float*)seg_output.buf;
                
                // 假设输出格式为 [1, 32, h, w] 或类似
                int n = 1;
                int c = 32; // 分割通道数（需要根据实际模型调整）
                int h = 80; // 需要根据实际模型调整
                int w = 80; // 需要根据实际模型调整
                
                // 验证并调整尺寸
                int total_elements = seg_output.size / sizeof(float);
                if (total_elements != n * c * h * w) {
                    for (int test_h = 20; test_h <= 160; test_h += 20) {
                        for (int test_w = 20; test_w <= 160; test_w += 20) {
                            if (total_elements == n * c * test_h * test_w) {
                                h = test_h;
                                w = test_w;
                                break;
                            }
                        }
                        if (h != 80) break;
                    }
                }
                
                // 使用sp_flatten展平
                std::vector<float> flattened_seg(n * h * w * c);
                sp_flatten(seg_data, n, c, h, w, flattened_seg.data());
                
                // 添加到总列表
                for (float val : flattened_seg) {
                    all_seg_parts.push_back(val);
                }
            }
        }
        
        // 3. 应用置信度阈值过滤
        std::vector<float> filtered_boxes;
        std::vector<int> filtered_classes;
        std::vector<float> filtered_scores;
        std::vector<float> filtered_seg_parts;
        
        int valid_count = filter_boxes(all_boxes, all_scores, all_classes, all_seg_parts,
                                     conf_threshold, filtered_boxes, filtered_classes,
                                     filtered_scores, filtered_seg_parts);
        
        if (valid_count == 0) {
            printf("没有检测到有效的仪表目标\n");
            return 0;
        }
        
        // 4. 执行NMS
        std::vector<meter_detection_t> detections;
        for (int i = 0; i < valid_count; i++) {
            meter_detection_t det;
            det.box.left = filtered_boxes[i * 4];
            det.box.top = filtered_boxes[i * 4 + 1];
            det.box.right = filtered_boxes[i * 4 + 2];
            det.box.bottom = filtered_boxes[i * 4 + 3];
            det.confidence = filtered_scores[i];
            det.class_id = filtered_classes[i];
            det.mask = nullptr; // 暂时不处理掩码
            det.mask_width = 0;
            det.mask_height = 0;
            detections.push_back(det);
        }
        
        // 执行NMS
        int nms_count = non_max_suppression_meter(detections.data(), detections.size(), nms_threshold);
        
        // 5. 生成最终结果
        int count = std::min(nms_count, METER_MAX_DETECTIONS);
        for (int i = 0; i < count; i++) {
            results->detections[results->count] = detections[i];
            results->count++;
        }
        
        printf("仪表后处理完成，检测到 %d 个目标\n", results->count);
        
    } catch (const std::exception& e) {
        printf("仪表后处理异常: %s\n", e.what());
        return -1;
    }
    
    return 0;
}

// 仪表读数处理
int meter_reading_process(meter_result_list_t *detections,
                         meter_reading_result_list *results) {
    if (!g_meter_reader_initialized) {
        printf("仪表读数器未初始化\n");
        return -1;
    }
    
    results->count = 0;
    
    if (detections->count == 0) {
        return 0;
    }
    
    // 简化的读数处理逻辑
    // 实际实现需要：
    // 1. 检测圆形区域
    // 2. 提取指针和刻度掩码
    // 3. 计算指针位置
    // 4. 计算读数
    
    printf("仪表读数处理: 处理 %d 个检测结果\n", detections->count);
    
    // 模拟读数结果
    for (int i = 0; i < detections->count && i < METER_MAX_DETECTIONS; i++) {
        meter_reading_result* result = &results->results[results->count];
        
        result->id = i;
        result->box_x = detections->detections[i].box.left;
        result->box_y = detections->detections[i].box.top;
        result->box_w = detections->detections[i].box.right - detections->detections[i].box.left;
        result->box_h = detections->detections[i].box.bottom - detections->detections[i].box.top;
        result->score = detections->detections[i].confidence;
        result->cls_id = detections->detections[i].class_id;
        result->item_id = i;
        result->mask_offset = 0;
        
        strcpy(result->name, meter_cls_to_name(detections->detections[i].class_id));
        strcpy(result->meter_type, "pressure_gauge");
        result->reading_value = 0.0f;  // 实际读数计算
        
        results->count++;
    }
    
    return 0;
}
