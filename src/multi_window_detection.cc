#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <map>
#include <signal.h>
#include <memory>

#include "postprocess.h"
#include "retinaface_postprocess.h"
#include "meter_postprocess.h"
#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "shared_camera_manager.h"

// 全局变量
std::atomic<bool> g_running(true);

// 中文文本绘制函数
void putChineseText(cv::Mat& img, const std::string& text, cv::Point pos, 
                   cv::Scalar color, double font_scale = 0.6, int thickness = 2) {
    // 将中文转换为英文显示，避免问号问题
    std::string english_text;
    
    // 检查是否包含中文姓名（人脸识别结果）
    if (text.find("范喆洋") != std::string::npos) {
        english_text = "Fan Zheyang";
    } else if (text.find("陈俊杰") != std::string::npos) {
        english_text = "Chen Junjie";
    } else if (text.find("张蕊蕊") != std::string::npos) {
        english_text = "Zhang Ruirui";
    } else if (text.find("安全帽") != std::string::npos) {
        english_text = "Helmet";
    } else if (text.find("无安全帽") != std::string::npos) {
        english_text = "No Helmet";
    } else if (text.find("火焰") != std::string::npos) {
        english_text = "Flame";
    } else if (text.find("吸烟") != std::string::npos) {
        english_text = "Smoking";
    } else if (text.find("人脸") != std::string::npos) {
        english_text = "Face";
    } else if (text.find("仪表") != std::string::npos) {
        english_text = "Meter";
    } else if (text.find("FPS") != std::string::npos) {
        english_text = text; // FPS保持原样
    } else if (text.find("队列") != std::string::npos) {
        english_text = "Queue";
    } else if (text.find("处理时间") != std::string::npos) {
        english_text = "Time";
    } else if (text.find("任务") != std::string::npos) {
        english_text = "Task";
    } else {
        // 如果包含中文字符，尝试提取置信度并显示为"Face"
        if (text.length() > 0 && (unsigned char)text[0] > 127) {
            // 包含中文字符，提取置信度
            size_t space_pos = text.find(' ');
            if (space_pos != std::string::npos) {
                std::string confidence = text.substr(space_pos + 1);
                english_text = "Face " + confidence;
            } else {
                english_text = "Face";
            }
        } else {
            english_text = text; // 其他文本保持原样
        }
    }
    
    cv::putText(img, english_text, pos, cv::FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness);
}

// 信号处理函数
void signal_handler(int sig) {
    printf("\n收到退出信号，正在停止...\n");
    g_running = false;
}

// FPS计算
class FPSCounter {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        frame_count_ = 0;
    }
    
    double get_fps() {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();
        if (elapsed > 0) {
            return (frame_count_ * 1000.0) / elapsed;
        }
        return 0.0;
    }
    
    void update() {
        frame_count_++;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int frame_count_ = 0;
};

// 多窗口检测系统 - 参照Python架构
class MultiWindowDetector {
public:
    struct DetectionResult {
        std::shared_ptr<cv::Mat> frame;  // 使用智能指针共享帧数据，避免拷贝
        std::string task_name;
        object_detect_result_list results;
        double processing_time_ms;
        int64_t frame_id;
        int class_result;  // 检测结果分类值
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    MultiWindowDetector(const std::vector<std::string>& model_paths) 
        : running_(true), start_time_(std::chrono::high_resolution_clock::now()) {
        
        // 初始化全局摄像头管理器
        camera_manager_ = &GlobalCameraManager::getInstance();
        
        // 初始化检测结果状态
        detection_results_["helmet"] = "Not Detected";
        detection_results_["flame"] = "Not Detected";
        detection_results_["smoking"] = "Not Detected";
        detection_results_["face"] = "Not Detected";
        detection_results_["meter"] = "Not Detected";
        
        // 初始化异常计数
        abnormal_counts_["helmet"] = 0;
        abnormal_counts_["flame"] = 0;
        abnormal_counts_["smoking"] = 0;
        abnormal_counts_["face"] = 0;
        abnormal_counts_["meter"] = 0;
        
        // 为每个任务创建RKNN应用上下文
        for (size_t i = 0; i < model_paths.size(); i++) {
            std::vector<rknn_app_context_t> contexts;
            for (int j = 0; j < 3; j++) {  // 3个NPU核心
                rknn_app_context_t app_ctx;
                int ret = init_yolov8_model(model_paths[i].c_str(), &app_ctx);
                if (ret != 0) {
                    printf("init_yolov8_model failed for %s, ret=%d\n", model_paths[i].c_str(), ret);
                    exit(-1);
                }
                
                // 绑定到不同的NPU核心
                rknn_core_mask core_mask;
                if (j == 0) core_mask = RKNN_NPU_CORE_0;
                else if (j == 1) core_mask = RKNN_NPU_CORE_1;
                else if (j == 2) core_mask = RKNN_NPU_CORE_2;
                else core_mask = RKNN_NPU_CORE_0_1_2;
                
                ret = rknn_set_core_mask(app_ctx.rknn_ctx, core_mask);
                if (ret != RKNN_SUCC) {
                    printf("rknn_set_core_mask failed, ret=%d\n", ret);
                    exit(-1);
                }
                
                contexts.push_back(app_ctx);
            }
            task_contexts_[task_names_[i]] = contexts;
            printf("%s 模型初始化成功\n", task_names_[i].c_str());
        }
    }
    
    ~MultiWindowDetector() {
        stop();
        
        // 释放RKNN应用上下文
        for (auto& [task_name, contexts] : task_contexts_) {
            for (auto& app_ctx : contexts) {
                release_yolov8_model(&app_ctx);
            }
        }
    }
    
    void start() {
        printf("启动多窗口检测系统...\n");
        
        // 先检测可用摄像头
        printf("正在检测可用摄像头...\n");
        auto available_cameras = camera_manager_->getAvailableCameras();
        if (available_cameras.empty()) {
            printf("未找到可用摄像头!\n");
            return;
        }
        
        printf("发现 %zu 个可用摄像头: ", available_cameras.size());
        for (int cam_id : available_cameras) {
            printf("%d ", cam_id);
        }
        printf("\n");
        
        // 获取摄像头
        printf("正在获取摄像头 0...\n");
        camera_ = camera_manager_->getCamera(0);
        if (!camera_) {
            printf("无法获取摄像头 0，尝试其他摄像头...\n");
            // 尝试其他可用摄像头
            for (int cam_id : available_cameras) {
                printf("尝试获取摄像头 %d...\n", cam_id);
                camera_ = camera_manager_->getCamera(cam_id);
                if (camera_) {
                    printf("成功获取摄像头 %d\n", cam_id);
                    break;
                }
            }
            if (!camera_) {
                printf("所有摄像头都无法获取!\n");
                return;
            }
        } else {
            printf("成功获取摄像头 0\n");
        }
        
        printf("摄像头信息: ID=%d, 宽度=%d, 高度=%d\n", 
               camera_->getCameraId(), camera_->getWidth(), camera_->getHeight());
        
        // 启动摄像头帧捕获
        printf("启动摄像头帧捕获线程...\n");
        camera_->startFrameCapture();
        
        // 启动各个检测线程
        for (const auto& task_name : task_names_) {
            std::thread worker(&MultiWindowDetector::detectionWorker, this, task_name);
            workers_.push_back(std::move(worker));
        }
        
        // 启动显示线程
        display_thread_ = std::thread(&MultiWindowDetector::displayWorker, this);
        
        printf("所有检测窗口启动成功\n");
        printf("按 'q' 键退出\n");
    }
    
    void stop() {
        running_ = false;
        
        // 停止摄像头帧捕获
        if (camera_) {
            camera_->stopFrameCapture();
        }
        
        // 等待所有工作线程结束
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        // 等待显示线程结束
        if (display_thread_.joinable()) {
            display_thread_.join();
        }
        
        printf("多窗口检测系统已停止\n");
    }

private:
    void detectionWorker(const std::string& task_name) {
        printf("启动 %s 检测线程\n", task_name.c_str());
        
        // 检查模型池是否存在
        auto it = task_contexts_.find(task_name);
        if (it == task_contexts_.end()) {
            printf("%s 模型未初始化，跳过此任务\n", task_name.c_str());
            return;
        }
        
        auto& contexts = it->second;
        int worker_id = 0;
        
        FPSCounter fps_counter;
        fps_counter.start();
        
        while (running_) {
            try {
                // 从摄像头队列获取帧 - 使用智能指针避免拷贝
                std::shared_ptr<cv::Mat> frame_ptr;
                if (!camera_->getFrameFromQueue(frame_ptr)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 增加等待时间
                    continue;
                }
                
                // 执行推理
                auto start_time = std::chrono::high_resolution_clock::now();
                
                rknn_app_context_t& app_ctx = contexts[worker_id % contexts.size()];
                
                // 准备输入 - 使用智能指针
                image_buffer_t src_img;
                src_img.width = frame_ptr->cols;
                src_img.height = frame_ptr->rows;
                src_img.format = IMAGE_FORMAT_RGB888;
                src_img.virt_addr = frame_ptr->data;
                src_img.size = frame_ptr->cols * frame_ptr->rows * 3;
                
                // 执行推理
                object_detect_result_list od_results;
                int ret = inference_yolov8_model(&app_ctx, &src_img, &od_results);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                double processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
                
                if (ret == 0) {
                    // 分析检测结果
                    int class_result = analyzeDetectionResults(od_results, task_name);
                    
                    // 更新检测状态和异常计数
                    updateDetectionStatus(task_name, class_result);
                    
                    // 创建检测结果 - 使用智能指针避免帧拷贝
                    DetectionResult result;
                    result.frame = frame_ptr;  // 直接使用智能指针，避免拷贝
                    result.task_name = task_name;
                    result.results = od_results;
                    result.processing_time_ms = processing_time;
                    result.frame_id = frame_id_++;
                    result.class_result = class_result;
                    result.timestamp = std::chrono::high_resolution_clock::now();
                    
                    // 将结果放入显示队列 - 优化版本
                    {
                        std::lock_guard<std::mutex> lock(display_mutex_);
                        if (display_queue_.size() < MAX_DISPLAY_QUEUE_SIZE) {
                            display_queue_.push(result);
                            display_cv_.notify_one();  // 通知显示线程
                        }
                    }
                    
                    // 更新性能统计
                    total_frames_processed_++;
                    total_detection_time_us_ += static_cast<int64_t>(processing_time * 1000);
                }
                
                worker_id++;
                fps_counter.update();
                
                // 每30帧打印一次FPS
                static int frame_count = 0;
                if (++frame_count % 30 == 0) {
                    printf("%s Detection: %.1f FPS\n", task_name.c_str(), fps_counter.get_fps());
                }
                
            } catch (const std::exception& e) {
                printf("%s 检测异常: %s\n", task_name.c_str(), e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        printf("%s 检测线程结束\n", task_name.c_str());
    }
    
    void displayWorker() {
        printf("启动显示线程\n");
        
        FPSCounter fps_counter;
        fps_counter.start();
        
        while (running_) {
            DetectionResult result;
            bool has_result = false;
            
            // 从显示队列获取结果 - 使用条件变量优化等待
            {
                std::unique_lock<std::mutex> lock(display_mutex_);
                display_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                    return !display_queue_.empty() || !running_;
                });
                
                if (!display_queue_.empty()) {
                    result = display_queue_.front();
                    display_queue_.pop();
                    has_result = true;
                }
            }
            
            if (has_result) {
                // 绘制检测结果 - 使用智能指针
                drawDetectionResults(*result.frame, result.results, result.task_name);
                
                // 显示检测状态信息
                std::string status_text;
                cv::Scalar status_color;
                
                {
                    std::lock_guard<std::mutex> lock(results_mutex_);
                    status_text = detection_results_[result.task_name];
                    
                    if (result.class_result == 1) {
                        // 异常状态 - 红色
                        status_color = cv::Scalar(0, 0, 255);
                        if (result.task_name == "flame") {
                            status_text += " (Count: " + std::to_string(abnormal_counts_[result.task_name]) + ")";
                        }
                    } else {
                        // 正常状态 - 绿色
                        status_color = cv::Scalar(0, 255, 0);
                    }
                }
                
                // 显示状态文字 - 使用智能指针
                putChineseText(*result.frame, status_text, cv::Point(10, 30), status_color, 0.8, 2);
                
                // 显示统计信息 - 增强版本
                char text[256];
                sprintf(text, "FPS: %.1f | Queue: %d", 
                       fps_counter.get_fps(), 
                       static_cast<int>(display_queue_.size()));
                cv::putText(*result.frame, text, cv::Point(10, 70), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                
                sprintf(text, "Time: %.1fms | Task: %s", 
                       result.processing_time_ms, result.task_name.c_str());
                cv::putText(*result.frame, text, cv::Point(10, 100), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                
                sprintf(text, "Detection Value: %d", result.class_result);
                cv::putText(*result.frame, text, cv::Point(10, 130), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                
                // 显示性能统计
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time_).count();
                if (elapsed > 0) {
                    double avg_fps = static_cast<double>(total_frames_processed_) / elapsed;
                    double avg_time = static_cast<double>(total_detection_time_us_) / total_frames_processed_ / 1000.0;
                    sprintf(text, "Avg FPS: %.1f | Avg Time: %.1fms", avg_fps, avg_time);
                    cv::putText(*result.frame, text, cv::Point(10, 160), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
                }
                
                // 在独立窗口中显示 - 使用智能指针
                cv::imshow(result.task_name + " Detection", *result.frame);
                
                fps_counter.update();
            }
            
            // 检查退出条件
            char key = cv::waitKey(30) & 0xFF;
            if (key == 'q' || key == 27) {
                printf("用户退出\n");
                g_running = false;
                break;
            }
            
            // 优化等待时间，减少CPU占用
            if (!has_result) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        printf("显示线程结束\n");
    }
    
    void drawDetectionResults(cv::Mat& frame, const object_detect_result_list& results, const std::string& task_name) {
        cv::Scalar color;
        std::string label_prefix;
        
        if (task_name == "helmet") {
            color = cv::Scalar(0, 255, 0);      // 绿色
            label_prefix = "安全帽";
        } else if (task_name == "flame") {
            color = cv::Scalar(0, 165, 255);    // 橙色
            label_prefix = "火焰";
        } else if (task_name == "smoking") {
            color = cv::Scalar(255, 0, 255);    // 紫色
            label_prefix = "吸烟";
        } else if (task_name == "face") {
            color = cv::Scalar(255, 255, 0);    // 青色
            label_prefix = "人脸";
        } else if (task_name == "meter") {
            color = cv::Scalar(0, 255, 255);    // 黄色
            label_prefix = "仪表";
        } else {
            color = cv::Scalar(255, 255, 255);
            label_prefix = "未知";
        }
        
        for (int i = 0; i < results.count; i++) {
            const object_detect_result* det_result = &(results.results[i]);
            
            // 绘制边界框
            cv::rectangle(frame, 
                         cv::Point(det_result->box.left, det_result->box.top),
                         cv::Point(det_result->box.right, det_result->box.bottom),
                         color, 2);
            
            // 获取类别名称
            const char* class_name = nullptr;
            if (task_name == "helmet") {
                class_name = coco_cls_to_name(det_result->cls_id);
            } else if (task_name == "flame") {
                class_name = flame_cls_to_name(det_result->cls_id);
            } else if (task_name == "smoking") {
                class_name = smoking_cls_to_name(det_result->cls_id);
            } else if (task_name == "face") {
                class_name = face_cls_to_name(det_result->cls_id);
            } else if (task_name == "meter") {
                class_name = meter_cls_to_name(det_result->cls_id);
            }
            
            if (class_name) {
                // 绘制标签
                std::string label = std::string(class_name) + " " + std::to_string(det_result->prop).substr(0, 4);
                putChineseText(frame, label,
                              cv::Point(det_result->box.left, det_result->box.top - 10),
                              color, 0.6, 2);
            }
        }
    }
    
    // 辅助方法
    int analyzeDetectionResults(const object_detect_result_list& results, const std::string& task_name) {
        if (results.count == 0) {
            return 0;  // 无检测结果
        }
        
        // 根据任务类型分析结果
        if (task_name == "helmet") {
            // 安全帽检测：0=有安全帽，1=无安全帽
            for (int i = 0; i < results.count; i++) {
                if (results.results[i].cls_id == 0) {  // 无安全帽
                    return 1;
                }
            }
            return 0;  // 有安全帽
        } else if (task_name == "flame") {
            // 火焰检测：0=正常，1=检测到火焰
            return results.count > 0 ? 1 : 0;
        } else if (task_name == "smoking") {
            // 吸烟检测：0=正常，1=检测到吸烟
            return results.count > 0 ? 1 : 0;
        } else if (task_name == "face") {
            // 人脸检测：0=无人脸，1=检测到人脸
            return results.count > 0 ? 1 : 0;
        } else if (task_name == "meter") {
            // 仪表检测：0=未检测到，1=检测到仪表
            return results.count > 0 ? 1 : 0;
        }
        
        return 0;
    }
    
    void updateDetectionStatus(const std::string& task_name, int class_result) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        
        if (class_result == 1) {
            // 检测到异常
            abnormal_counts_[task_name]++;
            if (task_name == "helmet") {
                detection_results_[task_name] = "Hardhat: No Hardhat Detected!";
            } else if (task_name == "flame") {
                detection_results_[task_name] = "Fire Detected!";
            } else if (task_name == "smoking") {
                detection_results_[task_name] = "Smoking: Detected!";
            } else if (task_name == "face") {
                detection_results_[task_name] = "Face: Detected";
            } else if (task_name == "meter") {
                detection_results_[task_name] = "Meter: Detected";
            }
        } else {
            // 正常状态
            abnormal_counts_[task_name] = 0;
            if (task_name == "helmet") {
                detection_results_[task_name] = "Hardhat: Normal";
            } else if (task_name == "flame") {
                detection_results_[task_name] = "Fire: Normal";
            } else if (task_name == "smoking") {
                detection_results_[task_name] = "Smoking: Normal";
            } else if (task_name == "face") {
                detection_results_[task_name] = "Face: No face";
            } else if (task_name == "meter") {
                detection_results_[task_name] = "Meter: Not Detected";
            }
        }
    }
    
    // 成员变量
    std::atomic<bool> running_;
    std::vector<std::string> task_names_ = {"helmet", "flame", "smoking", "face", "meter"};
    std::map<std::string, std::vector<rknn_app_context_t>> task_contexts_;
    
    // 检测状态管理
    std::map<std::string, std::string> detection_results_;
    std::map<std::string, int> abnormal_counts_;
    std::mutex results_mutex_;
    
    // 摄像头管理
    GlobalCameraManager* camera_manager_;
    std::shared_ptr<SharedCameraManager> camera_;
    
    // 线程管理
    std::vector<std::thread> workers_;
    std::thread display_thread_;
    
    // 显示队列 - 优化版本
    std::queue<DetectionResult> display_queue_;
    std::mutex display_mutex_;
    std::condition_variable display_cv_;  // 添加条件变量优化等待
    static const int MAX_DISPLAY_QUEUE_SIZE = 3;  // 减少队列大小，降低延迟
    
    // 性能监控
    std::atomic<int64_t> total_frames_processed_{0};
    std::atomic<int64_t> total_detection_time_us_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // 帧ID计数
    std::atomic<int64_t> frame_id_{0};
};

int main(int argc, char** argv) {
    printf("=== 多窗口检测系统 ===\n");
    
    if (argc < 6) {
        printf("用法: %s <helmet_model> <flame_model> <smoking_model> <face_model> <meter_model>\n", argv[0]);
        printf("示例: %s ../model/helmet.rknn ../model/fire.rknn ../model/smoking.rknn ../model/retinaface_mob.rknn ../model/yolov8_seg_newer.rknn\n", argv[0]);
        return -1;
    }
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 模型路径
    std::vector<std::string> model_paths = {argv[1], argv[2], argv[3], argv[4], argv[5]};
    
    printf("模型路径:\n");
    std::vector<std::string> task_names = {"helmet", "flame", "smoking", "face", "meter"};
    for (size_t i = 0; i < model_paths.size(); i++) {
        printf("  %s: %s\n", task_names[i].c_str(), model_paths[i].c_str());
    }
    
    // 初始化后处理
    init_post_process();
    init_flame_post_process();
    init_smoking_post_process();
    init_face_post_process();
    init_meter_post_process();
    init_meter_reader();
    
    // 创建多窗口检测系统
    MultiWindowDetector detector(model_paths);
    
    try {
        // 启动检测系统
        detector.start();
        
        // 主循环
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    } catch (const std::exception& e) {
        printf("检测系统异常: %s\n", e.what());
    }
    
    // 清理资源
    cv::destroyAllWindows();
    deinit_post_process();
    deinit_flame_post_process();
    deinit_smoking_post_process();
    deinit_face_post_process();
    deinit_meter_post_process();
    deinit_meter_reader();
    
    // 释放所有摄像头
    GlobalCameraManager::getInstance().releaseAllCameras();
    
    printf("程序退出\n");
    return 0;
}
