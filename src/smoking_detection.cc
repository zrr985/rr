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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <vector>

#include "postprocess.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "shared_camera_manager.h"
#include "camera_resource_pool.h"

// RKNN线程池配置
#define NPU_CORES 3
#define TPEs 3  // 线程池执行器数量，对应3个NPU核心

// 全局变量
static std::atomic<bool> g_running(true);
static std::atomic<int> g_total_frames(0);
static std::atomic<int> g_total_smoking_count(0);

// 推理结果结构体
struct InferenceResult {
    int64_t frame_id;
    cv::Mat frame;
    object_detect_result_list results;
    double processing_time;
    int core_id;
};

// RKNN线程池类
class RKNNPoolExecutor {
public:
    RKNNPoolExecutor(const char* model_path, int tpes) 
        : tpes_(tpes), num_(0) {
        // 初始化RKNN池
        rknn_pool_.resize(tpes_);
        for (int i = 0; i < tpes_; i++) {
            rknn_pool_[i] = std::make_shared<rknn_app_context_t>();
            memset(rknn_pool_[i].get(), 0, sizeof(rknn_app_context_t));
            
            // 初始化模型，绑定到不同的NPU核心
            int ret = init_yolov8_model(model_path, rknn_pool_[i].get());
            if (ret != 0) {
                printf("初始化RKNN模型 %d 失败! ret=%d\n", i, ret);
                exit(-1);
            }
            
            // 设置NPU核心掩码
            if (i == 0) {
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_0);
                printf("RKNN实例 %d 绑定到NPU核心0\n", i);
            } else if (i == 1) {
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_1);
                printf("RKNN实例 %d 绑定到NPU核心1\n", i);
            } else if (i == 2) {
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_2);
                printf("RKNN实例 %d 绑定到NPU核心2\n", i);
            }
        }
        
        // 初始化线程池
        for (int i = 0; i < tpes_; i++) {
            threads_.emplace_back(&RKNNPoolExecutor::worker_thread, this, i);
        }
        
        printf("RKNN线程池初始化完成，TPEs=%d\n", tpes_);
    }
    
    ~RKNNPoolExecutor() {
        // 停止所有线程
        g_running = false;
        cv_.notify_all();
        
        // 等待所有线程结束
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // 释放RKNN上下文
        for (int i = 0; i < tpes_; i++) {
            if (rknn_pool_[i]) {
                release_yolov8_model(rknn_pool_[i].get());
            }
        }
        
        printf("RKNN线程池已释放\n");
    }
    
    void put_task(const InferenceResult& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        task_queue_.push(task);
        cv_.notify_one();
    }
    
    bool get_result(InferenceResult& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (result_queue_.empty()) {
            return false;
        }
        result = result_queue_.front();
        result_queue_.pop();
        return true;
    }
    
    int task_queue_size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return task_queue_.size();
    }
    
    int result_queue_size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return result_queue_.size();
    }
    
private:
    void worker_thread(int thread_id) {
        printf("工作线程 %d 启动 (线程ID: %lu)\n", thread_id, pthread_self());
        
        while (g_running) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !task_queue_.empty() || !g_running; });
            
            if (!g_running) {
                break;
            }
            
            if (task_queue_.empty()) {
                continue;
            }
            
            InferenceResult task = task_queue_.front();
            task_queue_.pop();
            lock.unlock();
            
            // 准备输入图像
            image_buffer_t src_image;
            src_image.width = task.frame.cols;
            src_image.height = task.frame.rows;
            src_image.format = IMAGE_FORMAT_RGB888;
            src_image.virt_addr = task.frame.data;
            
            // 执行推理
            object_detect_result_list od_results;
            struct timeval start, end;
            gettimeofday(&start, NULL);
            
            int ret = inference_yolov8_model(rknn_pool_[thread_id].get(), &src_image, &od_results);
            if (ret != 0) {
                printf("线程 %d 推理失败! ret=%d\n", thread_id, ret);
                continue;
            }
            
            gettimeofday(&end, NULL);
            double processing_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                                    (end.tv_usec - start.tv_usec) / 1000.0;
            
            // 更新结果
            task.results = od_results;
            task.processing_time = processing_time;
            task.core_id = thread_id % NPU_CORES;
            
            // 统计吸烟检测数量
            for (int i = 0; i < od_results.count; i++) {
                if (od_results.results[i].cls_id == 2) { // smoking类别
                    g_total_smoking_count.fetch_add(1);
                }
            }
            
            // 将结果放入结果队列
            lock.lock();
            result_queue_.push(task);
            lock.unlock();
        }
        
        printf("工作线程 %d 退出\n", thread_id);
    }
    
    int tpes_;
    std::atomic<int> num_;
    std::vector<std::shared_ptr<rknn_app_context_t>> rknn_pool_;
    std::vector<std::thread> threads_;
    
    std::queue<InferenceResult> task_queue_;
    std::queue<InferenceResult> result_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// 信号处理函数
void signal_handler(int sig) {
    printf("\n收到退出信号，正在停止...\n");
    g_running = false;
    
    // 请求停止摄像头管理器
    // SharedCameraManager 现在不需要全局停止请求
}

// FPS计数器
class FPSCounter {
public:
    FPSCounter() : frame_count_(0), start_time_(0) {}
    
    void start() {
        frame_count_ = 0;
        start_time_ = get_current_time_ms();
    }
    
    double get_fps() {
        double current_time = get_current_time_ms();
        double elapsed = (current_time - start_time_) / 1000.0;
        if (elapsed > 0) {
            return frame_count_ / elapsed;
        }
        return 0.0;
    }
    
    void update() {
        frame_count_++;
    }
    
private:
    int frame_count_;
    double start_time_;
    
    double get_current_time_ms() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }
};

int main(int argc, char* argv[]) {
    // 检查命令行参数 - 支持单进程和多进程模式
    if (argc < 2 || argc > 3) {
        printf("用法: %s <model_path> [--multi-process]\n", argv[0]);
        printf("示例: %s ../model/smoking.rknn\n", argv[0]);
        printf("示例: %s ../model/smoking.rknn --multi-process\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    bool use_multi_process = false;
    
    // 检查是否使用多进程模式
    if (argc == 3 && strcmp(argv[2], "--multi-process") == 0) {
        use_multi_process = true;
    }
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("=== 吸烟RKNN线程池实时检测系统 ===\n");
    printf("模型路径: %s\n", model_path);
    printf("运行模式: %s\n", use_multi_process ? "多进程共享摄像头" : "单进程独立摄像头");
    printf("NPU核心数: %d, 线程池大小: %d\n", NPU_CORES, TPEs);
    printf("按 'q' 键退出\n\n");
    
    // 初始化后处理
    init_smoking_post_process();
    
    // 根据模式选择摄像头管理方式
    std::string client_id;
    if (use_multi_process) {
        // 多进程模式 - 使用摄像头资源池
        printf("正在初始化多进程摄像头资源池...\n");
        CameraResourcePool& pool = CameraResourcePool::getInstance();
        
        // 初始化摄像头资源池 - 添加重试和等待机制
        int retry_count = 0;
        const int max_retries = 5;
        bool init_success = false;
        
        while (retry_count < max_retries && !init_success) {
            if (pool.initialize()) {
                init_success = true;
            } else {
                retry_count++;
                if (retry_count < max_retries) {
                    printf("摄像头资源池初始化失败，等待%d秒后重试... (第%d次)\n", 3, retry_count);
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                }
            }
        }
        
        if (!init_success) {
            printf("摄像头资源池初始化失败，已重试%d次\n", max_retries);
            return -1;
        }
        
        client_id = pool.registerClient("吸烟检测");
        if (client_id.empty()) {
            printf("客户端注册失败\n");
            return -1;
        }
        
        printf("客户端已注册，ID: %s\n", client_id.c_str());
    } else {
        // 单进程模式 - 使用共享摄像头管理器
        printf("正在初始化单进程摄像头管理器...\n");
    }
    
    // 单进程模式需要初始化摄像头管理器
    SharedCameraManager camera_manager;
    if (!use_multi_process) {
        if (!camera_manager.initCamera()) {
            printf("摄像头初始化失败\n");
            return -1;
        }
        camera_manager.startFrameCapture();
    }
    
    // 等待摄像头初始化
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // 创建RKNN线程池
    RKNNPoolExecutor rknn_pool(model_path, TPEs);
    
    // 创建显示窗口
    cv::namedWindow("吸烟检测", cv::WINDOW_AUTOSIZE);
    
    FPSCounter fps_counter;
    fps_counter.start();
    
    int64_t frame_id = 0;
    
    printf("开始实时检测...\n");
    
    while (g_running) {
        // 根据模式选择不同的帧获取方式
        cv::Mat frame;
        bool frame_available = false;
        
        if (use_multi_process) {
            // 多进程模式 - 从摄像头资源池获取帧
            std::shared_ptr<cv::Mat> frame_ptr;
            if (CameraPool::getFrame(client_id, frame_ptr, 100)) {
                frame = *frame_ptr;
                frame_available = true;
            }
        } else {
            // 单进程模式 - 从共享摄像头管理器获取帧
            if (camera_manager.getFrameFromQueue(frame)) {
                frame_available = true;
            }
        }
        
        // 检查是否成功获取帧
        if (!frame_available) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        // 验证帧数据完整性
        if (frame.rows <= 0 || frame.cols <= 0 || frame.channels() <= 0) {
            printf("警告: 收到无效帧数据 (rows=%d, cols=%d, channels=%d)\n", 
                   frame.rows, frame.cols, frame.channels());
            continue;
        }
        
        // 创建推理任务
        InferenceResult task;
        task.frame_id = frame_id++;
        task.frame = frame;
        
        // 提交任务到线程池
        rknn_pool.put_task(task);
        
        // 处理推理结果
        InferenceResult result;
        if (rknn_pool.get_result(result)) {
            // 更新全局帧计数
            g_total_frames.fetch_add(1);
            // 绘制检测结果
            cv::Mat display_frame = result.frame.clone();
            char text[256];
            
            // 分析当前帧的检测结果
            bool has_face = false;
            bool has_cigarette = false;
            bool has_smoking = false;
            
            // 统计各类别检测结果
            for (int i = 0; i < result.results.count; i++) {
                object_detect_result *det_result = &(result.results.results[i]);
                if (det_result->cls_id == 0) { // cigarette
                    has_cigarette = true;
                } else if (det_result->cls_id == 1) { // face
                    has_face = true;
                } else if (det_result->cls_id == 2) { // smoking
                    has_smoking = true;
                }
            }
            
            // 记录face和cigarette的组合检测历史
            static std::vector<int> face_cigarette_history;
            static std::vector<int> smoking_detection_history;
            static const int window_size = 10;
            static const double detection_threshold = 0.7;
            
            // 记录face和cigarette的组合检测结果
            int face_cigarette_combo = (has_face && has_cigarette) ? 1 : 0;
            face_cigarette_history.push_back(face_cigarette_combo);
            
            // 保持历史记录的长度不超过窗口大小
            if (face_cigarette_history.size() > window_size) {
                face_cigarette_history.erase(face_cigarette_history.begin());
            }
            
            // 计算face和cigarette组合的检测比例
            double face_cigarette_ratio = 0.0;
            if (!face_cigarette_history.empty()) {
                int sum = 0;
                for (int val : face_cigarette_history) {
                    sum += val;
                }
                face_cigarette_ratio = (double)sum / face_cigarette_history.size();
            }
            
            // 判断是否为吸烟行为
            // 条件1：直接检测到smoking类别
            // 条件2：同时检测到face和cigarette，且持续一定时间
            bool smoking_detected = has_smoking || (face_cigarette_ratio >= 0.6 && face_cigarette_history.size() >= 5);
            
            // 记录吸烟检测结果
            smoking_detection_history.push_back(smoking_detected ? 1 : 0);
            
            // 保持历史记录的长度不超过窗口大小
            if (smoking_detection_history.size() > window_size) {
                smoking_detection_history.erase(smoking_detection_history.begin());
            }
            
            // 计算窗口内的检测比例
            double detection_ratio = 0.0;
            if (!smoking_detection_history.empty()) {
                int sum = 0;
                for (int val : smoking_detection_history) {
                    sum += val;
                }
                detection_ratio = (double)sum / smoking_detection_history.size();
            }
            
            // 根据检测比例判断是否在吸烟
            bool final_smoking_detected = false;
            if (smoking_detection_history.size() >= window_size) {
                final_smoking_detected = (detection_ratio >= detection_threshold);
            }
            
            // 绘制检测框 - 根据防误检逻辑决定是否显示
            if (final_smoking_detected) {
                // 如果防误检逻辑判断为吸烟，绘制所有相关检测框
                for (int i = 0; i < result.results.count; i++) {
                    object_detect_result *det_result = &(result.results.results[i]);
                    
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;
                    
                    // 根据类别设置不同颜色
                    cv::Scalar box_color, text_color;
                    const char* class_name;
                    
                    if (det_result->cls_id == 0) { // cigarette
                        box_color = cv::Scalar(0, 255, 255); // 黄色
                        text_color = cv::Scalar(0, 255, 255);
                        class_name = "cigarette";
                    } else if (det_result->cls_id == 1) { // face
                        box_color = cv::Scalar(0, 255, 0); // 绿色
                        text_color = cv::Scalar(0, 255, 0);
                        class_name = "face";
                    } else if (det_result->cls_id == 2) { // smoking
                        box_color = cv::Scalar(0, 0, 255); // 红色
                        text_color = cv::Scalar(0, 0, 255);
                        class_name = "smoking";
                    } else {
                        continue; // 跳过未知类别
                    }
                    
                    // 绘制检测框
                    cv::rectangle(display_frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 3);
                    
                    // 绘制标签
                    sprintf(text, "%s %.1f%%", class_name, det_result->prop * 100);
                    cv::putText(display_frame, text, cv::Point(x1, y1 - 20), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
                }
            } else {
                // 如果防误检逻辑判断为不吸烟，只显示smoking类别的检测框
                for (int i = 0; i < result.results.count; i++) {
                    object_detect_result *det_result = &(result.results.results[i]);
                    
                    // 只显示smoking类别 (cls_id == 2)
                    if (det_result->cls_id != 2) {
                        continue;
                    }
                    
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;
                    
                    // smoking类别使用红色
                    cv::Scalar box_color = cv::Scalar(0, 0, 255); // 红色
                    cv::Scalar text_color = cv::Scalar(0, 0, 255);
                    
                    // 绘制检测框
                    cv::rectangle(display_frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 3);
                    
                    // 绘制标签
                    const char* class_name = "smoking";
                    sprintf(text, "%s %.1f%%", class_name, det_result->prop * 100);
                    cv::putText(display_frame, text, cv::Point(x1, y1 - 20), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
                }
            }
            
            // 绘制FPS信息
            sprintf(text, "FPS: %.1f", fps_counter.get_fps());
            cv::putText(display_frame, text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            // 绘制统计信息
            sprintf(text, "Smoking: %d", g_total_smoking_count.load());
            cv::putText(display_frame, text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            
            // 绘制防误检逻辑状态
            sprintf(text, "Face+Cig: %.1f%% (%d/%d)", 
                   face_cigarette_ratio * 100, 
                   (int)face_cigarette_history.size() > 0 ? face_cigarette_history.back() : 0,
                   (int)face_cigarette_history.size());
            cv::putText(display_frame, text, cv::Point(10, 110), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
            
            // 绘制最终检测状态
            sprintf(text, "Final: %s (%.1f%%)", 
                   final_smoking_detected ? "YES" : "NO",
                   detection_ratio * 100);
            cv::Scalar status_color = final_smoking_detected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::putText(display_frame, text, cv::Point(10, 140), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);
            
            // 绘制线程池信息
            sprintf(text, "Core: %d | Processing: %.1fms", 
                   result.core_id, result.processing_time);
            cv::putText(display_frame, text, cv::Point(10, 170), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            // 绘制队列状态
            sprintf(text, "Queue: %d/%d", rknn_pool.task_queue_size(), rknn_pool.result_queue_size());
            cv::putText(display_frame, text, cv::Point(10, 200), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            
            // 绘制控制提示
            cv::putText(display_frame, "Press 'q' to quit", 
                       cv::Point(10, display_frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            // 显示结果
            cv::imshow("吸烟检测", display_frame);
            
            // 更新FPS
            fps_counter.update();
        } else {
            // 如果没有推理结果，显示原始帧
            static int no_result_count = 0;
            no_result_count++;
            if (no_result_count % 30 == 0) {  // 每30帧显示一次
                printf("等待推理结果中... (已提交 %ld 帧)\n", frame_id);
            }
            
            // 显示原始帧（带简单信息）
            cv::Mat display_frame = frame.clone();
            char text[256];
            sprintf(text, "Waiting for inference... Frame: %ld", frame_id);
            cv::putText(display_frame, text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            
            sprintf(text, "Task Queue: %d, Result Queue: %d", 
                   rknn_pool.task_queue_size(), rknn_pool.result_queue_size());
            cv::putText(display_frame, text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            cv::putText(display_frame, "Press 'q' to quit", 
                       cv::Point(10, display_frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            cv::imshow("吸烟检测", display_frame);
        }
        
        // 检查按键
        char key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' 或 ESC
            printf("用户退出\n");
            break;
        }
        
        // 定期打印统计信息
        static int print_counter = 0;
        if (++print_counter % 100 == 0) {
            printf("已处理 %d 帧, 总检测: 吸烟=%d, FPS=%.1f\n",
                   g_total_frames.load(), g_total_smoking_count.load(), fps_counter.get_fps());
        }
    }
    
    // 清理资源
    printf("正在清理资源...\n");
    
    // 根据模式进行不同的资源清理
    if (use_multi_process) {
        // 多进程模式 - 注销客户端
        if (!client_id.empty()) {
            CameraPool::unregisterClient(client_id);
            printf("客户端已注销: %s\n", client_id.c_str());
        }
    } else {
        // 单进程模式 - 停止帧捕获并释放摄像头
        camera_manager.stopFrameCapture();
        camera_manager.release();
        printf("摄像头管理器已停止\n");
    }
    
    cv::destroyAllWindows();
    deinit_smoking_post_process();
    
    printf("\n=== 检测统计 ===\n");
    printf("总处理帧数: %d\n", g_total_frames.load());
    printf("总检测到吸烟: %d\n", g_total_smoking_count.load());
    printf("程序退出\n");
    
    return 0;
}
