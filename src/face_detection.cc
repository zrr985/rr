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

// RKNN线程池配置
#define NPU_CORES 3
#define TPEs 3  // 线程池执行器数量，对应3个NPU核心

// 全局变量
static std::atomic<bool> g_running(true);
static std::atomic<int> g_total_frames(0);
static std::atomic<int> g_total_face_count(0);

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
            
            // 统计人脸检测数量
            for (int i = 0; i < od_results.count; i++) {
                if (od_results.results[i].cls_id == 0) { // face类别
                    g_total_face_count.fetch_add(1);
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
    if (argc != 2) {
        printf("用法: %s <model_path>\n", argv[0]);
        printf("示例: %s ../model/face.rknn\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("=== 人脸检测系统 ===\n");
    printf("模型路径: %s\n", model_path);
    printf("按 'q' 键退出\n\n");
    
    // 初始化后处理
    int ret = init_face_post_process();
    if (ret != 0) {
        printf("初始化人脸检测后处理失败! ret=%d\n", ret);
        return -1;
    }
    
    // 初始化全局摄像头管理器 - 支持多窗口检测
    GlobalCameraManager& global_camera_manager = GlobalCameraManager::getInstance();
    auto camera_manager = global_camera_manager.getCamera(0);
    if (!camera_manager) {
        printf("初始化摄像头失败!\n");
        return -1;
    }
    
    // 启动帧捕获线程 - 支持多窗口检测
    camera_manager->startFrameCapture();
    printf("摄像头帧捕获线程已启动，支持多窗口检测\n");
    
    // 创建RKNN线程池
    RKNNPoolExecutor pool(model_path, TPEs);
    
    // 创建显示窗口
    cv::namedWindow("人脸检测", cv::WINDOW_AUTOSIZE);
    
    FPSCounter fps_counter;
    fps_counter.start();
    
    int64_t frame_id = 0;
    
    printf("开始实时检测...\n");
    
    while (g_running) {
        // 从摄像头队列获取帧 - 支持多窗口检测
        std::shared_ptr<cv::Mat> frame_ptr;
        if (!camera_manager->getFrameFromQueue(frame_ptr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (!frame_ptr || frame_ptr->empty()) {
            continue;
        }
        
        // 创建推理任务 - 使用智能指针避免拷贝
        InferenceResult task;
        task.frame_id = frame_id++;
        task.frame = *frame_ptr;
        
        // 提交任务到线程池
        pool.put_task(task);
        
        // 处理推理结果
        InferenceResult current_result;
        if (pool.get_result(current_result)) {
            // 更新全局帧计数
            g_total_frames.fetch_add(1);
            // 绘制检测结果
            cv::Mat display_frame = current_result.frame.clone();
            char text[256];
            
            // 绘制检测框
            for (int i = 0; i < current_result.results.count; i++) {
                object_detect_result* det_result = &current_result.results.results[i];
                
                // 绘制边界框
                cv::Rect rect(det_result->box.left, det_result->box.top,
                             det_result->box.right - det_result->box.left,
                             det_result->box.bottom - det_result->box.top);
                cv::rectangle(display_frame, rect, cv::Scalar(0, 255, 0), 2);
                
                // 绘制标签和置信度
                sprintf(text, "%s: %.2f", face_cls_to_name(det_result->cls_id), det_result->prop);
                cv::putText(display_frame, text, cv::Point(det_result->box.left, det_result->box.top - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
            
            // 绘制FPS信息
            sprintf(text, "FPS: %.1f", fps_counter.get_fps());
            cv::putText(display_frame, text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            // 绘制统计信息
            sprintf(text, "人脸: %d", g_total_face_count.load());
            cv::putText(display_frame, text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            
            // 绘制线程池信息
            sprintf(text, "Core: %d | Processing: %.1fms", 
                   current_result.core_id, current_result.processing_time);
            cv::putText(display_frame, text, cv::Point(10, 110), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            // 绘制队列状态
            sprintf(text, "队列: %d/%d", pool.task_queue_size(), pool.result_queue_size());
            cv::putText(display_frame, text, cv::Point(10, 150), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            
            // 绘制控制提示
            cv::putText(display_frame, "按 'q' 键退出", 
                       cv::Point(10, display_frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            // 显示结果
            cv::imshow("人脸检测", display_frame);
            
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
            cv::Mat display_frame = frame_ptr->clone();
            char text[256];
            sprintf(text, "Waiting for inference... Frame: %ld", frame_id);
            cv::putText(display_frame, text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            
            sprintf(text, "Task Queue: %d, Result Queue: %d", 
                   pool.task_queue_size(), pool.result_queue_size());
            cv::putText(display_frame, text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            cv::putText(display_frame, "按 'q' 键退出", 
                       cv::Point(10, display_frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            cv::imshow("人脸检测", display_frame);
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
            printf("已处理 %d 帧, 总检测: 人脸=%d, FPS=%.1f\n",
                   g_total_frames.load(), g_total_face_count.load(), fps_counter.get_fps());
        }
    }
    
    // 清理资源
    cv::destroyAllWindows();
    
    // 停止帧捕获并释放摄像头
    camera_manager->stopFrameCapture();
    global_camera_manager.forceReleaseAllCameras();
    
    deinit_face_post_process();
    
    printf("\n=== 检测统计 ===\n");
    printf("总处理帧数: %d\n", g_total_frames.load());
    printf("总检测到人脸: %d\n", g_total_face_count.load());
    printf("程序退出\n");
    
    return 0;
}
