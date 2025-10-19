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
static std::atomic<int> g_total_flame_count(0);

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
        
        // 释放RKNN模型
        for (auto& rknn_ctx : rknn_pool_) {
            if (rknn_ctx) {
                release_yolov8_model(rknn_ctx.get());
            }
        }
        
        printf("RKNN线程池已释放\n");
    }
    
    // 提交任务到线程池
    void put(const cv::Mat& frame, int64_t frame_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        task_queue_.push({frame.clone(), frame_id});
        cv_.notify_one();
    }
    
    // 获取结果
    bool get(InferenceResult& result) {
        std::unique_lock<std::mutex> lock(result_mutex_);
        if (result_queue_.empty()) {
            return false;
        }
        result = result_queue_.front();
        result_queue_.pop();
        return true;
    }
    
    // 获取队列大小
    size_t task_queue_size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return task_queue_.size();
    }
    
    size_t result_queue_size() {
        std::unique_lock<std::mutex> lock(result_mutex_);
        return result_queue_.size();
    }

private:
    struct Task {
        cv::Mat frame;
        int64_t frame_id;
    };
    
    int tpes_;
    std::atomic<int> num_;
    std::vector<std::shared_ptr<rknn_app_context_t>> rknn_pool_;
    std::vector<std::thread> threads_;
    
    std::queue<Task> task_queue_;
    std::queue<InferenceResult> result_queue_;
    std::mutex mutex_;
    std::mutex result_mutex_;
    std::condition_variable cv_;
    
    // 工作线程函数
    void worker_thread(int thread_id) {
        printf("工作线程 %d 启动\n", thread_id);
        
        while (g_running) {
            Task task;
            
            // 获取任务
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !task_queue_.empty() || !g_running; });
                
                if (!g_running) break;
                if (task_queue_.empty()) continue;
                
                task = task_queue_.front();
                task_queue_.pop();
            }
            
            // 准备推理数据
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
            
            // 放入结果队列
            {
                std::unique_lock<std::mutex> lock(result_mutex_);
                result_queue_.push({task.frame_id, task.frame, od_results, 
                                  processing_time, thread_id});
            }
        }
        
        printf("工作线程 %d 结束\n", thread_id);
    }
};

// 信号处理函数
void signal_handler(int sig) {
    printf("\n收到退出信号，正在停止...\n");
    g_running = false;
    
    // 请求停止摄像头管理器
    // SharedCameraManager 现在不需要全局停止请求
}

// FPS计算
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

double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    // 检查命令行参数 - 支持单进程和多进程模式
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <model_path> [--multi-process]\n", argv[0]);
        printf("Example: %s ../model/fire.rknn\n", argv[0]);
        printf("Example: %s ../model/fire.rknn --multi-process\n", argv[0]);
        return -1;
    }
    
    const char *model_path = argv[1];
    bool use_multi_process = false;
    
    // 检查是否使用多进程模式
    if (argc == 3 && strcmp(argv[2], "--multi-process") == 0) {
        use_multi_process = true;
    }
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("=== 火焰RKNN线程池实时检测系统 ===\n");
    printf("模型路径: %s\n", model_path);
    printf("运行模式: %s\n", use_multi_process ? "多进程共享摄像头" : "单进程独立摄像头");
    printf("NPU核心数: %d, 线程池大小: %d\n", NPU_CORES, TPEs);
    printf("按 'q' 键退出\n\n");
    
    // 初始化后处理
    init_flame_post_process();
    
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
        
        client_id = pool.registerClient("火焰检测");
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
    
    // 初始化RKNN线程池
    RKNNPoolExecutor rknn_pool(model_path, TPEs);
    
    // 创建显示窗口
    cv::namedWindow("火焰检测", cv::WINDOW_AUTOSIZE);
    
    FPSCounter fps_counter;
    fps_counter.start();
    
    int64_t frame_id = 0;
    std::queue<InferenceResult> frame_buffer;
    int64_t expected_frame_id = 0;
    
    printf("开始检测循环...\n");
    
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
        
        // 提交到RKNN线程池
        rknn_pool.put(frame, frame_id++);
        
        // 获取推理结果
        InferenceResult result;
        if (rknn_pool.get(result)) {
            // 缓冲帧以保持顺序
            frame_buffer.push(result);
            
            // 处理按顺序的帧
            while (!frame_buffer.empty() && frame_buffer.front().frame_id == expected_frame_id) {
                auto current_result = frame_buffer.front();
                frame_buffer.pop();
                
                // 统计检测结果
                int flame_count = 0;
                for (int i = 0; i < current_result.results.count; i++) {
                    if (current_result.results.results[i].cls_id == 0) {  // 火焰类别
                        flame_count++;
                    }
                }
                
                // 更新全局统计
                g_total_flame_count.fetch_add(flame_count);
                g_total_frames.fetch_add(1);
                
                // 更新FPS
                fps_counter.update();
                double fps = fps_counter.get_fps();
                
                // 验证帧数据
                if (current_result.frame.empty() || current_result.frame.rows <= 0 || current_result.frame.cols <= 0) {
                    printf("警告: 跳过无效帧 (frame_id=%ld)\n", current_result.frame_id);
                    expected_frame_id++;
                    continue;
                }
                
                // 绘制检测结果
                cv::Mat display_frame = current_result.frame.clone();
                char text[256];
                
                // 绘制检测框
                for (int i = 0; i < current_result.results.count; i++) {
                    object_detect_result *det_result = &(current_result.results.results[i]);
                    
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;
                    
                    // 火焰检测使用红色
                    cv::Scalar box_color = cv::Scalar(0, 0, 255);
                    cv::Scalar text_color = cv::Scalar(0, 0, 255);
                    
                    // 绘制检测框
                    cv::rectangle(display_frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 3);
                    
                    // 绘制标签
                    sprintf(text, "%s %.1f%%", flame_cls_to_name(det_result->cls_id), det_result->prop * 100);
                    cv::putText(display_frame, text, cv::Point(x1, y1 - 20), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
                }
                
                // 绘制FPS信息
                sprintf(text, "FPS: %.1f", fps);
                cv::putText(display_frame, text, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                
                // 绘制统计信息
                sprintf(text, "Flame Count: %d", g_total_flame_count.load());
                cv::putText(display_frame, text, cv::Point(10, 70), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                
                // 绘制线程池信息
                sprintf(text, "Core: %d | Processing: %.1fms", 
                       current_result.core_id, current_result.processing_time);
                cv::putText(display_frame, text, cv::Point(10, 110), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                
                // 绘制队列状态
                sprintf(text, "队列: %zu/%zu",
                        rknn_pool.task_queue_size(), rknn_pool.result_queue_size());
                cv::putText(display_frame, text, cv::Point(10, 150), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                
                // 绘制控制提示
                cv::putText(display_frame, "按 'q' 键退出", 
                           cv::Point(10, display_frame.rows - 20), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                // 显示结果
                cv::imshow("火焰检测", display_frame);
                
                expected_frame_id++;
            }
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
            
            sprintf(text, "Task Queue: %zu, Result Queue: %zu", 
                   rknn_pool.task_queue_size(), rknn_pool.result_queue_size());
            cv::putText(display_frame, text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            cv::putText(display_frame, "按 'q' 键退出", 
                       cv::Point(10, display_frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            
            cv::imshow("火焰检测", display_frame);
        }
        
        // 处理键盘输入
        char key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 27) {  // 'q' 或 ESC 键退出
            g_running = false;
            break;
        }
        
        // 每100帧输出一次统计信息
        if (g_total_frames.load() % 100 == 0) {
            printf("已处理 %d 帧, 总检测: 火焰=%d, FPS=%.1f\n",
                   g_total_frames.load(), g_total_flame_count.load(), fps_counter.get_fps());
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
    deinit_flame_post_process();
    
    printf("\n=== 检测统计 ===\n");
    printf("总处理帧数: %d\n", g_total_frames.load());
    printf("总检测到火焰: %d\n", g_total_flame_count.load());
    
    printf("程序退出\n");
    return 0;
}
