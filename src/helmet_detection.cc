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

// 全局变量 - 使用原子类型确保多线程安全
static std::atomic<bool> g_running(true);              // 全局运行标志，控制程序主循环和所有线程的生命周期
static std::atomic<int> g_total_frames(0);             // 总处理帧数统计，记录系统处理的所有帧数
static std::atomic<int> g_total_helmet_count(0);       // 安全帽检测计数，统计检测到的安全帽总数
static std::atomic<int> g_total_no_helmet_count(0);    // 无安全帽检测计数，统计检测到的无安全帽总数

// 推理结果结构体 - 存储单次推理的完整结果信息
struct InferenceResult {
    int64_t frame_id;                    // 帧ID，用于跟踪和同步不同线程的处理结果
    cv::Mat frame;                       // 原始输入帧图像，用于后续显示和绘制
    object_detect_result_list results;   // 检测结果列表，包含所有检测到的目标信息
    double processing_time;              // 推理处理时间（毫秒），用于性能监控
    int core_id;                        // 使用的NPU核心ID，标识哪个核心处理了此帧
};

// RKNN线程池类 - 管理多个RKNN实例和工作线程，实现并行推理
class RKNNPoolExecutor {
public:
    // 构造函数 - 初始化RKNN线程池，创建多个模型实例和工作线程
    // 参数：model_path-模型文件路径，tpes-线程池执行器数量
    RKNNPoolExecutor(const char* model_path, int tpes) 
        : tpes_(tpes), num_(0) {                                              // 初始化成员变量：线程数和计数器
        // 初始化RKNN池
        rknn_pool_.resize(tpes_);                                             // 调整RKNN池大小为线程数
        for (int i = 0; i < tpes_; i++) {                                    // 为每个线程创建RKNN实例
            rknn_pool_[i] = std::make_shared<rknn_app_context_t>();          // 创建RKNN应用上下文的智能指针
            memset(rknn_pool_[i].get(), 0, sizeof(rknn_app_context_t));      // 将上下文结构体清零初始化
            
            // 初始化模型，绑定到不同的NPU核心
            int ret = init_yolov8_model(model_path, rknn_pool_[i].get());     // 使用模型文件初始化YOLOv8模型
            if (ret != 0) {                                                   // 检查模型初始化是否成功
                printf("初始化RKNN模型 %d 失败! ret=%d\n", i, ret);            // 输出错误信息
                exit(-1);                                                     // 初始化失败，退出程序
            }
            
            // 设置NPU核心掩码 - 将不同的RKNN实例绑定到不同的NPU核心
            if (i == 0) {                                                     // 第一个实例绑定到核心0
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_0); // 设置NPU核心0掩码
                printf("RKNN实例 %d 绑定到NPU核心0\n", i);                     // 输出绑定信息
            } else if (i == 1) {                                             // 第二个实例绑定到核心1
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_1); // 设置NPU核心1掩码
                printf("RKNN实例 %d 绑定到NPU核心1\n", i);                     // 输出绑定信息
            } else if (i == 2) {                                             // 第三个实例绑定到核心2
                rknn_set_core_mask(rknn_pool_[i]->rknn_ctx, RKNN_NPU_CORE_2); // 设置NPU核心2掩码
                printf("RKNN实例 %d 绑定到NPU核心2\n", i);                     // 输出绑定信息
            }
        }
        
        // 初始化线程池 - 创建工作线程
        for (int i = 0; i < tpes_; i++) {                                    // 创建指定数量的工作线程
            threads_.emplace_back(&RKNNPoolExecutor::worker_thread, this, i); // 创建线程并绑定工作函数
        }
        
        printf("RKNN线程池初始化完成，TPEs=%d\n", tpes_);                     // 输出初始化完成信息
    }
    
    // 析构函数 - 清理所有资源，确保线程和模型正确释放
    ~RKNNPoolExecutor() {
        // 停止所有线程
        g_running = false;                                                    // 设置全局停止标志，通知所有工作线程停止
        cv_.notify_all();                                                     // 唤醒所有等待条件变量的工作线程
        
        // 等待所有线程结束
        for (auto& thread : threads_) {                                       // 遍历所有工作线程
            if (thread.joinable()) {                                          // 检查线程是否可以join
                thread.join();                                                // 等待线程结束，确保线程安全退出
            }
        }
        
        // 释放RKNN模型
        for (auto& rknn_ctx : rknn_pool_) {                                   // 遍历所有RKNN上下文
            if (rknn_ctx) {                                                   // 检查上下文是否有效
                release_yolov8_model(rknn_ctx.get());                         // 释放YOLOv8模型占用的内存和资源
            }
        }
        
        printf("RKNN线程池已释放\n");                                          // 输出资源释放完成信息
    }
    
    // 提交任务到线程池 - 将帧数据添加到任务队列
    // 参数：frame-输入帧图像，frame_id-帧ID用于跟踪
    void put(const cv::Mat& frame, int64_t frame_id) {
        std::unique_lock<std::mutex> lock(mutex_);                            // 获取任务队列的互斥锁
        task_queue_.push({frame.clone(), frame_id});                          // 克隆帧数据并添加到任务队列
        cv_.notify_one();                                                     // 通知一个等待的工作线程有新任务
    }
    
    // 获取结果 - 从结果队列中获取推理结果
    // 参数：result-输出结果的引用
    // 返回值：成功获取返回true，队列为空返回false
    bool get(InferenceResult& result) {
        std::unique_lock<std::mutex> lock(result_mutex_);                     // 获取结果队列的互斥锁
        if (result_queue_.empty()) {                                          // 检查结果队列是否为空
            return false;                                                     // 队列为空，返回false
        }
        result = result_queue_.front();                                       // 获取队列前端的结果
        result_queue_.pop();                                                  // 从队列中移除已获取的结果
        return true;                                                          // 成功获取结果，返回true
    }
    
    // 获取队列大小 - 返回任务队列中的任务数量
    // 返回值：当前任务队列的大小
    size_t task_queue_size() {
        std::unique_lock<std::mutex> lock(mutex_);                            // 获取任务队列的互斥锁
        return task_queue_.size();                                            // 返回任务队列的当前大小
    }
    
    // 获取结果队列大小 - 返回结果队列中的结果数量
    // 返回值：当前结果队列的大小
    size_t result_queue_size() {
        std::unique_lock<std::mutex> lock(result_mutex_);                     // 获取结果队列的互斥锁
        return result_queue_.size();                                          // 返回结果队列的当前大小
    }

private:
    // 任务结构体 - 存储单个推理任务的信息
    struct Task {
        cv::Mat frame;                        // 待处理的帧图像数据
        int64_t frame_id;                     // 帧ID，用于跟踪和同步
    };
    
    // 成员变量
    int tpes_;                                                                // 线程池执行器数量
    std::atomic<int> num_;                                                    // 原子计数器，用于线程同步
    std::vector<std::shared_ptr<rknn_app_context_t>> rknn_pool_;             // RKNN上下文池，每个线程对应一个上下文
    std::vector<std::thread> threads_;                                        // 工作线程向量，存储所有工作线程
    
    std::queue<Task> task_queue_;                                             // 任务队列，存储待处理的推理任务
    std::queue<InferenceResult> result_queue_;                                // 结果队列，存储已完成的推理结果
    std::mutex mutex_;                                                        // 任务队列互斥锁，保护任务队列的线程安全访问
    std::mutex result_mutex_;                                                 // 结果队列互斥锁，保护结果队列的线程安全访问
    std::condition_variable cv_;                                              // 条件变量，用于工作线程间的同步通信
    
    // 工作线程函数 - 每个工作线程的主要执行逻辑
    // 参数：thread_id-线程ID，用于标识线程和选择对应的RKNN实例
    void worker_thread(int thread_id) {
        printf("工作线程 %d 启动\n", thread_id);                               // 输出线程启动信息
        
        while (g_running) {                                                   // 主工作循环，直到收到停止信号
            Task task;                                                        // 创建任务变量存储当前处理的任务
            
            // 获取任务 - 从任务队列中获取待处理的任务
            {
                std::unique_lock<std::mutex> lock(mutex_);                    // 获取任务队列的互斥锁
                cv_.wait(lock, [this] { return !task_queue_.empty() || !g_running; }); // 等待任务队列非空或收到停止信号
                
                if (!g_running) break;                                        // 如果收到停止信号，退出循环
                if (task_queue_.empty()) continue;                            // 如果队列为空，继续等待
                
                task = task_queue_.front();                                   // 获取队列前端的任务
                task_queue_.pop();                                            // 从队列中移除已获取的任务
            }
            
            // 准备推理数据 - 将OpenCV Mat格式转换为RKNN输入格式
            image_buffer_t src_image;                                         // 创建RKNN图像缓冲区结构体
            src_image.width = task.frame.cols;                                // 设置图像宽度
            src_image.height = task.frame.rows;                               // 设置图像高度
            src_image.format = IMAGE_FORMAT_RGB888;                           // 设置图像格式为RGB888
            src_image.virt_addr = task.frame.data;                            // 设置图像数据指针
            
            // 执行推理 - 使用对应的RKNN实例进行推理
            object_detect_result_list od_results;                             // 创建检测结果列表
            struct timeval start, end;                                        // 创建时间结构体用于计时
            gettimeofday(&start, NULL);                                       // 记录推理开始时间
            
            int ret = inference_yolov8_model(rknn_pool_[thread_id].get(), &src_image, &od_results); // 执行YOLOv8推理
            if (ret != 0) {                                                   // 检查推理是否成功
                printf("线程 %d 推理失败! ret=%d\n", thread_id, ret);          // 输出推理失败信息
                continue;                                                     // 跳过当前任务，继续处理下一个
            }
            
            gettimeofday(&end, NULL);                                         // 记录推理结束时间
            double processing_time = (end.tv_sec - start.tv_sec) * 1000.0 +  // 计算处理时间（毫秒）
                                    (end.tv_usec - start.tv_usec) / 1000.0;   // 秒部分转毫秒 + 微秒部分转毫秒
            
            // 放入结果队列 - 将推理结果添加到结果队列
            {
                std::unique_lock<std::mutex> lock(result_mutex_);             // 获取结果队列的互斥锁
                result_queue_.push({task.frame_id, task.frame, od_results,    // 创建推理结果并添加到队列
                                  processing_time, thread_id});               // 包含帧ID、原始帧、检测结果、处理时间、线程ID
            }
        }
        
        printf("工作线程 %d 结束\n", thread_id);                               // 输出线程结束信息
    }
};

// 信号处理函数 - 处理程序退出信号，实现优雅退出
// 参数：sig-接收到的信号类型（如SIGINT、SIGTERM等）
void signal_handler(int sig) {
    printf("\n收到退出信号，正在停止...\n");                                      // 输出退出信息到控制台
    g_running = false;                                                            // 设置全局停止标志为false，通知所有线程停止运行
    
    // 请求停止摄像头管理器
    // SharedCameraManager 现在不需要全局停止请求，会在析构时自动清理
}

// FPS计算器类 - 用于计算和监控系统帧率性能
class FPSCounter {
public:
    // 构造函数 - 初始化FPS计算器
    FPSCounter() : frame_count_(0), start_time_(0) {}                             // 初始化帧计数器和开始时间为0
    
    // 开始计时 - 重置计数器并开始新的FPS统计周期
    void start() {
        frame_count_ = 0;                                                         // 重置帧计数器为0
        start_time_ = get_current_time_ms();                                      // 记录当前时间作为开始时间
    }
    
    // 获取当前FPS - 计算从开始到现在的平均帧率
    // 返回值：当前的FPS值（帧/秒）
    double get_fps() {
        double current_time = get_current_time_ms();                              // 获取当前时间
        double elapsed = (current_time - start_time_) / 1000.0;                   // 计算经过的时间（秒）
        if (elapsed > 0) {                                                        // 检查经过时间是否大于0，避免除零错误
            return frame_count_ / elapsed;                                        // 计算FPS：帧数 / 时间(秒)
        }
        return 0.0;                                                               // 如果时间为0，返回0避免除零错误
    }
    
    // 更新帧计数 - 每处理一帧时调用此函数
    void update() {
        frame_count_++;                                                           // 帧计数器递增1
    }
    
private:
    int frame_count_;                                                             // 帧计数器，记录已处理的帧数
    double start_time_;                                                           // 开始时间（毫秒），用于计算经过时间
    
    // 获取当前时间（毫秒） - 内部工具函数
    // 返回值：当前时间的毫秒表示
    double get_current_time_ms() {
        struct timeval tv;                                                        // 创建时间结构体
        gettimeofday(&tv, NULL);                                                  // 获取当前时间
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;                        // 转换为毫秒：秒*1000 + 微秒/1000
    }
};

// 全局工具函数 - 获取当前时间（毫秒）
// 返回值：当前时间的毫秒表示
double get_current_time_ms() {
    struct timeval tv;                                                            // 创建时间结构体
    gettimeofday(&tv, NULL);                                                      // 获取当前系统时间
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;                            // 转换为毫秒：秒部分*1000 + 微秒部分/1000
}

// 主函数 - 安全帽RKNN线程池实时检测系统的入口点
// 参数：argc-命令行参数个数，argv-命令行参数数组
// 返回值：程序执行状态码，0表示成功，-1表示失败
int main(int argc, char **argv) {
    // 检查命令行参数 - 支持单进程和多进程模式
    if (argc < 2 || argc > 3) {                                                   // 检查参数个数是否正确
        printf("Usage: %s <model_path> [--multi-process]\n", argv[0]);            // 输出使用方法
        printf("Example: %s ../model/helmet.rknn\n", argv[0]);                   // 单进程模式示例
        printf("Example: %s ../model/helmet.rknn --multi-process\n", argv[0]);   // 多进程模式示例
        return -1;                                                                // 参数错误，返回-1
    }
    
    const char *model_path = argv[1];                                             // 获取模型文件路径参数
    bool use_multi_process = false;                                               // 多进程模式标志，默认为单进程模式
    
    // 检查是否使用多进程模式
    if (argc == 3 && strcmp(argv[2], "--multi-process") == 0) {                  // 检查第三个参数是否为多进程标志
        use_multi_process = true;                                                 // 设置多进程模式标志
    }
    
    // 设置信号处理 - 注册信号处理函数，支持优雅退出
    signal(SIGINT, signal_handler);                                               // 注册Ctrl+C信号处理函数
    signal(SIGTERM, signal_handler);                                              // 注册终止信号处理函数
    
    // 输出系统信息 - 显示程序启动信息和配置参数
    printf("=== 安全帽RKNN线程池实时检测系统 ===\n");                             // 输出系统标题
    printf("模型路径: %s\n", model_path);                                          // 输出模型文件路径
    printf("运行模式: %s\n", use_multi_process ? "多进程共享摄像头" : "单进程独立摄像头"); // 输出运行模式
    printf("NPU核心数: %d, 线程池大小: %d\n", NPU_CORES, TPEs);                    // 输出NPU配置信息
    printf("按 'q' 键退出\n\n");                                                   // 输出退出提示
    
    // 初始化后处理 - 加载标签文件和初始化后处理模块
    init_post_process();                                                          // 初始化YOLOv8后处理，加载类别标签
    
    // 根据模式选择摄像头管理方式
    std::string client_id;                                                        // 客户端ID，用于多进程模式
    if (use_multi_process) {
        // 多进程模式 - 使用摄像头资源池
        printf("正在初始化多进程摄像头资源池...\n");                                // 输出多进程模式提示
        CameraResourcePool& pool = CameraResourcePool::getInstance();             // 获取摄像头资源池单例
        
        // 初始化摄像头资源池 - 添加重试和等待机制
        int retry_count = 0;
        const int max_retries = 5;
        bool init_success = false;
        
        while (retry_count < max_retries && !init_success) {
            if (pool.initialize()) {                                              // 尝试初始化资源池
                init_success = true;                                              // 初始化成功
            } else {
                retry_count++;                                                    // 增加重试次数
                if (retry_count < max_retries) {
                    printf("摄像头资源池初始化失败，等待%d秒后重试... (第%d次)\n", 3, retry_count);
                    std::this_thread::sleep_for(std::chrono::seconds(3));         // 等待3秒后重试
                }
            }
        }
        
        if (!init_success) {                                                      // 所有重试都失败
            printf("摄像头资源池初始化失败，已重试%d次\n", max_retries);           // 输出失败信息
            return -1;                                                            // 初始化失败，退出程序
        }
        
        // 注册客户端到资源池
        client_id = pool.registerClient("安全帽检测");                             // 注册客户端，获取客户端ID
        if (client_id.empty()) {                                                  // 检查注册是否成功
            printf("客户端注册失败\n");                                             // 输出失败信息
            return -1;                                                            // 注册失败，退出程序
        }
        
        printf("客户端已注册，ID: %s\n", client_id.c_str());                       // 输出客户端ID
    } else {
        // 单进程模式 - 使用共享摄像头管理器
        printf("正在初始化单进程摄像头管理器...\n");                               // 输出单进程模式提示
    }
    
    // 单进程模式需要初始化摄像头管理器
    SharedCameraManager camera_manager;                                           // 创建共享摄像头管理器实例（仅单进程模式使用）
    if (!use_multi_process) {                                                     // 如果是单进程模式
        // 初始化摄像头 - 尝试打开和配置摄像头设备
        if (!camera_manager.initCamera()) {                                       // 尝试初始化摄像头
            printf("摄像头初始化失败\n");                                           // 输出失败信息
            return -1;                                                            // 初始化失败，退出程序
        }
        
        // 启动帧捕获 - 开始后台线程持续捕获摄像头帧
        camera_manager.startFrameCapture();                                       // 启动帧捕获线程
    }
    
    // 等待摄像头初始化 - 给摄像头一些时间完成初始化
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));                // 等待1秒确保摄像头完全初始化
    
    // 初始化RKNN线程池 - 创建多线程推理执行器
    RKNNPoolExecutor rknn_pool(model_path, TPEs);                                // 创建RKNN线程池，加载模型并绑定NPU核心
    
    // 创建显示窗口 - 初始化OpenCV显示窗口
    cv::namedWindow("安全帽检测", cv::WINDOW_AUTOSIZE);                            // 创建自适应大小的显示窗口
    
    // 初始化FPS计算器 - 创建性能监控工具
    FPSCounter fps_counter;                                                       // 创建FPS计算器实例
    fps_counter.start();                                                          // 开始FPS计时
    
    // 主循环变量初始化 - 设置帧跟踪和结果缓冲相关变量
    int64_t frame_id = 0;                                                         // 帧ID计数器，用于跟踪每一帧
    std::queue<InferenceResult> frame_buffer;                                     // 结果缓冲队列，用于保持帧的处理顺序
    int64_t expected_frame_id = 0;                                                // 期望的帧ID，用于按顺序处理结果
    
    printf("开始检测循环...\n");                                                   // 输出检测循环开始信息
    
    // 主检测循环 - 持续处理摄像头帧并进行安全帽检测
    while (g_running) {                                                           // 主循环，直到收到停止信号
        // 根据模式选择不同的帧获取方式
        cv::Mat frame;                                                            // 创建帧变量存储摄像头图像
        bool frame_available = false;                                             // 帧获取成功标志
        
        if (use_multi_process) {
            // 多进程模式 - 从摄像头资源池获取帧
            std::shared_ptr<cv::Mat> frame_ptr;                                   // 智能指针存储帧数据
            if (CameraPool::getFrame(client_id, frame_ptr, 100)) {                // 从资源池获取帧，超时100毫秒
                frame = *frame_ptr;                                               // 解引用智能指针获取帧数据
                frame_available = true;                                           // 设置获取成功标志
            }
        } else {
            // 单进程模式 - 从共享摄像头管理器获取帧
            if (camera_manager.getFrameFromQueue(frame)) {                        // 尝试从摄像头队列获取帧
                frame_available = true;                                           // 设置获取成功标志
            }
        }
        
        // 检查是否成功获取帧
        if (!frame_available) {                                                   // 如果没有获取到帧
            std::this_thread::sleep_for(std::chrono::milliseconds(5));            // 短暂休眠5毫秒
            continue;                                                             // 继续下一次循环尝试
        }
        
        // 验证帧数据完整性 - 检查帧数据是否有效
        if (frame.rows <= 0 || frame.cols <= 0 || frame.channels() <= 0) {       // 检查帧的尺寸和通道数是否有效
            printf("警告: 收到无效帧数据 (rows=%d, cols=%d, channels=%d)\n",      // 输出警告信息
                   frame.rows, frame.cols, frame.channels());
            continue;                                                             // 跳过无效帧，继续下一次循环
        }
        
        // 提交到RKNN线程池 - 将帧数据提交给推理线程池处理
        rknn_pool.put(frame, frame_id++);                                        // 提交帧到线程池，同时递增帧ID
        
        // 获取推理结果 - 尝试从结果队列获取已完成的推理结果
        InferenceResult result;                                                   // 创建结果变量存储推理结果
        if (rknn_pool.get(result)) {                                              // 尝试获取推理结果
            // 缓冲帧以保持顺序 - 将结果添加到缓冲队列中
            frame_buffer.push(result);                                            // 将结果添加到缓冲队列
            
            // 处理按顺序的帧 - 按照帧ID顺序处理结果，确保显示顺序正确
            while (!frame_buffer.empty() && frame_buffer.front().frame_id == expected_frame_id) { // 检查是否有按顺序的结果可处理
                auto current_result = frame_buffer.front();                       // 获取队列前端的结果
                frame_buffer.pop();                                               // 从缓冲队列中移除已处理的结果
                
                // 统计检测结果 - 统计当前帧中的安全帽和无安全帽数量
                int helmet_count = 0, no_helmet_count = 0;                        // 初始化当前帧的检测计数器
                for (int i = 0; i < current_result.results.count; i++) {          // 遍历当前帧的所有检测结果
                    if (current_result.results.results[i].cls_id == 1) {          // 检查类别ID是否为安全帽（类别1）
                        helmet_count++;                                           // 安全帽计数器递增
                    } else {                                                      // 其他类别（无安全帽，类别0）
                        no_helmet_count++;                                        // 无安全帽计数器递增
                    }
                }
                
                // 更新全局统计 - 使用原子操作更新全局统计数据
                g_total_helmet_count.fetch_add(helmet_count);                     // 原子操作：累加安全帽总数
                g_total_no_helmet_count.fetch_add(no_helmet_count);               // 原子操作：累加无安全帽总数
                g_total_frames.fetch_add(1);                                      // 原子操作：累加总处理帧数
                
                // 更新FPS - 更新帧率计算器并获取当前FPS
                fps_counter.update();                                             // 更新FPS计算器的帧计数
                double fps = fps_counter.get_fps();                               // 获取当前计算的FPS值
                
                // 创建显示帧 - 克隆原始帧用于绘制检测结果
                cv::Mat display_frame = current_result.frame.clone();             // 克隆原始帧，避免修改原始数据
                char text[256];                                                   // 创建文本缓冲区用于显示信息
                
                // 绘制检测框 - 在图像上绘制所有检测到的目标
                for (int i = 0; i < current_result.results.count; i++) {          // 遍历所有检测结果
                    object_detect_result *det_result = &(current_result.results.results[i]); // 获取当前检测结果的指针
                    
                    // 获取边界框坐标
                    int x1 = det_result->box.left;                                // 边界框左上角X坐标
                    int y1 = det_result->box.top;                                 // 边界框左上角Y坐标
                    int x2 = det_result->box.right;                               // 边界框右下角X坐标
                    int y2 = det_result->box.bottom;                              // 边界框右下角Y坐标
                    
                    // 根据检测结果选择颜色 - 安全帽用绿色，无安全帽用红色
                    cv::Scalar box_color = (det_result->cls_id == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);   // 边界框颜色
                    cv::Scalar text_color = (det_result->cls_id == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);  // 文本颜色
                    
                    // 绘制检测框 - 在图像上绘制矩形边界框
                    cv::rectangle(display_frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 3); // 绘制3像素宽的矩形框
                    
                    // 绘制标签 - 在边界框上方显示类别名称和置信度
                    sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100); // 格式化标签文本
                    cv::putText(display_frame, text, cv::Point(x1, y1 - 20),      // 在边界框上方20像素处绘制文本
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);     // 使用0.7大小的字体，2像素粗细
                }
                
                // 绘制FPS信息 - 在图像左上角显示当前帧率
                sprintf(text, "FPS: %.1f", fps);                                 // 格式化FPS文本
                cv::putText(display_frame, text, cv::Point(10, 30),               // 在(10,30)位置绘制FPS信息
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2); // 白色字体，1.0大小，2像素粗细
                
                // 绘制统计信息 - 显示累计的检测统计数据
                sprintf(text, "Helmet: %d | No Helmet: %d",                      // 格式化统计信息文本
                       g_total_helmet_count.load(), g_total_no_helmet_count.load()); // 获取原子变量的当前值
                cv::putText(display_frame, text, cv::Point(10, 70),               // 在(10,70)位置绘制统计信息
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2); // 白色字体，1.0大小，2像素粗细
                
                // 绘制线程池信息 - 显示处理该帧的NPU核心和处理时间
                sprintf(text, "Core: %d | Processing: %.1fms",                   // 格式化线程池信息文本
                       current_result.core_id, current_result.processing_time);  // 显示NPU核心ID和处理时间
                cv::putText(display_frame, text, cv::Point(10, 110),               // 在(10,110)位置绘制线程池信息
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2); // 白色字体，0.7大小，2像素粗细
                
                // 绘制队列状态 - 显示任务队列和结果队列的当前大小
                sprintf(text, "队列: %zu/%zu",                                    // 格式化队列状态文本
                        rknn_pool.task_queue_size(), rknn_pool.result_queue_size()); // 获取任务队列和结果队列大小
                cv::putText(display_frame, text, cv::Point(10, 150),              // 在(10,150)位置绘制队列状态
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2); // 白色字体，0.8大小，2像素粗细
                
                // 绘制控制提示 - 在图像底部显示退出提示
                cv::putText(display_frame, "按 'q' 键退出",                        // 退出提示文本
                           cv::Point(10, display_frame.rows - 20),               // 在图像底部20像素处绘制
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2); // 青色字体，0.7大小，2像素粗细
                
                // 显示结果 - 在OpenCV窗口中显示处理后的帧
                cv::imshow("安全帽检测", display_frame);                           // 在指定窗口中显示图像
                
                expected_frame_id++;                                              // 递增期望的帧ID，准备处理下一帧
            }
        } else {
            // 如果没有推理结果，显示原始帧 - 处理推理队列为空的情况
            static int no_result_count = 0;                                      // 静态计数器，记录无结果的次数
            no_result_count++;                                                    // 无结果计数器递增
            if (no_result_count % 30 == 0) {                                     // 每30帧显示一次提示信息
                printf("等待推理结果中... (已提交 %ld 帧)\n", frame_id);           // 输出等待信息到控制台
            }
            
            // 显示原始帧（带简单信息） - 在等待推理结果时显示原始摄像头帧
            cv::Mat display_frame = frame.clone();                               // 克隆原始帧用于显示
            char text[256];                                                       // 创建文本缓冲区
            sprintf(text, "Waiting for inference... Frame: %ld", frame_id);      // 格式化等待信息文本
            cv::putText(display_frame, text, cv::Point(10, 30),                   // 在(10,30)位置绘制等待信息
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); // 青色字体，0.8大小，2像素粗细
            
            sprintf(text, "Task Queue: %zu, Result Queue: %zu",                  // 格式化队列状态文本
                   rknn_pool.task_queue_size(), rknn_pool.result_queue_size());   // 获取队列大小信息
            cv::putText(display_frame, text, cv::Point(10, 70),                   // 在(10,70)位置绘制队列信息
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2); // 白色字体，0.7大小，2像素粗细
            
            cv::putText(display_frame, "按 'q' 键退出",                            // 退出提示文本
                       cv::Point(10, display_frame.rows - 20),                   // 在图像底部绘制
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2); // 青色字体，0.7大小，2像素粗细
            
            cv::imshow("安全帽检测", display_frame);                               // 显示等待状态的图像
        }
        
        // 处理键盘输入 - 检查用户是否请求退出程序
        char key = cv::waitKey(30) & 0xFF;                                       // 等待30毫秒获取键盘输入，屏蔽高位
        if (key == 'q' || key == 27) {                                          // 检查是否按下'q'键或ESC键
            g_running = false;                                                    // 设置全局停止标志
            break;                                                                // 退出主循环
        }
        
        // 每100帧输出一次统计信息 - 定期在控制台输出处理统计
        if (g_total_frames.load() % 100 == 0) {                                 // 检查是否处理了100的倍数帧
            printf("已处理 %d 帧, 总检测: 安全帽=%d, 无安全帽=%d, FPS=%.1f\n",    // 输出详细统计信息
                   g_total_frames.load(), g_total_helmet_count.load(), g_total_no_helmet_count.load(), fps_counter.get_fps());
        }
    }
    
    // 清理资源 - 程序退出前的资源清理工作
    printf("正在清理资源...\n");                                                  // 输出清理开始信息
    
    // 根据模式进行不同的资源清理
    if (use_multi_process) {
        // 多进程模式 - 注销客户端
        if (!client_id.empty()) {                                                // 如果客户端ID不为空
            CameraPool::unregisterClient(client_id);                             // 注销客户端
            printf("客户端已注销: %s\n", client_id.c_str());                      // 输出注销信息
        }
    } else {
        // 单进程模式 - 停止帧捕获并释放摄像头
        camera_manager.stopFrameCapture();                                       // 停止帧捕获线程
        camera_manager.release();                                                 // 释放摄像头资源
        printf("摄像头管理器已停止\n");                                            // 输出摄像头停止信息
    }
    
    cv::destroyAllWindows();                                                      // 销毁所有OpenCV窗口
    deinit_post_process();                                                        // 清理后处理模块资源
    
    // 输出最终统计信息 - 显示程序运行期间的完整统计数据
    printf("\n=== 检测统计 ===\n");                                               // 输出统计标题
    printf("总处理帧数: %d\n", g_total_frames.load());                            // 输出总处理帧数
    printf("总检测到安全帽: %d\n", g_total_helmet_count.load());                   // 输出安全帽检测总数
    printf("总检测到无安全帽: %d\n", g_total_no_helmet_count.load());             // 输出无安全帽检测总数
    
    printf("程序退出\n");                                                          // 输出程序退出信息
    return 0;                                                                     // 返回成功状态码
}
