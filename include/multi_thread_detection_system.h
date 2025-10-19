#ifndef MULTI_THREAD_DETECTION_SYSTEM_H
#define MULTI_THREAD_DETECTION_SYSTEM_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <signal.h>

#include "postprocess.h"
#include "retinaface_postprocess.h"
#include "meter_postprocess.h"
#include "yolov8.h"
#include "shared_camera_manager.h"

// 全局运行标志
extern std::atomic<bool> g_running;

// 颜色定义 - 为不同检测类型定义专用颜色，便于视觉区分
const cv::Scalar HELMET_COLOR(0, 255, 0);      // 绿色(BGR格式) - 安全帽检测框颜色
const cv::Scalar NO_HELMET_COLOR(0, 0, 255);   // 红色(BGR格式) - 无安全帽检测框颜色
const cv::Scalar FLAME_COLOR(0, 100, 255);     // 橙色(BGR格式) - 火焰检测框颜色
const cv::Scalar SMOKING_COLOR(255, 0, 0);     // 蓝色(BGR格式) - 吸烟检测框颜色
const cv::Scalar FACE_COLOR(255, 255, 0);      // 青色(BGR格式) - 人脸检测框颜色
const cv::Scalar METER_COLOR(0, 255, 255);     // 黄色(BGR格式) - 仪表检测框颜色

// FPS计算器类 - 用于计算和显示帧率
class FPSCounter {
public:
    // 开始计时 - 重置计时器并开始新的FPS统计周期
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        frame_count_ = 0;
    }
    
    // 获取当前FPS - 计算从开始到现在的平均帧率
    double get_fps() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time_).count();
        if (elapsed > 0) {
            return (frame_count_ * 1000.0) / elapsed;
        }
        return 0.0;
    }
    
    // 更新帧计数 - 每处理一帧时调用此函数
    void update() {
        frame_count_++;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    int frame_count_ = 0;
};

// 信号处理函数 - 处理程序退出信号
void signal_handler(int sig);

// 中文文本绘制函数 - 解决OpenCV中文显示问题
void putChineseText(cv::Mat& img, const std::string& text, cv::Point pos, 
                   cv::Scalar color, double font_scale = 0.6, int thickness = 2);

// 绘制检测结果函数
void draw_detection_results(cv::Mat& frame, const object_detect_result_list& results, 
                           const std::string& task_name);

// 多线程检测系统主类
class MultiThreadDetectionSystem {
private:
    // 检测任务结构体 - 存储单个检测任务的所有信息
    struct DetectionTask {
        std::string name;                    // 任务名称（helmet、flame、smoking等）
        std::string model_path;              // 模型文件路径
        std::vector<rknn_app_context_t> app_contexts;  // RKNN应用上下文池（3个）
        std::thread thread;                   // 任务专用线程
        std::atomic<bool> running{false};    // 任务运行状态
        std::atomic<int> detection_count{0}; // 检测计数
        FPSCounter fps_counter;              // FPS计算器
        int tpes_;                           // 线程池执行器数量
        
        // 默认构造函数
        DetectionTask() = default;
        
        // 移动构造函数
        DetectionTask(DetectionTask&& other) noexcept
            : name(std::move(other.name))
            , model_path(std::move(other.model_path))
            , app_contexts(std::move(other.app_contexts))
            , thread(std::move(other.thread))
            , running(other.running.load())
            , detection_count(other.detection_count.load())
            , fps_counter(std::move(other.fps_counter))
            , tpes_(other.tpes_)
        {
            // 重置原子变量的值
            running.store(false);
            detection_count.store(0);
        }
        
        // 移动赋值运算符
        DetectionTask& operator=(DetectionTask&& other) noexcept {
            if (this != &other) {
                name = std::move(other.name);
                model_path = std::move(other.model_path);
                app_contexts = std::move(other.app_contexts);
                thread = std::move(other.thread);
                running.store(other.running.load());
                detection_count.store(other.detection_count.load());
                fps_counter = std::move(other.fps_counter);
                tpes_ = other.tpes_;
            }
            return *this;
        }
        
        // 禁用复制构造函数和复制赋值运算符
        DetectionTask(const DetectionTask&) = delete;
        DetectionTask& operator=(const DetectionTask&) = delete;
    };
    
    std::shared_ptr<SharedCameraManager> camera_manager_;  // 摄像头管理器
    std::vector<DetectionTask> tasks_;                     // 检测任务列表
    std::atomic<bool> system_running_{false};             // 系统运行状态
    int camera_id_;                                        // 摄像头ID
    
public:
    // 构造函数 - 初始化多线程检测系统
    MultiThreadDetectionSystem(int camera_id = 0) : camera_id_(camera_id) {}
    
    // 析构函数 - 确保资源正确释放
    ~MultiThreadDetectionSystem() { stop(); }
    
    // 初始化系统 - 设置摄像头和信号处理
    bool initialize();
    
    // 添加检测任务 - 配置新的检测任务
    void addTask(const std::string& task_name, const std::string& model_path);
    
    // 启动系统 - 启动所有检测线程
    void start();
    
    // 停止系统 - 优雅地停止所有线程和资源
    void stop();
    
    // 打印统计信息 - 显示各任务的检测统计
    void printStatistics() const;

private:
    // 任务工作线程 - 每个检测任务的专用工作函数
    void taskWorker(DetectionTask& task);
};

#endif // MULTI_THREAD_DETECTION_SYSTEM_H
