#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>
#include <map>
#include <string>
#include <signal.h>

#include "postprocess.h"
#include "yolov8.h"
#include "shared_camera_manager.h"

// 全局运行标志
extern std::atomic<bool> g_running;
void signal_handler(int sig);

// 帧数据智能指针
using FramePtr = std::shared_ptr<cv::Mat>;

// 简化的FPS计数器
class FPSCounter {
private:
    std::chrono::steady_clock::time_point start_time;
    int frame_count;
    
public:
    FPSCounter() : frame_count(0) {
        start_time = std::chrono::steady_clock::now();
    }
    
    void update() {
        frame_count++;
    }
    
    double get_fps() const {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        return elapsed > 0 ? frame_count / elapsed : 0.0;
    }
    
    void reset() {
        frame_count = 0;
        start_time = std::chrono::steady_clock::now();
    }
};

// 异步显示管理器 - 避免显示阻塞推理
class DisplayManager {
private:
    struct DisplayTask {
        std::string window_name;
        cv::Mat frame;
    };
    
    std::queue<DisplayTask> display_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread display_thread_;
    std::atomic<bool> running_{false};
    std::map<std::string, bool> window_created_;
    
public:
    DisplayManager();
    ~DisplayManager();
    
    void start();
    void stop();
    void update_display(const std::string& window_name, const cv::Mat& frame);
    
private:
    void display_worker();
};

// 全局显示管理器
extern std::unique_ptr<DisplayManager> g_display_manager;

// 高性能独立队列缓冲区
class HighPerformanceBuffer {
private:
    struct ConsumerQueue {
        std::queue<FramePtr> queue;
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<bool> has_frame{false};
        size_t max_size;
    };
    
    std::map<std::string, std::shared_ptr<ConsumerQueue>> consumer_queues_;
    
public:
    HighPerformanceBuffer(size_t max_size_per_queue, const std::vector<std::string>& consumer_names);
    bool produce(const cv::Mat& frame);
    FramePtr consume(const std::string& consumer_name);
    size_t get_size(const std::string& consumer_name) const;
};

// 简化的检测消费者
class DetectionConsumer {
private:
    std::string task_name_;
    std::string model_path_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    
    HighPerformanceBuffer& buffer_;
    std::vector<rknn_app_context_t> app_contexts_;
    std::atomic<int> detection_count_{0};
    FPSCounter fps_counter_;
    std::atomic<int> context_index_{0};
    
public:
    DetectionConsumer(const std::string& task_name, 
                     const std::string& model_path, 
                     HighPerformanceBuffer& buffer);
    ~DetectionConsumer();
    
    bool initialize();
    void start();
    void stop();
    void join();
    int get_detection_count() const { return detection_count_.load(); }
    double get_fps() const { return fps_counter_.get_fps(); }
    const std::string& get_task_name() const { return task_name_; }

private:
    void worker_thread();
};

// 高性能摄像头生产者
class CameraProducer {
private:
    std::shared_ptr<SharedCameraManager> camera_manager_;
    HighPerformanceBuffer& buffer_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    int camera_id_;
    FPSCounter fps_counter_;
    std::atomic<int> frame_count_{0};
    
public:
    CameraProducer(int camera_id, HighPerformanceBuffer& buffer);
    ~CameraProducer();
    
    bool initialize();
    void start();
    void stop();
    void join();
    double get_fps() const { return fps_counter_.get_fps(); }
    int get_frame_count() const { return frame_count_.load(); }

private:
    void producer_thread();
};

#endif // CAMERA_H
