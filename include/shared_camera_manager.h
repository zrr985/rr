#ifndef _SHARED_CAMERA_MANAGER_H_
#define _SHARED_CAMERA_MANAGER_H_

#include <opencv2/opencv.hpp>  // OpenCV库，用于摄像头操作和图像处理
#include <iostream>            // 标准输入输出流，用于调试输出
#include <vector>              // C++标准库向量容器，用于动态数组操作
#include <thread>              // 多线程支持，用于并发处理
#include <atomic>              // 原子操作，用于无锁编程
#include <mutex>               // 互斥锁，用于线程同步
#include <condition_variable>  // 条件变量，用于线程间通信
#include <queue>               // 队列容器，用于帧缓冲
#include <functional>          // 函数对象支持，用于回调函数
#include <map>                 // 映射容器，用于摄像头管理
#include <memory>              // 智能指针支持，用于内存管理
// 单进程模式，不需要共享内存相关头文件

// 单进程模式，不需要共享内存数据结构

// 多窗口检测系统的摄像头管理器（单进程模式）
class SharedCameraManager {
public:
    // 构造函数和析构函数
    SharedCameraManager();  // 构造函数，初始化成员变量
    ~SharedCameraManager(); // 析构函数，清理资源
    
    // 初始化摄像头 - 支持多窗口检测
    // 参数：camera_id-摄像头设备ID，width-图像宽度，height-图像高度，fps-帧率
    // 返回值：成功返回true，失败返回false
    bool initCamera(int camera_id = 0, int width = 640, int height = 480, int fps = 30);
    
    // 读取帧 - 线程安全
    // 参数：frame-输出参数，存储读取的帧数据
    // 返回值：成功返回true，失败返回false
    bool readFrame(cv::Mat& frame);
    
    // 释放摄像头资源
    void release();
    
    // 获取摄像头状态
    bool isOpened() const { return cap_.isOpened(); }  // 检查摄像头是否已打开
    
    // 获取摄像头信息
    int getCameraId() const { return camera_id_; }     // 获取摄像头设备ID
    int getWidth() const { return width_; }            // 获取图像宽度
    int getHeight() const { return height_; }          // 获取图像高度
    
    // 多窗口检测专用方法
    void startFrameCapture();  // 启动帧捕获线程，开始连续捕获帧
    void stopFrameCapture();   // 停止帧捕获线程，停止捕获
    bool getFrameFromQueue(cv::Mat& frame);  // 从队列获取帧（值拷贝）
    bool getFrameFromQueue(std::shared_ptr<cv::Mat>& frame_ptr);  // 从队列获取帧（智能指针版本，避免拷贝）
    int getQueueSize() const;  // 获取当前队列中的帧数量

private:
    // 禁用拷贝构造和赋值 - 防止对象被意外复制
    SharedCameraManager(const SharedCameraManager&) = delete;  // 删除拷贝构造函数
    SharedCameraManager& operator=(const SharedCameraManager&) = delete;  // 删除赋值操作符
    
    // 智能摄像头检测函数
    std::vector<int> findAvailableCameras();  // 扫描系统中可用的摄像头设备
    bool tryOpenCamera(int camera_id, int width, int height, int fps);  // 尝试打开指定摄像头
    bool setupCamera(int width, int height, int fps);  // 配置摄像头参数（分辨率、帧率等）
    
    // 帧捕获线程函数
    void frameCaptureWorker();  // 帧捕获工作线程的主函数
    
    // 成员变量
    cv::VideoCapture cap_;  // OpenCV摄像头对象
    int camera_id_;         // 摄像头设备ID
    int width_;             // 图像宽度
    int height_;            // 图像高度
    
    // 多窗口检测相关
    std::atomic<bool> capture_running_;  // 原子标志：帧捕获线程是否正在运行
    std::thread capture_thread_;         // 帧捕获线程对象
    std::queue<cv::Mat> frame_queue_;    // 帧队列（值拷贝版本）
    std::queue<std::shared_ptr<cv::Mat>> frame_ptr_queue_;  // 智能指针队列，避免拷贝，提高性能
    mutable std::mutex queue_mutex_;     // 队列互斥锁，保护队列的线程安全访问
    std::condition_variable queue_cv_;   // 队列条件变量，用于线程间同步
    static const int MAX_QUEUE_SIZE = 10;  // 最大队列大小，防止内存溢出
    
    // 单进程模式，不需要多进程支持
};

// 全局摄像头管理器 - 单例模式，支持多窗口检测
class GlobalCameraManager {
public:
    static GlobalCameraManager& getInstance();  // 获取单例实例，确保全局只有一个管理器
    
    // 获取摄像头实例
    // 参数：camera_id-摄像头设备ID
    // 返回值：摄像头管理器的智能指针
    std::shared_ptr<SharedCameraManager> getCamera(int camera_id = 0);
    
    // 释放所有摄像头
    void releaseAllCameras();  // 正常释放所有摄像头资源
    void forceReleaseAllCameras();  // 强制释放所有摄像头，用于异常情况
    
    // 获取可用摄像头列表
    // 返回值：可用摄像头设备ID的向量
    std::vector<int> getAvailableCameras();

private:
    GlobalCameraManager() = default;  // 私有构造函数，单例模式
    ~GlobalCameraManager() = default; // 私有析构函数，单例模式
    
    // 禁用拷贝构造和赋值 - 防止单例模式被破坏
    GlobalCameraManager(const GlobalCameraManager&) = delete;  // 删除拷贝构造函数
    GlobalCameraManager& operator=(const GlobalCameraManager&) = delete;  // 删除赋值操作符
    
    std::mutex cameras_mutex_;  // 摄像头映射表的互斥锁，保护线程安全
    std::map<int, std::shared_ptr<SharedCameraManager>> cameras_;  // 摄像头映射表，key为camera_id，value为摄像头管理器
};

#endif //_SHARED_CAMERA_MANAGER_H_  // 头文件保护宏结束，防止重复包含