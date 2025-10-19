#pragma once  // 编译器指令，防止头文件被重复包含（现代C++替代#ifndef的方式）

#include <opencv2/opencv.hpp>  // OpenCV库，用于摄像头操作和图像处理
#include <memory>              // 智能指针支持，用于内存管理
#include <thread>              // 多线程支持，用于并发处理
#include <mutex>               // 互斥锁，用于线程同步
#include <condition_variable>  // 条件变量，用于线程间通信
#include <queue>               // 队列容器，用于帧缓冲
#include <map>                 // 映射容器，用于客户端管理
#include <atomic>              // 原子操作，用于无锁编程
#include <string>              // 字符串处理
#include <chrono>              // 时间处理，用于超时和统计
#include <vector>              // 动态数组，用于存储列表

/**
 * 摄像头资源池 - 基于多窗口检测架构设计
 * 
 * 设计理念：
 * 1. 单例资源池 - 全局唯一的摄像头资源池
 * 2. 多客户端支持 - 支持多个检测程序同时连接
 * 3. 帧分发机制 - 一个摄像头为多个客户端提供帧
 * 4. 客户端管理 - 自动管理客户端的连接和断开
 * 5. 资源保护 - 防止资源冲突和内存泄漏
 * 
 * 架构参考：
 * - 基于多窗口检测的MultiWindowDetector设计
 * - 使用GlobalCameraManager的单例模式
 * - 采用多线程帧分发机制
 * - 智能指针管理帧数据，避免拷贝
 */

// 客户端信息结构 - 参考多窗口检测的DetectionResult
struct CameraClient {
    std::string client_id;                    // 客户端唯一标识符，用于区分不同的检测程序
    std::string client_name;                  // 客户端名称（如"火焰检测"、"安全帽检测"等）
    std::queue<std::shared_ptr<cv::Mat>> frame_queue;  // 客户端专用帧队列，存储待处理的图像帧
    std::mutex queue_mutex;                   // 队列互斥锁，保护帧队列的线程安全访问
    std::condition_variable queue_cv;         // 队列条件变量，用于线程间同步和通知
    std::atomic<bool> active{true};           // 客户端是否活跃的原子标志，用于快速检查状态
    std::chrono::high_resolution_clock::time_point last_access;  // 最后访问时间，用于超时检测
    int max_queue_size;                       // 最大队列大小，防止内存溢出
    std::atomic<int64_t> frame_id{0};         // 帧ID计数器，用于跟踪帧的顺序
    
    // 构造函数：初始化客户端信息
    // 参数：id-客户端ID，name-客户端名称，max_size-最大队列大小（默认5）
    CameraClient(const std::string& id, const std::string& name, int max_size = 5)
        : client_id(id), client_name(name), max_queue_size(max_size) {
        last_access = std::chrono::high_resolution_clock::now();  // 记录创建时间
    }
};

// 摄像头资源池类 
class CameraResourcePool {
public:
    // 获取单例实例 
    static CameraResourcePool& getInstance();
    
    // 禁用拷贝构造和赋值 - 防止单例模式被破坏
    CameraResourcePool(const CameraResourcePool&) = delete;  // 删除拷贝构造函数
    CameraResourcePool& operator=(const CameraResourcePool&) = delete;  // 删除赋值操作符
    
    // 析构函数 - 清理所有资源
    ~CameraResourcePool();
    
    // 初始化摄像头资源池 
    // 参数：camera_id-摄像头设备ID，width-图像宽度，height-图像高度
    bool initialize(int camera_id = 0, int width = 640, int height = 480);
    
    // 客户端管理 - 参考多窗口检测的线程管理
    std::string registerClient(const std::string& client_name);  // 注册新客户端，返回客户端ID
    bool unregisterClient(const std::string& client_id);         // 注销客户端
    bool isClientActive(const std::string& client_id);           // 检查客户端是否活跃
    
    // 帧获取接口 
    // 参数：client_id-客户端ID，frame-输出帧指针，timeout_ms-超时时间（毫秒）
    bool getFrame(const std::string& client_id, std::shared_ptr<cv::Mat>& frame, int timeout_ms = 1000);
    
    // 状态查询接口
    int getActiveClientCount() const;                    // 获取活跃客户端数量
    std::vector<std::string> getActiveClients() const;   // 获取活跃客户端列表
    bool isInitialized() const;                          // 检查是否已初始化
    
    // 资源管理 
    bool startFrameCapture();  // 开始帧捕获
    void stopFrameCapture();   // 停止帧捕获
    void cleanup();            // 清理所有资源
    
    // 调试信息 
    void printStatus() const;  // 打印当前状态信息
    
private:
    // 私有构造函数 - 单例模式，防止外部直接创建实例
    CameraResourcePool();
    
    // 帧捕获线程 
    // 功能：从摄像头连续捕获帧，放入原始帧队列
    void frameCaptureWorker();
    
    // 帧分发线程 
    // 功能：从原始帧队列取帧，分发给所有活跃客户端
    void frameDistributionWorker();
    
    // 客户端清理线程 
    // 功能：定期清理超时或非活跃的客户端
    void clientCleanupWorker();
    
    // 分发帧到所有活跃客户端 
    // 参数：frame-要分发的帧
    void distributeFrame(const cv::Mat& frame);
    
    // 清理非活跃客户端
    // 功能：移除超时或已断开的客户端
    void cleanupInactiveClients();
    
    // 成员变量
    std::atomic<bool> initialized_{false};    // 原子标志：资源池是否已初始化
    std::atomic<bool> running_{false};        // 原子标志：资源池是否正在运行
    std::atomic<int> next_client_id_{1};      // 原子计数器：下一个客户端ID
    
    // 摄像头相关 
    cv::VideoCapture camera_;                 // OpenCV摄像头对象
    int camera_id_;                           // 摄像头设备ID
    int width_, height_;                      // 图像宽度和高度
    
    // 线程管理 - 参考MultiWindowDetector的线程管理
    std::thread frame_capture_thread_;        // 帧捕获线程
    std::thread frame_distribution_thread_;   // 帧分发线程
    std::thread client_cleanup_thread_;       // 客户端清理线程
    
    // 客户端管理 
    std::map<std::string, std::shared_ptr<CameraClient>> clients_;  // 客户端映射表
    mutable std::mutex clients_mutex_;        // 客户端映射表的互斥锁
    
    // 帧队列 
    std::queue<cv::Mat> raw_frame_queue_;     // 原始帧队列
    std::mutex raw_frame_mutex_;              // 原始帧队列的互斥锁
    std::condition_variable raw_frame_cv_;    // 原始帧队列的条件变量
    static const int MAX_RAW_FRAME_QUEUE_SIZE = 10;  // 最大原始帧队列大小
    
    // 统计信息 
    std::atomic<int64_t> total_frames_captured_{0};     // 总捕获帧数
    std::atomic<int64_t> total_frames_distributed_{0};  // 总分发帧数
    std::chrono::high_resolution_clock::time_point start_time_;  // 开始时间
    
    // 配置参数 
    static constexpr int CLIENT_CLEANUP_INTERVAL_MS = 5000;  // 客户端清理间隔（5秒）
    static constexpr int CLIENT_TIMEOUT_MS = 30000;          // 客户端超时时间（30秒）
    static constexpr int FRAME_CAPTURE_INTERVAL_MS = 33;     // 帧捕获间隔（约30FPS）
};

// 便捷的全局函数
namespace CameraPool {
    // 注册客户端 - 便捷接口，直接调用单例实例的方法
    std::string registerClient(const std::string& client_name);
    
    // 注销客户端 - 便捷接口，直接调用单例实例的方法
    bool unregisterClient(const std::string& client_id);
    
    // 获取帧 - 便捷接口，直接调用单例实例的方法
    bool getFrame(const std::string& client_id, std::shared_ptr<cv::Mat>& frame, int timeout_ms = 1000);
    
    // 检查客户端是否活跃 - 便捷接口，直接调用单例实例的方法
    bool isClientActive(const std::string& client_id);
    
    // 获取状态信息 - 便捷接口，直接调用单例实例的方法
    int getActiveClientCount();                    // 获取活跃客户端数量
    std::vector<std::string> getActiveClients();   // 获取活跃客户端列表
    
    // 打印状态 - 便捷接口，直接调用单例实例的方法
    void printStatus();  // 打印当前状态信息
}
