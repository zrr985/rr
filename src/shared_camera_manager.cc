#include "shared_camera_manager.h"
#include "camera_detector.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>

// SharedCameraManager构造函数：初始化所有成员变量为默认值
SharedCameraManager::SharedCameraManager() 
    : camera_id_(-1),               // 摄像头ID初始化为-1（表示未初始化）
      width_(640),                  // 默认图像宽度640像素
      height_(480),                 // 默认图像高度480像素
      capture_running_(false) {     // 帧捕获线程运行标志初始化为false
}

// SharedCameraManager析构函数：确保资源正确释放
SharedCameraManager::~SharedCameraManager() {
    stopFrameCapture();             // 停止帧捕获线程，避免资源泄露
    release();                      // 释放摄像头资源
}

// 初始化摄像头：尝试打开指定摄像头，如果失败则尝试其他可用摄像头
bool SharedCameraManager::initCamera(int camera_id, int width, int height, int fps) {
    std::cout << "SharedCameraManager: 初始化摄像头 " << camera_id 
              << " (单进程模式) 目标: " << width << "x" << height << "@" << fps << "fps" << std::endl;
    
    width_ = width;                     // 保存目标图像宽度
    height_ = height;                   // 保存目标图像高度
    camera_id_ = camera_id;             // 保存摄像头ID（可能在后续搜索中改变）
    
    // 单进程模式，直接初始化物理摄像头
    
    // 智能摄像头搜索：优先尝试指定的摄像头ID，然后尝试其他可用摄像头
    std::vector<int> camera_ids_to_try = {camera_id, 0, 1, 2, 3, 4, 5};  // 创建摄像头ID候选列表
    
    // 移除重复的ID
    std::sort(camera_ids_to_try.begin(), camera_ids_to_try.end());        // 对ID列表进行排序
    camera_ids_to_try.erase(std::unique(camera_ids_to_try.begin(), camera_ids_to_try.end()), camera_ids_to_try.end());  // 移除重复的ID
    
    // 遍历所有候选摄像头ID，尝试打开
    for (int cam_id : camera_ids_to_try) {
        std::cout << "尝试打开摄像头 " << cam_id << std::endl;           // 输出当前尝试的摄像头ID
        if (tryOpenCamera(cam_id, width, height, fps)) {                  // 尝试打开当前摄像头
            camera_id_ = cam_id;                                          // 更新实际使用的摄像头ID
            std::cout << "成功打开摄像头 " << cam_id << std::endl;        // 输出成功信息
            return true;                                                  // 返回成功
        }
    }
    
    std::cerr << "无法找到可用的摄像头，初始化失败" << std::endl;          // 输出失败信息
    return false;                                                         // 返回失败
}

// 读取单帧图像：从摄像头直接读取一帧数据
bool SharedCameraManager::readFrame(cv::Mat& frame) {
    // 单进程模式，直接从摄像头读取
    if (cap_.isOpened()) {                  // 检查摄像头是否已打开
        return cap_.read(frame);            // 从摄像头读取一帧到frame参数中
    }
    
    return false;                           // 摄像头未打开，返回失败
}

// 查找系统中所有可用的摄像头设备
std::vector<int> SharedCameraManager::findAvailableCameras() {
    // 使用新的摄像头检测功能
    return ::findAvailableCameras();        // 调用全局函数查找可用摄像头，返回设备ID列表
}

bool SharedCameraManager::tryOpenCamera(int camera_id, int width, int height, int fps) {
    // 确保之前的摄像头已释放
    if (cap_.isOpened()) {
        cap_.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // 按优先级尝试不同的后端
    std::vector<std::pair<int, std::string>> backends = {
        {cv::CAP_V4L2, "V4L2"},
        {cv::CAP_ANY, "默认"}
    };
    
    for (const auto& [backend, name] : backends) {
        std::cout << "尝试使用 " << name << " 后端打开摄像头..." << std::endl;
        
        cap_.open(camera_id, backend);
        if (cap_.isOpened()) {
            std::cout << "✅ 使用 " << name << " 后端成功打开摄像头 " << camera_id << std::endl;
            
            // 配置摄像头参数
            if (setupCamera(width, height, fps)) {
                return true;
            } else {
                std::cout << "❌ " << name << " 后端参数配置失败，尝试下一个后端" << std::endl;
                cap_.release();
            }
        }
    }
    
    return false;
}

// 配置摄像头参数：按正确顺序设置参数以达到最佳性能
bool SharedCameraManager::setupCamera(int width, int height, int fps) {
    std::cout << "������ 正在配置摄像头参数..." << std::endl;
    
    // ⚡ 第一步：设置FOURCC格式为MJPEG（最重要！必须第一个设置）
    bool mjpg_success = cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    std::cout << "  步骤1: 设置MJPEG格式... " << (mjpg_success ? "✅" : "❌") << std::endl;
    
    // 第二步：设置缓冲区大小（在分辨率之前）
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 2);
    std::cout << "  步骤2: 设置缓冲区大小为2" << std::endl;
    
    // 第三步：设置分辨率
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    std::cout << "  步骤3: 设置分辨率 " << width << "x" << height << std::endl;
    
    // 第四步：设置帧率（必须在FOURCC和分辨率之后）
    cap_.set(cv::CAP_PROP_FPS, fps);
    std::cout << "  步骤4: 设置帧率 " << fps << " FPS" << std::endl;
    
    // 第五步：设置其他优化参数
    cap_.set(cv::CAP_PROP_AUTOFOCUS, 0);
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
    std::cout << "  步骤5: 设置其他参数完成" << std::endl;
    
    // 获取实际参数
    int actual_fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));  // 转换为int用于位运算
    int actual_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap_.get(cv::CAP_PROP_FPS);
    double actual_buffer = cap_.get(cv::CAP_PROP_BUFFERSIZE);
    
    // 将FOURCC代码转换为可读格式
    char fourcc_str[5] = {0};
    fourcc_str[0] = static_cast<char>(actual_fourcc & 0xFF);
    fourcc_str[1] = static_cast<char>((actual_fourcc >> 8) & 0xFF);
    fourcc_str[2] = static_cast<char>((actual_fourcc >> 16) & 0xFF);
    fourcc_str[3] = static_cast<char>((actual_fourcc >> 24) & 0xFF);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "������ 摄像头实际配置:" << std::endl;
    std::cout << "  编码格式: " << fourcc_str << " (MJPEG: " << (mjpg_success ? "✅" : "❌") << ")" << std::endl;
    std::cout << "  分辨率: " << actual_width << "x" << actual_height 
              << " (请求: " << width << "x" << height << ")" << std::endl;
    std::cout << "  帧率: " << actual_fps << " FPS (请求: " << fps << " FPS)" << std::endl;
    std::cout << "  缓冲区: " << actual_buffer << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 性能测试：测试实际帧率
    std::cout << "\n������ 测试摄像头实际性能..." << std::endl;
    cv::Mat test_frame;
    auto start_time = std::chrono::steady_clock::now();
    int test_frames = 0;
    const int target_frames = 60; // 测试2秒性能
    
    for (int i = 0; i < target_frames; i++) {
        if (cap_.read(test_frame) && !test_frame.empty()) {
            test_frames++;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double measured_fps = (duration > 0) ? (test_frames * 1000.0 / duration) : 0;
    
    std::cout << "  性能测试: " << test_frames << "帧/" << duration << "ms = " 
              << measured_fps << " FPS" << std::endl;
    
    // 判断是否成功
    if (measured_fps >= 25) {
        std::cout << "\n✅ 摄像头配置成功！实测FPS: " << measured_fps << std::endl;
        std::cout << "========================================\n" << std::endl;
        return true;
    } else {
        std::cout << "\n⚠️ 警告: 实测帧率(" << measured_fps << " FPS)低于预期" << std::endl;
        std::cout << "========================================\n" << std::endl;
        return false; // 返回false尝试其他后端
    }
}

// 释放摄像头资源：停止所有线程并释放硬件资源
void SharedCameraManager::release() {
    stopFrameCapture();                     // 先停止帧捕获线程
    
    // 单进程模式，直接释放摄像头
    if (cap_.isOpened()) {                  // 检查摄像头是否已打开
        cap_.release();                     // 释放摄像头硬件资源
        std::cout << "摄像头 " << camera_id_ << " 已释放" << std::endl;  // 输出释放信息
    }
}

// 多窗口检测专用方法实现
// 启动帧捕获线程：开始后台连续捕获帧到队列
void SharedCameraManager::startFrameCapture() {
    if (capture_running_) {                 // 检查帧捕获线程是否已在运行
        return;                             // 如果已运行，直接返回
    }
    
    // 单进程模式，直接启动帧捕获线程
    if (!cap_.isOpened()) {                 // 检查摄像头是否已初始化
        std::cerr << "摄像头未初始化，无法启动帧捕获" << std::endl;  // 输出错误信息
        return;                             // 摄像头未初始化，返回
    }
    
    capture_running_ = true;                // 设置运行标志为true
    capture_thread_ = std::thread(&SharedCameraManager::frameCaptureWorker, this);  // 创建并启动帧捕获工作线程
    std::cout << "摄像头 " << camera_id_ << " 帧捕获线程已启动" << std::endl;  // 输出启动信息
}

void SharedCameraManager::stopFrameCapture() {
    if (!capture_running_) {
        return;
    }
    
    capture_running_ = false;
    queue_cv_.notify_all();
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    // 清空队列
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!frame_queue_.empty()) {
        frame_queue_.pop();
    }
    while (!frame_ptr_queue_.empty()) {
        frame_ptr_queue_.pop();
    }
    
    std::cout << "摄像头 " << camera_id_ << " 帧捕获线程已停止" << std::endl;
}

// 从队列获取帧（Mat版本）：线程安全地从队列中获取一帧数据
bool SharedCameraManager::getFrameFromQueue(cv::Mat& frame) {
    // 单进程模式，从本地队列读取
    std::unique_lock<std::mutex> lock(queue_mutex_);        // 获取队列的唯一锁
    
    // 使用条件变量等待帧，避免忙等待
    if (frame_ptr_queue_.empty()) {                         // 检查智能指针队列是否为空
        // 等待更长时间，让帧捕获线程有机会添加帧
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {  // 等待100ms或直到条件满足
            return !frame_ptr_queue_.empty() || !capture_running_;         // 等待条件：队列非空或捕获线程停止
        });
        
        if (frame_ptr_queue_.empty()) {                     // 等待后再次检查队列是否为空
            return false;                                   // 队列仍为空，返回失败
        }
    }
    
    // 从智能指针队列获取帧
    auto frame_ptr = frame_ptr_queue_.front();              // 获取队列前端的帧指针
    frame_ptr_queue_.pop();                                 // 从队列中移除该帧
    
    // 复制到输出参数
    frame = *frame_ptr;                                     // 将智能指针指向的帧数据复制到输出参数
    return true;                                            // 返回成功
}

// 从队列获取帧（智能指针版本）：高效地获取帧指针，避免数据拷贝
bool SharedCameraManager::getFrameFromQueue(std::shared_ptr<cv::Mat>& frame_ptr) {
    // 单进程模式，从本地队列读取
    std::unique_lock<std::mutex> lock(queue_mutex_);        // 获取队列的唯一锁
    
    // 使用条件变量等待帧，避免忙等待
    if (frame_ptr_queue_.empty()) {                         // 检查智能指针队列是否为空
        // 等待更长时间，让帧捕获线程有机会添加帧
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {  // 等待100ms或直到条件满足
            return !frame_ptr_queue_.empty() || !capture_running_;         // 等待条件：队列非空或捕获线程停止
        });
        
        if (frame_ptr_queue_.empty()) {                     // 等待后再次检查队列是否为空
            return false;                                   // 队列仍为空，返回失败
        }
    }
    
    frame_ptr = frame_ptr_queue_.front();                   // 获取队列前端的帧智能指针
    frame_ptr_queue_.pop();                                 // 从队列中移除该帧指针
    
    // 添加调试信息
    static int get_count = 0;                               // 静态计数器，记录获取帧的次数
    if (++get_count % 50 == 0) {                           // 每50次输出一次调试信息
        std::cout << "成功获取帧 " << get_count << " 次，剩余队列大小: " << frame_ptr_queue_.size() << std::endl;
    }
    
    return true;                                            // 返回成功
}

// 获取当前队列大小：返回队列中等待处理的帧数量
int SharedCameraManager::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);         // 获取队列互斥锁（const版本）
    return static_cast<int>(frame_ptr_queue_.size());       // 返回智能指针队列的大小
}

// 帧捕获工作线程：后台持续捕获帧并添加到队列中
void SharedCameraManager::frameCaptureWorker() {
    std::cout << "摄像头 " << camera_id_ << " 帧捕获工作线程启动" << std::endl;  // 输出线程启动信息
    
    int frame_count = 0;                                    // 成功捕获的帧计数器
    int fail_count = 0;                                     // 失败次数计数器
    
    while (capture_running_) {                              // 主循环：当捕获标志为true时持续运行
        cv::Mat frame;                                      // 创建帧变量存储捕获的图像
        bool read_success = false;                          // 读取成功标志
        
        // 从真实摄像头读取帧
        read_success = cap_.read(frame);                    // 从摄像头读取一帧数据
        
        if (read_success && !frame.empty()) {               // 检查读取是否成功且帧不为空
            frame_count++;                                  // 成功帧计数器递增
            
            // 单进程模式，不需要共享内存
            
            std::unique_lock<std::mutex> lock(queue_mutex_); // 获取队列互斥锁
            
            // 如果队列满了，丢弃最老的帧
            if (frame_ptr_queue_.size() >= MAX_QUEUE_SIZE) { // 检查队列是否已满
                frame_ptr_queue_.pop();                     // 移除最老的帧，为新帧腾出空间
            }
            
            // 使用智能指针避免拷贝，提高性能
            auto frame_ptr = std::make_shared<cv::Mat>(std::move(frame));  // 创建智能指针，使用move避免拷贝
            frame_ptr_queue_.push(frame_ptr);               // 将帧指针添加到队列
            queue_cv_.notify_all();                         // 通知所有等待队列的线程
            
            // 每100帧打印一次调试信息
            if (frame_count % 100 == 0) {                   // 每100帧输出一次统计信息
                std::cout << "摄像头 " << camera_id_ << " 已捕获 " << frame_count 
                         << " 帧，队列大小: " << frame_ptr_queue_.size() << std::endl;
            }
        } else {                                            // 读取失败或帧为空的处理
            fail_count++;                                   // 失败计数器递增
            // 每10次失败打印一次调试信息（更频繁）
            if (fail_count % 10 == 0) {                     // 每10次失败输出一次错误信息
                std::cout << "摄像头 " << camera_id_ << " 读取帧失败 " << fail_count 
                         << " 次，摄像头状态: " << (cap_.isOpened() ? "已打开" : "未打开")
                         << "，读取结果: " << (read_success ? "成功但帧为空" : "失败") << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 短暂休眠，减少CPU占用
        }
    }
    
    std::cout << "摄像头 " << camera_id_ << " 帧捕获工作线程结束，总捕获: "   // 输出线程结束统计信息
              << frame_count << " 帧，总失败: " << fail_count << " 次" << std::endl;
}

// 全局摄像头管理器实现
// 获取全局摄像头管理器单例实例：确保整个程序只有一个管理器实例
GlobalCameraManager& GlobalCameraManager::getInstance() {
    static GlobalCameraManager instance;               // 静态局部变量，确保单例模式
    return instance;                                   // 返回单例实例的引用
}

// 获取或创建摄像头实例：智能管理摄像头资源，支持复用和自动清理
std::shared_ptr<SharedCameraManager> GlobalCameraManager::getCamera(int camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);  // 获取摄像头映射表的互斥锁
    
    auto it = cameras_.find(camera_id);                // 在映射表中查找指定摄像头ID
    if (it != cameras_.end()) {                        // 如果找到了现有的摄像头实例
        // 如果摄像头已经存在，检查是否可用
        if (it->second && it->second->isOpened()) {    // 检查摄像头实例是否存在且已打开
            std::cout << "返回已存在的摄像头 " << camera_id << std::endl;  // 输出复用信息
            return it->second;                         // 返回现有的摄像头实例（资源复用）
        } else {
            // 如果摄像头不可用，先释放
            std::cout << "摄像头 " << camera_id << " 不可用，先释放" << std::endl;  // 输出清理信息
            if (it->second) {                          // 如果摄像头实例存在
                it->second->release();                 // 释放摄像头资源
            }
            cameras_.erase(it);                        // 从映射表中移除该摄像头实例
        }
    }
    
    // 创建新的摄像头实例
    std::cout << "创建新的摄像头实例 " << camera_id << std::endl;  // 输出创建信息
    auto camera = std::make_shared<SharedCameraManager>();        // 创建新的摄像头管理器智能指针
    if (camera->initCamera(camera_id)) {                          // 尝试初始化摄像头
        cameras_[camera_id] = camera;                             // 将新实例添加到映射表中
        std::cout << "摄像头 " << camera_id << " 创建成功" << std::endl;  // 输出成功信息
        return camera;                                            // 返回新创建的摄像头实例
    }
    
    std::cout << "摄像头 " << camera_id << " 创建失败" << std::endl;  // 输出失败信息
    return nullptr;                                               // 创建失败，返回空指针
}

// 单进程模式，不需要共享内存相关方法

// 释放所有摄像头：优雅地停止所有摄像头的帧捕获，但保留实例供其他程序使用
void GlobalCameraManager::releaseAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);  // 获取摄像头映射表的互斥锁
    
    for (auto& [id, camera] : cameras_) {              // 遍历所有摄像头实例
        if (camera) {                                  // 检查摄像头实例是否存在
            // 停止帧捕获线程，但不释放摄像头，让其他程序可以继续使用
            camera->stopFrameCapture();               // 停止帧捕获线程
            std::cout << "摄像头 " << id << " 帧捕获已停止" << std::endl;  // 输出停止信息
        }
    }
    // 不清空cameras_，让其他程序可以继续使用
    std::cout << "所有摄像头帧捕获已停止，但摄像头实例保留供其他程序使用" << std::endl;  // 输出总体状态
}

// 强制释放所有摄像头：完全释放所有摄像头资源，用于程序退出或异常情况
void GlobalCameraManager::forceReleaseAllCameras() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);  // 获取摄像头映射表的互斥锁
    
    for (auto& [id, camera] : cameras_) {              // 遍历所有摄像头实例
        if (camera) {                                  // 检查摄像头实例是否存在
            camera->stopFrameCapture();               // 停止帧捕获线程
            camera->release();                         // 完全释放摄像头资源
            std::cout << "摄像头 " << id << " 已强制释放" << std::endl;  // 输出释放信息
        }
    }
    cameras_.clear();                                  // 清空摄像头映射表
    
    std::cout << "所有摄像头已强制释放" << std::endl;    // 输出完成信息
}

// 获取系统中所有可用的摄像头设备列表
std::vector<int> GlobalCameraManager::getAvailableCameras() {
    // 使用新的摄像头检测功能
    return ::findAvailableCameras();                   // 调用全局函数获取可用摄像头列表
}