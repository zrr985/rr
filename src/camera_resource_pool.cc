#include "camera_resource_pool.h"  // 摄像头资源池头文件，包含类定义和函数声明
#include "camera_detector.h"        // 摄像头检测器头文件，用于检测可用摄像头
#include <iostream>                 // 标准输入输出流，用于调试输出
#include <algorithm>                // 算法库，用于数据处理
#include <sstream>                  // 字符串流，用于字符串操作
#include <fstream>                  // 文件流，用于文件操作
#include <sys/file.h>              // 文件锁，用于进程间协调
#include <unistd.h>                // Unix标准定义，用于进程ID
#include <signal.h>                // 信号处理，用于kill函数

// 静态常量成员变量已在头文件中定义并初始化，直接使用即可

// 文件锁路径，用于进程间协调
const std::string LOCK_FILE_PATH = "/tmp/camera_resource_pool.lock";

// 检查是否有其他进程正在使用摄像头
bool isCameraInUse() {
    std::ifstream lock_file(LOCK_FILE_PATH);
    if (!lock_file.is_open()) {
        return false;  // 锁文件不存在，摄像头未被使用
    }
    
    pid_t pid;
    lock_file >> pid;
    lock_file.close();
    
    // 检查进程是否还在运行
    if (kill(pid, 0) == 0) {
        return true;  // 进程还在运行，摄像头被占用
    } else {
        // 进程已退出，删除锁文件
        unlink(LOCK_FILE_PATH.c_str());
        return false;
    }
    
    return false;  // 默认返回false
}

// 创建锁文件
bool createLockFile() {
    std::ofstream lock_file(LOCK_FILE_PATH);
    if (!lock_file.is_open()) {
        return false;
    }
    lock_file << getpid() << std::endl;
    lock_file.close();
    return true;
}

// 删除锁文件
void removeLockFile() {
    unlink(LOCK_FILE_PATH.c_str());
}

// 获取单例实例 - 线程安全的懒汉式单例模式
CameraResourcePool& CameraResourcePool::getInstance() {
    static CameraResourcePool instance;  // 静态局部变量，C++11保证线程安全
    return instance;                     // 返回单例实例的引用
}

// 私有构造函数 - 单例模式，只能通过getInstance()获取实例
CameraResourcePool::CameraResourcePool() 
    : camera_id_(0), width_(640), height_(480) {  // 初始化摄像头ID和分辨率
    start_time_ = std::chrono::high_resolution_clock::now();  // 记录创建时间
    std::cout << "摄像头资源池创建" << std::endl;
}

// 析构函数 - 自动清理资源
CameraResourcePool::~CameraResourcePool() {
    cleanup();  // 调用清理函数释放所有资源
    removeLockFile();  // 删除锁文件，释放摄像头访问权限
    std::cout << "摄像头资源池销毁" << std::endl;
}

// 初始化摄像头资源池
// 参数：camera_id-摄像头设备ID，width-图像宽度，height-图像高度
// 返回值：成功返回true，失败返回false
// 功能：初始化摄像头资源池，检测可用摄像头，配置摄像头参数
bool CameraResourcePool::initialize(int camera_id, int width, int height) {
    std::lock_guard<std::mutex> lock(clients_mutex_);  // 线程安全锁，保护初始化过程
    
    // 检查是否有其他进程正在使用摄像头
    if (isCameraInUse()) {
        std::cout << "检测到其他进程正在使用摄像头，等待资源释放..." << std::endl;
        // 等待其他进程释放摄像头
        int wait_count = 0;
        while (isCameraInUse() && wait_count < 30) {  // 最多等待30秒
            std::this_thread::sleep_for(std::chrono::seconds(1));
            wait_count++;
            if (wait_count % 5 == 0) {
                std::cout << "等待摄像头资源释放... (" << wait_count << "秒)" << std::endl;
            }
        }
        
        if (isCameraInUse()) {
            std::cout << "等待超时，摄像头仍被其他进程占用" << std::endl;
            return false;
        }
    }
    
    // // 创建锁文件，标记当前进程正在使用摄像头
    // if (!createLockFile()) {
    //     std::cout << "无法创建锁文件，摄像头可能被占用" << std::endl;
    //     return false;
    // }
    
    // std::cout << "当前进程获得摄像头访问权限，开始初始化..." << std::endl;
    
    // 使用摄像头检测器找到可用的摄像头 - 调用camera_detector.cc中的函数
    std::vector<int> available_cameras = findAvailableCameras();
    
    // 设置摄像头参数 - 保存用户指定的分辨率
    width_ = width;
    height_ = height;
    
    if (!available_cameras.empty()) {
        // 使用检测到的摄像头 - 优先使用智能检测到的摄像头
        camera_id_ = available_cameras[0];  // 使用第一个可用摄像头
        std::cout << "使用检测到的摄像头 " << camera_id_ << " (从 " << available_cameras.size() << " 个可用摄像头中选择)" << std::endl;
        
        // 尝试不同的后端 - 参考SharedCameraManager的后端选择
        std::vector<int> backends = {cv::CAP_V4L2, cv::CAP_ANY};  // V4L2后端优先，通用后端备选
        bool camera_opened = false;
        
        for (int backend : backends) {
            camera_.open(camera_id_, backend);  // 尝试使用指定后端打开摄像头
            if (camera_.isOpened()) {
                std::cout << "使用后端 " << backend << " 成功打开摄像头" << std::endl;
                camera_opened = true;
                break;  // 成功打开，跳出循环
            }
        }
        
        if (!camera_opened) {
            std::cout << "无法打开检测到的摄像头 " << camera_id_ << "，尝试通用摄像头..." << std::endl;
        }
    }
    
    // 如果检测到的摄像头无法打开，或者没有检测到摄像头，尝试通用摄像头
    if (!camera_.isOpened()) {
        std::cout << "尝试直接打开通用摄像头..." << std::endl;
        
        for (int i = 0; i <= 5; i++) {  // 测试video0到video5
            std::cout << "尝试通用摄像头 " << i << "..." << std::endl;
            camera_.open(i);  // 尝试打开通用摄像头
            if (camera_.isOpened()) {
                cv::Mat test_frame;  // 测试帧
                if (camera_.read(test_frame) && !test_frame.empty()) {
                    // 成功读取帧，摄像头可用
                    camera_id_ = i;
                    std::cout << "成功打开通用摄像头 " << i << std::endl;
                    break;  // 找到可用摄像头，跳出循环
                } else {
                    camera_.release();  // 无法读取帧，释放摄像头
                }
            }
        }
        
        if (!camera_.isOpened()) {
            std::cout << "所有摄像头都无法打开，初始化失败" << std::endl;
            // 确保资源被清理
            initialized_.store(false);
            running_.store(false);
            return false;  // 所有摄像头都无法使用，初始化失败
        }
    }
    
    // 设置摄像头参数 - 参考SharedCameraManager的参数设置
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, width_);    // 设置图像宽度
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);  // 设置图像高度
    camera_.set(cv::CAP_PROP_FPS, 30);                // 设置帧率为30FPS
    camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);          // 设置缓冲区大小为1，减少延迟
    
    // 验证设置 - 检查实际设置的参数是否与期望一致
    int actual_width = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = camera_.get(cv::CAP_PROP_FPS);
    
    std::cout << "摄像头参数设置完成:" << std::endl;
    std::cout << "  分辨率: " << actual_width << "x" << actual_height << std::endl;
    std::cout << "  帧率: " << actual_fps << " FPS" << std::endl;
    
    initialized_.store(true);  // 标记为已初始化
    
    // 启动帧捕获线程
    if (!startFrameCapture()) {
        std::cout << "启动帧捕获失败" << std::endl;
        camera_.release();
        initialized_.store(false);
        return false;
    }
    
    std::cout << "摄像头资源池初始化成功" << std::endl;
    return true;  // 初始化成功
}

// 注册客户端 - 参考多窗口检测的线程管理
// 参数：client_name-客户端名称
// 返回值：客户端ID字符串，失败返回空字符串
// 功能：为新的检测程序注册客户端，分配唯一的客户端ID和帧队列
std::string CameraResourcePool::registerClient(const std::string& client_name) {
    std::lock_guard<std::mutex> lock(clients_mutex_);  // 线程安全锁
    
    if (!initialized_.load()) {
        std::cout << "摄像头资源池未初始化，无法注册客户端" << std::endl;
        return "";  // 资源池未初始化，无法注册
    }
    
    // 生成客户端ID - 参考多窗口检测的任务命名
    int client_id_num = next_client_id_.fetch_add(1);  // 原子操作获取唯一ID
    std::string client_id = client_name + "_" + std::to_string(client_id_num);  // 组合客户端ID
    
    // 创建客户端 - 参考多窗口检测的DetectionResult结构
    auto client = std::make_shared<CameraClient>(client_id, client_name, 5);  // 创建客户端对象，队列大小5
    clients_[client_id] = client;  // 添加到客户端映射表
    
    std::cout << "客户端注册成功: " << client_id << " (" << client_name << ")" << std::endl;
    std::cout << "当前活跃客户端数量: " << clients_.size() << std::endl;
    
    return client_id;  // 返回客户端ID
}

// 注销客户端 - 参考MultiWindowDetector的资源清理
// 参数：client_id-客户端ID
// 返回值：成功返回true，失败返回false
// 功能：注销指定的客户端，清理相关资源
bool CameraResourcePool::unregisterClient(const std::string& client_id) {
    std::lock_guard<std::mutex> lock(clients_mutex_);  // 线程安全锁
    
    auto it = clients_.find(client_id);  // 查找客户端
    if (it != clients_.end()) {
        // 标记客户端为非活跃 - 参考多窗口检测的线程停止
        it->second->active.store(false);  // 原子操作标记为非活跃
        
        // 通知等待的线程 - 参考多窗口检测的条件变量通知
        it->second->queue_cv.notify_all();  // 唤醒所有等待的线程
        
        // 从客户端列表中移除
        clients_.erase(it);  // 从映射表中删除客户端
        
        std::cout << "客户端注销成功: " << client_id << std::endl;
        std::cout << "当前活跃客户端数量: " << clients_.size() << std::endl;
        return true;  // 注销成功
    }
    
    std::cout << "客户端不存在: " << client_id << std::endl;
    return false;  // 客户端不存在
}

// 检查客户端是否活跃
bool CameraResourcePool::isClientActive(const std::string& client_id) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    auto it = clients_.find(client_id);
    return it != clients_.end() && it->second->active.load();
}

// 获取帧 - 参考多窗口检测的getFrameFromQueue
// 参数：client_id-客户端ID，frame-输出帧数据，timeout_ms-超时时间（毫秒）
// 返回值：成功返回true，失败返回false
// 功能：从指定客户端的帧队列中获取一帧图像数据
bool CameraResourcePool::getFrame(const std::string& client_id, std::shared_ptr<cv::Mat>& frame, int timeout_ms) {
    std::shared_ptr<CameraClient> client;
    
    // 获取客户端 - 参考多窗口检测的客户端管理
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);  // 保护客户端映射表
        auto it = clients_.find(client_id);  // 查找客户端
        if (it == clients_.end() || !it->second->active.load()) {
            return false;  // 客户端不存在或非活跃
        }
        client = it->second;  // 获取客户端对象
    }
    
    // 从客户端队列获取帧 - 参考多窗口检测的帧获取机制
    std::unique_lock<std::mutex> lock(client->queue_mutex);  // 锁定客户端队列
    
    if (client->queue_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                                 [&client] { return !client->frame_queue.empty() || !client->active.load(); })) {
        // 等待队列中有帧或客户端变为非活跃
        
        if (!client->active.load()) {
            return false;  // 客户端已非活跃
        }
        
        if (!client->frame_queue.empty()) {
            frame = client->frame_queue.front();  // 获取队列中的第一帧
            client->frame_queue.pop();            // 从队列中移除
            client->last_access = std::chrono::high_resolution_clock::now();  // 更新最后访问时间
            return true;  // 成功获取帧
        }
    }
    
    return false;  // 超时或队列为空
}

// 获取活跃客户端数量
int CameraResourcePool::getActiveClientCount() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return static_cast<int>(clients_.size());
}

// 获取活跃客户端列表
std::vector<std::string> CameraResourcePool::getActiveClients() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    std::vector<std::string> active_clients;
    for (const auto& [client_id, client] : clients_) {
        if (client->active.load()) {
            active_clients.push_back(client_id);
        }
    }
    
    return active_clients;
}

// 检查是否已初始化
bool CameraResourcePool::isInitialized() const {
    return initialized_.load();
}

// 启动帧捕获 - 参考MultiWindowDetector::start()
bool CameraResourcePool::startFrameCapture() {
    if (running_.load()) {
        std::cout << "帧捕获已经在运行，新程序加入共享" << std::endl;
        return true;
    }
    
    if (!initialized_.load()) {
        std::cout << "摄像头资源池未初始化，无法启动帧捕获" << std::endl;
        return false;
    }
    
    // 如果摄像头没有被实际初始化，启动失败
    if (!camera_.isOpened()) {
        std::cout << "摄像头未打开，无法启动帧捕获线程" << std::endl;
        return false;
    }
    
    running_.store(true);
    
    // 启动帧捕获线程 - 参考多窗口检测的线程启动
    frame_capture_thread_ = std::thread(&CameraResourcePool::frameCaptureWorker, this);
    
    // 启动帧分发线程 - 参考多窗口检测的帧分发
    frame_distribution_thread_ = std::thread(&CameraResourcePool::frameDistributionWorker, this);
    
    // 启动客户端清理线程 - 参考多窗口检测的资源管理
    client_cleanup_thread_ = std::thread(&CameraResourcePool::clientCleanupWorker, this);
    
    std::cout << "帧捕获线程启动成功" << std::endl;
    return true;
}

// 停止帧捕获 - 参考MultiWindowDetector::stop()
void CameraResourcePool::stopFrameCapture() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // 通知所有等待的线程 - 参考多窗口检测的线程停止
    raw_frame_cv_.notify_all();
    
    // 等待线程结束 - 参考多窗口检测的线程等待
    if (frame_capture_thread_.joinable()) {
        frame_capture_thread_.join();
    }
    
    if (frame_distribution_thread_.joinable()) {
        frame_distribution_thread_.join();
    }
    
    if (client_cleanup_thread_.joinable()) {
        client_cleanup_thread_.join();
    }
    
    std::cout << "帧捕获线程停止" << std::endl;
}

// 清理资源 - 参考MultiWindowDetector的析构函数
void CameraResourcePool::cleanup() {
    stopFrameCapture();
    
    if (camera_.isOpened()) {
        camera_.release();
        std::cout << "摄像头资源释放" << std::endl;
    }
    
    initialized_.store(false);
}

// 帧捕获工作线程 - 参考MultiWindowDetector::detectionWorker
// 功能：独立线程持续从摄像头捕获帧，并放入原始帧队列
void CameraResourcePool::frameCaptureWorker() {
    std::cout << "帧捕获线程启动" << std::endl;
    
    cv::Mat frame;  // 当前捕获的帧
    int frame_count = 0;  // 成功捕获的帧数
    int fail_count = 0;   // 失败的帧数
    int consecutive_failures = 0;  // 连续失败次数
    const int MAX_CONSECUTIVE_FAILURES = 100;  // 连续失败100次后尝试重连
    
    while (running_.load()) {  // 主循环，直到停止标志
        if (!camera_.isOpened()) {
            std::cout << "摄像头未打开，帧捕获线程退出" << std::endl;
            break;  // 摄像头未打开，退出线程
        }
        
        // 捕获帧 - 参考多窗口检测的帧捕获
        if (camera_.read(frame)) {  // 尝试从摄像头读取帧
            if (!frame.empty()) {   // 帧不为空
                // 将帧添加到原始帧队列 - 参考多窗口检测的队列管理
                {
                    std::lock_guard<std::mutex> lock(raw_frame_mutex_);  // 保护原始帧队列
                    if (raw_frame_queue_.size() >= MAX_RAW_FRAME_QUEUE_SIZE) {
                        raw_frame_queue_.pop();  // 队列已满，移除最旧的帧
                    }
                    raw_frame_queue_.push(frame.clone());  // 克隆帧并添加到队列
                }
                raw_frame_cv_.notify_one();  // 通知帧分发线程
                
                total_frames_captured_.fetch_add(1);  // 原子操作增加总帧数
                frame_count++;                        // 增加成功帧数
                consecutive_failures = 0;             // 重置连续失败计数
                
                // 每100帧打印一次状态 - 参考多窗口检测的状态显示
                if (frame_count % 100 == 0) {
                    std::cout << "已捕获 " << frame_count << " 帧，失败 " << fail_count << " 次" << std::endl;
                }
            } else {
                fail_count++;        // 帧为空，增加失败计数
                consecutive_failures++;
            }
        } else {
            fail_count++;            // 读取失败，增加失败计数
            consecutive_failures++;
            
            if (fail_count % 50 == 0) {
                std::cout << "帧捕获失败 " << fail_count << " 次" << std::endl;
            }
            
            // 如果连续失败次数过多，尝试重新初始化摄像头
            if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
                std::cout << "连续失败 " << consecutive_failures << " 次，尝试重新初始化摄像头..." << std::endl;
                
                // 释放当前摄像头
                camera_.release();
                
                // 等待一段时间让硬件稳定
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                
                // 尝试重新打开摄像头
                std::vector<int> backends = {cv::CAP_V4L2, cv::CAP_ANY};
                bool camera_reopened = false;
                
                for (int backend : backends) {
                    camera_.open(camera_id_, backend);
                    if (camera_.isOpened()) {
                        // 重新设置摄像头参数
                        camera_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
                        camera_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
                        camera_.set(cv::CAP_PROP_FPS, 30);
                        camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
                        
                        std::cout << "摄像头重新初始化成功，使用后端 " << backend << std::endl;
                        camera_reopened = true;
                        consecutive_failures = 0;  // 重置连续失败计数
                        break;
                    }
                }
                
                if (!camera_reopened) {
                    std::cout << "摄像头重新初始化失败，帧捕获线程退出" << std::endl;
                    break;
                }
            }
        }
        
        // 控制帧率 - 参考多窗口检测的帧率控制
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_CAPTURE_INTERVAL_MS));
    }
    
    std::cout << "帧捕获线程结束，总共捕获 " << frame_count << " 帧" << std::endl;
}

// 帧分发工作线程 - 参考多窗口检测的帧分发机制
void CameraResourcePool::frameDistributionWorker() {
    std::cout << "帧分发线程启动" << std::endl;
    
    while (running_.load()) {
        cv::Mat frame;
        
        // 从原始帧队列获取帧 - 参考多窗口检测的帧获取
        {
            std::unique_lock<std::mutex> lock(raw_frame_mutex_);
            if (raw_frame_cv_.wait_for(lock, std::chrono::milliseconds(100), 
                                      [this] { return !raw_frame_queue_.empty() || !running_.load(); })) {
                
                if (!running_.load()) {
                    break;
                }
                
                if (!raw_frame_queue_.empty()) {
                    frame = raw_frame_queue_.front();
                    raw_frame_queue_.pop();
                }
            }
        }
        
        if (!frame.empty()) {
            // 分发帧到所有活跃客户端 - 参考多窗口检测的结果分发
            distributeFrame(frame);
            total_frames_distributed_.fetch_add(1);
        }
    }
    
    std::cout << "帧分发线程结束" << std::endl;
}

// 客户端清理工作线程 - 参考多窗口检测的资源管理
void CameraResourcePool::clientCleanupWorker() {
    std::cout << "客户端清理线程启动" << std::endl;
    
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(CLIENT_CLEANUP_INTERVAL_MS));
        
        if (running_.load()) {
            cleanupInactiveClients();
        }
    }
    
    std::cout << "客户端清理线程结束" << std::endl;
}

// 分发帧到所有活跃客户端 - 参考多窗口检测的结果分发
void CameraResourcePool::distributeFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    for (auto& [client_id, client] : clients_) {
        if (!client->active.load()) {
            continue;
        }
        
        std::lock_guard<std::mutex> client_lock(client->queue_mutex);
        
        // 如果队列已满，移除最旧的帧 - 参考多窗口检测的队列管理
        while (client->frame_queue.size() >= client->max_queue_size) {
            client->frame_queue.pop();
        }
        
        // 添加新帧 - 参考多窗口检测的智能指针使用
        client->frame_queue.push(std::make_shared<cv::Mat>(frame.clone()));
        // 投喂帧即视为活跃，刷新心跳，避免被误清理
        client->last_access = std::chrono::high_resolution_clock::now();
        client->queue_cv.notify_one();
    }
}

// 清理非活跃客户端 - 参考MultiWindowDetector的资源清理
void CameraResourcePool::cleanupInactiveClients() {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    auto current_time = std::chrono::high_resolution_clock::now();
    auto it = clients_.begin();
    
    while (it != clients_.end()) {
        auto& [client_id, client] = *it;
        
        // 宽松判定：有队列帧或最近有心跳的不清理
        bool should_erase = false;
        {
            std::lock_guard<std::mutex> qlock(client->queue_mutex);
            if (!client->frame_queue.empty()) {
                should_erase = false;  // 队列里还有帧，认为活跃
            } else {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - client->last_access).count();
                should_erase = (!client->active.load() || elapsed > CLIENT_TIMEOUT_MS);
            }
        }
        
        if (should_erase) {
            std::cout << "清理非活跃客户端: " << client_id << std::endl;
            it = clients_.erase(it);
        } else {
            ++it;
        }
    }
}

// 打印状态信息 - 参考多窗口检测的状态显示
void CameraResourcePool::printStatus() const {
    std::cout << "\n=== 摄像头资源池状态 ===" << std::endl;
    std::cout << "初始化状态: " << (initialized_.load() ? "已初始化" : "未初始化") << std::endl;
    std::cout << "运行状态: " << (running_.load() ? "运行中" : "已停止") << std::endl;
    std::cout << "摄像头ID: " << camera_id_ << std::endl;
    std::cout << "分辨率: " << width_ << "x" << height_ << std::endl;
    std::cout << "活跃客户端数量: " << getActiveClientCount() << std::endl;
    std::cout << "总捕获帧数: " << total_frames_captured_.load() << std::endl;
    std::cout << "总分发帧数: " << total_frames_distributed_.load() << std::endl;
    
    auto active_clients = getActiveClients();
    if (!active_clients.empty()) {
        std::cout << "活跃客户端列表:" << std::endl;
        for (const auto& client_id : active_clients) {
            std::cout << "  - " << client_id << std::endl;
        }
    }
    std::cout << "========================\n" << std::endl;
}

// 便捷的全局函数实现 - 参考多窗口检测的接口设计
// 功能：提供简化的全局接口，避免直接使用单例模式
namespace CameraPool {
    // 注册客户端 - 便捷接口
    std::string registerClient(const std::string& client_name) {
        return CameraResourcePool::getInstance().registerClient(client_name);
    }
    
    // 注销客户端 - 便捷接口
    bool unregisterClient(const std::string& client_id) {
        return CameraResourcePool::getInstance().unregisterClient(client_id);
    }
    
    // 获取帧 - 便捷接口
    bool getFrame(const std::string& client_id, std::shared_ptr<cv::Mat>& frame, int timeout_ms) {
        return CameraResourcePool::getInstance().getFrame(client_id, frame, timeout_ms);
    }
    
    // 检查客户端是否活跃 - 便捷接口
    bool isClientActive(const std::string& client_id) {
        return CameraResourcePool::getInstance().isClientActive(client_id);
    }
    
    // 获取活跃客户端数量 - 便捷接口
    int getActiveClientCount() {
        return CameraResourcePool::getInstance().getActiveClientCount();
    }
    
    // 获取活跃客户端列表 - 便捷接口
    std::vector<std::string> getActiveClients() {
        return CameraResourcePool::getInstance().getActiveClients();
    }
    
    // 打印状态信息 - 便捷接口
    void printStatus() {
        CameraResourcePool::getInstance().printStatus();
    }
}
