#include "camera.h"
#include <signal.h>

// 全局运行标志
std::atomic<bool> g_running(true);

// 全局显示管理器
std::unique_ptr<DisplayManager> g_display_manager = nullptr;

// 信号处理函数
void signal_handler(int sig) {
    std::cout << "\n接收到退出信号，正在停止..." << std::endl;
    g_running = false;
}

// DisplayManager 实现
DisplayManager::DisplayManager() {
}

DisplayManager::~DisplayManager() {
    stop();
}

void DisplayManager::start() {
    if (running_) return;
    
    running_ = true;
    display_thread_ = std::thread(&DisplayManager::display_worker, this);
    printf("显示管理器启动\n");
}

void DisplayManager::stop() {
    if (!running_) return;
    
    running_ = false;
    cv_.notify_all();
    
    if (display_thread_.joinable()) {
        display_thread_.join();
    }
    
    // 销毁所有窗口
    for (const auto& [name, created] : window_created_) {
        if (created) {
            cv::destroyWindow(name);
        }
    }
    
    printf("显示管理器停止\n");
}

void DisplayManager::update_display(const std::string& window_name, const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 如果队列太大，移除旧帧
    while (display_queue_.size() > 10) {
        display_queue_.pop();
    }
    
    DisplayTask task;
    task.window_name = window_name;
    task.frame = frame.clone();
    
    display_queue_.push(task);
    cv_.notify_one();
}

void DisplayManager::display_worker() {
    printf("显示工作线程启动\n");
    
    while (running_ && g_running) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待显示任务
        if (cv_.wait_for(lock, std::chrono::milliseconds(10), 
            [this] { return !display_queue_.empty() || !running_ || !g_running; })) {
            
            if (!running_ || !g_running) {
                break;
            }
            
            if (display_queue_.empty()) {
                continue;
            }
            
            DisplayTask task = display_queue_.front();
            display_queue_.pop();
            lock.unlock();
            
            // 创建窗口（如果还没创建）
            if (window_created_.find(task.window_name) == window_created_.end()) {
                cv::namedWindow(task.window_name, cv::WINDOW_AUTOSIZE);
                window_created_[task.window_name] = true;
            }
            
            // 显示帧
            cv::imshow(task.window_name, task.frame);
            
            // 处理按键
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                g_running = false;
                break;
            }
        }
    }
    
    printf("显示工作线程结束\n");
}

// 统一的后处理绘制函数
void draw_detection_results(cv::Mat& frame, const object_detect_result_list& results, const std::string& task_name) {
    for (int i = 0; i < results.count; i++) {
        const auto& det = results.results[i];
        
        std::string class_name;
        cv::Scalar color;
        
        // 根据任务名称和类别ID确定类别名称和颜色
        if (task_name == "helmet") {
            // 安全帽检测：根据helmet_labels_list.txt，0=no_helmet, 1=helmet
            if (det.cls_id == 0) {
                class_name = "no_helmet";      // cls_id=0 对应标签文件第1行
                color = cv::Scalar(0, 0, 255); // 红色 - 危险，没戴安全帽
            } else if (det.cls_id == 1) {
                class_name = "helmet";         // cls_id=1 对应标签文件第2行
                color = cv::Scalar(0, 255, 0); // 绿色 - 安全，戴了安全帽
            } else {
                continue; // 跳过未知类别
            }
        } else if (task_name == "flame") {
            // 火焰检测
            if (det.cls_id == 0) {
                class_name = "flame";
                color = cv::Scalar(0, 100, 255); // 橙色
            } else {
                continue; // 跳过未知类别
            }
        } else if (task_name == "smoking") {
            // 吸烟检测：0=cigarette, 1=face, 2=smoking
            if (det.cls_id == 0) {
                class_name = "cigarette";
                color = cv::Scalar(0, 255, 255); // 黄色
            } else if (det.cls_id == 1) {
                class_name = "face";
                color = cv::Scalar(255, 255, 0); // 青色
            } else if (det.cls_id == 2) {
                class_name = "smoking";
                color = cv::Scalar(255, 0, 0); // 蓝色
            } else {
                continue; // 跳过未知类别
            }
        } else {
            // 默认处理
            class_name = "object_" + std::to_string(det.cls_id);
            color = cv::Scalar(128, 128, 128); // 灰色
        }
        
        // 绘制边界框
        cv::Point pt1(det.box.left, det.box.top);
        cv::Point pt2(det.box.right, det.box.bottom);
        cv::rectangle(frame, pt1, pt2, color, 2);
        
        // 绘制标签
        std::string label = class_name + " " + std::to_string((int)(det.prop * 100)) + "%";
        cv::Point text_pt(det.box.left, det.box.top - 10);
        cv::putText(frame, label, text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

// HighPerformanceBuffer 实现
HighPerformanceBuffer::HighPerformanceBuffer(size_t max_size_per_queue, const std::vector<std::string>& consumer_names) {
    for (const auto& name : consumer_names) {
        auto queue = std::make_shared<ConsumerQueue>();
        queue->max_size = max_size_per_queue;
        consumer_queues_[name] = queue;
    }
}

bool HighPerformanceBuffer::produce(const cv::Mat& frame) {
    auto frame_ptr = std::make_shared<cv::Mat>(frame.clone());
    bool all_success = true;
    
    for (auto& [name, queue] : consumer_queues_) {
        std::unique_lock<std::mutex> lock(queue->mutex);
        
        // 如果队列满了，移除最旧的帧
        if (queue->queue.size() >= queue->max_size) {
            queue->queue.pop();
        }
        
        queue->queue.push(frame_ptr);
        queue->has_frame = true;
        queue->cv.notify_one();
    }
    
    return all_success;
}

FramePtr HighPerformanceBuffer::consume(const std::string& consumer_name) {
    auto it = consumer_queues_.find(consumer_name);
    if (it == consumer_queues_.end()) {
        return nullptr;
    }
    
    auto& queue = it->second;
    std::unique_lock<std::mutex> lock(queue->mutex);
    
    // 等待有帧可用
    if (queue->cv.wait_for(lock, std::chrono::microseconds(100), 
        [&queue] { return queue->has_frame || !g_running; })) {
        
        if (!g_running || queue->queue.empty()) {
            return nullptr;
        }
        
        FramePtr frame_ptr = queue->queue.front();
        queue->queue.pop();
        queue->has_frame = !queue->queue.empty();
        
        return frame_ptr;
    }
    return nullptr;
}

size_t HighPerformanceBuffer::get_size(const std::string& consumer_name) const {
    auto it = consumer_queues_.find(consumer_name);
    if (it == consumer_queues_.end()) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(it->second->mutex);
    return it->second->queue.size();
}

// DetectionConsumer 实现 - 简化版本
DetectionConsumer::DetectionConsumer(const std::string& task_name, 
                                   const std::string& model_path, 
                                   HighPerformanceBuffer& buffer)
    : task_name_(task_name), model_path_(model_path), buffer_(buffer) {
}

DetectionConsumer::~DetectionConsumer() {
    stop();
    join();
    
    // 释放RKNN上下文
    for (auto& app_ctx : app_contexts_) {
        if (app_ctx.rknn_ctx != 0) {
            release_yolov8_model(&app_ctx);
        }
    }
}

bool DetectionConsumer::initialize() {
    printf("初始化检测消费者: %s\n", task_name_.c_str());
    
    // 初始化3个RKNN上下文，每个绑定到不同的NPU核心
    app_contexts_.resize(3);
    
    for (int i = 0; i < 3; i++) {
        if (init_yolov8_model(model_path_.c_str(), &app_contexts_[i]) != 0) {
            printf("错误: 无法初始化模型 %s (上下文 %d)\n", model_path_.c_str(), i);
            // 清理已初始化的上下文
            for (int j = 0; j < i; j++) {
                release_yolov8_model(&app_contexts_[j]);
            }
            return false;
        }
        
        // 设置NPU核心掩码
        rknn_core_mask core_mask;
        if (i == 0) {
            core_mask = RKNN_NPU_CORE_0;
        } else if (i == 1) {
            core_mask = RKNN_NPU_CORE_1;
        } else {
            core_mask = RKNN_NPU_CORE_2;
        }
        
        int ret = rknn_set_core_mask(app_contexts_[i].rknn_ctx, core_mask);
        if (ret != RKNN_SUCC) {
            printf("警告: 无法设置NPU核心掩码 (上下文 %d)\n", i);
        }
        
        printf("  RKNN上下文 %d 绑定到NPU核心 %d\n", i, i);
    }
    
    printf("检测消费者 %s 初始化成功\n", task_name_.c_str());
    return true;
}

void DetectionConsumer::start() {
    if (running_) return;
    
    running_ = true;
    thread_ = std::thread(&DetectionConsumer::worker_thread, this);
    printf("启动检测消费者: %s\n", task_name_.c_str());
}

void DetectionConsumer::stop() {
    running_ = false;
}

void DetectionConsumer::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DetectionConsumer::worker_thread() {
    std::string window_name = task_name_ + " Detection";
    
    printf("检测消费者 %s 线程启动\n", task_name_.c_str());
    
    int processed_frames = 0;
    auto last_status_time = std::chrono::steady_clock::now();
    
    while (running_ && g_running) {
        // 从缓冲区获取帧
        FramePtr frame_ptr = buffer_.consume(task_name_);
        if (!frame_ptr || frame_ptr->empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        
        processed_frames++;
        
        // 准备推理输入
        image_buffer_t src_img;
        src_img.width = frame_ptr->cols;
        src_img.height = frame_ptr->rows;
        src_img.format = IMAGE_FORMAT_RGB888;
        src_img.virt_addr = frame_ptr->data;
        src_img.size = frame_ptr->cols * frame_ptr->rows * 3;
        
        // 轮询选择RKNN上下文
        int selected_context = context_index_.fetch_add(1) % app_contexts_.size();
        
        // 执行推理
        object_detect_result_list results;
        int ret = inference_yolov8_model(&app_contexts_[selected_context], &src_img, &results);
        
        if (ret == 0) {
            // 更新检测计数
            if (results.count > 0) {
                detection_count_ += results.count;
                // 进一步减少输出频率，提升性能
                if (processed_frames % 100 == 0) {
                    printf("检测到 %d 个目标 (任务: %s, 核心: %d)\n", results.count, task_name_.c_str(), selected_context);
                }
            }
            
            // 更新FPS
            fps_counter_.update();
            
            // 只在有检测结果或每30帧时更新显示
            if (results.count > 0 || processed_frames % 30 == 0) {
                cv::Mat display_frame = frame_ptr->clone();
                
                // 绘制检测结果
                if (results.count > 0) {
                    draw_detection_results(display_frame, results, task_name_);
                }
                
                // 添加统计信息
                char info_text[256];
                sprintf(info_text, "FPS: %.1f | Detections: %d", 
                       fps_counter_.get_fps(), detection_count_.load());
                cv::putText(display_frame, info_text, cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                
                sprintf(info_text, "Task: %s | Core: %d", 
                       task_name_.c_str(), selected_context);
                cv::putText(display_frame, info_text, cv::Point(10, 60),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                
                sprintf(info_text, "Buffer: %zu", buffer_.get_size(task_name_));
                cv::putText(display_frame, info_text, cv::Point(10, 90),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                
                // 使用异步显示管理器更新显示
                if (g_display_manager) {
                    g_display_manager->update_display(window_name, display_frame);
                }
            }
        }
        
        // 定期打印状态（减少打印频率）
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 10) {
            printf("消费者 %s: 已处理 %d 帧, FPS: %.1f, 检测: %d\n", 
                   task_name_.c_str(), processed_frames, fps_counter_.get_fps(), detection_count_.load());
            last_status_time = now;
        }
    }
    
    printf("检测消费者 %s 线程结束，总共处理 %d 帧\n", task_name_.c_str(), processed_frames);
}

// CameraProducer 实现 - 高性能版本
CameraProducer::CameraProducer(int camera_id, HighPerformanceBuffer& buffer)
    : camera_id_(camera_id), buffer_(buffer) {
}

CameraProducer::~CameraProducer() {
    stop();
    join();
}

bool CameraProducer::initialize() {
    printf("初始化摄像头生产者: %d (固定30FPS模式)\n", camera_id_);
    
    camera_manager_ = GlobalCameraManager::getInstance().getCamera(camera_id_);
    if (!camera_manager_) {
        printf("错误: 无法获取摄像头 %d\n", camera_id_);
        return false;
    }
    
    // 无论是否已打开，都重新初始化以确保30FPS
    if (camera_manager_->isOpened()) {
        printf("摄像头 %d 已经打开，重新初始化以确保30FPS...\n", camera_id_);
        camera_manager_->release();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    printf("初始化摄像头 %d 为30FPS...\n", camera_id_);
    
    // 固定为30FPS配置，按优先级尝试不同分辨率
    std::vector<std::tuple<int, int, int>> resolutions = {
        {640, 480, 30},   // 首选：640x480@30fps，适合YOLOv8-640
        {320, 240, 30},   // 备选：320x240@30fps，更高帧率可能性
        {1280, 720, 30},  // 备选：1280x720@30fps，高分辨率
        {1920, 1080, 30}  // 备选：1080p@30fps
    };
    
    bool init_success = false;
    for (const auto& res : resolutions) {
        int width = std::get<0>(res);
        int height = std::get<1>(res);
        int fps = std::get<2>(res);
        
        printf("尝试分辨率: %dx%d @ %dfps\n", width, height, fps);
        if (camera_manager_->initCamera(camera_id_, width, height, fps)) {
            printf("✅ 摄像头 %d 初始化成功: %dx%d @ %dfps\n", camera_id_, width, height, fps);
            init_success = true;
            break;
        } else {
            printf("❌ 分辨率 %dx%d@%dfps 初始化失败，尝试下一个\n", width, height, fps);
        }
    }
    
    if (!init_success) {
        printf("错误: 所有分辨率都无法初始化摄像头 %d\n", camera_id_);
        return false;
    }
    
    // 启动帧捕获
    camera_manager_->startFrameCapture();
    printf("启动摄像头生产者帧捕获，目标帧率: 30FPS\n");
    
    // 等待一段时间让帧捕获线程开始工作
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // 测试摄像头
    cv::Mat test_frame;
    bool test_success = false;
    
    for (int i = 0; i < 10; i++) {
        if (camera_manager_->getFrameFromQueue(test_frame) && !test_frame.empty()) {
            printf("摄像头测试成功，实际帧大小: %dx%d\n", test_frame.cols, test_frame.rows);
            test_success = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!test_success) {
        printf("错误: 摄像头测试失败\n");
        return false;
    }
    
    printf("✅ 摄像头生产者初始化成功，目标帧率: 30FPS\n");
    return true;
}

void CameraProducer::start() {
    if (running_) return;
    
    running_ = true;
    thread_ = std::thread(&CameraProducer::producer_thread, this);
    printf("启动摄像头生产者\n");
}

void CameraProducer::stop() {
    running_ = false;
}

void CameraProducer::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
    camera_manager_->stopFrameCapture();
    camera_manager_->release();
}

void CameraProducer::producer_thread() {
    printf("摄像头生产者线程启动\n");
    
    auto last_status_time = std::chrono::steady_clock::now();
    int empty_frame_count = 0;
    
    while (running_ && g_running) {
        cv::Mat frame;
        
        // 从摄像头获取帧
        if (camera_manager_->getFrameFromQueue(frame)) {
            if (!frame.empty()) {
                // 生产帧到缓冲区
                buffer_.produce(frame);
                fps_counter_.update();
                frame_count_++;
                empty_frame_count = 0;
            } else {
                empty_frame_count++;
                if (empty_frame_count > 10) {
                    printf("警告: 连续收到空帧\n");
                    empty_frame_count = 0;
                }
            }
        } else {
            // 如果没有帧可用，短暂休息
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        // 定期打印状态
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status_time).count() >= 5) {
            printf("摄像头生产者: 已生产 %d 帧, FPS: %.1f\n", 
                   frame_count_.load(), fps_counter_.get_fps());
            last_status_time = now;
        }
        
        // 不控制帧率，让生产者尽可能快
    }
    
    printf("摄像头生产者线程结束，总共生产 %d 帧\n", frame_count_.load());
}
