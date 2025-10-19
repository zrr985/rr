#include "multi_thread_detection_system.h"
#include <cstring>

// 全局运行标志
std::atomic<bool> g_running(true);

// 信号处理函数 - 处理程序退出信号
void signal_handler(int sig) {
    printf("\n收到退出信号，正在停止...\n");
    g_running = false;
}

// 中文文本绘制函数 - 解决OpenCV中文显示问题
void putChineseText(cv::Mat& img, const std::string& text, cv::Point pos, 
                   cv::Scalar color, double font_scale, int thickness) {
    std::string english_text;
    
    if (text.find("范喆洋") != std::string::npos) {
        english_text = "Fan Zheyang";
    } else if (text.find("陈俊杰") != std::string::npos) {
        english_text = "Chen Junjie";
    } else if (text.find("张蕊蕊") != std::string::npos) {
        english_text = "Zhang Ruirui";
    } else if (text.find("安全帽") != std::string::npos) {
        english_text = "Helmet";
    } else if (text.find("无安全帽") != std::string::npos) {
        english_text = "No Helmet";
    } else if (text.find("火焰") != std::string::npos) {
        english_text = "Flame";
    } else if (text.find("吸烟") != std::string::npos) {
        english_text = "Smoking";
    } else if (text.find("人脸") != std::string::npos) {
        english_text = "Face";
    } else if (text.find("仪表") != std::string::npos) {
        english_text = "Meter";
    } else if (text.find("FPS") != std::string::npos) {
        english_text = text;
    } else if (text.find("队列") != std::string::npos) {
        english_text = "Queue";
    } else if (text.find("处理时间") != std::string::npos) {
        english_text = "Time";
    } else if (text.find("任务") != std::string::npos) {
        english_text = "Task";
    } else {
        if (text.length() > 0 && (unsigned char)text[0] > 127) {
            size_t space_pos = text.find(' ');
            if (space_pos != std::string::npos) {
                std::string confidence = text.substr(space_pos + 1);
                english_text = "Face " + confidence;
            } else {
                english_text = "Face";
            }
        } else {
            english_text = text;
        }
    }
    
    cv::putText(img, english_text, pos, cv::FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness);
}

// 绘制检测结果函数
void draw_detection_results(cv::Mat& frame, const object_detect_result_list& results, 
                           const std::string& task_name) {
    for (int i = 0; i < results.count; i++) {
        const object_detect_result* det_result = &(results.results[i]);
        
        const char* class_name = nullptr;
        cv::Scalar color;
        
        if (task_name == "helmet") {
            class_name = coco_cls_to_name(det_result->cls_id);
            color = (det_result->cls_id == 0) ? NO_HELMET_COLOR : HELMET_COLOR;
        } else if (task_name == "flame") {
            class_name = flame_cls_to_name(det_result->cls_id);
            if (strcmp(class_name, "null") == 0) continue;
            color = FLAME_COLOR;
        } else if (task_name == "smoking") {
            if (det_result->cls_id == 2) {
                class_name = "smoking";
            } else if (det_result->cls_id == 0 || det_result->cls_id == 1) {
                bool has_face = false, has_cigarette = false;
                for (int j = 0; j < results.count; j++) {
                    if (results.results[j].cls_id == 1) has_face = true;
                    if (results.results[j].cls_id == 0) has_cigarette = true;
                }
                if (has_face && has_cigarette) {
                    class_name = "smoking";
                } else {
                    continue;
                }
            }
            color = SMOKING_COLOR;
        } else if (task_name == "face") {
            class_name = face_cls_to_name(det_result->cls_id);
            color = FACE_COLOR;
        } else if (task_name == "meter") {
            class_name = meter_cls_to_name(det_result->cls_id);
            color = METER_COLOR;
        } else {
            class_name = "unknown";
            color = cv::Scalar(255, 255, 255);
        }
        
        if (class_name) {
            cv::rectangle(frame, 
                         cv::Point(det_result->box.left, det_result->box.top),
                         cv::Point(det_result->box.right, det_result->box.bottom),
                         color, 2);
            
            char confidence_text[32];
            sprintf(confidence_text, "%.2f", det_result->prop);
            std::string label = std::string(class_name) + " " + confidence_text;
            
            putChineseText(frame, label,
                          cv::Point(det_result->box.left, det_result->box.top - 10),
                          color, 0.6, 2);
        }
    }
}

// MultiThreadDetectionSystem 方法实现
bool MultiThreadDetectionSystem::initialize() {
    camera_manager_ = GlobalCameraManager::getInstance().getCamera(camera_id_);
    if (!camera_manager_) {
        std::cout << "无法初始化摄像头 " << camera_id_ << std::endl;
        return false;
    }
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "多线程检测系统初始化成功，摄像头: " << camera_id_ << std::endl;
    return true;
}

void MultiThreadDetectionSystem::addTask(const std::string& task_name, const std::string& model_path) {
    DetectionTask task;
    task.name = task_name;
    task.model_path = model_path;
    task.tpes_ = 3;  // 每个任务使用3个线程推理
    
    std::cout << "初始化任务: " << task_name << "，模型: " << model_path << "，TPEs: " << task.tpes_ << std::endl;
    
    // 为每个任务创建3个RKNN应用上下文
    task.app_contexts.resize(task.tpes_);
    for (int i = 0; i < task.tpes_; i++) {
        // 初始化RKNN模型
        if (init_yolov8_model(model_path.c_str(), &task.app_contexts[i]) != 0) {
            std::cout << "错误: 无法初始化模型 " << model_path << " 用于任务 " << task_name << " 线程 " << i << std::endl;
            return;
        }
        
        // 设置NPU核心（轮询分配）
        rknn_core_mask core_mask;
        if (i == 0) core_mask = RKNN_NPU_CORE_0;
        else if (i == 1) core_mask = RKNN_NPU_CORE_1;
        else core_mask = RKNN_NPU_CORE_2;
        
        int ret = rknn_set_core_mask(task.app_contexts[i].rknn_ctx, core_mask);
        if (ret != RKNN_SUCC) {
            std::cout << "警告: 无法设置NPU核心掩码，使用默认核心" << std::endl;
        }
        
        std::cout << "  线程 " << i << " 绑定到NPU核心 " << i << std::endl;
    }
    
    tasks_.push_back(std::move(task));
    std::cout << "任务添加成功: " << task_name << " (3个推理线程)" << std::endl;
}

void MultiThreadDetectionSystem::start() {
    if (!camera_manager_) {
        std::cout << "错误: 摄像头管理器未初始化" << std::endl;
        return;
    }
    
    if (tasks_.empty()) {
        std::cout << "错误: 没有添加任何检测任务" << std::endl;
        return;
    }
    
    // 启动摄像头
    if (!camera_manager_->initCamera(camera_id_)) {
        std::cout << "错误: 无法启动摄像头" << std::endl;
        return;
    }
    camera_manager_->startFrameCapture();
    
    system_running_ = true;
    g_running = true;
    
    // 为每个任务启动线程
    for (auto& task : tasks_) {
        task.running = true;
        task.fps_counter.start();
        task.thread = std::thread(&MultiThreadDetectionSystem::taskWorker, this, std::ref(task));
        std::cout << "启动检测任务线程: " << task.name << std::endl;
    }
    
    std::cout << "多线程检测系统启动成功，运行 " << tasks_.size() << " 个检测任务" << std::endl;
    std::cout << "按 'q' 键在任何窗口退出程序" << std::endl;
}

void MultiThreadDetectionSystem::stop() {
    if (!system_running_) return;
    
    std::cout << "正在停止多线程检测系统..." << std::endl;
    
    system_running_ = false;
    g_running = false;
    
    // 停止所有任务线程
    for (auto& task : tasks_) {
        task.running = false;
        if (task.thread.joinable()) {
            task.thread.join();
            std::cout << "任务线程停止: " << task.name << std::endl;
        }
        // 释放所有RKNN上下文
        for (auto& app_ctx : task.app_contexts) {
            release_yolov8_model(&app_ctx);
        }
    }
    
    // 停止摄像头
    if (camera_manager_) {
        camera_manager_->stopFrameCapture();
        camera_manager_->release();
    }
    
    // 销毁所有OpenCV窗口
    cv::destroyAllWindows();
    
    std::cout << "多线程检测系统已完全停止" << std::endl;
}

void MultiThreadDetectionSystem::taskWorker(DetectionTask& task) {
    std::string window_name = task.name + " Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    
    std::cout << "任务线程运行: " << task.name << std::endl;
    
    while (task.running && system_running_ && g_running) {
        // 从摄像头获取帧
        cv::Mat frame;
        if (!camera_manager_->getFrameFromQueue(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }
        
        // 准备推理输入
        image_buffer_t src_img;
        src_img.width = frame.cols;
        src_img.height = frame.rows;
        src_img.format = IMAGE_FORMAT_RGB888;
        src_img.virt_addr = frame.data;
        src_img.size = frame.cols * frame.rows * 3;
        
        // 执行推理（使用轮询选择RKNN上下文）
        auto start_time = std::chrono::high_resolution_clock::now();
        object_detect_result_list results;
        
        // 轮询选择RKNN上下文，实现负载均衡
        static int context_index = 0;
        int selected_context = context_index % task.app_contexts.size();
        context_index++;
        
        int ret = inference_yolov8_model(&task.app_contexts[selected_context], &src_img, &results);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;
        
        if (ret == 0) {
            // 更新检测计数
            if (results.count > 0) {
                task.detection_count.fetch_add(results.count);
            }
            
            // 绘制检测结果
            cv::Mat display_frame = frame.clone();
            draw_detection_results(display_frame, results, task.name);
            
            // 更新FPS
            task.fps_counter.update();
            
            // 添加统计信息
            char info_text[256];
            sprintf(info_text, "FPS: %.1f | Detections: %d", 
                   task.fps_counter.get_fps(), task.detection_count.load());
            cv::putText(display_frame, info_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            sprintf(info_text, "Task: %s | Core: %d | Time: %.1fms", 
                   task.name.c_str(), selected_context, processing_time);
            cv::putText(display_frame, info_text, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            sprintf(info_text, "TPEs: %d | NPU Cores: 0,1,2", task.tpes_);
            cv::putText(display_frame, info_text, cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            
            // 显示帧
            cv::imshow(window_name, display_frame);
        }
        
        // 处理按键输入（任何窗口按'q'都会退出）
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            g_running = false;
            system_running_ = false;
            break;
        }
    }
    
    cv::destroyWindow(window_name);
    std::cout << "任务线程结束: " << task.name << std::endl;
}

void MultiThreadDetectionSystem::printStatistics() const {
    std::cout << "\n=== 检测统计 ===" << std::endl;
    for (const auto& task : tasks_) {
        std::cout << task.name << ": " << task.detection_count.load() << " 次检测" << std::endl;
    }
    std::cout << "=================" << std::endl;
}

// 主函数 - 多线程RKNN实时检测系统
int main(int argc, char** argv) {
    std::cout << "=== 多线程RKNN实时检测系统 ===" << std::endl;
    
    // 解析命令行参数
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " [任务配置...]" << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " --helmet ../model/helmet.rknn --flame ../model/fire.rknn" << std::endl;
        std::cout << "  " << argv[0] << " --helmet ../model/helmet.rknn --flame ../model/fire.rknn --smoking ../model/smoking.rknn" << std::endl;
        std::cout << "  " << argv[0] << " --helmet ../model/helmet.rknn --camera 1" << std::endl;
        return -1;
    }
    
    // 默认配置
    std::vector<std::pair<std::string, std::string>> task_configs;
    int camera_id = 0;
    
    // 解析参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--helmet" && i + 1 < argc) {
            task_configs.push_back({"helmet", argv[++i]});
        } else if (arg == "--flame" && i + 1 < argc) {
            task_configs.push_back({"flame", argv[++i]});
        } else if (arg == "--smoking" && i + 1 < argc) {
            task_configs.push_back({"smoking", argv[++i]});
        } else if (arg == "--face" && i + 1 < argc) {
            task_configs.push_back({"face", argv[++i]});
        } else if (arg == "--meter" && i + 1 < argc) {
            task_configs.push_back({"meter", argv[++i]});
        } else if (arg == "--camera" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else {
            std::cout << "未知参数: " << arg << std::endl;
            return -1;
        }
    }
    
    if (task_configs.empty()) {
        std::cout << "错误: 必须至少指定一个检测任务" << std::endl;
        return -1;
    }
    
    // 初始化后处理模块
    std::cout << "初始化后处理模块..." << std::endl;
    init_post_process();
    init_flame_post_process();
    init_smoking_post_process();
    // init_face_post_process();   // 如果需要人脸检测
    // init_meter_post_process();  // 如果需要仪表检测
    
    // 创建并配置多线程检测系统
    MultiThreadDetectionSystem system(camera_id);
    
    if (!system.initialize()) {
        std::cout << "系统初始化失败" << std::endl;
        return -1;
    }
    
    // 添加检测任务
    std::cout << "配置检测任务..." << std::endl;
    for (const auto& config : task_configs) {
        system.addTask(config.first, config.second);
    }
    
    // 启动系统
    std::cout << "启动多线程检测系统..." << std::endl;
    system.start();
    
    // 主循环（简单等待）
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 每10秒打印一次状态
        static auto last_print = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 10) {
            system.printStatistics();
            last_print = now;
        }
    }
    
    // 停止系统
    system.stop();
    system.printStatistics();
    
    // 清理后处理模块
    deinit_post_process();
    deinit_flame_post_process();
    deinit_smoking_post_process();
    // deinit_face_post_process();   // 如果需要人脸检测
    // deinit_meter_post_process();  // 如果需要仪表检测
    
    std::cout << "程序正常退出" << std::endl;
    return 0;
}