#include "camera.h"
#include <iostream>
#include <vector>
#include <memory>

int main(int argc, char** argv) {
    std::cout << "=== 多任务检测系统 (简化高性能版本) ===" << std::endl;
    
    // 解析命令行参数
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " [任务配置...]" << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " --helmet ../model/helmet.rknn --flame ../model/fire.rknn" << std::endl;
        std::cout << "  " << argv[0] << " --helmet ../model/helmet.rknn --flame ../model/fire.rknn --smoking ../model/smoking.rknn" << std::endl;
        return -1;
    }
    
    // 配置参数
    std::vector<std::pair<std::string, std::string>> task_configs;
    int camera_id = 0;
    int buffer_size_per_queue = 5;  // 每个消费者的队列大小
    
    // 解析参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--helmet" && i + 1 < argc) {
            task_configs.push_back({"helmet", argv[++i]});
        } else if (arg == "--flame" && i + 1 < argc) {
            task_configs.push_back({"flame", argv[++i]});
        } else if (arg == "--smoking" && i + 1 < argc) {
            task_configs.push_back({"smoking", argv[++i]});
        } else if (arg == "--camera" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (arg == "--buffer" && i + 1 < argc) {
            buffer_size_per_queue = std::stoi(argv[++i]);
        }
    }
    
    if (task_configs.empty()) {
        std::cout << "错误: 必须至少指定一个检测任务" << std::endl;
        return -1;
    }
    
    std::cout << "配置参数:" << std::endl;
    std::cout << "  摄像头ID: " << camera_id << std::endl;
    std::cout << "  每个消费者缓冲区大小: " << buffer_size_per_queue << std::endl;
    for (const auto& config : task_configs) {
        std::cout << "  任务: " << config.first << ", 模型: " << config.second << std::endl;
    }
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 初始化后处理模块
    std::cout << "初始化后处理模块..." << std::endl;
    init_post_process();  // 只需要初始化通用的后处理
    
    // 提取消费者名称列表
    std::vector<std::string> consumer_names;
    for (const auto& config : task_configs) {
        consumer_names.push_back(config.first);
    }
    
    // 创建高性能缓冲区
    std::cout << "创建高性能缓冲区..." << std::endl;
    HighPerformanceBuffer buffer(buffer_size_per_queue, consumer_names);
    
    // 创建摄像头生产者
    std::cout << "创建摄像头生产者..." << std::endl;
    CameraProducer camera_producer(camera_id, buffer);
    if (!camera_producer.initialize()) {
        std::cout << "摄像头生产者初始化失败" << std::endl;
        return -1;
    }
    
    // 创建检测消费者
    std::cout << "创建检测消费者..." << std::endl;
    std::vector<std::unique_ptr<DetectionConsumer>> consumers;
    for (const auto& config : task_configs) {
        std::cout << "初始化检测消费者: " << config.first << std::endl;
        auto consumer = std::make_unique<DetectionConsumer>(config.first, config.second, buffer);
        if (consumer->initialize()) {
            consumers.push_back(std::move(consumer));
            std::cout << "成功创建检测消费者: " << config.first << std::endl;
        } else {
            std::cout << "检测消费者初始化失败: " << config.first << std::endl;
        }
    }
    
    if (consumers.empty()) {
        std::cout << "错误: 没有成功的检测消费者" << std::endl;
        return -1;
    }
    
    std::cout << "系统配置完成:" << std::endl;
    std::cout << "  摄像头: " << camera_id << std::endl;
    std::cout << "  每个消费者缓冲区大小: " << buffer_size_per_queue << std::endl;
    std::cout << "  检测任务: " << consumers.size() << " 个" << std::endl;
    std::cout << "  NPU核心分配: 每个任务使用3个NPU核心" << std::endl;
    std::cout << "按 'q' 键在任何窗口退出程序" << std::endl;
    
    // 创建并启动显示管理器
    std::cout << "创建显示管理器..." << std::endl;
    g_display_manager = std::make_unique<DisplayManager>();
    g_display_manager->start();
    
    // 启动系统
    std::cout << "启动系统..." << std::endl;
    camera_producer.start();
    for (auto& consumer : consumers) {
        consumer->start();
    }
    
    // 主循环
    std::cout << "系统运行中..." << std::endl;
    
    auto program_start_time = std::chrono::steady_clock::now();
    int last_total_frames = 0;
    
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 定期打印状态
        static auto last_print = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 5) {
            int current_total_frames = camera_producer.get_frame_count();
            int frames_since_last = current_total_frames - last_total_frames;
            last_total_frames = current_total_frames;
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - program_start_time).count();
            
            std::cout << "\n=== 系统状态 ===" << std::endl;
            std::cout << "运行时间: " << elapsed << " 秒" << std::endl;
            std::cout << "摄像头FPS: " << camera_producer.get_fps() << std::endl;
            std::cout << "总生产帧数: " << current_total_frames << std::endl;
            std::cout << "最近5秒帧数: " << frames_since_last << " (" << (frames_since_last / 5.0) << " FPS)" << std::endl;
            
            for (const auto& consumer : consumers) {
                std::cout << consumer->get_task_name() << ": " 
                          << consumer->get_detection_count() << " 次检测 | " 
                          << consumer->get_fps() << " FPS | "
                          << "缓冲区: " << buffer.get_size(consumer->get_task_name()) << std::endl;
            }
            std::cout << "===============" << std::endl;
            last_print = now;
        }
    }
    
    // 优雅停止系统
    std::cout << "正在停止系统..." << std::endl;
    
    // 先停止显示管理器
    if (g_display_manager) {
        g_display_manager->stop();
    }
    
    // 停止生产者
    camera_producer.stop();
    
    // 停止所有消费者
    for (auto& consumer : consumers) {
        consumer->stop();
    }
    
    // 等待线程结束
    camera_producer.join();
    for (auto& consumer : consumers) {
        consumer->join();
    }
    
    // 打印最终统计
    std::cout << "\n=== 最终统计 ===" << std::endl;
    std::cout << "总生产帧数: " << camera_producer.get_frame_count() << std::endl;
    for (const auto& consumer : consumers) {
        std::cout << consumer->get_task_name() << ": " 
                  << consumer->get_detection_count() << " 次检测" << std::endl;
    }
    
    // 清理后处理模块
    deinit_post_process();
    
    std::cout << "程序正常退出" << std::endl;
    return 0;
}
