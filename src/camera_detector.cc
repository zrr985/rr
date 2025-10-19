#include "camera_detector.h"
#include <iostream>
#include <fstream>
#include <regex>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

// 检测特定摄像头设备编号的函数实现
// 参数：inf_numbers-红外摄像头编号列表（输出），rgb_numbers-RGB摄像头编号列表（输出）
// 功能：扫描系统中的摄像头设备，区分红外摄像头和RGB摄像头，并将设备编号分别存储到对应的vector中
void detectCameraNumbers(std::vector<int>& inf_numbers, std::vector<int>& rgb_numbers) {
    
    // 检查/sys/class/video4linux/目录 - Linux系统中视频设备的系统目录
    std::string video_dir = "/sys/class/video4linux/";
    
    try {
        // 遍历所有video设备 - 使用C++17的filesystem库遍历目录
        for (const auto& entry : std::filesystem::directory_iterator(video_dir)) {
            std::string device_path = entry.path().string();        // 设备完整路径
            std::string device_name = entry.path().filename().string(); // 设备名称（如video0）
            
            // 检查是否是video设备 - 设备名称以"video"开头
            if (device_name.substr(0, 5) == "video") {
                std::string modalias_path = device_path + "/device/modalias"; // modalias文件路径，包含设备型号信息
                
                try {
                    std::ifstream modalias_file(modalias_path);  // 打开modalias文件
                    if (modalias_file.is_open()) {
                        std::string modalias;                    // 存储modalias内容
                        std::getline(modalias_file, modalias);   // 读取第一行（modalias信息）
                        modalias_file.close();                   // 关闭文件
                        
                        // 解析modalias - modalias格式为"v4l2:型号信息"
                        size_t colon_pos = modalias.find(':');   // 查找冒号分隔符
                        if (colon_pos != std::string::npos) {
                            std::string model = modalias.substr(colon_pos + 1); // 提取型号信息部分
                            
                            // 提取video编号 - 使用正则表达式从设备名称中提取数字
                            std::regex video_regex(R"(video(\d+))");  // 匹配"video"后跟数字的模式
                            std::smatch match;                        // 存储匹配结果
                            if (std::regex_search(device_name, match, video_regex)) {
                                int video_number = std::stoi(match[1].str()); // 将匹配的数字字符串转换为整数
                                
                                // 根据model判断摄像头类型 - 通过modalias中的型号信息识别特定摄像头
                                if (model == "v1514p0001d0200dcEFdsc02dp01ic0Eisc01ip00in00") {
                                    // 红外摄像头的特定型号标识
                                    inf_numbers.push_back(video_number);
                                    std::cout << "发现红外摄像头: video" << video_number << std::endl;
                                } else if (model == "v1BCFp0C18d0508dcEFdsc02dp01ic0Eisc01ip00in00") {
                                    // RGB摄像头的特定型号标识
                                    rgb_numbers.push_back(video_number);
                                    std::cout << "发现RGB摄像头: video" << video_number << std::endl;
                                } else {
                                    // 对于其他型号的摄像头（如USB摄像头），尝试用OpenCV测试是否可用
                                    std::cout << "测试通用摄像头: video" << video_number << " (model: " << model << ")" << std::endl;
                                    cv::VideoCapture test_cap(video_number);  // 尝试打开摄像头
                                    if (test_cap.isOpened()) {
                                        cv::Mat test_frame;                    // 测试帧
                                        if (test_cap.read(test_frame) && !test_frame.empty()) {
                                            // 成功读取帧，确认为可用摄像头
                                            rgb_numbers.push_back(video_number);
                                            std::cout << "发现通用摄像头: video" << video_number << " (model: " << model << ")" << std::endl;
                                        } else {
                                            std::cout << "通用摄像头 video" << video_number << " 无法读取帧" << std::endl;
                                        }
                                    } else {
                                        std::cout << "通用摄像头 video" << video_number << " 无法打开" << std::endl;
                                    }
                                    test_cap.release();  // 释放摄像头资源
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    // 忽略无法读取的设备 - 某些设备可能没有modalias文件或权限不足
                    continue;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "无法访问/sys/class/video4linux/目录: " << e.what() << std::endl;
    }
    
    // 输出检测结果 - 显示找到的红外摄像头编号
    std::cout << "红外摄像头编号: ";
    for (int num : inf_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 输出检测结果 - 显示找到的RGB摄像头编号
    std::cout << "RGB摄像头编号: ";
    for (int num : rgb_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

// 查找可用摄像头的函数实现
// 返回值：包含所有可用摄像头设备编号的vector<int>
// 功能：扫描系统中的所有摄像头设备（通常是/dev/video*），返回可用的摄像头设备编号列表
std::vector<int> findAvailableCameras() {
    std::vector<int> available_cameras;  // 存储可用摄像头编号的向量
    
    // 首先尝试检测特定的摄像头设备 - 调用detectCameraNumbers函数获取特定型号的摄像头
    std::vector<int> inf_numbers, rgb_numbers;  // 红外和RGB摄像头编号列表
    detectCameraNumbers(inf_numbers, rgb_numbers);
    
    // 优先测试RGB摄像头（用于安全帽检测） - RGB摄像头更适合目标检测任务
    if (!rgb_numbers.empty()) {
        for (int cam_id : rgb_numbers) {
            cv::VideoCapture test_cap;  // 创建OpenCV摄像头对象
            test_cap.open(cam_id);      // 尝试打开指定编号的摄像头
            if (test_cap.isOpened()) {
                cv::Mat test_frame;     // 测试帧
                if (test_cap.read(test_frame) && !test_frame.empty()) {
                    // 成功读取帧，摄像头可用
                    available_cameras.push_back(cam_id);
                    std::cout << "RGB摄像头 " << cam_id << " 可用" << std::endl;
                } else {
                    std::cout << "RGB摄像头 " << cam_id << " 无法读取帧（可能被占用）" << std::endl;
                }
                test_cap.release();     // 释放摄像头资源
            } else {
                std::cout << "RGB摄像头 " << cam_id << " 无法打开（可能被占用）" << std::endl;
            }
        }
    }
    
    // 如果RGB摄像头不可用，尝试红外摄像头 - 作为备选方案
    if (available_cameras.empty() && !inf_numbers.empty()) {
        for (int cam_id : inf_numbers) {
            cv::VideoCapture test_cap;  // 创建OpenCV摄像头对象
            test_cap.open(cam_id);      // 尝试打开红外摄像头
            if (test_cap.isOpened()) {
                cv::Mat test_frame;     // 测试帧
                if (test_cap.read(test_frame) && !test_frame.empty()) {
                    // 成功读取帧，红外摄像头可用
                    available_cameras.push_back(cam_id);
                    std::cout << "红外摄像头 " << cam_id << " 可用" << std::endl;
                }
                test_cap.release();     // 释放摄像头资源
            }
        }
    }
    
    // 如果特定摄像头不可用，尝试通用摄像头 - 最后的备选方案
    if (available_cameras.empty()) {
        std::cout << "未找到特定RGB摄像头，尝试通用摄像头..." << std::endl;
        for (int i = 0; i <= 5; i++) {  // 测试video0到video5
            std::cout << "测试通用摄像头 " << i << "..." << std::endl;
            cv::VideoCapture test_cap;  // 创建OpenCV摄像头对象
            test_cap.open(i);           // 尝试打开通用摄像头
            if (test_cap.isOpened()) {
                cv::Mat test_frame;     // 测试帧
                if (test_cap.read(test_frame) && !test_frame.empty()) {
                    // 成功读取帧，通用摄像头可用
                    available_cameras.push_back(i);
                    std::cout << "通用摄像头 " << i << " 可用" << std::endl;
                } else {
                    std::cout << "通用摄像头 " << i << " 无法读取帧" << std::endl;
                }
                test_cap.release();     // 释放摄像头资源
            } else {
                std::cout << "通用摄像头 " << i << " 无法打开" << std::endl;
            }
        }
    }
    
    return available_cameras;  // 返回可用摄像头编号列表
}
