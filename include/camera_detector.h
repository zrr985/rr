#ifndef _CAMERA_DETECTOR_H_  // 防止头文件被重复包含的宏定义开始
#define _CAMERA_DETECTOR_H_  // 定义头文件保护宏，确保此头文件只被包含一次

#include <vector>  // 包含标准库的vector容器，用于存储动态数组
#include <string>  // 包含标准库的string类，用于处理字符串

// 检测特定摄像头设备编号的函数声明
// 参数说明：
// - inf_numbers: 引用传递的红外摄像头设备编号列表（输出参数）
// - rgb_numbers: 引用传递的RGB摄像头设备编号列表（输出参数）
// 功能：扫描系统中的摄像头设备，区分红外摄像头和RGB摄像头，并将设备编号分别存储到对应的vector中
void detectCameraNumbers(std::vector<int>& inf_numbers, std::vector<int>& rgb_numbers);

// 查找可用摄像头的函数声明
// 返回值：包含所有可用摄像头设备编号的vector<int>
// 功能：扫描系统中的所有摄像头设备（通常是/dev/video*），返回可用的摄像头设备编号列表
// 例如：如果/dev/video0和/dev/video2可用，则返回{0, 2}
std::vector<int> findAvailableCameras();

#endif //_CAMERA_DETECTOR_H_  // 头文件保护宏结束，防止重复包含
