#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸烟检测测试脚本
用于测试修改后的吸烟检测功能
"""

import cv2
import numpy as np
import time
from func_smoke import myFunc_smoke
from rknnlite.api import RKNNLite

def test_smoke_detection():
    """测试吸烟检测功能"""
    print("开始测试吸烟检测功能...")
    
    # 初始化RKNN模型
    rknn_lite = RKNNLite()
    model_path = "./smoking.rknn"
    
    print(f"加载模型: {model_path}")
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print("Load RKNN model failed")
        return
    
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        return
    
    print("模型加载成功，开始测试...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 调用吸烟检测函数
            processed_frame, result = myFunc_smoke(rknn_lite, frame, frame_count)
            
            # 显示结果
            if result is not None:
                status = "检测到吸烟" if result == 1 else "未检测到吸烟"
                print(f"帧 {frame_count}: {status}")
            else:
                print(f"帧 {frame_count}: 检测中...")
            
            # 显示图像
            cv2.imshow('Smoke Detection Test', processed_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # 每30帧显示一次帧率
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                print(f"当前帧率: {fps:.2f} FPS")
                start_time = time.time()
    
    except KeyboardInterrupt:
        print("测试被用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rknn_lite.release()
        print("测试结束")

if __name__ == "__main__":
    test_smoke_detection() 