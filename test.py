import cv2
import time
import threading
import numpy as np
from ctypes import *
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc

# 添加海康SDK路径
import sys
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")

try:
    from MvCameraControl_class import *
except ImportError:
    print("请确保已正确添加海康SDK路径")
    sys.exit(1)

# 全局变量
g_bExit = False
latest_frame = None
frame_lock = threading.Lock()
frame_available = threading.Event()  # 用于通知新帧可用

def work_thread(cam):
    """工作线程：从海康摄像头获取图像并转换格式"""
    global latest_frame, g_bExit, frame_available
    
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    
    # 转换参数结构体
    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
    
    while not g_bExit:
        # 获取图像缓冲区
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            # MV_E_NO_DATA = 0x80000004 (无数据错误，可以忽略)
            if ret != 0x80000004:  # 忽略无数据错误
                print(f"获取图像失败: 0x{ret:x}")
            continue
        
        # 获取帧信息
        frame_info = stOutFrame.stFrameInfo
        
        # 设置转换参数
        stConvertParam.nWidth = frame_info.nWidth
        stConvertParam.nHeight = frame_info.nHeight
        stConvertParam.pSrcData = stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = frame_info.nFrameLen
        stConvertParam.enSrcPixelType = frame_info.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # 转换为BGR格式
        
        # 计算目标缓冲区大小
        nDstBufferSize = frame_info.nWidth * frame_info.nHeight * 3
        dst_buf = (c_ubyte * nDstBufferSize)()
        stConvertParam.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        stConvertParam.nDstBufferSize = nDstBufferSize
        
        # 转换像素格式
        ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret != 0:
            print(f"像素格式转换失败: 0x{ret:x}")
            cam.MV_CC_FreeImageBuffer(stOutFrame)
            continue
        
        # 转换为NumPy数组
        img = np.frombuffer(dst_buf, dtype=np.uint8)
        img = img.reshape((frame_info.nHeight, frame_info.nWidth, 3))
        
        # 更新最新帧
        with frame_lock:
            latest_frame = img.copy()
        frame_available.set()  # 通知有新帧可用
        
        # 释放图像缓冲区
        cam.MV_CC_FreeImageBuffer(stOutFrame)

def camera_capture(cam):
    """相机捕获线程"""
    # 启动工作线程
    worker = threading.Thread(target=work_thread, args=(cam,))
    worker.daemon = True
    worker.start()
    return worker

def reconnect_camera():
    """重新连接相机"""
    global cam
    print("尝试重新连接相机...")
    
    # 等待一段时间让设备释放
    time.sleep(2)
    
    # 先清理之前的连接
    try:
        if 'cam' in globals() and cam:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
    except:
        pass
    
    MvCamera.MV_CC_Finalize()
    time.sleep(1)  # 等待SDK完全释放
    
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"重新初始化SDK失败! ret=0x{ret:x}")
        return None

    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
    if ret != 0:
        print(f"重新枚举设备失败! ret=0x{ret:x}")
        return None

    if device_list.nDeviceNum == 0:
        print("重新枚举未找到USB相机设备!")
        return None

    stDeviceInfo = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print(f"重新创建句柄失败! ret=0x{ret:x}")
        return None

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"重新打开设备失败! ret=0x{ret:x}")
        cam.MV_CC_DestroyHandle()
        return None

    print("相机重新连接成功!")

    # 设置触发模式为关闭（连续取流）
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print(f"重新设置触发模式失败! ret=0x{ret:x}")

    # 新增：设置自动曝光模式
    ret = cam.MV_CC_SetEnumValue("ExposureMode", MV_EXPOSURE_MODE_AUTO)  # 自动曝光模式
    if ret != 0:
        print(f"重新设置自动曝光模式失败! ret=0x{ret:x}")

    # 设置分辨率（示例设置为 2560x1080）
    ret = cam.MV_CC_SetIntValue("Width", 2560)
    if ret != 0:
        print(f"重新设置宽度失败! ret=0x{ret:x}")
    ret = cam.MV_CC_SetIntValue("Height", 1080)
    if ret != 0:
        print(f"重新设置高度失败! ret=0x{ret:x}")

    # 设置帧率（调整为30fps）
    ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30)
    if ret != 0:
        print(f"重新设置帧率失败! ret=0x{ret:x}")

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"重新开始取流失败! ret=0x{ret:x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None

    return cam

def check_system_temperature():
    """检查系统温度"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
            print(f"当前CPU温度: {temp:.1f}°C")
            if temp > 80:
                print("警告: CPU温度过高，可能影响设备稳定性")
                return False
            return True
    except:
        print("无法读取温度信息")
        return True

def main():
    global g_bExit, cam
    
    print("海康摄像头视频流推理程序")
    print("=" * 50)
    
    # 检查系统温度
    if not check_system_temperature():
        print("系统温度过高，建议等待降温后再运行")
        return
    
    # 初始化SDK
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"初始化SDK失败! ret=0x{ret:x}")
        return
    
    # 枚举设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_USB_DEVICE  # 只枚举USB设备
    
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
    if ret != 0:
        print(f"枚举设备失败! ret=0x{ret:x}")
        MvCamera.MV_CC_Finalize()
        return
        
    if device_list.nDeviceNum == 0:
        print("未找到USB相机设备!")
        MvCamera.MV_CC_Finalize()
        return
        
    print(f"找到 {device_list.nDeviceNum} 个USB相机设备")
    
    # 获取第一个相机信息
    stDeviceInfo = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    
    # 创建相机实例
    cam = MvCamera()
    
    # 创建句柄
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print(f"创建句柄失败! ret=0x{ret:x}")
        MvCamera.MV_CC_Finalize()
        return
        
    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"打开设备失败! ret=0x{ret:x}")
        if ret == 0x80000203:
            print("设备被占用或权限不足，请检查:")
            print("1. 是否有其他程序在使用相机")
            print("2. 用户是否有访问USB设备的权限")
            print("3. 尝试重新插拔USB设备")
            print("4. 运行: sudo usermod -a -G video $USER 然后重新登录")
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
        
    print("相机连接成功!")
    
    # 设置触发模式为关闭（连续取流）
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print(f"设置触发模式失败! ret=0x{ret:x}")

    # 新增：设置自动曝光模式
    ret = cam.MV_CC_SetEnumValue("ExposureMode", 2)  # 自动曝光模式
    if ret != 0:
        print(f"设置自动曝光模式失败! ret=0x{ret:x}")

    # 设置分辨率（示例设置为 2560x1080）
    ret = cam.MV_CC_SetIntValue("Width", 2560)
    if ret != 0:
        print(f"设置宽度失败! ret=0x{ret:x}")
    ret = cam.MV_CC_SetIntValue("Height", 1080)
    if ret != 0:
        print(f"设置高度失败! ret=0x{ret:x}")

    # 设置帧率（调整为30fps）
    ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30)
    if ret != 0:
        print(f"设置帧率失败! ret=0x{ret:x}")
    
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"开始取流失败! ret=0x{ret:x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    
    print("开始取流...")
    
    # 初始化rknn池
    modelPath = "./rknnModel/md1.rknn"
    TPEs = 9  # 线程数, 增大可提高帧率
    print(f"初始化RKNN池，线程数: {TPEs}")
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        TPEs=TPEs,
        func=myFunc)
    
    try:
        # 启动相机捕获线程
        capture_thread = camera_capture(cam)
        
        # 等待第一帧
        print("等待相机帧...")
        frame_available.wait()
        
        # 获取原始帧尺寸（新增）
        with frame_lock:
            original_width = latest_frame.shape[1]
            original_height = latest_frame.shape[0]
        print(f"原始帧尺寸: {original_width}x{original_height}")

        # 设置显示比例（可调整0.5-1.0之间，0.8为示例）
        display_scale = 0.8 
        display_width = int(original_width * display_scale)
        display_height = int(original_height * display_scale)

        # 初始化异步所需要的帧
        print("初始化推理队列...")
        with frame_lock:
            init_frame = latest_frame.copy()
        for i in range(TPEs + 1):
            pool.put(init_frame)
            # 等待新帧
            frame_available.wait()
            frame_available.clear()
            with frame_lock:
                next_frame = latest_frame.copy()
            init_frame = next_frame
        
        # 开始处理和显示
        frames, loopTime, initTime = 0, time.time(), time.time()
        window_name = "Hikvision Camera - RKNN Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)  # 设置窗口初始大小
        
        print("开始处理视频流...")
        while not g_bExit:
            frames += 1
            if not capture_thread.is_alive():
                print("取流线程异常终止，尝试重新连接相机...")
                new_cam = reconnect_camera()
                if new_cam:
                    cam = new_cam
                    capture_thread = camera_capture(cam)
                else:
                    break
                    
            # 等待新帧
            frame_available.wait()
            frame_available.clear()
            
            # 获取新帧并放入处理队列
            with frame_lock:
                frame = latest_frame.copy()
            pool.put(frame)
            
            # 获取处理结果并显示
            result_frame, flag = pool.get()
            if not flag:
                break
                
            # 按比例缩放显示
            resized_frame = cv2.resize(result_frame, (display_width, display_height), 
                                      interpolation=cv2.INTER_AREA)
            
            # 在图像上显示帧率
            current_fps = frames / (time.time() - initTime)
            cv2.putText(resized_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, resized_frame)  # 显示缩放后画面
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                g_bExit = True
                break
                
            # 每30帧打印一次帧率
            if frames % 30 == 0:
                elapsed = time.time() - loopTime
                print(f"30帧平均帧率: {30 / elapsed:.1f} 帧/秒")
                loopTime = time.time()
        
        # 打印总平均帧率
        total_time = time.time() - initTime
        print(f"总平均帧率: {frames / total_time:.1f} 帧/秒")
        
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 清理资源
        g_bExit = True
        
        # 等待捕获线程结束
        if 'capture_thread' in locals() and capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        
        # 停止取流
        if 'cam' in locals():
            cam.MV_CC_StopGrabbing()
        
        # 关闭设备
        if 'cam' in locals():
            cam.MV_CC_CloseDevice()
        
        # 销毁句柄
        if 'cam' in locals():
            cam.MV_CC_DestroyHandle()
        
        # 反初始化SDK
        MvCamera.MV_CC_Finalize()
        
        # 释放RKNN池和窗口
        if 'pool' in locals():
            pool.release()
        cv2.destroyAllWindows()
        
        print("所有资源已释放")

if __name__ == "__main__":
    main()