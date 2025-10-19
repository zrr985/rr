import cv2
import time
import threading
import numpy as np
from ctypes import *
from rknnpool import rknnPoolExecutor
import subprocess
import os
import signal
import atexit
import queue
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
frame_available = threading.Event()

# RTSP推流相关变量
ffmpeg_process = None
rtsp_server_process = None
stream_queue = queue.Queue(maxsize=8)  # 适中的队列大小，平衡内存和延迟
stream_thread = None
stream_enabled = False

def get_ip_address():
    """获取本机IP地址"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def test_network_connectivity():
    """测试网络连接性"""
    print("=== 网络连接诊断 ===")
    
    # 测试localhost解析
    try:
        import socket
        localhost_ip = socket.gethostbyname('localhost')
        print(f"1. localhost解析为: {localhost_ip}")
    except Exception as e:
        print(f"1. localhost解析失败: {e}")
    
    # 测试127.0.0.1连接
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(('127.0.0.1', 8554))
        s.close()
        if result == 0:
            print("2. 127.0.0.1:8554 连接成功")
        else:
            print("2. 127.0.0.1:8554 连接失败")
    except Exception as e:
        print(f"2. 127.0.0.1:8554 连接测试失败: {e}")
    
    # 测试本机IP连接
    try:
        local_ip = get_ip_address()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((local_ip, 8554))
        s.close()
        if result == 0:
            print(f"3. {local_ip}:8554 连接成功")
        else:
            print(f"3. {local_ip}:8554 连接失败")
    except Exception as e:
        print(f"3. 本机IP连接测试失败: {e}")
    
    print("=== 诊断完成 ===")

def stream_worker():
    """推流工作线程 - 修复版本"""
    global ffmpeg_process, stream_enabled, g_bExit
    
    while not g_bExit and stream_enabled:
        try:
            # 从队列获取帧，超时1秒
            frame = stream_queue.get(timeout=1.0)
            
            if ffmpeg_process is None or ffmpeg_process.poll() is not None:
                continue
                
            try:
                # 直接写入字节数据（不再需要转换）
                ffmpeg_process.stdin.write(frame.tobytes())
                # 注意：移除了 .flush()，因为 bufsize=0 已经是无缓冲
            except BrokenPipeError:
                print("FFmpeg管道断开，推流失败")
                break
            except Exception as e:
                print(f"推流失败: {e}")
                # 如果推流失败，跳过此帧
                continue
                
            stream_queue.task_done()
            
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"推流线程错误: {e}")
            break

def start_stream_thread():
    """启动推流线程"""
    global stream_thread, stream_enabled
    
    if stream_enabled:
        return True
        
    stream_enabled = True
    stream_thread = threading.Thread(target=stream_worker)
    stream_thread.daemon = True
    stream_thread.start()
    print("推流线程已启动")
    return True

def push_frame_async(frame):
    """异步推送帧到RTSP流 - 优化版本"""
    global stream_enabled, stream_queue
    
    if not stream_enabled:
        return False
        
    try:
        # 如果队列已满，选择性丢弃旧帧而不是清空整个队列
        if stream_queue.full():
            try:
                # 只丢弃一半的队列内容，保留较新的帧
                current_size = stream_queue.qsize()
                drop_count = current_size // 2
                for _ in range(drop_count):
                    stream_queue.get_nowait()
                    stream_queue.task_done()
                print(f"丢弃 {drop_count} 帧以缓解积压")
            except:
                pass
        
        # 非阻塞方式放入队列
        stream_queue.put_nowait(frame)
        return True
    except queue.Full:
        # 如果队列仍然满，跳过这一帧
        return False

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
            if ret != MV_E_NO_DATA:  # 忽略无数据错误
            #if ret != 0x80000004:  # 忽略无数据错误
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

def start_rtsp_server():
    """启动RTSP服务器 - 使用当前目录的mediamtx"""
    global rtsp_server_process
    
    # 检查当前目录是否有mediamtx
    mediamtx_path = './mediamtx'
    if not os.path.exists(mediamtx_path):
        print(f"错误: 在当前目录找不到mediamtx可执行文件")
        return False
    
    # 确保mediamtx有执行权限
    os.chmod(mediamtx_path, 0o755)
    
    # 检查配置文件
    config_path = './mediamtx.yml'
    if not os.path.exists(config_path):
        # 创建优化的IPv4配置文件
        config_content = """# mediamtx.yml - IPv4优化版本
logLevel: info
rtspAddress: 0.0.0.0:8554    # 强制绑定到IPv4
httpAddress: 0.0.0.0:8888
metricsAddress: 0.0.0.0:8889

paths:
  cam:
    source: publisher
    sourceProtocol: tcp
    readBufferCount: 512
    sourceOnDemand: false
    sourceOnDemandStartTimeout: 10s
    sourceOnDemandCloseAfter: 10s
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("已创建优化的mediamtx.yml配置文件（IPv4版本）")
    
    try:
        # 启动MediaMTX RTSP服务器（使用当前目录的mediamtx）
        rtsp_server_process = subprocess.Popen(
            [mediamtx_path, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # 等待服务器启动
        time.sleep(2)
        
        # 检查进程是否还在运行
        if rtsp_server_process.poll() is None:
            local_ip = get_ip_address()
            print("RTSP服务器启动成功!")
            print(f"RTSP流地址: rtsp://{local_ip}:8554/cam")
            print(f"RTSP流地址: rtsp://localhost:8554/cam")
            return True
        else:
            stdout, stderr = rtsp_server_process.communicate()
            print(f"RTSP服务器启动失败: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"启动RTSP服务器失败: {e}")
        return False

def init_ffmpeg_pipeline(width, height, fps=10):
    """使用FFmpeg创建推流管道 - 终极优化版本"""
    global ffmpeg_process
    
    try:
        # 终极优化的FFmpeg推流命令
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',      # 最快的编码预设
            '-tune', 'zerolatency',      # 零延迟调优
            '-b:v', '500k',              # 提高码率以获得更好质量
            '-maxrate', '500k',          # 最大码率
            '-bufsize', '1000k',         # 缓冲区大小
            '-g', '10',                  # GOP大小，减少关键帧间隔
            '-r', str(fps),              # 输出帧率
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',    # 强制使用TCP
            '-muxdelay', '0.1',          # 减少复用延迟
            '-loglevel', 'error',        # 只显示错误信息，减少输出
            'rtsp://127.0.0.1:8554/cam'  # 关键修改：使用127.0.0.1而不是localhost
        ]
        
        print(f"启动FFmpeg推流: {width}x{height} @ {fps}fps")
        
        # 关键修改：移除 text=True，使用字节模式
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # 忽略标准输出
            stderr=subprocess.PIPE,     # 只捕获错误
            bufsize=0,
            preexec_fn=os.setsid
            # 注意：移除了 text=True
        )
        
        # 启动错误监控线程
        def monitor_errors():
            while ffmpeg_process.poll() is None:
                try:
                    # 读取错误输出（字节模式）
                    error_line = ffmpeg_process.stderr.readline()
                    if error_line:
                        # 解码字节为字符串
                        error_text = error_line.decode('utf-8', errors='ignore').strip()
                        if error_text and 'error' in error_text.lower():
                            print(f"FFmpeg错误: {error_text}")
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor_errors)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 等待FFmpeg启动
        time.sleep(3)
        
        if ffmpeg_process.poll() is not None:
            print("FFmpeg启动失败")
            # 获取错误信息
            try:
                stderr_output = ffmpeg_process.stderr.read()
                if stderr_output:
                    print(f"FFmpeg错误详情: {stderr_output.decode('utf-8', errors='ignore')}")
            except:
                pass
            return False
        
        print("FFmpeg推流管道初始化成功!")
        return True
            
    except Exception as e:
        print(f"FFmpeg推流初始化失败: {e}")
        return False


def stop_rtsp_stream():
    """停止RTSP推流"""
    global ffmpeg_process, rtsp_server_process, stream_enabled, stream_thread
    
    # 停止推流线程
    stream_enabled = False
    
    try:
        # 清空队列
        while not stream_queue.empty():
            try:
                stream_queue.get_nowait()
                stream_queue.task_done()
            except:
                pass
                
        # 等待推流线程结束
        if stream_thread and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)
            print("推流线程已停止")
    except Exception as e:
        print(f"停止推流线程时出错: {e}")
    
    try:
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=5)
            ffmpeg_process = None
            print("FFmpeg推流已停止")
            
    except Exception as e:
        print(f"停止FFmpeg时出错: {e}")
        try:
            if ffmpeg_process:
                os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGKILL)
        except:
            pass
    
    try:
        if rtsp_server_process and rtsp_server_process.poll() is None:
            os.killpg(os.getpgid(rtsp_server_process.pid), signal.SIGTERM)
            rtsp_server_process.wait(timeout=5)
            rtsp_server_process = None
            print("RTSP服务器已停止")
            
    except Exception as e:
        print(f"停止RTSP服务器时出错: {e}")
        try:
            if rtsp_server_process:
                os.killpg(os.getpgid(rtsp_server_process.pid), signal.SIGKILL)
        except:
            pass

# 注册退出时的清理函数
atexit.register(stop_rtsp_stream)

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
    MvCamera.MV_CC_Finalize()
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

    # 设置曝光模式为手动，并调整曝光时间
    ret = cam.MV_CC_SetEnumValue("ExposureMode", 1)  # 手动曝光模式
    if ret != 0:
        print(f"重新设置手动曝光模式失败! ret=0x{ret:x}")
    
    # 设置曝光时间（增加曝光时间让画面更亮）
    ret = cam.MV_CC_SetFloatValue("ExposureTime", 10000)  # 10ms曝光时间
    if ret != 0:
        print(f"重新设置曝光时间失败! ret=0x{ret:x}")
    
    # 设置增益
    ret = cam.MV_CC_SetFloatValue("Gain", 10.0)  # 增加增益
    if ret != 0:
        print(f"重新设置增益失败! ret=0x{ret:x}")

    # 设置分辨率（调整为1920x1080，获得更高清晰度）
    ret = cam.MV_CC_SetIntValue("Width", 1920)
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

def main():
    global g_bExit, cam, stream_enabled, ffmpeg_process, rtsp_server_process, stream_thread
    
    local_ip = get_ip_address()
    
    print("海康摄像头视频流推理程序 - 带RTSP推流（优化版）")
    print("=" * 50)
    print(f"本机IP地址: {local_ip}")
    print(f"RTSP流地址: rtsp://{local_ip}:8554/cam")
    print("=" * 50)
    
    # 初始化全局变量
    stream_enabled = False
    rtsp_enabled = False
    
    # 启动RTSP服务器
    if not start_rtsp_server():
        print("警告: RTSP服务器启动失败，继续运行但不推流")
        rtsp_enabled = False
    else:
        rtsp_enabled = True
        # 进行网络连接诊断
        test_network_connectivity()
    
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
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
        
    print("相机连接成功!")
    
    # 设置触发模式为关闭（连续取流）
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print(f"设置触发模式失败! ret=0x{ret:x}")

    # 设置曝光模式为手动，并调整曝光时间
    ret = cam.MV_CC_SetEnumValue("ExposureMode", 1)  # 手动曝光模式
    if ret != 0:
        print(f"设置手动曝光模式失败! ret=0x{ret:x}")
    
    # 设置曝光时间（增加曝光时间让画面更亮）
    ret = cam.MV_CC_SetFloatValue("ExposureTime", 10000)  # 10ms曝光时间
    if ret != 0:
        print(f"设置曝光时间失败! ret=0x{ret:x}")
    
    # 设置增益
    ret = cam.MV_CC_SetFloatValue("Gain", 10.0)  # 增加增益
    if ret != 0:
        print(f"设置增益失败! ret=0x{ret:x}")

    # 设置分辨率（调整为1920x1080，获得更高清晰度）
    ret = cam.MV_CC_SetIntValue("Width", 1920)
    if ret != 0:
        print(f"设置宽度失败! ret=0x{ret:x}, 尝试获取当前分辨率...")
        # 获取当前分辨率
        width_value = MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue("Width", width_value)
        if ret == 0:
            print(f"当前宽度: {width_value.nCurValue}")
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
    TPEs = 3  # 线程数, 与RK3588的3核NPU对齐，避免资源争抢
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

        # 设置显示比例（针对1920x1080优化）
        if original_width <= 640:
            display_scale = 1.0  # 对于640x480，使用1:1显示比例
        elif original_width <= 960:
            display_scale = 0.8  # 对于960x540，使用0.8显示比例
        elif original_width <= 1280:
            display_scale = 0.5  # 对于1280x720，使用0.5显示比例
        elif original_width <= 1920:
            display_scale = 0.4  # 对于1920x1080，使用0.4显示比例，避免窗口过大
        else:
            display_scale = 0.3  # 对于更高分辨率，使用更小的显示比例
        display_width = int(original_width * display_scale)
        display_height = int(original_height * display_scale)

        # 初始化推流管道（使用异步推流）
        if rtsp_enabled:
            print("初始化FFmpeg推流管道...")
            # 使用更低的分辨率和帧率
            push_width = 480  # 进一步降低推流分辨率
            push_height = 270
            if init_ffmpeg_pipeline(push_width, push_height, fps=10):
                print(f"推流分辨率: {push_width}x{push_height} @ 10fps")
                # 启动异步推流线程
                start_stream_thread()
            else:
                print("FFmpeg推流管道初始化失败，继续运行但不推流")
                rtsp_enabled = False
        else:
            print("RTSP服务器未启动，跳过推流初始化")

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
        window_name = f"Hikvision Camera - RTSP (Async)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)  # 设置窗口初始大小
        
        print("开始处理视频流和RTSP推流...")
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
            
            # 推流到RTSP（使用异步推流，不阻塞主线程）
            if rtsp_enabled and stream_enabled:
                # 为推流专门调整分辨率
                push_frame = cv2.resize(result_frame, (480, 270), interpolation=cv2.INTER_AREA)
                push_frame_async(push_frame)
            
            # 在图像上显示帧率和连接信息
            current_fps = frames / (time.time() - initTime)
            queue_size = stream_queue.qsize() if rtsp_enabled else 0
            status_lines = [
                f"FPS: {current_fps:.1f}",
                f"RTSP: {'ON' if rtsp_enabled else 'OFF'}",
                f"Queue: {queue_size}/8" if rtsp_enabled else ""
            ]
            
            for i, line in enumerate(status_lines):
                if line:  # 只显示非空行
                    cv2.putText(resized_frame, line, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, resized_frame)  # 显示缩放后画面
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                g_bExit = True
                break
                
            # 每30帧打印一次帧率
            if frames % 30 == 0:
                elapsed = time.time() - loopTime
                rtsp_status = "推流中..." if rtsp_enabled else "无推流"
                print(f"30帧平均帧率: {30 / elapsed:.1f} 帧/秒 | {rtsp_status}")
                loopTime = time.time()
        
        # 打印总平均帧率
        total_time = time.time() - initTime
        print(f"总平均帧率: {frames / total_time:.1f} 帧/秒")
        
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 清理资源
        g_bExit = True
        stream_enabled = False
        
        # 等待推流线程结束
        if 'stream_thread' in locals() and stream_thread and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)
        
        # 停止推流
        stop_rtsp_stream()
        
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