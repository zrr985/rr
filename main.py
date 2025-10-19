import cv2
import time
import os
from rknnpool import rknnPoolExecutor
from func import myFunc
from moviepy.editor import VideoFileClip, vfx

# 配置参数
input_path = './5.mp4'
output_path = './output.mp4'  # 当前目录，确保有权限
modelPath = "./rknnModel/md1.rknn"
TPEs = 9

# 将输入视频转换为灰度视频
print("正在将输入视频转换为灰度...")
clip = VideoFileClip(input_path)
clip_blackwhite = clip.fx(vfx.blackwhite)
gray_input_path = './gray_input.mp4'
clip_blackwhite.write_videofile(gray_input_path, verbose=False, logger=None)
clip.close()
clip_blackwhite.close()
print(f"灰度视频已生成: {gray_input_path}")

# 使用灰度视频作为输入
input_path = gray_input_path

# 确保输出目录存在
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开视频
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: 无法打开输入视频 {input_path}")
    exit(-1)

# 获取视频参数（并处理默认值）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 else 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0 else 720
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# 初始化视频写入器（使用兼容性编码器）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 替换avc1为mp4v
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error: 视频写入器初始化失败！编码器: {fourcc}, 尺寸: {frame_width}x{frame_height}")
    cap.release()
    exit(-1)

# 初始化rknn池
pool = rknnPoolExecutor(rknnModel=modelPath, TPEs=TPEs, func=myFunc)

# 预加载帧
for i in range(TPEs + 1):
    ret, frame = cap.read()
    if not ret:
        print("Error: 初始化时读取帧失败")
        cap.release()
        out.release()
        pool.release()
        exit(-1)
    pool.put(frame)

# 主循环
frames = 0
loopTime = time.time()
initTime = time.time()
write_count = 0  # 写入帧计数

try:
    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
        
        pool.put(frame)
        processed_frame, flag = pool.get()
        if not flag or processed_frame is None:
            print(f"警告: 第 {frames} 帧处理失败，跳过写入")
            continue
        
        # 写入帧并计数
        out.write(processed_frame)
        write_count += 1
        
        # 显示
        cv2.imshow('yolov8', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 打印帧率
        if frames % 30 == 0:
            current_fps = 30 / (time.time() - loopTime)
            print(f"30帧平均帧率: {current_fps:.2f} 帧/秒 | 已写入 {write_count} 帧")
            loopTime = time.time()

except Exception as e:
    print(f"程序异常终止: {e}")

finally:
    # 释放所有资源
    print(f"\n总处理帧: {frames} | 成功写入帧: {write_count}")
    cap.release()
    print("释放视频捕获...")
    out.release()
    print("释放视频写入器...")
    cv2.destroyAllWindows()
    pool.release()
    print("所有资源已释放")

# 检查输出文件
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"输出视频已生成: {output_path} (大小: {file_size:.2f} KB)")
else:
    print(f"警告: 未找到输出视频 {output_path}")

# 清理临时灰度文件
if os.path.exists(gray_input_path):
    os.remove(gray_input_path)
    print(f"已清理临时文件: {gray_input_path}")
