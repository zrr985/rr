import cv2
import os
import video_number

def open_camera(camera_numbers):
    for number in camera_numbers:
        cap = cv2.VideoCapture(number)
        if cap.isOpened():
            print(f"Found openable camera: {number}")
            return cap
    return None

cap = open_camera(video_number.rgb_numbers)

# 文件夹名称，存储图片
folder = "face_images"
if not os.path.exists(folder):
    os.makedirs(folder)

# 存储人名与图片编号的对应关系的文件
info_file = "name_to_image_map.txt"

# 读取或创建记录文件
if not os.path.exists(info_file):
    with open(info_file, 'w') as f:
        f.write("Image Number,Name\n")  # 文件头部写入

# 获取用户姓名
name = input("请输入姓名：")

# 图片编号，从1开始
start_image_number = 1

# 检查文件夹内是否有已存在的图片，确定从哪个编号开始
existing_images = [f for f in os.listdir(folder) if f.endswith(".jpg")]
if existing_images:
    existing_numbers = [int(f.split(".")[0]) for f in existing_images]
    start_image_number = max(existing_numbers) + 1

print("按 's' 键拍照，按 'q' 键退出程序。")

# 循环拍照
while True:
    ret, frame = cap.read()  # 从摄像头读取一帧
    if not ret:
        print("无法读取摄像头")
        break

    # 显示摄像头画面
    cv2.imshow("Camera", frame)

    # 等待用户按键操作
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 按 's' 键拍照
        image_filename = f"{start_image_number}.jpg"
        image_path = os.path.join(folder, image_filename)

        # 保存图片
        cv2.imwrite(image_path, frame)
        print(f"保存图片: {image_path}")

        # 写入对应关系文件
        with open(info_file, 'a') as f:
            f.write(f"{start_image_number},{name}\n")

        start_image_number += 1  # 图片编号递增

    elif key == ord('q'):  # 按 'q' 键退出程序
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
