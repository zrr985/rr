import os
import re
camera_info={}


# 定义两个数组分别用于存储两个希望的 model 的 video number
inf_numbers = []
rgb_numbers = []

for device in os.listdir('/sys/class/video4linux/'):
    try:
        # 读取设备的硬件信息
        with open(f"/sys/class/video4linux/{device}/device/modalias", "r") as f:
            modalias = f.read().strip()
            # 从 modalias 中提取设备厂商、型号等信息
            manufacturer, model = modalias.split(":")
            print(model)
            # 如果 model 是红外摄像
            if model == 'v1514p0001d0200dcEFdsc02dp01ic0Eisc01ip00in00':
                # 使用正则表达式提取数字部分
                video_number = re.search(r'\d+', device).group()
                inf_numbers.append(int(video_number))
            #model是普通usb摄像
            elif model == 'v1BCFp0C18d0508dcEFdsc02dp01ic0Eisc01ip00in00':
                # 使用正则表达式提取数字部分
                video_number = re.search(r'\d+', device).group()
                rgb_numbers.append(int(video_number))
    except FileNotFoundError:
        pass

print(f"inf Video Numbers: {inf_numbers}")
print(f"rgb Video Numbers: {rgb_numbers}")

