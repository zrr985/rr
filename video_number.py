import os
import re

inf_numbers = []
rgb_numbers = []

for device in os.listdir('/sys/class/video4linux/'):
    if device.startswith("video"):
        # 提取数字部分
        video_number = re.search(r'\d+', device).group()
        rgb_numbers.append(int(video_number))

print(f"inf Video Numbers: {inf_numbers}")
print(f"rgb Video Numbers: {rgb_numbers}")



