import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_circle_on_mask(dail_mask_resized, center, radius):
    # 将dail_mask_resized转为彩色图像以便于绘图
    color_img = cv2.cvtColor(dail_mask_resized, cv2.COLOR_GRAY2BGR)

    # 绘制圆心
    center_coordinates = (int(center[0]), int(center[1]))
    cv2.circle(color_img, center_coordinates, 5, (0, 0, 255), -1)  # 红色圆心

    # 绘制圆
    cv2.circle(color_img, center_coordinates, int(radius), (0, 255, 0), 2)  # 绿色圆

    return color_img

def find_circle_center_and_radius(image):
    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 转换为灰度图并中值滤波
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 使用HoughCircles方法检测圆形
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0] // 8,
        param1=100,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    # 确保至少检测到一个圆
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 将圆心和半径转为整数

        # 取第一个检测到的圆的参数
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))  # 圆心坐标 (x, y)
            radius = i[2]  # 圆的半径
            radius = int((radius * 9) / 10)

            # 在图像上绘制检测到的圆
            # cv2.circle(image, center, radius, (0, 255, 0), 2)  # 画出圆
            # cv2.circle(image, center, 2, (0, 0, 255), 3)  # 画出圆心
            # # 展示结果
            # cv2.imshow("Detected Circle", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print(f"Center: {center}, Radius: {radius}")

            return center, radius
    else:
        print("No circles were detected.")
        return None, None


class MeterReader(object):

    def __init__(self, img):
        center, radius = find_circle_center_and_radius(img)

        # Custom parameters
        self.line_width = int(2 * np.pi * radius)  # 表盘展开为直线的长度,按照表盘的图像像素周长计算
        self.line_height = int(radius*0.4)  # 表盘展开为直线的宽度，该设定按照围绕圆心的扇形宽度计算，需要保证包含刻度以及一部分指针
        self.circle_radius = int(radius)  # 预设圆盘直径，扫描的最大直径，需要小于图幅，否者可能会报错
        self.circle_center = center  # 圆盘指针的旋转中心，预设的指针旋转中心
        self.threshold = 0.5

    def __call__(self, point_mask_resized, dail_mask_resized):
        if (self.circle_radius is None) | (self.circle_center is None):
            return None
        # 可视化掩码
        plt.subplot(1, 2, 1)
        plt.title("Pointer Mask")
        plt.imshow(point_mask_resized, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Scale Mask")
        plt.imshow(dail_mask_resized, cmap='gray')

        print(f"dail_mask size:{dail_mask_resized.shape}")

        plt.show()
        # color_image = draw_circle_on_mask(dail_mask_resized, self.circle_center, self.circle_radius)
        # cv2.imshow("Detected Circle", color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        relative_value = self.get_relative_value(point_mask_resized, dail_mask_resized)

        return relative_value['ratio']

    def get_relative_value(self, image_pointer, image_dail):
        line_image_pointer = self.create_line_image(image_pointer)
        line_image_dail = self.create_line_image(image_dail)

        plt.subplot(1, 2, 1)
        plt.title("Unfolded Pointer Image")
        plt.imshow(line_image_pointer, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Unfolded Scale Image")
        plt.imshow(line_image_dail, cmap='gray')
        plt.show()

        data_1d_pointer = self.convert_1d_data(line_image_pointer)
        data_1d_dail = self.convert_1d_data(line_image_dail)
        data_1d_dail = self.mean_filtration(data_1d_dail)

        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0

        for i in range(self.line_width - 1):
            # 检测刻度位置
            if data_1d_dail[i] > 0:
                if not dail_flag:
                    one_dail_start = i
                    dail_flag = True
            if dail_flag and data_1d_dail[i] == 0:
                one_dail_end = i - 1
                one_dail_location = (one_dail_start + one_dail_end) / 2
                dail_location.append(one_dail_location)
                dail_flag = False

            # 检测指针位置
            if data_1d_pointer[i] > 0:
                if not pointer_flag:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag and data_1d_pointer[i] == 0:
                one_pointer_end = i - 1
                pointer_location = (one_pointer_start + one_pointer_end) / 2
                pointer_flag = False

        scale_num = len(dail_location)
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1
            ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)

        print(f"Pointer location: {pointer_location}")
        print(f"Dail locations: {dail_location}")

        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        return result

    def create_line_image(self, image_mask):
        """
        Create a linear image
        :param image_mask: mask image
        :return:
        """
        line_image = np.zeros((self.line_height, self.line_width), dtype=np.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                """Calculate the angle with the -y axis"""
                theta = ((2 * np.pi) / self.line_width) * (col + 1)
                '''Calculate the diameter corresponding to the original image'''
                radius = self.circle_radius - row - 1

                # # 提前终止条件: 如果半径超出图像的尺寸，跳过这一行
                if radius < 0 or radius >= min(image_mask.shape[0], image_mask.shape[1]):
                    continue

                '''Calculate the position of the current scan point corresponding to the original image'''
                y = int(self.circle_center[1] + radius * np.cos(theta) + 0.5)
                x = int(self.circle_center[0] - radius * np.sin(theta) + 0.5)

                # # 检查 x 和 y 是否在图像范围内
                if 0 <= y < image_mask.shape[0] and 0 <= x < image_mask.shape[1]:
                    line_image[row, col] = image_mask[y, x]
                #else:
                    # Optionally log or handle the invalid coordinates
                    #print(f"Skipping out-of-bound coordinates: y={y}, x={x}")
        return line_image

    def convert_1d_data(self, line_image):
        """
        Convert the image to a 1D array
        :param line_image: Unfolded image
        :return: 1D array
        """
        data_1d = np.zeros((self.line_width), dtype=np.int16)
        threshold = 127  # 设置阈值
        for col in range(self.line_width):
            for row in range(self.line_height):
                if line_image[row, col] > threshold:  # 如果像素值大于阈值
                    data_1d[col] += 1
        return data_1d

    def mean_filtration(self, data_1d_dail):
        """
        Mean filtering
        :param data_1d_dail: 1D data array
        :return: Filtered data array
        """
        new_data_1d_dail = data_1d_dail.copy()
        for i in range(1, self.line_width - 1):
            new_data_1d_dail[i] = (data_1d_dail[i - 1] + data_1d_dail[i] + data_1d_dail[i + 1]) / 3
        return new_data_1d_dail


