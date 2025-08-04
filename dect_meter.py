import cv2
import apriltag

# 预定义仪表类别信息的映射表，key 为 AprilTag 的 ID，value 为对应的类别
TAG_TO_CATEGORY = {
    0: "消防水压表",
    1: "空气房气压表",
    2: "机械气压表"
}


def capture_meter_frame(cap):
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 检测 AprilTag
        tags = detector.detect(gray)

        # 遍历所有检测到的标签，并在图像中标注
        detected_categories = []
        for tag in tags:
            # 获取标签的角点和中心点
            corners = tag.corners
            center = (int(tag.center[0]), int(tag.center[1]))

            # 绘制标签的外部矩形框
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # 在标签中心绘制一个小圆点
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # 标注标签的 ID 并记录其类别信息
            tag_id = tag.tag_id
            category = TAG_TO_CATEGORY.get(tag_id, "未知类别")  # 根据ID获取类别
            detected_categories.append(category)

            cv2.putText(frame, f"ID: {tag_id} ({category})", (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(tags) >= 2:  # 至少需要两个 AprilTag 才能确定仪表区域
            # 假设只处理两个标签，分别是对角的两个标签
            tag1, tag2 = tags[0], tags[1]

            # 取出两个标签的中心位置
            center1 = (int(tag1.center[0]), int(tag1.center[1]))
            center2 = (int(tag2.center[0]), int(tag2.center[1]))

            # 计算对角的两个点
            top_left = (min(center1[0], center2[0]), min(center1[1], center2[1]))
            bottom_right = (max(center1[0], center2[0]), max(center1[1], center2[1]))

            meter_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            if meter_roi.size > 0:
                # 返回检测到的仪表区域和其对应的类别信息
                cv2.destroyAllWindows()
                return meter_roi, detected_categories  # 返回图像和检测到的类别列表


    return None, None


# 示例使用
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    meter_roi, categories = capture_meter_frame(cap)

    if meter_roi is not None:
        print(f"检测到的仪表类别: {categories}")
        cv2.imshow("Meter ROI", meter_roi)
        cv2.imwrite('w14.jpg', meter_roi)
        k = cv2.waitKey(0)
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("未检测到仪表区域")
