import cv2
import numpy as np
import time

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.3, 0.45, 640

CLASSES = ('flame')

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape


    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)

    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        top = (top - padding[0]) / ratio[0]
        left = (left - padding[1]) / ratio[1]
        right = (right - padding[0]) / ratio[0]
        bottom = (bottom - padding[1]) / ratio[1]
        
        if top is not None and not np.isnan(top):
            top = int(top)
        if left is not None and not np.isnan(left):
            left = int(left)

        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (left, top)


class FlameDetector:
    def __init__(self, detection_threshold=0.7, window_size=10):
        """
        初始化火焰检测器
        :param detection_threshold: 检测阈值，当窗口内检测到火焰的比例超过此值时，才认为有火焰
        :param window_size: 滑动窗口大小
        """
        self.detection_threshold = detection_threshold
        self.window_size = window_size
        self.detection_history = []
        
    def detect(self, rknn_lite, img, frame_num):
        """
        检测图像中的火焰
        :param rknn_lite: RKNN模型
        :param img: 输入图像
        :param frame_num: 当前帧号
        :return: 处理后的图像, 火焰检测结果 (1: 有火焰, 0: 无火焰, None: 不确定)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb, ratio, padding = letterbox(img_rgb)
        img_rgb = np.expand_dims(img_rgb, 0)

        outputs = rknn_lite.inference(inputs=[img_rgb], data_format=['nhwc'])
        boxes, classes, scores = yolov8_post_process(outputs)
        
        # 记录当前帧的检测结果
        has_flame = 1 if classes is not None and len(classes) > 0 else 0
        self.detection_history.append(has_flame)
        
        # 保持历史记录的长度不超过窗口大小
        if len(self.detection_history) > self.window_size:
            self.detection_history.pop(0)
            
        # 计算窗口内的检测比例
        detection_ratio = sum(self.detection_history) / len(self.detection_history)
        
        # 根据检测比例判断是否有火焰
        if len(self.detection_history) >= self.window_size:
            class_real = 1 if detection_ratio >= self.detection_threshold else 0
        else:
            class_real = None  # 窗口未满，返回不确定
            
        # 绘制检测结果
        if boxes is not None:
            draw(img, boxes, scores, classes, ratio, padding)
            
        # 显示检测状态
        #status_text = f"Flame: {class_real}"
        #cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return img, class_real


# 创建全局火焰检测器实例
flame_detector = FlameDetector(detection_threshold=0.7, window_size=10)

def myFunc_flame(rknn_lite, img, frame_num):
    """
    火焰检测函数
    :param rknn_lite: RKNN模型
    :param img: 输入图像
    :param frame_num: 当前帧号
    :return: 处理后的图像, 火焰检测结果
    """
    return flame_detector.detect(rknn_lite, img, frame_num)