import cv2
import numpy as np

# -------------------------- 常量定义 --------------------------
# 关键修改1：新增电子警戒阈值
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.3, 0.45, 640  # IMG_SIZE改为1280（需与模型输入尺寸匹配）
ALERT_DISTANCE = 3.0  # 电子警戒触发距离阈值（单位：米）

# 关键修改2：新增face类别实际尺寸配置
TARGET_DIMENSIONS = {
    'person': {'height': 1.7, 'min_pixel': 30},  # 人-平均身高
    'head': {'height': 0.2, 'min_pixel': 15},    # 头-平均高度
    'face': {'height': 0.2, 'min_pixel': 15}     # 新增：人脸-平均高度（需根据实际校准）
}

# 相机参数（建议通过标定获取准确值，示例值需根据实际相机校准）
FOCAL_LENGTH = 800  # 像素焦距（公式：f = (pixel_height * distance) / real_height）

# 原始类别列表（已包含person、head、face）
CLASSES = ('0', '1', '2', 'Bicycle', 'Bike', 'Car', 'Cyclist', 'Pedestrian', 'Pedestrians', 'Persona', 'Pessoa', 
           'Signboard', 'Stopper', 'aeroplane', 'bag', 'berdiri', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
           'cat', 'chair', 'cow', 'cyclist', 'dianzhuan', 'diningtable', 'dog', 'face', 'forklift', 'handbag', 'head',  # 包含face类别
           'helmet', 'high', 'horse', 'jatuh', 'laptop', 'low', 'medium', 'motorbike', 'people', 'person', 'persons', 
           'pottedplant', 'refrigerator', 'sheep', 'sofa', 'teddy bear', 'train', 'tv', 'tvmonitor', 'vase')

# 关键修改3：新增face类别到目标检测集合
TARGET_CLASSES = {
    'person',  # 保留核心类别person
    'head',    # 保留head类别
    'face'     # 新增：face类别
}

# -------------------------- 辅助函数 --------------------------
def filter_boxes(boxes, box_confidences, box_class_probs):
    # 未修改，保持原有逻辑
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
    # 未修改，保持原有逻辑
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
    # 未修改，保持原有逻辑
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    # 未修改，保持原有逻辑（依赖IMG_SIZE，修改后自动适配新尺寸）
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)  # 自动使用新的IMG_SIZE

    position = dfl(position)
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    # 未修改，保持原有逻辑（自动过滤新增的face类别）
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 只保留person、head、face类别的检测结果（自动生效）
    target_indices = []
    for i, class_idx in enumerate(classes):
        class_name = CLASSES[class_idx].lower()
        if class_name in TARGET_CLASSES:
            target_indices.append(i)
    
    if not target_indices:
        return None, None, None
    
    boxes = boxes[target_indices]
    classes = classes[target_indices]
    scores = scores[target_indices]

    # 对检测到的目标进行NMS处理
    keep = nms_boxes(boxes, scores)
    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]

    return boxes, classes, scores


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # 未修改，保持原有逻辑（new_shape默认值不影响，实际使用IMG_SIZE）
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)


def calculate_distance(real_size, focal_length, pixel_size):
    # 未修改，保持原有逻辑
    if pixel_size <= 0 or pixel_size < 1e-3:  # 防止除零错误
        return 0.0, False
    distance = (real_size * focal_length) / pixel_size
    return round(distance, 2), True  # 保留2位小数


# 关键修改4：新增报警触发函数
def trigger_alert(class_name, distance):
    """触发报警反馈（示例使用打印日志，实际可扩展为声音/灯光报警）"""
    alert_msg = f"警告：检测到{class_name}距离过近！当前距离：{distance}米（阈值：{ALERT_DISTANCE}米）"
    print(alert_msg)  # 实际使用时可替换为硬件控制代码（如GPIO输出/蜂鸣器控制）
    # 可选扩展：cv2.putText(image, alert_msg, ...) 显示红色警告文字
    # 可选扩展：使用pygame播放音频（需安装pygame库）


def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, class_idx in zip(boxes, scores, classes):
        # 原始坐标格式：[x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        
        # 将坐标转换回原始图像尺寸（保持不变）
        orig_x1 = (x1 - padding[0]) / ratio[0]
        orig_y1 = (y1 - padding[1]) / ratio[1]
        orig_x2 = (x2 - padding[0]) / ratio[0]
        orig_y2 = (y2 - padding[1]) / ratio[1]
        orig_x1 = max(0, min(int(orig_x1), image.shape[1]))
        orig_y1 = max(0, min(int(orig_y1), image.shape[0]))
        orig_x2 = max(0, min(int(orig_x2), image.shape[1]))
        orig_y2 = max(0, min(int(orig_y2), image.shape[0]))
        pixel_height = orig_y2 - orig_y1  # 垂直方向像素高度

        # 获取当前类别名称（自动支持face）
        class_name = CLASSES[class_idx].lower()

        # 根据类别获取实际尺寸和最小像素阈值（自动使用face的配置）
        if class_name not in TARGET_DIMENSIONS:
            continue  # 非目标类别跳过
        dim = TARGET_DIMENSIONS[class_name]
        real_height = dim['height']
        min_pixel = dim['min_pixel']

        # 计算距离（关键修改5：增加报警判断）
        distance_text = ""
        if pixel_height < min_pixel:
            distance_text = f" {class_name.capitalize()} too far"
        else:
            distance, reliable = calculate_distance(real_height, FOCAL_LENGTH, pixel_height)
            if reliable:
                # 触发报警逻辑
                if distance < ALERT_DISTANCE:
                    trigger_alert(class_name, distance)  # 调用报警函数
                distance_text = f' {distance}m'
            else:
                distance_text = " Distance invalid"

        # 绘制边界框（保持不变）
        cv2.rectangle(image, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
        
        # 显示标签（包含新增的face类别）
        label = f'{class_name.capitalize()} {score:.2f}{distance_text}'
        cv2.putText(image, label, (orig_x1, orig_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image


# -------------------------- 必须包含的 myFunc 函数 --------------------------
def myFunc(rknn_lite, IMG):
    """主处理函数，供 main.py 调用 - 优化版本"""
    import time
    start_time = time.time()
    
    # 图像预处理
    preprocess_start = time.time()
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)  # BGR转RGB
    IMG2, ratio, padding = letterbox(IMG2, new_shape=IMG_SIZE)  # 使用修改后的IMG_SIZE
    IMG2 = np.expand_dims(IMG2, 0)               # 增加batch维度
    preprocess_time = time.time() - preprocess_start
    
    # 模型推理
    inference_start = time.time()
    outputs = rknn_lite.inference(inputs=[IMG2], data_format=['nhwc'])
    inference_time = time.time() - inference_start
    
    # 后处理
    postprocess_start = time.time()
    boxes, classes, scores = yolov8_post_process(outputs)
    postprocess_time = time.time() - postprocess_start
    
    # 绘制结果
    draw_start = time.time()
    if boxes is not None:
        draw(IMG, boxes, scores, classes, ratio, padding)
    draw_time = time.time() - draw_start
    
    total_time = time.time() - start_time
    
    # 性能监控（每50次推理输出一次统计）
    if hasattr(myFunc, 'call_count'):
        myFunc.call_count += 1
    else:
        myFunc.call_count = 1
    
    if myFunc.call_count % 50 == 0:
        print(f"推理性能统计 (第{myFunc.call_count}次):")
        print(f"  预处理: {preprocess_time*1000:.1f}ms, 推理: {inference_time*1000:.1f}ms, "
              f"后处理: {postprocess_time*1000:.1f}ms, 绘制: {draw_time*1000:.1f}ms, "
              f"总计: {total_time*1000:.1f}ms, 理论FPS: {1.0/total_time:.1f}")
    
    return IMG  # 返回标注后的图像