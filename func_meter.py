#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
import time

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640

CLASSES = ("background", "pointer", "scale")


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
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
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

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

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     import torch
#     x = torch.tensor(position)
#     n,c,h,w = x.shape
#     p_num = 4
#     mc = c//p_num
#     y = x.reshape(n,p_num,mc,h,w)
#     y = y.softmax(2)
#     acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
#     y = (y*acc_metrix).sum(2)
#     return y.numpy()

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     n, c, h, w = position.shape
#     p_num = 4
#     mc = c // p_num
#     y = position.reshape(n, p_num, mc, h, w)
#     exp_y = np.exp(y)
#     y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
#     acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
#     y = (y * acc_metrix).sum(2)
#     return y

def dfl(position):
    # Distribution Focal Loss (DFL)
    # x = np.array(position)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    
    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y
    

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def yolov8_seg_post_process(input_data):
    proto = input_data[-1]
    boxes, scores, classes_conf, seg_parts = [], [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
        seg_parts.append(input_data[pair_per_branch*i+3])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_parts = [sp_flatten(_v) for _v in seg_parts]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_parts = np.concatenate(seg_parts)

    boxes, classes, scores, seg_parts = filter_boxes(boxes, scores, classes_conf, seg_parts)

    zipped = zip(boxes, classes, scores, seg_parts)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return None, None, None, None
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_parts = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_parts = [np.array(x) for x in result]

    nboxes, nclasses, nscores, nseg_parts = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        n = seg_parts[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
            nseg_parts.append(n[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_parts = np.concatenate(nseg_parts)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_parts.shape[-1], -1)
    seg_imgs = np.matmul(seg_parts, proto)
    seg_imgs = sigmoid(seg_imgs)
    seg_imgs = seg_imgs.reshape(-1, ph, pw)

    downsampled_bboxes = boxes.copy()
    downsampled_bboxes[:, 0] *= pw / 640
    downsampled_bboxes[:, 2] *= pw / 640
    downsampled_bboxes[:, 3] *= ph / 640
    downsampled_bboxes[:, 1] *= ph / 640

    seg_imgs_cropped = _crop_mask(seg_imgs, downsampled_bboxes)
    seg_imgs_resize = np.array([cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) for img in seg_imgs_cropped])
    seg_imgs = seg_imgs_resize > 0.5

    # 根据类别提取 pointer 和 scale 的掩码
    pointer_mask = seg_imgs[classes == 1]  # 类别 1: pointer
    scale_mask = seg_imgs[classes == 2]  # 类别 2: scale

    return boxes, classes, scores, seg_imgs, pointer_mask, scale_mask


# def _crop_mask(masks, boxes):
#     """
#     "Crop" predicted masks by zeroing out everything not in the predicted bbox.
#     Vectorized by Chong (thanks Chong).

#     Args:
#         - masks should be a size [n, h, w] numpy array of masks
#         - boxes should be a size [n, 4] numpy array of bbox coords in relative point form
#     """

#     n, h, w = masks.shape
#     x1, y1, x2, y2 = np.split(boxes, 4, axis=1)  # x1 shape(n,1,1)
#     x1, y1, x2, y2 = [np.expand_dims(coord, axis=(1,2)) for coord in [x1, y1, x2, y2]]  # expand dimensions to match r and c
#     r = np.arange(w)[None, None, :]  # rows shape(1,w,1)
#     c = np.arange(h)[None, :, None]  # cols shape(h,1,1)
    
#     return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def _crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    r = np.arange(w, dtype=np.float32)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=np.float32)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        top = (top - padding[0])/ratio[0]
        left = (left - padding[1])/ratio[1]
        right = (right - padding[0])/ratio[0]
        bottom = (bottom - padding[1])/ratio[1]
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)

        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        
# def merge_seg(image, seg_img, classes):
#     color = Colors()
#     for i in range(len(seg_img)):
#         seg = seg_img[i]
#         seg = seg.astype(np.uint8)
#         seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
#         seg = seg * color(classes[i])
#         seg = seg.astype(np.uint8)
#         image = cv2.add(image, seg)
#     return image

def merge_seg(image, seg_imgs, classes,padding):
    seg_img = np.sum(seg_imgs, axis=0)
    seg = seg_img.astype(np.uint8)
    seg = seg * 128
    seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    
    if padding[1] == 0 :
        if padding[0] != 0:
            seg2 = seg[:, padding[0]:-padding[2]]
        else:
            seg2 = seg
    else:
        if padding[0] == 0:
            seg2 = seg[padding[1]:-padding[3], :]
        else:
            seg2 = seg[padding[1]:-padding[3], padding[0]:-padding[2]]
            
    seg = cv2.resize(seg2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    image = cv2.add(image, seg)    
    return image

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    #return im
    return im, ratio, (left, top, right, bottom)
    
def myFunc(rknn_lite, IMG):
    initTime = time.time()
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

    IMG2, ratio, padding = letterbox(IMG2)
    # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    IMG2 = np.expand_dims(IMG2, 0)
    prepostTime = time.time()
    # print("预处理时间:\t", prepostTime - initTime, "秒")
    outputs = rknn_lite.inference(inputs=[IMG2],data_format=['nhwc'])
    inferenceTime = time.time()
    # print("推理时间:\t", inferenceTime - prepostTime, "秒")
    #print("oups1",len(outputs))
    #print("oups2",outputs[0].shape)

    boxes, classes, scores, seg_imgs, pointer_mask, scale_mask = yolov8_seg_post_process(outputs)
    postprocessTime = time.time()
    # print("后处理时间:\t", postprocessTime - inferenceTime, "秒")

    if boxes is not None:
        IMG = merge_seg(IMG, seg_imgs, classes, padding)
        #draw(IMG, boxes, scores, classes, ratio, padding)
    return IMG, pointer_mask, scale_mask
