import cv2
import time
import video_number
from rknnpool_rgb import rknnPoolExecutor_face
from func_face import myFunc_face
from rec_co import command

def face():
    print("rgb camera start")
    for number in video_number.rgb_numbers:
        cap = cv2.VideoCapture(number)
        if cap.isOpened():
            print(f"Found openable camera: {number}")
            break


    #cap = cv2.VideoCapture(0)

    model_path = 'model_data/retinaface_mob.rknn'
    model_path2 = 'model_data/mobilefacenet.rknn'
    # 线程数, 增大可提高帧率
    TPEs = 3
    # 初始化rknn池
    pool = rknnPoolExecutor_face(
        rknnModel1=model_path,
        rknnModel2=model_path2,
        TPEs=TPEs,
        func=myFunc_face)

    if not cap.isOpened():
        print("could not open camera")

    # 初始化异步所需要的帧
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

    frames, loopTime, initTime = 0, time.time(), time.time()
    # 添加一个循环来持续读取摄像头的帧图像并放入队列
    while cap.isOpened() and command == "0":
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, flag = pool.get()

        if not flag:
            break
        cv2.imshow('test', frame)
        frames += 1  # 更新处理的帧数
        if frames % 30 == 0:
            #print(time.time() - loopTime)
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
            loopTime = time.time()
    cap.release()
    cv2.destroyAllWindows()
    pool.release()


