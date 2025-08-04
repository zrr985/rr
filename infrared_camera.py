import cv2
import time
import threading
from rec_co import command
from rknnpool_inf import rknnPoolExecutor_inf
from func_v7 import myFunc_inf

def infrared():
    print("infrard camera start")
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    modelPath = "./yolov7_tiny-a.rknn"
    # 线程数, 增大可提高帧率
    TPEs = 3
    #初始化rknn池
    pool = rknnPoolExecutor_inf(
        rknnModel=modelPath,
        TPEs=TPEs,
        func=myFunc_inf)
    if not cap.isOpened():
        print("could not open camera")

    # 初始化异步所需要的帧
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

    frames, loopTime, initTime = 0, time.time(), time.time()
    while (cap.isOpened() and command == '1'):
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, flag = pool.get()
        if flag == False:
            break
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
            print(threading.active_count())
            print(threading.enumerate())
            loopTime = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
           # 释放cap和rknn线程池
            cap.release()
            cv2.destroyAllWindows()
            pool.release()

    cap.release()
    cv2.destroyAllWindows()
    pool.release()

