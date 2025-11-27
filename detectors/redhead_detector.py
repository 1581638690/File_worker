import cv2
import numpy as np
import glob
import os
from os import path as osp
import shutil

def is_red_image(img_path, top_ratio=0.25, red_ratio_threshold=0.01):
    """
    检测图片顶部是否为红头文件

    参数:
        img_path: 图片路径
        top_ratio: 只检测顶部区域高度占比，默认 0.25
        red_ratio_threshold: 红色像素比例阈值，超过则判定为红头，默认 1%
    
    返回:
        (is_red_head, red_ratio)
        is_red_head: bool, 是否为红头
        red_ratio: 红色像素占比
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片：{img_path}")
        return False, 0.0
    
    h, w = img.shape[:2]
    top_region = img[0:int(h*top_ratio), :]

    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)

    # 红色区间1
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    # 红色区间2
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 可选：去噪
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    red_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    red_ratio = red_pixels / total_pixels

    is_red_head = red_ratio >= red_ratio_threshold
    return is_red_head, red_ratio

def detect_red_header(filepath):
    """
        红头文件检测逻辑
        返回：
            is_red_header:bool是否为红头文件
            confidence:float置信度
    """

if __name__ == "__main__":
    filepaths = [
        "2eefa80f-f3bd-4e4f-8278-fb336f0c1d59.png",
        "7e72dd2e-4b80-4daf-9e16-08cb0fab8d1d.png",
        "967bd36b-def0-4ff9-bd16-c7eb28265425.png",
        "86205ced5a889596c3bfcfd589785f15.png",
        "11073889-ea3a-4813-8d2d-bd16b2bbc1d8.png",
        "533856970.jpg",
        "webwxgetmsgimg.jpeg"
    ]
    for i in filepaths:
        filepath = f"/opt/openfbi/pylibs/File_worker/test_file/{i}"
        s = is_red_image(filepath)
        print(s)