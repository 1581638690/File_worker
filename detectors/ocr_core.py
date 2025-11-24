import sys
import os

# 添加 paddleocr 可能所在的路径
possible_paths = [
    #"/usr/local/lib/python3.8/site-packages",
    #"/home/zhds/.local/bin",
    # 添加其他可能的路径
]

for path in possible_paths:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)
from paddleocr import PaddleOCRVL

class OCRCore:
    _ocr = None

    @classmethod
    def get_ocr(cls):
        if cls._ocr is None:
            cls._ocr = PaddleOCRVL()
        return cls._ocr

from paddlex import create_pipeline

class SEALReco:
    _seal = None
    @classmethod
    def get_seal(cls):
        if cls._seal is None:
            cls._seal = create_pipeline(pipeline="seal_recognition")
        return cls._seal
    