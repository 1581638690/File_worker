from fastapi import FastAPI, UploadFile, File
from typing import List
import os, shutil, uuid
from detectors.ocr_core import OCRCore, SEALReco
from utils.ocr_extract import extract_ocr_info_v2, detect_stamp_from_image
import uvicorn
app = FastAPI()

# 初始化模型（只加载一次）
OCR = OCRCore.get_ocr()
SEAL_PIPE = SEALReco.get_seal()

TMP_DIR = "/tmp/file_service"
os.makedirs(TMP_DIR, exist_ok=True)


def save_upload_file(upload_file: UploadFile) -> str:
    """保存上传文件到临时目录，返回保存路径"""
    file_ext = os.path.splitext(upload_file.filename)[1]
    tmp_filename = f"{uuid.uuid4().hex}{file_ext}"
    tmp_path = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_path, "wb") as f:
        f.write(upload_file.file.read())
    return tmp_path


def process_ocr_images(img_paths: List[str]):
    """OCR处理"""
    ocr_text = ""
    ocr_detail = []
    for img_path in img_paths:
        try:
            output = OCR.predict(img_path)
            for res in output:
                info = extract_ocr_info_v2(res._to_json())
                ocr_detail.append({"image": img_path, "ocr": info})
                ocr_text += info.get("full_text", "")
        except Exception:
            continue
    return ocr_text, ocr_detail


def process_stamp_images(img_paths: List[str]):
    """印章识别"""
    stamp_all = []
    for img_path in img_paths:
        try:
            stamp_info = detect_stamp_from_image(img_path, SEAL_PIPE)
            if stamp_info:
                stamp_all.append({"image": img_path, "stamps": stamp_info})
        except Exception:
            continue
    return stamp_all


@app.post("/ocr")
async def ocr_service(files: List[UploadFile] = File(...)):
    tmp_paths = [save_upload_file(f) for f in files]
    try:
        ocr_text, ocr_detail = process_ocr_images(tmp_paths)
        return {"ocr_text": ocr_text, "ocr_detail": ocr_detail}
    finally:
        # 清理临时文件
        for path in tmp_paths:
            os.remove(path)


@app.post("/stamp")
async def stamp_service(files: List[UploadFile] = File(...)):
    tmp_paths = [save_upload_file(f) for f in files]
    try:
        stamp_detail = process_stamp_images(tmp_paths)
        stamp_detected = len(stamp_detail) > 0
        return {"stamp_detected": stamp_detected, "stamp_detail": stamp_detail}
    finally:
        # 清理临时文件
        for path in tmp_paths:
            os.remove(path)



if __name__ == "__main__":
    #uvicorn.run("ocr_main:app",host="0.0.0.0",port=5110,reload=True)
    pass
