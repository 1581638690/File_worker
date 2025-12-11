import os
import json
import sys
sys.path.append("/opt/openfbi/pylibs/")
from File_worker.utils.filetype_detect import guess_magic_from_path
from File_worker.detectors.zip_detector import ZipDetector
from File_worker.detectors.rar_detector import RarDetector
from File_worker.detectors.tar_detector import TarDetector
from File_worker.detectors.gz_detector import GzDetector
from File_worker.detectors.office_detector import OfficeDetector
from File_worker.detectors.pdf_detector import PdfDetector
from File_worker.detectors.text_detector import TextDetector
from File_worker.detectors.sevenz_detector import SevenZDetector
from File_worker.utils.ocr_extract import *
from File_worker.utils.encrypt_detector import *
from File_worker.detectors.redhead_detector import *
from File_worker.utils.ssdb_config import *
import requests
import shutil
import tempfile
import base64


# 类型处理器表
DETECTOR_TABLE = {
    "zip": ZipDetector(),
    "rar": RarDetector(),
    "tar": TarDetector(),
    "gz": GzDetector(),
    "bz2": GzDetector(),
    "xz": GzDetector(),
    "7z": SevenZDetector(),

    "tgz": TarDetector(),
    "tbz": TarDetector(),
    "txz": TarDetector(),
    "tar.gz": TarDetector(),
    "tar.bz2": TarDetector(),
    "tar.xz": TarDetector(),

    "docx": OfficeDetector(),
    "xlsx": OfficeDetector(),
    "pptx": OfficeDetector(),

    "pdf": PdfDetector(),
    "text": TextDetector(),
}


def ensure_workdir(workdir):
    os.makedirs(workdir, exist_ok=True)
    return workdir


def check_encryption(filepath, ftype):
    """统一加密检测入口"""
    if ftype == "pdf":
        return FileEncryptDetector.is_pdf_encrypted(filepath)
    elif ftype in ("docx", "xlsx", "pptx"):
        return FileEncryptDetector.is_office_encrypted(filepath)
    elif ftype == "zip":
        return FileEncryptDetector.is_zip_encrypted(filepath)
    elif ftype == "7z":
        return FileEncryptDetector.is_7z_encrypted(filepath)
    return False, ""


#def down_config(strs):
    #ocr_config(key)


def ocr_identify(image_path,API_URL="http://192.168.124.78:8080/layout-parsing"):
    #API_URL = "http://192.168.124.78:8080/layout-parsing" # 服务URL

    # 对本地图像进行Base64编码
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "file": image_data, # Base64编码的文件内容或者文件URL
        "fileType": 1, # 文件类型，1表示图像文件
    }

    # 调用API
    response = requests.post(API_URL, json=payload)

    # 处理接口返回数据
    if response.status_code == 200:
        result = response.json()["result"]
        for i, res in enumerate(result["layoutParsingResults"]):
            return res["prunedResult"]
    print(f"ocr请求状态码：{response.status_code}")
    return {}
def call_ocr_service_new(image_paths,model_url = None):
    ####### ocr 处理 多图片处理#######
    ocr_detail =[] 
    ocr_text = ""
    for p in image_paths:
        output = ocr_identify(p,API_URL=model_url)
        info = extract_ocr_info_v2(output) # 从提取数据的图片中获取数据信息
        ocr_detail.append({"image": p, "ocr": info})
        ocr_text += info.get("full_text", "")
    return ocr_text, ocr_detail


def stmp_identify(image_path,API_URL="http://192.168.124.78:8081/seal-recognition"):
    #API_URL = "http://192.168.124.78:8081/seal-recognition" # 服务URL

    with open(image_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    payload = {"file": file_data, "fileType": 1}

    response = requests.post(API_URL, json=payload)
    print(response.status_code)
    if response.status_code == 200:
        result = response.json()["result"]
        for i, res in enumerate(result["sealRecResults"]):
            return res["prunedResult"]
    else:
        print(f"印章请求状态码：{response.status_code}")
        return {}

def call_stamp_service_new(image_paths,model_url = None):
    stamp_all = []
    stamp_detail = ""
    for p in image_paths:
        output = stmp_identify(p,API_URL=model_url)
        stamp_info = detect_stamp_from_image_new(output) # 印章识别
        if stamp_info:
            stamp_all.append({"image": p, "stamps": stamp_info})
        for i in stamp_info:
            stamp_detail += i.get("text","") + "\n"
    stamp_detected = len(stamp_detail) > 0
    return stamp_all,stamp_detail,stamp_detected




# 创建临时文件信息
def prepare_for_ocr(filepath, ftype):
    """
    生成唯一临时目录和临时文件路径，用于 OCR 调用
    返回 (tmp_file_path, tmp_dir)
    """
    ocr_types = ["pdf", "jpg", "jpeg", "png", "bmp"]
    if ftype not in ocr_types:
        return None, None

    tmp_dir = tempfile.mkdtemp(prefix="ocr_task")  # 自动生成唯一目录
    tmp_path = os.path.join(tmp_dir, os.path.basename(filepath) + f".{ftype}")
    shutil.copy(filepath, tmp_path)
    return tmp_path, tmp_dir


######################主文件提取入口函数#####################
def process_file(filepath, 
                 magic_str=None, 
                 workdir="/tmp/service/",
                 OCR_SERVICE_URL="http://192.168.124.78:8080/layout-parsing",
                 STAMP_SERVICE_URL="http://192.168.124.78:8081/seal-recognition"):
    ensure_workdir(workdir)
    filename = os.path.basename(filepath)
    ftype = guess_magic_from_path(filepath, magic_str)

    # 获取对应的 detector
    detector = DETECTOR_TABLE.get(ftype, TextDetector())

    # 加密检测
    encrypted, encrypted_msg = check_encryption(filepath, ftype)
    if encrypted:
        return {
            "filename": filename,
            "filepath": os.path.abspath(filepath),
            "filetype": ftype,
            "encrypted": True,
            "red_header": False,
            "stamp_detected": False,
            "msg": encrypted_msg
        }

    # 初始化返回结果
    result = {
        "filename": filename,
        "filepath": os.path.abspath(filepath),
        "filetype": ftype,
        "sha256": "",
        "text": "",
        "ocr_text": "",
        "ocr_detail": [],
        "images": [],
        "red_header": False,
        "stamp_detected": False,
        "stamp_detail": [],
        "stamp_text":"",
        "children": [],
    }

    # 处理压缩或容器文件
    try:
        extracted = detector.extract(filepath, workdir)
    except Exception:
        extracted = None

    if extracted:
        for child in extracted:
            child_path = child if os.path.isabs(child) else os.path.join(workdir, child)
            try:
                child_res = process_file(child_path, None, workdir,OCR_SERVICE_URL,STAMP_SERVICE_URL)
                result["children"].append(child_res)
            except Exception:
                continue
        return result

    # ----------------------------
    # 判断文件是否为图片类型
    # ----------------------------
    if ftype in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        result["images"] = [os.path.abspath(filepath)]
    else:
        # 文本抽取
        try:
            result["text"] = detector.extract_text(filepath, ftype)
        except Exception:
            result["text"] = ""

        # 图片抽取
        try:
            result["images"] = detector.extract_images(filepath, ftype=ftype) or []
        except Exception:
            result["images"] = []


    # ------------------ OCR & 印章处理 ------------------
    if result["images"]:
        tmp_files = []
        tmp_dirs = []

        for img_path in result["images"]:
            img_ftype = guess_magic_from_path(img_path)
            tmp_path, tmp_dir = prepare_for_ocr(img_path, img_ftype)

            if tmp_path:
                tmp_files.append(tmp_path)
                tmp_dirs.append(tmp_dir)

        if tmp_files:
            # OCR
            ocr_text, ocr_detail = call_ocr_service_new(result["images"],model_url = OCR_SERVICE_URL)
            result["ocr_text"] = ocr_text
            result["ocr_detail"] = ocr_detail

            # 印章
            stamp_all,stamp_detail,stamp_detected = call_stamp_service_new(result["images"],model_url = STAMP_SERVICE_URL)
            result["stamp_detected"] = stamp_detected # 是否包含
            result["stamp_detail"] = stamp_all # 印章详情
            result["stamp_text"] = stamp_detail # 文本

        # 进行红头文件识别
        
        is_red_head, red_ratio = is_red_image(img_path, top_ratio=0.25, red_ratio_threshold=0.01)
        result["red_header"] = is_red_head
        # 删除自己生成的临时目录
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    

    return result

################## 分级目录递归解压####################
import uuid

def get_file_text(result):
    """
    文件级文本优先级：
      1) result['text']（PDF/Office 直接解析文本）
      2) 如果没有，则用 ocr_detail 拼接所有图片 OCR
      3) 否则返回空字符串
    """
    if result.get("text"):
        return result.get("text", "")
    ocr_texts = []
    for item in result.get("ocr_detail", []):
        txt = item.get("ocr", {}).get("full_text", "")
        if txt:
            ocr_texts.append(txt)
    return "\n".join(ocr_texts) if ocr_texts else ""

def flatten_image_data(result, parent_path=None, parent_type_hierarchy=None):
    """
    打平：文件级 + 图片级（但仅在有意义时生成文件级）
    规则：
      - 如果文件本身是图片：只生成一条图片级记录（含 OCR/stamp）
      - 否则（文档/容器）：
          * 如果有 file_text 或 stamp_detail -> 生成文件级记录
          * 对 result['images'] 中每张图片都生成图片级记录
      - 递归处理 children
    """
    rows = []

    parent_path = parent_path or result.get("filepath")
    parent_type_hierarchy = parent_type_hierarchy or []

    # 共同信息
    filetype = result.get("filetype")
    filepath = result.get("filepath")
    filename = result.get("filename")
    current_type_hierarchy = parent_type_hierarchy + [filetype]
    # -------------- 判断文件是否为加密文件 ------------------
    if result.get("encrypted", False):
        rows.append({

            "level": "file",
            "parent_path": parent_path,
            "file_name": filename,
            "file_type": filetype,
            "type_hierarchy": current_type_hierarchy,
            "image_path": "",
            "file_text": "",
            "ocr_text": "",
            "stamp_detected": False,
            "stamps": [],
            "stamp_text":"",
            "download_path": filepath,
            "msg": result.get("msg", "加密文件，无法处理")
        })
        return rows

    image_types = {"jpg", "jpeg", "png", "bmp", "tiff"}

    # 如果当前文件本身是图片 -> 只生成一条图片级记录
    if filetype in image_types:
        # 找匹配的 OCR & stamps
        ocr_text = ""
        for ocr_item in result.get("ocr_detail", []):
            if ocr_item.get("image") == filepath:
                ocr_text = ocr_item.get("ocr", {}).get("full_text", "")
                break

        stamps_for_img = []
        stamp_text = ""
        for item in result.get("stamp_detail", []):
            if item.get("image") == filepath:
                stamps_for_img = [s for s in item.get("stamps", []) if s.get("score", 0) >= 0.3]
                for s in item.get("stamps",[]):
                    stamp_text += s.get("text","") + "\n"
                break

        img_ext = filetype
        rows.append({
            
            "level": "image",
            "parent_path": parent_path,
            "file_name": filename,
            "file_type": filetype,
            "image_type": img_ext,
            "type_hierarchy": current_type_hierarchy,
            "image_path": filepath,
            "file_text": "",
            "ocr_text": ocr_text,
            "stamp_detected": len(stamps_for_img) > 0,
            "red_header":result.get("red_header"),
            "stamps": stamps_for_img,
            "stamp_text":stamp_text,
            "download_path": filepath  # 新增统一下载字段
        })
        return rows

    # 否则：文档或容器（pdf / zip / office 等）
    file_text = get_file_text(result)
    has_stamp = bool(result.get("stamp_detail"))

    # 文件级记录（仅在有意义时生成）
    if file_text or has_stamp:
        rows.append({
            
            "level": "file",
            "parent_path": parent_path,
            "file_name": filename,
            "file_type": filetype,
            "type_hierarchy": current_type_hierarchy,
            "image_path": "",
            "file_text": file_text,
            "ocr_text": result.get("ocr_text", ""),
            "stamp_detected": result.get("stamp_detected", False),
            "stamps": result.get("stamp_detail", []),
            "stamp_text":result.get("stamp_text",""),
            "download_path": filepath  # 文件下载
        })

    # 图片级记录
    for img_path in result.get("images", []):
        stamps_for_img = []
        stamp_text=""
        for item in result.get("stamp_detail", []):
            if item.get("image") == img_path:
                stamps_for_img = [s for s in item.get("stamps", []) if s.get("score", 0) >= 0.3]
                for s in item.get("stamps",[]):
                    stamp_text += s.get("text","") + "\n"
                break

        img_ocr_text = ""
        for ocr_item in result.get("ocr_detail", []):
            if ocr_item.get("image") == img_path:
                img_ocr_text = ocr_item.get("ocr", {}).get("full_text", "")

                break
        img_ext = os.path.splitext(img_path)[-1].lower().replace(".", "")
        rows.append({
            
            "level": "image",
            "parent_path": parent_path,
            "file_name": filename,
            "file_type": filetype,
            "image_type": img_ext,
            "type_hierarchy": current_type_hierarchy + [img_ext],
            "image_path": img_path,
            "file_text": "",
            "ocr_text": img_ocr_text,
            "red_header":result.get("red_header"),
            "stamp_detected": len(stamps_for_img) > 0,
            "stamps": stamps_for_img,
            "stamp_text":stamp_text,
            "download_path": img_path  # 图片下载
        })

    # 递归处理子文件
    for child in result.get("children", []):
        rows.extend(flatten_image_data(
            child,
            parent_path=filepath,
            parent_type_hierarchy=current_type_hierarchy
        ))

    return rows

def run_file(filepath, magic_str=None,workdir='/tmp/service/',OCR_SERVICE_URL="http://192.168.124.78:8080/layout-parsing",STAMP_SERVICE_URL="http://192.168.124.78:8081/seal-recognition"):
    workdir = workdir or os.path.join(os.getcwd(), "tmp_processor")
    res = process_file(filepath, magic_str, workdir,OCR_SERVICE_URL,STAMP_SERVICE_URL)
    files_flatten = flatten_image_data(res)
    return res,files_flatten

if __name__ == "__main__":
    #res,files_flatten = run_file("/data/files/65/6533623063343432393866633163313439616662663463383939366662393234",magic_str = "JPEG image data,JFIF standard 1.01,resilution,density 72*72",workdir = '/opt/openfbi/pylibs/File_worker/')
    #print(res)

    #res,files_flatten = run_file("/opt/openfbi/pylibs/File_worker/test_file/2eefa80f-f3bd-4e4f-8278-fb336f0c1d59.png",workdir = '/opt/openfbi/pylibs/File_worker/tmp_processor')
    res,files_flatten = run_file("/opt/openfbi/pylibs/File_worker/test_file/加密111_2.7z")
    print(res)
    #with open("./11.json","w")as fp:
    #    json.dump(res,fp)
    #print(files_flatten)


    #res = {'filename': '1111.zip', 'filepath': '/opt/openfbi/pylibs/File_worker/test_file/1111.zip', 'filetype': 'zip', 'sha256': None, 'text': '', 'ocr_text': '', 'ocr_detail': [], 'images': [], 'red_header': False, 'stamp_detected': False, 'stamp_detail': [], 'children': [{'filename': 'dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621.pdf', 'filepath': '/opt/openfbi/pylibs/File_worker/1111.zip_zip/dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621.pdf', 'filetype': 'pdf', 'sha256': None, 'text': '电子发票（普通发票）\n发票号码：25352000000097909249\n开票日期：2025年08月27日\n购 名称：杭州瀚海数安科技有限公司 销 名称：福州闽侯县华超酒店有限责任公司\n买 售\n方 方\n信 统一社会信用代码/纳税人识别号：91330108MA2H2FMAXF 信 统一社会信用代码/纳税人识别号：91350121MACECMKX4B\n息 息\n项目名称 规格型号 单 位 数 量 单 价 金 额 税率/征收率 税 额\n*住宿服务*住宿服务 天 1 188.584158415842 188.58 1% 1.89\n合 计 ¥188.58 ¥1.89\n价税合计（大写） 壹佰玖拾圆肆角柒分 （小写）¥190.47\n备\n注\n开票人：刘昌辉\n刘昌辉\n', 'ocr_text': '税\n', 'ocr_detail': [{'image': '/home/rzc/File_worker/images/dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621/dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621.pdf_page0_img0.png', 'ocr': {'full_text': '税\n', 'blocks': [{'content': '税', 'label': 'paragraph_title', 'bbox': [0, 1, 300, 298], 'id': 0, 'order': 1}], 'layout_boxes': [{'label': 'paragraph_title', 'score': 0.7613851428031921, 'bbox': [0.01934814453125, 1.5437774658203125, 300, 298.591064453125]}], 'seal_candidates': []}}], 'images': ['/home/rzc/File_worker/images/dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621/dzfp_25352000000097909249_杭州瀚海数安科技有限公司_20250827122621.pdf_page0_img0.png'], 'red_header': False, 'stamp_detected': False, 'stamp_detail': [], 'children': []}]}
    rows = flatten_image_data(res)
    print(rows)
"""

{ "filename": "3535653631383932326538616536636233336234323637333361636162386663", "filepath": "/data/files/35/3535653631383932326538616536636233336234323637333361636162386663", "filetype": "", "sha256": null, "text": "{\"c\":0,\"flag\":\"1\"}", "ocr_text": "", "ocr_detail": [], "images": [], "red_header": false, "stamp_detected": false, "stamp_detail": [], "children": [] }
{ "filename": "3139396437633634383966323534363839303261376135323664333333323466", "filepath": "/data/files/31/3139396437633634383966323534363839303261376135323664333333323466", "filetype": "", "sha256": null, "text": "adddd\r\n", "ocr_text": "", "ocr_detail": [], "images": [], "red_header": false, "stamp_detected": false, "stamp_detail": [], "children": [] }
{ "filename": "6533623063343432393866633163313439616662663463383939366662393234", "filepath": "/data/files/65/6533623063343432393866633163313439616662663463383939366662393234", "filetype": "jpg", "sha256": null, "text": "", "ocr_text": "", "ocr_detail": [], "images": [ "/data/files/65/6533623063343432393866633163313439616662663463383939366662393234" ], "red_header": false, "stamp_detected": false, "stamp_detail": [], "children": [] }
{ "filename": "6533623063343432393866633163313439616662663463383939366662393234", "filepath": "/data/files/65/6533623063343432393866633163313439616662663463383939366662393234", "filetype": "jpg", "sha256": null, "text": "", "ocr_text": "", "ocr_detail": [], "images": [ "/data/files/65/6533623063343432393866633163313439616662663463383939366662393234" ], "red_header": false, "stamp_detected": false, "stamp_detail": [], "children": [] }

"""


