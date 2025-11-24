import os

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

def detect_office_type(magic_str):
    """
    根据 magic_str 判断 Office 类型（doc, docx, xls, xlsx, ppt, pptx）
    """
    if not magic_str:
        return None
    ms = magic_str.lower()

    # 新版 Office 2007+ OOXML
    if "microsoft word" in ms and "xml" in ms:
        return "docx"
    if "microsoft excel" in ms and "xml" in ms:
        return "xlsx"
    if "microsoft powerpoint" in ms and "xml" in ms:
        return "pptx"
    if "office open xml" in ms:
        if "word" in ms:
            return "docx"
        if "excel" in ms or "sheet" in ms:
            return "xlsx"
        if "presentation" in ms or "powerpoint" in ms:
            return "pptx"
        return "ooxml"   # fallback

    # 旧版 Office 97-2003
    if "cdfv2" in ms or "composite document file v2" in ms:
        if "word" in ms:
            return "doc"
        if "excel" in ms:
            return "xls"
        if "powerpoint" in ms or "presentation" in ms:
            return "ppt"
        return "ole"

    return None

def guess_magic_from_path(path, magic_str=""):
    """
    返回文件类型字符串
    逻辑：
    1. 优先使用传入的 magic_str 进行分类
    2. 如果 magic_str 不存在或没有匹配到类型，则使用 python-magic 获取
    3. fallback 扩展名
    """
    path = os.path.abspath(path)
    file_type = None

    def classify(ms):
        ms = ms.lower()
        # 压缩包
        if "zip" in ms:
            return "zip"
        elif "rar" in ms:
            return "rar"
        elif "tar" in ms:
            return "tar"
        elif "gzip" in ms or "gz" in ms:
            return "gz"
        elif "bzip2" in ms or "bz2" in ms:
            return "bz2"
        elif "xz" in ms:
            return "xz"
        # PDF
        elif "pdf" in ms:
            return "pdf"
        # 图片类型
        elif "image/png" in ms or "png" in ms:
            return "png"
        elif "image/jpeg" in ms or "jpeg" in ms or "jpg" in ms:
            return "jpg"
        elif "image/bmp" in ms or "bmp" in ms:
            return "bmp"
        elif "image/tiff" in ms or "tiff" in ms:
            return "tiff"
        # 文本/HTML/JSON
        elif "html document" in ms:
            return "html"
        elif "json" in ms:
            return "json"
        elif "ascii text" in ms or "utf-8 text" in ms or "unicode text" in ms:
            return "txt"
        else:
            office_type = detect_office_type(ms)
            return office_type

    # Step 1: 尝试用传入的 magic_str 分类
    if magic_str:
        file_type = classify(magic_str)

    # Step 2: 如果没有分类成功，则用 python-magic 获取
    if not file_type and MAGIC_AVAILABLE:
        try:
            import magic
            m = magic.Magic(mime=True)
            magic_str_real = m.from_file(path)
            print("python-magic 获取类型:", magic_str_real)
            file_type = classify(magic_str_real)
        except Exception as e:
            print("获取文件信息错误：", e)

    # Step 3: fallback 扩展名
    if not file_type:
        ext = os.path.splitext(path)[1].lower().strip(".")
        file_type = ext if ext else "unknown"

    return file_type