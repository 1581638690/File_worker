import pdfplumber
from .base import BaseDetector
import os
import fitz  # PyMuPDF for image extraction
from pathlib import Path
def generate_image_dir(filepath, base_dir="/tmp/service/images"):
    """
    根据源文件生成图片保存目录
    """
    file_stem = Path(filepath).stem
    save_dir = os.path.join(base_dir, file_stem)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

class PdfDetector(BaseDetector):

    def extract_text(self, filepath,ftype=None):
        try:
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text += t + "\n"
            return text
        except Exception:
            return ""

    def extract_images(self, filepath, base_dir="/tmp/service/images", ftype=None):
        save_dir = generate_image_dir(filepath, base_dir)
        imgs = []
        try:
            doc = fitz.open(filepath)

            # 创建目录
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            for i, page in enumerate(doc):
                imglist = page.get_images(full=True)
                for idx, img in enumerate(imglist):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # 统一保存为 PNG
                    ext = "png"
                    save_path = os.path.join(
                        save_dir,
                        f"{os.path.basename(filepath)}_page{i}_img{idx}.png"
                    )

                    # 保存
                    if pix.n < 5:  
                        pix.save(save_path)
                    else:          
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(save_path)

                    imgs.append(save_path)

            return imgs

        except Exception:
            return []
