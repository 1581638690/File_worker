# detectors/sevenz_detector.py
import os
import py7zr
import shutil
import hashlib

class SevenZDetector:
    """
    解压 7z 文件
    """

    def extract(self, filepath, workdir):
        """
        解压 7z 文件到工作目录，返回子文件列表
        """
        extracted_files = []

        try:
            # 目标解压目录
            extract_dir = os.path.join(
                workdir,
                os.path.splitext(os.path.basename(filepath))[0]
            )
            os.makedirs(extract_dir, exist_ok=True)

            try:
                with py7zr.SevenZipFile(filepath, mode='r') as archive:
                    archive.extractall(path=extract_dir)
            except py7zr.exceptions.PasswordRequired:
                raise Exception(f"7Z 文件加密，需要密码: {filepath}")
            except Exception as e:
                raise Exception(f"7Z 解压失败: {e}")

            # 枚举所有文件
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    extracted_files.append(full_path)

            return extracted_files

        except Exception as e:
            print("7z 解压失败：", e)
            return None  # return None = 交给主逻辑判断 zip 失败继续走文本解析

    def extract_text(self, filepath, ftype=None):
        return ""

    def extract_images(self, filepath, ftype=None):
        return []


