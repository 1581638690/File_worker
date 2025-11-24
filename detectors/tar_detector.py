# detectors/tar_like_detector.py
import os
import tarfile

class TarDetector:
    """
    统一处理 tar.gz / tgz / tar.bz2 / tbz / tar.xz / txz
    """

    def extract(self, filepath, workdir):
        extracted_files = []

        # 判断是否是 tar 系列压缩
        if not tarfile.is_tarfile(filepath):
            return None

        try:
            extract_dir = os.path.join(
                workdir,
                os.path.splitext(os.path.basename(filepath))[0]
            )
            os.makedirs(extract_dir, exist_ok=True)

            with tarfile.open(filepath, "r:*") as tar:
                tar.extractall(extract_dir)

            # 获取所有文件
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    extracted_files.append(os.path.join(root, f))

            return extracted_files

        except Exception as e:
            print("tar 解压失败：", e)
            return None

    def extract_text(self, filepath, ftype=None):
        return ""

    def extract_images(self, filepath, ftype=None):
        return []
