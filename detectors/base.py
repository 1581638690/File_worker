class BaseDetector:

    def extract(self, filepath, workdir):
        """
        如果是压缩文件 → 解压并返回子文件列表
        默认返回 None，表示非压缩文件。
        """
        return None

    def extract_text(self, filepath):
        """
        抽取文件正文内容
        """
        return ""

    def extract_images(self, filepath):
        """
        返回图片路径列表
        """
        return []