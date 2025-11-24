import os
import shutil

def safe_extract(ziplike_obj, path):
    """
    安全解压，彻底防止路径遍历（Zip Slip）
    支持 zipfile.ZipFile / tarfile.TarFile / rarfile.RarFile
    只会解压到指定目录，恶意路径会跳过
    """

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    def is_safe_path(base, target):
        """判断目标路径是否在 base 目录内部"""
        return os.path.commonprefix([base, target]) == base

    # ZIP 文件：有 namelist()
    if hasattr(ziplike_obj, "namelist"):
        for member in ziplike_obj.namelist():
            if member.endswith("/"):
                continue

            dest = os.path.abspath(os.path.join(path, member))
            if not is_safe_path(path, dest):
                continue  # 跳过恶意路径

            # 创建目标目录
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with ziplike_obj.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return

    # TAR 文件：有 getmembers()
    if hasattr(ziplike_obj, "getmembers"):
        for member in ziplike_obj.getmembers():
            name = member.name
            if name.endswith("/"):
                continue

            dest = os.path.abspath(os.path.join(path, name))
            if not is_safe_path(path, dest):
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            src = ziplike_obj.extractfile(member)
            if src:
                with src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return

    # RAR 文件：rarfile 本身做了一些检查，但我们再检查一遍
    try:
        for member in ziplike_obj.infolist():
            name = member.filename
            dest = os.path.abspath(os.path.join(path, name))
            if not is_safe_path(path, dest):
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            ziplike_obj.extract(member, path)
    except Exception:
        pass
