import os
import rarfile
#3from unrar import rarfile
from .base import BaseDetector
from File_worker.utils.safe_extract import safe_extract

class RarDetector(BaseDetector):

    def extract(self, filepath, workdir):
        try:
            rf = rarfile.RarFile(filepath)
            namelist = rf.namelist()
            if not namelist:
                return None
            subdir = os.path.join(workdir, os.path.basename(filepath) + "_rar")
            os.makedirs(subdir, exist_ok=True)
            safe_extract(rf, subdir)
            return [os.path.join(subdir, n) for n in namelist if n and not n.endswith('/')]
        except Exception:
            return None
