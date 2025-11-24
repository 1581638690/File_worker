import zipfile
import os
from .base import BaseDetector
from File_worker.utils.safe_extract import safe_extract

class ZipDetector(BaseDetector):

    def extract(self, filepath, workdir):
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                namelist = zf.namelist()
                if not namelist:
                    return None
                # safe extract to a subdir named after file
                subdir = os.path.join(workdir, os.path.basename(filepath) + "_zip")
                os.makedirs(subdir, exist_ok=True)
                safe_extract(zf, subdir)
                # return list of extracted paths relative to subdir
                return [os.path.join(subdir, n) for n in namelist if n and not n.endswith('/')]
        except Exception:
            return None
