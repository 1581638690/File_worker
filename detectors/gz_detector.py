import gzip
import bz2
import lzma
import os
from .base import BaseDetector
from File_worker.utils.guess_encoding import decode_bytes

class GzDetector(BaseDetector):

    def extract_text(self, filepath,ftype=None):
        try:
            data = None
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rb") as f:
                    data = f.read()
            elif filepath.endswith(".bz2"):
                with bz2.BZ2File(filepath) as f:
                    data = f.read()
            elif filepath.endswith(".xz"):
                with lzma.open(filepath, "rb") as f:
                    data = f.read()
            if data:
                return decode_bytes(data)
        except Exception:
            pass
        return ""
