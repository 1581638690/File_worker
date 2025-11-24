from .base import BaseDetector
from File_worker.utils.guess_encoding import decode_bytes

class TextDetector(BaseDetector):

    def extract_text(self, filepath,ftype=None):
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
            return decode_bytes(raw)
        except Exception:
            return ""
