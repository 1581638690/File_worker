import sys
sys.path.append("/opt/openfbi/fbi-bin/")
from avenger.fglobals import *
from avenger.fsys import b64
ssdb0 = fbi_global.get_ssdb0()
# 获取需要进行配置的信息
def ocr_config(key):
    # 获取ocr配置信息
    ocr_config = ssdb0.get(b64(key))
    if ocr_config:
        return ocr_config
    return {}