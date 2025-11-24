
def decode_bytes(data: bytes):
    if data is None:
        return ""
    try:
        return data.decode("utf-8")
    except Exception:
        try:
            return data.decode("gbk")
        except Exception:
            try:
                return data.decode("latin1")
            except:
                return ""
