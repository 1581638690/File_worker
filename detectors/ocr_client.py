import base64
import requests
import pathlib
def ocr_identify(image_path):
    API_URL = "http://192.168.124.78:8080/layout-parsing" # 服务URL

    # 对本地图像进行Base64编码
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "file": image_data, # Base64编码的文件内容或者文件URL
        "fileType": 1, # 文件类型，1表示图像文件
    }

    # 调用API
    response = requests.post(API_URL, json=payload)

    # 处理接口返回数据
    if response.status_code == 200:
        result = response.json()["result"]
        for i, res in enumerate(result["layoutParsingResults"]):
            return res["prunedResult"]
    return {}

def stmp_identify(image_path):
    API_URL = "http://192.168.124.78:8081/seal-recognition" # 服务URL

    with open(image_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    payload = {"file": file_data, "fileType": 1}

    response = requests.post(API_URL, json=payload)
    print(response.status_code)
    if response.status_code == 200:
        result = response.json()["result"]
        for i, res in enumerate(result["sealRecResults"]):
            return res["prunedResult"]
    else:
        print(f"请求状态码:{response.status_code}")
        return {}
if __name__ == "__main__":

    image_path = "/opt/openfbi/pylibs/File_worker/test_file/webwxgetmsgimg.jpeg"
    image_path1 = "/opt/openfbi/pylibs/File_worker/test_file/533856970.jpg"
    res = stmp_identify(image_path1)
    print(res)