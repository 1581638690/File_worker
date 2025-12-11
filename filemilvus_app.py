from fastapi import FastAPI,UploadFile,File
from typing import List
import os
import shutil
from pydantic import BaseModel

f_app = FastAPI() # 创建实例化
UPLOAD_TMP = "/tmp/service/"
import sys
sys.path.append("/opt/openfbi/pylibs/")
#from File_worker.milvus_embed import VectorSerive
from File_worker.core import *
from File_worker.milvus_embed_con import *

# ------------------ 全局 VectorSerive ------------------
vs = None
print("连接：",vs)

class MilvusConfig(BaseModel):
    milvus_host: str = "192.168.124.250"
    milvus_port: str = "19530"


@f_app.post("/update_milvus_conf")
async def update_milvus_conf(conf: MilvusConfig):
    """更新 Milvus 配置"""
    global vs
    vs = VectorSerive(milvus_host=conf.milvus_host, milvus_port=conf.milvus_port)
    vs.connect_milvus()
    return {"code": 200, "msg": f"Milvus 配置已更新: {conf.milvus_host}:{conf.milvus_port}"}


@f_app.post("/upload_file")
async def upload_files(files:List[UploadFile]=File(...)):
    """
    支持：单个文件上传
    多文件上传
    """

    os.makedirs(UPLOAD_TMP,exist_ok=True) 

    uploaded_file_paths = []

    for f in files:
        file_path = os.path.join(UPLOAD_TMP,f.filename)
        with open(file_path,"wb")as buffer:
            shutil.copyfileobj(f.file,buffer)
        
        uploaded_file_paths.append(file_path)
    
    # 批量处理文件->向量化->搜索相似文件

    results = []
    
    for file_path in uploaded_file_paths:
        try:
            res = process_file(filepath)
        except:
            res = {}
    
        # 向量化
        combine_text = vs.flatten_result_text(res) # 获取文档中文本信息包含ocr文本信息，印章文本信息
        vector = vs.embed_text_in_chunks(combine_text) #获取文本的向量信息

        # 进行相似向量搜索
        search_results = vs.search_documents(vector,top_k=10)
        results.append({
            "file":os.path.basename(file_path),
            "similar_files":search_results
        })
    return {"code": 200, "data": {"success": True,"similar_datas":results}, "msg": "成功"}


@f_app.post("/upload_single")
async def upload_single(file: UploadFile = File(...)):
    """
    单文件上传接口
    返回文件信息 JSON
    """
    os.makedirs(UPLOAD_TMP, exist_ok=True)

    # 防止路径穿越
    file_name = os.path.basename(file.filename)
    print("上传文件名：",file_name)
    file_path = os.path.join(UPLOAD_TMP, file_name)

    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "code": 200,
        "data": {
            "file_name": file_name,
            "file_path": file_path
        },
        "msg": "上传成功"
    }



class InsertRequest(BaseModel):
    data: List[Dict]


@f_app.post("/insert")
async def insert_data(req: InsertRequest):
    #print("插入数据：",req.data[0])
    #global vs
    if vs is None:
        return {"code": 500, "data": {"success": False}, "msg": "Milvus 尚未初始化"}

    if not req.data:
        return {"code": 400, "data": {"success": False}, "msg": "数据不能为空"}

    try:
        
        ids = vs.batch_insert_documents(req.data)
       
    except Exception as e:
        
        return {"code": 500, "data": {"success": False}, "msg": f"插入失败: {e}"}

    return {"code": 200, "data": {"success": True, "inserted_ids": ids}, "msg": "插入成功"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(f_app,host="0.0.0.0",port=7053)