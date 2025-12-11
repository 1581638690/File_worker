import os
import json
import time
import numpy as np
from typing import List, Dict
from pymilvus import MilvusClient, DataType

import logging
import sys

sys.path.append("/opt/openfbi/pylibs/")
from File_worker.utils.embedding_con import *
from File_worker.core import *
logger = logging.getLogger("VectorService")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("vector_service.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class VectorSerive:
    def __init__(self,
                 milvus_host="127.0.0.1",
                 milvus_port="19530",
                 embedding_service_url="http://192.168.124.78:8084",
                 api_key="no-key",
                 collection_name="documents_xlink_con",
                 embedding_dim=1024):

        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_service_url = embedding_service_url
        self.api_key = api_key
        self.client = None  # MilvusClient 初始化为空

        # 初始化 embedding 服务
        self.ai_client = Llama_Cpp_AIService(
            embedding_base_url=self.embedding_service_url,
            api_key=self.api_key
        )

        logger.info(f"VectorService instance created for collection '{self.collection_name}'.")

    # ------------------ Milvus 连接函数 ------------------
    def connect_milvus(self):
        """显式连接 Milvus，必要时创建 collection"""
        if self.client is not None:
            logger.info("MilvusClient already connected.")
            return

        self.client = MilvusClient(
            uri=f"http://{self.milvus_host}:{self.milvus_port}",
            token="root:Milvus"
        )
        logger.info(f"Connecting MilvusClient: {self.milvus_host}:{self.milvus_port}")

        if not self.client.has_collection(self.collection_name):
            self._create_collection()

        logger.info(f"Milvus collection '{self.collection_name}' ready.")

        
    # ---------------- 更新向量服务 URL ----------------
    def update_embedding_url(self, new_url: str):
        self.ai_client.base_url = new_url

    # ---------------- 创建 schema + collection ----------------
    def _create_collection(self):

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )

        schema.add_field("file_id", DataType.INT64, is_primary=True)  # auto_id 主键
        schema.add_field("id", DataType.VARCHAR, max_length=200)
        schema.add_field("file", DataType.VARCHAR, max_length=200)
        schema.add_field("text", DataType.VARCHAR, max_length=60000)
        schema.add_field("category_id", DataType.INT64)
        schema.add_field("tags", DataType.VARCHAR, max_length=50)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.embedding_dim)

        index_params = self.client.prepare_index_params()
        # 创建向量索引
        index_params.add_index(
            collection_name=self.collection_name,
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )

        # 创建 collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        self.client.get_load_state(self.collection_name) # 加载
        logger.info("Collection created with HNSW index.")

    # ---------------- 文本合并 / embedding ----------------
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.replace("\r", "\n").split()).strip()

    def flatten_result_text(self, result: Dict) -> str:
        parts = []
        for key in ["text", "ocr_text", "stamp_text"]:
            if result.get(key):
                parts.append(result[key])

        children = result.get("children", [])
        if isinstance(children, str):
            children = json.loads(children)

        for ch in children:
            parts.append(self.flatten_result_text(ch))

        return self.normalize_text("\n".join(parts))

    def text_to_vector(self, text: str) -> List[float]:
        if not text:
            return []
        try:
            resp = self.ai_client.embedding(text, model="GPT-4")
            return resp[0]["embedding"]
        except Exception as e:
            logger.error(f"[Embedding Error] {e}")
            return []

    def embed_text_in_chunks(self, text: str, chunk_size=512, overlap=50) -> List[float]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += chunk_size - overlap

        vectors = []
        for ch in chunks:
            v = self.text_to_vector(ch)
            if v:
                vectors.append(v)

        if not vectors:
            return []

        return np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()

    # ---------------- 批量插入 ----------------

    def batch_insert_documents(self, fingers: List[Dict]) -> List[str]:

        rows = []

        for f in fingers:
            vec = np.array(f.get("vector", []), dtype=np.float32).tolist()
            if not vec:
                continue

            row = {
                #"file_id": f.get("id", ""),       # 主键
                "id": f.get("id", ""),
                "file": f.get("filename", ""),
                "text": f.get("combined_text", ""),
                "category_id": f.get("category_id", -1),
                "tags": f.get("tags", ""),
                "embedding": vec
            }

            rows.append(row)

        if rows:
            #logger.info(f"插入数据文件名：{[(f['file'], f['id']) for f in rows]}")
            res = self.client.insert(collection_name=self.collection_name, data=rows)
        else:
            res = {}
        return res

    # def batch_insert_documents(self, fingers: List[Dict]) -> List[str]:
    #     if not fingers:
    #         return []

    #     data_list = []
    #     ids = []

    #     for finger in fingers:
    #         text = finger.get("combined_text", "")
    #         vector = finger.get("vector", [])

    #         if not vector:
    #             vector = self.embed_text_in_chunks(text)

    #         if not vector:
    #             continue

    #         #ids.append(finger["id"])

    #         data_list.append({
    #             "id": finger["id"],
    #             "file": finger.get("filename", ""),
    #             "text": text,
    #             "category_id": finger.get("category_id", -1),
    #             "tags": finger.get("tags", ""),
    #             "embedding": vector
    #         })
    #     logger.info(f"输出连接器：{self.client}")
    #     if data_list:
    #         logger.info(f"插入数据文件名：{[(f['file'], f['id']) for f in data_list]}")
    #         res = self.client.insert(collection_name=self.collection_name, data = data_list)
    #         self.client.flush(self.collection_name)

    #     return res

    # ---------------- 搜索 ----------------
    def search_documents(self,
                         vector,
                         top_k=5,
                         metric_type="COSINE",
                         score_threshold=None,
                         filter_category_id=None,
                         extra_filters=None):

        expr_parts = []
        if filter_category_id is not None:
            expr_parts.append(f"category_id == {filter_category_id}")

        if extra_filters:
            for k, v in extra_filters.items():
                if isinstance(v, str):
                    expr_parts.append(f'{k} == "{v}"')
                else:
                    expr_parts.append(f"{k} == {v}")

        expr = " and ".join(expr_parts) if expr_parts else None

        res = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="embedding",
            limit=top_k,
            filter=expr,
            search_params={"metric_type": metric_type, "params": {"ef": 64}},
            output_fields=["id", "file", "text", "tags", "category_id"]
        )

        results = []
        for hit in res[0]:
            score = float(hit["distance"])
            if score_threshold is not None:
                if metric_type in ["COSINE", "IP"]:
                    if score < score_threshold:
                        continue
                else:
                    if score > score_threshold:
                        continue

            results.append(hit)

        return results
    
if __name__ == "__main__":
    vs = VectorSerive()
    data = [
            {
            "id": "doc_001",
            "filename": "/opt/openfbi/pylibs/File_worker/test_file/1111.pdf",
            "text": "",
            "ocr_text": "",
            "stamp_text": "",
            "children": []
        },
        {
            "id": "doc_002",
            "filename": "/opt/openfbi/pylibs/File_worker/test_file/25339190041004130573-电子发票.pdf",
            "text": "",
            "ocr_text": "",
            "stamp_text": "",
            "children": []
        },
        {
            "id": "doc_003",
            "filename": "/opt/openfbi/pylibs/File_worker/test_file/CRUD-postgresql.docx",
            "text": "",
            "ocr_text": "",
            "stamp_text": "",
            "children": []
        }
    ]
    datas = []
    for i in data:
        filename = i["filename"]
        res = process_file(filename)

        text = vs.flatten_result_text(res)
        #text = "这是一个用于测试文本向量化的示例文本。它包含多个句子，以确保向量化过程能够正确处理不同的文本结构。"
        vector = vs.embed_text_in_chunks(text)
        # i["text"] = res.get("text","")
        # i["ocr_text"] = res.get("ocr_text","")
        # i["stamp_text"] = res.get("stamp_text","")
        # i["children"] = res.get("children",[])
        finger = {
            "id": i["id"],
            "filename": filename,
            "combined_text": text,
            "vector": vector
        }
        datas.append(finger)
        #oc_id = vs.insert_document(finger)
       # print(f"Vector Length: {len(vector)}")
    print(datas)
    vs.connect_milvus()
    ids = vs.batch_insert_documents(datas)
    print(ids)
    # 用某条文本向量搜索相似文档
    query_text = "电子客票,12306,二等座"
    #res = process_file("/opt/openfbi/pylibs/File_worker/test_file/25339190041004130573-电子发票.pdf")

    #text = vs.flatten_result_text(res)
    query_vector = vs.embed_text_in_chunks(query_text)
    results = vs.search_documents(query_vector, top_k=5,score_threshold=0.5)
    print(results)
    for r in results:
        print(r)