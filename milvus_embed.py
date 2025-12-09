import os
import time
import json
import numpy as np
from typing import List, Optional, Dict
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import sys
sys.path.append("/opt/openfbi/pylibs/")
from File_worker.utils.embedding_con import *
from File_worker.core import *

import logging
logger = logging.getLogger("VerctorSerive")
logger.setLevel(logging.INFO)
# ----------------- 日志 -----------------
logger = logging.getLogger("VectorService")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

# 控制台输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 文件输出
file_handler = logging.FileHandler("vector_service.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ------------------ Monkey patch 兼容 Linux ------------------
if not hasattr(os, "add_dll_directory"):
    os.add_dll_directory = lambda path: None
class VectorSerive:
    def __init__(self,
                 milvus_host="192.168.124.250",
                 milvus_port="19530",
                 embedding_service_url="http://192.168.124.78:8084",
                 api_key="no-key",
                 collection_name="documents_xlink",
                 embedding_dim=1024):

        """
        初始化 Milvus + Embedding 服务
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        if "default" in connections.list_connections():
            connections.disconnect("default")
        # 连接 Milvus
        connections.connect(alias="default", host=milvus_host, port=milvus_port)

        # 初始化 embedding 服务
        self.ai_client = Llama_Cpp_AIService(
            embedding_base_url=embedding_service_url,
            api_key=api_key
        )

        # 如果不存在集合则创建
        if collection_name not in utility.list_collections():
            self._create_collection()
        logger.info(f"Milvus collection '{collection_name}' is ready.")
        # 加载集合
        self.collection = Collection(self.collection_name)
        self.collection.load()

    # ------------------- 支持动态更新 embedding 服务 URL -------------------
    def update_embedding_url(self, new_url: str):
        self.ai_client.base_url = new_url

    def test_connection(self):
        pass
    # ------------------- 创建 Collection -------------------

    def _create_collection(self):
        fields = [
            FieldSchema(name="file_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=60000),
            FieldSchema(name="category_id", dtype=DataType.INT64),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]

        schema = CollectionSchema(fields, description="文档向量存储")
        collection = Collection(self.collection_name, schema)

        # 创建索引（HNSW）
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )

        collection.load()

    # ------------------- 文本处理 -------------------

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        s = text.replace("\r", "\n")
        s = " ".join(s.split())
        return s.strip()

    def flatten_result_text(self, result: Dict) -> str:
        """递归提取 text / ocr_text / stamp_text + children"""

        parts = []
        if result.get("text"):
            parts.append(result["text"])
        if result.get("ocr_text"):
            parts.append(result["ocr_text"])
        if result.get("stamp_text"):
            parts.append(result["stamp_text"])

        children = result.get("children", [])
        if isinstance(children, str):
            children = json.loads(children)

        for ch in children:
            parts.append(self.flatten_result_text(ch))

        combined = "\n".join([p for p in parts if p])
        return self.normalize_text(combined)

    # ------------------- embedding 生成 -------------------

    def text_to_vector(self, text: str, model: str = "GPT-4") -> List[float]:
        if not text:
            return []

        try:
            resp = self.ai_client.embedding(text, model=model)
            if not resp:
                return []
            return resp[0]["embedding"]
        except Exception as e:
            print(f"[Embedding Error] {e}")
            return []

    def embed_text_in_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[float]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        start, n = 0, len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            start += chunk_size - overlap

        vectors = []
        for ch in chunks:
            vec = self.text_to_vector(ch)
            if vec:
                vectors.append(vec)

        if not vectors:
            return []

        arr = np.array(vectors, dtype=np.float32)
        return np.mean(arr, axis=0).tolist()

    # ------------------- 插入 Milvus -------------------

    def insert_document(self, finger: dict) -> int:
        """
        finger: {
            "id": "...",
            "filename": "...",
            "vector": [...],
            "combined_text": "...",
        }
        """
        combined_text = finger.get("combined_text", "")
        vector = finger.get("vector", [])

        if not vector:
            vector = self.embed_text_in_chunks(combined_text)

        # 强制 float32
        vector = np.array(vector, dtype=np.float32).tolist()

        id = finger.get("id", "")

        # Milvus insert 格式：List[List]s
        data = [
            {"id":finger.get("id", ""),"file":finger.get("filename", ""),"text":combined_text,"category_id":-1,"tags":"","embedding":vector}                                    # file_id
        ]

        self.collection.insert(data)
        self.collection.flush()

        return id

    # ------------------- 向量检索 -------------------

    def search_similar(self, vector, top_k=10) -> List[Dict]:

        vector = np.array(vector, dtype=np.float32).tolist()

        search_params = {
            "metric_type": "COSINE", # 余弦相似度 L2 IP JACCARD HAMMING
            "params": {"nprobe": 10} 
        }

        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["file_id", "id", "file", "text", "tags", "category_id"]
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "file_id": hit.entity.get("file_id"),
                "id": hit.entity.get("id"),
                "file": hit.entity.get("file"),
                "text": hit.entity.get("text"),
                "tags": hit.entity.get("tags"),
                "category_id": hit.entity.get("category_id"),
                "score": float(hit.distance),
            })

        return hits

    def search_documents(
        self,
        vector,
        top_k=5,
        metric_type="COSINE",
        score_threshold=None,
        filter_category_id=None,
        extra_filters=None
    ):
        # -------------------------------
        # 构造 Milvus 搜索参数
        # -------------------------------
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}
        }

        # -------------------------------
        # 构造过滤条件
        # -------------------------------
        expr_parts = []

        if filter_category_id is not None:
            expr_parts.append(f"category_id == {filter_category_id}")

        if extra_filters:
            for k, v in extra_filters.items():
                if isinstance(v, str):
                    expr_parts.append(f'{k} == "{v}"')
                else:
                    expr_parts.append(f'{k} == {v}')

        expr = " and ".join(expr_parts) if expr_parts else None

        # -------------------------------
        # 执行向量检索
        # -------------------------------
        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["file_id", "id", "file", "text", "tags", "category_id"]
        )

        # -------------------------------
        # 基于阈值过滤并统一返回格式
        # -------------------------------
        filtered_results = []
        for hit in results[0]:
            score = float(hit.distance)
            # 阈值判断，根据 metric 类型
            if score_threshold is not None:
                if metric_type in ["COSINE", "IP"]:
                    if score < score_threshold:
                        continue
                elif metric_type in ["L2", "JACCARD", "HAMMING"]:
                    if score > score_threshold:
                        continue

            filtered_results.append({
                "file_id": hit.entity.get("file_id"),
                "id": hit.entity.get("id"),
                "file_name": hit.entity.get("file"),
                "tags": hit.entity.get("tags"),
                "category_id": hit.entity.get("category_id"),
                "score": score
            })

        # -------------------------------
        # 如果需要聚类逻辑，可在这里根据 cluster_threshold 进行二次过滤/聚合
        # cluster_top_k 可用于限制返回的聚类数量
        # -------------------------------
        # 示例：简单按 score 排序并截取 cluster_top_k
        # if cluster_top_k and filtered_results:
        #     filtered_results.sort(key=lambda x: x["score"], reverse=(metric_type in ["COSINE", "IP"]))
        #     filtered_results = filtered_results[:cluster_top_k]

        return filtered_results
    # ------------------- 集合删除 -------------------
    def drop_collection(self):
        if self.collection_name in utility.list_collections():
            utility.drop_collection(self.collection_name)
            print(f"Collection {self.collection_name} dropped.")
    
    # ------------------- 批量插入 -------------------
    def batch_insert_documents(self,fingers:List[Dict])->List[str]:
        """
        批量插入文档数据信息
        """
        #logger.info(f"Batch inserting {len(fingers)} documents into Milvus.")
        #logger.info(f"数据全部，{fingers}")
        if not fingers:
            return []
        ids = []
        data_list = []
        for finger in fingers:
            combined_text = finger.get("combined_text","")
            vector = finger.get("vector",[])

            #if not vector and combined_text:
            #    vector = self.embed_text_in_chunks(combined_text)
            
            if not vector:
                continue
            
            vector = np.array(vector,dtype=np.float32).tolist()
            data_list.append({
            "id": finger.get("id", ""),
            "file": finger.get("filename", ""),
            "text": combined_text,
            "category_id": finger.get("category_id", -1),
            "tags": finger.get("tags", ""),
            "embedding": vector
            })
            ids.append(finger.get("id", ""))
        #logger.info(f"Prepared {len(data_list)} documents for insertion.")
        if data_list:
            self.collection.insert(data_list)
            self.collection.flush()
        return ids
"""
| Metric  | 距离大小  | 相似度关系     | 忽略最相似向量条件                          |
| ------- | ----- | --------- | ---------------------------------- |
| L2      | 越小越相似 | 距离小 = 高相似 | range_filter <= distance < radius  |
| IP      | 越大越相似 | 距离大 = 高相似 | radius < distance <= range_filter  |
| COSINE  | 越大越相似 | 距离大 = 高相似 | radius < distance <= range_filter  |
| JACCARD | 越小越相似 | 距离小 = 高相似 | range_filter <= distance <= radius |
| HAMMING | 越小越相似 | 距离小 = 高相似 | range_filter <= distance < radius  |


"""
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