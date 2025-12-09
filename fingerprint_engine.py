"""
针对文件进行指纹信息的获取
"""
import hashlib
import json
import os
from typing import Dict,List,Tuple,Optional
import numpy as np
from simhash import Simhash # 用于文本的Simhash快速近似文本匹配

try:
    import tlsh 
    TLSH_AVAILABLE=True
except Exception:
    TLSH_AVAILABLE=False


# 用于向量化

import sys 
sys.path.append("/opt/openfbi/pylibs")
from File_worker.core import *
from File_worker.utils.embedding_con import *

AI_CLIENT = Llama_Cpp_AIService(
    embedding_base_url= "http://192.168.124.78:8084",
    api_key="no-key"

)



######## 文本标准化与分段 #############

def normalize_text(text:str) -> str:
    if not text:
        return ""
    
    s = text.replace("\r","\n")
    s = " ".join(s.split())
    return s.strip()


def flatten_result_text(result:Dict) -> str:
    """
    从获取文本信息提取合并所有的文本来源
    result["text"] 文本信息
    每个 ocr_detail
    递归children
    返回合并后的长文本
    """

    parts = []
    if result.get("text"):
        parts.append(result["text"])
    
    # 添加当前文件下所有的图片ocr识别文本
    ocr_text = result.get("ocr_text","")
    parts.append(ocr_text)
    # 印章文本
    stamp_text = result.get("stamp_text","")
    parts.append(stamp_text)

    # 递归children
    children_msg = result.get("children",[])
    if isinstance(children_msg,str):
        children_msg = json.loads(children_msg)
    for ch in children_msg:
        parts.append(flatten_result_text(ch))
    
    combined = "\n".join([p for p in parts if p])

    return normalize_text(combined)


def segment_text(text:str,seg_size:int=1024) -> List[str]:
    """
    将文本切成若干断用户段指纹/段匹配
    """

    if not text:
        return []
    
    text = text.strip()

    segments = [text[i:i+seg_size] for i in range(0,len(text),seg_size)]
    return segments


######## 指纹生成函数 #########
def sha256_file(file_path:str) -> str:
    """针对文件二进制的精确指纹"""
    # fileinfo日志中存在sha256指纹

    h = hashlib.sha256()
    with open(file_path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def text_simhash(text:str) -> int:
    """对文本计算simhash"""
    s = normalize_text(text)
    if not s:
        return 0
    
    return Simhash(s).value # 返回整数，通常是64位或者128位


def file_tlsh(file_path:str) ->Optional[str]:
    if not TLSH_AVAILABLE:
        return ""
    
    with open(file_path,"rb")as f:
        data = f.read()
    
    try:
        return tlsh.hash(data)
    except Exception:
        return ""


########## 使用 llama.cpp embedding文本向量化 ############
def text_to_vector(text:str,model:str = 'GPT-4') -> List[float]:
    """
    调用llama.cppHTTP服务生成文本 embedding
    """
    if not text:
        return []
    try:
        resp = AI_CLIENT.embedding(text,model=model)
        if not resp:
            return []
        return resp[0]["embedding"]
    except Exception as e:
        return []

def embed_text_in_chunks(text:str,chunk_size:int=512,overlap:int = 50) -> List[float]:
    """
    将文本切分后逐段生成向量，然后平均池化得到整个文档的语义向量。

    :param text: 输入文本（可能很长）
    :param chunk_size: 每段的字符数（安全值：512~800）
    :param overlap: 段之间的重叠字符数（用于提高语义连续性）
    :return: 文档整体向量（平均池化）
    """
    text = text.strip()
    if not text:
        return []
    
    # 生成chunks
    chunks = []
    start = 0 
    n = len(text)

    while start <n:
        end = min(start + chunk_size,n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap) # 滑动窗口
    
    vectors = []
    for ch in chunks:
        vec = text_to_vector(ch)
        if vec:
            vectors.append(vec)
    if not vectors:
        return []
    
    # --- 平均池化 (mean pooling) ---
    arr = np.array(vectors, dtype=np.float32)
    pooled = np.mean(arr, axis=0)

    return pooled.tolist()



# -------------------------
# Segment fingerprints（用作部分匹配）
# -------------------------
def segment_hashes(text: str, seg_size: int = 1024) -> List[str]: # 将文本进行分段 然后对每段进行md5(hex)
    segs = segment_text(text, seg_size)
    return [hashlib.md5(seg.encode('utf-8')).hexdigest() for seg in segs]


# -------------------------
# 匹配逻辑 相似度函数
# -------------------------
def hamming_distance_int(a: int, b: int) -> int: # 比较两个simbash值
    return bin(a ^ b).count("1")


def cosine_similarity(v1: List[float], v2: List[float]) -> float: # 计算余弦相似度
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)



# -------------------------
# 主函数：从 result 生成指纹包
# -------------------------
def generate_fingerprints_from_result(result: Dict, compute_file_sha: bool = True) -> Dict:
    """
    输入：根据已有的 result dict（包含 filepath, text, ocr_detail, children）
    输出：指纹字典包含：
      - sha256 (如果 compute_file_sha True 且 filepath 存在)
      - simhash (整文)
      - tlsh (如果可用)
      - vector (整文向量)
      - segment_hashes (列表)
    """
    if isinstance(result,str):
        result = json.loads(result)
    out = {}

    # 合并文本信息
    combined_text = flatten_result_text(result)
    out["combined_text"] = combined_text

    # 文件二进制sha256
    file_path = result.get("filepath")
    

    # 尽量先使用流量包解析出来的数据信息
    if result.get("sha256",""):
        out["sha256"] = result.get("sha256","")
    elif compute_file_sha and file_path and os.path.exists(file_path):
        try:
            out["sha256"] = sha256_file(file_path)
        except Exception:
            out["sha256"] = ""
    

    # simhash
    out["simhash"] = text_simhash(combined_text)

    # tlsh
    if file_path and os.path.exists(file_path):
        out["tlsh"] = file_tlsh(file_path) if TLSH_AVAILABLE else ""
    else:
        out["tlsh"] = ""
    
    # segments
    out["segment_hashes"] = segment_hashes(combined_text, seg_size=1024)

    #txt_for_vec = combined_text if len(combined_text) <= 4000 else combined_text[:4000]
    #out["vector"] = text_to_vector(txt_for_vec)
    # 实现向量化
    out["vector"] = embed_text_in_chunks(combined_text, chunk_size=512, overlap=50)
    print(len(out["vector"]))
    # 带上基本元信息
    out["file_id"] = result.get("filepath")  # 或者你系统的 fileinfo id
    out["filename"] = result.get("filename")
    out["filetype"] = result.get("filetype")

    return out

# 用作udf匹配指纹信息
def match_fingerprint_with_docs(mysql_df, ckh_df, file_fp, thresholds=None):
    """
    根据规则匹配文件指纹，返回匹配规则及相关文档信息。

    :param mysql_df: MySQL规则表 DataFrame
    :param ckh_df: ClickHouse 文件指纹表 DataFrame
    :param file_fp: 当前上传文件指纹 dict
    :param thresholds: 可选默认阈值覆盖规则表
    :return: List[Dict], 每个元素包含:
             {"rule_name":..., "match_type":..., "score":..., "related_file_id":..., "filename":...}
    """
    if thresholds is None:
        thresholds = {
            "simhash_hamming": 5,
            "vector_cos": 0.82,
            "segment_overlap": 0.4
        }

    matches = []

    for _, rule in mysql_df.iterrows():
        if not rule.get("enabled", 1):
            continue

        r_name = rule.get("rule_name", "Unnamed Rule")
        r_type = rule.get("match_type", "")
        threshold = rule.get("threshold", None)

        # 若规则没有 threshold，就用默认阈值
        if threshold is None:
            if r_type == "SIMHASH":
                threshold = thresholds["simhash_hamming"]
            elif r_type == "VECTOR":
                threshold = thresholds["vector_cos"]
            elif r_type == "SEGMENT":
                threshold = thresholds["segment_overlap"]

        # SHA256 精确匹配
        if r_type.upper() == "SHA256":
            for _, rec in ckh_df.iterrows():
                if file_fp.get("sha256") and rec.get("sha256") and file_fp["sha256"] == rec["sha256"]:
                    matches.append({
                        "rule_name": r_name,
                        "match_type": "SHA256",
                        "score": 1.0,
                        "related_file_id": rec.get("id"),
                        "filename": rec.get("filename")
                    })
                    break

        # SIMHASH 近似匹配
        elif r_type.upper() == "SIMHASH":
            for _, rec in ckh_df.iterrows():
                sim = int(rec.get("simhashs")) # 库中是字符串
                if sim is not None:
                    dist = hamming_distance_int(file_fp.get("simhash", 0), sim)
                    if dist <= threshold:
                        score = max(0.0, 1.0 - dist / max(1, threshold))
                        matches.append({
                            "rule_name": r_name,
                            "match_type": "SIMHASH",
                            "score": score,
                            "related_file_id": rec.get("id"),
                            "filename": rec.get("filename")
                        })
                        break

        # SEGMENT 分段匹配
        elif r_type.upper() == "SEGMENT":
            file_segs = set(file_fp.get("segment_hashes", []))
            for _, rec in ckh_df.iterrows():
                segment = rec.get("segment_hashes", [])
                if isinstance(segment, str):
                    try:
                        segment = json.loads(segment)
                    except:
                        segment = []
                set_b = set(segment)
                if set_b:
                    overlap_ratio = len(file_segs & set_b) / max(len(file_segs | set_b), 1)
                    if overlap_ratio >= threshold:
                        matches.append({
                            "rule_name": r_name,
                            "match_type": "SEGMENT",
                            "score": overlap_ratio,
                            "related_file_id": rec.get("file_id"),
                            "filename": rec.get("filename")
                        })
                        break

        # VECTOR 语义向量匹配
        elif r_type.upper() == "VECTOR":
            file_vec = file_fp.get("vector", [])
            if not file_vec:
                continue
            for _, rec in ckh_df.iterrows():
                vec = rec.get("vector", [])
                if isinstance(vec, str):
                    try:
                        vec = json.loads(vec)
                    except:
                        vec = []
                if not vec:
                    continue
                cos = cosine_similarity(file_vec, vec)
                if cos >= threshold:
                    matches.append({
                        "rule_name": r_name,
                        "match_type": "VECTOR",
                        "score": cos,
                        "related_file_id": rec.get("id"),
                        "filename": rec.get("filename")
                    })
                    break

    return matches

def test_fingerprint_pipeline(file1, file2, workdir="/tmp/fp_test"):
    """
    文件指纹 + 相似度测试主函数
    :param file1: 文件路径1
    :param file2: 文件路径2
    :param workdir: 临时工作目录
    """

    os.makedirs(workdir, exist_ok=True)

    print("======= 文件指纹系统测试开始 =======")

    # 1. 提取文件内容
    


    print("[1] 提取文件文本:", file1)
    result1 = process_file(file1)

    print("[2] 提取文件文本:", file2)
    result2 = process_file(file2)

    if not result1 or not result2:
        print("❌ 提取内容失败，请检查文件解析模块")
        return

    # 2. 生成文件指纹
    

    print("[3] 生成文件指纹…")
    fp1 = generate_fingerprints_from_result(result1)
    fp2 = generate_fingerprints_from_result(result2)

    # 3. 指纹详细展示
    print("\n-------- 文件1指纹结果 --------")
    for k, v in fp1.items():
        print(f"{k}: {str(v)[:100]}")

    print("\n-------- 文件2指纹结果 --------")
    for k, v in fp2.items():
        print(f"{k}: {str(v)[:100]}")

    # 4. 相似度比对
    

    print("\n[4] 相似度计算…")

    simhash_distance = hamming_distance_int(fp1["simhash"], fp2["simhash"])
    vector_similarity = cosine_similarity(fp1["vector"], fp2["vector"])

    print("\n======= 相似度结果 =======")
    
    print("SimHash 海明距离:", simhash_distance)
    print("向量余弦相似度:", round(vector_similarity, 4))

    # 5. 简单判断
    if simhash_distance <= 5:
        print("✅ 文本高度相似!")
    elif simhash_distance <= 10:
        print("⚠️ 文本中度相似")
    else:
        print("❌ 文本差异较大")

    if vector_similarity >= 0.85:
        print("✅ 语义高度相似")
    elif vector_similarity >= 0.6:
        print("⚠️ 语义中度相似")
    else:
        print("❌ 语义相似度低")
    vectors = []
    vectors.append(fp1["vector"])
    vectors.append(fp2["vector"])
    file_ids = []
    cate = cluster_vectors(file_ids, vectors)
    print("类别信息：",cate)
    print("\n======= 测试结束 =======")

"""
数据泄露检测（DLP）
机密文件溯源
红头文件复用检测
邮件附件内容识别
OCR 图片转文字内容匹配
网盘/文件服务器文件查重
垃圾文件去重与归档
"""


######################### 语义向量分类 ###################
def cluster_vectors(file_ids, vectors):
    n = len(vectors)
    categories = [-1] * n
    next_cat = 1
    from tqdm import tqdm
    for i in tqdm(range(n), desc="分类中…"):
        if categories[i] != -1:
            continue

        categories[i] = next_cat

        for j in range(i + 1, n):
            if categories[j] != -1:
                continue
            #json.loads(vectors[i])
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= 0.70:
                categories[j] = next_cat

        next_cat += 1

    return categories


#from collections import defaultdict,deque
#from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    import datetime
    #test_fingerprint_pipeline("/opt/openfbi/pylibs/File_worker/test_file/25339190041004130573-电子发票.pdf","/opt/openfbi/pylibs/File_worker/test_file/25359134682000718615-电子发票.pdf")
    #o = {'filename': '6533623063343432393866633163313439616662663463383939366662393234', 'filepath': '/data/files/65/6533623063343432393866633163313439616662663463383939366662393234', 'filetype': 'jpg', 'sha256': '6533623063343432393866633163313439616662663463383939366662393234', 'text': '', 'ocr_text': '', 'ocr_detail': '[]', 'images': '["\\/data\\/files\\/65\\/6533623063343432393866633163313439616662663463383939366662393234"]', 'red_header': '否', 'stamp_detected': '否', 'stamp_detail': '[]', 'stamp_text': '', 'children': '[]', 'req_header': False, 'size': 1022, 'state': 'TRUNCATED', 'srcip': '72.233.69.5', 'app_proto': 'http', 'srcport': 80, 'dstip': '192.168.4.120', 'dstport': 4642, 'url': 'http://72.233.69.5:80/libhtp::request_uri_not_seen', 'parameter': '', 'id': '1764667249251387558', 'encrypted': '否', 'timestamp': datetime.datetime(2025, 12, 2, 17, 20, 49, 237538), 'msg': ''}
    #generate_fingerprints_from_result(o)
    file_path = [
		"test_file/1111.pdf",
		"test_file/1111.zip",
		"test_file/533856970.jpg",
		"test_file/加密111_2.7z",
		"test_file/加密111.7z",
		"test_file/加密111.pdf",
		"test_file/25339190041004130573-电子发票.pdf",
		"test_file/25359134682000718615-电子发票.pdf",
		"test_file/CRUD-postgresql.docx",
		"test_file/webwxgetmsgimg.jpeg"

	]
    #from .core import process_file
    dic = {"vector":[],"filename":[]}
    
    for i in file_path:
        res = process_file(f"/opt/openfbi/pylibs/File_worker/{i}")
        out = generate_fingerprints_from_result(res)
        dic["vector"] = out.get("vector")
        dic["filename"] = out.get("filename")
    
    # 然后 循环 dic 进行聚类分析
    print(len(dic))
    cate = cluster_vectors(dic["filename"],dic["vector"])
    print(cate)