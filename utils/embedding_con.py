import requests
from typing import List, Union, Optional, Dict, Any

class Llama_Cpp_AIService:
    def __init__(
        self,
        embedding_base_url: str = "http://localhost:8080",
        rerank_base_url: str = "http://127.0.0.1:8012",
        api_key: str = "no-key",
        timeout: int = 30
    ):
        """
        初始化本地 AI 服务客户端。
        
        :param embedding_base_url: Embedding 服务的基础 URL（如 http://localhost:8080）
        :param rerank_base_url: Rerank 服务的基础 URL（如 http://127.0.0.1:8012）
        :param api_key: 用于认证的 Bearer Token
        :param timeout: 请求超时时间（秒）
        """
        self.embedding_url = f"{embedding_base_url.rstrip('/')}/v1/embeddings"
        self.rerank_url = f"{rerank_base_url.rstrip('/')}/v1/rerank"
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embedding(
        self,
        input_text: Union[str, List[str]],
        model: str = "bge-m3",  #
        encoding_format: str = "float"
    ) -> Optional[Dict[str, Any]]:
        """
        获取文本的向量嵌入（Embedding）。
        
        :param input_text: 单个字符串或字符串列表
        :param model: 使用的 embedding 模型名称
        :param encoding_format: 编码格式，如 "float" 或 "base64"
        :return: API 返回的 JSON 响应（含 embeddings），失败时返回 None
        """
        payload = {
            "input": input_text,
            "model": model,
            "encoding_format": encoding_format
        }

        try:
            response = requests.post(
                self.embedding_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            """
            {"embedding":[向量]}
            """
            return response.json()["data"]
        except requests.RequestException as e:
            print(f"[Embedding Error] {e}")
            return None

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = "bge-m3-rerank",
        top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        对文档列表按与查询的相关性进行重排序（Rerank）。
        
        :param query: 查询文本
        :param documents: 待排序的文档列表
        :param model: 使用的 rerank 模型名称
        :param top_n: 返回前 top_n 个结果
        :return: API 返回的 JSON 响应（含排序结果），失败时返回 None
        """
        payload = {
            "model": model,
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        # 注意：某些 rerank 服务可能不需要 Authorization，这里保留以兼容
        # 如果你的 rerank 服务不需要 token，可单独处理 headers
        try:
            response = requests.post(
                self.rerank_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            """
            [{index:0,relevance_score:0.638}]
            """
            return response.json()["results"]
        except requests.RequestException as e:
            print(f"[Rerank Error] {e}")
            return None

if __name__=="__main__":

    # 初始化客户端
    ai = Llama_Cpp_AIService(
        embedding_base_url="http://192.168.124.78:8084",
        rerank_base_url="http://192.168.124.78:8085",
        api_key="no-key"
    )

    # 1. 获取 embedding
    emb_result = ai.embedding("hello", model="GPT-4")  # 注意：建议改用真实 embedding 模型
    if emb_result:
        print("Embedding:", emb_result[0]["embedding"])

    # 2. 执行 rerank
    documents=[
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ]
    rerank_result = ai.rerank(
        query="What is panda?",
        documents=documents,
        model="some-model",
        top_n=2
    )
    if rerank_result:
        for item in rerank_result:
            print(f"Doc {item['index']}: score={item['relevance_score']}, text={documents[item['index']]}")