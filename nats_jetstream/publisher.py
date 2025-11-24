import asyncio
import json
from nats.aio.client import Client as NATS

async def publish_fileinfos(fileinfos:list):
    """
    将fileinfo日志批量发送到JetStream
    """
    nc = NATS()
    await nc.connect("nats://127.0.0.1:4222") # 连接nats server
    js = nc.jetstream()

    # 创建Stream
    try:
        await js.add_stream(name="FILES",subjects=["files.*"])
    except Exception:
        pass

    # 异步发送每条fileinfo
    tasks= []

    for fi in fileinfos:
        tasks.append(js.publish("files.upload",json.dumps(fi,ensure_ascii=False).encode()))
    
    await asyncio.gather(*tasks)
    await nc.drain()  # 优雅关闭连接


# 示例 fileinfo 日志（来自流计算层）
fileinfos = [
    {
        "timestamp": "2025-07-01T17:35:22.307510",
        "flow_id": 1047975525085093,
        "file_path": "/home/rzc/File_worker/test_file/1111.pdf",
        "filename": "1111.pdf",
        "md5": "56beeb226fbc8bc59d26670f2472a1f2",
        "sha256": "6664386531383633316632623765613236396132643739623935633337613532"
    },
    {
        "timestamp": "2025-07-01T17:36:10.123456",
        "flow_id": 1047975525085094,
        "file_path": "/home/rzc/File_worker/test_file/加密111.7z",
        "filename": "加密111.7z",
        "md5": "abcd1234abcd1234abcd1234abcd1234",
        "sha256": "efgh5678efgh5678efgh5678efgh5678"
    },
        {
        "timestamp": "2025-07-01T17:37:22.307510",
        "flow_id": 1047975525085095,
        "file_path": "/home/rzc/File_worker/test_file/加密111.pdf",
        "filename": "加密111.pdf",
        "md5": "56beeb226fbc8bc59d26670f2472a1f2",
        "sha256": "6664386531383633316632623765613236396132643739623935633337613532"
    },
    {
        "timestamp": "2025-07-01T17:38:10.123456",
        "flow_id": 1047975525085096,
        "file_path": "/home/rzc/File_worker/test_file/加密111.zip",
        "filename": "加密111.zip",
        "md5": "abcd1234abcd1234abcd1234abcd1234",
        "sha256": "efgh5678efgh5678efgh5678efgh5678"
    },
    {
        "timestamp": "2025-07-01T17:379:10.123456",
        "flow_id": 1047975525085097,
        "file_path": "/home/rzc/File_worker/test_file/CRUD-postgresql.docx",
        "filename": "CRUD-postgresql.docx",
        "md5": "abcd1234abcd1234abcd1234abcd1234",
        "sha256": "efgh5678efgh5678efgh5678efgh5678"
    }

]

asyncio.run(publish_fileinfos(fileinfos))