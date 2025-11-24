import asyncio
import json
import time
from nats.aio.client import Client as NATS
from core import run  # 你的完整文件处理逻辑

# -------------------- 配置 --------------------
MAX_CONCURRENCY = 5         # 同时处理文件数
BATCH_SIZE = 10             # 每次拉取消息批量处理
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
# ---------------------------------------------

async def process_file_async(fileinfo):
    """
    异步处理单个文件
    """
    filepath = fileinfo["file_path"]
    start_time = time.time()
    async with SEMAPHORE:
        print(f"[Worker] Start processing {filepath}")
        try:
            result_json = run(filepath)  # 调用 core.py
            # TODO: 写入数据库或存储结果
        except Exception as e:
            print(f"[Worker] Error processing {filepath}: {e}")
        end_time = time.time()
        print(f"[Worker] Finished processing {filepath} in {end_time - start_time:.2f}s")

async def handle_messages(msgs):
    """
    批量处理消息
    """
    tasks = []
    for msg in msgs:
        try:
            fileinfo = json.loads(msg.data.decode())
            tasks.append(asyncio.create_task(process_file_async(fileinfo)))
            await msg.ack()  # ack 消息
        except Exception as e:
            print(f"[Consumer] Error decoding message: {e}")
    if tasks:
        await asyncio.gather(*tasks)

async def consumer_loop(js):
    """
    持续拉取消息并批量处理
    """
    # durable name 保证断线重连后消息不会丢
    sub = await js.subscribe("files.*", durable="file_worker", max_inflight=MAX_CONCURRENCY)

    buffer = []
    while True:
        try:
            msg = await sub.next_msg(timeout=1)
            buffer.append(msg)
            if len(buffer) >= BATCH_SIZE:
                await handle_messages(buffer)
                buffer.clear()
        except asyncio.TimeoutError:
            if buffer:
                await handle_messages(buffer)
                buffer.clear()
        except Exception as e:
            print(f"[Consumer] Error in consumer loop: {e}")

async def main():
    nc = NATS()
    await nc.connect("nats://127.0.0.1:4222")
    js = nc.jetstream()

    # 创建 Stream（如果不存在）
    try:
        await js.add_stream(name="FILES", subjects=["files.*"])
    except Exception:
        pass

    print("[Consumer] Optimized consumer started, waiting for messages...")
    try:
        await consumer_loop(js)
    except KeyboardInterrupt:
        await nc.drain()
        print("[Consumer] Shutdown gracefully.")

if __name__ == "__main__":
    asyncio.run(main())
