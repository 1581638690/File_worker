from nats.aio.client import Client as NATS
import asyncio

async def main():
    nc = NATS()
    await nc.connect("nats://127.0.0.1:4222")
    await nc.publish("foo",b"hello Natas")
    await nc.close()
asyncio.run(main())