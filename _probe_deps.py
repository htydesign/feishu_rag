import asyncio

from deal_pilot.config.minio_settings import minio_settings
from deal_pilot.config.nacos.nacos_config import get_config
from deal_pilot.config.nacos.nacos_schemas import RagConfig
from deal_pilot.utils.minio import MinIO


async def main():
    try:
        cfg = await get_config("rag_config", RagConfig)
        print("rag_config OK bucket=", cfg.image_parser.bucket_name,
              "sibling=", cfg.image_parser.sibling_count)
    except Exception as e:
        print("rag_config FAIL", type(e).__name__, e)
    try:
        m = MinIO(
            endpoint=minio_settings.endpoint,
            access_key=minio_settings.access_key,
            secret_key=minio_settings.secret_key,
            secure=minio_settings.secure,
        )
        exists = await m.bucket_exists("feishu-images")
        print("minio OK endpoint=", minio_settings.endpoint,
              " secure=", minio_settings.secure,
              " feishu-images exists=", exists)
    except Exception as e:
        print("minio FAIL", type(e).__name__, e)


asyncio.run(main())
