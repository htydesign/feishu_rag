"""MinIO.upload_bytes 的单元测试；替换底层 aiobotocore 客户端，不连真实服务。

只验证 upload_bytes 的参数校验与 put_object 的元数据约定：Content-Type 使用传入
mime，Cache-Control 仍由构造器的 cache_strategy 决定。真实 S3 验收见
tests/intergrations/test_minio.py。
"""

import asyncio
from io import BytesIO

import pytest

from deal_pilot.utils.minio import CacheStrategy, MinIO


class FakeAioClient:
    """记录 put_object 参数的假 aiobotocore 异步客户端。"""

    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def put_object(self, **kwargs):
        self.log.append(kwargs)


def _client(monkeypatch, *, cache_strategy=None):
    log = []
    client = MinIO(
        endpoint="localhost:9000",
        access_key="access",
        secret_key="secret",
        secure=False,
        cache_strategy=cache_strategy,
    )
    monkeypatch.setattr(client, "_client", lambda: FakeAioClient(log))
    return client, log


def test_upload_bytes_uses_mime_content_type_and_no_store_default(monkeypatch):
    client, log = _client(monkeypatch)

    asyncio.run(client.upload_bytes(
        bucket_name="bucket", filename="img.png", data=b"\x89PNG", mime="image/png",
    ))

    assert log == [{
        "Bucket": "bucket", "Key": "img.png", "Body": b"\x89PNG",
        "ContentType": "image/png", "CacheControl": "no-store",
    }]


def test_upload_bytes_applies_cache_strategy_by_extension(monkeypatch):
    client, log = _client(monkeypatch, cache_strategy=[CacheStrategy([".png"], max_age=120)])

    asyncio.run(client.upload_bytes(
        bucket_name="bucket", filename="img.png", data=b"\x89PNG", mime="image/png",
    ))

    assert log[0]["CacheControl"] == "max-age=120"
    assert log[0]["ContentType"] == "image/png"


def test_upload_bytes_invalid_data_raises_without_put(monkeypatch):
    client, log = _client(monkeypatch)

    with pytest.raises(ValueError, match="data"):
        asyncio.run(client.upload_bytes(
            bucket_name="bucket", filename="img.png", data=BytesIO(b"x"), mime="image/png",
        ))
    assert log == []


def test_upload_bytes_invalid_mime_raises_without_put(monkeypatch):
    client, log = _client(monkeypatch)

    with pytest.raises(ValueError, match="Content-Type"):
        asyncio.run(client.upload_bytes(
            bucket_name="bucket", filename="img.png", data=b"\x89PNG", mime="image/png\nbad",
        ))
    assert log == []


def test_upload_bytes_invalid_bucket_or_key_raises_without_put(monkeypatch):
    client, log = _client(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(client.upload_bytes(
            bucket_name="bad bucket", filename="img.png", data=b"x", mime="image/png",
        ))
    with pytest.raises(ValueError):
        asyncio.run(client.upload_bytes(
            bucket_name="bucket", filename="", data=b"x", mime="image/png",
        ))
    assert log == []
