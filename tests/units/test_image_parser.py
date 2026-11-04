"""image_parser 的单元测试；Mock Nacos 配置与 MinIO，不依赖真实服务。

沿用工程内 asyncio.run 包裹的写法（无 pytest-asyncio）。Nacos 配置、飞书
图片下载与 MinIO 均替换为假对象，验证纯逻辑与编排行为，不代表真实服务验证。
"""

import asyncio
import base64

import pytest

from deal_pilot.config.nacos.nacos_schemas import RagConfig, RagImageParser
from deal_pilot.rag.structure.feishu_doc import image_parser
from deal_pilot.rag.structure.feishu_doc.block_node import BlockNode
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType


def make_tree(*, sibling_count=2):
    """构造含目标图片节点及其父链、前后兄弟的树。"""
    page = BlockNode(block_id="page", block_type=BlockType.PAGE, content="")
    h1 = BlockNode(block_id="h1", block_type=BlockType.HEADING_1, content="章节")
    page.append(h1)
    t1 = BlockNode(block_id="t1", block_type=BlockType.TEXT, content="前文本")
    i1 = BlockNode(block_id="i1", block_type=BlockType.IMAGE, content="IMG-1")
    img = BlockNode(block_id="img2", block_type=BlockType.IMAGE, content="IMG-2")
    t2 = BlockNode(block_id="t2", block_type=BlockType.TEXT, content="后文本")
    i2 = BlockNode(block_id="i3", block_type=BlockType.IMAGE, content="IMG-3")
    for child in (t1, i1, img, t2, i2):
        h1.append(child)
    return img, RagConfig(image_parser=RagImageParser(sibling_count=sibling_count))


def _patch_config(monkeypatch, config):
    async def fake_get_config(_name, _schema):
        return config

    monkeypatch.setattr(image_parser, "get_config", fake_get_config)


class FakeMinIO:
    """记录 create_bucket / upload_bytes 调用的假 MinIO 客户端。"""

    def __init__(self):
        self.created = []
        self.uploads = []

    async def create_bucket(self, bucket_name, allow_public_read=True):
        self.created.append((bucket_name, allow_public_read))

    async def upload_bytes(self, **kwargs):
        self.uploads.append(kwargs)


def _patch_minio(monkeypatch, fake):
    monkeypatch.setattr(image_parser, "MinIO", lambda **kwargs: fake)


# ---------- _generate_dataurl ----------


def test_generate_dataurl_encodes_mime_and_base64():
    data = b"\x89PNG\r\n\x1a\n"
    url = image_parser._generate_dataurl(data, "image/png")
    expected = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
    assert url == expected


# ---------- _attachment_info ----------


def test_attachment_info_parent_trace_and_siblings(monkeypatch):
    img, config = make_tree(sibling_count=2)
    _patch_config(monkeypatch, config)

    info = asyncio.run(image_parser._attachment_info(img))

    # 父链：根到直接父，去除空内容，不含图片自身。
    assert info.parent_trace == ["章节"]
    # 前 n 个兄弟，不跳过图片，按文档顺序。
    assert info.prevs == ["前文本", "IMG-1"]
    # 后 n 个非图片兄弟，跳过图片并补足，按文档顺序；不足则返回实际。
    assert info.nexts == ["后文本"]


def test_attachment_info_skips_images_and_pads_nexts(monkeypatch):
    img, config = make_tree(sibling_count=3)
    _patch_config(monkeypatch, config)

    info = asyncio.run(image_parser._attachment_info(img))

    assert info.prevs == ["前文本", "IMG-1"]  # 只有 2 个前兄弟，不足 3。
    # 跳过 IMG-3 后无可用的非图片节点，返回实际找到的 1 个。
    assert info.nexts == ["后文本"]


def test_attachment_info_empty_parent_content_filtered(monkeypatch):
    page = BlockNode(block_id="page", block_type=BlockType.PAGE, content="")
    inner = BlockNode(block_id="inner", block_type=BlockType.TEXT, content="")
    page.append(inner)
    img = BlockNode(block_id="img", block_type=BlockType.IMAGE, content="IMG")
    inner.append(img)
    _patch_config(monkeypatch, RagConfig(image_parser=RagImageParser(sibling_count=1)))

    info = asyncio.run(image_parser._attachment_info(img))

    assert info.parent_trace == []
    assert info.prevs == []
    assert info.nexts == []


# ---------- _upload_minio ----------


def test_upload_minio_uses_configured_bucket_and_filename(monkeypatch):
    fake = FakeMinIO()
    _patch_config(
        monkeypatch, RagConfig(image_parser=RagImageParser(bucket_name="feishu-images"))
    )
    _patch_minio(monkeypatch, fake)

    location = asyncio.run(image_parser._upload_minio(b"\x89PNG", "image/png", "blk1"))

    assert location.bucket_name == "feishu-images"
    assert location.filename == "blk1.png"
    assert fake.created == [("feishu-images", False)]  # 桶禁用公共读。
    assert fake.uploads == [{
        "bucket_name": "feishu-images",
        "filename": "blk1.png",
        "data": b"\x89PNG",
        "mime": "image/png",
    }]


def test_upload_minio_unknown_mime_raises(monkeypatch):
    fake = FakeMinIO()
    _patch_config(monkeypatch, RagConfig())
    _patch_minio(monkeypatch, fake)

    with pytest.raises(ValueError, match="不支持的图片 MIME"):
        asyncio.run(image_parser._upload_minio(b"\x00", "image/x-unknown", "blk1"))
    assert fake.created == []
    assert fake.uploads == []


# ---------- parse_image_block ----------


def test_parse_image_block_download_upload_and_write_back(monkeypatch):
    img, config = make_tree(sibling_count=2)
    fake = FakeMinIO()
    _patch_config(monkeypatch, config)
    _patch_minio(monkeypatch, fake)

    async def fake_download(token):
        assert token == "IMG-2"
        from deal_pilot.feishu.api.image import DownloadedImage
        return DownloadedImage(mime="image/png", data=b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(image_parser, "download_image", fake_download)

    asyncio.run(image_parser.parse_image_block(img))

    expected_url = (
        f"data:image/png;base64,{base64.b64encode(b'\x89PNG\r\n\x1a\n').decode('ascii')}"
    )
    assert img.content == expected_url
    assert img.payload["token"] == "IMG-2"
    assert img.payload["mime"] == "image/png"
    assert img.payload["dataurl"] == expected_url
    assert img.payload["location"] == {
        "bucket_name": "feishu-images", "filename": "img2.png",
    }
    assert img.payload["context"]["parent_trace"] == ["章节"]
    assert img.payload["context"]["prevs"] == ["前文本", "IMG-1"]
    assert img.payload["context"]["nexts"] == ["后文本"]
    assert fake.created == [("feishu-images", False)]
    assert fake.uploads[0]["filename"] == "img2.png"


# ---------- build_tree 串行处理 ----------


def test_build_tree_processes_images_serially_in_document_order(monkeypatch):
    """图片按文档顺序逐个串行处理：前一张完成后再处理下一张。"""
    from deal_pilot.feishu.api.models import DocxBlock
    from deal_pilot.rag.structure.feishu_doc.tree_builder import build_tree

    blocks = [
        DocxBlock(
            block_id="page", block_type=BlockType.PAGE, parent_id=None,
            children=["img_a", "t", "img_b"],
            raw={"block_id": "page", "block_type": BlockType.PAGE,
                 "children": ["img_a", "t", "img_b"]},
        ),
        DocxBlock(
            block_id="img_a", block_type=BlockType.IMAGE, parent_id="page",
            children=[], raw={"block_id": "img_a", "block_type": BlockType.IMAGE,
                              "image": {"token": "tok_a"}},
        ),
        DocxBlock(
            block_id="t", block_type=BlockType.TEXT, parent_id="page", children=[],
            raw={"block_id": "t", "block_type": BlockType.TEXT,
                 "text": {"elements": []}},
        ),
        DocxBlock(
            block_id="img_b", block_type=BlockType.IMAGE, parent_id="page",
            children=[], raw={"block_id": "img_b", "block_type": BlockType.IMAGE,
                              "image": {"token": "tok_b"}},
        ),
    ]
    order = []

    async def fake_parse(node):
        order.append(("start", node.block_id))
        await asyncio.sleep(0)  # 挂起一次，若并发则 start 会相互交错。
        order.append(("end", node.block_id))
        node.content = f"processed:{node.block_id}"

    monkeypatch.setattr(image_parser, "parse_image_block", fake_parse)

    root = asyncio.run(build_tree(blocks))

    assert order == [
        ("start", "img_a"), ("end", "img_a"),
        ("start", "img_b"), ("end", "img_b"),
    ]
    assert root.children[0].content == "processed:img_a"
    assert root.children[2].content == "processed:img_b"
