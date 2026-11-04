"""Markdown 到飞书新版文档（docx）块的最小转换器。

只覆盖本知识库使用的语法子集：标题（#）、段落、管道表格、
指向本地图片的行内引用（![]() 与 []() 两种写法）。
表格通过「创建表格块 + 逐单元格写入文本」实现；图片先上传为
docx_image 素材再创建图片块，因此可保留图片内联显示。
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from lark_oapi.api.docx.v1 import (
    Block,
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    GetDocumentBlockRequest,
    Image,
    Table,
    TableProperty,
    Text,
    TextElement,
    TextRun,
)
from lark_oapi.api.drive.v1 import UploadAllMediaRequest, UploadAllMediaRequestBody

from deal_pilot.config.feishu_settings import feishu_settings
from deal_pilot.feishu.api._client import sdk_call
from deal_pilot.feishu.api.errors import FeishuAPIError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
HEADING_BLOCK_TYPES = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11}
MAX_BATCH = 40  # children 接口单次块数上限为 50，留出余量
MAX_IMAGE_WIDTH = 600  # 图片显示最大宽度（px），等比缩放

HEADING_RE = re.compile(r"^(#{1,9})\s+(.*?)\s*#*\s*$")
IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)\s*$")


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class TableItem:
    rows: list[list[str]]


@dataclass
class ImageItem:
    alt: str
    path: Path


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _image_target(md_path: Path, target: str) -> Path | None:
    candidate = (md_path.parent / target).resolve()
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
        return candidate
    return None


def parse_markdown(md_path: Path) -> list[Heading | Para | TableItem | ImageItem]:
    items: list[Heading | Para | TableItem | ImageItem] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            items.append(TableItem(rows=table_rows))
            table_rows = []

    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            if _is_table_separator(line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()
        if not line.strip():
            continue
        if heading := HEADING_RE.match(line):
            items.append(Heading(level=len(heading.group(1)), text=heading.group(2)))
            continue
        if image := IMAGE_RE.match(line):
            target = _image_target(md_path, image.group(2))
            if target is not None:
                items.append(ImageItem(alt=image.group(1), path=target))
                continue
        if link := LINK_RE.match(line):
            target = _image_target(md_path, link.group(2))
            if target is not None:
                items.append(ImageItem(alt=link.group(1), path=target))
                continue
        items.append(Para(text=line.strip()))
    flush_table()
    return items


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    return MAX_IMAGE_WIDTH, MAX_IMAGE_WIDTH // 2


def _text_block(content: str) -> Block:
    return (
        Block.builder()
        .block_type(2)
        .text(
            Text.builder()
            .elements([TextElement.builder().text_run(TextRun.builder().content(content).build()).build()])
            .build()
        )
        .build()
    )


def _heading_block(level: int, content: str) -> Block:
    text = Text.builder().elements(
        [TextElement.builder().text_run(TextRun.builder().content(content).build()).build()]
    ).build()
    builder = Block.builder().block_type(HEADING_BLOCK_TYPES[min(level, 9)])
    getattr(builder, f"heading{min(level, 9)}")(text)
    return builder.build()


def _table_block(rows: list[list[str]]) -> Block:
    column_count = max(len(row) for row in rows)
    width = 1500 // max(column_count, 1)
    return (
        Block.builder()
        .block_type(31)
        .table(
            Table.builder()
            .property(
                TableProperty.builder()
                .row_size(len(rows))
                .column_size(column_count)
                .column_width([width] * column_count)
                .header_row(True)
                .build()
            )
            .build()
        )
        .build()
    )


async def create_children(docx, document_id: str, parent_id: str, blocks: list[Block]) -> list[dict]:
    if not blocks:
        return []
    request = (
        CreateDocumentBlockChildrenRequest.builder()
        .document_id(document_id)
        .block_id(parent_id)
        .request_body(CreateDocumentBlockChildrenRequestBody.builder().children(blocks).build())
        .build()
    )
    # 服务端偶发 1770001 invalid param（与内容无关，重试即可成功）。
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            data = await sdk_call(lambda: docx.document_block_children.create(request))
            return data.get("children") or []
        except FeishuAPIError as error:
            if error.code != 1770001:
                raise
            last_error = error
            await asyncio.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


async def _table_cell_ids(docx, document_id: str, table_block_id: str) -> list[str]:
    data = await sdk_call(
        lambda: docx.document_block.get(
            GetDocumentBlockRequest.builder().document_id(document_id).block_id(table_block_id).build()
        )
    )
    block = data.get("block") or {}
    table = block.get("table") or {}
    cells = table.get("cells") or []
    if not isinstance(cells, list):
        raise RuntimeError(f"表格块 {table_block_id} 响应缺少 cells")
    return [str(cell) for cell in cells]


async def write_document(
    docx, drive, document_id: str, items: list[Heading | Para | TableItem | ImageItem]
) -> dict:
    """把解析后的块写入文档，返回统计信息。"""
    counts = {"heading": 0, "para": 0, "table": 0, "image": 0}
    pending: list[Block] = []

    async def flush() -> None:
        nonlocal pending
        for start in range(0, len(pending), MAX_BATCH):
            await create_children(docx, document_id, document_id, pending[start : start + MAX_BATCH])
        pending = []

    for item in items:
        if isinstance(item, Heading):
            pending.append(_heading_block(item.level, item.text))
            counts["heading"] += 1
        elif isinstance(item, Para):
            pending.append(_text_block(item.text))
            counts["para"] += 1
        elif isinstance(item, ImageItem):
            await flush()
            await create_image_block(docx, drive, document_id, item.path)
            counts["image"] += 1
        elif isinstance(item, TableItem):
            await flush()
            # 服务端限制：单次创建的表格最多 9 行，超出的表格分段创建，
            # 续段重复表头行以保持可读性。
            header, body_rows = item.rows[0], item.rows[1:]
            if not body_rows:
                chunks = [item.rows]
            else:
                chunk_size = 8
                chunks = [
                    [header] + body_rows[start : start + chunk_size]
                    for start in range(0, len(body_rows), chunk_size)
                ]
            for chunk in chunks:
                created = await create_children(docx, document_id, document_id, [_table_block(chunk)])
                table_block = created[0] if created else {}
                table_id = table_block.get("block_id")
                if not table_id:
                    raise RuntimeError(f"创建表格块失败：{document_id}")
                cell_ids = await _table_cell_ids(docx, document_id, table_id)
                column_count = max(len(row) for row in chunk)
                expected = len(chunk) * column_count
                if len(cell_ids) != expected:
                    raise RuntimeError(
                        f"表格块 {table_id} 单元格数量不符：预期 {expected}，实际 {len(cell_ids)}"
                    )
                for row_index, row in enumerate(chunk):
                    for column_index in range(column_count):
                        text = row[column_index] if column_index < len(row) else ""
                        cell_id = cell_ids[row_index * column_count + column_index]
                        await create_children(
                            docx, document_id, cell_id, [_text_block(text if text else " ")]
                        )
            counts["table"] += 1
    await flush()
    return counts


def _tenant_token() -> str:
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": feishu_settings.app_id, "app_secret": feishu_settings.app_secret},
        timeout=30,
    )
    return response.json()["tenant_access_token"]


async def create_image_block(docx, drive, document_id: str, path: Path) -> None:
    """按官方 FAQ 三步法插入内联图片：建空图片块 → 上传素材到该块 → 绑定素材。

    docx 建块接口拒绝带 token 的图片块（1770001 invalid param），且素材必须
    上传到图片块（parent_node=图片块 ID），绑定用 replace_image（SDK 的 patch
    序列化与该接口不匹配，此处用原始 HTTP）。
    """
    created = await create_children(
        docx, document_id, document_id,
        [Block.builder().block_type(27).image(Image.builder().build()).build()],
    )
    image_block_id = created[0].get("block_id")
    if not image_block_id:
        raise RuntimeError(f"创建空图片块失败：{document_id}")
    with path.open("rb") as handle:
        data = await sdk_call(
            lambda: drive.media.upload_all(
                UploadAllMediaRequest.builder()
                .request_body(
                    UploadAllMediaRequestBody.builder()
                    .file_name(path.name)
                    .parent_type("docx_image")
                    .parent_node(image_block_id)
                    .size(path.stat().st_size)
                    .file(handle)
                    .build()
                )
                .build()
            )
        )
    token = data.get("file_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"上传图片素材失败，未返回 file_token：{path.name}")
    patch = requests.patch(
        f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks/{image_block_id}",
        headers={"Authorization": f"Bearer {_tenant_token()}"},
        json={"replace_image": {"token": token}},
        timeout=30,
    )
    payload = patch.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"绑定图片素材失败：{payload}")
