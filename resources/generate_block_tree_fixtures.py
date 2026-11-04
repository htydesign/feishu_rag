"""一次性脚本：从真实飞书文档拉取全量块，生成 build_block_tree 的树测试数据。

用法（项目根目录）：
    uv run python resources/generate_block_tree_fixtures.py

对每个输入 token 先尝试作为文档 ID 读取，失败时尝试作为知识库节点
token 解析出 obj_token 再读取。产物写入 tests/fixtures/block_tree/：
- <token>.blocks.json：list_blocks 的完整扁平块列表及文档元信息；
- <token>.tree.json：用 build_block_tree 构树后的结构摘要（只沿
  children 遍历，不含 parent/prev/next 引用），用于核对树关系。
凭证使用 .env 中的 FEISHU_APP_ID / FEISHU_APP_SECRET，需要应用身份
对相应文档或知识空间有阅读权限。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from deal_pilot.feishu.api import DocxAPI, WikiAPI
from deal_pilot.feishu.api.errors import FeishuError
from deal_pilot.rag.structure.feishu_doc.block_tree import build_block_tree

TOKENS = [
    "DFUWw5Ci7iYKKJkcYnWcIHUFnwb",
    "EDiDwCmTTic5WIkH3J3cL9aKnqc",
]
OUTPUT_DIR = Path("tests/fixtures/block_tree")


def tree_summary(node) -> dict:
    """只沿 children 遍历的树摘要，避免 parent/prev/next 循环引用。"""
    return {
        "block_id": node.block_id,
        "block_type": int(node.block_type),
        "child_count": len(node.children),
        "children": [tree_summary(child) for child in node.children],
    }


async def fetch_document(token: str) -> tuple[str, dict, list]:
    """把 token 解析为文档 ID 并读取元信息与全量块。"""
    docx = DocxAPI()
    document = await docx.get_document(token)
    if document is None:
        node = await WikiAPI().get_node(token)
        if node is None:
            raise FeishuError(f"token {token} 既不是文档 ID 也不是知识库节点")
        print(f"  {token} 是知识库节点，obj_token={node.obj_token}")
        document = await docx.get_document(node.obj_token)
        if document is None:
            raise FeishuError(f"节点 {token} 指向的文档 {node.obj_token} 不存在")
    blocks = await docx.list_blocks(document.document_id)
    meta = {
        "source_token": token,
        "document_id": document.document_id,
        "title": document.title,
        "revision_id": document.revision_id,
    }
    return document.document_id, meta, blocks


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for token in TOKENS:
        print(f"拉取 {token} ...")
        document_id, meta, blocks = await fetch_document(token)
        print(
            f"  文档《{meta['title']}》共 {len(blocks)} 个块，"
            f"根块 {sum(1 for b in blocks if not b.parent_id)} 个"
        )

        blocks_file = OUTPUT_DIR / f"{token}.blocks.json"
        blocks_file.write_text(
            json.dumps({**meta, "blocks": [b.model_dump() for b in blocks]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        root = build_block_tree(blocks)
        tree_file = OUTPUT_DIR / f"{token}.tree.json"
        tree_file.write_text(
            json.dumps({**meta, "tree": tree_summary(root)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  已写入 {blocks_file} 与 {tree_file}")


if __name__ == "__main__":
    asyncio.run(main())
