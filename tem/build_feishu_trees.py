"""拉取两个飞书知识库节点的 DocxBlock 列表，用 build_tree 构树并保存到 tem 文件夹。"""

import asyncio
import json
from pathlib import Path

# 脚本在 LangGraph 运行时之外执行，get_thread_id 依赖的 runnable 上下文不存在。
# 按其预期行为返回 None（无 thread_id 则不加会话头），仅在本脚本进程内生效。
import deal_pilot.utils.create_model as _create_model_module

_create_model_module.get_thread_id = lambda: None

# LangGraph 服务启动时会把 .env 注入进程环境变量，独立脚本需手动加载，
# 否则 ChatOpenAI 从 os.environ 读不到 OPENAI_API_KEY / OPENAI_BASE_URL。
from dotenv import load_dotenv

load_dotenv()

from deal_pilot.feishu.api import DocxAPI, WikiAPI

# 上游网关对视觉请求有频控（短窗口约 3 次，超限返回裸 404 而非 429）。
# 给图片解析加 404 重试与退避，仅在本脚本进程内生效。
import deal_pilot.rag.structure.feishu_doc.image_parser as _image_parser

_orig_parse_image_block = _image_parser.parse_image_block


async def _parse_image_block_with_retry(image_block):
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            await _orig_parse_image_block(image_block)
            if attempt:
                print(f"    [重试成功] {image_block.block_id} 第 {attempt + 1} 次尝试")
            return
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if "404" in message and "Not Found" in message and attempt < 7:
                last_error = error
                print(f"    [频控退避] {image_block.block_id} 404，等待 65 秒后重试（{attempt + 1}/8）")
                await asyncio.sleep(65)
                continue
            raise
    raise last_error  # pragma: no cover


_image_parser.parse_image_block = _parse_image_block_with_retry

from deal_pilot.rag.structure.feishu_doc.block_type import BlockType
from deal_pilot.rag.structure.feishu_doc.tree_builder import build_tree

NODE_IDS = ["DFUWw5Ci7iYKKJkcYnWcIHUFnwb", "EDiDwCmTTic5WIkH3J3cL9aKnqc"]
OUT_DIR = Path(__file__).parent


def node_to_dict(node) -> dict:
    """沿 children 序列化 BlockNode；不触碰 parent/prev/next 避免循环引用。"""
    block_type = node.block_type
    if isinstance(block_type, BlockType):
        block_type = block_type.name
    return {
        "block_id": node.block_id,
        "block_type": block_type,
        "content": node.content,
        "payload": node.payload,
        "children": [node_to_dict(child) for child in node.children],
    }


async def main() -> None:
    wiki = WikiAPI()
    docx = DocxAPI()

    for node_id in NODE_IDS:
        node = await wiki.get_node(node_id)
        if node is None:
            print(f"[跳过] {node_id}: 节点不存在或应用无权限")
            continue
        doc = await docx.get_document(node.obj_token)
        blocks = await docx.list_blocks(node.obj_token)
        print(f"[拉取] {node.title} ({node_id}): obj_token={node.obj_token}, blocks={len(blocks)}")

        root = await build_tree(blocks)

        result = {
            "node_id": node.node_token,
            "space_id": node.space_id,
            "obj_token": node.obj_token,
            "title": node.title,
            "url": node.url,
            "revision_id": doc.revision_id if doc else None,
            "block_count": len(blocks),
            "tree": node_to_dict(root),
        }
        out_path = OUT_DIR / f"{node_id}.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[保存] {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
