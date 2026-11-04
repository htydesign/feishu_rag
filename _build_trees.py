import asyncio
import json
import os

from deal_pilot.feishu.api import DocxAPI, WikiAPI
from deal_pilot.rag.structure.feishu_doc.block_node import BlockNode
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType
from deal_pilot.rag.structure.feishu_doc.tree_builder import build_tree

NODES = ["DFUWw5Ci7iYKKJkcYnWcIHUFnwb", "EDiDwCmTTic5WIkH3J3cL9aKnqc"]
OUT = r"F:\AI-FullStack\AI\deal_pilot_rag_learn\tem"


def _type_name(t):
    try:
        return BlockType(t).name
    except ValueError:
        return str(t)


def _serialize(node: BlockNode) -> dict:
    return {
        "block_id": node.block_id,
        "block_type": _type_name(node.block_type),
        "block_type_code": int(node.block_type),
        "content": node.content,
        "payload": node.payload,
        "children": [_serialize(c) for c in node.children],
    }


async def main():
    os.makedirs(OUT, exist_ok=True)
    wiki = WikiAPI()
    docx = DocxAPI()
    for nid in NODES:
        try:
            node = await wiki.get_node(nid)
        except Exception as e:
            print("NODE_FAIL", nid, type(e).__name__, e)
            continue
        try:
            blocks = await docx.list_blocks(node.obj_token)
        except Exception as e:
            print("BLOCKS_FAIL", nid, type(e).__name__, e)
            continue
        blocks_path = os.path.join(OUT, f"{nid}_blocks.json")
        with open(blocks_path, "w", encoding="utf-8") as f:
            json.dump([b.model_dump() for b in blocks], f, ensure_ascii=False, indent=2)
        print("BLOCKS_SAVED", nid, node.title, "count=", len(blocks))
        try:
            root = await build_tree(blocks)
            tree_path = os.path.join(OUT, f"{nid}_tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(_serialize(root), f, ensure_ascii=False, indent=2)
            print("TREE_OK", nid, node.title, "saved=", tree_path)
        except Exception as e:
            print("TREE_FAIL", nid, node.title, type(e).__name__, e)


asyncio.run(main())
