import asyncio

from deal_pilot.feishu.api import DocxAPI, WikiAPI
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType

NODES = ["DFUWw5Ci7iYKKJkcYnWcIHUFnwb", "EDiDwCmTTic5WIkH3J3cL9aKnqc"]


async def main():
    wiki = WikiAPI()
    docx = DocxAPI()
    for nid in NODES:
        node = await wiki.get_node(nid)
        print(f"node {nid}")
        print(f"  obj_type={node.obj_type} title={node.title!r} space={node.space_id}")
        blocks = await docx.list_blocks(node.obj_token)
        print(f"  blocks={len(blocks)}")
        imgs = [b for b in blocks if b.block_type == BlockType.IMAGE]
        print(f"  image_blocks={len(imgs)}")
        # 各类型分布（前 15 个类型）
        from collections import Counter
        dist = Counter(b.block_type for b in blocks)
        top = ", ".join(f"{BlockType(k).name if k in BlockType._value2member_map_ else k}:{v}" for k, v in list(dist.items())[:15])
        print(f"  types: {top}")


asyncio.run(main())
