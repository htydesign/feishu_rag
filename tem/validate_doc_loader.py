"""从 tem/ 的标准树 JSON 恢复 BlockNode 树，调用 loader 并保存 Documents 结果。"""

import json
from pathlib import Path

from deal_pilot.rag.loader.feishu_doc.feishu_doc_loader import load_feishu_doc_structre
from deal_pilot.rag.structure.feishu_doc.block_node import BlockNode
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType

OUT_DIR = Path(__file__).parent
TREE_FILES = [
    "DFUWw5Ci7iYKKJkcYnWcIHUFnwb.json",
    "EDiDwCmTTic5WIkH3J3cL9aKnqc.json",
]


def restore_node(data: dict, nodes: dict[str, BlockNode]) -> BlockNode:
    block_type = data["block_type"]
    if isinstance(block_type, str):
        block_type = BlockType[block_type]
    node = BlockNode(
        block_id=data["block_id"],
        block_type=block_type,
        content=data["content"],
    )
    node.payload = data.get("payload", {})
    nodes[node.block_id] = node
    for child_data in data.get("children", []):
        node.append(restore_node(child_data, nodes))
    return node


def documents_to_markdown(documents: list) -> str:
    parts = []
    for document in documents:
        meta = document.metadata
        references = "".join(
            f"\n  - {ref['kind']} {ref['id']}" for ref in meta["references"]
        )
        parts.append(
            f"<!-- order={meta['order']} kind={meta['kind']} id={document.id} "
            f"parent={meta['parent_document_id']} "
            f"title_path={' > '.join(meta['title_path'])!r} "
            f"references={references or ' 无'} -->"
        )
        parts.append(document.page_content)
        parts.append("\n" + "-" * 60 + "\n")
    return "\n".join(parts)


for name in TREE_FILES:
    data = json.loads((OUT_DIR / name).read_text(encoding="utf-8"))
    nodes: dict[str, BlockNode] = {}
    root = restore_node(data["tree"], nodes)
    assert root.block_id == data["tree"]["block_id"]

    documents = load_feishu_doc_structre(root, metadata={
        "node_id": data["node_id"],
        "title": data["title"],
        "url": data["url"],
        "revision_id": data["revision_id"],
    })

    stem = data["node_id"]
    json_path = OUT_DIR / f"{stem}.documents.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "id": document.id,
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    md_path = OUT_DIR / f"{stem}.documents.md"
    md_path.write_text(documents_to_markdown(documents), encoding="utf-8")
    print(f"[完成] {data['title']}: {len(documents)} 份 Document -> {json_path.name}, {md_path.name}")
