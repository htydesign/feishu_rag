"""build_block_tree 的单元测试；不依赖真实飞书服务（fixtures 为离线数据）。"""

import copy
import json
from pathlib import Path

import pytest

from deal_pilot.feishu.api.models import DocxBlock
from deal_pilot.rag.structure.feishu_doc.block_node import BlockNode
from deal_pilot.rag.structure.feishu_doc.block_tree import build_block_tree
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType


def make_block(
    block_id: str,
    block_type: int,
    *,
    parent_id: str | None = None,
    children: list[str] | None = None,
    **raw_extra,
) -> DocxBlock:
    """构造测试用 DocxBlock，raw 保留块 ID 与类型及附加字段。"""
    raw: dict = {"block_id": block_id, "block_type": block_type}
    raw.update(raw_extra)
    return DocxBlock(
        block_id=block_id,
        block_type=block_type,
        parent_id=parent_id,
        children=children or [],
        raw=raw,
    )


def test_builds_tree_with_ordered_children_and_siblings():
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1", "t2", "t3"]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
        make_block("t2", BlockType.TEXT, parent_id="page"),
        make_block("t3", BlockType.TEXT, parent_id="page"),
    ]

    root = build_block_tree(blocks)

    assert root.parent is None
    assert root.prev is None
    assert root.next is None
    assert root.block_id == "page"
    assert root.block_type is BlockType.PAGE
    assert [child.block_id for child in root.children] == ["t1", "t2", "t3"]

    first, second, third = root.children
    assert all(child.parent is root for child in root.children)
    assert first.prev is None and first.next is second
    assert second.prev is first and second.next is third
    assert third.prev is second and third.next is None


def test_node_content_empty_and_raw_kept_in_payload():
    raw = {
        "block_id": "t1",
        "block_type": BlockType.TEXT,
        "text": {"elements": [{"text_run": {"content": "你好"}}]},
    }
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1"]),
        DocxBlock(
            block_id="t1",
            block_type=BlockType.TEXT,
            parent_id="page",
            children=[],
            raw=raw,
        ),
    ]

    root = build_block_tree(blocks)

    assert root.content == ""
    assert root.payload == {"raw": {"block_id": "page", "block_type": BlockType.PAGE}}
    node = root.children[0]
    assert node.content == ""
    assert node.payload["raw"]["text"]["elements"][0]["text_run"]["content"] == "你好"


def test_child_order_follows_parent_children_list():
    # 服务返回顺序与父块 children 顺序不同，构树以 children 为准。
    blocks = [
        make_block("b", BlockType.TEXT, parent_id="page"),
        make_block("page", BlockType.PAGE, children=["c", "a", "b"]),
        make_block("a", BlockType.TEXT, parent_id="page"),
        make_block("c", BlockType.TEXT, parent_id="page"),
    ]

    root = build_block_tree(blocks)

    assert [child.block_id for child in root.children] == ["c", "a", "b"]


def test_deeply_nested_tree():
    blocks = [
        make_block("page", BlockType.PAGE, children=["h1", "callout"]),
        make_block("h1", BlockType.HEADING_1, parent_id="page", children=["p1"]),
        make_block("p1", BlockType.TEXT, parent_id="h1"),
        make_block("callout", BlockType.CALLOUT, parent_id="page", children=["p2"]),
        make_block("p2", BlockType.TEXT, parent_id="callout"),
    ]

    root = build_block_tree(blocks)

    heading, callout = root.children
    assert heading.block_type is BlockType.HEADING_1
    assert [child.block_id for child in heading.children] == ["p1"]
    assert heading.children[0].children == []
    assert callout.block_type is BlockType.CALLOUT
    assert callout.children[0].block_id == "p2"


def test_unlisted_block_type_keeps_raw_int():
    blocks = [
        make_block("page", BlockType.PAGE, children=["odd"]),
        make_block("odd", 777, parent_id="page"),
    ]

    root = build_block_tree(blocks)

    assert root.block_type is BlockType.PAGE
    assert root.children[0].block_type == 777
    assert not isinstance(root.children[0].block_type, BlockType)


def test_input_blocks_are_not_mutated():
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1"]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
    ]
    snapshot = copy.deepcopy(blocks)

    build_block_tree(blocks)

    assert [block.model_dump() for block in blocks] == [
        block.model_dump() for block in snapshot
    ]


def test_empty_block_list_raises():
    with pytest.raises(ValueError, match="块列表为空"):
        build_block_tree([])


def test_duplicate_block_id_raises():
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1"]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
        make_block("t1", BlockType.BULLET, parent_id="page"),
    ]

    with pytest.raises(ValueError, match="block_id 重复"):
        build_block_tree(blocks)


def test_missing_root_raises():
    blocks = [make_block("t1", BlockType.TEXT, parent_id="page")]

    with pytest.raises(ValueError, match="不存在无父块的根块"):
        build_block_tree(blocks)


def test_multiple_roots_raise():
    blocks = [
        make_block("page1", BlockType.PAGE),
        make_block("page2", BlockType.PAGE),
    ]

    with pytest.raises(ValueError, match="多个无父块的根块"):
        build_block_tree(blocks)


def test_parent_id_referencing_missing_block_raises():
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1"]),
        make_block("t1", BlockType.TEXT, parent_id="ghost"),
    ]

    with pytest.raises(ValueError, match="parent_id 指向不存在的块"):
        build_block_tree(blocks)


def test_children_referencing_missing_block_raises():
    blocks = [make_block("page", BlockType.PAGE, children=["ghost"])]

    with pytest.raises(ValueError, match="children 引用了不存在的块"):
        build_block_tree(blocks)


def test_block_missing_from_parent_children_raises():
    blocks = [
        make_block("page", BlockType.PAGE, children=[]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
    ]

    with pytest.raises(ValueError, match="children 未包含声称的子块"):
        build_block_tree(blocks)


def test_child_listed_under_wrong_parent_raises():
    # t1 声称父块是 other，但只登记在 page 的 children 中。
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1", "other"]),
        make_block("t1", BlockType.TEXT, parent_id="other"),
        make_block("other", BlockType.QUOTE, parent_id="page"),
    ]

    with pytest.raises(ValueError, match="未包含声称的子块"):
        build_block_tree(blocks)


def test_duplicate_entry_in_children_raises():
    blocks = [
        make_block("page", BlockType.PAGE, children=["t1", "t1"]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
    ]

    with pytest.raises(ValueError, match="重复子块"):
        build_block_tree(blocks)


def test_parent_cycle_unreachable_from_root_raises():
    blocks = [
        make_block("page", BlockType.PAGE),
        make_block("a", BlockType.TEXT, parent_id="b", children=["b"]),
        make_block("b", BlockType.TEXT, parent_id="a", children=["a"]),
    ]

    with pytest.raises(ValueError, match="无法从根块到达"):
        build_block_tree(blocks)


def test_empty_parent_id_string_treated_as_root():
    # 真实飞书服务对根页面块返回空字符串 parent_id，而非 None。
    blocks = [
        make_block("page", BlockType.PAGE, parent_id="", children=["t1"]),
        make_block("t1", BlockType.TEXT, parent_id="page"),
    ]

    root = build_block_tree(blocks)

    assert root.block_id == "page"
    assert root.parent is None
    assert [child.block_id for child in root.children] == ["t1"]


# ---------------------------------------------------------------------------
# 真实数据 fixtures：由 resources/generate_block_tree_fixtures.py 生成，
# tests/fixtures/block_tree/*.blocks.json 保存 list_blocks 的完整结果。
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "block_tree"


def load_fixture_blocks(path: Path) -> list[DocxBlock]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DocxBlock.model_validate(block) for block in data["blocks"]]


@pytest.mark.parametrize(
    "blocks_file", sorted(FIXTURE_DIR.glob("*.blocks.json")), ids=lambda p: p.stem
)
def test_build_tree_from_real_document_fixture(blocks_file: Path):
    blocks = load_fixture_blocks(blocks_file)

    root = build_block_tree(blocks)

    assert root.block_type is BlockType.PAGE
    assert root.parent is None
    # 树恰好包含全部输入块，且根页面块与文档同 ID。
    collected: list[BlockNode] = []

    def walk(node: BlockNode) -> None:
        collected.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    assert {node.block_id for node in collected} == {b.block_id for b in blocks}
    assert len(collected) == len(blocks)
    # 兄弟引用与 children 列表一致。
    for node in collected:
        for index, child in enumerate(node.children):
            assert child.parent is node
            assert child.prev is (node.children[index - 1] if index else None)
            assert child.next is (
                node.children[index + 1] if index + 1 < len(node.children) else None
            )
