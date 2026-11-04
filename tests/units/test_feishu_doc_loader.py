"""load_feishu_doc_structre 的单元测试；不依赖真实飞书服务。"""

import pytest
from langchain_core.documents import Document

from deal_pilot.rag.loader.feishu_doc.feishu_doc_loader import load_feishu_doc_structre
from deal_pilot.rag.structure.feishu_doc.block_node import BlockNode
from deal_pilot.rag.structure.feishu_doc.block_type import BlockType
from deal_pilot.rag.structure.feishu_doc.code_language import CodeLanguage

INTERNAL_FIELDS = {
    "block_id", "block_type", "kind", "payload", "order",
    "parent_document_id", "title_path", "references",
}


def make_node(
    block_id: str,
    block_type: BlockType | int,
    content: str = "",
    *children: BlockNode,
    payload: dict | None = None,
) -> BlockNode:
    """构造测试用 BlockNode 并按给定顺序挂载子节点。"""
    node = BlockNode(block_id=block_id, block_type=block_type, content=content)
    if payload is not None:
        node.payload = payload
    for child in children:
        node.append(child)
    return node


def find(documents: list[Document], document_id: str) -> Document:
    return next(document for document in documents if document.id == document_id)


def test_page_and_headings_become_documents_with_kind_and_order():
    page = make_node(
        "page", BlockType.PAGE, "产品手册",
        make_node("h1", BlockType.HEADING_1, "快速开始",
                  make_node("h2", BlockType.HEADING_2, "安装")),
    )

    documents = load_feishu_doc_structre(page)

    assert [document.id for document in documents] == ["page", "h1", "h2"]
    assert [document.metadata["order"] for document in documents] == [0, 1, 2]
    assert [document.metadata["kind"] for document in documents] == ["page", "heading", "heading"]
    assert [document.metadata["block_type"] for document in documents] == [1, 3, 4]
    assert documents[0].metadata["parent_document_id"] is None
    assert documents[1].metadata["parent_document_id"] == "page"
    assert documents[2].metadata["parent_document_id"] == "h1"


def test_unstructured_nodes_stay_in_nearest_document_without_new_documents():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "章节",
                  make_node("quote", BlockType.QUOTE, "引用正文",
                            make_node("text", BlockType.TEXT, "引用里的正文")),
                  make_node("callout", BlockType.CALLOUT, "",
                            make_node("text2", BlockType.TEXT, "高亮正文"))),
        make_node("t", BlockType.TEXT, "页面正文"),
    )

    documents = load_feishu_doc_structre(page)

    assert [document.id for document in documents] == ["page", "h1"]
    page_content = find(documents, "h1").page_content
    assert "引用正文" in page_content and "引用里的正文" in page_content
    assert "高亮正文" in page_content
    assert "页面正文" in find(documents, "page").page_content
    assert "> 引用正文" in page_content
    assert "<aside>" in page_content and "</aside>" in page_content


def test_all_heading_levels_and_image_table_code_kinds():
    page = make_node(
        "page", BlockType.PAGE, "根",
        *(
            make_node(f"h{level}", BlockType(level), f"{level}级")
            for level in range(3, 12)
        ),
        make_node("img", BlockType.IMAGE, "一张截图"),
        make_node(
            "tbl", BlockType.TABLE, "",
            make_node("cell", BlockType.TABLE_CELL, "单元格"),
            payload={"row_size": 1, "column_size": 1},
        ),
        make_node("code", BlockType.CODE, "print(1)", payload={"language": CodeLanguage.PYTHON}),
    )

    documents = load_feishu_doc_structre(page)
    kinds = {document.id: document.metadata["kind"] for document in documents}

    assert kinds["page"] == "page"
    assert all(kinds[f"h{level}"] == "heading" for level in range(3, 12))
    assert kinds["img"] == "image" and kinds["tbl"] == "table" and kinds["code"] == "code"
    assert find(documents, "code").metadata["block_type"] == 14
    assert find(documents, "img").metadata["block_type"] == 27
    assert find(documents, "tbl").metadata["block_type"] == 31


def test_title_path_inheritance_and_markdown_prefix():
    page = make_node(
        "page", BlockType.PAGE, "手册",
        make_node("h1", BlockType.HEADING_1, "入门",
                  make_node("h2", BlockType.HEADING_2, "准备",
                            make_node("img", BlockType.IMAGE, "截图"),
                            make_node("code", BlockType.CODE, "pip install", payload={"language": 1}))),
    )

    documents = load_feishu_doc_structre(page)

    assert find(documents, "page").metadata["title_path"] == ["手册"]
    assert find(documents, "h1").metadata["title_path"] == ["手册", "入门"]
    assert find(documents, "img").metadata["title_path"] == ["手册", "入门", "准备"]
    assert find(documents, "h2").page_content.startswith("# 手册\n\n## 入门\n\n### 准备")
    assert find(documents, "h2").page_content.count("准备") == 1
    assert find(documents, "code").page_content.startswith("# 手册\n\n## 入门\n\n### 准备")


def test_empty_heading_kept_with_placeholder_and_references():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "", make_node("text", BlockType.TEXT, "正文")),
    )

    documents = load_feishu_doc_structre(page)

    assert [document.id for document in documents] == ["page", "h1"]
    page_doc = find(documents, "page")
    assert '<document-ref id="h1" kind="heading">标题</document-ref>' in page_doc.page_content
    assert page_doc.metadata["references"] == [{
        "id": "h1",
        "kind": "heading",
        "placeholder": '<document-ref id="h1" kind="heading">标题</document-ref>',
    }]


def test_references_bidirectional_consistency():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "章节",
                  make_node("img", BlockType.IMAGE, "图"),
                  make_node("code", BlockType.CODE, "x=1")),
    )

    documents = load_feishu_doc_structre(page)

    for document in documents:
        for reference in document.metadata["references"]:
            target = find(documents, reference["id"])
            assert target.metadata["parent_document_id"] == document.id
            assert reference["placeholder"] in document.page_content
    assert [reference["id"] for reference in find(documents, "page").metadata["references"]] == ["h1"]
    assert [reference["id"] for reference in find(documents, "h1").metadata["references"]] == ["img", "code"]


def test_placeholder_escapes_attribute_and_label():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node('a"b<c>&d', BlockType.CODE, "x"),
    )

    documents = load_feishu_doc_structre(page)

    placeholder = find(documents, "page").metadata["references"][0]["placeholder"]
    assert placeholder in find(documents, "page").page_content
    assert 'id="a&quot;b&lt;c&gt;&amp;d"' in placeholder
    assert placeholder.count("<document-ref") == 1


def test_metadata_conflict_raises():
    page = make_node("page", BlockType.PAGE, "首页")

    with pytest.raises(ValueError, match="metadata 与内部字段冲突"):
        load_feishu_doc_structre(page, {"order": 1})
    with pytest.raises(ValueError, match="metadata 与内部字段冲突"):
        load_feishu_doc_structre(page, {"references": []})


def test_external_metadata_attached_and_independent():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "章节"),
    )
    external = {"space": {"name": "知识库"}, "tags": ["a"]}

    documents = load_feishu_doc_structre(page, external)

    assert all(document.metadata["space"] == {"name": "知识库"} for document in documents)
    assert all(document.metadata["tags"] == ["a"] for document in documents)
    documents[0].metadata["space"]["name"] = "已修改"
    documents[0].metadata["tags"].append("b")
    assert documents[1].metadata["space"] == {"name": "知识库"}
    assert documents[1].metadata["tags"] == ["a"]
    assert external == {"space": {"name": "知识库"}, "tags": ["a"]}


def test_internal_metadata_exactly_eight_fields_and_payload_deepcopy():
    payload = {"language": 49}
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("code", BlockType.CODE, "print(1)", payload=payload),
    )

    documents = load_feishu_doc_structre(page)
    code_doc = find(documents, "code")

    assert set(code_doc.metadata) == INTERNAL_FIELDS
    assert code_doc.metadata["payload"] == {"language": 49}
    code_doc.metadata["payload"]["language"] = -1
    assert payload == {"language": 49}
    assert find(documents, "page").metadata["references"] != code_doc.metadata["references"]
    assert find(documents, "page").metadata["title_path"] is not code_doc.metadata["title_path"]


def test_code_fence_longer_than_inner_backticks_and_language_name():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("code1", BlockType.CODE, "```python\nx = 1\n```", payload={"language": CodeLanguage.MARKDOWN}),
        make_node("code2", BlockType.CODE, "神秘代码", payload={"language": 999}),
        make_node("code3", BlockType.CODE, "无语言"),
    )

    documents = load_feishu_doc_structre(page)

    content1 = find(documents, "code1").page_content
    assert "````markdown" in content1
    assert content1.rstrip().endswith("````")
    content2 = find(documents, "code2").page_content
    assert "```神秘代码" not in content2 and "\n神秘代码\n```" in content2
    assert find(documents, "code2").metadata["payload"] == {"language": 999}
    content3 = find(documents, "code3").page_content
    assert "\n无语言\n```" in content3


def test_table_rendered_as_html_with_merge_info():
    cells = [make_node(f"c{i}", BlockType.TABLE_CELL, f"格{i}") for i in range(4)]
    cells[1].content = ""
    table = make_node(
        "tbl", BlockType.TABLE, "",
        *cells,
        payload={
            "row_size": 2, "column_size": 2,
            "merge_info": [
                {"row_span": 1, "col_span": 2},
                {"row_span": 0, "col_span": 0},
                {"row_span": 1, "col_span": 1},
                {"row_span": 1, "col_span": 1},
            ],
        },
    )
    page = make_node("page", BlockType.PAGE, "首页", table)

    documents = load_feishu_doc_structre(page)
    content = find(documents, "tbl").page_content

    assert '<td colspan="2"><p>格0</p></td>' in content
    assert "<td><p>格2</p></td>" in content and "<td><p>格3</p></td>" in content
    assert content.count("<td") == 3
    assert find(documents, "tbl").metadata["payload"]["merge_info"][1] == {"row_span": 0, "col_span": 0}


def test_table_cell_structured_child_ownership_and_placeholder():
    image = make_node("img", BlockType.IMAGE, "单元格里的图")
    cell = make_node("cell", BlockType.TABLE_CELL, "", image)
    table = make_node(
        "tbl", BlockType.TABLE, "", cell,
        payload={"row_size": 1, "column_size": 1},
    )
    page = make_node("page", BlockType.PAGE, "首页", table)

    documents = load_feishu_doc_structre(page)

    img_doc = find(documents, "img")
    assert img_doc.metadata["parent_document_id"] == "tbl"
    assert img_doc.metadata["title_path"] == ["首页"]
    table_content = find(documents, "tbl").page_content
    assert '<document-ref id="img" kind="image">图片</document-ref>' in table_content


def test_table_validation_errors():
    def table_with(payload, *cells):
        page = make_node("page", BlockType.PAGE, "首页",
                         make_node("tbl", BlockType.TABLE, "", *cells, payload=payload))
        return load_feishu_doc_structre(page)

    cell = make_node("cell", BlockType.TABLE_CELL, "格")
    with pytest.raises(ValueError, match="row_size、column_size"):
        table_with({"row_size": 0, "column_size": 1}, cell)
    with pytest.raises(ValueError, match="children 必须"):
        table_with({"row_size": 2, "column_size": 2}, cell)
    bad_type = make_node("bad", BlockType.TEXT, "文本")
    with pytest.raises(ValueError, match="TABLE_CELL"):
        table_with({"row_size": 1, "column_size": 1}, bad_type)
    with pytest.raises(ValueError, match="merge_info 数量"):
        table_with({"row_size": 1, "column_size": 1, "merge_info": []}, cell)
    with pytest.raises(ValueError, match="越界"):
        table_with(
            {"row_size": 1, "column_size": 2, "merge_info": [
                {"row_span": 1, "col_span": 3}, {"row_span": 1, "col_span": 1}]},
            make_node("c0", BlockType.TABLE_CELL, "0"),
            make_node("c1", BlockType.TABLE_CELL, "1"),
        )
    with pytest.raises(ValueError, match="重叠"):
        table_with(
            {"row_size": 2, "column_size": 1, "merge_info": [
                {"row_span": 2, "col_span": 1}, {"row_span": 2, "col_span": 1}]},
            make_node("c0", BlockType.TABLE_CELL, "0"),
            make_node("c1", BlockType.TABLE_CELL, "1"),
        )
    with pytest.raises(ValueError, match="覆盖位置"):
        table_with(
            {"row_size": 2, "column_size": 1, "merge_info": [
                {"row_span": 2, "col_span": 1}, {"row_span": 0, "col_span": 0}]},
            make_node("c0", BlockType.TABLE_CELL, "0"),
            make_node("c1", BlockType.TABLE_CELL, "非空内容"),
        )


def test_lists_quote_equation_divider_rendering_kept_in_document():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "格式",
                  make_node("b1", BlockType.BULLET, "要点一"),
                  make_node("o1", BlockType.ORDERED, "步骤"),
                  make_node("todo", BlockType.TODO, "待办事项"),
                  make_node("eq", BlockType.EQUATION, "E=mc^2"),
                  make_node("div", BlockType.DIVIDER, ""),
                  make_node("q", BlockType.QUOTE, "引用"),
                  make_node("qc", BlockType.QUOTE_CONTAINER, "",
                            make_node("qc_text", BlockType.TEXT, "容器内正文"))),
    )

    documents = load_feishu_doc_structre(page)
    content = find(documents, "h1").page_content

    assert "- 要点一" in content
    assert "1. 步骤" in content
    assert "- 待办事项" in content
    assert "$$\nE=mc^2\n$$" in content
    assert "---" in content
    assert "> 引用" in content
    assert "> 容器内正文" in content
    assert len(documents) == 2


def test_unknown_type_marker_and_children_still_processed():
    unknown = make_node("u1", 888, "", make_node("text", BlockType.TEXT, "未知块的正文"))
    page = make_node("page", BlockType.PAGE, "首页", unknown)

    documents = load_feishu_doc_structre(page)

    page_content = find(documents, "page").page_content
    assert "[block type=888 id=u1]" in page_content
    assert "未知块的正文" in page_content


def test_invalid_root_and_ids_and_structure_raise():
    text_root = make_node("t", BlockType.TEXT, "不是页面")
    with pytest.raises(ValueError, match="root 必须为 PAGE"):
        load_feishu_doc_structre(text_root)

    with pytest.raises(ValueError, match="block_id"):
        load_feishu_doc_structre(make_node("  ", BlockType.PAGE, "空ID"))

    duplicated = make_node("dup", BlockType.HEADING_1, "A")
    other = make_node("dup", BlockType.HEADING_1, "B")
    with pytest.raises(ValueError, match="重复 block_id"):
        load_feishu_doc_structre(make_node("page", BlockType.PAGE, "首页", duplicated, other))

    # append() 会把节点从原父节点摘除，直接改 children 才能构造真正的重复挂载。
    shared = make_node("shared", BlockType.TEXT, "正文")
    h1 = make_node("h1", BlockType.HEADING_1, "甲")
    h2 = make_node("h2", BlockType.HEADING_1, "乙")
    page = make_node("page", BlockType.PAGE, "首页", h1, h2)
    h1.children.append(shared)
    h2.children.append(shared)
    with pytest.raises(ValueError, match="重复挂载"):
        load_feishu_doc_structre(page)

    child = make_node("child", BlockType.TEXT, "子")
    parent = make_node("parent", BlockType.QUOTE_CONTAINER, "", child)
    child.children.append(parent)  # 直接改 children 构造环，append 会拒绝祖先
    with pytest.raises(ValueError, match="环"):
        load_feishu_doc_structre(make_node("page", BlockType.PAGE, "首页", parent))


def test_deep_tree_does_not_hit_recursion_limit():
    depth = 800
    node = make_node("leaf", BlockType.TEXT, "底部正文")
    for level in range(depth):
        node = make_node(f"q{level}", BlockType.QUOTE_CONTAINER, "", node)
    page = make_node("page", BlockType.PAGE, "首页", node)

    documents = load_feishu_doc_structre(page)

    assert len(documents) == 1
    assert "底部正文" in find(documents, "page").page_content


def test_input_tree_not_modified():
    page = make_node(
        "page", BlockType.PAGE, "首页",
        make_node("h1", BlockType.HEADING_1, "章节",
                  make_node("code", BlockType.CODE, "x=1", payload={"language": CodeLanguage.GO})),
    )
    snapshot = (
        page.content, [child.block_id for child in page.children],
        page.children[0].children[0].payload,
        page.children[0].parent is page,
        page.children[0].prev is None and page.children[0].next is None,
    )

    load_feishu_doc_structre(page, {"k": "v"})

    assert snapshot == (
        page.content, [child.block_id for child in page.children],
        page.children[0].children[0].payload,
        page.children[0].parent is page,
        page.children[0].prev is None and page.children[0].next is None,
    )
    assert page.children[0].children[0].payload == {"language": CodeLanguage.GO}
