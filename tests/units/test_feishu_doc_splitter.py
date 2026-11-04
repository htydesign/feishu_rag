"""split_feishu_doc 的单元测试：切片、引用保护、表格与非法输入。"""

from copy import deepcopy

import pytest
from langchain_core.documents import Document

from deal_pilot.rag.splitter.feishu_doc.feishu_doc_splitter import split_feishu_doc


PLACEHOLDER = '<document-ref id="child-1" kind="code">代码块</document-ref>'
REF = {"id": "child-1", "kind": "code", "placeholder": PLACEHOLDER}


def make_doc(
    doc_id: str,
    kind: str,
    content: str,
    *,
    title_path: list[str] | None = None,
    references: list[dict[str, str]] | None = None,
    payload: dict | None = None,
    **extra,
) -> Document:
    metadata = {
        "block_id": doc_id,
        "block_type": 1,
        "kind": kind,
        "payload": payload or {},
        "order": 0,
        "parent_document_id": None,
        "title_path": title_path or ["页面标题"],
        "references": references or [],
    }
    metadata.update(extra)
    return Document(id=doc_id, page_content=content, metadata=metadata)


def table_doc(
    doc_id: str,
    rows: list[str],
    payload_size: tuple[int, int] | None = None,
    *,
    title_path: list[str] | None = None,
    **kwargs,
) -> Document:
    html_rows = ["<tr>\n" + "\n".join(f"<td>{cell}</td>" for cell in row.split("\x00")) + "\n</tr>" for row in rows]
    content = "# 页面标题\n\n<table>\n" + "\n".join(html_rows) + "\n</table>"
    payload = {"row_size": payload_size[0], "column_size": payload_size[1]} if payload_size else {
        "row_size": len(rows), "column_size": len(rows[0].split("\x00"))
    }
    return make_doc(doc_id, "table", content, payload=payload, title_path=title_path or ["页面标题"], **kwargs)


def raw_table_doc(doc_id: str, html_rows: list[str], payload: dict, **kwargs) -> Document:
    content = "# 页面标题\n\n<table>\n" + "\n".join(html_rows) + "\n</table>"
    return make_doc(doc_id, "table", content, payload=payload, title_path=["页面标题"], **kwargs)


def test_empty_list_returns_empty() -> None:
    assert split_feishu_doc([]) == []


def test_image_and_code_kept_whole_beyond_chunk_size() -> None:
    code = "```python\nprint('hi')\n```"
    docs = [
        make_doc("img-1", "image", "一张长图描述" * 100, payload={"bucket_name": "b"}),
        make_doc("code-1", "code", code),
    ]
    chunks = split_feishu_doc(docs, chunk_size=50, chunk_overlap=10)
    assert [c.id for c in chunks] == ["img-1", "code-1"]
    assert [c.page_content for c in chunks] == ["一张长图描述" * 100, code]
    for chunk in chunks:
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["chunk_count"] == 1
        assert chunk.metadata["kind"] in {"image", "code"}


def test_image_literal_tags_not_treated_as_references() -> None:
    content = '描述中有标签示例 <document-ref id="x" kind="code">示例</document-ref> 结束'
    chunk = split_feishu_doc([make_doc("img-1", "image", content)])[0]
    assert chunk.page_content == content
    assert chunk.metadata["references"] == []


def test_short_text_single_chunk_keeps_id_and_metadata() -> None:
    doc = make_doc("h-1", "heading", "# 页面标题\n\n正文一段。", order=3)
    original = deepcopy(doc)
    chunks = split_feishu_doc([doc])
    assert len(chunks) == 1
    assert chunks[0].id == "h-1"
    assert chunks[0].page_content == doc.page_content
    assert chunks[0].metadata["order"] == 3
    assert doc.page_content == original.page_content
    assert doc.metadata == original.metadata
    assert chunks[0].metadata is not doc.metadata


def test_long_text_split_with_title_path_and_order() -> None:
    paragraphs = "\n\n".join(f"第{i}段。" + "内容" * 40 for i in range(6))
    content = "# 页面标题\n\n## 第一章\n\n" + paragraphs
    doc = make_doc("h-1", "heading", content, title_path=["页面标题", "第一章"])
    chunks = split_feishu_doc([doc], chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["chunk_count"] == len(chunks) for c in chunks)
    body_of = lambda text: text.split("## 第一章\n\n", 1)[1]
    joined = "\n\n".join(body_of(c.page_content) for c in chunks)
    for i in range(6):
        assert f"第{i}段。" in joined
    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.page_content.startswith("# 页面标题\n\n## 第一章\n\n")
    assert [c.id for c in chunks] == [f"h-1::chunk:{i}" for i in range(len(chunks))]
    assert all(c.metadata["block_id"] == "h-1" for c in chunks)


def test_adjacent_chunks_have_overlap() -> None:
    sentences = "".join(f"这是第{i}句话，包含一些内容。" for i in range(40))
    content = "# 页面标题\n\n" + sentences
    doc = make_doc("h-1", "heading", content)
    chunks = split_feishu_doc([doc], chunk_size=100, chunk_overlap=40)
    assert len(chunks) > 1
    first_body = chunks[0].page_content.split("# 页面标题\n\n", 1)[1]
    second_body = chunks[1].page_content.split("# 页面标题\n\n", 1)[1]
    assert any(
        first_body.endswith(second_body[:length])
        for length in range(10, min(len(first_body), len(second_body)) + 1)
    )


def test_placeholder_atomic_and_references_filtered_per_chunk() -> None:
    long_before = "前文。" * 80
    long_after = "后文。" * 80
    content = f"# 页面标题\n\n{long_before}\n\n{PLACEHOLDER}\n\n{long_after}"
    doc = make_doc("h-1", "heading", content, references=[REF])
    chunks = split_feishu_doc([doc], chunk_size=250, chunk_overlap=50)
    assert all(PLACEHOLDER in c.page_content for c in chunks if PLACEHOLDER in c.page_content)
    with_ref = [c for c in chunks if PLACEHOLDER in c.page_content]
    assert with_ref, "占位符应至少完整出现在一个切片中"
    for chunk in with_ref:
        assert chunk.metadata["references"] == [REF]
    for chunk in chunks:
        if PLACEHOLDER not in chunk.page_content:
            assert chunk.metadata["references"] == []
    assert all("<F" not in c.page_content and "\U000f0000" not in c.page_content for c in chunks)


def test_title_path_exhausting_budget_raises() -> None:
    content = "# " + "很长" * 120 + "\n\n正文。"
    doc = make_doc("h-1", "heading", content, title_path=["很长" * 120])
    with pytest.raises(ValueError, match="预算"):
        split_feishu_doc([doc], chunk_size=100, chunk_overlap=10)


def test_placeholder_longer_than_budget_raises() -> None:
    big_ref = {"id": "big", "kind": "table", "placeholder": "<document-ref id=\"big\" kind=\"table\">" + "标" * 300 + "</document-ref>"}
    content = "# 页面标题\n\n" + big_ref["placeholder"]
    doc = make_doc("h-1", "heading", content, references=[big_ref])
    with pytest.raises(ValueError, match="占位符"):
        split_feishu_doc([doc], chunk_size=200, chunk_overlap=10)


def test_table_split_into_header_plus_each_row() -> None:
    doc = table_doc("t-1", ["表头A\x00表头B", "甲1\x00甲2", "乙1\x00乙2"])
    chunks = split_feishu_doc([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert [c.metadata["table_row_index"] for c in chunks] == [1, 2]
    assert [c.id for c in chunks] == ["t-1::chunk:0", "t-1::chunk:1"]
    assert all(c.metadata["chunk_count"] == 2 for c in chunks)
    for chunk in chunks:
        assert chunk.page_content.count("<tr") == 2
        assert chunk.page_content.startswith("# 页面标题\n\n")
        assert "<td>表头A</td>" in chunk.page_content and "<td>表头B</td>" in chunk.page_content
    assert "<td>甲1</td>" in chunks[0].page_content and "<td>乙1</td>" not in chunks[0].page_content
    assert "<td>乙1</td>" in chunks[1].page_content
    assert all(c.metadata["payload"] == {"row_size": 3, "column_size": 2} for c in chunks)


def test_table_rowspan_expanded_and_colspan_kept() -> None:
    content = (
        "# 页面标题\n\n"
        "<table>\n"
        "<tr>\n<td>表头1</td>\n<td>表头2</td>\n<td>表头3</td>\n</tr>\n"
        "<tr>\n<td rowspan=\"2\">合并X</td>\n<td colspan=\"2\">宽Y</td>\n</tr>\n"
        "<tr>\n<td>普通</td>\n<td>补位</td>\n</tr>\n"
        "</table>"
    )
    doc = make_doc("t-1", "table", content, payload={"row_size": 3, "column_size": 3}, title_path=["页面标题"])
    chunks = split_feishu_doc([doc], chunk_size=800, chunk_overlap=50)
    assert len(chunks) == 2
    second = chunks[1].page_content
    assert "<td>合并X</td>" in second and "rowspan" not in second
    assert 'colspan="2"' in chunks[0].page_content


def test_table_with_surrounding_text_and_references() -> None:
    content = (
        f"# 页面标题\n\n表格前说明。\n\n<table>\n<tr>\n<td>表头</td>\n</tr>\n"
        f"<tr>\n<td>{PLACEHOLDER}</td>\n</tr>\n<tr>\n<td>普通行</td>\n</tr>\n</table>\n\n表格后说明。"
    )
    doc = make_doc("t-1", "table", content, references=[REF], payload={"row_size": 3, "column_size": 1}, title_path=["页面标题"])
    chunks = split_feishu_doc([doc], chunk_size=600, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("# 页面标题\n\n表格前说明。")
    assert all("表格后说明。" in c.page_content for c in chunks)
    assert chunks[0].metadata["references"] == [REF]
    assert chunks[1].metadata["references"] == []


def test_table_only_header_kept_single_chunk() -> None:
    doc = table_doc("t-1", ["表头A\x00表头B"])
    chunks = split_feishu_doc([doc])
    assert len(chunks) == 1
    assert chunks[0].id == "t-1"
    assert chunks[0].metadata["table_row_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_table_row_exceeding_chunk_size_raises() -> None:
    doc = table_doc("t-1", ["表头\x00" + "超长" * 200])
    with pytest.raises(ValueError, match="超过 chunk_size"):
        split_feishu_doc([doc], chunk_size=100, chunk_overlap=10)


def test_table_invalid_structure_raises() -> None:
    doc = table_doc("t-1", ["A", "B"], payload_size=(3, 1))
    with pytest.raises(ValueError, match="行数"):
        split_feishu_doc([doc], chunk_size=500, chunk_overlap=50)
    doc = raw_table_doc(
        "t-1",
        ["<tr>\n<td rowspan=\"3\">X</td>\n</tr>"],
        {"row_size": 1, "column_size": 1},
    )
    with pytest.raises(ValueError, match="越界"):
        split_feishu_doc([doc], chunk_size=500, chunk_overlap=50)


def test_invalid_parameters_and_inputs_raise() -> None:
    doc = make_doc("d-1", "page", "# 页面标题\n\n正文")
    with pytest.raises(ValueError, match="chunk_size"):
        split_feishu_doc([doc], chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_feishu_doc([doc], chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="block_id"):
        split_feishu_doc([Document(id="x", page_content="c", metadata={"block_id": "y", "kind": "page", "payload": {}, "title_path": [], "references": []})])
    with pytest.raises(ValueError, match="kind"):
        split_feishu_doc([make_doc("d-1", "unknown", "c")])
    with pytest.raises(ValueError, match="冲突"):
        split_feishu_doc([make_doc("d-1", "page", "c", chunk_index=0)])
    with pytest.raises(ValueError, match="重复"):
        split_feishu_doc([doc, make_doc("d-1", "page", "正文")])


def test_references_inconsistent_with_content_raises() -> None:
    content = "# 页面标题\n\n正文没有占位符。"
    doc = make_doc("h-1", "heading", content, references=[REF])
    with pytest.raises(ValueError, match="不一致"):
        split_feishu_doc([doc])


def test_metadata_deep_copied_and_not_shared() -> None:
    doc = table_doc("t-1", ["表头\x00数", "甲\x00乙", "丙\x00丁"])
    chunks = split_feishu_doc([doc], chunk_size=500, chunk_overlap=50)
    assert chunks[0].metadata is not chunks[1].metadata
    assert chunks[0].metadata["payload"] is not chunks[1].metadata["payload"]
    doc.metadata["payload"]["extra"] = "后加"
    assert "extra" not in chunks[0].metadata["payload"]


def test_order_and_external_metadata_preserved() -> None:
    doc = table_doc(
        "t-1", ["表头\x00数", "甲\x00乙", "丙\x00丁"],
        order=7, parent_document_id="p-1", node_title="自定义",
    )
    chunks = split_feishu_doc([doc], chunk_size=500, chunk_overlap=50)
    assert all(c.metadata["order"] == 7 for c in chunks)
    assert all(c.metadata["parent_document_id"] == "p-1" for c in chunks)
    assert all(c.metadata["title_path"] == ["页面标题"] for c in chunks)
    assert all(c.metadata["node_title"] == "自定义" for c in chunks)
