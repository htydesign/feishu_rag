"""本地演示：读取 loader 的 documents.json，切片并导出为 Markdown。

用法：.venv/Scripts/python.exe tem/export_feishu_doc_chunks.py
输出目录已存在时报错退出，不覆盖上次产物。
"""

import json
import re
import sys
from pathlib import Path

import yaml
from langchain_core.documents import Document

from deal_pilot.rag.splitter.feishu_doc.feishu_doc_splitter import split_feishu_doc

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INPUTS = [
    Path("tem/DFUWw5Ci7iYKKJkcYnWcIHUFnwb.documents.json"),
    Path("tem/EDiDwCmTTic5WIkH3J3cL9aKnqc.documents.json"),
]
OUTPUT_ROOT = Path("tem/document_chunks")


def load_documents(path: Path) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Document(id=item["id"], page_content=item["page_content"], metadata=item["metadata"]) for item in data]


def safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value)[:80]


def write_markdown(path: Path, document_id: str, metadata: dict, content: str) -> None:
    front = yaml.safe_dump(
        {"document_id": document_id, "metadata": metadata},
        allow_unicode=True, sort_keys=False, default_flow_style=False,
    )
    path.write_text(f"---\n{front}---\n\n{content}\n", encoding="utf-8")


def read_markdown(path: Path) -> tuple[str, dict, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines[0].strip() != "---":
        raise ValueError(f"{path.name} 缺少 front matter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    front = yaml.safe_load("\n".join(lines[1:end]))
    content = "\n".join(lines[end + 1:]).lstrip("\n")
    if content.endswith("\n"):
        content = content[:-1]
    return front["document_id"], front["metadata"], content


def main() -> int:
    missing = [p for p in INPUTS if not p.exists()]
    if missing:
        print("输入文件缺失，不伪造数据：", ", ".join(str(p) for p in missing))
        return 1
    if OUTPUT_ROOT.exists():
        print(f"输出目录已存在，不覆盖：{OUTPUT_ROOT}")
        return 1

    summary = []
    for path in INPUTS:
        node_id = path.name.removesuffix(".documents.json")
        docs = load_documents(path)
        chunks = split_feishu_doc(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        out_dir = OUTPUT_ROOT / node_id
        out_dir.mkdir(parents=True)

        mismatches = []
        for order, chunk in enumerate(chunks, 1):
            file_path = out_dir / f"{order:04d}-{safe_name(chunk.id)}.md"
            write_markdown(file_path, chunk.id, chunk.metadata, chunk.page_content)
            doc_id, metadata, content = read_markdown(file_path)
            if doc_id != chunk.id or metadata != chunk.metadata or content != chunk.page_content:
                mismatches.append(file_path.name)
        summary.append((node_id, len(docs), len(chunks), len(mismatches)))
        kinds: dict[str, int] = {}
        for doc in docs:
            kinds[doc.metadata["kind"]] = kinds.get(doc.metadata["kind"], 0) + 1
        print(f"{node_id}: 输入 {len(docs)} 份 ({', '.join(f'{k}×{v}' for k, v in kinds.items())}) -> 切片 {len(chunks)} 片")

    print()
    all_ok = True
    for node_id, in_count, out_count, mismatches in summary:
        status = "复读一致" if not mismatches else f"复读不一致: {mismatches}"
        if mismatches:
            all_ok = False
        print(f"{node_id}: 输入 {in_count} -> 输出 {out_count} | {status}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
