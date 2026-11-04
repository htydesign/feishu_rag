"""真实 Milvus 集成测试：验证 get_store() 的初始化、单例缓存、写入与稠密/BM25 检索；不使用 Mock。

固定写入 3 条测试数据（主键 it-feishu-0001 至 it-feishu-0003）；重跑时先按主键
删除旧数据再写入，测试结束后数据保留在集合中，不影响其他真实数据。
"""

import asyncio
import os

import pytest
from langchain_milvus import Milvus
from pymilvus import MilvusClient

from deal_pilot.config.milvus_settings import milvus_settings
from deal_pilot.rag.store.feishu_doc.get_store import get_store
from deal_pilot.rag.store.feishu_doc.init_store import COLLECTION_NAME

TEST_IDS = ["it-feishu-0001", "it-feishu-0002", "it-feishu-0003"]
ID_FILTER = 'id in ["it-feishu-0001", "it-feishu-0002", "it-feishu-0003"]'

TEST_TITLES = ["测试节点：向量数据库", "测试节点：飞书知识库", "测试节点：Agent 框架"]

TEST_TEXTS = [
    "Milvus 是一款开源向量数据库，支持稠密向量与稀疏向量的混合检索，常用于 RAG 知识库的语义召回。",
    "飞书知识库以知识空间为根，节点组成树形结构，文档挂在节点下协作编辑，支持导出为标准树结构。",
    "LangGraph 是构建有状态多 Agent 工作流的框架，支持检查点持久化、中断恢复与人工介入审批。",
]


def make_metadata(index: int, title: str) -> dict:
    """按 schema.py 的字段全集构造切片元数据，验证元数据可完整往返。"""
    return {
        "space_id": "it-space",
        "node_id": f"it-node-{index}",
        "node_title": title,
        "obj_id": f"it-obj-{index}",
        "obj_type": "docx",
        "revision_id": 1,
        "block_id": f"it-block-{index}",
        "block_type": 2,
        "kind": "text",
        "payload": {"language": "zh"},
        "order": index,
        "parent_document_id": None,
        "title_path": [title],
        "references": [],
        "chunk_index": 0,
        "chunk_count": 1,
        "table_row_index": None,
    }


@pytest.fixture
def verify_client():
    if os.environ.get("MILVUS_INTEGRATION") != "1":
        pytest.skip("真实 Milvus 集成测试需显式启用：MILVUS_INTEGRATION=1")
    client = MilvusClient(uri=milvus_settings.uri, db_name=milvus_settings.db_name)
    try:
        yield client
    finally:
        client.close()


def test_get_store_connects_initializes_and_caches_single_instance(verify_client):
    store = asyncio.run(get_store())
    assert isinstance(store, Milvus)
    # 再次调用必须返回同一个缓存实例，且不重复初始化。
    assert asyncio.run(get_store()) is store
    assert verify_client.has_collection(COLLECTION_NAME)


def test_add_three_test_documents_and_retrieve(verify_client):
    store = asyncio.run(get_store())

    # 固定主键重跑时覆盖旧数据，保证测试幂等；flush 确保删除先于写入生效。
    verify_client.delete(collection_name=COLLECTION_NAME, filter=ID_FILTER)
    verify_client.flush(COLLECTION_NAME)

    metadatas = [make_metadata(index, title) for index, title in enumerate(TEST_TITLES)]
    inserted = store.add_texts(texts=TEST_TEXTS, metadatas=metadatas, ids=TEST_IDS)
    assert inserted == TEST_IDS
    # 增长段数据需 flush 后才对查询可见。
    verify_client.flush(COLLECTION_NAME)

    # 行级校验：主键齐全、元数据往返一致。
    rows = verify_client.query(
        collection_name=COLLECTION_NAME,
        filter=ID_FILTER,
        output_fields=[
            "id", "text", "node_title", "kind", "payload", "title_path",
            "chunk_index", "chunk_count",
        ],
    )
    assert sorted(row["id"] for row in rows) == sorted(TEST_IDS)
    for row in rows:
        index = TEST_IDS.index(row["id"])
        assert row["text"] == TEST_TEXTS[index]
        assert row["node_title"] == TEST_TITLES[index]
        assert row["kind"] == "text"
        assert row["payload"] == {"language": "zh"}
        assert row["title_path"] == [TEST_TITLES[index]]
        assert row["chunk_index"] == 0 and row["chunk_count"] == 1

    # 稠密检索：语义查询命中对应文档并排在首位。
    dense_hits = store.similarity_search(query="向量数据库的混合检索", k=3)
    assert [doc.metadata["id"] for doc in dense_hits][0] == "it-feishu-0001"
    assert dense_hits[0].page_content == TEST_TEXTS[0]

    # BM25 全文检索：直接以文本作为查询，经服务端 BM25 函数稀疏化。
    sparse_hits = verify_client.search(
        collection_name=COLLECTION_NAME,
        data=["知识空间 节点 树形结构"],
        anns_field="sparse",
        limit=3,
        output_fields=["id"],
        search_params={"metric_type": "BM25"},
    )
    assert [hit["id"] for hit in sparse_hits[0]][0] == "it-feishu-0002"
