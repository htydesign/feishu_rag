from dataclasses import dataclass
from langchain_core.documents import Document
from pymilvus import Function, FunctionType
from deal_pilot.config.nacos.nacos_schemas import RerankerConfig, RetrieverConfig
from deal_pilot.rag.store.feishu_doc.get_store import get_store


@dataclass
class SearchResult:
    doc: Document
    score: float


def _resolve_reranker_function(question: str, config: RerankerConfig) -> Function:
    if config.reranker == "model":
        return Function(
            name="model_rerank",
            input_field_names=["text"],  # 用于评分的正文字段
            function_type=FunctionType.RERANK,
            params={
                "reranker": "model",
                "provider": "cohere",
                "model_name": config.params.get("model_name", "rerank"),
                "queries": [question],
            },
        )
    if config.reranker == "rrf":
        return Function(
            name="dense_sparse_rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": config.params.get("k", 60)},
        )

    if config.reranker == "weighted":
        return Function(
            name="dense_sparse_weighted",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "weighted",
                "weights": config.params.get("weights", [0.5, 0.5]),
                "norm_score": True,
            },
        )
    raise ValueError(f"reranker取值{config.reranker}无效")


async def search(
    question: str,
    *,
    filter: str | None = None,
    config: RetrieverConfig | None = None,
    threshold: float = 0,
) -> list[SearchResult]:
    store = await get_store()

    if config is None:
        config = RetrieverConfig()

    reranker = _resolve_reranker_function(question, config.reranker)

    reuslts = await store.asimilarity_search_with_score(
        query=question,  # 查询文本；dense 路生成 embedding，sparse 路交给内置 BM25 分析
        fetch_k=config.fetch_k,  # 每一路最多召回 10 条候选，默认 4；两路候选可能重合
        k=config.k,  # 融合后最多返回 4 条，默认 4；候选不足时可能更少
        param=config.param,
        expr=filter,  # 默认 None；同一过滤表达式在两路召回前生效,
        reranker=reranker,
    )

    return [SearchResult(doc, score) for doc, score in reuslts if score > threshold]
