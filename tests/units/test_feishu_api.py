"""飞书只读 API 单元测试（基于官方 lark-oapi SDK）。

通过替换 lark-oapi 的传输层（``Transport.aexecute``）返回构造的 RawResponse，
并 stub ``TokenManager.get_self_tenant_token``，完全屏蔽真实网络与凭证。
异步测试沿用工程内 asyncio.run 包裹的写法（无 pytest-asyncio）。
"""

import asyncio
import json

import httpx
import pytest
from lark_oapi.core.http.transport import Transport
from lark_oapi.core.model.raw_response import RawResponse
from lark_oapi.core.token.manager import TokenManager

from deal_pilot.feishu.api import client as client_module
from deal_pilot.feishu.api.docx import DocxAPI
from deal_pilot.feishu.api.image import download_image
from deal_pilot.feishu.api.errors import (
    FeishuAPIError,
    FeishuAuthError,
    FeishuNetworkError,
    FeishuNotFoundError,
    FeishuPermissionError,
    FeishuRateLimitError,
    FeishuUnsupportedObjectTypeError,
)
from deal_pilot.feishu.api.models import DocxBlock, DocxDocument, WikiNode, WikiSpace
from deal_pilot.feishu.api.wiki import WikiAPI

_ok = {"code": 0, "msg": "success"}

# ---- SDK 数据类占位（构造 RawResponse） ----


def _raw(status, content):
    r = RawResponse()
    r.status_code = status
    r.content = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
    return r


def _add_route(routes, uri, responder):
    routes.append(lambda req: responder(req) if req.uri == uri else None)


def _q(req, key, default=None):
    for k, v in req.queries:
        if k == key:
            return v
    return default


@pytest.fixture
def feishu_api(monkeypatch):
    """注入 SDK 传输层与假 token，屏蔽真实网络与凭证。"""
    from deal_pilot.config.feishu_settings import feishu_settings

    monkeypatch.setattr(feishu_settings, "app_id", "cli_test")
    monkeypatch.setattr(feishu_settings, "app_secret", "secret")
    routes = []

    async def fake_aexecute(conf, req, option):
        for responder in routes:
            response = responder(req)
            if response is not None:
                return response
        return _raw(500, {"code": 1, "msg": f"未匹配路由: {req.uri}"})

    monkeypatch.setattr(Transport, "aexecute", fake_aexecute)
    monkeypatch.setattr(TokenManager, "get_self_tenant_token", staticmethod(lambda conf: "t-test-token"))
    return routes


def _raw_binary(status, content, content_type=None):
    """构造带响应头的二进制 RawResponse，模拟 media 下载响应。"""
    r = _raw(status, content)
    if content_type is not None:
        r.headers = {"Content-Type": content_type}
    return r


# ---------- WikiAPI.list_spaces ----------


def test_list_spaces_single_page(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces",
        lambda req: _raw(
            200,
            {
                **_ok,
                "data": {
                    "items": [
                        {"space_id": "s1", "name": "A", "description": "descA"},
                        {"space_id": "s2", "name": "B"},
                    ],
                    "page_token": "",
                    "has_more": False,
                },
            },
        ),
    )

    async def scenario():
        spaces = await WikiAPI().list_spaces()
        assert spaces == [
            WikiSpace(space_id="s1", name="A", description="descA"),
            WikiSpace(space_id="s2", name="B", description=None),
        ]

    asyncio.run(scenario())


def test_list_spaces_paginates(feishu_api):
    pages = {
        None: {"items": [{"space_id": "s1", "name": "A"}], "page_token": "p1", "has_more": True},
        "p1": {"items": [{"space_id": "s2", "name": "B"}], "page_token": "", "has_more": False},
    }
    requested = []

    def responder(req):
        requested.append(_q(req, "page_token"))
        return _raw(200, {**_ok, "data": pages.get(_q(req, "page_token"))})

    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", responder)

    async def scenario():
        spaces = await WikiAPI().list_spaces()
        assert [s.space_id for s in spaces] == ["s1", "s2"]

    asyncio.run(scenario())
    assert requested == [None, "p1"]


def test_list_spaces_keeps_paging_when_page_empty_but_has_more(feishu_api):
    """权限过滤下可能整页为空但 has_more 仍为 true，翻页必须继续。"""
    pages = {
        None: {"items": [], "page_token": "p1", "has_more": True},
        "p1": {"items": [{"space_id": "s1", "name": "A"}], "page_token": "", "has_more": False},
    }

    def responder(req):
        return _raw(200, {**_ok, "data": pages.get(_q(req, "page_token"))})

    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", responder)

    async def scenario():
        assert await WikiAPI().list_spaces() == [WikiSpace(space_id="s1", name="A")]

    asyncio.run(scenario())


def test_list_spaces_raises_on_page_failure(feishu_api):
    pages = {
        None: {"items": [{"space_id": "s1", "name": "A"}], "page_token": "p1", "has_more": True},
        "p1": None,
    }

    def responder(req):
        data = pages.get(_q(req, "page_token"))
        if data is None:
            return _raw(200, {"code": 131007, "msg": "internal err"})
        return _raw(200, {**_ok, "data": data})

    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", responder)

    async def scenario():
        with pytest.raises(FeishuAPIError):
            await WikiAPI().list_spaces()

    asyncio.run(scenario())


# ---------- WikiAPI.get_space ----------


def test_get_space_returns_none_when_not_found(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/:space_id",
        lambda req: _raw(200, {"code": 131005, "msg": "space not found"}),
    )

    async def scenario():
        assert await WikiAPI().get_space("s1") is None

    asyncio.run(scenario())


def test_get_space_permission_denied(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/:space_id",
        lambda req: _raw(200, {"code": 131006, "msg": "wiki space permission denied"}),
    )

    async def scenario():
        with pytest.raises(FeishuPermissionError):
            await WikiAPI().get_space("s1")

    asyncio.run(scenario())


def test_get_space_success(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/:space_id",
        lambda req: _raw(
            200, {**_ok, "data": {"space": {"space_id": "s1", "name": "研发", "description": "d"}}}
        ),
    )

    async def scenario():
        assert await WikiAPI().get_space("s1") == WikiSpace(space_id="s1", name="研发", description="d")

    asyncio.run(scenario())


# ---------- WikiAPI.get_node ----------


def _node(item):
    return _raw(200, {**_ok, "data": {"node": item}})


def test_get_node_success_builds_url(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/get_node",
        lambda req: _node(
            {
                "space_id": "s1",
                "node_token": "wikcn1",
                "obj_token": "doxcn1",
                "obj_type": "docx",
                "parent_node_token": None,
                "has_child": False,
                "title": "架构",
            }
        ),
    )

    async def scenario():
        node = await WikiAPI().get_node("wikcn1")
        assert node == WikiNode(
            node_token="wikcn1",
            space_id="s1",
            obj_token="doxcn1",
            obj_type="docx",
            title="架构",
            url="https://www.feishu.cn/wiki/wikcn1",
            has_child=False,
            parent_node_token=None,
        )

    asyncio.run(scenario())


def test_get_node_not_found_returns_none(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/get_node",
        lambda req: _raw(404, {"code": 1, "msg": "not found"}),
    )

    async def scenario():
        assert await WikiAPI().get_node("missing") is None

    asyncio.run(scenario())


def test_get_node_unsupported_type_raises(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/get_node",
        lambda req: _node(
            {
                "space_id": "s1",
                "node_token": "wikcn1",
                "obj_token": "bit001",
                "obj_type": "bitable",
                "parent_node_token": None,
                "has_child": False,
                "title": "台账",
            }
        ),
    )

    async def scenario():
        with pytest.raises(FeishuUnsupportedObjectTypeError):
            await WikiAPI().get_node("wikcn1")

    asyncio.run(scenario())


# ---------- WikiAPI.list_nodes ----------


def test_list_nodes_empty_dir(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/:space_id/nodes",
        lambda req: _raw(200, {**_ok, "data": {"items": [], "page_token": "", "has_more": False}}),
    )

    async def scenario():
        assert await WikiAPI().list_nodes("s1") == []

    asyncio.run(scenario())


def test_list_nodes_paginates_and_passes_parent(feishu_api):
    pages = {
        ("root", None): {"items": [{"space_id": "s1", "node_token": "a", "obj_token": "da",
                                    "obj_type": "docx", "title": "A", "has_child": False}],
                         "page_token": "p1", "has_more": True},
        ("root", "p1"): {"items": [{"space_id": "s1", "node_token": "b", "obj_token": "db",
                                    "obj_type": "docx", "title": "B", "has_child": False}],
                         "page_token": "", "has_more": False},
    }
    seen_params = []

    def responder(req):
        key = (_q(req, "parent_node_token"), _q(req, "page_token"))
        seen_params.append(key)
        return _raw(200, {**_ok, "data": pages[key]})

    _add_route(feishu_api, "/open-apis/wiki/v2/spaces/:space_id/nodes", responder)

    async def scenario():
        nodes = await WikiAPI().list_nodes("s1", parent_node_token="root")
        assert [n.node_token for n in nodes] == ["a", "b"]

    asyncio.run(scenario())
    assert seen_params[0][0] == "root"


def test_list_nodes_unsupported_type_raises(feishu_api):
    _add_route(
        feishu_api,
        "/open-apis/wiki/v2/spaces/:space_id/nodes",
        lambda req: _raw(
            200,
            {
                **_ok,
                "data": {
                    "items": [{"space_id": "s1", "node_token": "a", "obj_token": "db",
                               "obj_type": "sheet", "title": "S", "has_child": False}],
                    "page_token": "",
                    "has_more": False,
                },
            },
        ),
    )

    async def scenario():
        with pytest.raises(FeishuUnsupportedObjectTypeError):
            await WikiAPI().list_nodes("s1")

    asyncio.run(scenario())


# ---------- WikiAPI.list_all_space_nodes ----------


def _all_nodes_router(pages):
    def responder(req):
        space_id = req.paths["space_id"]
        parent = _q(req, "parent_node_token")
        return _raw(200, {**_ok, "data": pages[(space_id, parent)]})

    return responder


def test_list_all_space_nodes_recurses_and_dedup(feishu_api):
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", lambda req: _raw(
        200, {**_ok, "data": {"items": [
            {"space_id": "s1", "name": "一"}, {"space_id": "s2", "name": "二"},
        ], "page_token": "", "has_more": False}}))
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces/:space_id/nodes", _all_nodes_router({
        ("s1", None): {"items": [{"space_id": "s1", "node_token": "a", "obj_token": "da", "obj_type": "docx",
                                  "title": "A", "has_child": True}], "page_token": "", "has_more": False},
        ("s1", "a"): {"items": [
            {"space_id": "s1", "node_token": "a1", "obj_token": "da1", "obj_type": "docx",
             "title": "A1", "has_child": False, "parent_node_token": "a"},
            {"space_id": "s1", "node_token": "a2", "obj_token": "da2", "obj_type": "docx",
             "title": "A2", "has_child": False, "parent_node_token": "a"},
        ], "page_token": "", "has_more": False},
        ("s2", None): {"items": [{"space_id": "s2", "node_token": "b", "obj_token": "db", "obj_type": "docx",
                                  "title": "B", "has_child": False}], "page_token": "", "has_more": False},
    }))

    async def scenario():
        nodes = await WikiAPI().list_all_space_nodes()
        assert [(n.space_id, n.node_token) for n in nodes] == [
            ("s1", "a"), ("s1", "a1"), ("s1", "a2"), ("s2", "b"),
        ]

    asyncio.run(scenario())


# ---------- DocxAPI.get_document ----------


def test_get_document_success(feishu_api):
    _add_route(feishu_api, "/open-apis/docx/v1/documents/:document_id", lambda req: _raw(
        200, {**_ok, "data": {"document": {"document_id": "doxcn1", "title": "指南", "revision_id": 3}}}))
    async def scenario():
        doc = await DocxAPI().get_document("doxcn1")
        assert doc == DocxDocument(document_id="doxcn1", title="指南", revision_id=3)
    asyncio.run(scenario())


def test_get_document_not_found_returns_none(feishu_api):
    _add_route(feishu_api, "/open-apis/docx/v1/documents/:document_id", lambda req: _raw(
        404, {"code": 1770002, "msg": "not found"}))
    async def scenario():
        assert await DocxAPI().get_document("doxcn1") is None
    asyncio.run(scenario())


# ---------- DocxAPI.list_blocks ----------


def test_list_blocks_paginates_with_revision(feishu_api):
    pages = {
        (3, None): {"items": [{"block_id": "b1", "block_type": 1, "parent_id": None, "children": ["b2"],
                               "text": {"elements": []}}],
                    "page_token": "pt1", "has_more": True},
        (3, "pt1"): {"items": [{"block_id": "b2", "block_type": 2, "parent_id": "b1", "children": []}],
                     "page_token": "", "has_more": False},
    }
    seen = []

    def responder(req):
        rev = int(_q(req, "document_revision_id", -1))
        pt = _q(req, "page_token")
        seen.append((rev, pt))
        return _raw(200, {**_ok, "data": pages[(rev, pt)]})

    _add_route(feishu_api, "/open-apis/docx/v1/documents/:document_id/blocks", responder)

    async def scenario():
        blocks = await DocxAPI().list_blocks("doxcn1", document_revision_id=3)
        assert blocks == [
            DocxBlock(block_id="b1", block_type=1, parent_id=None, children=["b2"],
                      raw={"block_id": "b1", "block_type": 1, "parent_id": None, "children": ["b2"],
                           "text": {"elements": []}}),
            DocxBlock(block_id="b2", block_type=2, parent_id="b1", children=[], raw={
                "block_id": "b2", "block_type": 2, "parent_id": "b1", "children": []}),
        ]
    asyncio.run(scenario())
    assert seen[0][0] == 3


def test_list_blocks_omits_revision_when_default(feishu_api):
    seen = []

    def responder(req):
        seen.append(_q(req, "document_revision_id"))
        return _raw(200, {**_ok, "data": {"items": [], "page_token": "", "has_more": False}})

    _add_route(feishu_api, "/open-apis/docx/v1/documents/:document_id/blocks", responder)

    async def scenario():
        assert await DocxAPI().list_blocks("doxcn1") == []
    asyncio.run(scenario())
    assert seen == [None]


def test_list_blocks_raises_when_document_missing(feishu_api):
    _add_route(feishu_api, "/open-apis/docx/v1/documents/:document_id/blocks", lambda req: _raw(
        404, {"code": 1770002, "msg": "not found"}))
    async def scenario():
        with pytest.raises(FeishuNotFoundError):
            await DocxAPI().list_blocks("doxcn1")
    asyncio.run(scenario())


# ---------- 客户端与错误分类 ----------


def test_missing_credentials_raises(monkeypatch):
    from deal_pilot.config.feishu_settings import feishu_settings
    monkeypatch.setattr(feishu_settings, "app_id", "")
    monkeypatch.setattr(feishu_settings, "app_secret", "")
    async def scenario():
        with pytest.raises(FeishuAuthError, match="FEISHU_APP_ID"):
            await WikiAPI().list_spaces()
    asyncio.run(scenario())


def test_rate_limit_mapped_to_specific_error(feishu_api):
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", lambda req: _raw(
        429, {"code": 99991400, "msg": "rate limited"}))
    async def scenario():
        with pytest.raises(FeishuRateLimitError):
            await WikiAPI().list_spaces()
    asyncio.run(scenario())


# ---------- 补充：网络与边界行为 ----------


def test_network_error_raises(monkeypatch):
    """传输层网络失败（非业务响应）映射为 FeishuNetworkError。"""
    from deal_pilot.config.feishu_settings import feishu_settings
    monkeypatch.setattr(feishu_settings, "app_id", "cli_test")
    monkeypatch.setattr(feishu_settings, "app_secret", "secret")
    monkeypatch.setattr(TokenManager, "get_self_tenant_token", staticmethod(lambda conf: "t-test-token"))

    async def boom(conf, req, option):
        raise httpx.ConnectError("connect failed", request=None)

    monkeypatch.setattr(Transport, "aexecute", boom)

    async def scenario():
        with pytest.raises(FeishuNetworkError):
            await WikiAPI().list_spaces()

    asyncio.run(scenario())


def test_non_json_response_raises_api_error(feishu_api):
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", lambda req: _raw(200, b"not json"))

    async def scenario():
        with pytest.raises(FeishuAPIError):
            await WikiAPI().list_spaces()

    asyncio.run(scenario())


def test_paginate_raises_when_has_more_without_token(feishu_api):
    """has_more=true 但 page_token 为空，视为服务异常，避免无限翻页。"""
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", lambda req: _raw(
        200, {**_ok, "data": {"items": [{"space_id": "s1", "name": "A"}],
                              "page_token": "", "has_more": True}}))

    async def scenario():
        with pytest.raises(FeishuAPIError):
            await WikiAPI().list_spaces()

    asyncio.run(scenario())


def test_list_all_space_nodes_unsupported_type_propagates(feishu_api):
    """list_all_space_nodes 遇到非 docx 节点，直接抛错而非静默跳过。"""
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces", lambda req: _raw(
        200, {**_ok, "data": {"items": [{"space_id": "s1", "name": "一"}],
                              "page_token": "", "has_more": False}}))
    _add_route(feishu_api, "/open-apis/wiki/v2/spaces/:space_id/nodes", lambda req: _raw(
        200, {**_ok, "data": {"items": [{"space_id": "s1", "node_token": "a", "obj_token": "da",
                                         "obj_type": "file", "title": "F", "has_child": False}],
                              "page_token": "", "has_more": False}}))

    async def scenario():
        with pytest.raises(FeishuUnsupportedObjectTypeError):
            await WikiAPI().list_all_space_nodes()

    asyncio.run(scenario())


# ---------- 图片下载 download_image ----------

_MEDIA_URI = "/open-apis/drive/v1/medias/:file_token/download"


def test_download_image_success_returns_mime_and_bytes(feishu_api):
    payload = b"\x89PNG\r\n\x1a\nbinary"
    _add_route(feishu_api, _MEDIA_URI,
               lambda req: _raw_binary(200, payload, "image/png"))

    async def scenario():
        image = await download_image("img_tok")
        assert image.mime == "image/png"
        assert image.data == payload

    asyncio.run(scenario())


def test_download_image_falls_back_mime_without_content_type_header(feishu_api):
    _add_route(feishu_api, _MEDIA_URI, lambda req: _raw_binary(200, b"\x00\xff"))

    async def scenario():
        image = await download_image("img_tok")
        assert image.mime == "application/octet-stream"
        assert image.data == b"\x00\xff"

    asyncio.run(scenario())


def test_download_image_not_found_classified(feishu_api):
    _add_route(feishu_api, _MEDIA_URI,
               lambda req: _raw(404, {"code": 131005, "msg": "media not found"}))

    async def scenario():
        with pytest.raises(FeishuNotFoundError):
            await download_image("img_tok")

    asyncio.run(scenario())


def test_download_image_permission_denied(feishu_api):
    _add_route(feishu_api, _MEDIA_URI,
               lambda req: _raw(403, {"code": 131006, "msg": "permission denied"}))

    async def scenario():
        with pytest.raises(FeishuPermissionError):
            await download_image("img_tok")

    asyncio.run(scenario())
