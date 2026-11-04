#!/usr/bin/env python3
"""将 resources/企业知识库 下的文档真实上传为飞书知识库。

规则：根目录下每个子文件夹 = 一个知识空间；子文件夹内 .md 文件通过
媒体上传 + 导入任务（mount_type=2 挂载到知识空间）导入为 docx 文档；
图片素材子文件夹内的图片上传为云空间文件并创建为知识库文件节点。

幂等：已完成的导入记录写入 tem/feishu_wiki_state.json，重跑自动跳过。
需要的应用权限（企业自建应用，发布版本后生效）：
- wiki:wiki（查看、评论、编辑和管理知识库）
- drive:drive（查看、编辑和管理云空间中所有文件）
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import requests
from lark_oapi.api.docx.v1 import CreateDocumentRequest, CreateDocumentRequestBody
from lark_oapi.api.drive.v1 import (
    UploadAllFileRequest,
    UploadAllFileRequestBody,
)
from lark_oapi.api.wiki.v2 import (
    CreateSpaceNodeRequest,
    ListSpaceNodeRequest,
    ListSpaceRequest,
    MoveDocsToWikiSpaceNodeRequest,
    MoveDocsToWikiSpaceNodeRequestBody,
    Node,
)
from rich.console import Console
from rich.table import Table

from deal_pilot.feishu.api._client import docx_client, drive_client, sdk_call, wiki_client
from deal_pilot.feishu.api.errors import FeishuAPIError

# 支持直接 `python scripts/upload_wiki.py` 运行（此时 sys.path 优先级不同）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.md2docx import parse_markdown, write_document  # noqa: E402
import scripts.md2docx as md2docx  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "resources" / "企业知识库"
STATE_PATH = PROJECT_ROOT / "tem" / "feishu_wiki_state.json"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 120.0

console = Console()
USER_TOKEN: str | None = None


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"spaces": {}, "docs": {}, "files": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def list_root_nodes(wiki, space_id: str) -> list[dict]:
    """读取空间根目录全部节点（原始字典，允许任意 obj_type）。"""
    nodes: list[dict] = []
    page_token: str | None = None
    while True:
        builder = ListSpaceNodeRequest.builder().space_id(space_id).page_size(50)
        if page_token is not None:
            builder.page_token(page_token)
        data = await sdk_call(lambda: wiki.space_node.list(builder.build()))
        nodes.extend(data.get("items") or [])
        if not data.get("has_more"):
            return nodes
        page_token = data["page_token"]


async def create_space_via_user_token(name: str) -> str:
    """创建知识空间的接口只支持用户身份；用 user_access_token 调用原始 HTTP。"""
    assert USER_TOKEN
    response = requests.post(
        "https://open.feishu.cn/open-apis/wiki/v2/spaces",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={"name": name},
        timeout=30,
    )
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"以用户身份创建知识空间 {name} 失败：{payload}")
    space_id = payload["data"]["space"]["space_id"]
    console.print(f"[green]已创建知识空间（用户身份）[/] {name} -> {space_id}")
    return space_id


async def ensure_space(wiki, existing: dict[str, str], name: str) -> str:
    if name in existing:
        return existing[name]
    if USER_TOKEN:
        space_id = await create_space_via_user_token(name)
    else:
        raise RuntimeError(
            f"缺少名为 {name} 的知识空间。创建知识空间的接口仅支持用户身份，二选一：\n"
            "  A. 用 --user-token <user_access_token> 提供用户令牌，脚本自动创建空间；\n"
            "  B. 在飞书客户端手动创建同名空知识空间，并在「设置-成员设置」中添加本应用为\n"
            "     成员（可编辑权限），脚本会按名称复用。"
        )
    existing[name] = space_id
    return space_id


async def upload_drive_file(drive, path: Path, root_folder: str) -> str:
    """上传普通文件到应用云空间根目录，用于创建知识库文件节点。"""
    with path.open("rb") as handle:
        data = await sdk_call(
            lambda: drive.file.upload_all(
                UploadAllFileRequest.builder()
                .request_body(
                    UploadAllFileRequestBody.builder()
                    .file_name(path.name)
                    .parent_type("explorer")
                    .parent_node(root_folder)
                    .size(path.stat().st_size)
                    .file(handle)
                    .build()
                )
                .build()
            )
        )
    file_token = data.get("file_token")
    if not isinstance(file_token, str) or not file_token:
        raise RuntimeError(f"上传 {path.name} 未返回有效 file_token")
    return file_token


async def create_document_from_markdown(drive, wiki, md_path: Path, space_id: str) -> str:
    """docx API 直接创建文档并写入 Markdown 内容，移入知识空间，返回 node_token。

    背景：该租户的「导入文件」异步任务始终静默失败（job_status=2 且无错误信息，
    与文件类型和挂载点无关），故改用 docx API 逐块构建文档；图片以 docx_image
    素材内联写入，效果优于导入通道（导入不处理本地图片引用）。
    """
    items = parse_markdown(md_path)
    data = await sdk_call(
        lambda: docx_client().document.create(
            CreateDocumentRequest.builder()
            .request_body(CreateDocumentRequestBody.builder().title(md_path.stem).build())
            .build()
        )
    )
    document_id = data["document"]["document_id"]
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError(f"创建文档失败，未返回 document_id：{md_path.name}")
    counts = await write_document(docx_client(), drive, document_id, items)

    await sdk_call(
        lambda: wiki.space_node.move_docs_to_wiki(
            MoveDocsToWikiSpaceNodeRequest.builder()
            .space_id(space_id)
            .request_body(
                MoveDocsToWikiSpaceNodeRequestBody.builder()
                .apply(True)
                .obj_type("docx")
                .obj_token(document_id)
                .build()
            )
            .build()
        )
    )
    # 移入是异步任务，轮询等待节点出现在空间根目录。
    deadline = time.monotonic() + 30.0
    while True:
        node = await find_node_by_obj_token(wiki, space_id, document_id)
        if node is not None:
            console.print(
                f"    [dim]内容统计[/] 标题 {counts['heading']} / 段落 {counts['para']} / "
                f"表格 {counts['table']} / 图片 {counts['image']}"
            )
            return node["node_token"]
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"文档已创建但未在知识空间出现节点：{md_path.name}，document_id={document_id}"
            )
        await asyncio.sleep(2.0)


async def get_app_root_folder() -> str:
    """获取应用云空间根目录 token（该接口未包含在当前 SDK 版本，走原始 HTTP）。"""
    import requests

    from deal_pilot.config.feishu_settings import feishu_settings

    token_response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": feishu_settings.app_id, "app_secret": feishu_settings.app_secret},
        timeout=30,
    )
    tenant_token = token_response.json()["tenant_access_token"]
    response = requests.get(
        "https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta",
        headers={"Authorization": f"Bearer {tenant_token}"},
        timeout=30,
    )
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"获取应用根文件夹失败：{payload}")
    root_token = payload["data"]["token"]
    if not isinstance(root_token, str) or not root_token:
        raise RuntimeError(f"应用根文件夹响应缺少 token：{payload}")
    return root_token


async def upload_image_as_file_node(
    wiki, drive, image_path: Path, space_id: str
) -> tuple[str, str]:
    """为图片创建内嵌该图片的文档节点，返回 (node_token, title)。

    官方限制：创建节点接口不支持 obj_type=file（file 仅支持快捷方式），
    因此每张图片创建为一个仅包含内联图片的新版文档。
    """
    items = [md2docx.ImageItem(alt=image_path.stem, path=image_path)]
    data = await sdk_call(
        lambda: docx_client().document.create(
            CreateDocumentRequest.builder()
            .request_body(CreateDocumentRequestBody.builder().title(image_path.stem).build())
            .build()
        )
    )
    document_id = data["document"]["document_id"]
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError(f"创建图片文档失败，未返回 document_id：{image_path.name}")
    await md2docx.create_image_block(docx_client(), drive, document_id, image_path)
    await sdk_call(
        lambda: wiki.space_node.move_docs_to_wiki(
            MoveDocsToWikiSpaceNodeRequest.builder()
            .space_id(space_id)
            .request_body(
                MoveDocsToWikiSpaceNodeRequestBody.builder()
                .apply(True)
                .obj_type("docx")
                .obj_token(document_id)
                .build()
            )
            .build()
        )
    )
    deadline = time.monotonic() + 30.0
    while True:
        node = await find_node_by_obj_token(wiki, space_id, document_id)
        if node is not None:
            return node["node_token"], image_path.stem
        if time.monotonic() > deadline:
            raise TimeoutError(f"图片文档已创建但未出现节点：{image_path.name}")
        await asyncio.sleep(2.0)


async def find_node_by_obj_token(wiki, space_id: str, obj_token: str) -> dict | None:
    for node in await list_root_nodes(wiki, space_id):
        if node.get("obj_token") == obj_token:
            return node
    return None


async def process_document_folder(
    wiki, drive, folder: Path, space_id: str, state: dict
) -> list[tuple[str, str]]:
    """导入一个文档子文件夹，返回 [(标题, wiki 链接)]。"""
    results: list[tuple[str, str]] = []
    existing_titles = {node.get("title") for node in await list_root_nodes(wiki, space_id)}
    for md_path in sorted(folder.glob("*.md")):
        record = state["docs"].get(md_path.name)
        if record and record.get("space_id") == space_id:
            results.append((record["title"], record["url"]))
            console.print(f"[dim]跳过（已导入）[/] {md_path.name}")
            continue
        if md_path.stem in existing_titles:
            console.print(f"[yellow]跳过（空间内同名节点已存在）[/] {md_path.name}")
            continue
        node_token = await create_document_from_markdown(drive, wiki, md_path, space_id)
        node = await find_node_by_obj_token(wiki, space_id, node_token) or {}
        title = node.get("title") if node.get("title") else md_path.stem
        url = f"https://www.feishu.cn/wiki/{node_token}"
        state["docs"][md_path.name] = {
            "space_id": space_id,
            "node_token": node_token,
            "title": title,
            "url": url,
        }
        save_state(state)
        results.append((title, url))
        console.print(f"[green]已导入[/] {md_path.name} -> {url}")
    return results


async def process_image_folder(
    wiki, drive, folder: Path, space_id: str, state: dict
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for image_path in sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    ):
        record = state["files"].get(image_path.name)
        if record and record.get("space_id") == space_id:
            results.append((record["title"], record["url"]))
            console.print(f"[dim]跳过（已上传）[/] {image_path.name}")
            continue
        node_token, title = await upload_image_as_file_node(wiki, drive, image_path, space_id)
        url = f"https://www.feishu.cn/wiki/{node_token}"
        state["files"][image_path.name] = {
            "space_id": space_id,
            "node_token": node_token,
            "title": title,
            "url": url,
        }
        save_state(state)
        results.append((title, url))
        console.print(f"[green]已上传图片节点[/] {image_path.name} -> {url}")
    return results


async def main() -> int:
    global USER_TOKEN
    parser = argparse.ArgumentParser(description="上传企业知识库到飞书知识空间")
    parser.add_argument(
        "--user-token",
        default=None,
        help="user_access_token（创建知识空间必需；从开放平台调试台或 OAuth 获取）",
    )
    args = parser.parse_args()
    USER_TOKEN = args.user_token

    if not SOURCE_ROOT.is_dir():
        console.print(f"[bold red]目录不存在：{SOURCE_ROOT}[/]")
        return 1
    folders = sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())
    if not folders:
        console.print(f"[bold red]目录下没有子文件夹：{SOURCE_ROOT}[/]")
        return 1

    wiki, drive = wiki_client(), drive_client()
    state = load_state()

    spaces_data = await sdk_call(
        lambda: wiki.space.list(
            ListSpaceRequest.builder().page_size(50).build()
        )
    )
    existing: dict[str, str] = {
        item["name"]: item["space_id"]
        for item in spaces_data.get("items") or []
        if isinstance(item.get("name"), str) and isinstance(item.get("space_id"), str)
    }
    # 已记录的空间直接信任，避免列表接口分页遗漏。
    existing.update(state["spaces"])

    summary: list[tuple[str, list[tuple[str, str]]]] = []
    for folder in folders:
        space_id = await ensure_space(wiki, existing, folder.name)
        state["spaces"][folder.name] = space_id
        save_state(state)
        if folder.name == "图片素材":
            items = await process_image_folder(
                wiki, drive, folder, space_id, state
            )
        else:
            items = await process_document_folder(
                wiki, drive, folder, space_id, state
            )
        summary.append((folder.name, items))

    console.print()
    table = Table(title="企业知识库上传结果")
    table.add_column("知识空间")
    table.add_column("条目")
    table.add_column("链接")
    for space_name, items in summary:
        for title, url in items:
            table.add_row(space_name, title, url)
    console.print(table)
    console.print(f"[dim]状态文件：{STATE_PATH}[/]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except FeishuAPIError as error:
        console.print(f"[bold red]飞书 API 错误[/] code={error.code} http={error.http_status}")
        console.print(
            "若为权限类错误（如 code=99991672 或提示 scope 不足），请在飞书开放平台为应用开通：\n"
            "  1. wiki:wiki（查看、评论、编辑和管理知识库）\n"
            "  2. drive:drive（查看、编辑和管理云空间中所有文件）\n"
            "  然后发布应用新版本使权限生效，再重新运行本脚本（已完成条目会自动跳过）。"
        )
        sys.exit(2)
