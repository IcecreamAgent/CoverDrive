"""MCP 工具加载：把多个远程 MCP server 暴露成 LangChain 工具。

当前接入两个源：
- 飞书项目（Meego）MCP：streamable HTTP，``X-Mcp-Token`` 头鉴权。
- 保险业务 MCP：本机 ``http://127.0.0.1:8010/mcp``（报价/退保/折扣/评分）。

工具通过 ``langchain-mcp-adapters`` 的 ``MultiServerMCPClient`` 拉取，
返回的 ``BaseTool`` 列表可直接并入轻量 ``create_agent(tools=...)``。
每个源独立加载，任一失败不阻塞其它源。
"""
from __future__ import annotations

import asyncio
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import settings

logger = logging.getLogger(__name__)

# 持有 client 引用，避免会话被 GC 提前关闭；长连接随进程存亡，不做显式 close。
_clients: list[MultiServerMCPClient] = []


def _mcp_servers() -> dict:
    """按配置组装各 MCP 源的连接参数；未配置的源自动跳过。"""
    servers: dict = {}
    if settings.feishu_mcp_token:
        servers["feishu"] = {
            "transport": "http",
            "url": settings.feishu_mcp_url,
            "headers": {"X-Mcp-Token": settings.feishu_mcp_token},
        }
    if settings.insurance_mcp_url:
        servers["insurance"] = {
            "transport": "http",
            "url": settings.insurance_mcp_url,
        }
    return servers


async def _load_one(name: str, config: dict) -> list:
    # 保险服务可能和 voice-agent 同时启动，启动瞬间连接失败时重试，
    # 避免一次启动竞态导致 agent 永久缺少业务工具。
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        client = MultiServerMCPClient({name: config})
        try:
            tools = await client.get_tools()
            _clients.append(client)  # 成功后持有引用
            logger.info(f"{name} MCP 加载成功：{len(tools)} 个工具")
            return tools
        except Exception as exc:
            last_exc = exc
            logger.warning("%s MCP 第 %s/3 次加载失败：%s", name, attempt, type(exc).__name__)
            if attempt < 3:
                await asyncio.sleep(attempt)
    logger.error("%s MCP 加载失败，跳过：%s", name, type(last_exc).__name__)
    return []


async def load_mcp_tools() -> list:
    """加载所有已配置的 MCP 源工具；无配置或全部失败时返回空列表（不阻塞启动）。"""
    tools: list = []
    for name, config in _mcp_servers().items():
        tools.extend(await _load_one(name, config))
    return tools
