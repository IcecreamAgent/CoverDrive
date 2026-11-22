"""Discover only insurance tools once, without eager or unrelated MCP access."""

import asyncio

_catalog = None
_lock = None
_loop = None


async def load_business_tools():
    global _catalog, _lock, _loop
    if _catalog is not None:
        return list(_catalog)
    loop = asyncio.get_running_loop()
    if _lock is None or _loop is not loop:
        _loop, _lock = loop, asyncio.Lock()
    async with _lock:
        if _catalog is None:
            from app.mcp_tools import _load_one, _mcp_servers
            config = _mcp_servers().get("insurance")
            if config is None:
                return []
            tools = await _load_one("insurance", config)
            if not tools:
                return []  # An unavailable source can recover on a later task.
            _catalog = tuple(tools)
        return list(_catalog)
