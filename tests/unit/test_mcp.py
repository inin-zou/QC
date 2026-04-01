"""Tests for robotq/mcp/server.py — MCP tool registration and list_augmentations."""

from __future__ import annotations

import asyncio


# ---------------------------------------------------------------------------
# Import / smoke tests
# ---------------------------------------------------------------------------


def test_mcp_server_imports():
    """Importing mcp from robotq.mcp.server succeeds without error."""
    from robotq.mcp.server import mcp  # noqa: F401


def test_mcp_server_name():
    """The FastMCP server is registered with name 'robotq'."""
    from robotq.mcp.server import mcp

    assert mcp.name == "robotq"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_mcp_has_six_tools():
    """Exactly 6 tools are registered on the MCP server."""
    from robotq.mcp.server import mcp

    tools = mcp._tool_manager.list_tools()
    assert len(tools) == 6


def test_mcp_tool_names():
    """All 6 expected tools are registered."""
    from robotq.mcp.server import mcp

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "list_augmentations",
        "augment_dataset",
        "preview_augmentation",
        "list_adapters",
        "inspect_dataset",
        "generate_config",
    }


# ---------------------------------------------------------------------------
# list_augmentations — direct call (sync function)
# ---------------------------------------------------------------------------


def test_list_augmentations_returns_expected_items():
    """list_augmentations() returns exactly 5 augmentation dicts."""
    from robotq.mcp.server import list_augmentations

    result = list_augmentations()
    assert isinstance(result, list)
    assert len(result) == 5


def test_list_augmentations_has_required_keys():
    """Every dict returned by list_augmentations has name, type, adapter, description keys."""
    from robotq.mcp.server import list_augmentations

    result = list_augmentations()
    required_keys = {"name", "type", "adapter", "description"}
    for item in result:
        assert isinstance(item, dict)
        assert required_keys.issubset(item.keys()), (
            f"Item missing required keys: {item}"
        )


def test_list_augmentations_includes_mirror():
    """list_augmentations() includes an entry with name='mirror'."""
    from robotq.mcp.server import list_augmentations

    names = [item["name"] for item in list_augmentations()]
    assert "mirror" in names


def test_list_augmentations_includes_speed_warp():
    """list_augmentations() includes an entry with name='speed_warp'."""
    from robotq.mcp.server import list_augmentations

    names = [item["name"] for item in list_augmentations()]
    assert "speed_warp" in names


# ---------------------------------------------------------------------------
# list_augmentations — via async tool manager (MCP call path)
# ---------------------------------------------------------------------------


def test_list_augmentations_via_tool_manager():
    """Calling list_augmentations through the MCP tool manager returns 5 dicts."""
    from robotq.mcp.server import mcp

    async def _run():
        return await mcp._tool_manager.call_tool("list_augmentations", {})

    result = asyncio.run(_run())
    assert isinstance(result, list)
    assert len(result) == 5
    for item in result:
        assert isinstance(item, dict)
        assert {"name", "type", "adapter", "description"}.issubset(item.keys())
