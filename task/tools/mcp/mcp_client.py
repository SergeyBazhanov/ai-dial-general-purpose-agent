from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        instance = cls(mcp_server_url)
        await instance.connect()
        return instance

    async def connect(self):
        """Connect to MCP server"""
        if self.session:
            return
        self._streams_context = streamablehttp_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        result = await self.session.initialize()
        print(f"MCP Session initialized: {result}")

    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        result = await self.session.list_tools()
        return [
            MCPToolModel(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if tool.inputSchema else {}
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        response_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                response_parts.append(content.text)
            else:
                response_parts.append(str(content))
        return "\n".join(response_parts)

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        result: ReadResourceResult = await self.session.read_resource(uri)
        for resource in result.contents:
            if isinstance(resource, TextResourceContents):
                return resource.text
            elif isinstance(resource, BlobResourceContents):
                return resource.blob
        return ""

    async def close(self):
        """Close connection to MCP server"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
        self.session = None
        self._session_context = None
        self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
