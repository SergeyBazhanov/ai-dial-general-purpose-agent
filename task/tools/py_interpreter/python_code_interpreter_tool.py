import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break
        if self._code_execute_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in MCP tool models. Cannot set up PythonCodeInterpreterTool.")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        mcp_client = await MCPClient.create(mcp_url)
        tools = await mcp_client.get_tools()
        return cls(mcp_client=mcp_client, mcp_tool_models=tools, tool_name=tool_name, dial_endpoint=dial_endpoint)

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        code = arguments.get("code", "")
        session_id = arguments.get("session_id")
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")

        if session_id and session_id != 0:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")

        tool_args = {"code": code}
        if session_id:
            tool_args["session_id"] = session_id

        result = await self.mcp_client.call_tool(self._code_execute_tool.name, tool_args)
        result_json = json.loads(result)
        execution_result = _ExecutionResult.model_validate(result_json)

        if execution_result.files:
            dial_client = Dial(base_url=self.dial_endpoint, api_key=tool_call_params.api_key)
            files_home = dial_client.my_appdata_home
            for file_ref in execution_result.files:
                file_name = file_ref.name
                mime_type = file_ref.mime_type
                resource = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))

                if mime_type.startswith("text/") or mime_type in ('application/json', 'application/xml'):
                    file_bytes = resource.encode('utf-8') if isinstance(resource, str) else resource
                else:
                    file_bytes = base64.b64decode(resource) if isinstance(resource, str) else resource

                upload_url = f"files/{(files_home / file_name).as_posix()}"
                dial_client.files.upload(upload_url, file_bytes, mime_type)

                attachment = Attachment(
                    url=upload_url,
                    type=mime_type,
                    title=file_name
                )
                stage.add_attachment(type=attachment.type, title=attachment.title, url=attachment.url)
                tool_call_params.choice.add_attachment(type=attachment.type, title=attachment.title, url=attachment.url)

            result_json["files_uploaded"] = True

        if execution_result.output:
            for i, output_item in enumerate(execution_result.output):
                if len(output_item) > 1000:
                    execution_result.output[i] = output_item[:1000] + "... (truncated)"

        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        return execution_result.model_dump_json()
