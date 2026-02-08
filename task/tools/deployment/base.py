import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt", "")
        arguments.pop("prompt", None)

        client = AsyncDial(base_url=self.endpoint, api_key=tool_call_params.api_key)

        messages = [{"role": "user", "content": prompt}]

        chunks = await client.chat.completions.create(
            messages=messages,
            stream=True,
            deployment_name=self.deployment_name,
            extra_body={"custom_fields": arguments} if arguments else None,
            api_version="2025-01-01-preview",
            **self.tool_parameters
        )

        content = ""
        attachments = []
        stage = tool_call_params.stage

        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        content += delta.content
                        stage.append_content(delta.content)
                    if hasattr(delta, 'custom_content') and delta.custom_content:
                        if hasattr(delta.custom_content, 'attachments') and delta.custom_content.attachments:
                            for attachment in delta.custom_content.attachments:
                                attachments.append(attachment)
                                stage.add_attachment(
                                    type=attachment.type,
                                    title=attachment.title,
                                    url=attachment.url
                                )

        custom_content = None
        if attachments:
            from aidial_sdk.chat_completion import Attachment as SdkAttachment
            sdk_attachments = []
            for att in attachments:
                sdk_attachments.append(SdkAttachment(
                    type=att.type,
                    title=att.title,
                    url=att.url
                ))
            custom_content = CustomContent(attachments=sdk_attachments)

        message = Message(
            role=Role.TOOL,
            content=StrictStr(content) if content else None,
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
        return message
