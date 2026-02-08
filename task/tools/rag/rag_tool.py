import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided context. 
Use ONLY the information from the provided context to answer the question. 
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be concise and accurate in your responses.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2',
            device='cpu'
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Performs semantic search on uploaded documents to find relevant content and answer questions. "
            "Supports PDF, TXT, CSV, HTML files. This tool indexes the document, searches for the most relevant "
            "passages matching the query, and generates an answer based on those passages. "
            "Best for answering specific questions about large documents efficiently without reading the entire file. "
            "Prefer this tool over file_content_extraction when asking questions about document content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to search in."
                }
            },
            "required": ["request", "file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments["request"]
        file_url = arguments["file_url"]
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data:
            index, chunks = cached_data
        else:
            extractor = DialFileContentExtractor(self.endpoint, tool_call_params.api_key)
            text_content = extractor.extract_text(file_url)
            if not text_content:
                stage.append_content("Error: File content not found.\n")
                return "Error: File content not found."
            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32'))
            self.document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request]).astype('float32')
        distances, indices = index.search(query_embedding, k=3)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]

        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        client = AsyncDial(base_url=self.endpoint, api_key=tool_call_params.api_key)
        chunks_response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt}
            ],
            stream=True,
            deployment_name=self.deployment_name,
            api_version="2025-01-01-preview"
        )

        collected_content = ""
        async for chunk in chunks_response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    collected_content += delta.content

        return collected_content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return f"""Based on the following context, answer the question.

Context:
{context}

Question: {request}

Answer:"""
