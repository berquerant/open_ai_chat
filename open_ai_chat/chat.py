from dataclasses import dataclass
from typing import Iterator

from openai import ChatCompletion

from open_ai_chat.message import MessageList


@dataclass
class Delta:
    """Result of stream chat."""

    content: str


class Chat:
    """Chat client."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def chat(self, messages: MessageList, stream: bool = True) -> Iterator[Delta]:
        """Create a new chat."""
        response = ChatCompletion.create(
            model=self.model,
            messages=messages.into_request(),
            stream=stream,
        )
        if not stream:
            yield Delta(content=response["choices"][0]["message"]["content"])
            return
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if not delta:
                return
            msg = delta.get("content", "")
            if not msg:
                continue
            yield Delta(content=msg)
