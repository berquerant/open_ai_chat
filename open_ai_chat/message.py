from collections.abc import Sequence
from typing import Any, Optional, TextIO

from pydantic import BaseModel, validator


class Message(BaseModel):
    """Message model."""

    role: str
    content: str

    @validator("role")
    def role_not_empty(cls, v) -> str:
        """Validate role is not empty."""
        if not v:
            raise ValueError("cannot be empty")
        return v

    @validator("content")
    def content_not_empty(cls, v) -> str:
        """Validate content is not empty"""
        if not v:
            raise ValueError("cannot be empty")
        return v


class MessageList(list[Message]):
    """List of `Message`."""

    @staticmethod
    def from_raw(seq: Sequence[str]) -> "MessageList":
        """Parse json string."""
        return MessageList([Message.parse_raw(x) for x in seq])

    @staticmethod
    def from_obj(seq: Sequence[Any]) -> "MessageList":
        """Parse dict."""
        return MessageList([Message.parse_obj(x) for x in seq])

    @classmethod
    def from_str(cls, seq: Sequence[str], sep: Optional[str] = "$") -> "MessageList":
        """Parse string.
        Format:
          ROLE{sep}CONTENT
        or
          CONTENT
        """

        def parse(x: str) -> Any:
            if sep not in x:
                role = "user"
                content = x
            else:
                role, content = x.split(sep, maxsplit=1)
            return {
                "role": role,
                "content": "\\n".join(content.splitlines()),
            }

        return cls.from_obj([parse(x) for x in seq])

    @classmethod
    def from_src(
        cls,
        src: TextIO,
        sep: Optional[str] = "$$",
        role_sep: Optional[str] = "$",
    ) -> "MessageList":
        """Parse source like sys.stdin, StringIO.
        :sep: seprator of message
        :role_sep: separator of role and content
        """
        return cls.from_str(
            [x.lstrip() for x in src.read().split(sep)],
            sep=role_sep,
        )

    def into_request(self) -> list[Any]:
        """Into list of dictionaries."""
        return [x.dict() for x in self]
