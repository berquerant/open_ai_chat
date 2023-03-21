"""Entry point of CLI."""
import json
import sys

from pkommand import Parser, Wrapper

from open_ai_chat.chat import Chat
from open_ai_chat.message import MessageList
from open_ai_chat.token import Tokenizer


def chat(
    role_sep: str = "$",
    chat_sep: str = "$$",
    dry: bool = False,
    bulk: bool = False,
    temperature: float = 1,
    max_tokens: int = 1024,
):
    """
    Create a new chat.
    Receive messages from stdin in the following format:

    ROLE$CONTENT

    Pass multiple messages:

    ROLE$CONTENT$$
    ROLE$CONTENT

    The line breaks immediately following the '$$' is ignored.
    '$' is `role_sep`, '$$' is `chat_sep`.
    If `dry`, no API calls, print the actual messages and number of tokens you plan to send
    If `bulk`, no streaming.
    Default `temperature` is 1.
    Default `max_tokens` is 1024.
    """
    messages = MessageList.from_src(sys.stdin, chat_sep, role_sep)
    params = {"temperature": temperature, "max_tokens": max_tokens}
    if not dry:
        for delta in Chat().chat(messages=messages, stream=not bulk, **params):
            print(delta.content, end="", flush=True)
        print()
        return

    print("request messages:")
    for message in messages:
        print(json.dumps(message.dict(), separators=(",", ":"), ensure_ascii=False))
    token_count = Tokenizer().count(messages)
    print(f"token count: {token_count}")


def main() -> int:
    """Entry point of CLI."""
    w = Wrapper(Parser("open_ai_chat"))
    w.add(chat)
    w.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
