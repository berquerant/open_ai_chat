"""Entry point of CLI."""
import json
import sys

from pkommand import Parser, Wrapper

from open_ai_chat.chat import Chat
from open_ai_chat.message import MessageList
from open_ai_chat.token import Tokenizer


def chat(
    role_separator: str = "$",
    msg_separator: str = "$$",
    dry: bool = False,
    bulk: bool = False,
):
    """
    Create a new chat.
    Receive messages from stdin in the following format:

    ROLE$CONTENT

    Pass multiple messages:

    ROLE$CONTENT$$
    ROLE$CONTENT

    The line breaks immediately following the '$$' is ignored.

    '$' is `role_seprator`, '$$' is `msg_separator`.

    If `dry`, no API calls, print the actual messages and number of tokens you plan to send.
    If `bulk`, no streaming.
    """
    messages = MessageList.from_src(sys.stdin, msg_separator, role_separator)
    if not dry:
        for delta in Chat().chat(messages, not bulk):
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
