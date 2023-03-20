from tiktoken import encoding_for_model

from open_ai_chat.message import MessageList


class Tokenizer:
    def __init__(self, model: str = "gpt-3.5-turbo-0301"):
        self.encoding = encoding_for_model(model)

    def __count(self, msg: str) -> int:
        return len(self.encoding.encode(msg))

    def count(self, messages: MessageList) -> int:
        """Return the number of tokens."""
        return 2 + sum(4 + self.__count(m.role) + self.__count(m.content) for m in messages)
