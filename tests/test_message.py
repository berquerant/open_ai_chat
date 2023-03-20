from io import StringIO

import pytest

import open_ai_chat.message as message


@pytest.mark.parametrize(
    "src,want",
    [
        (
            """system$You are a psychiatrist""",
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist",
                ),
            ],
        ),
        (
            """system$You are a psychiatrist.
You have a reputation for improving symptoms without drugs.""",
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist.\\nYou have a reputation for improving symptoms without drugs.",
                ),
            ],
        ),
        (
            """system$You are a psychiatrist.
You have a reputation for improving symptoms without drugs.$$
user$I have been stressed lately""",
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist.\\nYou have a reputation for improving symptoms without drugs.",
                ),
                message.Message(
                    role="user",
                    content="I have been stressed lately",
                ),
            ],
        ),
    ],
)
def test_normal_message_list_from_src(src: str, want: message.MessageList):
    got = message.MessageList.from_src(StringIO(src))
    assert got == want


@pytest.mark.parametrize(
    "lines",
    [
        ([""],),
        (["system"],),
        (["system$"],),
    ],
)
def test_error_message_list_from_str(lines: list[str]):
    with pytest.raises(Exception):
        message.MessageList.from_str(lines)


@pytest.mark.parametrize(
    "lines,want",
    [
        (
            [
                "system$You are a psychiatrist",
            ],
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist",
                ),
            ],
        ),
        (
            [
                """system$You are a psychiatrist.
You have a reputation for improving symptoms without drugs.""",
            ],
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist.\\nYou have a reputation for improving symptoms without drugs.",
                ),
            ],
        ),
        (
            [
                "system$You are a psychiatrist$tail",
            ],
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist$tail",
                ),
            ],
        ),
    ],
)
def test_normal_message_list_from_str(lines: list[str], want: message.MessageList):
    got = message.MessageList.from_str(lines)
    assert got == want


@pytest.mark.parametrize(
    "lines",
    [
        (['{"role":"system"}'],),
        (['{"content":"You are a psychiatrist"}'],),
    ],
)
def test_error_message_list_from_raw(lines: list[str]):
    with pytest.raises(ValueError):
        message.MessageList.from_raw(lines)


@pytest.mark.parametrize(
    "lines,want",
    [
        (
            [
                '{"role":"system","content":"You are a psychiatrist"}',
            ],
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist",
                ),
            ],
        ),
        (
            [
                '{"role":"system","content":"You are a psychiatrist"}',
                '{"role":"user","content":"I have been stressed lately"}',
            ],
            [
                message.Message(
                    role="system",
                    content="You are a psychiatrist",
                ),
                message.Message(
                    role="user",
                    content="I have been stressed lately",
                ),
            ],
        ),
    ],
)
def test_normal_message_list_from_raw(lines: list[str], want: message.MessageList):
    got = message.MessageList.from_raw(lines)
    assert got == want
