import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def cd(p: Path):
    now = Path.cwd()
    try:
        os.chdir(str(p))
        yield
    finally:
        os.chdir(str(now))


def run(cmd: str | list[str], dir: Path, *args, **kwargs) -> subprocess.CompletedProcess:
    with cd(dir):
        return subprocess.run(cmd, check=True, *args, **kwargs)


def test_e2e():
    pwd = Path.cwd()
    run(["pipenv", "run", "dist"], pwd)
    run(["pip", "install", "dist/open_ai_chat-0.1.1.tar.gz"], pwd)

    msg = r"system$You are ELIZA.$$user$I have a headache."
    r = run(["python", "-m", "open_ai_chat.cli", "chat", "--dry"], pwd, input=msg, text=True, capture_output=True)

    want = """request messages:
{"role":"system","content":"You are ELIZA."}
{"role":"user","content":"I have a headache."}
token count: 23
"""
    assert want == r.stdout
