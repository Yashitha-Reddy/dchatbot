"""Persistent server configuration; deployment environment takes precedence."""
import os
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_settings():
    path = HERE / ".env"
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name.isidentifier():
            raise ValueError(f"Invalid .env assignment at line {number}; use KEY=value")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote at line {number}")
            value = value[1:-1]
        os.environ.setdefault(name, value)


def server_key():
    configured = os.getenv("CHAT_API_KEY", "").strip()
    if configured:
        return configured
    path = Path(os.getenv("CHAT_KEY_FILE", str(HERE / "data" / "chat_api_key")))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(48))
    key = path.read_text(encoding="utf-8").strip()
    if len(key) < 32:
        raise RuntimeError("Persisted CHAT_KEY_FILE is empty or too short; restore a strong key")
    return key
