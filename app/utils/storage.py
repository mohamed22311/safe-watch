import os
import uuid
from pathlib import Path
from typing import Optional
from ..config import settings


class Storage:
    async def store_bytes(self, data: bytes, suffix: str, subdir: str = "") -> str:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self) -> None:
        self.base = Path(settings.storage_base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    async def store_bytes(self, data: bytes, suffix: str, subdir: str = "") -> str:
        sub = self.base / subdir
        sub.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{suffix}"
        path = sub / name
        with open(path, "wb") as f:
            f.write(data)
        if settings.external_base_url:
            return f"{settings.external_base_url.rstrip('/')}/{subdir.strip('/')}/{name}"
        return str(path)


_storage_singleton: Optional[Storage] = None


def get_storage() -> Storage:
    global _storage_singleton
    if _storage_singleton is None:
        # For now only local storage is implemented
        _storage_singleton = LocalStorage()
    return _storage_singleton

