from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(raw=data)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
