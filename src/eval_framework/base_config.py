from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    def as_dict(self) -> dict[str, Any]:
        def simplify_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: simplify_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [simplify_recursive(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        return simplify_recursive(self.model_dump())

    @classmethod
    def from_yaml(cls, yml_filename: str | Path) -> "BaseConfig":
        with open(yml_filename) as conf_file:
            config_dict = yaml.load(conf_file, Loader=yaml.FullLoader)

        return cls(**config_dict)

    def save(self, out_file: Path) -> None:
        with open(out_file, "w", encoding="UTF-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f)
