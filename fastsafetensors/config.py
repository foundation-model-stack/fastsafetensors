# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict

from .common import init_logger

logger = init_logger(__name__)

CONFIG_ENV_VAR = "FASTSAFETENSORS_CONFIG"
DEFAULT_CONFIG_PATH = "fastsafetensors.yaml"


@dataclass
class LoaderConfig:
    """Configuration for fastsafetensors unified loader.

    Core fields live as dataclass attributes. Per-loader extension settings
    (e.g., ``base.copier_type``, ``3fs.mount_point``) are stored in
    ``_extensions`` and accessed via :meth:`get_extension_config`.
    """

    loader: str = "base"
    framework: str = "pytorch"
    debug_log: bool = False
    set_numa: bool = True
    disable_cache: bool = True

    use_pipeline: bool = False
    max_concurrent_producers: int = 1
    queue_size: int = 0
    use_tqdm_on_load: bool = True

    _extensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_concurrent_producers != 1:
            raise ValueError(
                f"max_concurrent_producers must be 1 "
                f"(got {self.max_concurrent_producers}). "
                "Concurrent producers > 1 are not yet supported because broadcast "
                "batches must be processed in strict order across all ranks."
            )

    _COMMON_GROUPS = {"parallel", "debug"}
    _COMMON_FIELDS_FOR_EXTENSION = {
        "framework",
        "debug_log",
        "set_numa",
        "disable_cache",
    }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoaderConfig":
        """Create from dict. ``parallel``/``copy``/``debug`` groups are
        flattened; other dict-valued keys become extension sections."""
        valid_fields = {f.name for f in fields(cls) if not f.name.startswith("_")}
        flat: Dict[str, Any] = {}
        extensions: Dict[str, Dict[str, Any]] = {}

        for key, value in data.items():
            if key in cls._COMMON_GROUPS and isinstance(value, dict):
                # parallel / copy / debug -> flatten into top-level fields
                for sub_key, sub_value in value.items():
                    if sub_key in valid_fields:
                        flat[sub_key] = sub_value
                    else:
                        logger.debug(
                            "Ignoring unknown config field: %s.%s", key, sub_key
                        )
            elif isinstance(value, dict):
                # Any other dict-valued top-level key is treated as an
                # extension section (e.g., base / 3fs / oss / s3).
                extensions[key] = dict(value)
            elif key in valid_fields:
                flat[key] = value
            else:
                logger.debug("Ignoring unknown config field: %s", key)

        flat["_extensions"] = extensions
        return cls(**flat)

    def get_extension_config(self, name: str) -> Dict[str, Any]:
        """Return a shallow copy of the extension section for *name*,
        with cross-loader common fields stripped."""
        raw = self._extensions.get(name, {})
        return {
            k: v for k, v in raw.items() if k not in self._COMMON_FIELDS_FOR_EXTENSION
        }

    @classmethod
    def _from_yaml(cls, path: str) -> "LoaderConfig":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        logger.info("Loaded config from YAML: %s", path)
        return cls.from_dict(data)

    @classmethod
    def _from_json(cls, path: str) -> "LoaderConfig":
        with open(path, "r") as f:
            data = json.load(f)

        logger.info("Loaded config from JSON: %s", path)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "LoaderConfig":
        """Load from file, auto-detecting format by extension."""
        if path.endswith((".yaml", ".yml")):
            return cls._from_yaml(path)
        elif path.endswith(".json"):
            return cls._from_json(path)
        else:
            # Try JSON first, then YAML
            try:
                return cls._from_json(path)
            except (json.JSONDecodeError, ValueError):
                return cls._from_yaml(path)

    def create_parallel_kwargs(self) -> Dict[str, Any]:
        if not self.use_pipeline:
            # queue_size=-1: fully serial (copy_files → broadcast → copy_files),
            # only 1 batch in GPU memory at a time.
            return {"queue_size": -1}
        return {
            "max_concurrent_producers": self.max_concurrent_producers,
            "queue_size": self.queue_size,
            "use_tqdm_on_load": self.use_tqdm_on_load,
        }


def load_config() -> LoaderConfig:
    """Load config: env var > default path > defaults."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path is not None:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(
                f"Config file specified by {CONFIG_ENV_VAR} not found: {env_path}"
            )
        logger.info("Loading config from %s=%s", CONFIG_ENV_VAR, env_path)
        return LoaderConfig.from_file(env_path)

    # 2. Default path
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        logger.info("Loading config from default path: %s", DEFAULT_CONFIG_PATH)
        return LoaderConfig.from_file(DEFAULT_CONFIG_PATH)

    # 3. Defaults
    logger.debug("No config file found, using defaults")
    return LoaderConfig()
