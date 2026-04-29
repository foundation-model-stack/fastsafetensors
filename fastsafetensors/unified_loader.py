# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Generator, List, Optional, Tuple, Type

from .common import init_logger
from .config import LoaderConfig, load_config
from .loader import BaseSafeTensorsFileLoader, SafeTensorsFileLoader
from .parallel_loader import PipelineParallel
from .threefs_loader import ThreeFSLoader

logger = init_logger(__name__)

# Used by _resolve_loader_class() via globals().
__all__ = ["UnifiedLoader", "SafeTensorsFileLoader", "ThreeFSLoader"]

# Maps loader name -> module-level class attribute name (strings, not
# references, so that unittest.mock.patch intercepts construction).
_LOADER_REGISTRY: Dict[str, str] = {
    "base": "SafeTensorsFileLoader",
    "3fs": "ThreeFSLoader",
}


def _resolve_loader_class(loader_name: str) -> Type[BaseSafeTensorsFileLoader]:
    attr_name = _LOADER_REGISTRY.get(loader_name)
    if attr_name is None:
        raise ValueError(
            f"Unknown loader type: {loader_name!r}. "
            f"Available: {list(_LOADER_REGISTRY.keys())}"
        )
    cls = globals().get(attr_name)
    if cls is None:
        raise ValueError(
            f"Loader class '{attr_name}' for loader type {loader_name!r} "
            f"is not imported in this module."
        )
    return cls


class UnifiedLoader:
    """Config-driven parallel loader. Dispatches to the loader class
    registered in ``_LOADER_REGISTRY`` based on ``LoaderConfig.loader``.

    Usage::

        loader = UnifiedLoader(pg, files, device="cuda:0")
        for key, tensor in loader.iterate_weights():
            process(key, tensor)
        loader.close()
    """

    def __init__(
        self,
        pg: Optional[Any],
        hf_weights_files: List[str],
        device: str = "cpu",
    ):
        self._config = load_config()
        loader_cls = _resolve_loader_class(self._config.loader)

        common_kwargs: Dict[str, Any] = {
            "framework": self._config.framework,
            "debug_log": self._config.debug_log,
            "set_numa": self._config.set_numa,
            "disable_cache": self._config.disable_cache,
        }

        raw_ext = self._config.get_extension_config(self._config.loader)
        ext_kwargs = loader_cls.process_extension_config(
            raw_ext,
            hf_weights_files=hf_weights_files,
        )

        self._loader = loader_cls(pg, device=device, **common_kwargs, **ext_kwargs)  # type: ignore[arg-type]

        self._pipeline = PipelineParallel(
            pg=pg,
            loader=self._loader,
            hf_weights_files=hf_weights_files,
            **self._config.create_parallel_kwargs(),
        )

        self._log_config_summary(device, len(hf_weights_files), ext_kwargs)

    def _log_config_summary(
        self, device: str, num_files: int, ext_kwargs: Dict[str, Any]
    ) -> None:
        cfg = self._config
        parts = [
            f"loader={cfg.loader}",
            f"framework={cfg.framework}",
            f"device={device}",
            f"files={num_files}",
        ]
        # Extension config (dynamic -- no hardcoded field names)
        for k, v in ext_kwargs.items():
            parts.append(f"{k}={v}")
        parts += [
            f"max_concurrent_producers={cfg.max_concurrent_producers}",
            f"queue_size={cfg.queue_size}",
            f"use_tqdm_on_load={cfg.use_tqdm_on_load}",
        ]
        logger.info("UnifiedLoader initialized: %s", ", ".join(parts))

    @property
    def config(self) -> LoaderConfig:
        return self._config

    def iterate_weights(self) -> Generator[Tuple[str, Any], None, None]:
        return self._pipeline.iterate_weights()

    def close(self):
        # PipelineParallel.close() already closes the underlying loader;
        # do NOT call self._loader.close() to avoid double-close.
        self._pipeline.close()
