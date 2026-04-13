# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

from . import cpp as fstcpp
from .common import init_logger
from .frameworks import get_framework_op
from .loader import BaseSafeTensorsFileLoader, loaded_library
from .parallel_loader import PipelineParallel

logger = init_logger(__name__)


class ThreeFSLoader(BaseSafeTensorsFileLoader):
    """Load .safetensors files using 3FS USRBIO for high-performance I/O.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed loading.
        device (str): Target device where tensors will be loaded (CPU, CUDA, etc.).
        mount_point (str): 3FS mount point path (e.g., "/mnt/3fs").
        debug_log (bool): Enable detailed debug logging.
        disable_cache (bool): Whether to disable caching of loaded tensors.
        framework (str): Deep learning framework to use ("pytorch" or "paddle").
        **kwargs: Additional arguments passed to BaseSafeTensorsFileLoader.

    Examples:
        >>> from fastsafetensors.threefs_loader import ThreeFSLoader
        >>> loader = ThreeFSLoader(None, device="cuda:0", mount_point="/mnt/3fs")
        >>> loader.add_filenames({0: ["/mnt/3fs/model.safetensors"]})
        >>> bufs = loader.copy_files_to_device()
        >>> tensor = bufs.get_tensor("weight")
        >>> loader.close()
    """

    def __init__(
        self,
        pg: Optional[Any],
        device: str = "cpu",
        mount_point: str = "/mnt/3fs",
        debug_log: bool = False,
        disable_cache: bool = True,
        framework: str = "pytorch",
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = self.framework.get_device(device, self.pg)

        global loaded_library
        if not loaded_library:
            fstcpp.load_library_functions()
            loaded_library = True
        fstcpp.set_debug_log(debug_log)
        super().__init__(
            pg,
            self.device,
            copier_type="3fs",
            set_numa=True,
            disable_cache=disable_cache,
            framework=framework,
            mount_point=mount_point,
            **kwargs,
        )


class ParallelThreeFSLoader(PipelineParallel):
    """Parallel loader for .safetensors files using 3FS USRBIO.

    This class provides pipeline-parallel loading of multiple safetensors files
    using 3FS for high-performance I/O operations.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed operations.
        hf_weights_files (List[str]): List of safetensors files to load from 3FS.
        mount_point (str): 3FS mount point path (e.g., "/mnt/3fs").
        max_concurrent_producers (int): Maximum number of concurrent producer threads.
        queue_size (int): Size of the queue for buffering loaded file batches.
                         Default 0 for unbuffered behavior.
        use_tqdm_on_load (bool): Enable progress bar during loading.
        device (str): Target device for tensor loading.
        debug_log (bool): Enable debug logs.
        framework (str): Framework to use for tensor operations ("pytorch" or "paddle").
        **kwargs: Additional arguments passed to the loader.

    Examples:
        >>> from fastsafetensors.threefs_loader import ParallelThreeFSLoader
        >>> files = ["/mnt/3fs/model-00001.safetensors", "/mnt/3fs/model-00002.safetensors"]
        >>> loader = ParallelThreeFSLoader(
        ...     pg=None,
        ...     hf_weights_files=files,
        ...     mount_point="/mnt/3fs",
        ...     device="cuda:0"
        ... )
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """

    def __init__(
        self,
        pg: Optional[Any],
        hf_weights_files: List[str],
        max_concurrent_producers: int = 1,
        queue_size: int = 0,
        use_tqdm_on_load: bool = True,
        device: str = "cpu",
        debug_log: bool = False,
        framework: str = "pytorch",
        **kwargs,
    ):
        from fastsafetensor_3fs_reader import extract_mount_point

        mount_point: str = extract_mount_point(hf_weights_files[0])

        loader = ThreeFSLoader(
            pg,
            device=device,
            mount_point=mount_point,
            disable_cache=True,
            debug_log=debug_log,
            framework=framework,
            **kwargs,
        )

        super().__init__(
            pg,
            loader,
            hf_weights_files,
            max_concurrent_producers,
            queue_size,
            use_tqdm_on_load,
            **kwargs,
        )
