# SPDX-License-Identifier: Apache-2.0

import os
import queue
import threading
import time
from typing import Any, Generator, List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from . import BaseSafeTensorsFileLoader, SafeTensorsFileLoader
from . import cpp as fstcpp


def enable_tqdm(use_tqdm_on_load: bool):
    """Determine whether to enable tqdm progress bar based on distributed settings.

    Progress bar is enabled only on rank 0 when in distributed mode, or always
    enabled in single GPU mode.

    Args:
        use_tqdm_on_load: User preference for enabling tqdm

    Returns:
        bool: True if tqdm should be enabled, False otherwise
    """
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"


def parse_time(timestamp_ms):
    """Convert timestamp in milliseconds to formatted string with millisecond precision.

    Args:
        timestamp_ms: Timestamp in milliseconds

    Returns:
        str: Formatted timestamp string with millisecond precision
    """
    return (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_ms))
        + f".{int((timestamp_ms % 1) * 1000):03d}"
    )


class TimingContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        name: str,
        logger=None,
        batch_id=None,
        threshold_ms: float = 0,
        log_on_exit: bool = True,
    ):
        self.name = name
        self.logger = logger
        self.threshold_ms = threshold_ms
        self.log_on_exit = log_on_exit
        self.start_time = None
        self.elapsed_ms = 0
        self.batch_id = batch_id

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        msg = ""
        if self.batch_id is not None:
            msg += f"Batch {self.batch_id}: "
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        if (
            self.logger
            and self.log_on_exit
            and (self.elapsed_ms >= self.threshold_ms or exc_type is not None)
        ):
            msg += f"{self.name} took {self.elapsed_ms:.3f} ms"
            self.logger(msg)
        return False


class FileBatch:
    """Represents a batch of files loaded into device memory.

    This class encapsulates a batch of SafeTensors files that have been loaded
    into memory, along with metadata about the loading process for performance
    monitoring and debugging purposes.
    """

    def __init__(self, fb, keys: List[str], batch_id: int):
        """Initialize a FileBatch instance.

        Args:
            fb: SafeTensors file buffer containing loaded tensors
            keys: List of tensor keys available in this batch
            batch_id: Unique identifier for this batch
        """
        self.fb = fb  # SafeTensors file buffer
        self.keys = keys  # tensor keys
        self.batch_id = batch_id
        self.load_time: Optional[float] = None  # Record loading time for debugging
        self.add_filenames_time: Optional[float] = None  # Record add_filenames time
        self.copy_files_time: Optional[float] = None  # Record copy_files_to_device time


class PipelineParallel:
    r"""Load .safetensors files in parallel with producer-consumer pattern."""

    def __init__(
        self,
        pg: Optional[Any],
        loader: BaseSafeTensorsFileLoader,
        hf_weights_files: List[str],
        max_concurrent_producers: int = 1,
        queue_size: int = 0,  # Changed default to 0 for unbuffered behavior
        use_tqdm_on_load: bool = True,
        **kwargs,
    ):

        self.loader = loader
        self.hf_weights_files = hf_weights_files
        self.max_concurrent_producers = max_concurrent_producers
        self.queue_size = queue_size
        self.use_tqdm_on_load = use_tqdm_on_load

        # Batch files
        self.weight_files_batches = self._create_batches(pg)

        # Producer-consumer communication
        # For unbuffered behavior (queue_size=0), we use a maxsize of 1 to ensure synchronization
        # but modify the producer logic to wait for consumer to process before producing next
        self.batch_queue: queue.Queue[Union[FileBatch, Exception, None]] = queue.Queue(
            maxsize=max(1, queue_size)
        )  # Ensure at least size 1
        self.stop_event = threading.Event()
        self.error_event = threading.Event()
        self.error_info: Optional[str] = None

        # For unbuffered behavior, we need additional synchronization
        self.consumer_processed: Optional[threading.Event] = (
            threading.Event() if queue_size <= 0 else None
        )
        if queue_size <= 0 and self.consumer_processed is not None:
            self.consumer_processed.set()  # Initially set to allow first production

        # Logging setup - get from environment variable, default to False
        self.print_log = os.getenv("FASTSAFETENSORS_DEBUG", "false").lower() == "true"
        self.log_prefix = f"PG{pg.rank() if pg is not None else 0}"
        fstcpp.set_gil_release(True)

    def _create_batches(self, pg) -> List[List[str]]:
        """Create file batches based on distributed settings.

        In distributed mode, files are grouped by the process group size so that
        each process in the group handles one file from each batch.

        Args:
            pg: Process group for distributed operations

        Returns:
            List[List[str]]: List of file batches, where each batch contains
                            files for all processes in the group
        """
        batch_size = pg.size()
        return [
            self.hf_weights_files[i : i + batch_size]
            for i in range(0, len(self.hf_weights_files), batch_size)
        ]

    def _log_message(self, message: str, is_error: bool = False):
        """Unified logging method for conditional print statements.
        Args:
            message: The message to log
        """
        t = parse_time(time.time())
        if self.print_log or is_error:
            print(f"[{self.log_prefix}] {t} - {message}", flush=True)

    def _log_error(self, message: str):
        self._log_message(message, is_error=True)

    def _load_single_batch(self, batch_id: int, file_list: List[str]):
        """Load a single batch into device memory.

        This method handles the complete process of loading a batch of files:
        1. Preparing file mappings
        2. Adding filenames to the loader
        3. Copying files to device memory
        4. Creating a FileBatch object with timing information
        5. Putting the batch into the queue for consumer processing

        Args:
            batch_id: Identifier for this batch
            file_list: List of files to load in this batch
        """
        if self.stop_event.is_set() or self.error_event.is_set():
            self._log_message(
                f"Producer batch {batch_id} skipped: stop_event-{self.stop_event.is_set()} "
                f"or error_event-{self.error_event.is_set()},"
                f"error_info-{self.error_info}",
                is_error=True,
            )
            return

        try:
            # Prepare file mapping
            rank_file_map = {i: [f] for i, f in enumerate(file_list)}

            with TimingContext("add_filenames", self._log_message, batch_id) as timer:
                self.loader.add_filenames(rank_file_map)
            add_filenames_time = timer.elapsed_ms

            # For unbuffered behavior, wait for consumer to process previous item
            if self.queue_size <= 0 and self.consumer_processed is not None:
                if not self.consumer_processed.wait():
                    raise TimeoutError(
                        "can not waiting for consumer to process previous batch"
                    )
                # Clear the event after wait to ensure next wait will block
                self.consumer_processed.clear()

            with TimingContext(
                "copy_files_to_device", self._log_message, batch_id
            ) as timer:
                fb = self.loader.copy_files_to_device()
            copy_time = timer.elapsed_ms

            # Get tensor keys
            keys = list(fb.key_to_rank_lidx.keys())

            # Create batch object with timing info
            batch = FileBatch(fb, keys, batch_id)
            batch.add_filenames_time = add_filenames_time
            batch.copy_files_time = copy_time
            batch.load_time = add_filenames_time + copy_time  # Total load time

            # Put into queue for consumer processing
            if not self.stop_event.is_set():
                self.batch_queue.put(batch)
                with TimingContext("loader.reset", self._log_message, batch_id):
                    self.loader.reset()

        except Exception as e:
            self.error_info = f"Producer batch {batch_id} failed: {e}"
            self.error_event.set()
            self.batch_queue.put(e)  # Notify consumer of error

    def _producer_worker(self):
        """Producer worker thread: responsible for copy_files_to_device operations.

        This method runs in a separate thread and handles the production of file
        batches by loading them into memory and placing them in the queue for
        consumption by the main thread.
        """
        batch_id = 0
        try:
            for f_list in tqdm(
                self.weight_files_batches,
                desc="Loading fastsafetensors checkpoint shards",
                disable=True,
                bar_format=_BAR_FORMAT,
            ):
                self._load_single_batch(batch_id, f_list)
                batch_id += 1
        except Exception as e:
            if not self.error_event.is_set():
                self.error_info = f"Producer future failed: {e}"
                self.error_event.set()
                self.batch_queue.put(e)

        # Signal end of production
        self.batch_queue.put(None)

    def _consume_single_batch(self):
        with TimingContext("wait_queue", self._log_message) as timer:
            batch_item = self.batch_queue.get()
            # For unbuffered behavior, signal that we've processed this item
            if self.queue_size == 0 and self.consumer_processed is not None:
                self.consumer_processed.set()
            # Check end signal
            if batch_item is None:
                self._log_error(f"get batch_item is None, will break")
                return
            # Check for errors
            if isinstance(batch_item, Exception):
                self._log_error(f"get batch_item is Exception, will raise")
                raise batch_item
            # Process normal batch
            batch = batch_item
            timer.batch_id = batch.batch_id
        queue_wait_time = timer.elapsed_ms
        if queue_wait_time / 1000 > 10:  # Only log if wait time is significant > 10s
            self._log_message(
                f"Warning: Batch {batch.batch_id}: Queue wait took {queue_wait_time:.3f} ms",
                is_error=True,
            )
        try:
            self._log_message(
                f"Batch {batch.batch_id}: tensor key len: {len(batch.keys)}"
            )
            # Consumer operation: extract tensors
            with TimingContext(
                "get_tensor", self._log_message, batch.batch_id
            ) as timer:
                for key in batch.keys:
                    tensor = batch.fb.get_tensor(key)
                    yield key, tensor
            get_tensor_time = timer.elapsed_ms
        finally:
            # Close the file buffer
            with TimingContext("fb.close", self._log_message, batch.batch_id) as timer:
                batch.fb.close()
            close_time = timer.elapsed_ms
        # Log batch summary with all timing info
        self._log_message(
            f"Batch {batch.batch_id} summary: "
            f"add_filenames={batch.add_filenames_time:.3f}ms, "
            f"copy_files={batch.copy_files_time:.3f}ms, "
            f"get_tensor={get_tensor_time:.3f}ms, "
            f"close={close_time:.3f}ms"
        )
        # sync
        if self.queue_size < 0 and self.consumer_processed is not None:
            self.consumer_processed.set()

    def _consumer_worker(self):
        start_time_total = time.time()
        processed_batches = 0
        try:
            for _ in tqdm(
                self.weight_files_batches,
                desc="Loading fastsafetensors checkpoint shards",
                disable=not enable_tqdm(self.use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ):
                yield from self._consume_single_batch()
                processed_batches += 1
        except Exception as e:
            self._log_message(f"Consumer error: {e}", is_error=True)
            self.stop_event.set()
            raise
        finally:
            end_time_total = time.time()
            elapsed_time = end_time_total - start_time_total
            self._log_message(
                f"Completed ParallelLoader iterate_weights, "
                f"processed {processed_batches} batches, total time: {elapsed_time:.2f} seconds"
            )
        if processed_batches < len(self.weight_files_batches):
            self._log_error(f"Unexpected Error: not all tensors has been exported")

    def iterate_weights(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Main weight iterator: consumer logic.

        This method implements the consumer side of the producer-consumer pattern.
        It retrieves batches from the queue, extracts tensors, and yields them
        one by one. It also handles cleanup and error reporting.

        Yields:
            Tuple[str, torch.Tensor]: Key-value pairs of tensor names and tensors
        """
        self._log_message("Starting ParallelLoader iterate_weights")

        # Start producer thread
        producer_thread = threading.Thread(target=self._producer_worker)
        producer_thread.start()

        try:
            yield from self._consumer_worker()
        finally:
            # Cleanup work
            self.stop_event.set()
            producer_thread.join(timeout=5)

    def close(self):
        self.loader.close()


class ParallelLoader(PipelineParallel):
    r"""Load .safetensors files in parallel with producer-consumer pattern.

    This class is a convenience wrapper around PipelineParallel that handles
    the creation of the SafeTensorsFileLoader with appropriate parameters.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed operations.
                           None for single GPU use-cases.
        hf_weights_files (List[str]): List of safetensors files to load.
        max_concurrent_producers (int): Maximum number of concurrent producer threads
                                       for file loading. Currently only 1 is supported.
        queue_size (int): Size of the queue for buffering loaded file batches.
                         Set to 0 for unbuffered behavior.
        use_tqdm_on_load (bool): Enable progress bar during loading.
        device (str): Target device for tensor loading (e.g., "cpu", "cuda:0").
        bbuf_size_kb (int): Bounce buffer size for file copies in KB.
        max_threads (int): Maximum number of threads for memory copies.
        nogds (bool): If True, turn off GDS and fallback to pread with bounce buffer.
        use_shm (bool): If True, use shared memory for file loading.
        set_numa (bool): If True, set NUMA node for optimal memory allocation.
        debug_log (bool): Enable debug logs.
        framework (str): Framework to use for tensor operations, e.g., "pytorch".

    Additional GPU memory consumption: (max_concurrent_producers + queue_size) * file_size
    To reduce GPU memory consumption, re-accessing tensors that have already been accessed is prohibited.

    Examples:
        >> from fastsafetensors import ParallelLoader
        >> src_files = download(target_dir, "gpt2")
        >> iterator = ParallelLoader(None, src_files, max_concurrent_producers=1, queue_size=0)
        >> for key, tensor in iterator.iterate_weights():
        >>     print(f"Loaded tensor: {key}, shape: {tensor.shape}")
    """

    def __init__(
        self,
        pg: Optional[Any],
        hf_weights_files: List[str],
        max_concurrent_producers: int = 1,
        queue_size: int = 0,  # Changed default to 0 for unbuffered behavior
        use_tqdm_on_load: bool = True,
        device: str = "cpu",
        bbuf_size_kb: int = 16 * 1024,
        max_threads: int = 16,
        nogds: bool = False,
        set_numa: bool = True,
        debug_log: bool = False,
        framework="pytorch",
        **kwargs,
    ):
        """Initialize PipelineParallelLoader with a pre-configured SafeTensorsFileLoader.

        Args:
            pg (Optional[Any]): Process group-like objects for distributed operations.
            hf_weights_files (List[str]): List of safetensors files to load.
            max_concurrent_producers (int): Maximum number of concurrent producer threads.
            queue_size (int): Size of the queue for buffering loaded file batches.
            use_tqdm_on_load (bool): Enable progress bar during loading.
            device (str): Target device for tensor loading.
            bbuf_size_kb (int): Bounce buffer size for file copies in KB.
            max_threads (int): Maximum number of threads for memory copies.
            nogds (bool): If True, turn off GDS and fallback to pread with bounce buffer.
            set_numa (bool): If True, set NUMA node for optimal memory allocation.
            debug_log (bool): Enable debug logs.
            framework (str): Framework to use for tensor operations
        """
        loader = SafeTensorsFileLoader(
            pg,
            device,
            bbuf_size_kb=bbuf_size_kb,
            max_threads=max_threads,
            nogds=nogds,
            disable_cache=True,
            set_numa=set_numa,
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
