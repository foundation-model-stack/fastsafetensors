# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from loguru import logger

QUANTIZE_CONFIG_FILENAME = "quantize_config.json"


def get_config(device_index: int) -> Tuple[bool, int, int]:
    auto_config = os.getenv("FST_CONFIG", "auto")
    nogds = os.getenv("FST_NOGDS")  # disable GDS if FST_NOGDS==1
    nogds = nogds is not None and nogds == "1"
    max_copier_threads = int(
        os.getenv("FST_THREADS", "16")
    )  # number of copy threads at host CPU
    bbuf_size_kb_total = int(
        os.getenv("FST_BBUF_SIZE_KB", "163840")
    )  # size of bounce buffer at host memory for FST_NOGDS==1
    if auto_config == "auto":
        nogds = not os.path.exists("/run/udev")  # udev directory is required for GDS
        from fastsafetensors.common import get_device_numa_node

        node = get_device_numa_node(device_index)
        total_l2_size = 0
        phys_cpus = {}
        failed = False
        for cpudir in glob.glob(f"/sys/devices/system/node/node{node}/cpu[0-9]*"):
            try:
                with open(
                    f"{cpudir}/cache/index2/size"
                ) as f:  # L2 cache size for a cpu
                    size_str = f.read().strip()
                    if size_str[-1] != "K":
                        raise Exception(f"cannot parse {cpudir}/cache/index2/size")
                    total_l2_size += int(size_str[:-1])
                with open(f"{cpudir}/topology/core_id") as f:  # physical core ID
                    phys_cpus[f.read().strip()] = True
            except Exception as e:
                failed = True
                print(f"Failed to auto-configure fastsafetensors. reason: {e}")
                break
        if not failed and total_l2_size > 0:
            bbuf_size_kb_total = total_l2_size
        if not failed and len(phys_cpus) > 0:
            max_copier_threads = len(phys_cpus)
    return (nogds, max_copier_threads, bbuf_size_kb_total)


class FastWeights:
    def __init__(
        self,
        filenames: List[str],
        device: torch.device,
        dtype: torch.dtype,
        pg: dist.ProcessGroup,
        debug_log: bool = False,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None,
    ):
        from fastsafetensors.loader import SafeTensorsFileLoader

        (nogds, max_copier_threads, bbuf_size_kb_total) = get_config(device.index)
        self._loader = SafeTensorsFileLoader(
            device,
            pg=pg,
            bbuf_size_kb=bbuf_size_kb_total // pg.size(),
            max_threads=max_copier_threads,
            nogds=nogds,
            debug_log=debug_log,
        )
        rank_filenames: Dict[str, List[str]] = {
            rank: [] for rank in range(0, pg.size())
        }
        max_copy_block_size = 1
        total_size = 0
        for idx, filename in enumerate(
            sorted(filenames, key=lambda x: os.path.basename(x))
        ):
            rank_filenames[idx % pg.size()].append(filename)
            s = os.stat(filename)
            total_size += s.st_size
            if max_copy_block_size < s.st_size:
                max_copy_block_size = s.st_size
        self._loader.add_filenames(rank_filenames)
        if len(filenames) < max_copier_threads:
            max_copy_block_size = total_size // pg.size() // max_copier_threads
            if max_copy_block_size % bbuf_size_kb_total * 1024 > 0:
                max_copy_block_size = (
                    max_copy_block_size
                    - max_copy_block_size % (bbuf_size_kb_total * 1024)
                    + (bbuf_size_kb_total * 1024)
                )
        msg = f"Fastsafetensors configuration: GDS={not nogds}, maximum number of file copy threads={max_copier_threads}, copy block size={max_copy_block_size}B"
        if nogds:
            msg += f", total bounce buffer size={bbuf_size_kb_total * 1024}B"
        print(msg)
        self._fb = self._loader.copy_files_to_device(
            dtype, max_copy_block_size=max_copy_block_size
        )
        self.device = device
        self.dtype = dtype
        if aliases is None:
            aliases = {}
        self.prefix = prefix
        self.aliases = aliases
        self.process_group = pg
        self.routing = {}
        for key in self._loader.get_keys():
            self.routing[key] = True

    def close(self):
        self._fb.close()
        self._loader.close()
        torch.cuda.empty_cache()

    def _get_alias(self, tensor_name: str) -> str:
        if self._fb.get_filename(tensor_name) is None:
            if tensor_name in self.aliases:
                for alias in self.aliases[tensor_name]:
                    if self._fb.get_filename(alias) is not None:
                        return alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return tensor_name

    def get_shape(self, tensor_name: str) -> torch.Size:
        return torch.Size(self._fb.get_shape(self._get_alias(tensor_name)))

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        return self._fb.get_tensor(
            self._get_alias(tensor_name), device=self.device, dtype=self.dtype
        ).get_raw()

    def push_tensor(self, tensor_name: str, dst_rank: int) -> torch.Tensor:
        return self._fb.push_tensor(
            self._get_alias(tensor_name), dst_rank, device=self.device, dtype=self.dtype
        ).get_raw()

    def get_partial_sharded(self, tensor_name: str, dim: int) -> torch.Tensor:
        return self._fb.get_sharded(
            self._get_alias(tensor_name), dim, device=self.device, dtype=self.dtype
        ).get_raw()

    def get_sharded(self, tensor_name: str, dim: int = 1) -> torch.Tensor:
        return self._fb.get_sharded(
            self._get_alias(tensor_name), dim, device=self.device, dtype=self.dtype
        ).get_raw()

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int):
        if quantize == "gptq":
            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            bits, groupsize = self._get_gptq_params()
            use_gptq_cuda = False
            if bits == 4:
                from text_generation_server.utils.layers import HAS_GPTQ_CUDA

                use_gptq_cuda = HAS_GPTQ_CUDA
                if use_gptq_cuda:
                    logger.info(f"Using GPTQ cuda kernels for col {prefixes}")

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_gptq_cuda)
        else:
            tensor_names = [self._get_alias(f"{prefix}.weight") for prefix in prefixes]
            weight = self._fb.get_multi_cols(
                tensor_names, dim, device=self.device, dtype=self.dtype
            )
        return weight

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "gptq":
            bits, groupsize = self._get_gptq_params()

            use_gptq_cuda = bits == 4

            if self.process_group.size() > 1:
                g_idx = self.get_tensor(f"{prefix}.g_idx")
                if g_idx is not None:
                    if (
                        not torch.equal(
                            g_idx.cpu(),
                            torch.tensor(
                                [i // groupsize for i in range(g_idx.shape[0])],
                                dtype=torch.int32,
                            ),
                        )
                        and not (g_idx == 0).all()
                    ):
                        # Exllama implementation does not support row tensor parallelism with act-order, as
                        # it would require to reorder input activations that are split unto several GPUs
                        use_gptq_cuda = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            from text_generation_server.utils.layers import HAS_GPTQ_CUDA

            if use_gptq_cuda:
                use_gptq_cuda = HAS_GPTQ_CUDA
                if self.process_group.rank == 0:
                    if use_gptq_cuda:
                        logger.info(f"Using GPTQ cuda kernels for row {prefix}")
                    else:
                        logger.warning(
                            "GPTQ cuda kernels (which are faster) could have been used, but are disabled via the DISABLE_EXLLAMA env var,"
                            " or not currently installed, try using BUILD_EXTENSIONS=True"
                        )

            if use_gptq_cuda:
                if groupsize >= 0:
                    # Exllama reorders the weights in advance and the activations on the fly, thus
                    # the scales and zero-points do not need to be reordered.
                    qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)
                else:
                    qzeros = self.get_tensor(f"{prefix}.qzeros")
                    scales = self.get_tensor(f"{prefix}.scales")

                # For tp > 1, at this point we know we do not use act-order
                if self.process_group.size() == 1:
                    g_idx = self.get_tensor(f"{prefix}.g_idx")
                else:
                    g_idx = None
            else:
                # The triton kernel reorders the scales/zero points instead of the weight/activation.
                # Thus, each rank needs the full qzeros/scales.

                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_gptq_cuda)
        else:
            weight = self._fb.get_sharded(
                self._get_alias(f"{prefix}.weight"),
                1,
                device=self.device,
                dtype=self.dtype,
            ).get_raw()
        return weight

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except RuntimeError as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception:
                raise e
        return bits, groupsize

    def _set_gptq_params(self, model_config: Any, model_path: str):
        # Get quantization config from model's configuration
        # or else look for quantize_config.json in the model dir
        config = model_config.to_dict()
        quantize_config = config.get("quantization_config")
        if quantize_config is None:
            filename = os.path.join(model_path, QUANTIZE_CONFIG_FILENAME)
            if not os.path.exists(filename):
                return
            with open(filename, "r") as f:
                quantize_config = json.load(f)

        self.gptq_bits = quantize_config["bits"]
        self.gptq_groupsize = quantize_config["group_size"]
