# SPDX-License-Identifier: Apache-2.0

"""Bound failure-path tests so a stranded producer cannot hang pytest."""

import os
import subprocess
import sys

import pytest

_FAILURE_PROBE = """
import sys
import threading
from contextlib import closing

from fastsafetensors import ParallelLoader, ParallelSaver
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.st_types import Device, DType

framework_name, directory, stage, queue = sys.argv[1:]
fw = get_framework_op(framework_name)
expected = fw.randn([4], Device(), DType.F32)
paths = ParallelSaver(num_shards=2, framework=framework_name).save(
    {'a': expected.get_raw(), 'b': expected.get_raw()}, directory + '/model'
)
before_threads = set(threading.enumerate())
before_memory = fw.get_mem_used()
loader = ParallelLoader(
    None, paths, device='cpu', framework=framework_name, nogds=True,
    queue_size=int(queue), use_tqdm_on_load=False,
)
copiers, buffers = [], []
construct = loader.loader.copier_constructor
def record_copier(*args):
    copier = construct(*args)
    copiers.append(copier)
    return copier
loader.loader.copier_constructor = record_copier
copy = loader.loader.copy_files_to_device
def record_buffer(*args, **kwargs):
    buffer = copy(*args, **kwargs)
    buffers.append(buffer)
    return buffer
loader.loader.copy_files_to_device = record_buffer
method = 'add_filenames' if stage == 'add' else 'copy_files_to_device'
original = getattr(loader.loader, method)
calls = 0
def fail_second(*args, **kwargs):
    global calls
    calls += 1
    if calls == 2:
        raise RuntimeError('injected producer failure')
    return original(*args, **kwargs)
setattr(loader.loader, method, fail_second)
outputs = {}
try:
    with closing(loader.iterate_weights()) as weights:
        for name, tensor in weights:
            outputs[name] = tensor
except RuntimeError as error:
    assert str(error) == 'injected producer failure', str(error)
else:
    raise AssertionError('producer failure was not delivered to consumer')
finally:
    loader.close()
    loader.close()
assert calls == 2
assert outputs.keys() == {'a'}
assert fw.is_equal(fw.wrap_tensor(outputs['a']), expected.get_raw())
assert buffers and all(not buffer.rank_loaders for buffer in buffers)
assert copiers and all(copier.fd == -1 for copier in copiers)
assert fw.get_mem_used() == before_memory
assert not [t for t in threading.enumerate() if t not in before_threads and not t.daemon]
"""


@pytest.mark.parametrize("queue_size", [-1, 0, 2])
@pytest.mark.parametrize("stage", ["add", "copy"])
def test_producer_failure_propagates_and_releases_resources(
    tmp_path, framework, stage, queue_size
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _FAILURE_PROBE,
            framework.get_name(),
            str(tmp_path),
            stage,
            str(queue_size),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
