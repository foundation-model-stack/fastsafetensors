# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest
import vllm
from vllm.config import LoadFormat


def test_vllm_no_fastsafetensors(fstcpp_log):
    _ = vllm.LLM(model="ibm-granite/granite-3.3-2b-instruct", max_model_len=16)


def test_vllm_fastsafetensors(fstcpp_log):
    _ = vllm.LLM(
        model="ibm-granite/granite-3.3-2b-instruct",
        load_format=LoadFormat.FASTSAFETENSORS,
        max_model_len=16,
    )


def test_deepseek_r1(fstcpp_log):
    _ = vllm.LLM(
        model="silence09/DeepSeek-R1-Small-2layers",
        load_format=LoadFormat.FASTSAFETENSORS,
        max_model_len=16,
    )
