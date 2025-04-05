# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest
import vllm
from vllm.config import LoadFormat

def test_vllm_no_fastsafetensors(fstcpp_log):
    _ = vllm.LLM(model="ibm-granite/granite-3.0-8b-instruct")

def test_vllm_fastsafetensors(fstcpp_log):
    _ = vllm.LLM(model="ibm-granite/granite-3.0-8b-instruct", load_format=LoadFormat.FASTSAFETENSORS)
