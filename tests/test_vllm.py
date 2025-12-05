# SPDX-License-Identifier: Apache-2.0

import pytest
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import vllm
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=16,
)

prompts = [
    "Explain the difference between CPU and GPU.",
    "Write a haiku about programming.",
]


def run_llm(llm):
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        print("=========")
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output.outputs[0].text}")


def test_vllm_no_fastsafetensors(fstcpp_log):
    run_llm(LLM(model="ibm-granite/granite-3.3-2b-instruct", max_model_len=16))


def test_vllm_fastsafetensors(fstcpp_log):
    run_llm(
        LLM(
            model="ibm-granite/granite-3.3-2b-instruct",
            load_format="fastsafetensors",
            max_model_len=16,
        )
    )


def test_deepseek_r1(fstcpp_log):
    run_llm(
        LLM(
            model="silence09/DeepSeek-R1-Small-2layers",
            load_format="fastsafetensors",
            max_model_len=16,
        )
    )
