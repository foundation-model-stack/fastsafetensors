# SPDX-License-Identifier: Apache-2.0

import os

import pytest

pytest.importorskip("vllm", reason="vllm integration tests require vllm")
from vllm import LLM, SamplingParams  # type: ignore[attr-defined]

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


def test_vllm_no_fastsafetensors():
    run_llm(LLM(model="ibm-granite/granite-3.3-2b-instruct", max_model_len=16))


def test_vllm_fastsafetensors():
    run_llm(
        LLM(
            model="ibm-granite/granite-3.3-2b-instruct",
            load_format="fastsafetensors",
            max_model_len=16,
        )
    )


def test_vllm_fastsafetensors_tp():
    run_llm(
        LLM(
            model="ibm-granite/granite-3.3-2b-instruct",
            load_format="fastsafetensors",
            max_model_len=16,
            tensor_parallel_size=2,
        )
    )


def test_deepseek_r1():
    run_llm(
        LLM(
            model="silence09/DeepSeek-R1-Small-2layers",
            load_format="fastsafetensors",
            max_model_len=16,
            tensor_parallel_size=2,
        )
    )
