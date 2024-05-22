# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

CUDA_HOME ?= /usr/local/cuda
TORCH_PATH ?= auto

.PHONY: install
install:
	pip install setuptools
	CUDA_HOME=$(CUDA_HOME) TORCH_PATH=$(TORCH_PATH) pip install . --no-cache-dir --no-build-isolation

.PHONY: install-test
install-test:
	FST_TEST=1 TORCH_PATH=$(TORCH_PATH) pip install .[test] --no-cache-dir --no-build-isolation

.PHONY: unittest
unittest:
	pytest -s --cov=fastsafetensors --cov-report=html

.PHONY: builder
builder: Dockerfile.build
	docker build -t fastsafetensors-builder:latest - < Dockerfile.build

dist: builder
	docker run -u `id -u` --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.11 -m build --no-isolation /fastsafetensors

perf/dist:
	cd perf && python3 -m build