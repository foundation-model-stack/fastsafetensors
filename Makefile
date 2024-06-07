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
	rm -rf dist build fastsafetensors.egg-info
	docker run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.10 setup.py sdist bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	docker run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.11 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64

.PHONY: upload-test
upload-test:
	python3 -m twine upload -u __token__ --repository testpypi dist/fastsafetensors-$(shell grep version pyproject.toml | sed -e 's/version = "\([0-9.]\+\)"/\1/g')*

.PHONY: upload
upload:
	python3 -m twine upload -u __token__ dist/fastsafetensors-$(shell grep version pyproject.toml | sed -e 's/version = "\([0-9.]\+\)"/\1/g')*

perf/dist:
	cd perf && python3 -m build
