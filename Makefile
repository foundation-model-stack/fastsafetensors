# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

PODMAN := $(shell podman -v 2> /dev/null)
CONCMD := docker
ifdef PODMAN
	CONCMD = podman
endif

.PHONY: install
install:
	pip install . --no-cache-dir --no-build-isolation

.PHONY: unittest unittest-parallel unittest-paddle unittest-paddle-gpu htmlcov

FST_DIR := $(shell python3 -c "import os; os.chdir('/tmp'); import fastsafetensors; print(os.path.dirname(fastsafetensors.__file__))")

unittest:
	@FST_DIR=$(FST_DIR); \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_0 pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_1 CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_2 pytest -s --cov=$(FST_DIR) -s tests/test_vllm.py

unittest-parallel:
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_3 torchrun --nnodes=4 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/3.log 2>&1 & \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_4 torchrun --nnodes=4 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/4.log 2>&1 & \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_5 torchrun --nnodes=4 --master_addr=0.0.0.0 --master_port=1234 --node_rank=2 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/5.log 2>&1 & \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_6 torchrun --nnodes=4 --master_addr=0.0.0.0 --master_port=1234 --node_rank=3 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/6.log 2>&1 && \
	wait && \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_7 torchrun --nnodes=1 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/7.log 2>&1 & \
	wait

unittest-paddle:
	@FST_DIR=$(FST_DIR); \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_8 CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_9 CUDA_VISIBLE_DEVICES="" WORLD_SIZE=2 python3 -m paddle.distributed.launch --nnodes 2 --master 127.0.0.1:1234 --rank 0 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/9.log 2>&1 & \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_10 CUDA_VISIBLE_DEVICES="" WORLD_SIZE=2 python3 -m paddle.distributed.launch --nnodes 2 --master 127.0.0.1:1234 --rank 1 tests/test_multi.py --cov=$(FST_DIR) -s tests/test_multi.py > /tmp/10.log 2>&1 && \
	wait

unittest-paddle-gpu:
	@FST_DIR=$(FST_DIR); \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_11 pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py

htmlcov:
	coverage combine .coverage_* && \
	coverage html

.PHONY: builder
builder: Dockerfile.build
	$(CONCMD) build -t fastsafetensors-builder:latest - < Dockerfile.build

dist: builder
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.13 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.12 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.11 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.10 setup.py sdist bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.9 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64

.PHONY: upload-test
upload-test:
	python3 -m twine upload -u __token__ --repository testpypi dist/fastsafetensors-$(shell grep version pyproject.toml | sed -e 's/version = "\([0-9.]\+\)"/\1/g')*

.PHONY: upload
upload:
	python3 -m twine upload -u __token__ dist/fastsafetensors-$(shell grep version pyproject.toml | sed -e 's/version = "\([0-9.]\+\)"/\1/g')*

perf/dist:
	cd perf && pip install .

.PHONY: format
format:
	black8 .
	isort .

.PHONY: lint
lint:
	black .
	isort .
	flake8 . --select=E9,F63,F7,F82
	mypy . --ignore-missing-imports

.PHONY: clean
clean:
	rm -rf dist build fastsafetensors.egg-info