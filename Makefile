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

.PHONY: unittest
unittest:
	FST_DIR=$$(python3 -c "import os; os.chdir('/tmp'); import fastsafetensors; print(os.path.dirname(fastsafetensors.__file__))") && \
	COVERAGE_FILE=.coverage_0 pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py && \
	COVERAGE_FILE=.coverage_1 CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/test_fastsafetensors.py && \
	COVERAGE_FILE=.coverage_2 torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=0 --no-python pytest -s --cov=${FST_DIR} tests/test_multi.py > /tmp/2.log 2>&1 & \
	COVERAGE_FILE=.coverage_3 torchrun --nnodes=2 --master_addr=0.0.0.0 --master_port=1234 --node_rank=1 --no-python pytest -s --cov=${FST_DIR} tests/test_multi.py > /tmp/3.log 2>&1 && \
	coverage combine .coverage_0 .coverage_1 .coverage_2 .coverage_3 && \
	coverage html

.PHONY: integrationtest
integrationtest:
	cd tests && COVERAGE_FILE=.coverage pytest -s test_vllm.py

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
	black --check .
	isort --check-only .
	flake8 . --select=E9,F63,F7,F82
	mypy . --ignore-missing-imports

.PHONY: clean
clean:
	rm -rf dist build fastsafetensors.egg-info