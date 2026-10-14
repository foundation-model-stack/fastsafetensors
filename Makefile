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

FST_DIR := $(shell python3 -c "import os; import fastsafetensors; print(os.path.dirname(fastsafetensors.__file__))")

unittest:
	@TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_unit pytest -s --cov=$(FST_DIR) tests/unit --ignore=tests/unit/test_multi.py --ignore=tests/unit/test_subgroup.py --ignore=tests/unit/test_broadcast_runs.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_cpu CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/unit --ignore=tests/unit/test_multi.py --ignore=tests/unit/test_subgroup.py --ignore=tests/unit/test_broadcast_runs.py

test-vllm:
	@FST_DIR=$(FST_DIR); \
	cd tests && TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_2 pytest -s --cov=$(FST_DIR) -s vllm/test_vllm.py

unittest-parallel:
	TEST_FASTSAFETENSORS_FRAMEWORK=torch COVERAGE_FILE=.coverage_single pytest -s --cov=$(FST_DIR) tests/unit/test_multi.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=torch torchrun --standalone --nproc_per_node=4 --no-python bash -c 'COVERAGE_FILE=.coverage_dist_$$RANK pytest -s --cov=$(FST_DIR) tests/unit/test_multi.py tests/unit/test_subgroup.py tests/unit/test_broadcast_runs.py'

unittest-paddle:
	@FST_DIR=$(FST_DIR); \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_8 CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/unit/test_fastsafetensors.py && \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_paddle_regressions CUDA_VISIBLE_DEVICES="" pytest -s --cov=$(FST_DIR) tests/unit/test_chunk_loading.py tests/unit/test_metadata_validation.py tests/unit/test_pipeline_errors.py tests/unit/test_runtime_hint.py tests/unit/test_fgds.py && \
	{ \
		pids=""; \
		for rank in 0 1; do \
			TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_paddle_dist_$$rank CUDA_VISIBLE_DEVICES="" WORLD_SIZE=2 python3 -m paddle.distributed.launch --nnodes 2 --master 127.0.0.1:1234 --rank $$rank tests/unit/test_multi.py --cov=$(FST_DIR) -s tests/unit/test_multi.py > /tmp/paddle-dist-$$rank.log 2>&1 & \
			pids="$$pids $$!"; \
		done; \
		status=0; \
		for pid in $$pids; do wait $$pid || status=1; done; \
		exit $$status; \
	}

unittest-paddle-gpu:
	@FST_DIR=$(FST_DIR); \
	TEST_FASTSAFETENSORS_FRAMEWORK=paddle COVERAGE_FILE=.coverage_11 pytest -s --cov=$(FST_DIR) tests/unit/test_fastsafetensors.py

htmlcov:
	coverage combine .coverage_* $(wildcard tests/.coverage_*) && \
	coverage html

.PHONY: builder
builder: Dockerfile.build
	$(CONCMD) build -t fastsafetensors-builder:latest - < Dockerfile.build

dist: builder
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.13 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.12 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.11 setup.py bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64
	$(CONCMD) run -u `id -u` -w /fastsafetensors --rm -v $(CURDIR):/fastsafetensors -e CC=c++ -it fastsafetensors-builder:latest python3.10 setup.py sdist bdist_wheel --python-tag=py3 -p manylinux_2_34_x86_64

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
