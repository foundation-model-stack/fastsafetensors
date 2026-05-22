Development
===========

This project requires all commits to comply with the Developer Certificate of Origin (DCO). We can only accept contributions whose commits include a valid
`Signed-off-by` line.

To sign off a commit, use:

```bash
git commit -s
```

Each commit in a pull request must include a sign-off line such as:

```
Signed-off-by: Your Name <your.email@example.com>
```

# Tests

This repository has CI with CPU-only mode. They automatically run when you raise a PR. We only accept changes that can pass these tests and lint checks with DCO.

You can also use Makefile on your local environment.

```
make unittest
make unittest-parallel
make vllm
```

# Pre-commit Hooks

Our CI workflow checks code formatting and linting with Python 3.13. Therefore, we recommend testing your code with Python 3.13 and running the following pre-commit hooks before contributing your code.

To set up:

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

Now, every time you commit, the following checks will run automatically:
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Basic linting (syntax errors, undefined names)
- **mypy**: Type checking
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with a newline
- **check-yaml**: Validate YAML files
- **check-toml**: Validate TOML files
- **check-merge-conflict**: Detect merge conflict markers
- **debug-statements**: Detect debug statements

To manually run pre-commit on all files:
```bash
pre-commit run --all-files
```

To skip pre-commit hooks (not recommended):
```bash
git commit --no-verify
```

# Build & install

## Build & install from GitHub Source

```bash
pip install git+https://github.com/foundation-model-stack/fastsafetensors.git
```

## Build & install from source

```bash
pip install .
```
