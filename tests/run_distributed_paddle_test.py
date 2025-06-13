import os
import sys

import pytest

if __name__ == "__main__":
    # There are 4 commands before this test
    # GPU ditributed need at least 2 GPU
    rank = int(os.getenv("PADDLE_TRAINER_ID", -100)) + 4
    os.environ["COVERAGE_FILE"] = f".coverage_{rank}"
    pytest_args = sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
