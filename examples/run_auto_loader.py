# SPDX-License-Identifier: Apache-2.0

"""Example: Using AutoLoader with automatic configuration.

AutoLoader automatically creates the appropriate loader based on
configuration discovered from:
  1. FASTSAFETENSORS_CONFIG environment variable -> config file
  2. ./fastsafetensors.json (default path, if it exists)
  3. Built-in defaults (loader="base", copier_type="gds")

No manual loader creation is needed. The constructor itself emits a
``logger.info`` summary of the effective configuration, so callers do
not need to print it manually.
"""

import argparse
import logging

from fastsafetensors import AutoLoader, SingleGroup

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="AutoLoader example")
    parser.add_argument("files", nargs="+", help="safetensors file paths")
    parser.add_argument("--device", default="cpu", help="target device (default: cpu)")
    args = parser.parse_args()

    pg = SingleGroup()

    # --- Way 1: Pure defaults ---
    # Uses loader="base", copier_type="gds" by default.
    # No config file needed.
    logger.info("=== Way 1: Default config ===")
    loader = AutoLoader(pg, args.files, device=args.device)
    for key, tensor in loader.iterate_weights():
        logger.info("  %s: shape=%s", key, tensor.shape)
    loader.close()

    # --- Way 2: Config file in working directory ---
    # Place a fastsafetensors.json in the working directory:
    #
    #   {
    #     "loader": "3fs",
    #     "3fs": {
    #       "mount_point": "/mnt/3fs"
    #     }
    #   }
    #
    # Then just run:
    #   loader = AutoLoader(pg, args.files, device=args.device)
    logger.info(
        "=== Way 2: Config file (auto-discovered from ./fastsafetensors.json) ==="
    )
    logger.info("  (Place fastsafetensors.json in your working directory)")

    # --- Way 3: Environment variable ---
    # export FASTSAFETENSORS_CONFIG=/path/to/your/config.json
    # Then just run:
    #   loader = AutoLoader(pg, args.files, device=args.device)
    logger.info("=== Way 3: Environment variable ===")
    logger.info("  export FASTSAFETENSORS_CONFIG=/path/to/config.json")


if __name__ == "__main__":
    main()
