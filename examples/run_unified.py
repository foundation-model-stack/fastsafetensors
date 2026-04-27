# SPDX-License-Identifier: Apache-2.0

"""Example: Using UnifiedLoader with automatic configuration.

UnifiedLoader automatically creates the appropriate loader based on
configuration discovered from:
  1. FASTSAFETENSORS_CONFIG environment variable -> config file
  2. ./fastsafetensors.yaml (default path, if it exists)
  3. Built-in defaults (loader="base", copier_type="gds")

No manual loader creation is needed. The constructor itself emits a
``logger.info`` summary of the effective configuration, so callers do
not need to print it manually.
"""

import argparse
import logging

from fastsafetensors import SingleGroup, UnifiedLoader


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="UnifiedLoader example")
    parser.add_argument("files", nargs="+", help="safetensors file paths")
    parser.add_argument("--device", default="cpu", help="target device (default: cpu)")
    args = parser.parse_args()

    pg = SingleGroup()

    # --- Way 1: Pure defaults ---
    # Uses loader="base", copier_type="gds" by default.
    # No config file needed.
    print("=== Way 1: Default config ===")
    loader = UnifiedLoader(pg, args.files, device=args.device)
    for key, tensor in loader.iterate_weights():
        print(f"  {key}: shape={tensor.shape}")
    loader.close()

    # --- Way 2: Config file in working directory ---
    # Place a fastsafetensors.yaml in the working directory:
    #
    #   loader: "threefs"
    #   threefs:
    #     mount_point: "/mnt/3fs"
    #
    # Then just run:
    #   loader = UnifiedLoader(pg, args.files, device=args.device)
    print("\n=== Way 2: Config file (auto-discovered from ./fastsafetensors.yaml) ===")
    print("  (Place fastsafetensors.yaml in your working directory)")

    # --- Way 3: Environment variable ---
    # export FASTSAFETENSORS_CONFIG=/path/to/your/config.yaml
    # Then just run:
    #   loader = UnifiedLoader(pg, args.files, device=args.device)
    print("\n=== Way 3: Environment variable ===")
    print("  export FASTSAFETENSORS_CONFIG=/path/to/config.yaml")


if __name__ == "__main__":
    main()
