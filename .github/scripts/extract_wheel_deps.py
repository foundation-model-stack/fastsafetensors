#!/usr/bin/env python3
"""Extract dependencies from wheel METADATA file."""

import sys
import zipfile
import re
from pathlib import Path


def extract_dependencies(wheel_path):
    """Extract Requires-Dist from wheel METADATA."""
    dependencies = []

    try:
        with zipfile.ZipFile(wheel_path, "r") as whl:
            # Find METADATA file
            metadata_files = [
                f for f in whl.namelist() if f.endswith(".dist-info/METADATA")
            ]

            if not metadata_files:
                print(f"Warning: No METADATA found in {wheel_path}", file=sys.stderr)
                return dependencies

            metadata_content = whl.read(metadata_files[0]).decode("utf-8")

            # Parse Requires-Dist lines
            for line in metadata_content.split("\n"):
                line = line.strip()
                if line.startswith("Requires-Dist:"):
                    # Extract dependency specification
                    dep = line.split(":", 1)[1].strip()
                    # Remove extras and environment markers
                    dep = re.split(r"\s*;\s*", dep)[0]
                    dep = re.split(r"\s*\[", dep)[0]
                    dependencies.append(dep)

    except Exception as e:
        print(f"Error reading {wheel_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return dependencies


def main():
    if len(sys.argv) != 3:
        print("Usage: extract_wheel_deps.py <wheel_dir> <output_file>", file=sys.stderr)
        sys.exit(1)

    wheel_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Find fastsafetensors wheels
    wheel_files = list(wheel_dir.glob("fastsafetensors-*.whl"))

    if not wheel_files:
        print("Error: No fastsafetensors wheels found", file=sys.stderr)
        sys.exit(1)

    # Use first wheel (all should have same dependencies)
    wheel_path = wheel_files[0]
    print(f"Extracting dependencies from: {wheel_path.name}", file=sys.stderr)

    deps = extract_dependencies(wheel_path)

    # Write dependencies to output file
    with open(output_file, "w") as f:
        for dep in deps:
            f.write(dep + "\n")

    print(f"Extracted {len(deps)} dependencies to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
