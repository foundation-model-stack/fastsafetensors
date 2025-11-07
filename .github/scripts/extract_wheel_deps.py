#!/usr/bin/env python3
"""Extract dependencies from pyproject.toml for a specific version."""

import re
import sys
from pathlib import Path


def parse_pyproject_toml(pyproject_path):
    """Parse dependencies from pyproject.toml, excluding test dependencies."""
    dependencies = []

    try:
        with open(pyproject_path, "r") as f:
            content = f.read()

        # Find [project.dependencies] section
        in_dependencies = False
        in_optional_dependencies = False
        bracket_count = 0

        for line in content.split("\n"):
            line_stripped = line.strip()

            # Check if entering dependencies section
            if line_stripped == "[project.dependencies]" or line_stripped.startswith(
                "dependencies = ["
            ):
                in_dependencies = True
                if "[" in line:
                    bracket_count = line.count("[") - line.count("]")
                continue

            # Check if entering optional dependencies (skip these)
            if (
                "[project.optional-dependencies]" in line_stripped
                or "[tool.poetry.group" in line_stripped
            ):
                in_dependencies = False
                in_optional_dependencies = True
                continue

            # Exit sections when encountering new section header
            if line_stripped.startswith("[") and line_stripped.endswith("]"):
                in_dependencies = False
                in_optional_dependencies = False
                bracket_count = 0
                continue

            # Skip if in optional dependencies
            if in_optional_dependencies:
                continue

            # Parse dependency lines
            if in_dependencies:
                # Track bracket balance for multiline arrays
                bracket_count += line.count("[") - line.count("]")

                # Extract dependency from quoted string
                match = re.search(r'["\']([^"\']+)["\']', line)
                if match:
                    dep = match.group(1).strip()
                    # Skip comments and empty lines
                    if dep and not dep.startswith("#"):
                        # Remove any trailing commas
                        dep = dep.rstrip(",").strip()
                        dependencies.append(dep)

                # Check if array is closed
                if bracket_count == 0:
                    in_dependencies = False

    except Exception as e:
        print(f"Error reading {pyproject_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return dependencies


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: extract_wheel_deps.py <pyproject_path> <output_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    pyproject_path = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting dependencies from: {pyproject_path}", file=sys.stderr)

    deps = parse_pyproject_toml(pyproject_path)

    # Write dependencies to output file
    with open(output_file, "w") as f:
        for dep in deps:
            f.write(dep + "\n")

    print(f"Extracted {len(deps)} dependencies to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
