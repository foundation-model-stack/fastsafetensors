#!/usr/bin/env python3
"""
Generate PyPI-compatible simple index from GitHub releases.
This script fetches all releases and creates separate indexes for CUDA and ROCm wheels.
"""

import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

def fetch_releases(repo_owner, repo_name):
    """Fetch all releases from GitHub API."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PyPI-Index-Generator"
    }

    # Add GitHub token if available (for higher rate limits)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    request = Request(url, headers=headers)

    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching releases: {e}", file=sys.stderr)
        sys.exit(1)

def categorize_backend(release_tag):
    """Determine backend (cuda/rocm) from release tag."""
    tag_lower = release_tag.lower()

    if "rocm" in tag_lower:
        return "rocm"
    elif "cuda" in tag_lower:
        return "cuda"
    else:
        # Default to cuda for untagged releases
        return "cuda"

def extract_wheels_by_backend(releases):
    """Extract wheel files from releases, categorized by backend."""
    wheels_by_backend = {
        "cuda": [],
        "rocm": []
    }

    for release in releases:
        backend = categorize_backend(release.get("tag_name", ""))

        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".whl"):
                wheels_by_backend[backend].append({
                    "name": name,
                    "url": asset.get("browser_download_url"),
                    "version": release.get("tag_name"),
                })

    return wheels_by_backend

def generate_root_index(output_dir, packages):
    """Generate the root simple index."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Simple Index</title>
</head>
<body>
    <h1>Simple Index</h1>
"""

    for package in sorted(packages):
        html += f'    <a href="{package}/">{package}</a><br/>\n'

    html += """</body>
</html>
"""

    output_path = output_dir / "index.html"
    output_path.write_text(html)
    print(f"Generated: {output_path}")

def generate_package_index(output_dir, package_name, wheels):
    """Generate package-specific index with all wheels."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Links for {package_name}</title>
</head>
<body>
    <h1>Links for {package_name}</h1>
"""

    # Sort wheels by version and Python version
    sorted_wheels = sorted(wheels, key=lambda w: (w["name"], w["version"]), reverse=True)

    for wheel in sorted_wheels:
        # Extract package name from wheel filename to ensure consistency
        wheel_name = wheel["name"]
        url = wheel["url"]
        html += f'    <a href="{url}#sha256=">{wheel_name}</a><br/>\n'

    html += """</body>
</html>
"""

    package_dir = output_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    output_path = package_dir / "index.html"
    output_path.write_text(html)
    print(f"Generated: {output_path}")

def generate_landing_page(base_dir, repo_name):
    """Generate a landing page for the PyPI index."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{repo_name} - PyPI Index</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .backend {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        code {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{repo_name} - PyPI Index</h1>
    <p>Choose the appropriate index URL based on your GPU backend:</p>

    <div class="backend">
        <h2>🔥 ROCm (AMD GPUs)</h2>
        <p>For AMD GPUs using ROCm:</p>
        <pre>pip install fastsafetensors --index-url https://embeddedllm.github.io/{repo_name}/rocm/simple/</pre>
    </div>

    <div class="backend">
        <h2>💚 CUDA (NVIDIA GPUs)</h2>
        <p>For NVIDIA GPUs using CUDA:</p>
        <pre>pip install fastsafetensors --index-url https://embeddedllm.github.io/{repo_name}/cuda/simple/</pre>
    </div>

    <h3>Version Specific Installation</h3>
    <pre>pip install fastsafetensors==0.1.15 --index-url https://embeddedllm.github.io/{repo_name}/rocm/simple/</pre>

    <h3>In requirements.txt</h3>
    <pre>--index-url https://embeddedllm.github.io/{repo_name}/rocm/simple/
fastsafetensors>=0.1.15</pre>

    <hr>
    <p><small>Direct access: <a href="rocm/simple/">ROCm Index</a> | <a href="cuda/simple/">CUDA Index</a></small></p>
</body>
</html>
"""

    output_path = base_dir / "index.html"
    output_path.write_text(html)
    print(f"Generated landing page: {output_path}")

def main():
    # Configuration
    repo_owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "EmbeddedLLM")
    repo_full = os.environ.get("GITHUB_REPOSITORY", "EmbeddedLLM/fastsafetensors-rocm")
    repo_name = repo_full.split("/")[-1]

    print(f"Fetching releases from {repo_owner}/{repo_name}...")
    releases = fetch_releases(repo_owner, repo_name)
    print(f"Found {len(releases)} releases")

    # Extract wheels categorized by backend
    wheels_by_backend = extract_wheels_by_backend(releases)

    total_wheels = sum(len(wheels) for wheels in wheels_by_backend.values())
    print(f"Found {total_wheels} total wheel files")
    print(f"  CUDA: {len(wheels_by_backend['cuda'])} wheels")
    print(f"  ROCm: {len(wheels_by_backend['rocm'])} wheels")

    if total_wheels == 0:
        print("Warning: No wheel files found in any release", file=sys.stderr)
        return

    # Generate indexes for each backend
    for backend, wheels in wheels_by_backend.items():
        if not wheels:
            print(f"Skipping {backend} index (no wheels found)")
            continue

        print(f"\nGenerating {backend.upper()} index...")
        output_dir = Path(f"pypi-index/{backend}/simple")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group wheels by package name
        packages = {}
        for wheel in wheels:
            # Extract package name from wheel filename (before first dash)
            package_name = wheel["name"].split("-")[0]
            if package_name not in packages:
                packages[package_name] = []
            packages[package_name].append(wheel)

        # Generate indexes
        generate_root_index(output_dir, packages.keys())

        for package_name, package_wheels in packages.items():
            generate_package_index(output_dir, package_name, package_wheels)

        print(f"  Generated {backend.upper()} index with {len(packages)} package(s)")

    # Generate landing page
    base_dir = Path("pypi-index")
    generate_landing_page(base_dir, repo_name)

    print(f"\n✓ Successfully generated indexes for all backends")
    print(f"  Total wheels: {total_wheels}")

if __name__ == "__main__":
    main()
