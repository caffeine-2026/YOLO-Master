"""Validate the dependency-free WeChat Mini Program source tree."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PACKAGE_BYTES = 2 * 1024 * 1024
IGNORED_DIRECTORIES = {"dist", "tests", "tools", "node_modules", "miniprogram_npm", "__pycache__"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    app = load_json(ROOT / "app.json")
    project = load_json(ROOT / "project.config.json")
    assert project["compileType"] == "miniprogram"
    assert project["miniprogramRoot"] == "./"
    pages = app.get("pages", [])
    assert pages, "app.json must declare pages"
    for page in pages:
        for suffix in (".js", ".json", ".wxml", ".wxss"):
            path = ROOT / f"{page}{suffix}"
            assert path.is_file(), f"Missing page artifact: {path.relative_to(ROOT)}"
    package_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(ROOT).parts)
    ]
    package_bytes = sum(path.stat().st_size for path in package_files)
    assert package_bytes < MAX_PACKAGE_BYTES, f"Mini Program source package is too large: {package_bytes} bytes"
    bundled_onnx = [path for path in ROOT.rglob("*.onnx") if "dist" not in path.relative_to(ROOT).parts]
    assert not bundled_onnx, "ONNX artifacts must remain outside the code package"
    print(
        json.dumps(
            {
                "status": "PASS",
                "pages": pages,
                "source_files": len(package_files),
                "source_package_bytes": package_bytes,
                "limit_bytes": MAX_PACKAGE_BYTES,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
