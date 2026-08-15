"""Fail safely before publishing empty or example credential files to Modal."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = {
    "egoverse": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "SECRETS_ARN",
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    ),
    "r2": (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    ),
}
PLACEHOLDER_PREFIXES = ("replace-with", "your-", "changeme", "example")


def parse_dotenv(path: str | Path) -> dict[str, str]:
    source = Path(path)
    values: dict[str, str] = {}
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def validate(path: str | Path, kind: str) -> list[str]:
    if kind not in REQUIRED:
        raise ValueError(f"unknown secret kind: {kind}")
    values = parse_dotenv(path)
    return [
        key
        for key in REQUIRED[kind]
        if not values.get(key)
        or values[key].strip().lower().startswith(PLACEHOLDER_PREFIXES)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(REQUIRED))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    invalid = validate(args.path, args.kind)
    if invalid:
        parser.error(
            "refusing to publish placeholder/empty fields: " + ", ".join(invalid)
        )
    print(f"secret preflight passed: {args.kind} ({len(REQUIRED[args.kind])} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
