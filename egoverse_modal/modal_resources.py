"""Shared, pinned Modal images, secrets, and persistent volume definitions."""

from __future__ import annotations

import modal

from .runtime import EGOVERSE_COMMIT, EGOVERSE_REPOSITORY, EGOVERSE_ROOT


data_volume = modal.Volume.from_name("egoverse-data", create_if_missing=True)

egoverse_cloud_secret = modal.Secret.from_name(
    "egoverse-cloud",
    required_keys=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "SECRETS_ARN",
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    ],
)

egoverse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "git-lfs",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        "python -m pip install --upgrade pip uv",
        f"mkdir -p {EGOVERSE_ROOT}",
        f"cd {EGOVERSE_ROOT} && git init",
        f"cd {EGOVERSE_ROOT} && git remote add origin {EGOVERSE_REPOSITORY}",
        f"cd {EGOVERSE_ROOT} && git fetch --depth 1 origin {EGOVERSE_COMMIT}",
        f"cd {EGOVERSE_ROOT} && git -c advice.detachedHead=false checkout --detach FETCH_HEAD",
        f"cd {EGOVERSE_ROOT} && git submodule update --init --recursive --depth 1",
        f"uv pip install --system --editable {EGOVERSE_ROOT}",
        "python -c \"import egomimic, torch; print('EgoVerse ready; torch', torch.__version__)\"",
    )
    .add_local_python_source("egoverse_modal", copy=True)
)
