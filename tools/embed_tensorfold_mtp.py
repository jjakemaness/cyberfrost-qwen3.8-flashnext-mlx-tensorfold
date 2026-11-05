#!/usr/bin/env python3
"""Embed an MLX-VLM Qwen4Exp MTP sidecar into a TensorFold checkpoint.

The resulting checkpoint keeps the base shards unchanged and adds one
``model-mtp.safetensors`` file whose 76 tensor names are rooted at
``language_model.mtp.``.  The TensorFold weight index is then updated to
refer to that file.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path


PREFIX = "language_model.mtp."
DEST_NAME = "model-mtp.safetensors"


def read_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path} has no safetensors header")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"{path} has an invalid safetensors header")
    return header, header_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("mtp", type=Path)
    args = parser.parse_args()

    base = args.base.resolve(strict=True)
    mtp = args.mtp.resolve(strict=True)
    index_path = base / "model.safetensors.index.json"
    source = mtp / "model.safetensors"
    destination = base / DEST_NAME
    temp_destination = base / f".{DEST_NAME}.tmp"
    temp_index = base / ".model.safetensors.index.json.tmp"

    if destination.exists() or temp_destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if not index_path.is_file() or not source.is_file():
        raise FileNotFoundError("base index or MTP sidecar is missing")

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("base index has no weight_map")

    header, old_size = read_header(source)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if len(tensors) != 76:
        raise ValueError(f"expected 76 MTP tensors, found {len(tensors)}")
    if any(name.startswith(PREFIX) for name in tensors):
        raise ValueError("MTP sidecar already has TensorFold names")

    renamed = {PREFIX + name: value for name, value in tensors.items()}
    if set(renamed).intersection(weight_map):
        raise ValueError("base index already contains an MTP tensor")
    new_header: dict[str, object] = {"__metadata__": header.get("__metadata__", {})}
    new_header.update(renamed)
    encoded = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    with source.open("rb") as src, temp_destination.open("xb") as dst:
        src.seek(8 + old_size)
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        while chunk := src.read(16 * 1024 * 1024):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temp_destination, destination)

    weight_map.update({name: DEST_NAME for name in renamed})
    metadata = index.setdefault("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        metadata["total_size"] += destination.stat().st_size
    with temp_index.open("x") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_index, index_path)

    verified_header, _ = read_header(destination)
    if set(verified_header) - {"__metadata__"} != set(renamed):
        raise ValueError("written MTP header does not match the renamed tensor set")
    print(f"embedded {len(renamed)} MTP tensors in {destination}")


if __name__ == "__main__":
    main()
