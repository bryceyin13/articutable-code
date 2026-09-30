#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from gram.p3sam_worker import transfer_face_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--saved-primitives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.saved_primitives / "manifest.json").read_text(encoding="utf-8")
    )
    source = trimesh.load(args.input, force="mesh", process=False)
    frame = manifest["frame"]
    source.vertices = (
        source.vertices - np.asarray(frame["center"])
    ) * float(frame["scale"])
    segmented = trimesh.load(
        args.saved_primitives / manifest["mesh_path"],
        force="mesh",
        process=False,
    )
    saved_labels = np.load(args.saved_primitives / manifest["face_ids_path"])
    labels, report = transfer_face_labels(source, segmented, saved_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.export(args.output_dir / "working_mesh.glb")
    np.save(args.output_dir / "face_primitive_ids.npy", labels)
    manifest["mesh_path"] = "working_mesh.glb"
    manifest["face_ids_path"] = "face_primitive_ids.npy"
    manifest["texture_transfer"] = report
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    log = args.saved_primitives / "p3sam.log"
    if log.is_file():
        shutil.copy2(log, args.output_dir / "p3sam.log")


if __name__ == "__main__":
    main()
