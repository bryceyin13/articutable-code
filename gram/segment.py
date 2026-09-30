from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np

from .artifacts import PrimitiveManifest, read_json


def remap_labels_by_area(labels: np.ndarray, face_areas: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape != np.asarray(face_areas).shape:
        raise ValueError("labels and face areas must be aligned 1-D arrays")
    old_ids = sorted((int(x) for x in np.unique(labels)), key=lambda x: (-float(face_areas[labels == x].sum()), x))
    mapping = {old: new for new, old in enumerate(old_ids)}
    return np.array([mapping[int(x)] for x in labels], dtype=np.int32), mapping


def build_p3sam_command(*, python: Path, worker: Path, checkout: Path, checkpoint: Path,
                        sonata_dir: Path, input_glb: Path, output_dir: Path,
                        seed: int) -> list[str]:
    return [str(python), str(worker), "--checkout", str(checkout), "--checkpoint", str(checkpoint),
            "--sonata-dir", str(sonata_dir),
            "--input", str(input_glb), "--output-dir", str(output_dir), "--seed", str(seed)]


def run_p3sam(input_glb: Path, output_dir: Path, python: Path, checkout: Path,
              checkpoint: Path, sonata_dir: Path, seed: int,
              worker: Path | None = None) -> PrimitiveManifest:
    for path, label in ((input_glb, "input GLB"), (python, "P3-SAM Python"),
                        (checkout, "P3-SAM checkout"), (checkpoint, "P3-SAM checkpoint")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    worker = worker or Path(__file__).with_name("p3sam_worker.py")
    output_dir.mkdir(parents=True, exist_ok=True)
    server_socket = os.environ.get("GRAM_P3SAM_SOCKET")
    if server_socket:
        request = {
            "input": str(input_glb.resolve()),
            "output_dir": str(output_dir.resolve()),
            "seed": seed,
        }
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(server_socket)
            connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
            with connection.makefile("r", encoding="utf-8") as response_file:
                response = json.loads(response_file.readline())
        finally:
            connection.close()
        (output_dir / "p3sam.log").write_text(
            f"persistent P3-SAM server: {server_socket}\n\n"
            f"STDOUT\n{response.get('stdout', '')}\n"
            f"STDERR\n{response.get('stderr', '')}",
            encoding="utf-8",
        )
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error") or "persistent P3-SAM failed")
        return PrimitiveManifest.from_dict(read_json(output_dir / "manifest.json"))
    command = build_p3sam_command(python=python, worker=worker, checkout=checkout, checkpoint=checkpoint,
                                  sonata_dir=sonata_dir, input_glb=input_glb,
                                  output_dir=output_dir, seed=seed)
    result = subprocess.run(command, text=True, capture_output=True, cwd=checkout / "demo")
    (output_dir / "p3sam.log").write_text(
        f"$ {' '.join(command)}\n\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr}", encoding="utf-8")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return PrimitiveManifest.from_dict(read_json(output_dir / "manifest.json"))
