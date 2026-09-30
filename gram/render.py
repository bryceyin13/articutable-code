from __future__ import annotations

import colorsys
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np


def primitive_color(i: int) -> tuple[int, int, int]:
    if not 0 <= i < 256:
        raise ValueError("primitive ID must be in [0, 255]")
    hue = (i * 0.618033988749895) % 1.0
    saturation = 0.72 if i % 2 == 0 else 0.92
    value = 0.95 if (i // 2) % 2 == 0 else 0.78
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def _triangle_keys(triangles: np.ndarray, tolerance: float) -> list[bytes]:
    triangles = np.asarray(triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("face triangles must have shape (N, 3, 3)")
    quantized = np.rint(triangles / tolerance).astype(np.int64)
    order = np.lexsort(
        (quantized[:, :, 2], quantized[:, :, 1], quantized[:, :, 0]), axis=1,
    )
    ordered = np.take_along_axis(quantized, order[:, :, None], axis=1)
    return [row.tobytes() for row in ordered]


def remap_face_labels(reference_triangles: np.ndarray, labels: np.ndarray,
                      imported_triangles: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    labels = np.asarray(labels)
    reference_triangles = np.asarray(reference_triangles)
    imported_triangles = np.asarray(imported_triangles)
    if labels.shape != (len(reference_triangles),):
        raise ValueError("face labels do not match reference triangles")
    dropped = len(reference_triangles) - len(imported_triangles)
    limit = min(100, max(1, int(np.ceil(len(reference_triangles) * 5e-4))))
    if dropped < 0 or dropped > limit:
        raise RuntimeError(
            f"Blender changed face count by {dropped} faces; allowed dropped-face limit is {limit}"
        )
    by_triangle = {}
    for key, label in zip(_triangle_keys(reference_triangles, tolerance), labels):
        label = int(label)
        if key in by_triangle and by_triangle[key] != label:
            raise RuntimeError("duplicate reference triangles have conflicting face labels")
        by_triangle[key] = label
    imported_keys = _triangle_keys(imported_triangles, tolerance)
    missing = [key for key in imported_keys if key not in by_triangle]
    if missing:
        raise RuntimeError(f"could not map {len(missing)} Blender faces to segmentation labels")
    return np.asarray([by_triangle[key] for key in imported_keys], dtype=labels.dtype)


def build_blender_command(blender: Path, worker: Path, mesh_path: Path, labels_path: Path,
                          output_dir: Path, source_mesh: Path | None = None,
                          save_blend: bool = True) -> list[str]:
    resolve = lambda path: str(path.resolve())
    command = [resolve(blender), "--background", "--python", resolve(worker), "--python-exit-code", "1",
            "--", "--mesh", resolve(mesh_path),
            "--face-ids", resolve(labels_path),
            "--face-triangles", resolve(output_dir / ".face_triangles.npy"),
            "--output-dir", resolve(output_dir)]
    if source_mesh is not None:
        command.extend(["--source-mesh", resolve(source_mesh)])
    if not save_blend:
        command.append("--no-save-blend")
    return command


def _persistent_blender_request(server_socket: str, request: dict) -> dict:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(server_socket)
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        with connection.makefile("r", encoding="utf-8") as response_file:
            response = response_file.readline()
    finally:
        connection.close()
    if not response:
        raise RuntimeError("persistent Blender server closed without a response")
    return json.loads(response)


def validate_rendered_views(view_paths: Sequence[Path]) -> None:
    from PIL import Image

    hashes = []
    for path in view_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"rendered view is missing or empty: {path}")
        try:
            rgba = np.asarray(Image.open(path).convert("RGBA"))
        except Exception as exc:
            raise RuntimeError(f"rendered view is not a readable image: {path}") from exc
        if rgba.ndim != 3 or rgba.shape[0] == 0 or rgba.shape[1] == 0:
            raise RuntimeError(f"rendered view has invalid dimensions: {path}")
        visible = rgba[..., 3] > 0
        visible_count = int(visible.sum())
        if visible_count == 0:
            raise RuntimeError(f"rendered view is fully transparent: {path}")
        _, counts = np.unique(rgba[..., :3][visible], axis=0, return_counts=True)
        if len(counts) < 2:
            raise RuntimeError(f"rendered view contains too few visible colors: {path}")
        foreground_count = visible_count - int(counts.max())
        if foreground_count < max(16, round(visible_count * 0.001)):
            raise RuntimeError(f"rendered view contains too few non-background pixels: {path}")
        hashes.append(hashlib.sha256(path.read_bytes()).digest())
    if len(view_paths) > 1 and len(set(hashes)) == 1:
        raise RuntimeError("rendered views are byte-identical")


def render_primitive_views(mesh_path: Path, labels_path: Path, output_dir: Path, blender: Path,
                           worker: Path | None = None, source_mesh: Path | None = None,
                           save_blend: bool = True) -> Path:
    for path, label in ((mesh_path, "working mesh"), (labels_path, "face labels"), (blender, "Blender")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    worker = worker or Path(__file__).with_name("render_primitives_blender.py")
    output_dir.mkdir(parents=True, exist_ok=True)
    view_paths = [output_dir / f"view_{name}.png"
                  for name in ("front", "right", "back", "left", "top")]
    rgb_paths = [output_dir / f"view_{name}_rgb.png"
                 for name in ("front", "right", "back", "left", "top")]
    manifest = output_dir / "views.json"
    face_triangles = output_dir / ".face_triangles.npy"
    all_inspection_paths = [output_dir / "primitives.glb", output_dir / "primitives.blend"]
    inspection_paths = [output_dir / "primitives.glb"]
    if save_blend:
        inspection_paths.append(output_dir / "primitives.blend")
    for stale in (*view_paths, *rgb_paths, manifest, face_triangles, *all_inspection_paths):
        stale.unlink(missing_ok=True)
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    labels = np.load(labels_path)
    if labels.shape != (len(mesh.faces),):
        raise RuntimeError(f"face labels {labels.shape} do not match {len(mesh.faces)} mesh faces")
    np.save(face_triangles, np.asarray(mesh.triangles))
    server_socket = os.environ.get("GRAM_BLENDER_SOCKET")
    try:
        if server_socket:
            response = _persistent_blender_request(server_socket, {
                "mesh": str(mesh_path.resolve()),
                "face_ids": str(labels_path.resolve()),
                "face_triangles": str(face_triangles.resolve()),
                "output_dir": str(output_dir.resolve()),
                "source_mesh": str(source_mesh.resolve()) if source_mesh else None,
                "save_blend": save_blend,
            })
            (output_dir / "blender.log").write_text(
                f"persistent Blender server: {server_socket}\n\n"
                f"STDOUT\n{response.get('stdout', '')}\n"
                f"STDERR\n{response.get('stderr', '')}\n",
                encoding="utf-8",
            )
            if response.get("ok") is not True:
                raise RuntimeError(response.get("error") or "persistent Blender failed")
        else:
            command = build_blender_command(
                blender, worker, mesh_path, labels_path, output_dir, source_mesh,
                save_blend,
            )
            result = subprocess.run(command, text=True, capture_output=True)
            (output_dir / "blender.log").write_text(
                result.stdout + "\n" + result.stderr, encoding="utf-8",
            )
            if result.returncode:
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stdout, result.stderr,
                )
    finally:
        face_triangles.unlink(missing_ok=True)
    if not manifest.exists():
        raise RuntimeError("Blender completed without writing views.json")
    for path in inspection_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Blender completed without writing primitive inspection file: {path}")
    validate_rendered_views(view_paths)
    if source_mesh is not None:
        validate_rendered_views(rgb_paths)
    return manifest
