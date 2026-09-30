from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def face_keys(mesh, tolerance=1e-6):
    triangles = np.asarray(mesh.triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("mesh must contain triangular faces")
    quantized = np.rint(triangles / tolerance).astype(np.int64)
    vertex_dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    vertices = np.ascontiguousarray(quantized).reshape(-1, 3).view(vertex_dtype).reshape(-1, 3)
    vertices.sort(axis=1, order=("x", "y", "z"))
    rows = np.ascontiguousarray(vertices.view("<i8").reshape(-1, 9))
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def transfer_face_labels(source, segmented, labels, tolerance=1e-6):
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(segmented.faces),):
        raise ValueError("segmentation labels do not match segmented mesh faces")
    segmented_keys = face_keys(segmented, tolerance)
    order = np.argsort(segmented_keys)
    sorted_keys, sorted_labels = segmented_keys[order], labels[order]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    minimum = np.minimum.reduceat(sorted_labels, starts)
    maximum = np.maximum.reduceat(sorted_labels, starts)
    if np.any(minimum != maximum):
        raise ValueError("duplicate segmented triangles have conflicting labels")

    distances, positions = cKDTree(segmented.triangles_center).query(
        source.triangles_center, workers=-1,
    )
    source_triangles = np.asarray(source.triangles)
    segmented_triangles = np.asarray(segmented.triangles)[positions]
    source_order = np.lexsort(
        (source_triangles[:, :, 2], source_triangles[:, :, 1], source_triangles[:, :, 0]),
        axis=1,
    )
    segmented_order = np.lexsort(
        (segmented_triangles[:, :, 2], segmented_triangles[:, :, 1], segmented_triangles[:, :, 0]),
        axis=1,
    )
    source_sorted = np.take_along_axis(source_triangles, source_order[:, :, None], axis=1)
    segmented_sorted = np.take_along_axis(
        segmented_triangles, segmented_order[:, :, None], axis=1,
    )
    matched = np.max(np.abs(source_sorted - segmented_sorted), axis=(1, 2)) <= tolerance
    edges = np.roll(source_triangles, -1, axis=1) - source_triangles
    max_edge = np.linalg.norm(edges, axis=2).max(axis=1)
    degenerate = (
        np.asarray(source.area_faces) <= 10 * tolerance * tolerance
    ) & (distances <= np.maximum(max_edge, tolerance * 10))
    matched |= degenerate
    unmatched_count = int((~matched).sum())
    approximation_limit = min(
        100, max(1, int(np.ceil(len(source.faces) * 3e-4))),
    )
    approximate = (~matched) & (
        distances <= np.maximum(max_edge, tolerance * 10)
    )
    if (
        unmatched_count <= approximation_limit
        and int(approximate.sum()) == unmatched_count
    ):
        matched |= approximate
    if not np.all(matched):
        raise ValueError(f"could not map {int((~matched).sum())} source faces to segmented mesh")
    return labels[positions], {
        "source_faces": len(source.faces),
        "segmented_faces": len(segmented.faces),
        "matched_source_faces": int(matched.sum()),
        "degenerate_source_faces": int(degenerate.sum()),
        "approximated_source_faces": int(approximate.sum()),
        "approximation_limit": approximation_limit,
        "tolerance": tolerance,
    }


def remove_duplicate_faces(mesh, face_ids):
    _, first = np.unique(face_keys(mesh), return_index=True)
    keep = np.sort(first)
    if len(keep) != len(mesh.faces):
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        face_ids = face_ids[keep]
    return mesh, face_ids


def remove_degenerate_faces(mesh, tolerance=1e-6):
    keep = np.asarray(mesh.area_faces) > 50 * tolerance * tolerance
    removed = int((~keep).sum())
    if removed:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return removed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sonata-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server-socket", type=Path)
    args = parser.parse_args()
    if args.server_socket is None and not all((args.input, args.output_dir)):
        parser.error("--input and --output-dir are required outside server mode")
    if args.server_socket is not None and any((args.input, args.output_dir)):
        parser.error("--server-socket cannot be combined with --input or --output-dir")
    return args


def load_automask(checkout, checkpoint, sonata_dir):
    sys.path.insert(0, str(checkout / "demo"))
    from auto_mask import AutoMask
    from model import sonata

    sonata_dir.mkdir(parents=True, exist_ok=True)
    local_sonata = sonata_dir / "sonata.pth"
    if local_sonata.is_file():
        load_sonata = sonata.load
        sonata.load = lambda name, *args, **kwargs: load_sonata(
            str(local_sonata) if name == "sonata" else name, *args, **kwargs,
        )
    return AutoMask(str(checkpoint), sonata_dir=str(sonata_dir))


def segment(input_path, output_dir, seed, automask) -> None:
    random.seed(seed)
    np.random.seed(seed)
    source = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or not len(source.faces):
        raise ValueError("input must contain one non-empty mesh")
    center = tuple(map(float, source.bounding_box.centroid))
    diagonal = float(np.linalg.norm(source.bounds[1] - source.bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("input mesh has invalid bounds")
    scale = 1.0 / diagonal
    source.vertices = (source.vertices - center) * scale
    removed_degenerate_faces = remove_degenerate_faces(source)
    textured_source = source.copy()
    _, face_ids, working_mesh = automask.predict_aabb(
        source, seed=seed, is_parallel=False, post_process=True, prompt_bs=8
    )
    face_ids = np.asarray(face_ids, dtype=np.int32)
    if face_ids.shape != (len(working_mesh.faces),):
        raise RuntimeError(f"P3-SAM returned {face_ids.shape} labels for {len(working_mesh.faces)} faces")
    working_mesh, face_ids = remove_duplicate_faces(working_mesh, face_ids)
    areas = np.asarray(working_mesh.area_faces)
    old_ids = sorted((int(x) for x in np.unique(face_ids)), key=lambda x: (-float(areas[face_ids == x].sum()), x))
    mapping = {old: new for new, old in enumerate(old_ids)}
    face_ids = np.array([mapping[int(x)] for x in face_ids], dtype=np.int32)
    face_ids, texture_transfer = transfer_face_labels(textured_source, working_mesh, face_ids)
    working_mesh = textured_source
    output_dir.mkdir(parents=True, exist_ok=True)
    working_mesh.export(output_dir / "working_mesh.glb")
    np.save(output_dir / "face_primitive_ids.npy", face_ids)
    manifest = {
        "mesh_path": "working_mesh.glb", "face_ids_path": "face_primitive_ids.npy",
        "primitive_ids": list(range(len(mapping))), "seed": seed,
        "old_to_new": sorted(mapping.items()), "frame": {"center": center, "scale": scale},
        "texture_transfer": texture_transfer,
        "removed_degenerate_faces": removed_degenerate_faces,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def serve(socket_path, automask) -> None:
    if socket_path.exists():
        raise FileExistsError(f"P3-SAM server socket already exists: {socket_path}")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        listener.listen()
        print(f"[p3sam-server] ready: {socket_path}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                with connection.makefile("r", encoding="utf-8") as request_file:
                    request = json.loads(request_file.readline())
                if request.get("shutdown") is True:
                    connection.sendall(b'{"ok": true}\n')
                    return
                stdout, stderr = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        segment(
                            Path(request["input"]), Path(request["output_dir"]),
                            int(request.get("seed", 0)), automask,
                        )
                    response = {
                        "ok": True, "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(),
                    }
                except BaseException:
                    response = {
                        "ok": False, "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(), "error": traceback.format_exc(),
                    }
                connection.sendall((json.dumps(response) + "\n").encode("utf-8"))
                try:
                    import torch
                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    automask = load_automask(args.checkout, args.checkpoint, args.sonata_dir)
    if args.server_socket is not None:
        serve(args.server_socket, automask)
    else:
        segment(args.input, args.output_dir, args.seed, automask)


if __name__ == "__main__":
    main()
