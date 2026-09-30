"""Coordinate conversions defined by a table_alignment_v1 manifest."""

import json
from pathlib import Path

import numpy as np


def _parts(alignment):
    scale = float(alignment["metric_scale_world_per_vggt"])
    rotation = np.asarray(alignment["rotation_world_from_vggt"], dtype=float)
    translation = np.asarray(alignment["translation_world"], dtype=float)
    if scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid table alignment transform")
    if not np.isfinite(scale) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("table alignment transform must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("table alignment rotation must be right-handed and orthonormal")
    return scale, rotation, translation


def load_alignment(path):
    alignment = json.loads(Path(path).read_text(encoding="utf-8"))
    if alignment.get("format") != "table_alignment_v1":
        raise ValueError("unsupported table alignment format")
    if alignment.get("unit") != "meter" or alignment.get("up_axis") != "Z":
        raise ValueError("table alignment must use meters and Z-up")
    scale, rotation, translation = _parts(alignment)
    matrix = np.asarray(alignment.get("similarity_matrix_world_from_vggt"), dtype=float)
    expected = np.eye(4)
    expected[:3, :3] = scale * rotation
    expected[:3, 3] = translation
    table_size = np.asarray(alignment.get("table_size_m"), dtype=float)
    if matrix.shape != (4, 4) or not np.allclose(matrix, expected, atol=1e-6):
        raise ValueError("table alignment similarity matrix is inconsistent")
    if table_size.shape != (3,) or not np.isfinite(table_size).all() or np.any(table_size <= 0):
        raise ValueError("invalid aligned table size")
    return alignment


def vggt_to_world(points, alignment):
    scale, rotation, translation = _parts(alignment)
    points = np.asarray(points, dtype=float)
    return points @ rotation.T * scale + translation


def world_to_vggt(points, alignment):
    scale, rotation, translation = _parts(alignment)
    points = np.asarray(points, dtype=float)
    return (points - translation) @ rotation / scale


def blueprint_xy_to_world(xy_norm, alignment, z=0.0):
    xy_norm = np.asarray(xy_norm, dtype=float)
    if xy_norm.shape[-1:] != (2,):
        raise ValueError("blueprint XY coordinates must end in two values")
    table_size = np.asarray(alignment["table_size_m"], dtype=float)
    xy = xy_norm * table_size[:2]
    return np.concatenate((xy, np.full(xy.shape[:-1] + (1,), float(z))), axis=-1)
