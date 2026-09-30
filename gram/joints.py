from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class JointEstimate:
    type: str
    subtype: str | None
    axis: np.ndarray
    pivot: np.ndarray
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=float)
        pivot = np.asarray(self.pivot, dtype=float)
        norm = np.linalg.norm(axis)
        if axis.shape != (3,) or pivot.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(pivot).all() or norm <= 0:
            raise ValueError("joint axis and pivot must be finite 3-vectors with a nonzero axis")
        object.__setattr__(self, "axis", axis / norm)
        object.__setattr__(self, "pivot", pivot)

    def with_limits(self, lower: float, upper: float) -> "JointEstimate":
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError("joint lower limit must not exceed upper limit")
        return replace(self, lower=float(lower), upper=float(upper))

    def to_dict(self):
        value = asdict(self)
        value["axis"], value["pivot"] = self.axis.tolist(), self.pivot.tolist()
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(value["type"], value.get("subtype"), np.array(value["axis"]), np.array(value["pivot"]),
                   value.get("lower"), value.get("upper"))


def _boundary_vertices(mesh) -> np.ndarray:
    boundary = getattr(mesh, "edges_boundary", np.empty((0, 2), dtype=int))
    indices = np.unique(boundary) if len(boundary) else np.arange(len(mesh.vertices))
    return np.asarray(mesh.vertices)[indices]


def extract_contact_points(child, parent, threshold: float) -> np.ndarray:
    if threshold <= 0:
        raise ValueError("contact threshold must be positive")
    child_vertices, parent_vertices = _boundary_vertices(child), _boundary_vertices(parent)
    if not len(child_vertices) or not len(parent_vertices):
        return np.empty((0, 3))
    distances, _ = cKDTree(parent_vertices).query(child_vertices)
    return child_vertices[distances <= threshold]


def _principal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("at least three finite 3-D contact points are required")
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
    return center, values, vectors


def initialize_hinge(contact_points: np.ndarray) -> JointEstimate:
    center, _, vectors = _principal(contact_points)
    return JointEstimate("revolute", "hinge", vectors[:, -1], center)


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    a, b, c = points
    matrix = 2 * np.array([b - a, c - a])
    rhs = np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)])
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(a - center))


def initialize_spin(contact_points: np.ndarray, seed: int) -> JointEstimate:
    points = np.asarray(contact_points, dtype=float)
    center3, _, vectors = _principal(points)
    axis, basis = vectors[:, 0], vectors[:, 1:]
    plane = (points - center3) @ basis
    rng = np.random.default_rng(seed)
    threshold = max(float(np.ptp(plane, axis=0).max()) * 0.02, 1e-8)
    best = np.ones(len(plane), dtype=bool)
    best_count = 0
    for _ in range(256):
        candidate = _circle_from_three(plane[rng.choice(len(plane), 3, replace=False)])
        if candidate is None:
            continue
        circle_center, radius = candidate
        inliers = np.abs(np.linalg.norm(plane - circle_center, axis=1) - radius) <= threshold
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    fit = plane[best]
    if len(fit) < 3:
        raise ValueError("spin circle fit found fewer than three inliers")
    matrix = np.c_[2 * fit, np.ones(len(fit))]
    solution, *_ = np.linalg.lstsq(matrix, np.sum(fit * fit, axis=1), rcond=None)
    circle_center = solution[:2]
    try:
        hull = ConvexHull(plane)
        if np.any(hull.equations[:, :2] @ circle_center + hull.equations[:, 2] > 1e-9):
            circle_center = np.zeros(2)
    except QhullError:
        circle_center = np.zeros(2)
    return JointEstimate("revolute", "spin", axis, center3 + basis @ circle_center)


def spin_axis_support(contact_points: np.ndarray, axis: np.ndarray) -> dict[str, float | bool]:
    """Report whether contacts geometrically support a proposed spin axis."""
    _, values, vectors = _principal(contact_points)
    planarity_ratio = float(values[0] / values[1]) if values[1] > 1e-12 else 1.0
    normal_alignment = abs(float(np.dot(
        np.asarray(axis, dtype=float) / np.linalg.norm(axis), vectors[:, 0],
    )))
    return {
        "planarity_ratio": planarity_ratio,
        "normal_alignment": normal_alignment,
        "supported": planarity_ratio <= 0.1 and normal_alignment >= 0.9,
    }


def _thin_axis(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    _, vectors = np.linalg.eigh(np.cov(vertices - vertices.mean(axis=0), rowvar=False))
    return vectors[:, 0]


def revolute_hypotheses(contact_points: np.ndarray, parent, child, seed: int
                        ) -> tuple[JointEstimate, JointEstimate | None, str | None]:
    hinge = initialize_hinge(contact_points)
    _, values, _ = _principal(contact_points)
    line_like = values[2] > 1e-12 and values[1] <= 0.05 * values[2]
    try:
        spin = initialize_spin(contact_points, seed)
    except ValueError:
        spin = None
    if line_like:
        return hinge, spin, "hinge"
    if spin is None or values[1] <= 1e-12 or values[0] > 0.1 * values[1]:
        return hinge, spin, None
    parent_normal, child_normal = _thin_axis(parent), _thin_axis(child)
    normals_agree = abs(float(parent_normal @ child_normal)) >= 0.9
    spin_aligned = min(abs(float(spin.axis @ parent_normal)),
                       abs(float(spin.axis @ child_normal))) >= 0.9
    hinge_in_plane = max(abs(float(hinge.axis @ parent_normal)),
                         abs(float(hinge.axis @ child_normal))) <= 0.25
    preference = "spin" if normals_agree and spin_aligned and hinge_in_plane else None
    return hinge, spin, preference


def prismatic_pca_candidates(child) -> dict[str, dict[str, float | list[float]]]:
    vertices = np.asarray(child.vertices, dtype=float)
    center = vertices.mean(axis=0)
    variances, axes = np.linalg.eigh(np.cov(vertices - center, rowvar=False))
    result = {}
    for index, (variance, axis) in enumerate(zip(variances, axes.T)):
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        result[f"pca_axis_{index}"] = {
            "axis": (axis / np.linalg.norm(axis)).tolist(), "variance": float(variance),
        }
    return result


def initialize_prismatic(child, parent, contact_points: np.ndarray,
                         evaluator: Callable[[np.ndarray, float], float | tuple[float, float]],
                         selected_axis: np.ndarray | None = None) -> JointEstimate:
    if selected_axis is not None:
        return JointEstimate(
            "prismatic", None, selected_axis, np.asarray(contact_points).mean(axis=0))
    vertices = np.asarray(child.vertices, dtype=float)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    distances = np.array([-0.08, -0.04, 0.04, 0.08]) * diagonal
    best_score, best_axis = float("inf"), None
    for item in prismatic_pca_candidates(child).values():
        candidate = np.asarray(item["axis"], dtype=float)
        score = 0.0
        for distance in distances:
            value = evaluator(candidate, float(distance))
            score += float(value[0] + 20 * value[1] if isinstance(value, tuple) else value)
        if score < best_score:
            best_score, best_axis = score, candidate
    return JointEstimate("prismatic", None, best_axis, np.asarray(contact_points).mean(axis=0))


def require_contacts(points: np.ndarray, child_id: int, diagnostic: Path | None = None, minimum: int = 32) -> None:
    if len(points) >= minimum:
        return
    if diagnostic is not None and len(points):
        import trimesh
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        trimesh.points.PointCloud(points).export(diagnostic)
    raise ValueError(f"fit-joints refused joint {child_id}: only {len(points)} contact samples, minimum is {minimum}")
