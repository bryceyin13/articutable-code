from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

from .spin_closure import (
    _closed_revolute_bbox_direction,
    _persistent_interface_faces,
)


@dataclass(frozen=True)
class HingePlaneState:
    axis: np.ndarray
    mesh_current_q: float
    closing_rotation: float
    parent_normal: np.ndarray
    child_normal: np.ndarray
    parent_patch_area: float
    child_patch_area: float
    normal_alignment_error: float
    parent_patch_count: int
    child_patch_count: int
    collision_upper: float | None = None
    plane_closing_rotation: float | None = None
    collision_candidate_rank: int | None = None
    collision_candidates_tested: int = 0
    collision_evaluations: int = 0
    axis_flipped_by_collision: bool = False
    locked_due_to_initial_visual_collision: bool = False
    reconstruction_clamped_to_zero: bool = False
    plane_closed_q: float = 0.0
    closure_method: str = "plane"
    parent_patch_area_ratio: float = 0.0
    child_patch_area_ratio: float = 0.0
    projected_overlap_ratio: float = 0.0
    parent_plane_footprint_area: float = 0.0
    child_plane_footprint_area: float = 0.0
    mechanical_rest_rotation: float | None = None
    mechanical_rest_score: float | None = None
    mechanical_rest_height: float | None = None
    mechanical_rest_compactness: float | None = None
    mechanical_rest_surface_gap: float | None = None
    mechanical_scan_safe_samples: int = 0
    opening_direction_method: str = "reconstruction_pose"
    opening_direction_bbox: dict | None = None

    def to_dict(self) -> dict:
        return {
            "axis_direction": self.axis.tolist(),
            "mesh_current_q": self.mesh_current_q,
            "closing_rotation": self.closing_rotation,
            "parent_normal": self.parent_normal.tolist(),
            "child_normal": self.child_normal.tolist(),
            "selected_patch_area": {
                "parent": self.parent_patch_area,
                "child": self.child_patch_area,
            },
            "normal_alignment_error": self.normal_alignment_error,
            "candidate_patch_counts": {
                "parent": self.parent_patch_count,
                "child": self.child_patch_count,
            },
            "collision_upper": self.collision_upper,
            "plane_closing_rotation": self.plane_closing_rotation,
            "collision_candidate_rank": self.collision_candidate_rank,
            "collision_candidates_tested": self.collision_candidates_tested,
            "collision_evaluations": self.collision_evaluations,
            "axis_flipped_by_collision": self.axis_flipped_by_collision,
            "locked_due_to_initial_visual_collision": (
                self.locked_due_to_initial_visual_collision),
            "reconstruction_clamped_to_zero": self.reconstruction_clamped_to_zero,
            "plane_closed_q": self.plane_closed_q,
            "closure_method": self.closure_method,
            "parent_patch_area_ratio": self.parent_patch_area_ratio,
            "child_patch_area_ratio": self.child_patch_area_ratio,
            "projected_overlap_ratio": self.projected_overlap_ratio,
            "selected_plane_footprint_area": {
                "parent": self.parent_plane_footprint_area,
                "child": self.child_plane_footprint_area,
            },
            "mechanical_rest_rotation": self.mechanical_rest_rotation,
            "mechanical_rest_score": self.mechanical_rest_score,
            "mechanical_rest_height": self.mechanical_rest_height,
            "mechanical_rest_compactness": self.mechanical_rest_compactness,
            "mechanical_rest_surface_gap": self.mechanical_rest_surface_gap,
            "mechanical_scan_safe_samples": self.mechanical_scan_safe_samples,
            "opening_direction_method": self.opening_direction_method,
            "opening_direction_bbox": self.opening_direction_bbox,
        }


def planar_patches(vertices, faces, normal_angle, plane_tolerance, min_area):
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(crosses, axis=1) * 0.5
    normals = crosses / np.maximum(2.0 * areas[:, None], 1e-12)
    centers = triangles.mean(axis=1)

    edge_faces = defaultdict(list)
    for face_index, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces[tuple(sorted((int(start), int(end))))].append(face_index)
    neighbors = [[] for _ in faces]
    for adjacent in edge_faces.values():
        if len(adjacent) == 2:
            neighbors[adjacent[0]].append(adjacent[1])
            neighbors[adjacent[1]].append(adjacent[0])

    cosine = math.cos(normal_angle)
    unused, patches = set(range(len(faces))), []
    while unused:
        seed = unused.pop()
        seed_normal, seed_center = normals[seed], centers[seed]
        queue, members = deque([seed]), [seed]
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if neighbor not in unused:
                    continue
                aligned = abs(float(normals[neighbor] @ seed_normal)) >= cosine
                coplanar = abs(float((centers[neighbor] - seed_center) @ seed_normal)) <= plane_tolerance
                if aligned and coplanar:
                    unused.remove(neighbor)
                    queue.append(neighbor)
                    members.append(neighbor)

        area = float(areas[members].sum())
        if area < min_area:
            continue
        patch_faces = faces[members]
        patch_normal = sum(
            areas[index] * normals[index] * (1 if normals[index] @ seed_normal >= 0 else -1)
            for index in members
        )
        patch_normal /= np.linalg.norm(patch_normal)
        patches.append({
            "area": area,
            "center": np.average(centers[members], axis=0, weights=areas[members]),
            "normal": patch_normal,
            "points": vertices[np.unique(patch_faces)],
        })
    return patches


def fit_large_planes(
        mesh, axis, scale, normal_angle, axis_angle, plane_tolerance,
        min_support_ratio, max_planes, sample_count=30000,
        hypotheses=512, seed=0):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    count = min(sample_count, max(6000, len(mesh.faces)))
    points, face_ids = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=float)
    eligible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (np.abs(normals @ axis) < math.sin(axis_angle))
    )
    remaining = np.flatnonzero(eligible)
    minimum = max(20, math.ceil(min_support_ratio * len(points)))
    cosine = math.cos(normal_angle)
    rng = np.random.default_rng(seed)
    planes = []
    for _ in range(max_planes):
        if len(remaining) < minimum:
            break
        best = None
        candidates = rng.choice(
            remaining, min(hypotheses, len(remaining)), replace=False)
        for index in candidates:
            normal = normals[index] - axis * float(normals[index] @ axis)
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            normal /= length
            offset = -float(normal @ points[index])
            selected = remaining[
                (np.abs(points[remaining] @ normal + offset) <= plane_tolerance)
                & (np.abs(normals[remaining] @ normal) >= cosine)
            ]
            residual = (
                float(np.median(np.abs(points[selected] @ normal + offset)))
                if len(selected) else math.inf
            )
            score = (len(selected), -residual)
            if best is None or score > best[0]:
                best = score, normal, selected
        if best is None or len(best[2]) < minimum:
            break
        normal, selected = best[1], best[2]
        for _ in range(3):
            if len(selected) < minimum:
                break
            center = points[selected].mean(axis=0)
            _, _, vectors = np.linalg.svd(
                points[selected] - center, full_matrices=False)
            refined = vectors[-1] - axis * float(vectors[-1] @ axis)
            length = np.linalg.norm(refined)
            if length < 1e-12:
                break
            refined /= length
            normal = refined if refined @ normal >= 0 else -refined
            offset = -float(normal @ center)
            selected = remaining[
                (np.abs(points[remaining] @ normal + offset) <= plane_tolerance)
                & (np.abs(normals[remaining] @ normal) >= cosine)
            ]
        if len(selected) < minimum:
            break
        radial = np.cross(normal, axis)
        radial /= np.linalg.norm(radial)
        projected = np.column_stack((points[selected] @ axis, points[selected] @ radial))
        try:
            footprint_area = float(ConvexHull(projected).volume)
        except QhullError:
            footprint_area = 0.0
        planes.append({
            "area": float(mesh.area * len(selected) / len(points)),
            "footprint_area": footprint_area,
            "center": points[selected].mean(axis=0),
            "normal": normal,
            "points": points[selected],
        })
        remaining = np.setdiff1d(remaining, selected, assume_unique=True)
    return planes


def rotate_vectors(values, axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    values = np.asarray(values, dtype=float)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (values * cosine + np.cross(axis, values) * sine
            + np.outer(values @ axis, axis) * (1.0 - cosine))


def rotate_points(points, origin, axis, angle):
    return rotate_vectors(np.asarray(points) - origin, axis, angle) + origin


def interval_overlap(low_a, high_a, low_b, high_b):
    return max(0.0, min(high_a, high_b) - max(low_a, low_b))


def pair_score(parent, child, origin, axis, delta, scale):
    """Return the legacy visualization score and reusable local geometry terms."""
    child_normal = rotate_vectors([child["normal"]], axis, delta)[0]
    child_points = rotate_points(child["points"], origin, axis, delta)
    child_center = rotate_points([child["center"]], origin, axis, delta)[0]
    align = 1.0 - abs(float(parent["normal"] @ child_normal))
    distance = abs(float((child_center - parent["center"]) @ parent["normal"])) / scale

    radial = np.cross(parent["normal"], axis)
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-12:
        return math.inf, {"normal_alignment_error": align}
    radial /= radial_norm
    parent_2d = np.column_stack((parent["points"] @ axis, parent["points"] @ radial))
    child_2d = np.column_stack((child_points @ axis, child_points @ radial))
    p_min, p_max = parent_2d.min(axis=0), parent_2d.max(axis=0)
    c_min, c_max = child_2d.min(axis=0), child_2d.max(axis=0)
    overlap_bbox = interval_overlap(p_min[0], p_max[0], c_min[0], c_max[0]) * interval_overlap(
        p_min[1], p_max[1], c_min[1], c_max[1])
    overlap_area = min(overlap_bbox, parent["area"], child["area"])
    overlap = overlap_area / (scale * scale)
    axis_distance = (
        abs(float((origin - parent["center"]) @ parent["normal"]))
        + abs(float((origin - child["center"]) @ child["normal"]))
    ) / scale
    open_side = np.sign((child["center"] - parent["center"]) @ parent["normal"]) or 1.0
    signed_depth = open_side * ((child_points - parent["center"]) @ parent["normal"])
    penetration = float(np.maximum(0.0, -signed_depth).mean()) / scale
    return 4.0 * align + 2.0 * distance + 2.0 * axis_distance - 6.0 * overlap + penetration, {
        "normal_alignment_error": align,
        "plane_distance": distance,
        "axis_plane_distance": axis_distance,
        "projected_overlap": overlap,
        "projected_overlap_area": overlap_area,
        "penetration": penetration,
    }


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _articulated_aabb_volume(stationary_bounds, moving_vertices, origin, axis, angle):
    moved = rotate_points(moving_vertices, origin, axis, angle)
    low = np.minimum(stationary_bounds[0], moved.min(axis=0))
    high = np.maximum(stationary_bounds[1], moved.max(axis=0))
    return float(np.prod(high - low))


def _minmax_columns(values):
    values = np.asarray(values, dtype=float)
    low = values.min(axis=0)
    span = np.ptp(values, axis=0)
    return np.divide(
        values - low, span, out=np.zeros_like(values), where=span > 0.0)


def _without_faces(mesh: trimesh.Trimesh, excluded: np.ndarray) -> trimesh.Trimesh | None:
    selected = np.flatnonzero(~excluded)
    if not len(selected):
        return None
    result = mesh.submesh([selected], append=True, repair=False)
    return result if isinstance(result, trimesh.Trimesh) and len(result.faces) else None


def _filtered_collision_managers(
        parent_mesh, child_mesh, stationary_meshes, moving_meshes,
        axis, pivot, collision_manager_factory,
        interface_probe_degrees, interface_distance_ratio):
    vertices = np.vstack((parent_mesh.vertices, child_mesh.vertices))
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    parent_interface, child_interface = _persistent_interface_faces(
        parent_mesh, child_mesh, axis, pivot, scale,
        probe_degrees=interface_probe_degrees,
        distance_ratio=interface_distance_ratio)
    parent, child = (_without_faces(parent_mesh, parent_interface),
                     _without_faces(child_mesh, child_interface))
    if parent is None or child is None:
        raise ValueError("hinge interface filtering removed an entire direct link")

    stationary, moving = collision_manager_factory(), collision_manager_factory()
    stationary.add_object("parent", parent)
    moving.add_object("child", child)
    for index, mesh in enumerate(stationary_meshes):
        stationary.add_object(f"stationary_{index}", mesh)
    moving_names = []
    for index, mesh in enumerate(moving_meshes):
        name = f"moving_{index}"
        moving.add_object(name, mesh)
        moving_names.append(name)
    return stationary, moving, ("child", *moving_names)


def _collision_probe(parent_mesh, child_mesh, stationary_meshes, moving_meshes,
                     axis, pivot, collision_manager_factory,
                     interface_probe_degrees, interface_distance_ratio):
    stationary, moving, names = _filtered_collision_managers(
        parent_mesh, child_mesh, stationary_meshes, moving_meshes,
        axis, pivot, collision_manager_factory,
        interface_probe_degrees, interface_distance_ratio)
    evaluations = 0

    def collides(angle):
        nonlocal evaluations
        evaluations += 1
        transform = trimesh.transformations.rotation_matrix(angle, axis, pivot)
        for name in names:
            moving.set_transform(name, transform)
        return bool(stationary.in_collision_other(moving))

    return collides, lambda: evaluations


def _mechanical_rest_rotation(
        parent_mesh, child_mesh, stationary_meshes, moving_meshes,
        axis, pivot, collision_manager_factory,
        interface_probe_degrees, interface_distance_ratio,
        step_degrees=2.0, refine_degrees=0.1):
    if step_degrees <= 0 or refine_degrees <= 0 or refine_degrees > step_degrees:
        raise ValueError("mechanical hinge scan steps must be positive and ordered")
    stationary, moving, names = _filtered_collision_managers(
        parent_mesh, child_mesh, stationary_meshes, moving_meshes,
        axis, pivot, collision_manager_factory,
        interface_probe_degrees, interface_distance_ratio)
    moving_group = (child_mesh, *moving_meshes)
    areas = np.asarray([mesh.area for mesh in moving_group], dtype=float)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("mechanical hinge scan requires positive moving surface area")
    centroid = np.average(
        np.asarray([mesh.centroid for mesh in moving_group]), axis=0, weights=areas)
    moving_vertices = np.vstack([mesh.vertices for mesh in moving_group])
    stationary_vertices = np.vstack(
        [parent_mesh.vertices, *(mesh.vertices for mesh in stationary_meshes)])
    stationary_bounds = stationary_vertices.min(axis=0), stationary_vertices.max(axis=0)
    up = np.array([0.0, 1.0, 0.0])
    evaluations = 0

    def sample(angle):
        nonlocal evaluations
        evaluations += 1
        transform = trimesh.transformations.rotation_matrix(angle, axis, pivot)
        for name in names:
            moving.set_transform(name, transform)
        gap = float(stationary.min_distance_other(moving))
        if not math.isfinite(gap) or gap <= 0.0:
            return None
        height = float(rotate_points([centroid], pivot, axis, angle)[0] @ up)
        compactness = _articulated_aabb_volume(
            stationary_bounds, moving_vertices, pivot, axis, angle)
        return float(angle), height, compactness, gap

    coarse_step = math.radians(step_degrees)
    coarse = [
        state for state in (
            sample(angle)
            for angle in np.arange(-math.pi, math.pi, coarse_step))
        if state is not None
    ]
    if not coarse:
        raise ValueError("mechanical hinge scan found no collision-free pose")

    def scores(states):
        metrics = _minmax_columns([
            [state[1], state[2], state[3]] for state in states
        ])
        return metrics @ np.array([0.5, 0.3, 0.2])

    coarse_best = coarse[int(np.argmin(scores(coarse)))][0]
    refine_step = math.radians(refine_degrees)
    refined = [
        state for state in (
            sample(angle)
            for angle in np.arange(
                max(-math.pi, coarse_best - coarse_step),
                min(math.pi, coarse_best + coarse_step) + refine_step / 2,
                refine_step))
        if state is not None
    ]
    candidates = coarse + refined
    candidate_scores = scores(candidates)
    index = int(np.argmin(candidate_scores))
    return candidates[index][0], {
        "score": float(candidate_scores[index]),
        "height": float(candidates[index][1]),
        "compactness": float(candidates[index][2]),
        "surface_gap": float(candidates[index][3]),
        "safe_samples": len(candidates),
        "evaluations": evaluations,
    }


def infer_hinge_plane_state(
        parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh,
        axis, pivot, *, normal_angle_degrees: float = 5.0,
        axis_angle_degrees: float = 3.0, min_plane_support_ratio: float = 0.005,
        max_planes: int = 12,
        stationary_meshes=(), moving_meshes=(),
        angle_tolerance_degrees: float = 0.01,
        interface_probe_degrees=(1.0, 3.0, 5.0),
        interface_distance_ratio: float = 0.005,
        min_selected_plane_footprint_area: float = 0.05,
        min_projected_overlap_ratio: float = 0.25,
        mechanical_step_degrees: float = 2.0,
        mechanical_refine_degrees: float = 0.1,
        collision_manager_factory=trimesh.collision.CollisionManager) -> HingePlaneState:
    axis, pivot = np.asarray(axis, dtype=float), np.asarray(pivot, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or not np.isfinite(axis).all() or not np.isfinite(pivot).all():
        raise ValueError("hinge plane inference requires a finite nonzero axis and pivot")
    axis /= norm
    vertices = np.vstack((parent_mesh.vertices, child_mesh.vertices))
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("hinge plane inference requires nonzero mesh extent")
    stationary_meshes, moving_meshes = tuple(stationary_meshes), tuple(moving_meshes)
    stationary_vertices = np.vstack(
        (parent_mesh.vertices, *(mesh.vertices for mesh in stationary_meshes)))
    moving_vertices = np.vstack(
        (child_mesh.vertices, *(mesh.vertices for mesh in moving_meshes)))
    stationary_bounds = stationary_vertices.min(axis=0), stationary_vertices.max(axis=0)

    plane_args = (
        axis, scale, math.radians(normal_angle_degrees),
        math.radians(axis_angle_degrees), scale * 0.002,
        min_plane_support_ratio, max_planes,
    )
    parent_planes = fit_large_planes(parent_mesh, *plane_args, seed=0)
    child_planes = fit_large_planes(child_mesh, *plane_args, seed=1)
    candidates = []
    for parent in parent_planes:
        parent_normal = parent["normal"] - axis * (parent["normal"] @ axis)
        parent_normal /= np.linalg.norm(parent_normal)
        for child in child_planes:
            child_normal = child["normal"] - axis * (child["normal"] @ axis)
            child_normal /= np.linalg.norm(child_normal)
            for target in (parent_normal, -parent_normal):
                delta = _wrap_pi(math.atan2(
                    float(axis @ np.cross(child_normal, target)),
                    float(child_normal @ target)))
                _, terms = pair_score(parent, child, pivot, axis, delta, scale)
                if terms["projected_overlap_area"] <= np.finfo(float).eps * scale * scale:
                    continue
                raw_scores = np.array([
                    terms["plane_distance"],
                    terms["axis_plane_distance"],
                    terms["projected_overlap"],
                    _articulated_aabb_volume(
                        stationary_bounds, moving_vertices, pivot, axis, delta),
                    abs(delta),
                ])
                rotated = rotate_vectors([child_normal], axis, delta)[0]
                projected_error = max(
                    0.0, 1.0 - abs(float(parent_normal @ rotated)))
                if np.isfinite(raw_scores).all():
                    candidates.append((
                        raw_scores, delta, parent, child, projected_error, terms))
    if angle_tolerance_degrees <= 0:
        raise ValueError("hinge angle tolerance must be positive")
    if (not math.isfinite(min_selected_plane_footprint_area)
            or min_selected_plane_footprint_area < 0):
        raise ValueError("selected hinge plane footprint area must be finite and nonnegative")
    if not 0 <= min_projected_overlap_ratio <= 1:
        raise ValueError("hinge projected overlap ratio must lie in [0, 1]")

    if candidates:
        normalized = _minmax_columns([candidate[0] for candidate in candidates])
        scores = normalized @ np.array([1.0, 1.0, -1.0, 1.0, 1.0])
        _, plane_closing, parent, child, alignment_error, terms = candidates[
            int(np.argmin(scores))]
        parent_area_ratio = float(parent["area"] / parent_mesh.area)
        child_area_ratio = float(child["area"] / child_mesh.area)
        parent_footprint_area = float(parent.get("footprint_area", parent["area"]))
        child_footprint_area = float(child.get("footprint_area", child["area"]))
        overlap_ratio = float(
            terms["projected_overlap_area"] / min(parent["area"], child["area"]))
    else:
        plane_closing = 0.0
        parent = {"normal": np.zeros(3), "area": 0.0}
        child = {"normal": np.zeros(3), "area": 0.0}
        alignment_error = 1.0
        parent_footprint_area = child_footprint_area = 0.0
        parent_area_ratio = child_area_ratio = overlap_ratio = 0.0
    plane_reliable = (
        min(parent_footprint_area, child_footprint_area)
        >= min_selected_plane_footprint_area
        and overlap_ratio >= min_projected_overlap_ratio
    )
    closure_method = "plane"
    mechanical = {
        "score": None, "height": None, "compactness": None,
        "surface_gap": None, "safe_samples": 0, "evaluations": 0,
    }
    closing_rotation = float(plane_closing)
    if not plane_reliable:
        closing_rotation, mechanical = _mechanical_rest_rotation(
            parent_mesh, child_mesh, stationary_meshes, moving_meshes,
            axis, pivot, collision_manager_factory,
            interface_probe_degrees, interface_distance_ratio,
            mechanical_step_degrees, mechanical_refine_degrees)
        closure_method = "mechanical-rest"
    tolerance = math.radians(angle_tolerance_degrees)
    near_closed_tolerance = math.radians(max(map(float, interface_probe_degrees)))
    raw_input_q = -float(closing_rotation)
    axis_flipped = flipped_by_collision = False
    collision_evaluations = int(mechanical["evaluations"])
    direction_method = "reconstruction_pose"
    direction_bbox = None
    if abs(raw_input_q) <= max(tolerance, near_closed_tolerance):
        direction, direction_bbox = _closed_revolute_bbox_direction(
            parent_mesh, child_mesh, axis, pivot)
        if direction is None:
            collides, evaluation_count = _collision_probe(
                parent_mesh, child_mesh, tuple(stationary_meshes), tuple(moving_meshes),
                axis, pivot, collision_manager_factory,
                interface_probe_degrees, interface_distance_ratio)
            hits = [
                bool(collides(closing_rotation + math.radians(degrees)))
                for degrees in range(0, 360, 10)
            ]
            positive_hits = sum(hits[1:18])
            negative_hits = sum(hits[19:])
            direction = -1 if negative_hits < positive_hits else 1
            flipped_by_collision = direction == -1
            collision_evaluations += evaluation_count()
            direction_method = "collision_probe"
        else:
            direction_method = "original_minimum_bbox"
        axis_flipped = direction == -1
        mesh_current = plane_closed_q = 0.0
    elif raw_input_q > 0.0:
        mesh_current, plane_closed_q = raw_input_q, 0.0
    else:
        axis_flipped = True
        mesh_current, plane_closed_q = -raw_input_q, 0.0
    return HingePlaneState(
        -axis if axis_flipped else axis.copy(), float(mesh_current),
        float(plane_closed_q - mesh_current),
        parent["normal"].copy(), child["normal"].copy(), float(parent["area"]),
        float(child["area"]), float(alignment_error), len(parent_planes),
        len(child_planes), None, float(plane_closing),
        (1 if candidates else None), (1 if candidates else 0),
        collision_evaluations, flipped_by_collision, False, False,
        float(plane_closed_q), closure_method, parent_area_ratio,
        child_area_ratio, overlap_ratio, parent_footprint_area,
        child_footprint_area,
        (float(closing_rotation) if not plane_reliable else None),
        mechanical["score"], mechanical.get("height"),
        mechanical.get("compactness"), mechanical.get("surface_gap"),
        int(mechanical["safe_samples"]), direction_method, direction_bbox)
