from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import QhullError, cKDTree


_PROXIMITY_BATCH_SIZE = 2048
_SMALL_AXIAL_OVERLAP_RATIO = 0.15


@dataclass(frozen=True)
class SpinClosureState:
    axis: np.ndarray
    scan_axis: np.ndarray
    pivot: np.ndarray
    mesh_current_q: float
    closing_rotation: float
    first_colliding_rotation: float | None
    axis_flipped: bool
    exclusion_radius: float
    parent_face_count: int
    child_face_count: int
    evaluations: int
    search_span: float
    coarse_step: float
    angle_tolerance: float
    locked_due_to_initial_visual_collision: bool = False
    opposite_safe_rotation: float | None = None
    opposite_first_colliding_rotation: float | None = None
    collision_upper: float | None = None
    exclusion_radius_ratio: float = 0.0
    exclusion_radius_source: str = "area_weighted_projected_aabb"
    robust_extent_area_ratio: float = 0.95
    robust_projected_diagonal: float = 0.0
    closure_shell_radius: float | None = None
    closure_shell_skips: int = 0
    opening_blocked_at_input: bool = False
    interface_pair_count: int = 0
    parent_interface_face_count: int = 0
    child_interface_face_count: int = 0
    interface_pair_face_counts: tuple = ()
    parent_boundary_candidate_face_count: int = 0
    child_boundary_candidate_face_count: int = 0
    parent_planar_boundary_face_count: int = 0
    child_planar_boundary_face_count: int = 0
    parent_cylindrical_interface_face_count: int = 0
    child_cylindrical_interface_face_count: int = 0
    contact_area_threshold: float = 0.0
    contact_depth_threshold: float = 0.0
    ignored_subthreshold_contact_clusters: int = 0
    ignored_small_axial_overlap_clusters: int = 0
    opening_direction_method: str = "collision_probe"
    opening_direction_bbox: dict | None = None

    def to_dict(self) -> dict:
        return {
            "axis_direction": self.axis.tolist(),
            "scan_axis_direction": self.scan_axis.tolist(),
            "axis_point": self.pivot.tolist(),
            "axis_flipped_during_closure": self.axis_flipped,
            "mesh_current_q": self.mesh_current_q,
            "closing_rotation": self.closing_rotation,
            "first_colliding_rotation": self.first_colliding_rotation,
            "opposite_safe_rotation": self.opposite_safe_rotation,
            "opposite_first_colliding_rotation": self.opposite_first_colliding_rotation,
            "collision_upper": self.collision_upper,
            "exclusion_radius": self.exclusion_radius,
            "exclusion_radius_ratio": self.exclusion_radius_ratio,
            "exclusion_radius_source": self.exclusion_radius_source,
            "closure_shell_radius": self.closure_shell_radius,
            "closure_shell_skips": self.closure_shell_skips,
            "robust_extent_area_ratio": self.robust_extent_area_ratio,
            "robust_projected_diagonal": self.robust_projected_diagonal,
            "opening_collision_mode": "disabled",
            "opening_blocked_at_input": self.opening_blocked_at_input,
            "opening_direction_method": self.opening_direction_method,
            "opening_direction_bbox": self.opening_direction_bbox,
            "collision_mode": "robust_axis_cylinder",
            "interface_pair_count": self.interface_pair_count,
            "interface_face_counts": {
                "parent": self.parent_interface_face_count,
                "child": self.child_interface_face_count,
            },
            "interface_pairs": list(self.interface_pair_face_counts),
            "boundary_candidate_face_counts": {
                "parent": self.parent_boundary_candidate_face_count,
                "child": self.child_boundary_candidate_face_count,
            },
            "planar_boundary_face_counts": {
                "parent": self.parent_planar_boundary_face_count,
                "child": self.child_planar_boundary_face_count,
            },
            "cylindrical_interface_face_counts": {
                "parent": self.parent_cylindrical_interface_face_count,
                "child": self.child_cylindrical_interface_face_count,
            },
            "significant_contact": {
                "area_threshold": self.contact_area_threshold,
                "depth_threshold": self.contact_depth_threshold,
                "ignored_subthreshold_clusters": (
                    self.ignored_subthreshold_contact_clusters),
                "ignored_small_axial_overlap_clusters": (
                    self.ignored_small_axial_overlap_clusters),
                "coarse_confirmation_poses": 2,
            },
            "tested_face_counts": {
                "parent": self.parent_face_count,
                "child": self.child_face_count,
            },
            "evaluations": self.evaluations,
            "search_span_degrees": math.degrees(self.search_span),
            "opening_search_span_degrees": 0.0,
            "coarse_step_degrees": math.degrees(self.coarse_step),
            "angle_tolerance_degrees": math.degrees(self.angle_tolerance),
            "locked_due_to_initial_visual_collision": (
                self.locked_due_to_initial_visual_collision),
        }


def _axis_face_distances(mesh: trimesh.Trimesh, pivot, axis) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=float) - pivot
    radial = triangles - np.einsum("fvi,i->fv", triangles, axis)[..., None] * axis
    with np.errstate(all="ignore"):
        closest = trimesh.triangles.closest_point(radial, np.zeros((len(radial), 3)))
    distances = np.linalg.norm(closest, axis=1)
    invalid = ~np.isfinite(distances)
    if invalid.any():
        projected = radial[invalid]
        edge_distances = []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            begin, edge = projected[:, start], projected[:, end] - projected[:, start]
            denominator = np.einsum("fi,fi->f", edge, edge)
            parameter = np.divide(
                -np.einsum("fi,fi->f", begin, edge), denominator,
                out=np.zeros_like(denominator), where=denominator > 0)
            point = begin + np.clip(parameter, 0.0, 1.0)[:, None] * edge
            edge_distances.append(np.linalg.norm(point, axis=1))
        distances[invalid] = np.min(edge_distances, axis=0)
    return distances


def _dilate_faces(mesh: trimesh.Trimesh, selected: np.ndarray) -> np.ndarray:
    dilated = selected.copy()
    adjacency = np.asarray(mesh.face_adjacency, dtype=int)
    touching = selected[adjacency].any(axis=1)
    if touching.any():
        dilated[np.unique(adjacency[touching])] = True
    return dilated


class _SurfaceToleranceQuery:
    """Exact point-to-triangle threshold queries without a global candidate array."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.triangles = np.asarray(mesh.triangles, dtype=float)
        self.centers = np.asarray(mesh.triangles_center, dtype=float)
        self.max_radius = float(np.linalg.norm(
            self.triangles - self.centers[:, None, :], axis=2).max())
        self.tree = cKDTree(self.centers)

    def within(self, points, tolerance, *, batch_size=_PROXIMITY_BATCH_SIZE):
        points = np.asarray(points, dtype=float)
        result = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            batch = points[start:stop]
            candidates = self.tree.query_ball_point(
                batch, tolerance + self.max_radius)
            counts = np.fromiter(
                map(len, candidates), dtype=np.int64, count=len(candidates))
            if not counts.sum():
                continue
            point_indices = np.repeat(np.arange(len(batch)), counts)
            triangle_indices = np.concatenate(candidates).astype(np.intp, copy=False)
            closest = trimesh.triangles.closest_point(
                self.triangles[triangle_indices], batch[point_indices])
            close = np.linalg.norm(closest - batch[point_indices], axis=1) <= tolerance
            result[start + np.unique(point_indices[close])] = True
        return result


def _persistent_interface_faces(
        parent: trimesh.Trimesh, child: trimesh.Trimesh, axis, pivot, scale,
        *, probe_degrees=(1.0, 3.0, 5.0), distance_ratio=0.005):
    """Find joint-local surfaces that stay close through small bidirectional motion."""
    if not math.isfinite(distance_ratio) or distance_ratio <= 0 or not probe_degrees or any(
            not math.isfinite(float(value)) or not 0 < float(value) <= 5
            for value in probe_degrees):
        raise ValueError("revolute interface probes must be finite and within (0, 5] degrees")
    tolerance = scale * distance_ratio
    parent_centers = np.asarray(parent.triangles_center, dtype=float)
    child_centers = np.asarray(child.triangles_center, dtype=float)

    child_relative = np.asarray(child.vertices, dtype=float) - pivot
    child_axial = child_relative @ axis
    child_radial = np.linalg.norm(
        child_relative - np.outer(child_axial, axis), axis=1)
    parent_relative = parent_centers - pivot
    parent_axial = parent_relative @ axis
    parent_radial = np.linalg.norm(
        parent_relative - np.outer(parent_axial, axis), axis=1)
    parent_candidates = np.flatnonzero(
        (parent_radial <= child_radial.max() + tolerance)
        & (parent_axial >= child_axial.min() - tolerance)
        & (parent_axial <= child_axial.max() + tolerance))
    parent_persistent = np.ones(len(parent_candidates), dtype=bool)
    child_persistent = np.ones(len(child.faces), dtype=bool)
    parent_query = _SurfaceToleranceQuery(parent)
    child_query = _SurfaceToleranceQuery(child)
    angles = (0.0, *(math.radians(sign * float(value))
                     for value in probe_degrees for sign in (-1, 1)))
    for angle in angles:
        transform = trimesh.transformations.rotation_matrix(angle, axis, pivot)
        child_active = np.flatnonzero(child_persistent)
        if len(child_active):
            child_persistent[child_active] = parent_query.within(
                trimesh.transform_points(child_centers[child_active], transform),
                tolerance)
        parent_active = np.flatnonzero(parent_persistent)
        if len(parent_active):
            parent_persistent[parent_active] = child_query.within(
                trimesh.transform_points(
                    parent_centers[parent_candidates[parent_active]],
                    np.linalg.inv(transform)),
                tolerance)

    parent_faces = np.zeros(len(parent.faces), dtype=bool)
    parent_faces[parent_candidates] = parent_persistent
    return (_dilate_faces(parent, parent_faces),
            _dilate_faces(child, child_persistent))


def _robust_projected_extent(parent, child, axis, pivot, area_ratio):
    """Return an axis-radial AABB diagonal robust to low-area floating geometry."""
    if not math.isfinite(area_ratio) or not 0 < area_ratio <= 1:
        raise ValueError("robust spin extent area ratio must be within (0, 1]")
    triangles = np.concatenate((
        np.asarray(parent.triangles, dtype=float),
        np.asarray(child.triangles, dtype=float)))
    areas = np.concatenate((
        np.asarray(parent.area_faces, dtype=float),
        np.asarray(child.area_faces, dtype=float)))
    samples = triangles.reshape(-1, 3)
    weights = np.repeat(areas / 3.0, 3)
    valid = np.isfinite(samples).all(axis=1) & np.isfinite(weights) & (weights > 0)
    samples, weights = samples[valid], weights[valid]
    if not len(samples):
        raise ValueError("visual collision requires finite nondegenerate mesh faces")

    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    relative = samples - pivot
    projected = np.column_stack((relative @ first, relative @ second))
    order = np.argsort(np.linalg.norm(projected, axis=1), kind="stable")
    cumulative = np.cumsum(weights[order])
    count = int(np.searchsorted(
        cumulative, area_ratio * cumulative[-1], side="left")) + 1
    retained = projected[order[:count]]
    diagonal = float(np.linalg.norm(np.ptp(retained, axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("visual collision requires nonzero radial mesh extent")
    return diagonal


def _spin_collision_regions(
        parent, child, axis, pivot, radius_ratio, area_ratio, distance_ratio):
    axis, pivot = np.asarray(axis, dtype=float), np.asarray(pivot, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or not np.isfinite(axis).all() or not np.isfinite(pivot).all():
        raise ValueError("visual collision requires a finite nonzero axis and pivot")
    if not math.isfinite(radius_ratio) or not 0 <= radius_ratio < 1:
        raise ValueError("visual collision axis exclusion ratio must be in [0, 1)")
    if not math.isfinite(distance_ratio) or distance_ratio <= 0:
        raise ValueError("visual collision distance ratio must be positive")
    axis = axis / norm
    vertices = np.vstack((parent.vertices, child.vertices))
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("visual collision requires nonzero mesh extent")
    projected_diagonal = _robust_projected_extent(
        parent, child, axis, pivot, area_ratio)
    radius = radius_ratio * projected_diagonal
    parent_mask = _axis_face_distances(parent, pivot, axis) > radius
    child_mask = _axis_face_distances(child, pivot, axis) > radius
    return (
        axis, pivot, radius, projected_diagonal, scale * distance_ratio,
        _selected_submesh(parent, parent_mask),
        _selected_submesh(child, child_mask),
        int(parent_mask.sum()), int(child_mask.sum()))


def _selected_submesh(mesh, selected):
    faces = np.flatnonzero(selected)
    if not len(faces):
        return None
    result = mesh.submesh([faces], append=True, repair=False)
    return result if isinstance(result, trimesh.Trimesh) and len(result.faces) else None


def _small_axial_overlap(
        contacts, names, face_triangles, moving_names, transform, axis,
        axial_overlap_threshold):
    if face_triangles is None or axial_overlap_threshold is None:
        return False
    intervals = []
    for name in names:
        indices = np.unique([contact.index(name) for contact in contacts])
        points = np.asarray(face_triangles[name], dtype=float)[indices].reshape(-1, 3)
        if name in moving_names:
            points = trimesh.transform_points(points, transform)
        heights = points @ axis
        intervals.append(np.percentile(heights, (10.0, 90.0)))
    overlap = min(interval[1] for interval in intervals) - max(
        interval[0] for interval in intervals)
    return overlap <= axial_overlap_threshold


def _has_significant_contacts(
        contacts, face_areas, distance_threshold, *, face_triangles=None,
        moving_names=(), transform=None, axis=None,
        axial_overlap_threshold=None):
    """Return collision significance and rejected contact-cluster counts."""
    if not contacts:
        return True, 0, 0
    area_threshold = distance_threshold ** 2
    ignored = ignored_axial = 0
    grouped = {}
    for contact in contacts:
        key = tuple(sorted(contact.names))
        grouped.setdefault(key, []).append(contact)
    for names, group in grouped.items():
        points = np.asarray([contact.point for contact in group], dtype=float)
        tree = cKDTree(points)
        unseen = np.ones(len(group), dtype=bool)
        for start in range(len(group)):
            if not unseen[start]:
                continue
            component, queue = [], deque([start])
            unseen[start] = False
            while queue:
                index = queue.popleft()
                component.append(index)
                for neighbor in tree.query_ball_point(points[index], distance_threshold):
                    if unseen[neighbor]:
                        unseen[neighbor] = False
                        queue.append(neighbor)
            selected = [group[index] for index in component]
            support = min(
                sum(face_areas[name][indices])
                for name in names
                for indices in [np.unique([
                    contact.index(name) for contact in selected])]
            )
            depth = max(max(0.0, float(contact.depth)) for contact in selected)
            if support >= area_threshold or depth >= distance_threshold:
                if _small_axial_overlap(
                        selected, names, face_triangles, moving_names, transform,
                        axis, axial_overlap_threshold):
                    ignored_axial += 1
                    continue
                return True, ignored, ignored_axial
            ignored += 1
    return False, ignored, ignored_axial


def _cylinder_collision_probe(
        parent, child, filtered_parent, filtered_child, stationary_meshes,
        moving_meshes, axis, pivot, distance_threshold,
        collision_manager_factory, *, include_direct_parent_child=True):
    """Ignore the axis cylinder only for the direct parent-child pair."""
    probes = []

    def add_probe(stationary_objects, moving_objects, filter_axial_overlap=False):
        stationary_objects = [
            (name, mesh) for name, mesh in stationary_objects if mesh is not None]
        moving_objects = [
            (name, mesh) for name, mesh in moving_objects if mesh is not None]
        if not stationary_objects or not moving_objects:
            return
        stationary_manager = collision_manager_factory()
        moving_manager = collision_manager_factory()
        face_areas, face_triangles = {}, {}
        axial_overlap_threshold = None
        for name, mesh in stationary_objects:
            stationary_manager.add_object(name, mesh)
            face_areas[name] = np.asarray(mesh.area_faces, dtype=float)
            face_triangles[name] = np.asarray(mesh.triangles, dtype=float)
        moving_names = []
        for name, mesh in moving_objects:
            moving_manager.add_object(name, mesh)
            moving_names.append(name)
            face_areas[name] = np.asarray(mesh.area_faces, dtype=float)
            face_triangles[name] = np.asarray(mesh.triangles, dtype=float)
        if filter_axial_overlap:
            spans = [float(np.ptp(np.asarray(mesh.vertices) @ axis))
                     for _, mesh in (*stationary_objects, *moving_objects)]
            axial_overlap_threshold = _SMALL_AXIAL_OVERLAP_RATIO * min(spans)
        probes.append((stationary_manager, moving_manager, moving_names, face_areas,
                       face_triangles if filter_axial_overlap else None,
                       axial_overlap_threshold))

    if include_direct_parent_child:
        add_probe([("parent_filtered", filtered_parent)],
                  [("child_filtered", filtered_child)], True)
    add_probe([("parent", parent)], [
        (f"moving_{index}", mesh) for index, mesh in enumerate(moving_meshes)])
    add_probe([
        (f"stationary_{index}", mesh)
        for index, mesh in enumerate(stationary_meshes)
    ], [("child", child), *[
        (f"moving_from_stationary_{index}", mesh)
        for index, mesh in enumerate(moving_meshes)]])

    evaluations, ignored_clusters, ignored_axial_clusters = 0, 0, 0

    def collides(angle):
        nonlocal evaluations, ignored_clusters, ignored_axial_clusters
        evaluations += 1
        transform = trimesh.transformations.rotation_matrix(angle, axis, pivot)
        for (stationary_manager, moving_manager, names, face_areas,
             face_triangles, axial_overlap_threshold) in probes:
            for name in names:
                moving_manager.set_transform(name, transform)
            if not stationary_manager.in_collision_other(moving_manager):
                continue
            hit, contacts = stationary_manager.in_collision_other(
                moving_manager, return_data=True)
            if not hit:
                continue
            significant, ignored, ignored_axial = _has_significant_contacts(
                contacts, face_areas, distance_threshold,
                face_triangles=face_triangles, moving_names=names,
                transform=transform, axis=axis,
                axial_overlap_threshold=axial_overlap_threshold)
            ignored_clusters += ignored
            ignored_axial_clusters += ignored_axial
            if significant:
                return True
        return False

    return (collides, lambda: evaluations, lambda: ignored_clusters,
            lambda: ignored_axial_clusters)


def scan_spin_upper_limit(
        parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh,
        axis, pivot, mllm_upper: float, mesh_current_q: float, *,
        persistent_interface: bool = False, step_degrees: float = 1.0,
        confirmation_poses: int = 3, axis_exclusion_radius_ratio: float = 0.15,
        robust_extent_area_ratio: float = 0.95,
        interface_distance_ratio: float = 0.005,
        stationary_meshes=(), moving_meshes=(),
        collision_manager_factory=trimesh.collision.CollisionManager):
    """Clamp a semantic spin upper to the first confirmed collision barrier from q=0."""
    if (not math.isfinite(mllm_upper) or not math.isfinite(mesh_current_q)
            or mllm_upper < 0 or not 0 <= mesh_current_q <= mllm_upper):
        raise ValueError("spin upper scan requires 0 <= mesh_current_q <= mllm_upper")
    if not math.isfinite(step_degrees) or step_degrees <= 0 or confirmation_poses < 1:
        raise ValueError("spin upper scan step and confirmation count must be positive")
    (scan_axis, pivot, radius, projected_diagonal, distance_threshold,
     filtered_parent, filtered_child, parent_face_count, child_face_count) = (
        _spin_collision_regions(
            parent_mesh, child_mesh, axis, pivot,
            axis_exclusion_radius_ratio, robust_extent_area_ratio,
            interface_distance_ratio))
    stationary_meshes, moving_meshes = tuple(stationary_meshes), tuple(moving_meshes)
    collides, evaluations, ignored_clusters, ignored_axial_clusters = (
        _cylinder_collision_probe(
        parent_mesh, child_mesh, filtered_parent, filtered_child,
        stationary_meshes, moving_meshes, scan_axis, pivot,
        distance_threshold, collision_manager_factory,
        include_direct_parent_child=not persistent_interface))

    step = math.radians(step_degrees)
    samples = [mllm_upper - index * step
               for index in range(int(math.floor(mllm_upper / step)) + 1)]
    if samples[-1] > 1e-12:
        samples.append(0.0)
    else:
        samples[-1] = 0.0
    collisions = [collides(q - mesh_current_q) for q in samples]
    runs, start = [], 0
    for index in range(1, len(samples) + 1):
        if index == len(samples) or collisions[index] != collisions[start]:
            count = index - start
            runs.append({
                "upper_degrees": math.degrees(samples[start]),
                "lower_degrees": math.degrees(samples[index - 1]),
                "collides": collisions[start], "samples": count,
                "confirmed": collisions[start] and count >= confirmation_poses,
            })
            start = index

    fallback_reason = None
    final_upper = mllm_upper
    if all(collisions):
        if persistent_interface:
            final_upper = mesh_current_q
            fallback_reason = "all_samples_collide_persistent_external_blocked"
        else:
            fallback_reason = "all_samples_collide_nonpersistent_keep_mllm_upper"
    else:
        confirmed = [run for run in runs if run["confirmed"]]
        if confirmed:
            barrier = min(confirmed, key=lambda run: run["lower_degrees"])
            lower = math.radians(barrier["lower_degrees"])
            final_upper = max(
                (q for q, hit in zip(samples, collisions) if q < lower and not hit),
                default=0.0)
            if final_upper < mesh_current_q:
                final_upper = mesh_current_q
                fallback_reason = "clamped_to_mesh_current_q"

    diagnostics = {
        "persistent_interface": bool(persistent_interface),
        "direct_parent_child_enabled": not persistent_interface,
        "moving_descendant_count": len(moving_meshes),
        "stationary_link_count": len(stationary_meshes),
        "mllm_upper_degrees": math.degrees(mllm_upper),
        "mesh_current_q_degrees": math.degrees(mesh_current_q),
        "final_upper_degrees": math.degrees(final_upper),
        "step_degrees": step_degrees,
        "confirmation_poses": confirmation_poses,
        "runs": runs, "fallback_reason": fallback_reason,
        "evaluations": evaluations(),
        "ignored_subthreshold_contact_clusters": ignored_clusters(),
        "ignored_small_axial_overlap_clusters": ignored_axial_clusters(),
        "exclusion_radius": radius,
        "exclusion_radius_ratio": axis_exclusion_radius_ratio,
        "robust_extent_area_ratio": robust_extent_area_ratio,
        "robust_projected_diagonal": projected_diagonal,
        "contact_area_threshold": distance_threshold ** 2,
        "contact_depth_threshold": distance_threshold,
        "tested_face_counts": {
            "parent": parent_face_count, "child": child_face_count,
        },
    }
    return final_upper, diagnostics


def _closed_revolute_bbox_direction(parent_mesh, child_mesh, axis, pivot):
    """Score opening in the original minimum-area projected bounding-box frame."""
    axis, pivot = np.asarray(axis, dtype=float), np.asarray(pivot, dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)

    def project(vertices):
        relative = np.asarray(vertices, dtype=float) - pivot
        return np.column_stack((relative @ first, relative @ second))

    parent = project(parent_mesh.vertices)
    child = project(child_mesh.vertices)
    try:
        frame, extents = trimesh.bounds.oriented_bounds_2D(
            np.vstack((parent, child)))
    except (QhullError, ValueError):
        return None, {"fallback_reason": "degenerate_original_minimum_bbox"}
    parent = trimesh.transform_points(parent, frame)
    child_closed = trimesh.transform_points(child, frame)

    def box_metrics(child_points):
        overlap = np.maximum(
            0.0,
            np.minimum(parent.max(axis=0), child_points.max(axis=0))
            - np.maximum(parent.min(axis=0), child_points.min(axis=0)),
        )
        overall = np.ptp(np.vstack((parent, child_points)), axis=0)
        return float(np.prod(overlap)), float(np.prod(overall))

    closed_overlap, closed_area = box_metrics(child_closed)
    if not math.isfinite(closed_area) or closed_area <= 1e-12:
        return None, {"fallback_reason": "degenerate_original_minimum_bbox"}

    probes = (5.0, 10.0, 15.0)
    scores = {}
    for direction in (-1, 1):
        sampled = []
        for degrees in probes:
            transform = trimesh.transformations.rotation_matrix(
                math.radians(direction * degrees), axis, pivot)
            moved = trimesh.transform_points(child_mesh.vertices, transform)
            sampled.append(box_metrics(trimesh.transform_points(project(moved), frame)))
        intersection_reduction = float(np.mean([
            closed_overlap - overlap for overlap, _area in sampled
        ]) / closed_area)
        overall_area_growth = float(np.mean([
            area - closed_area for _overlap, area in sampled
        ]) / closed_area)
        scores[direction] = {
            "intersection_reduction": intersection_reduction,
            "overall_area_growth": overall_area_growth,
            "score": intersection_reduction + overall_area_growth,
        }

    best = max(scores, key=lambda direction: scores[direction]["score"])
    difference = scores[best]["score"] - scores[-best]["score"]
    selected = best if scores[best]["score"] > 1e-4 and difference > 1e-4 else None
    return selected, {
        "reference": "original_minimum_rotated_bbox",
        "original_extents": np.asarray(extents, dtype=float).tolist(),
        "original_intersection_area": closed_overlap,
        "original_overall_area": closed_area,
        "probe_degrees": list(probes),
        "directions": {str(key): value for key, value in scores.items()},
        "selected_direction": selected,
    }


def infer_spin_closure_state(
        parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh,
        axis, pivot, *, search_span_degrees: float = 180.0,
        coarse_step_degrees: float = 1.0, angle_tolerance_degrees: float = 0.01,
        axis_exclusion_radius_ratio: float = 0.15,
        robust_extent_area_ratio: float = 0.95,
        interface_distance_ratio: float = 0.005,
        stationary_meshes=(), moving_meshes=(),
        collision_manager_factory=trimesh.collision.CollisionManager) -> SpinClosureState:
    """Find the nearest first visual-mesh contact from the reconstruction pose."""
    if min(search_span_degrees, coarse_step_degrees, angle_tolerance_degrees) <= 0:
        raise ValueError("spin closure search parameters must be positive")
    (scan_axis, pivot, radius, projected_diagonal, distance_threshold,
     filtered_parent, filtered_child, parent_face_count, child_face_count) = (
        _spin_collision_regions(
            parent_mesh, child_mesh, axis, pivot,
            axis_exclusion_radius_ratio, robust_extent_area_ratio,
            interface_distance_ratio))
    stationary_meshes, moving_meshes = tuple(stationary_meshes), tuple(moving_meshes)
    collides, evaluations, ignored_clusters, ignored_axial_clusters = (
        _cylinder_collision_probe(
        parent_mesh, child_mesh, filtered_parent, filtered_child,
        stationary_meshes, moving_meshes, scan_axis, pivot,
        distance_threshold, collision_manager_factory))
    diagnostics = {
        "exclusion_radius_ratio": axis_exclusion_radius_ratio,
        "exclusion_radius_source": "area_weighted_projected_aabb",
        "robust_extent_area_ratio": robust_extent_area_ratio,
        "robust_projected_diagonal": projected_diagonal,
        "contact_area_threshold": distance_threshold ** 2,
        "contact_depth_threshold": distance_threshold,
    }

    def current_diagnostics():
        return {
            **diagnostics,
            "ignored_subthreshold_contact_clusters": ignored_clusters(),
            "ignored_small_axial_overlap_clusters": ignored_axial_clusters(),
        }

    span = math.radians(search_span_degrees)
    step = math.radians(coarse_step_degrees)
    tolerance = math.radians(angle_tolerance_degrees)
    if collides(0.0):
        bbox_direction, bbox_diagnostics = _closed_revolute_bbox_direction(
            parent_mesh, child_mesh, scan_axis, pivot)
        safe_directions = set()
        probe_span = min(span, math.radians(15.0))
        probe_step = min(probe_span, math.radians(1.0))
        for direction in (-1, 1):
            consecutive_safe = 0
            for magnitude in np.arange(probe_step, probe_span + probe_step * 0.5, probe_step):
                consecutive_safe = consecutive_safe + 1 if not collides(direction * magnitude) else 0
                if consecutive_safe == 2:
                    safe_directions.add(direction)
                    break
        direction = (
            bbox_direction if bbox_direction is not None
            else (-1 if safe_directions == {-1} else 1)
        )
        flipped = direction == -1
        method = (
            "original_minimum_bbox"
            if bbox_direction is not None else "collision_probe"
        )
        return SpinClosureState(
            -scan_axis if flipped else scan_axis.copy(), scan_axis.copy(), pivot.copy(),
            0.0, 0.0, 0.0, flipped, radius,
            parent_face_count, child_face_count, evaluations(), span, step, tolerance,
            opening_blocked_at_input=direction not in safe_directions,
            opening_direction_method=method,
            opening_direction_bbox=bbox_diagnostics,
            **current_diagnostics())
    previous, brackets, pending = {-1: 0.0, 1: 0.0}, {}, {-1, 1}
    candidates = {-1: None, 1: None}
    for magnitude in np.arange(step, span + step * 0.5, step):
        for direction in tuple(sorted(pending)):
            angle = float(direction * min(magnitude, span))
            if collides(angle):
                if candidates[direction] is None:
                    candidates[direction] = (previous[direction], angle)
                else:
                    brackets[direction] = candidates[direction]
                    pending.remove(direction)
            else:
                previous[direction] = angle
                candidates[direction] = None
        if not pending:
            break
    if not brackets:
        return SpinClosureState(
            scan_axis.copy(), scan_axis.copy(), pivot.copy(), 0.0, 0.0, None,
            False, radius, parent_face_count, child_face_count, evaluations(),
            span, step, tolerance, **current_diagnostics())

    def refine(bracket):
        safe, first_colliding = bracket
        while abs(first_colliding - safe) > tolerance:
            middle = 0.5 * (safe + first_colliding)
            if collides(middle):
                first_colliding = middle
            else:
                safe = middle
        return safe, first_colliding

    refined = {direction: refine(bracket) for direction, bracket in brackets.items()}
    closing_direction = min(refined, key=lambda direction: abs(refined[direction][1]))
    safe, first_colliding = refined[closing_direction]

    canonical_axis, flipped = scan_axis.copy(), safe > 0
    if flipped:
        canonical_axis = -canonical_axis
        safe, first_colliding = -safe, -first_colliding
    return SpinClosureState(
        canonical_axis, scan_axis.copy(), pivot.copy(), float(-safe), float(safe),
        float(first_colliding), flipped, radius,
        parent_face_count, child_face_count,
        evaluations(), span, step, tolerance,
        **current_diagnostics())
