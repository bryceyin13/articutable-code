#!/usr/bin/env python3
"""Joint differentiable front-view refinement for the fixed-camera pipeline."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from PIL import Image

from psgsr.table_alignment import gltf_y_up_to_z_up


NORMAL_EDGE_THRESHOLD = 0.0
CONTACT_TOLERANCE_M = 0.001
CONTACT_LOSS_WEIGHT = 0.1
ROLL_LIMIT_RAD = math.radians(15.0)
_FIXED_RASTER_LOCK = Lock()
_FIXED_RASTER_CONTEXTS = {}


@dataclass(frozen=True)
class FrontRefinementConfig:
    resolution: int = 256
    iterations: int = 200
    fine_resolution: int = 512
    fine_iterations: int = 100
    lr_xy: float = 0.03
    lr_yaw: float = 0.03
    lr_scale: float = 0.02
    max_xy_fraction: float = 0.10
    scale_limit: float = 2.0
    sdf_weight: float = 1.0
    iou_weight: float = 1.0
    bbox_weight: float = 0.1
    bbox_center_weight: float = 1.0
    edge_weight: float = 0.0
    edge_parameter_scope: str = "all"
    xy_weight: float = 0.005
    scale_weight: float = 0.005
    yaw_weight: float = 0.0001
    loss_combination: str = "geometric_product"
    rgb_weight: float = 0.0
    bbox_geometry: str = "aabb"
    pose_update_mode: str = "raster"
    gradient_clip: float = 5.0
    patience: int = 30
    early_stop_min_delta: float = 1e-4
    improvement_epsilon: float = 1e-6

    def __post_init__(self):
        if self.scale_limit <= 1.0:
            raise ValueError("scale_limit must be greater than 1")
        if self.resolution <= 0 or self.fine_resolution <= 0:
            raise ValueError("front refinement resolutions must be positive")
        if self.iterations < 0 or self.fine_iterations < 0:
            raise ValueError("front refinement iteration counts cannot be negative")
        if not 0.0 < self.max_xy_fraction:
            raise ValueError("max_xy_fraction must be positive")
        if self.bbox_weight < 0.0:
            raise ValueError("bbox_weight cannot be negative")
        if self.bbox_center_weight < 0.0:
            raise ValueError("bbox_center_weight cannot be negative")
        if self.edge_weight < 0.0:
            raise ValueError("edge_weight cannot be negative")
        if self.edge_parameter_scope not in ("all", "yaw_only"):
            raise ValueError(
                f"unknown edge parameter scope: {self.edge_parameter_scope}"
            )
        if self.patience < 0:
            raise ValueError("patience cannot be negative")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("early_stop_min_delta cannot be negative")
        if self.loss_combination not in ("additive", "geometric_product"):
            raise ValueError(f"unknown loss combination: {self.loss_combination}")
        if self.rgb_weight < 0.0:
            raise ValueError("rgb_weight cannot be negative")
        if self.bbox_geometry not in ("aabb", "obb"):
            raise ValueError(f"unknown bbox geometry: {self.bbox_geometry}")
        if self.pose_update_mode not in ("forward_search", "raster"):
            raise ValueError(f"unknown pose update mode: {self.pose_update_mode}")


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def bottom_center_pivot(vertices):
    vertices = np.asarray(vertices, dtype=float)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return np.array([(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, low[2]])


def center_vertices_with_reference(vertices, reference_vertices, centerer):
    """Center render vertices with the offset measured on the full reference mesh."""
    reference_vertices = np.asarray(reference_vertices)
    centered_reference = np.asarray(centerer(reference_vertices))
    offset = reference_vertices[0] - centered_reference[0]
    return np.asarray(vertices) - offset


def native_scene_geometry(scene, child_node, face_count=None):
    vertices, normals, faces, child_mask, offset = [], [], [], [], 0
    nodes = list(scene.graph.nodes_geometry)
    per_node_faces = max(4, face_count // len(nodes)) if face_count else None
    for node in nodes:
        transform, geometry_name = scene.graph.get(node)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        if per_node_faces and len(mesh.faces) > per_node_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=per_node_faces)
            except BaseException:
                pass
        node_vertices = gltf_y_up_to_z_up(np.asarray(mesh.vertices))
        vertices.append(node_vertices)
        normals.append(gltf_y_up_to_z_up(np.asarray(mesh.vertex_normals)))
        faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
        child_mask.extend([node == child_node] * len(node_vertices))
        offset += len(node_vertices)
    if not any(child_mask):
        raise ValueError(f"native animated child node {child_node!r} is missing")
    return (
        np.concatenate(vertices),
        np.concatenate(normals),
        np.concatenate(faces),
        np.asarray(child_mask, dtype=bool),
    )


def _rotation_z(yaw):
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack((
        torch.stack((cosine, -sine, zero)),
        torch.stack((sine, cosine, zero)),
        torch.stack((zero, zero, one)),
    ))


def _rotation_x(roll):
    cosine = torch.cos(roll)
    sine = torch.sin(roll)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack((
        torch.stack((one, zero, zero)),
        torch.stack((zero, cosine, -sine)),
        torch.stack((zero, sine, cosine)),
    ))


def _rotation_y(roll):
    cosine = torch.cos(roll)
    sine = torch.sin(roll)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack((
        torch.stack((cosine, zero, sine)),
        torch.stack((zero, one, zero)),
        torch.stack((-sine, zero, cosine)),
    ))


def _pose_rotation(yaw, roll_x=None, roll_y=None):
    roll_x = torch.zeros_like(yaw) if roll_x is None else roll_x
    roll_y = torch.zeros_like(yaw) if roll_y is None else roll_y
    return _rotation_z(yaw) @ _rotation_y(roll_y) @ _rotation_x(roll_x)


def base_anchor_from_legacy_translation(
    translation_xy, pivot, yaw, scale_xyz, roll_x=None, roll_y=None,
):
    rotated_pivot = _pose_rotation(yaw, roll_x, roll_y) @ (scale_xyz * pivot)
    return translation_xy + rotated_pivot[:2]


def legacy_translation_from_anchor(
    anchor_xy, pivot, yaw, scale_xyz, roll_x=None, roll_y=None,
):
    rotated_pivot = _pose_rotation(yaw, roll_x, roll_y) @ (scale_xyz * pivot)
    return anchor_xy - rotated_pivot[:2]


def legacy_world_vertices(
    vertices, translation_xy, yaw, scale_xyz, roll_x=None, roll_y=None,
):
    world = (vertices * scale_xyz) @ _pose_rotation(
        yaw, roll_x, roll_y,
    ).T
    offset = torch.cat((translation_xy, torch.zeros_like(translation_xy[:1])))
    return world + offset


def pivot_world_vertices(
    vertices, pivot, anchor_xy, yaw, scale_xyz, roll_x=None, roll_y=None,
):
    rotation = _pose_rotation(yaw, roll_x, roll_y)
    world = ((vertices - pivot) * scale_xyz) @ rotation.T
    pivot_offset = rotation @ (scale_xyz * pivot)
    anchor = torch.cat((anchor_xy, pivot_offset[2:]))
    return world + anchor


def support_pose_world_vertices(
    vertices, pivot, anchor_xy, z, roll_x, yaw, scale_xyz, roll_y=None,
):
    rotation = _pose_rotation(yaw, roll_x, roll_y)
    world = ((vertices - pivot) * scale_xyz) @ rotation.T
    rotated_pivot = rotation @ (scale_xyz * pivot)
    anchor = torch.cat((anchor_xy, (z + rotated_pivot[2]).reshape(1)))
    return world + anchor


def vertical_surface_heights(points, support_vertices, support_faces, chunk_size=1024):
    """Return the highest support-triangle Z intersected below each point XY."""
    points = torch.as_tensor(points)
    support_vertices = torch.as_tensor(
        support_vertices, dtype=points.dtype, device=points.device,
    )
    support_faces = torch.as_tensor(
        support_faces, dtype=torch.long, device=points.device,
    )
    best = torch.full(
        (len(points),), float("-inf"), dtype=points.dtype, device=points.device,
    )
    px, py = points[:, 0, None], points[:, 1, None]
    for start in range(0, len(support_faces), chunk_size):
        triangles = support_vertices[support_faces[start:start + chunk_size]]
        ax, ay = triangles[:, 0, 0], triangles[:, 0, 1]
        bx, by = triangles[:, 1, 0], triangles[:, 1, 1]
        cx, cy = triangles[:, 2, 0], triangles[:, 2, 1]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid_triangle = denominator.abs() > 1e-10
        safe_denominator = torch.where(
            valid_triangle, denominator, torch.ones_like(denominator),
        )
        weight_a = (
            (by - cy) * (px - cx) + (cx - bx) * (py - cy)
        ) / safe_denominator
        weight_b = (
            (cy - ay) * (px - cx) + (ax - cx) * (py - cy)
        ) / safe_denominator
        weight_c = 1.0 - weight_a - weight_b
        inside = (
            valid_triangle[None]
            & (weight_a >= -1e-6)
            & (weight_b >= -1e-6)
            & (weight_c >= -1e-6)
        )
        heights = (
            weight_a * triangles[None, :, 0, 2]
            + weight_b * triangles[None, :, 1, 2]
            + weight_c * triangles[None, :, 2, 2]
        )
        best = torch.maximum(
            best,
            torch.where(
                inside, heights, torch.full_like(heights, float("-inf")),
            )
            .amax(dim=1),
        )
    return best, torch.isfinite(best)


def contact_delta_for_support(
    child_vertices_at_base_z, support_vertices, support_faces,
    candidate_count=128,
):
    """Relative Z for support contact, or None when the mesh has no usable hit."""
    support_xy_min = support_vertices[:, :2].amin(dim=0)
    support_xy_max = support_vertices[:, :2].amax(dim=0)
    overlaps_support = (
        (child_vertices_at_base_z[:, :2] >= support_xy_min)
        & (child_vertices_at_base_z[:, :2] <= support_xy_max)
    ).all(dim=1)
    overlapping_indices = torch.nonzero(overlaps_support).flatten()
    if len(overlapping_indices) == 0:
        return None
    overlapping_vertices = child_vertices_at_base_z[overlapping_indices]
    count = min(int(candidate_count), len(overlapping_vertices))
    indices = torch.topk(
        overlapping_vertices[:, 2], count, largest=False,
    ).indices
    candidates = overlapping_vertices[indices]
    heights, valid = vertical_surface_heights(
        candidates, support_vertices, support_faces,
    )
    if not bool(torch.any(valid)):
        return None
    return (heights[valid] - candidates[valid, 2]).amax()


def contact_delta_interval(contact_deltas, tolerance=CONTACT_TOLERANCE_M):
    """Common relative-Z interval keeping every declared support in contact."""
    values = torch.stack(tuple(contact_deltas))
    lower = values.amax() - float(tolerance)
    upper = values.amin()
    violation = torch.relu(lower - upper)
    return lower, upper, violation


def uniform_scale_residual(base_scales, raw_scale, log_scale_limit):
    delta = log_scale_limit * torch.tanh(raw_scale)
    return delta, base_scales * torch.exp(delta)[:, None]


def xy_scale_residual(
    base_scales, raw_scale, log_scale_limit, tie_xy_mask=None, swap_xy_mask=None,
):
    delta = log_scale_limit * torch.tanh(raw_scale)
    if tie_xy_mask is not None:
        tied_delta = delta.mean(dim=1, keepdim=True).expand_as(delta)
        delta = torch.where(tie_xy_mask[:, None], tied_delta, delta)
    local_delta = (
        torch.where(swap_xy_mask[:, None], delta.flip(1), delta)
        if swap_xy_mask is not None else delta
    )
    xy_scales = base_scales[:, :2] * torch.exp(local_delta)
    if tie_xy_mask is not None:
        shared_xy = torch.sqrt(
            (base_scales[:, 0] * base_scales[:, 1]).clamp_min(1e-12)
        ) * torch.exp(delta[:, 0])
        xy_scales = torch.where(tie_xy_mask[:, None], shared_xy[:, None], xy_scales)
    scales = torch.cat((xy_scales, base_scales[:, 2:]), dim=1)
    return delta, scales


def xy_z_scale_residual(base_scales, raw_scale, log_scale_limit):
    delta = log_scale_limit * torch.tanh(raw_scale)
    base_xy = torch.sqrt(
        (base_scales[:, 0] * base_scales[:, 1]).clamp_min(1e-12)
    )
    shared_xy = base_xy * torch.exp(delta[:, 0])
    scale_z = base_scales[:, 2] * torch.exp(delta[:, 1])
    return delta, torch.stack((shared_xy, shared_xy, scale_z), dim=1)


def xyz_scale_residual(base_scales, raw_scale, log_scale_limit, tie_xy_mask=None):
    delta = log_scale_limit * torch.tanh(raw_scale)
    if tie_xy_mask is not None:
        tied_xy = delta[:, :2].mean(dim=1, keepdim=True).expand(-1, 2)
        delta = torch.cat((
            torch.where(tie_xy_mask[:, None], tied_xy, delta[:, :2]),
            delta[:, 2:],
        ), dim=1)
    scales = base_scales * torch.exp(delta)
    xy_scales = scales[:, :2]
    if tie_xy_mask is not None:
        shared_xy = torch.sqrt(
            (base_scales[:, 0] * base_scales[:, 1]).clamp_min(1e-12)
        ) * torch.exp(delta[:, 0])
        xy_scales = torch.where(tie_xy_mask[:, None], shared_xy[:, None], xy_scales)
    return delta, torch.cat((xy_scales, scales[:, 2:]), dim=1)


def sequential_parameter_groups(
    raw_xy, delta_yaw, raw_joint, raw_log_scale,
    scale_phase_name="uniform_scale",
):
    return (
        ("xy_yaw_joint", (raw_xy, delta_yaw, raw_joint)),
        (scale_phase_name, (raw_log_scale,)),
    )


def bbox_center_then_joint_parameter_groups(
    raw_xy, delta_yaw, raw_joint, raw_log_scale,
):
    """First align projected bottom centers, then refine every parameter together."""
    return (
        ("bbox_bottom_center_xy", (raw_xy,)),
        ("joint_xy_yaw_joint_scale", (raw_xy, delta_yaw, raw_joint, raw_log_scale)),
    )


def edge_yaw_only_parameter_groups(
    raw_xy, delta_yaw, raw_joint, raw_log_scale,
):
    return (
        ("base_xy_joint_scale", (raw_xy, raw_joint, raw_log_scale)),
        ("edge_yaw", (delta_yaw,)),
    )


def update_rowwise_best(
    parameters, best_parameters, best_scores, current_scores, epsilon, *, minimize=False,
):
    """Keep one best checkpoint per independently optimized object row."""
    improved = (
        current_scores < best_scores - epsilon
        if minimize else current_scores > best_scores + epsilon
    )
    for parameter, best in zip(parameters, best_parameters):
        mask = improved.reshape((len(improved),) + (1,) * (parameter.ndim - 1))
        best.copy_(torch.where(mask, parameter.detach(), best))
    best_scores.copy_(torch.where(improved, current_scores.detach(), best_scores))
    return improved


def early_stopping_update(current, best, stale_steps, min_delta):
    """Update scalar early-stopping state; patience is handled by the caller."""
    if current < best - min_delta:
        return current, 0
    return best, stale_steps + 1


def early_stopping_update_rows(current, best, stale_steps, active, min_delta):
    """Update independent early-stopping state for active object rows."""
    improved = active & (current < best - min_delta)
    best = torch.where(improved, current, best)
    stale_steps = torch.where(
        active & ~improved, stale_steps + 1, stale_steps,
    )
    return best, stale_steps


def target_bboxes(labels, ids, object_ids, target_mask_fn, device):
    """Return full-resolution target AABB centers/sizes in normalized coordinates."""
    centers = []
    sizes = []
    for object_id in object_ids:
        mask = np.asarray(target_mask_fn(labels, ids, object_id)) > 0
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError(f"empty target mask for {object_id}")
        height, width = mask.shape
        centers.append([
            (float(xs.min() + xs.max()) + 1.0) / (2.0 * max(float(width), 1.0)),
            (float(ys.min() + ys.max()) + 1.0) / (2.0 * max(float(height), 1.0)),
        ])
        sizes.append([
            (float(xs.max() - xs.min()) + 1.0) / max(float(width), 1.0),
            (float(ys.max() - ys.min()) + 1.0) / max(float(height), 1.0),
        ])
    return (
        torch.as_tensor(centers, dtype=torch.float32, device=device),
        torch.as_tensor(sizes, dtype=torch.float32, device=device),
    )


def target_oriented_bboxes(labels, ids, object_ids, target_mask_fn, device):
    """Return target min-area box centers/sizes in each box's own pixel axes."""
    centers, sizes, axes = [], [], []
    for object_id in object_ids:
        mask = np.asarray(target_mask_fn(labels, ids, object_id)) > 0
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError(f"empty target mask for {object_id}")
        points = np.column_stack((xs + 0.5, ys + 0.5)).astype(np.float32)
        _, (width, height), angle = cv2.minAreaRect(points)
        theta = math.radians(float(angle))
        width_axis = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        height_axis = np.array([-width_axis[1], width_axis[0]], dtype=np.float32)
        frame = np.stack(
            (width_axis, height_axis) if width >= height else (height_axis, width_axis),
        )
        coordinates = points @ frame.T
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        centers.append((low + high) * 0.5)
        sizes.append(high - low + 1.0)
        axes.append(frame)
    return tuple(
        torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
        for values in (centers, sizes, axes)
    )


def projected_bbox_centers_sizes(world_vertices, camera):
    image_size = world_vertices[0].new_tensor(camera["image_size"])
    centers, sizes = [], []
    for vertices in world_vertices:
        pixels, depth = torch_project_points(vertices, camera)
        valid = (depth > 1e-8) & torch.isfinite(pixels).all(dim=1)
        low = torch.where(valid[:, None], pixels, torch.full_like(pixels, float("inf"))).amin(dim=0)
        high = torch.where(valid[:, None], pixels, torch.full_like(pixels, float("-inf"))).amax(dim=0)
        centers.append((low + high) / (2.0 * image_size))
        sizes.append((high - low) / image_size)
    return torch.stack(centers), torch.stack(sizes)


def projected_oriented_bbox_centers_sizes(world_vertices, camera, target_axes):
    centers, sizes = [], []
    for vertices, axes in zip(world_vertices, target_axes):
        pixels, depth = torch_project_points(vertices, camera)
        valid = (depth > 1e-8) & torch.isfinite(pixels).all(dim=1)
        coordinates = pixels @ axes.T
        low = torch.where(
            valid[:, None], coordinates, torch.full_like(coordinates, float("inf")),
        ).amin(dim=0)
        high = torch.where(
            valid[:, None], coordinates, torch.full_like(coordinates, float("-inf")),
        ).amax(dim=0)
        centers.append((low + high) * 0.5)
        sizes.append(high - low)
    return torch.stack(centers), torch.stack(sizes)


def projected_bbox_size_losses(world_vertices, camera, target_sizes, eps=1e-6):
    """Compare continuous projected geometry extents with target mask extents."""
    _, predicted_sizes = projected_bbox_centers_sizes(world_vertices, camera)
    return torch.log(
        predicted_sizes.clamp_min(eps) / target_sizes.clamp_min(eps)
    ).square().sum(dim=1)


def projected_bbox_center_losses(
    world_vertices, camera, target_centers, target_sizes, eps=1e-6,
):
    """Penalize projected AABB center offsets relative to target box size."""
    predicted_centers, _ = projected_bbox_centers_sizes(world_vertices, camera)
    return (
        (predicted_centers - target_centers) / target_sizes.clamp_min(eps)
    ).square().sum(dim=1)


def bbox_bottom_center_losses(
    predicted_centers, predicted_sizes, target_centers, target_sizes, eps=1e-6,
):
    """Align horizontal centers and vertical bottom edges in screen space."""
    predicted = torch.stack((
        predicted_centers[:, 0],
        predicted_centers[:, 1] + 0.5 * predicted_sizes[:, 1],
    ), dim=1)
    target = torch.stack((
        target_centers[:, 0],
        target_centers[:, 1] + 0.5 * target_sizes[:, 1],
    ), dim=1)
    return ((predicted - target) / target_sizes.clamp_min(eps)).square().sum(dim=1)


def zero_parameter_rows(parameters, rows):
    """Reset rejected object rows without changing accepted rows."""
    for parameter in parameters:
        mask = rows.reshape((len(rows),) + (1,) * (parameter.ndim - 1))
        parameter.copy_(torch.where(mask, torch.zeros_like(parameter), parameter))


def nonoverlapping_loss_batches(centers, sizes, image_size, padding_pixels=2.0):
    """Greedily group projected boxes whose raster footprints cannot interact."""
    centers = np.asarray(centers, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    padding = float(padding_pixels) / np.asarray(image_size, dtype=float)
    low = centers - 0.5 * sizes - padding
    high = centers + 0.5 * sizes + padding
    batches = []
    for index in range(len(centers)):
        for batch in batches:
            if all(
                high[index, 0] < low[other, 0]
                or high[other, 0] < low[index, 0]
                or high[index, 1] < low[other, 1]
                or high[other, 1] < low[index, 1]
                for other in batch
            ):
                batch.append(index)
                break
        else:
            batches.append([index])
    return tuple(tuple(batch) for batch in batches)


def backward_independent_rows(
    losses, parameters, active_rows=None, loss_batches=None,
):
    """Backprop each object's loss only into that object's parameter row."""
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    active = (
        set(range(len(losses)))
        if active_rows is None else
        set(active_rows.nonzero(as_tuple=False).flatten().tolist())
    )
    batches = [
        tuple(index for index in batch if index in active)
        for batch in (
            loss_batches or tuple((index,) for index in range(len(losses)))
        )
    ]
    batches = [batch for batch in batches if batch]
    for position, batch in enumerate(batches):
        gradients = torch.autograd.grad(
            losses[list(batch)].sum(), parameters,
            retain_graph=position + 1 < len(batches), allow_unused=True,
        )
        for target, gradient in zip(accumulated, gradients):
            if gradient is not None:
                for index in batch:
                    target[index].copy_(gradient[index])
    for parameter, gradient in zip(parameters, accumulated):
        parameter.grad = gradient


def forward_coordinate_search(
    parameter, instance_losses_fn, current_losses, step, *,
    lower=None, upper=None, active_rows=None, improvement_epsilon=1e-8,
):
    """Choose the current, positive, or negative value from actual forward losses."""
    coordinates = tuple(np.ndindex(parameter.shape[1:])) or ((),)
    losses = current_losses.detach().clone()
    row_count = parameter.shape[0]
    step = torch.as_tensor(step, dtype=parameter.dtype, device=parameter.device)
    if step.ndim == 0:
        step = step.expand(row_count)
    active_rows = (
        torch.ones(row_count, dtype=torch.bool, device=parameter.device)
        if active_rows is None else active_rows.to(dtype=torch.bool, device=parameter.device)
    )
    if not bool(torch.any(active_rows)):
        return losses
    lower = None if lower is None else torch.as_tensor(
        lower, dtype=parameter.dtype, device=parameter.device,
    )
    upper = None if upper is None else torch.as_tensor(
        upper, dtype=parameter.dtype, device=parameter.device,
    )
    with torch.no_grad():
        for coordinate in coordinates:
            selection = (slice(None),) + coordinate
            original = parameter[selection].clone()
            try:
                plus_value = original + step
                minus_value = original - step
                if lower is not None:
                    plus_value = torch.maximum(plus_value, lower)
                    minus_value = torch.maximum(minus_value, lower)
                if upper is not None:
                    plus_value = torch.minimum(plus_value, upper)
                    minus_value = torch.minimum(minus_value, upper)
                plus_value = torch.where(active_rows, plus_value, original)
                minus_value = torch.where(active_rows, minus_value, original)
                parameter[selection].copy_(plus_value)
                plus_losses = instance_losses_fn().detach()
                parameter[selection].copy_(minus_value)
                minus_losses = instance_losses_fn().detach()
            finally:
                parameter[selection].copy_(original)
            candidate_losses = torch.stack((losses, plus_losses, minus_losses))
            best_losses, best_indices = candidate_losses.min(dim=0)
            improved = active_rows & (best_losses < losses - improvement_epsilon)
            selected = torch.where(
                best_indices == 1, plus_value,
                torch.where(best_indices == 2, minus_value, original),
            )
            parameter[selection].copy_(torch.where(improved, selected, original))
            losses = torch.where(improved, best_losses, losses)
    return losses


def revolute_vertices(vertices, origin, axis, angle):
    axis = axis / torch.linalg.vector_norm(axis)
    relative = vertices - origin
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return origin + (
        relative * cosine
        + torch.linalg.cross(axis.expand_as(relative), relative) * sine
        + axis * torch.sum(relative * axis, dim=1, keepdim=True) * (1.0 - cosine)
    )


def prismatic_vertices(vertices, axis, displacement):
    axis = axis / torch.linalg.vector_norm(axis)
    return vertices + axis * displacement


def joint_vertices(vertices, origin, axis, position, joint_type):
    if joint_type == "revolute":
        return revolute_vertices(vertices, origin, axis, position)
    if joint_type == "prismatic":
        return prismatic_vertices(vertices, axis, position)
    if joint_type == "fixed":
        return vertices
    raise ValueError(f"unsupported joint type: {joint_type}")


def joint_normals(normals, axis, position, joint_type):
    if joint_type == "revolute":
        return revolute_vertices(
            normals, torch.zeros_like(axis), axis, position,
        )
    if joint_type in ("prismatic", "fixed"):
        return normals
    raise ValueError(f"unsupported joint type: {joint_type}")


def pivot_world_normals(normals, yaw, scale_xyz, roll_x=None, roll_y=None):
    """Apply the inverse-transpose of the object's scale/yaw transform."""
    roll_x = (
        torch.zeros((), dtype=yaw.dtype, device=yaw.device)
        if roll_x is None else roll_x
    )
    roll_y = (
        torch.zeros((), dtype=yaw.dtype, device=yaw.device)
        if roll_y is None else roll_y
    )
    return torch.nn.functional.normalize(
        (normals / scale_xyz) @ _pose_rotation(yaw, roll_x, roll_y).T,
        dim=1,
        eps=1e-6,
    )


def bounded_joint_residual(raw_joint, lower, upper):
    return torch.clamp(raw_joint, min=lower, max=upper)


def _camera_tensors(camera, reference):
    return (
        torch.as_tensor(camera["center_world_m"], dtype=reference.dtype, device=reference.device),
        torch.as_tensor(
            camera["rotation_world_from_camera"],
            dtype=reference.dtype,
            device=reference.device,
        ),
        torch.as_tensor(camera["intrinsic"], dtype=reference.dtype, device=reference.device),
    )


def torch_project_points(points, camera):
    center, world_from_camera, intrinsic = _camera_tensors(camera, points)
    camera_points = (points - center) @ world_from_camera
    depth = camera_points[:, 2]
    if camera.get("projection") == "orthographic":
        u = intrinsic[0, 0] * camera_points[:, 0] + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_points[:, 1] + intrinsic[1, 2]
    else:
        u = intrinsic[0, 0] * camera_points[:, 0] / depth + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_points[:, 1] / depth + intrinsic[1, 2]
    pixels = torch.stack((u, v), dim=1)
    valid = depth > 1e-8
    pixels = torch.where(valid[:, None], pixels, torch.full_like(pixels, float("nan")))
    return pixels, depth


def torch_points_to_clip(points, camera, image_size=None, near=1e-3, far=100.0):
    center, world_from_camera, intrinsic = _camera_tensors(camera, points)
    camera_points = (points - center) @ world_from_camera
    x, y, z = camera_points.unbind(dim=1)
    width, height = image_size or camera["image_size"]
    if camera.get("projection") == "orthographic":
        x_clip = (2.0 * intrinsic[0, 0] / width) * x + 2.0 * intrinsic[0, 2] / width - 1.0
        y_clip = (2.0 * intrinsic[1, 1] / height) * y + 2.0 * intrinsic[1, 2] / height - 1.0
        z_clip = 2.0 * (z - near) / (far - near) - 1.0
        return torch.stack((x_clip, y_clip, z_clip, torch.ones_like(z)), dim=1)
    x_clip = (2.0 * intrinsic[0, 0] / width) * x + (2.0 * intrinsic[0, 2] / width - 1.0) * z
    y_clip = (2.0 * intrinsic[1, 1] / height) * y + (2.0 * intrinsic[1, 2] / height - 1.0) * z
    depth_a = (far + near) / (far - near)
    depth_b = -2.0 * far * near / (far - near)
    z_clip = depth_a * z + depth_b
    return torch.stack((x_clip, y_clip, z_clip, z), dim=1)


def signed_distance_field(mask):
    inside = np.asarray(mask, dtype=bool)
    diagonal = max(math.hypot(*inside.shape), 1.0)
    return (
        distance_transform_edt(~inside) - distance_transform_edt(inside)
    ).astype(np.float32) / diagonal


def front_mask_losses(predicted, target, sdf, eps=1e-6):
    reduce_dims = tuple(range(1, predicted.ndim))
    intersection = torch.sum(predicted * target, dim=reduce_dims)
    union = (
        torch.sum(predicted, dim=reduce_dims)
        + torch.sum(target, dim=reduce_dims)
        - intersection
    )
    iou = (intersection + eps) / (union + eps)
    outside = torch.sum(predicted * torch.relu(sdf), dim=reduce_dims) / (
        torch.sum(predicted, dim=reduce_dims) + eps
    )
    missing = torch.sum((1.0 - predicted) * target * torch.relu(-sdf), dim=reduce_dims) / (
        torch.sum(target, dim=reduce_dims) + eps
    )
    return {"iou": iou, "soft_iou": 1.0 - iou, "sdf": outside + missing}


def refinement_data_loss(
    mask_losses, bbox_losses, rgb_losses, config, edge_losses=None, eps=1e-6,
):
    edge_losses = edge_losses or {
        "iou": torch.ones_like(mask_losses["iou"]),
        "soft_iou": torch.zeros_like(mask_losses["soft_iou"]),
        "sdf": torch.zeros_like(mask_losses["sdf"]),
    }
    if config.loss_combination == "additive":
        return (
            config.sdf_weight * mask_losses["sdf"]
            + config.iou_weight * mask_losses["soft_iou"]
            + config.bbox_weight * bbox_losses
            + config.edge_weight * (
                config.sdf_weight * edge_losses["sdf"]
                + config.iou_weight * edge_losses["soft_iou"]
            )
        )
    edge_log_similarity = (
        0.5 * (
            config.iou_weight * torch.log(edge_losses["external"]["iou"].clamp_min(eps))
            - config.sdf_weight * edge_losses["external"]["sdf"]
        )
        + 0.5 * (
            config.iou_weight * torch.log(edge_losses["internal"]["iou"].clamp_min(eps))
            - config.sdf_weight * edge_losses["internal"]["sdf"]
        )
        if "external" in edge_losses else
        config.iou_weight * torch.log(edge_losses["iou"].clamp_min(eps))
        - config.sdf_weight * edge_losses["sdf"]
    )
    log_similarity = (
        config.iou_weight * torch.log(mask_losses["iou"].clamp_min(eps))
        - config.sdf_weight * mask_losses["sdf"]
        - config.rgb_weight * rgb_losses
        + config.edge_weight * edge_log_similarity
    )
    return (
        1.0 - torch.exp(log_similarity.clamp_max(0.0))
        + config.bbox_weight * bbox_losses
    )


def soft_mask_edges(masks):
    values = masks[:, None]
    maximum = torch.nn.functional.max_pool2d(values, 3, stride=1, padding=1)
    minimum = -torch.nn.functional.max_pool2d(-values, 3, stride=1, padding=1)
    return (maximum - minimum).clamp(0.0, 1.0)[:, 0]


def soft_external_mask_edges(masks):
    """Extract a one-sided one-pixel boundary directly from the rendered mask."""
    eroded = -torch.nn.functional.max_pool2d(
        -masks[:, None], 3, stride=1, padding=1,
    )[:, 0]
    return (masks - eroded).clamp(0.0, 1.0)


def differentiable_vertex_normals(vertices, faces):
    triangles = vertices[faces.long()]
    face_normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner].long(), face_normals)
    return torch.nn.functional.normalize(normals, dim=1, eps=1e-6)


def soft_normal_internal_edges(
    normal_buffer, masks, threshold=NORMAL_EDGE_THRESHOLD,
):
    """One-sided screen-space normal discontinuities inside the silhouette."""
    normal_buffer = torch.nn.functional.normalize(normal_buffer, dim=2, eps=1e-6)
    dx = torch.nn.functional.pad(
        normal_buffer[:, 1:] - normal_buffer[:, :-1], (0, 0, 0, 1),
    )
    dy = torch.nn.functional.pad(
        normal_buffer[1:] - normal_buffer[:-1], (0, 0, 0, 0, 0, 1),
    )
    magnitude = torch.sqrt(torch.sum(dx.square() + dy.square(), dim=2) + 1e-8)
    response = ((magnitude - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    interior = -torch.nn.functional.max_pool2d(
        -masks[:, None], 3, stride=1, padding=1,
    )[:, 0]
    return response[None] * interior


def soft_normal_discontinuity_edges(
    normal_buffer, masks, threshold=NORMAL_EDGE_THRESHOLD,
):
    """Visible silhouette plus differentiable screen-space normal discontinuities."""
    return torch.maximum(
        soft_external_mask_edges(masks),
        soft_normal_internal_edges(normal_buffer, masks, threshold),
    )


def split_target_edge_components(mask_targets, combined_edge_targets):
    external = soft_external_mask_edges(mask_targets)
    external_band = torch.nn.functional.max_pool2d(
        external[:, None], 3, stride=1, padding=1,
    )[:, 0]
    internal = combined_edge_targets * (1.0 - external_band)
    return external, internal


def balanced_edge_component_losses(
    predicted_external, predicted_internal,
    target_external, target_internal,
    external_sdfs, internal_sdfs,
):
    external = front_mask_losses(
        predicted_external, target_external, external_sdfs,
    )
    internal = front_mask_losses(
        predicted_internal, target_internal, internal_sdfs,
    )
    has_internal = torch.sum(
        target_internal, dim=tuple(range(1, target_internal.ndim)),
    ) > 0
    internal = {
        "iou": torch.where(has_internal, internal["iou"], torch.ones_like(internal["iou"])),
        "soft_iou": torch.where(
            has_internal, internal["soft_iou"], torch.zeros_like(internal["soft_iou"]),
        ),
        "sdf": torch.where(has_internal, internal["sdf"], torch.zeros_like(internal["sdf"])),
    }
    combined = {
        key: 0.5 * external[key] + 0.5 * internal[key]
        for key in ("iou", "soft_iou", "sdf")
    }
    return {**combined, "external": external, "internal": internal}


def differentiable_edge_losses(predicted_masks, target_edges, target_edge_sdfs):
    return front_mask_losses(
        soft_external_mask_edges(predicted_masks), target_edges, target_edge_sdfs,
    )


@lru_cache(maxsize=None)
def _source_vertex_texture(path):
    import trimesh

    mesh = trimesh.load(path, force="scene").to_geometry()
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if uv is None or image is None:
        return None
    image = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)[..., :3]
    uv = np.asarray(uv, dtype=np.float32)
    x = np.rint(np.clip(uv[:, 0], 0.0, 1.0) * (image.shape[1] - 1)).astype(int)
    y = np.rint((1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (image.shape[0] - 1)).astype(int)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if Path(path).suffix.lower() in {".glb", ".gltf"}:
        vertices = gltf_y_up_to_z_up(vertices)
    return cKDTree(vertices), image[y, x].astype(np.float32) / 255.0


def mesh_vertex_texture_colors(path, vertices):
    source = _source_vertex_texture(str(Path(path).resolve()))
    if source is None:
        return None
    source_tree, source_colors = source
    indices = source_tree.query(np.asarray(vertices), k=1)[1]
    return source_colors[indices]


def masked_texture_l1(rendered_rgb, target_rgb, predicted_masks, target_masks):
    pixel_error = torch.mean(torch.abs(rendered_rgb - target_rgb), dim=2)
    overlap = predicted_masks * target_masks
    return torch.sum(overlap * pixel_error[None], dim=(1, 2)) / (
        torch.sum(overlap, dim=(1, 2)).clamp_min(1e-6)
    )


def _nvdiffrast():
    try:
        import nvdiffrast.torch as dr
    except ImportError as error:
        raise RuntimeError(
            "nvdiffrast is required for front refinement; install it in the "
            "vggt environment from https://github.com/NVlabs/nvdiffrast"
        ) from error
    return dr


def rasterize_visible_instance_masks(
    clip_vertices,
    triangles,
    vertex_features,
    object_count,
    resolution,
    context=None,
    vertex_rgb=None,
    vertex_normals=None,
    vertex_coordinates=None,
):
    if not clip_vertices.is_cuda:
        raise RuntimeError("nvdiffrast front refinement requires CUDA tensors")
    dr = _nvdiffrast()
    context = context or dr.RasterizeCudaContext(device=clip_vertices.device)
    height, width = (int(resolution), int(resolution)) if isinstance(resolution, int) else resolution
    rast, rast_db = dr.rasterize(
        context,
        clip_vertices[None],
        triangles,
        resolution=[int(height), int(width)],
        grad_db=True,
    )
    attributes, _ = dr.interpolate(
        vertex_features[None], rast, triangles, rast_db=rast_db,
    )
    attributes = dr.antialias(
        attributes, rast, clip_vertices[None], triangles,
    )
    masks = attributes[0, :, :, :object_count].permute(2, 0, 1).clamp(0.0, 1.0)
    if vertex_rgb is None and vertex_normals is None and vertex_coordinates is None:
        return masks
    outputs = [masks]
    if vertex_rgb is not None:
        rgb, _ = dr.interpolate(vertex_rgb[None], rast, triangles, rast_db=rast_db)
        rgb = dr.antialias(rgb, rast, clip_vertices[None], triangles)
        outputs.append(rgb[0].clamp(0.0, 1.0))
    if vertex_normals is not None:
        normals, _ = dr.interpolate(
            vertex_normals[None], rast, triangles, rast_db=rast_db,
        )
        outputs.append(normals[0])
    if vertex_coordinates is not None:
        coordinates, _ = dr.interpolate(
            vertex_coordinates[None], rast, triangles, rast_db=rast_db,
        )
        outputs.append(coordinates[0])
    return tuple(outputs)


def restore_supported_regions_in_rendered_masks(
    visible_masks, full_parent_masks, children_by_parent,
):
    """Ignore direct supported-object occlusion in support-parent mask losses."""
    restored = list(visible_masks.unbind(0))
    for parent_index, child_indices in children_by_parent.items():
        child_mask = torch.stack([
            visible_masks[index] for index in child_indices
        ]).amax(dim=0)
        restored[parent_index] = torch.maximum(
            restored[parent_index],
            torch.minimum(full_parent_masks[parent_index], child_mask),
        )
    return torch.stack(restored)


def _fixed_raster_context(device):
    key = (device.type, device.index)
    context = _FIXED_RASTER_CONTEXTS.get(key)
    if context is None:
        context = _nvdiffrast().RasterizeCudaContext(device=device)
        _FIXED_RASTER_CONTEXTS[key] = context
    return context


def _render_normal_edge_buffers_nvdiffrast_unlocked(
    vertices, faces, camera, vertex_foreground=None, vertex_normals=None,
    vertex_rgb=None, return_normal_map=False, vertex_coordinates=None,
):
    """Render a fixed mesh's mask, normal edges, and optional vertex RGB."""
    if not torch.cuda.is_available():
        raise RuntimeError("normal-edge rendering requires CUDA")
    device = torch.device("cuda")
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.int32, device=device)
    width, height = map(int, camera["image_size"])
    clip = torch_points_to_clip(vertices, camera)
    features = torch.as_tensor(
        (
            np.ones((len(vertices), 1), dtype=np.float32)
            if vertex_foreground is None else
            np.asarray(vertex_foreground, dtype=np.float32).reshape(-1, 1)
        ),
        dtype=torch.float32, device=device,
    )
    normals = (
        differentiable_vertex_normals(vertices, faces)
        if vertex_normals is None else
        torch.as_tensor(vertex_normals, dtype=torch.float32, device=device)
    )
    rendered = rasterize_visible_instance_masks(
        clip, faces, features, 1, (height, width),
        context=_fixed_raster_context(device),
        vertex_rgb=(
            torch.as_tensor(vertex_rgb, dtype=torch.float32, device=device)
            if vertex_rgb is not None else None
        ),
        vertex_normals=normals,
        vertex_coordinates=(
            torch.as_tensor(
                vertex_coordinates, dtype=torch.float32, device=device,
            )
            if vertex_coordinates is not None else None
        ),
    )
    output_index = 1
    masks = rendered[0]
    rgb = rendered[output_index] if vertex_rgb is not None else None
    output_index += int(vertex_rgb is not None)
    normal_buffer = rendered[output_index]
    output_index += 1
    coordinate_buffer = (
        rendered[output_index] if vertex_coordinates is not None else None
    )
    edges = soft_normal_discontinuity_edges(normal_buffer, masks)[0]
    outputs = (
        masks[0].detach().cpu().numpy() > 0.5,
        edges.detach().cpu().numpy() > 0.25,
        (
            np.rint(rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)
            if rgb is not None else None
        ),
    )
    if not return_normal_map:
        return outputs
    world_from_camera = torch.as_tensor(
        camera["rotation_world_from_camera"],
        dtype=normal_buffer.dtype, device=normal_buffer.device,
    )
    camera_normals = torch.nn.functional.normalize(
        normal_buffer @ world_from_camera, dim=2, eps=1e-6,
    )
    camera_normals = torch.where(
        camera_normals[..., 2:3] < 0.0, -camera_normals, camera_normals,
    )
    normal_map = torch.round((camera_normals + 1.0) * 127.5).clamp(0, 255)
    normal_map = torch.where(
        masks[0, ..., None] > 0.5,
        normal_map,
        torch.full_like(normal_map, 127.0),
    ).to(torch.uint8)
    result = (*outputs, normal_map.detach().cpu().numpy())
    if coordinate_buffer is not None:
        result = (*result, coordinate_buffer.detach().cpu().numpy())
    return result


def render_normal_edge_buffers_nvdiffrast(
    vertices, faces, camera, vertex_foreground=None, vertex_normals=None,
    vertex_rgb=None, return_normal_map=False, vertex_coordinates=None,
):
    """Render fixed candidates through one reusable nvdiffrast CUDA context."""
    with _FIXED_RASTER_LOCK:
        return _render_normal_edge_buffers_nvdiffrast_unlocked(
            vertices, faces, camera, vertex_foreground, vertex_normals,
            vertex_rgb, return_normal_map, vertex_coordinates,
        )


def render_normal_edge_buffers_batch_nvdiffrast(candidates, camera, chunk_size=4):
    """Render unrelated fixed meshes in range-mode tensor batches."""
    candidates = list(candidates)
    if not candidates:
        return []
    if chunk_size < 1:
        raise ValueError("fixed candidate render chunk size must be positive")
    outputs = []
    with _FIXED_RASTER_LOCK:
        for start in range(0, len(candidates), int(chunk_size)):
            outputs.extend(_render_normal_edge_batch_unlocked(
                candidates[start:start + int(chunk_size)], camera,
            ))
    return outputs


def _render_normal_edge_batch_unlocked(candidates, camera):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("normal-edge rendering requires CUDA")
    outputs = []
    dr = _nvdiffrast()
    device = torch.device("cuda")
    width, height = map(int, camera["image_size"])
    candidate_cameras = [item.get("camera", camera) for item in candidates]
    if any(
        list(map(int, value["image_size"])) != [width, height]
        for value in candidate_cameras
    ):
        raise ValueError("batched fixed-candidate cameras must share one image size")
    vertices = [
        torch.as_tensor(item["vertices"], dtype=torch.float32, device=device)
        for item in candidates
    ]
    faces = [
        torch.as_tensor(item["faces"], dtype=torch.int32, device=device)
        for item in candidates
    ]
    clips = [
        torch_points_to_clip(value, value_camera)
        for value, value_camera in zip(vertices, candidate_cameras)
    ]
    vertex_offsets = np.cumsum([0, *[len(value) for value in vertices[:-1]]])
    triangle_offsets = np.cumsum([0, *[len(value) for value in faces[:-1]]])
    combined_vertices = torch.cat(vertices)
    combined_clip = torch.cat(clips)
    combined_faces = torch.cat([
        value + int(offset) for value, offset in zip(faces, vertex_offsets)
    ])
    ranges = torch.as_tensor([
        [int(offset), len(value)]
        for value, offset in zip(faces, triangle_offsets)
    ], dtype=torch.int32, device="cpu")
    rast, rast_db = dr.rasterize(
        _fixed_raster_context(device), combined_clip, combined_faces,
        resolution=[height, width], ranges=ranges, grad_db=True,
    )

    foreground = torch.cat([
        torch.ones((len(value), 1), dtype=torch.float32, device=device)
        if item.get("vertex_foreground") is None else
        torch.as_tensor(
            item["vertex_foreground"], dtype=torch.float32, device=device,
        ).reshape(-1, 1)
        for item, value in zip(candidates, vertices)
    ])
    masks, _ = dr.interpolate(
        foreground, rast, combined_faces, rast_db=rast_db,
    )
    masks = dr.antialias(
        masks, rast, combined_clip, combined_faces,
    )[..., 0].clamp(0.0, 1.0)

    normals = torch.cat([
        differentiable_vertex_normals(value, face)
        if item.get("vertex_normals") is None else
        torch.as_tensor(
            item["vertex_normals"], dtype=torch.float32, device=device,
        )
        for item, value, face in zip(candidates, vertices, faces)
    ])
    normal_buffers, _ = dr.interpolate(
        normals, rast, combined_faces, rast_db=rast_db,
    )
    rgb_requested = any(item.get("vertex_rgb") is not None for item in candidates)
    rgb_buffers = None
    if rgb_requested:
        rgb = torch.cat([
            torch.zeros((len(value), 3), dtype=torch.float32, device=device)
            if item.get("vertex_rgb") is None else
            torch.as_tensor(item["vertex_rgb"], dtype=torch.float32, device=device)
            for item, value in zip(candidates, vertices)
        ])
        rgb_buffers, _ = dr.interpolate(
            rgb, rast, combined_faces, rast_db=rast_db,
        )
        rgb_buffers = dr.antialias(
            rgb_buffers, rast, combined_clip, combined_faces,
        ).clamp(0.0, 1.0)

    coordinates_requested = any(
        item.get("vertex_coordinates") is not None for item in candidates
    )
    coordinate_buffers = None
    if coordinates_requested:
        coordinates = torch.cat([
            torch.zeros_like(value)
            if item.get("vertex_coordinates") is None else
            torch.as_tensor(
                item["vertex_coordinates"], dtype=torch.float32, device=device,
            )
            for item, value in zip(candidates, vertices)
        ])
        coordinate_buffers, _ = dr.interpolate(
            coordinates, rast, combined_faces, rast_db=rast_db,
        )

    world_from_camera = torch.as_tensor([
        value["rotation_world_from_camera"] for value in candidate_cameras
    ], dtype=normal_buffers.dtype, device=device)
    camera_normals = torch.nn.functional.normalize(
        torch.einsum("bhwc,bcd->bhwd", normal_buffers, world_from_camera),
        dim=3, eps=1e-6,
    )
    camera_normals = torch.where(
        camera_normals[..., 2:3] < 0.0, -camera_normals, camera_normals,
    )
    normal_maps = torch.round((camera_normals + 1.0) * 127.5).clamp(0, 255)
    normal_maps = torch.where(
        masks[..., None] > 0.5,
        normal_maps,
        torch.full_like(normal_maps, 127.0),
    ).to(torch.uint8)

    for index, item in enumerate(candidates):
        mask = masks[index:index + 1]
        edges = soft_normal_discontinuity_edges(
            normal_buffers[index], mask,
        )[0]
        result = (
            mask[0].detach().cpu().numpy() > 0.5,
            edges.detach().cpu().numpy() > 0.25,
            (
                np.rint(rgb_buffers[index].detach().cpu().numpy() * 255.0)
                .astype(np.uint8)
                if item.get("vertex_rgb") is not None else None
            ),
            normal_maps[index].detach().cpu().numpy(),
        )
        if item.get("vertex_coordinates") is not None:
            result = (*result, coordinate_buffers[index].detach().cpu().numpy())
        outputs.append(result)
    return outputs


def render_camera_normal_map_nvdiffrast(vertices, faces, camera):
    """Render a uint8 camera-space normal map with a neutral background."""
    mask, _, _, normal_map = render_normal_edge_buffers_nvdiffrast(
        vertices, faces, camera, return_normal_map=True,
    )
    return mask, normal_map


def render_normal_edges_nvdiffrast(
    vertices, faces, camera, vertex_foreground=None, vertex_normals=None,
):
    """Render method-3 normal discontinuity edges for a fixed mesh pose."""
    return render_normal_edge_buffers_nvdiffrast(
        vertices, faces, camera, vertex_foreground, vertex_normals,
    )[1]


def _scaled_camera(camera, width, height):
    source_width, source_height = camera["image_size"]
    intrinsic = np.asarray(camera["intrinsic"], dtype=float).copy()
    intrinsic[0, :] *= width / source_width
    intrinsic[1, :] *= height / source_height
    return {**camera, "image_size": [width, height], "intrinsic": intrinsic.tolist()}


def _optimization_size(camera, resolution):
    width, height = camera["image_size"]
    factor = float(resolution) / max(width, height)
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


def _resize_mask(mask, width, height, preserve_thin=False):
    values = np.asarray(mask)
    maximum = float(values.max()) if values.size else 0.0
    is_soft = (
        values.dtype != bool
        and np.any((values > 0) & (values < maximum))
    )
    image = Image.fromarray(
        values.astype(np.uint8)
        if is_soft else
        np.where(values > 0, 255, 0).astype(np.uint8)
    )
    resampling = (
        Image.Resampling.BOX
        if preserve_thin and (width < image.width or height < image.height)
        else Image.Resampling.NEAREST
    )
    resized = np.asarray(image.resize((width, height), resampling), dtype=np.float32)
    return (
        resized / 255.0
        if is_soft or not preserve_thin else
        (resized > 0.0).astype(np.float32)
    )


def _mask_tensors(
    labels, ids, object_ids, target_mask_fn, camera, resolution, device,
    preserve_thin=False,
):
    width, height = _optimization_size(camera, resolution)
    masks = np.stack([
        _resize_mask(
            target_mask_fn(labels, ids, object_id), width, height,
            preserve_thin=preserve_thin,
        )
        for object_id in object_ids
    ])
    sdfs = np.stack([signed_distance_field(mask) for mask in masks])
    return (
        torch.as_tensor(masks, dtype=torch.float32, device=device),
        torch.as_tensor(sdfs, dtype=torch.float32, device=device),
        _scaled_camera(camera, width, height),
    )


def _mask_grid(masks):
    masks = np.asarray(masks, dtype=float)
    panels = [np.repeat(np.clip(mask * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=2) for mask in masks]
    return np.concatenate(panels, axis=1)


def _overlay_grid(predicted, target):
    panels = []
    for current, reference in zip(np.asarray(predicted), np.asarray(target)):
        current = current > 0.5
        reference = reference > 0.5
        image = np.zeros((*current.shape, 3), dtype=np.uint8)
        image[reference] = (40, 220, 60)
        image[current] = (255, 60, 60)
        image[current & reference] = (255, 255, 255)
        panels.append(image)
    return np.concatenate(panels, axis=1)


def _save_edge_debug_outputs(output_dir, object_ids, initial_edges, final_edges, targets):
    directory = Path(output_dir) / "edge_visualization"
    directory.mkdir(parents=True, exist_ok=True)
    for object_id, initial, final, reference in zip(
        object_ids, initial_edges, final_edges, targets,
    ):
        initial, final, reference = (
            np.asarray(values) for values in (initial, final, reference)
        )
        occupied = (initial > 0.05) | (final > 0.05) | (reference > 0.05)
        rows, columns = np.nonzero(occupied)
        if len(rows):
            padding = 4
            y0 = max(0, int(rows.min()) - padding)
            y1 = min(occupied.shape[0], int(rows.max()) + padding + 1)
            x0 = max(0, int(columns.min()) - padding)
            x1 = min(occupied.shape[1], int(columns.max()) + padding + 1)
            initial, final, reference = (
                values[y0:y1, x0:x1]
                for values in (initial, final, reference)
            )

        def save_gray(name, values):
            image = np.clip(np.asarray(values) * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(directory / name)

        save_gray(f"{object_id}_reference_external_edge.png", reference)
        save_gray(f"{object_id}_initial_3d_model_edge.png", initial)
        save_gray(f"{object_id}_final_3d_model_edge.png", final)
        Image.fromarray(_overlay_grid(initial[None], reference[None])).save(
            directory / f"{object_id}_initial_overlay.png"
        )
        Image.fromarray(_overlay_grid(final[None], reference[None])).save(
            directory / f"{object_id}_final_overlay.png"
        )


def _save_debug_outputs(
    output_dir, initial_masks, final_masks, targets, history, parameters,
    visualization_kind="mask", object_ids=(), intermediate_masks=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_mask_grid(initial_masks)).save(output_dir / "initial_masks.png")
    Image.fromarray(_mask_grid(final_masks)).save(output_dir / "final_masks.png")
    Image.fromarray(_mask_grid(targets)).save(output_dir / "reference_masks.png")
    Image.fromarray(_overlay_grid(initial_masks, targets)).save(output_dir / "initial_overlay.png")
    Image.fromarray(_overlay_grid(final_masks, targets)).save(output_dir / "final_overlay.png")
    for name, masks in (intermediate_masks or {}).items():
        Image.fromarray(_mask_grid(masks)).save(output_dir / f"after_{name}_masks.png")
        Image.fromarray(_overlay_grid(masks, targets)).save(
            output_dir / f"after_{name}_overlay.png"
        )
    if visualization_kind == "edge_drawing":
        _save_edge_debug_outputs(
            output_dir, object_ids, initial_masks, final_masks, targets,
        )

    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in history], [row["total_loss"] for row in history])
    axis.set(xlabel="optimization step", ylabel="loss", title="Differentiable front refinement")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(figure)
    (output_dir / "parameters.json").write_text(
        json.dumps(json_value(parameters), indent=2) + "\n", encoding="utf-8",
    )


def refine_front_masks_differentiable(
    *,
    top_objects,
    front_camera,
    front_labels,
    front_ids,
    object_ids,
    table_mesh_path,
    table_scale_xyz,
    mesh_loader,
    object_vertex_centerer,
    table_vertex_centerer,
    target_mask_fn,
    output_dir,
    config=None,
    view_name="front",
    scale_mode="uniform",
    joint_pose_scale=False,
    tie_xy_scale_ids=(),
    loss_mode="mask",
    target_edge_fn=None,
    rgb_loss_fns=None,
    texture_target_rgb=None,
    object_optimization_mode="joint",
    optimization_schedule="legacy_sequential",
    support_pose_refinement=False,
    optimize_internal_joints=False,
    initial_yaw_offset_deg=0.0,
    yaw_delta_bounds_deg_by_object=None,
    restore_supported_occlusions=False,
):
    config = config or FrontRefinementConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast front refinement")
    _nvdiffrast()
    device = torch.device("cuda")
    dtype = torch.float32
    active_ids = tuple(
        object_id for object_id in object_ids
        if object_id in top_objects and object_id in front_ids
    )
    if not active_ids:
        raise ValueError("front refinement has no objects with instance masks")
    if object_optimization_mode not in ("joint", "independent_parallel"):
        raise ValueError(f"unknown object optimization mode: {object_optimization_mode}")
    if optimization_schedule not in ("legacy_sequential", "bbox_center_then_joint"):
        raise ValueError(f"unknown optimization schedule: {optimization_schedule}")
    if scale_mode not in ("uniform", "xy", "xy_z", "xyz"):
        raise ValueError(f"unknown scale mode: {scale_mode}")
    if config.edge_parameter_scope == "yaw_only" and loss_mode == "edge_drawing":
        raise ValueError("yaw-only edge routing requires mask_edge or mask loss")
    independent_parallel = object_optimization_mode == "independent_parallel"
    yaw_bounds_enabled = yaw_delta_bounds_deg_by_object is not None
    yaw_delta_bounds_deg_by_object = yaw_delta_bounds_deg_by_object or {}
    yaw_delta_bounds = [
        yaw_delta_bounds_deg_by_object.get(object_id, (-float("inf"), float("inf")))
        for object_id in active_ids
    ]
    if any(lower > upper for lower, upper in yaw_delta_bounds):
        raise ValueError("yaw delta lower bound must not exceed upper bound")
    if loss_mode not in ("mask", "edge_drawing", "mask_edge"):
        raise ValueError(f"unknown differentiable refinement loss mode: {loss_mode}")
    if loss_mode in ("edge_drawing", "mask_edge") and target_edge_fn is None:
        raise ValueError(f"{loss_mode} loss requires target_edge_fn")
    rgb_loss_fns = rgb_loss_fns or {}
    tie_xy_scale_ids = frozenset(tie_xy_scale_ids)
    tie_xy_mask = torch.as_tensor(
        [object_id in tie_xy_scale_ids for object_id in active_ids],
        dtype=torch.bool, device=device,
    )
    bbox_targets = (
        (
            target_oriented_bboxes(
                front_labels, front_ids, active_ids, target_mask_fn, device,
            )
            if config.bbox_geometry == "obb" else
            target_bboxes(
                front_labels, front_ids, active_ids, target_mask_fn, device,
            )
        )
        if config.bbox_weight > 0.0 or optimization_schedule == "bbox_center_then_joint"
        else None
    )
    bottom_center_targets = (
        target_bboxes(
            front_labels, front_ids, active_ids, target_mask_fn, device,
        )
        if optimization_schedule == "bbox_center_then_joint" else None
    )

    canonical_vertices = []
    canonical_normals = []
    object_faces = []
    child_vertex_masks = []
    joint_origins = []
    joint_axes = []
    joint_lowers = []
    joint_uppers = []
    base_joints = []
    joint_types = []
    joint_active = []
    pivots = []
    base_yaws = []
    base_scales = []
    base_anchors = []
    base_zs = []
    base_roll_x = []
    base_roll_y = []
    support_parents_by_object = []
    z_active_values = []
    roll_x_active_values = []
    roll_y_active_values = []
    object_vertex_rgb = []
    texture_loss_skips = {}
    use_vertex_texture = (
        texture_target_rgb is not None
        and config.rgb_weight > 0.0
        and config.loss_combination == "geometric_product"
    )
    for object_id in active_ids:
        initial = top_objects[object_id]
        articulation = initial.get("articulation")
        if articulation:
            object_vertex_rgb.append(None)
            if use_vertex_texture:
                texture_loss_skips[object_id] = "articulated_texture_mapping_not_supported"
            if articulation.get("geometry_asset") and articulation.get("child_node"):
                import trimesh

                native_scene = trimesh.load(articulation["geometry_asset"], force="scene")
                raw_canonical, raw_normals, articulated_faces, child_mask = native_scene_geometry(
                    native_scene, articulation["child_node"], face_count=None,
                )
                reference_canonical = raw_canonical
            else:
                vertices, normals, reference_vertices, faces, child_mask, offset = [], [], [], [], [], 0
                child_name = articulation["joint"]["child"]
                for link in articulation["links"]:
                    mesh = mesh_loader(Path(link["geometry_asset"]), face_count=None)
                    reference_mesh = mesh
                    link_vertices = np.asarray(mesh.vertices, dtype=np.float32)
                    vertices.append(link_vertices)
                    normals.append(np.asarray(mesh.vertex_normals, dtype=np.float32))
                    reference_vertices.append(np.asarray(reference_mesh.vertices, dtype=np.float32))
                    faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
                    child_mask.extend([link["name"] == child_name] * len(link_vertices))
                    offset += len(link_vertices)
                raw_canonical = np.concatenate(vertices)
                raw_normals = np.concatenate(normals)
                reference_canonical = np.concatenate(reference_vertices)
                articulated_faces = np.concatenate(faces)
            canonical = np.asarray(center_vertices_with_reference(
                raw_canonical, reference_canonical, object_vertex_centerer,
            ), dtype=np.float32)
            centered_reference = np.asarray(object_vertex_centerer(reference_canonical))
            center_offset = reference_canonical[0] - centered_reference[0]
            object_faces.append(articulated_faces)
            child_vertex_masks.append(torch.as_tensor(child_mask, dtype=torch.bool, device=device))
            joint = articulation["joint"]
            joint_origins.append(torch.as_tensor(
                np.asarray(joint["origin"], dtype=np.float32) - center_offset,
                dtype=dtype, device=device,
            ))
            joint_axes.append(torch.as_tensor(joint["axis"], dtype=dtype, device=device))
            joint_type = joint.get("type", "revolute")
            state = initial.get("joint_state", {})
            state_field = (
                "relative_angle_rad" if joint_type == "revolute"
                else "scene_current_q"
            )
            base_joint = float(state.get(
                state_field,
                state.get(
                    "relative_displacement",
                    joint.get(
                        "scene_current_q",
                        joint.get("requested_scene_q", 0.0),
                    ),
                ),
            ))
            limits = joint.get("limits", joint.get("limits_rad"))
            joint_lowers.append(float(limits[0]) - base_joint)
            joint_uppers.append(float(limits[1]) - base_joint)
            base_joints.append(base_joint)
            joint_types.append(joint_type)
            joint_active.append(1.0)
        else:
            mesh = mesh_loader(Path(initial["geometry_asset"]), face_count=None)
            reference_mesh = mesh
            try:
                colors = (
                    mesh_vertex_texture_colors(initial["geometry_asset"], mesh.vertices)
                    if use_vertex_texture else None
                )
            except Exception as error:
                colors = None
                texture_loss_skips[object_id] = f"texture_load_failed:{type(error).__name__}"
            object_vertex_rgb.append(colors)
            if use_vertex_texture and colors is None:
                texture_loss_skips.setdefault(object_id, "missing_uv_texture")
            canonical = np.asarray(center_vertices_with_reference(
                mesh.vertices, reference_mesh.vertices, object_vertex_centerer,
            ), dtype=np.float32)
            raw_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            object_faces.append(np.asarray(mesh.faces, dtype=np.int32))
            child_vertex_masks.append(torch.zeros(len(canonical), dtype=torch.bool, device=device))
            joint_origins.append(torch.zeros(3, dtype=dtype, device=device))
            joint_axes.append(torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device))
            joint_lowers.append(0.0)
            joint_uppers.append(0.0)
            base_joints.append(0.0)
            joint_types.append("fixed")
            joint_active.append(0.0)
        pivot = bottom_center_pivot(canonical).astype(np.float32)
        translation = np.asarray(initial["translation_world_m"][:2], dtype=np.float32)
        yaw = math.radians(float(initial["yaw_deg"]))
        scale = (
            np.asarray(initial.get("scale_xyz", [initial["uniform_scale"]] * 3), dtype=np.float32)
            if scale_mode in ("xy", "xy_z", "xyz")
            else np.full(3, float(initial["uniform_scale"]), dtype=np.float32)
        )
        vertices_tensor = torch.as_tensor(canonical, dtype=dtype, device=device)
        pivot_tensor = torch.as_tensor(pivot, dtype=dtype, device=device)
        translation_tensor = torch.as_tensor(translation, dtype=dtype, device=device)
        yaw_tensor = torch.tensor(yaw, dtype=dtype, device=device)
        scale_tensor = torch.as_tensor(scale, dtype=dtype, device=device)
        roll_x_tensor = torch.tensor(
            math.radians(float(initial.get("roll_x_deg", 0.0))),
            dtype=dtype, device=device,
        )
        roll_y_tensor = torch.tensor(
            math.radians(float(initial.get("roll_y_deg", 0.0))),
            dtype=dtype, device=device,
        )
        anchor = base_anchor_from_legacy_translation(
            translation_tensor, pivot_tensor, yaw_tensor, scale_tensor,
            roll_x_tensor, roll_y_tensor,
        )
        legacy = legacy_world_vertices(
            vertices_tensor, translation_tensor, yaw_tensor, scale_tensor,
            roll_x_tensor, roll_y_tensor,
        )
        pivoted = pivot_world_vertices(
            vertices_tensor, pivot_tensor, anchor, yaw_tensor, scale_tensor,
            roll_x_tensor, roll_y_tensor,
        )
        zero_error = float(torch.max(torch.abs(legacy - pivoted)).detach().cpu())
        if zero_error >= 1e-6:
            raise AssertionError(
                f"zero-residual pivot mismatch for {object_id}: {zero_error:.3g} m"
            )
        canonical_vertices.append(vertices_tensor)
        canonical_normals.append(torch.as_tensor(
            raw_normals, dtype=dtype, device=device,
        ))
        pivots.append(pivot_tensor)
        base_yaws.append(yaw_tensor)
        base_scales.append(scale_tensor)
        base_anchors.append(anchor)
        parents = tuple(initial.get("support_parents") or ("table_0",))
        mode = initial.get("support_pose_mode", "disabled")
        support_parents_by_object.append(parents)
        base_zs.append(float(initial["translation_world_m"][2]))
        base_roll_x.append(roll_x_tensor)
        base_roll_y.append(roll_y_tensor)
        z_active_values.append(
            support_pose_refinement
            and mode in {"z_only", "z_and_roll_x", "z_and_roll_xy"}
        )
        multi_support = (
            support_pose_refinement
            and mode in {"z_and_roll_x", "z_and_roll_xy"}
            and len(parents) > 1
        )
        roll_x_active_values.append(multi_support)
        roll_y_active_values.append(multi_support)

    base_yaws = torch.stack(base_yaws)
    base_scales = torch.stack(base_scales)
    base_anchors = torch.stack(base_anchors)
    base_zs = torch.as_tensor(base_zs, dtype=dtype, device=device)
    base_roll_x = torch.stack(base_roll_x)
    base_roll_y = torch.stack(base_roll_y)
    z_active = torch.as_tensor(z_active_values, dtype=dtype, device=device)
    roll_x_active = torch.as_tensor(
        roll_x_active_values, dtype=dtype, device=device,
    )
    roll_y_active = torch.as_tensor(
        roll_y_active_values, dtype=dtype, device=device,
    )
    joint_origins = torch.stack(joint_origins)
    joint_axes = torch.stack(joint_axes)
    joint_lowers = torch.as_tensor(joint_lowers, dtype=dtype, device=device)
    joint_uppers = torch.as_tensor(joint_uppers, dtype=dtype, device=device)
    base_joints = torch.as_tensor(base_joints, dtype=dtype, device=device)
    joint_prismatic = torch.as_tensor(
        [joint_type == "prismatic" for joint_type in joint_types],
        dtype=torch.bool, device=device,
    )
    joint_active = torch.as_tensor(joint_active, dtype=dtype, device=device)
    swap_xy_scale = torch.zeros(len(active_ids), dtype=torch.bool, device=device)
    if scale_mode == "xy" and config.bbox_geometry == "obb":
        for index, (pivot, anchor, yaw, scale) in enumerate(zip(
            pivots, base_anchors, base_yaws, base_scales,
        )):
            axis_points = torch.stack((
                pivot, pivot + pivot.new_tensor([1.0, 0.0, 0.0]),
                pivot + pivot.new_tensor([0.0, 1.0, 0.0]),
            ))
            world_axis_points = pivot_world_vertices(
                axis_points, pivot, anchor, yaw, scale,
            )
            pixels, _ = torch_project_points(world_axis_points, front_camera)
            x_direction = pixels[1] - pixels[0]
            y_direction = pixels[2] - pixels[0]
            target_major_axis = bbox_targets[2][index, 0]
            swap_xy_scale[index] = (
                torch.abs(torch.dot(y_direction, target_major_axis))
                > torch.abs(torch.dot(x_direction, target_major_axis))
            )
    table_mesh = mesh_loader(Path(table_mesh_path))
    table_vertices_np = np.asarray(table_vertex_centerer(table_mesh.vertices), dtype=np.float32)
    table_vertices_np *= np.asarray(table_scale_xyz, dtype=np.float32)
    table_vertices = torch.as_tensor(table_vertices_np, dtype=dtype, device=device)
    table_normals = torch.nn.functional.normalize(
        torch.as_tensor(
            np.asarray(table_mesh.vertex_normals, dtype=np.float32),
            dtype=dtype, device=device,
        ) / torch.as_tensor(table_scale_xyz, dtype=dtype, device=device),
        dim=1,
        eps=1e-6,
    )
    table_width, table_depth = np.ptp(table_vertices_np[:, :2], axis=0)
    max_xy = torch.tensor(
        [config.max_xy_fraction * table_width, config.max_xy_fraction * table_depth],
        dtype=dtype, device=device,
    )
    if torch.any(max_xy <= 0):
        raise ValueError("table width and depth must be positive")

    active_index = {
        object_id: index for index, object_id in enumerate(active_ids)
    }
    children_by_parent = {}
    if restore_supported_occlusions:
        for child_index, parents in enumerate(support_parents_by_object):
            for parent_id in parents:
                parent_index = active_index.get(parent_id)
                if parent_id != "table_0" and parent_index is not None:
                    children_by_parent.setdefault(parent_index, []).append(child_index)
    support_parent_triangles = {
        index: torch.as_tensor(
            object_faces[index], dtype=torch.int32, device=device,
        )
        for index in children_by_parent
    }
    support_parent_features = {
        index: torch.ones(
            (len(canonical_vertices[index]), 1), dtype=dtype, device=device,
        )
        for index in children_by_parent
    }
    fixed_support_geometry = {
        "table_0": (
            table_vertices,
            torch.as_tensor(
                np.asarray(table_mesh.faces, dtype=np.int32),
                dtype=torch.long, device=device,
            ),
        ),
    }
    referenced_supports = {
        parent
        for index, parents in enumerate(support_parents_by_object)
        if z_active_values[index]
        for parent in parents
        if parent != "table_0" and parent not in active_index
    }
    for parent_id in referenced_supports:
        if parent_id not in top_objects:
            raise ValueError(f"support object is missing from scene: {parent_id}")
        item = top_objects[parent_id]
        mesh = mesh_loader(Path(item["geometry_asset"]), face_count=None)
        canonical = np.asarray(center_vertices_with_reference(
            mesh.vertices, mesh.vertices, object_vertex_centerer,
        ), dtype=np.float32)
        vertices = torch.as_tensor(canonical, dtype=dtype, device=device)
        pivot = torch.as_tensor(
            bottom_center_pivot(canonical), dtype=dtype, device=device,
        )
        scale = torch.as_tensor(
            item.get("scale_xyz", [item["uniform_scale"]] * 3),
            dtype=dtype, device=device,
        )
        yaw = torch.tensor(
            math.radians(float(item["yaw_deg"])), dtype=dtype, device=device,
        )
        roll_x = torch.tensor(
            math.radians(float(item.get("roll_x_deg", 0.0))),
            dtype=dtype, device=device,
        )
        roll_y = torch.tensor(
            math.radians(float(item.get("roll_y_deg", 0.0))),
            dtype=dtype, device=device,
        )
        translation = torch.as_tensor(
            item["translation_world_m"][:2], dtype=dtype, device=device,
        )
        anchor = base_anchor_from_legacy_translation(
            translation, pivot, yaw, scale, roll_x, roll_y,
        )
        fixed_support_geometry[parent_id] = (
            support_pose_world_vertices(
                vertices, pivot, anchor,
                torch.tensor(
                    float(item["translation_world_m"][2]),
                    dtype=dtype, device=device,
                ),
                roll_x, yaw, scale, roll_y,
            ),
            torch.as_tensor(
                np.asarray(mesh.faces, dtype=np.int32),
                dtype=torch.long, device=device,
            ),
        )

    vertex_features = []
    triangles = []
    texture_features = []
    offset = 0
    channel_count = len(active_ids) + 1
    for index, (vertices, faces) in enumerate(zip(canonical_vertices, object_faces)):
        feature = torch.zeros((len(vertices), channel_count), dtype=dtype, device=device)
        feature[:, index] = 1.0
        vertex_features.append(feature)
        colors = object_vertex_rgb[index]
        texture_features.append(torch.as_tensor(
            colors if colors is not None else np.zeros((len(vertices), 3), dtype=np.float32),
            dtype=dtype, device=device,
        ))
        triangles.append(torch.as_tensor(faces + offset, dtype=torch.int32, device=device))
        offset += len(vertices)
    table_feature = torch.zeros((len(table_vertices), channel_count), dtype=dtype, device=device)
    table_feature[:, -1] = 1.0
    vertex_features.append(table_feature)
    texture_features.append(torch.zeros((len(table_vertices), 3), dtype=dtype, device=device))
    triangles.append(torch.as_tensor(
        np.asarray(table_mesh.faces, dtype=np.int32) + offset,
        dtype=torch.int32, device=device,
    ))
    vertex_features = torch.cat(vertex_features)
    texture_features = torch.cat(texture_features) if use_vertex_texture else None
    triangles = torch.cat(triangles)

    raw_xy = torch.nn.Parameter(torch.zeros((len(active_ids), 2), device=device))
    fixed_yaw_offset = math.radians(float(initial_yaw_offset_deg))
    delta_yaw = torch.nn.Parameter(torch.zeros(len(active_ids), device=device))
    yaw_delta_lowers = torch.tensor(
        [math.radians(float(bounds[0])) for bounds in yaw_delta_bounds], device=device,
    )
    yaw_delta_uppers = torch.tensor(
        [math.radians(float(bounds[1])) for bounds in yaw_delta_bounds], device=device,
    )
    with torch.no_grad():
        delta_yaw.clamp_(yaw_delta_lowers, yaw_delta_uppers)
    raw_joint = torch.nn.Parameter(
        torch.zeros(len(active_ids), device=device),
        requires_grad=optimize_internal_joints,
    )
    raw_z = torch.nn.Parameter(torch.zeros(len(active_ids), device=device))
    raw_roll_x = torch.nn.Parameter(torch.zeros(len(active_ids), device=device))
    raw_roll_y = torch.nn.Parameter(torch.zeros(len(active_ids), device=device))
    scale_parameter_width = {"xy": 2, "xy_z": 2, "xyz": 3}.get(scale_mode)
    raw_log_scale = torch.nn.Parameter(torch.zeros(
        (
            (len(active_ids), scale_parameter_width)
            if scale_parameter_width is not None
            else len(active_ids)
        ),
        device=device,
    ))
    scale_residual_axes = (
        ("x", "y", "z") if scale_mode == "xyz"
        else ("xy", "z") if scale_mode == "xy_z"
        else ()
    )
    parameters = (
        raw_xy, delta_yaw, raw_joint, raw_log_scale,
        raw_z, raw_roll_x, raw_roll_y,
    )
    pose_optimizer = torch.optim.Adam([
        {"params": [raw_xy], "lr": config.lr_xy},
        {"params": [delta_yaw], "lr": config.lr_yaw},
        {"params": [raw_joint], "lr": config.lr_yaw},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    scale_optimizer = torch.optim.Adam([
        {"params": [raw_log_scale], "lr": config.lr_scale},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    joint_optimizer = torch.optim.Adam([
        {"params": [raw_xy], "lr": config.lr_xy},
        {"params": [delta_yaw, raw_joint], "lr": config.lr_yaw},
        {"params": [raw_log_scale], "lr": config.lr_scale},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    bbox_joint_optimizer = torch.optim.Adam([
        {"params": [raw_xy], "lr": config.lr_xy},
        {"params": [delta_yaw, raw_joint], "lr": config.lr_yaw},
        {"params": [raw_log_scale], "lr": config.lr_scale},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    bbox_center_optimizer = torch.optim.Adam([
        {"params": [raw_xy], "lr": config.lr_xy},
    ])
    edge_base_optimizer = torch.optim.Adam([
        {"params": [raw_xy], "lr": config.lr_xy},
        {"params": [raw_joint], "lr": config.lr_yaw},
        {"params": [raw_log_scale], "lr": config.lr_scale},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    edge_yaw_optimizer = torch.optim.Adam([
        {"params": [delta_yaw], "lr": config.lr_yaw},
        {"params": [raw_z], "lr": config.lr_xy},
        {"params": [raw_roll_x, raw_roll_y], "lr": config.lr_yaw},
    ])
    pose_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pose_optimizer, T_max=max(config.iterations, 1), eta_min=0.0,
    )
    scale_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        scale_optimizer, T_max=max(config.fine_iterations, 1), eta_min=0.0,
    )
    joint_scheduler = (
        None
        if view_name == "front"
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            joint_optimizer, T_max=max(config.fine_iterations, 1), eta_min=0.0,
        )
    )
    bbox_center_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        bbox_center_optimizer, T_max=max(config.fine_iterations, 1), eta_min=0.0,
    )
    edge_base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        edge_base_optimizer, T_max=max(config.iterations, 1), eta_min=0.0,
    )
    edge_yaw_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        edge_yaw_optimizer, T_max=max(config.fine_iterations, 1), eta_min=0.0,
    )
    bbox_joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        bbox_joint_optimizer, T_max=max(config.iterations, 1), eta_min=0.0,
    )
    dr = _nvdiffrast()
    context = dr.RasterizeCudaContext(device=device)
    log_scale_limit = math.log(config.scale_limit)
    render_inputs = {}
    edge_render_inputs = {}
    edge_component_inputs = {}
    texture_mask_inputs = {}
    texture_rgb_inputs = {}
    latest_loss_batches = tuple((index,) for index in range(len(active_ids)))

    def current_state():
        normalized_xy = torch.tanh(raw_xy)
        delta_xy = normalized_xy * max_xy
        if scale_mode == "xy":
            delta_log_scale, scales = xy_scale_residual(
                base_scales, raw_log_scale, log_scale_limit,
                tie_xy_mask=tie_xy_mask, swap_xy_mask=swap_xy_scale,
            )
        elif scale_mode == "xy_z":
            delta_log_scale, scales = xy_z_scale_residual(
                base_scales, raw_log_scale, log_scale_limit,
            )
        elif scale_mode == "xyz":
            delta_log_scale, scales = xyz_scale_residual(
                base_scales, raw_log_scale, log_scale_limit,
                tie_xy_mask=tie_xy_mask,
            )
        else:
            delta_log_scale, scales = uniform_scale_residual(
                base_scales, raw_log_scale, log_scale_limit,
            )
        delta_joint = bounded_joint_residual(raw_joint, joint_lowers, joint_uppers) * joint_active
        joints = base_joints + delta_joint
        yaws = base_yaws + fixed_yaw_offset + delta_yaw
        anchors = base_anchors + delta_xy
        free_delta_z = 0.05 * torch.tanh(raw_z) * z_active
        delta_roll_x = ROLL_LIMIT_RAD * torch.tanh(raw_roll_x) * roll_x_active
        delta_roll_y = ROLL_LIMIT_RAD * torch.tanh(raw_roll_y) * roll_y_active
        rolls_x = base_roll_x + delta_roll_x
        rolls_y = base_roll_y + delta_roll_y
        return (
            delta_xy, delta_log_scale, delta_joint, joints, scales, yaws,
            anchors, normalized_xy, free_delta_z, delta_roll_x, delta_roll_y,
            rolls_x, rolls_y,
        )

    support_contact_fallbacks = set()

    def current_scene_state():
        (
            delta_xy, delta_log_scale, delta_joint, joints, scales, yaws,
            anchors, normalized_xy, free_delta_z, delta_roll_x, delta_roll_y,
            rolls_x, rolls_y,
        ) = current_state()
        posed_vertices = []
        posed_normals = []
        for vertices, normals, child_mask, origin, axis, joint, joint_type in zip(
            canonical_vertices, canonical_normals, child_vertex_masks,
            joint_origins, joint_axes, joints, joint_types,
        ):
            moved = joint_vertices(vertices, origin, axis, joint, joint_type)
            posed_vertices.append(torch.where(child_mask[:, None], moved, vertices))
            moved_normals = joint_normals(normals, axis, joint, joint_type)
            posed_normals.append(torch.where(
                child_mask[:, None], moved_normals, normals,
            ))

        world_vertices = [None] * len(active_ids)
        final_delta_z = torch.zeros_like(free_delta_z)
        contact_violations = torch.zeros_like(free_delta_z)
        contact_spreads = torch.zeros_like(free_delta_z)
        contact_deltas_by_object = [()] * len(active_ids)
        resolving = set()

        def resolve_world(index):
            if world_vertices[index] is not None:
                return world_vertices[index]
            if index in resolving:
                raise ValueError("support hierarchy is cyclic")
            resolving.add(index)
            base_world = support_pose_world_vertices(
                posed_vertices[index], pivots[index], anchors[index],
                base_zs[index], rolls_x[index], yaws[index], scales[index],
                rolls_y[index],
            )
            if not bool(z_active[index]):
                world = base_world
            else:
                contact_deltas = []
                for parent_id in support_parents_by_object[index]:
                    parent_index = active_index.get(parent_id)
                    if parent_index is None:
                        support_vertices, support_faces = fixed_support_geometry[
                            parent_id
                        ]
                    else:
                        support_vertices = resolve_world(parent_index)
                        support_faces = object_faces[parent_index]
                    delta = contact_delta_for_support(
                        base_world, support_vertices.detach(), support_faces,
                    )
                    if delta is None:
                        support_contact_fallbacks.add(
                            (active_ids[index], parent_id),
                        )
                    else:
                        contact_deltas.append(delta)
                if not contact_deltas:
                    world = base_world
                    world_vertices[index] = world
                    resolving.remove(index)
                    return world
                lower, upper, violation = contact_delta_interval(contact_deltas)
                feasible = lower <= upper
                constrained = torch.clamp(
                    free_delta_z[index], min=lower, max=upper,
                )
                selected_delta = torch.where(
                    feasible, constrained, 0.5 * (lower + upper),
                )
                offset = torch.stack((
                    selected_delta * 0.0,
                    selected_delta * 0.0,
                    selected_delta,
                ))
                world = base_world + offset
                final_delta_z[index] = selected_delta
                contact_violations[index] = violation
                values = torch.stack(contact_deltas)
                contact_spreads[index] = values.amax() - values.amin()
                contact_deltas_by_object[index] = tuple(contact_deltas)
            world_vertices[index] = world
            resolving.remove(index)
            return world

        world_vertices = [
            resolve_world(index) for index in range(len(active_ids))
        ]
        world_normals = [
            pivot_world_normals(normals, yaw, scale, roll_x, roll_y)
            for normals, yaw, scale, roll_x, roll_y in zip(
                posed_normals, yaws, scales, rolls_x, rolls_y,
            )
        ]
        return (
            delta_xy, delta_log_scale, delta_joint, joints, scales, yaws,
            anchors, normalized_xy, final_delta_z, delta_roll_x, delta_roll_y,
            rolls_x, rolls_y, world_vertices, world_normals,
            contact_violations, contact_spreads, contact_deltas_by_object,
        )

    def render(resolution, objective="full"):
        nonlocal latest_loss_batches
        if resolution not in render_inputs:
            render_inputs[resolution] = _mask_tensors(
                front_labels, front_ids, active_ids,
                target_edge_fn if loss_mode == "edge_drawing" else target_mask_fn,
                front_camera, resolution, device,
                preserve_thin=loss_mode == "edge_drawing",
            )
        targets, sdfs, camera = render_inputs[resolution]
        edge_targets = edge_sdfs = None
        if loss_mode == "mask_edge":
            if resolution not in edge_render_inputs:
                edge_render_inputs[resolution] = _mask_tensors(
                    front_labels, front_ids, active_ids, target_edge_fn,
                    front_camera, resolution, device, preserve_thin=True,
                )
            edge_targets, edge_sdfs, _ = edge_render_inputs[resolution]
            if resolution not in edge_component_inputs:
                external_targets, internal_targets = split_target_edge_components(
                    targets, edge_targets,
                )
                component_sdfs = tuple(
                    torch.as_tensor(
                        np.stack([
                            signed_distance_field(mask)
                            for mask in component.detach().cpu().numpy() > 0.5
                        ]),
                        dtype=dtype, device=device,
                    )
                    for component in (external_targets, internal_targets)
                )
                edge_component_inputs[resolution] = (
                    external_targets, internal_targets, *component_sdfs,
                )
        (
            delta_xy, delta_log_scale, delta_joint, joints, scales, yaws,
            anchors, normalized_xy, delta_z, delta_roll_x, delta_roll_y,
            rolls_x, rolls_y, world_vertices, world_normals,
            contact_violations, _, _,
        ) = current_scene_state()
        with torch.no_grad():
            projected_centers, projected_sizes = projected_bbox_centers_sizes(
                world_vertices, camera,
            )
            latest_loss_batches = nonoverlapping_loss_batches(
                projected_centers.cpu().numpy(), projected_sizes.cpu().numpy(),
                camera["image_size"],
            )
        clip_vertices = torch_points_to_clip(
            torch.cat([*world_vertices, table_vertices]), camera,
        )
        render_rgb = texture_features is not None and objective == "full"
        render_normals = loss_mode in ("edge_drawing", "mask_edge")
        vertex_normals = (
            torch.cat([*world_normals, table_normals])
            if render_normals else None
        )
        rasterized = rasterize_visible_instance_masks(
            clip_vertices, triangles, vertex_features, len(active_ids),
            resolution=(targets.shape[1], targets.shape[2]), context=context,
            vertex_rgb=texture_features if render_rgb else None,
            vertex_normals=vertex_normals,
        )
        if render_rgb and render_normals:
            masks, rendered_rgb, rendered_normals = rasterized
        elif render_rgb:
            masks, rendered_rgb = rasterized
            rendered_normals = None
        elif render_normals:
            masks, rendered_normals = rasterized
            rendered_rgb = None
        else:
            masks, rendered_rgb, rendered_normals = rasterized, None, None
        if children_by_parent:
            full_parent_masks = {
                index: rasterize_visible_instance_masks(
                    torch_points_to_clip(world_vertices[index], camera),
                    support_parent_triangles[index],
                    support_parent_features[index],
                    1,
                    resolution=(targets.shape[1], targets.shape[2]),
                    context=context,
                )[0]
                for index in children_by_parent
            }
            masks = restore_supported_regions_in_rendered_masks(
                masks, full_parent_masks, children_by_parent,
            )
        model_external_edges = (
            soft_external_mask_edges(masks) if render_normals else None
        )
        model_internal_edges = (
            soft_normal_internal_edges(rendered_normals, masks)
            if render_normals else None
        )
        model_edges = (
            torch.maximum(model_external_edges, model_internal_edges)
            if render_normals else None
        )
        comparison = (
            model_edges
            if loss_mode in ("edge_drawing", "mask_edge") else masks
        )
        mask_losses = (
            front_mask_losses(model_edges, targets, sdfs)
            if loss_mode == "edge_drawing"
            else front_mask_losses(masks, targets, sdfs)
        )
        edge_losses = (
            balanced_edge_component_losses(
                model_external_edges, model_internal_edges,
                *edge_component_inputs[resolution],
            )
            if loss_mode == "mask_edge" else None
        )
        if bbox_targets is not None:
            predicted_centers, predicted_sizes = (
                projected_oriented_bbox_centers_sizes(
                    world_vertices, front_camera, bbox_targets[2],
                )
                if config.bbox_geometry == "obb" else
                projected_bbox_centers_sizes(world_vertices, front_camera)
            )
            bbox_size_losses = torch.log(
                predicted_sizes.clamp_min(1e-6) / bbox_targets[1].clamp_min(1e-6)
            ).square().sum(dim=1)
            bbox_center_losses = (
                (predicted_centers - bbox_targets[0])
                / bbox_targets[1].clamp_min(1e-6)
            ).square().sum(dim=1)
        else:
            bbox_size_losses = torch.zeros_like(mask_losses["soft_iou"])
            bbox_center_losses = torch.zeros_like(mask_losses["soft_iou"])
        bbox_losses = bbox_size_losses + config.bbox_center_weight * bbox_center_losses
        bottom_center_losses = (
            bbox_bottom_center_losses(
                *projected_bbox_centers_sizes(world_vertices, front_camera),
                *bottom_center_targets,
            )
            if bottom_center_targets is not None
            else torch.zeros_like(mask_losses["soft_iou"])
        )
        rgb_losses = torch.stack([
            rgb_loss_fns[object_id].batch_metrics(delta_yaw[index])[1]
            if object_id in rgb_loss_fns else torch.zeros((), device=device)
            for index, object_id in enumerate(active_ids)
        ])
        if rendered_rgb is not None:
            if loss_mode == "edge_drawing":
                if resolution not in texture_mask_inputs:
                    texture_mask_inputs[resolution] = _mask_tensors(
                        front_labels, front_ids, active_ids, target_mask_fn,
                        front_camera, resolution, device,
                    )[0]
                texture_masks = texture_mask_inputs[resolution]
            else:
                texture_masks = targets
            if resolution not in texture_rgb_inputs:
                width, height = targets.shape[2], targets.shape[1]
                resized = cv2.resize(
                    np.asarray(texture_target_rgb, dtype=np.float32) / 255.0,
                    (width, height), interpolation=cv2.INTER_AREA,
                )
                texture_rgb_inputs[resolution] = torch.as_tensor(
                    resized, dtype=dtype, device=device,
                )
            texture_losses = masked_texture_l1(
                rendered_rgb, texture_rgb_inputs[resolution], masks, texture_masks,
            )
            texture_active = torch.as_tensor(
                [colors is not None for colors in object_vertex_rgb],
                dtype=dtype, device=device,
            )
            rgb_losses = rgb_losses + texture_active * texture_losses
        xy_regularization = torch.sum(normalized_xy * normalized_xy, dim=1)
        scale_regularization = delta_log_scale * delta_log_scale
        if scale_regularization.ndim == 2:
            scale_regularization = scale_regularization.sum(dim=1)
        yaw_regularization = 1.0 - torch.cos(delta_yaw)
        support_pose_regularization = delta_z.square() + (
            1.0 - torch.cos(delta_roll_x)
        ) + (
            1.0 - torch.cos(delta_roll_y)
        )
        contact_losses = (
            contact_violations / CONTACT_TOLERANCE_M
        ).square()
        joint_span = (joint_uppers - joint_lowers).abs().clamp_min(1e-6)
        joint_regularization = torch.where(
            joint_prismatic,
            (delta_joint / joint_span).square(),
            1.0 - torch.cos(delta_joint),
        )
        if objective == "bbox_bottom_center":
            instance_loss = bottom_center_losses
        elif objective in ("base", "full"):
            instance_loss = (
                refinement_data_loss(
                    mask_losses, bbox_losses, rgb_losses, config,
                    edge_losses=edge_losses if objective == "full" else None,
                )
                + config.xy_weight * xy_regularization
                + config.scale_weight * scale_regularization
                + config.yaw_weight * yaw_regularization
                + config.yaw_weight * joint_regularization
                + config.xy_weight * support_pose_regularization
                + CONTACT_LOSS_WEIGHT * contact_losses
            )
        else:
            raise ValueError(f"unknown refinement objective: {objective}")
        diagnostics_target = edge_targets if edge_targets is not None else targets
        return (
            instance_loss.mean(), comparison, diagnostics_target,
            mask_losses["iou"], instance_loss,
        )

    with torch.no_grad():
        initial_loss, _, _, initial_iou_coarse, initial_instance_losses = render(
            config.resolution,
        )
    history = [{
        "step": 0,
        "phase": "initial",
        "resolution": int(config.resolution),
        "total_loss": float(initial_loss.cpu()),
        "mean_iou": float(initial_iou_coarse.mean().cpu()),
        **{
            f"iou_{object_id}": float(value)
            for object_id, value in zip(active_ids, initial_iou_coarse.cpu())
        },
        **{
            f"loss_{object_id}": float(value)
            for object_id, value in zip(active_ids, initial_instance_losses.cpu())
        },
        **({
            f"scale_{axis}_{object_id}": 0.0
            for object_id in active_ids
            for axis in scale_residual_axes
        } if scale_residual_axes else {}),
    }]
    global_step = 0

    edge_yaw_only = (
        config.edge_parameter_scope == "yaw_only"
        and loss_mode == "mask_edge"
        and config.edge_weight > 0.0
    )
    if edge_yaw_only:
        parameter_groups = edge_yaw_only_parameter_groups(
            raw_xy, delta_yaw, raw_joint, raw_log_scale,
        )
        parameter_groups = tuple(
            (name, (*phase_parameters, raw_z, raw_roll_x, raw_roll_y))
            for name, phase_parameters in parameter_groups
        )
        base_iterations = (
            config.fine_iterations if joint_pose_scale else config.iterations
        )
        yaw_iterations = config.fine_iterations or base_iterations
        phase_settings = (
            (
                config.fine_resolution, base_iterations,
                edge_base_optimizer, edge_base_scheduler, "base",
            ),
            (
                config.fine_resolution, yaw_iterations,
                edge_yaw_optimizer, edge_yaw_scheduler, "full",
            ),
        )
        if (
            not joint_pose_scale
            and optimization_schedule == "bbox_center_then_joint"
        ):
            parameter_groups = (
                ("bbox_bottom_center_xy", (raw_xy,)),
                *parameter_groups,
            )
            phase_settings = (
                (
                    config.resolution, config.fine_iterations,
                    bbox_center_optimizer, bbox_center_scheduler,
                    "bbox_bottom_center",
                ),
                *phase_settings,
            )
    elif joint_pose_scale:
        parameter_groups = ((
            (
                "joint_pose_independent_xyz_scale"
                if scale_mode == "xyz"
                else "joint_pose_shared_xy_independent_z_scale"
                if scale_mode == "xy_z"
                else "xy_yaw_local_xy_scale"
            ),
            (
                raw_xy, delta_yaw, raw_joint, raw_log_scale,
                raw_z, raw_roll_x, raw_roll_y,
            ),
        ),)
        phase_settings = ((
            config.fine_resolution, config.fine_iterations,
            joint_optimizer, joint_scheduler, "full",
        ),)
    elif optimization_schedule == "bbox_center_then_joint":
        parameter_groups = bbox_center_then_joint_parameter_groups(
            raw_xy, delta_yaw, raw_joint, raw_log_scale,
        )
        parameter_groups = (
            parameter_groups[0],
            (
                parameter_groups[1][0],
                (*parameter_groups[1][1], raw_z, raw_roll_x, raw_roll_y),
            ),
        )
        phase_settings = (
            (
                config.resolution, config.fine_iterations,
                bbox_center_optimizer, bbox_center_scheduler, "bbox_bottom_center",
            ),
            (
                config.fine_resolution, config.iterations,
                bbox_joint_optimizer, bbox_joint_scheduler, "full",
            ),
        )
    else:
        parameter_groups = sequential_parameter_groups(
            raw_xy, delta_yaw, raw_joint, raw_log_scale,
            scale_phase_name=(
                "independent_xyz_scale"
                if scale_mode == "xyz"
                else "shared_xy_independent_z_scale"
                if scale_mode == "xy_z"
                else "uniform_scale"
            ),
        )
        parameter_groups = tuple(
            (name, (*phase_parameters, raw_z, raw_roll_x, raw_roll_y))
            for name, phase_parameters in parameter_groups
        )
        phase_settings = (
            (config.resolution, config.iterations, pose_optimizer, pose_scheduler, "full"),
            (
                config.fine_resolution, config.fine_iterations,
                scale_optimizer, scale_scheduler, "full",
            ),
        )
    if not optimize_internal_joints:
        parameter_groups = tuple(
            (name, tuple(parameter for parameter in active_parameters if parameter is not raw_joint))
            for name, active_parameters in parameter_groups
        )
    phase_actual_iterations = {}
    early_stopped_phases = []
    phase_object_stop_steps = {}
    phase_parameter_snapshots = []
    for (phase_name, active_parameters), (
        resolution, iterations, optimizer, scheduler, objective,
    ) in zip(
        parameter_groups,
        phase_settings,
    ):
        with torch.no_grad():
            _, _, _, phase_iou, phase_instance_losses = render(resolution, objective)
        minimize_checkpoint = (
            config.bbox_weight > 0.0 or objective == "bbox_bottom_center"
        )
        phase_scores = phase_instance_losses if minimize_checkpoint else phase_iou
        best_scores = (
            phase_scores.detach().clone()
            if independent_parallel else float(phase_scores.mean().cpu())
        )
        best_parameters = [parameter.detach().clone() for parameter in parameters]
        if independent_parallel:
            row_best_losses = torch.full_like(phase_instance_losses, float("inf"))
            row_stale_steps = torch.zeros_like(
                phase_instance_losses, dtype=torch.long,
            )
            active_rows = torch.ones_like(
                phase_instance_losses, dtype=torch.bool,
            )
        else:
            phase_best_loss = float("inf")
            stale_steps = 0
        phase_start_step = global_step
        phase_object_stop_steps[phase_name] = {}
        early_stopped = False
        search_step_multiplier = 4.0
        phase_iterations = (
            min(iterations, 24)
            if config.pose_update_mode == "forward_search" else iterations
        )
        for phase_iteration in range(phase_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, iou, instance_losses = render(resolution, objective)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite differentiable front loss at step {global_step}"
                )
            current_loss = float(loss.detach().cpu())
            mean_iou = float(iou.detach().mean().cpu())
            iteration_scale = current_state()[1].detach().cpu()
            scale_gradients = None
            if independent_parallel:
                update_rowwise_best(
                    parameters, best_parameters, best_scores,
                    (instance_losses if minimize_checkpoint else iou).detach(),
                    config.improvement_epsilon, minimize=minimize_checkpoint,
                )
            else:
                current_score = (
                    current_loss if minimize_checkpoint else mean_iou
                )
                improved = (
                    current_score < best_scores - config.improvement_epsilon
                    if minimize_checkpoint
                    else current_score > best_scores + config.improvement_epsilon
                )
            if not independent_parallel and improved:
                best_scores = current_score
                best_parameters = [parameter.detach().clone() for parameter in parameters]
            if independent_parallel:
                row_best_losses, row_stale_steps = early_stopping_update_rows(
                    instance_losses.detach(), row_best_losses, row_stale_steps,
                    active_rows, config.early_stop_min_delta,
                )
                if config.patience > 0:
                    previously_active = active_rows.clone()
                    active_rows &= row_stale_steps < config.patience
                    for index in (
                        previously_active & ~active_rows
                    ).nonzero(as_tuple=False).flatten().tolist():
                        phase_object_stop_steps[phase_name][
                            active_ids[index]
                        ] = global_step + 1
                    early_stopped = not bool(torch.any(active_rows))
            else:
                phase_best_loss, stale_steps = early_stopping_update(
                    current_loss, phase_best_loss, stale_steps,
                    config.early_stop_min_delta,
                )
                early_stopped = (
                    config.patience > 0 and stale_steps >= config.patience
                )
            if config.pose_update_mode == "forward_search":
                candidate_losses = instance_losses.detach()
                candidate_fn = lambda: render(resolution, objective)[4]
                if any(raw_xy is active for active in active_parameters):
                    candidate_losses = forward_coordinate_search(
                        raw_xy, candidate_fn, candidate_losses,
                        0.005 * search_step_multiplier,
                        active_rows=active_rows if independent_parallel else None,
                        improvement_epsilon=config.improvement_epsilon,
                    )
                if any(delta_yaw is active for active in active_parameters):
                    candidate_losses = forward_coordinate_search(
                        delta_yaw, candidate_fn, candidate_losses,
                        math.radians(0.5) * search_step_multiplier,
                        lower=yaw_delta_lowers, upper=yaw_delta_uppers,
                        active_rows=active_rows if independent_parallel else None,
                        improvement_epsilon=config.improvement_epsilon,
                    )
                if any(raw_joint is active for active in active_parameters):
                    joint_steps = (
                        0.01 * (joint_uppers - joint_lowers).abs().clamp_min(1e-6)
                        * search_step_multiplier
                    )
                    candidate_losses = forward_coordinate_search(
                        raw_joint, candidate_fn, candidate_losses, joint_steps,
                        lower=joint_lowers, upper=joint_uppers,
                        active_rows=(
                            (joint_active > 0) & active_rows
                            if independent_parallel else joint_active > 0
                        ),
                        improvement_epsilon=config.improvement_epsilon,
                    )
                if any(raw_log_scale is active for active in active_parameters):
                    candidate_losses = forward_coordinate_search(
                        raw_log_scale, candidate_fn, candidate_losses,
                        0.005 * search_step_multiplier,
                        active_rows=active_rows if independent_parallel else None,
                        improvement_epsilon=config.improvement_epsilon,
                    )
                search_improved = bool(torch.any(
                    (
                        candidate_losses
                        < instance_losses.detach() - config.improvement_epsilon
                    ) & (active_rows if independent_parallel else True)
                ))
                if not search_improved:
                    search_step_multiplier *= 0.5
            else:
                if independent_parallel:
                    backward_independent_rows(
                        instance_losses, active_parameters, active_rows,
                        latest_loss_batches,
                    )
                else:
                    loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in active_parameters
                ):
                    raise FloatingPointError(
                        f"non-finite differentiable front gradient at step {global_step}"
                    )
                torch.nn.utils.clip_grad_norm_(active_parameters, config.gradient_clip)
                if raw_log_scale.grad is not None:
                    scale_gradients = raw_log_scale.grad.detach().cpu()
                if not independent_parallel or bool(torch.any(active_rows)):
                    frozen_rows = (
                        [
                            parameter.detach()[~active_rows].clone()
                            for parameter in active_parameters
                        ]
                        if independent_parallel else ()
                    )
                    optimizer.step()
                    if independent_parallel:
                        with torch.no_grad():
                            for parameter, frozen in zip(
                                active_parameters, frozen_rows,
                            ):
                                parameter[~active_rows] = frozen
                with torch.no_grad():
                    raw_joint.clamp_(joint_lowers, joint_uppers)
                    delta_yaw.clamp_(yaw_delta_lowers, yaw_delta_uppers)
                if scheduler is not None:
                    scheduler.step()
            global_step += 1
            history.append({
                "step": global_step,
                "phase": phase_name,
                "resolution": int(resolution),
                "total_loss": current_loss,
                "mean_iou": mean_iou,
                **{
                    f"iou_{object_id}": float(value)
                    for object_id, value in zip(active_ids, iou.detach().cpu())
                },
                **{
                    f"loss_{object_id}": float(value)
                    for object_id, value in zip(active_ids, instance_losses.detach().cpu())
                },
                **({
                    f"scale_{axis}_{object_id}": float(iteration_scale[index, axis_index])
                    for index, object_id in enumerate(active_ids)
                    for axis_index, axis in enumerate(scale_residual_axes)
                } if scale_residual_axes else {}),
                **({
                    f"scale_grad_{axis}_{object_id}": float(
                        scale_gradients[index, axis_index]
                    )
                    for index, object_id in enumerate(active_ids)
                    for axis_index, axis in enumerate(scale_residual_axes)
                } if scale_residual_axes and scale_gradients is not None else {}),
            })
            if early_stopped:
                early_stopped = True
                break
            if (
                config.pose_update_mode == "forward_search"
                and search_step_multiplier < 0.25
            ):
                early_stopped = True
                break
        phase_actual_iterations[phase_name] = global_step - phase_start_step
        if early_stopped:
            early_stopped_phases.append(phase_name)
        with torch.no_grad():
            phase_loss, _, _, phase_iou, phase_instance_losses = render(
                resolution, objective,
            )
            if independent_parallel:
                update_rowwise_best(
                    parameters, best_parameters, best_scores,
                    phase_instance_losses if minimize_checkpoint else phase_iou,
                    config.improvement_epsilon, minimize=minimize_checkpoint,
                )
            else:
                final_score = (
                    float(phase_loss.cpu()) if minimize_checkpoint
                    else float(phase_iou.mean().cpu())
                )
                improved = (
                    final_score < best_scores - config.improvement_epsilon
                    if minimize_checkpoint
                    else final_score > best_scores + config.improvement_epsilon
                )
            if not independent_parallel and improved:
                best_parameters = [parameter.detach().clone() for parameter in parameters]
            for parameter, best in zip(parameters, best_parameters):
                parameter.copy_(best)
            phase_parameter_snapshots.append((
                phase_name,
                [parameter.detach().clone() for parameter in parameters],
            ))

    evaluation_resolution = (
        config.fine_resolution if config.fine_iterations > 0 else config.resolution
    )
    with torch.no_grad():
        optimized_parameters = [parameter.detach().clone() for parameter in parameters]
        for parameter in parameters:
            parameter.zero_()
        _, initial_masks, targets, initial_iou, initial_instance_losses = render(evaluation_resolution)
        initial_base_iou = initial_base_losses = None
        if edge_yaw_only:
            _, _, _, initial_base_iou, initial_base_losses = render(
                evaluation_resolution, "base",
            )
        for parameter, best in zip(parameters, optimized_parameters):
            parameter.copy_(best)
        _, final_masks, _, final_iou, final_instance_losses = render(evaluation_resolution)
        if edge_yaw_only:
            _, _, _, final_base_iou, final_base_losses = render(
                evaluation_resolution, "base",
            )
            base_rejected = (
                final_base_losses > initial_base_losses + 1e-8
                if config.bbox_weight > 0.0 else
                final_base_iou + 1e-8 < initial_base_iou
            )
            if independent_parallel:
                zero_parameter_rows(
                    (raw_xy, raw_joint, raw_log_scale), base_rejected,
                )
            else:
                if bool(base_rejected.any()):
                    for parameter in (raw_xy, raw_joint, raw_log_scale):
                        parameter.zero_()
            _, final_masks, _, final_iou, final_instance_losses = render(
                evaluation_resolution,
            )
            yaw_rejected = (
                final_instance_losses > initial_instance_losses + 1e-8
                if config.bbox_weight > 0.0 else
                final_iou + 1e-8 < initial_iou
            )
            if independent_parallel:
                zero_parameter_rows((delta_yaw,), yaw_rejected)
            else:
                if bool(yaw_rejected.any()):
                    delta_yaw.zero_()
            _, final_masks, _, final_iou, final_instance_losses = render(
                evaluation_resolution,
            )
        elif independent_parallel:
            rejected = (
                final_instance_losses > initial_instance_losses + 1e-8
                if config.bbox_weight > 0.0
                else final_iou + 1e-8 < initial_iou
            )
            if torch.any(rejected):
                zero_parameter_rows(parameters, rejected)
                _, final_masks, _, final_iou, final_instance_losses = render(evaluation_resolution)
        elif (
            float(final_instance_losses.mean().cpu()) > float(initial_instance_losses.mean().cpu()) + 1e-8
            if config.bbox_weight > 0.0
            else float(final_iou.mean().cpu()) + 1e-8 < float(initial_iou.mean().cpu())
        ):
            for parameter in parameters:
                parameter.zero_()
            _, final_masks, _, final_iou, final_instance_losses = render(
                evaluation_resolution,
            )
        (
            delta_xy, delta_log_scale, delta_joint, joints, scales, yaws,
            anchors, _, delta_z, delta_roll_x, delta_roll_y,
            rolls_x, rolls_y, _, _, _, contact_spreads,
            contact_deltas_by_object,
        ) = current_scene_state()
        zs = base_zs + delta_z
        selected_parameters = [parameter.detach().clone() for parameter in parameters]

        debug_resolution = min(
            1024, max(int(value) for value in front_camera["image_size"]),
        )
        if debug_resolution == evaluation_resolution:
            debug_initial_masks = initial_masks
            debug_targets = targets
        else:
            for parameter in parameters:
                parameter.zero_()
            _, debug_initial_masks, debug_targets, _, _ = render(debug_resolution)
        debug_intermediate_masks = {}
        for phase_name, snapshot in phase_parameter_snapshots[:-1]:
            for parameter, value in zip(parameters, snapshot):
                parameter.copy_(value)
            _, phase_masks, _, _, _ = render(debug_resolution)
            debug_intermediate_masks[phase_name] = phase_masks.detach().cpu().numpy()
        for parameter, selected in zip(parameters, selected_parameters):
            parameter.copy_(selected)
        if debug_resolution == evaluation_resolution:
            debug_final_masks = final_masks
        else:
            _, debug_final_masks, _, _, _ = render(debug_resolution)

    final_objects = {}
    object_reports = {}
    for index, object_id in enumerate(active_ids):
        translation_xy = legacy_translation_from_anchor(
            anchors[index], pivots[index], yaws[index], scales[index],
            rolls_x[index], rolls_y[index],
        )
        writeback_error = 0.0
        scale_values = scales[index].detach().cpu().numpy()
        yaw_degrees = (
            float(top_objects[object_id]["yaw_deg"])
            + float(initial_yaw_offset_deg)
            + math.degrees(float(delta_yaw[index].detach().cpu()))
        )
        yaw_degrees = (yaw_degrees + 180.0) % 360.0 - 180.0
        report = {
            "base_translation_world_m": list(top_objects[object_id]["translation_world_m"]),
            "base_yaw_deg": float(top_objects[object_id]["yaw_deg"]),
            "base_scale_xyz": base_scales[index].detach().cpu().tolist(),
            "bbox_bottom_center_local": pivots[index].detach().cpu().tolist(),
            "delta_xy_m": delta_xy[index].detach().cpu().tolist(),
            "delta_yaw_deg": (
                float(initial_yaw_offset_deg)
                + math.degrees(float(delta_yaw[index].detach().cpu()))
            ),
            "optimized_yaw_residual_deg": math.degrees(
                float(delta_yaw[index].detach().cpu())
            ),
            "delta_log_scale": delta_log_scale[index].detach().cpu().tolist(),
            "delta_joint": float(delta_joint[index].detach().cpu()),
            "joint_optimization_enabled": bool(optimize_internal_joints),
            "delta_z_m": float(delta_z[index].detach().cpu()),
            "delta_roll_x_deg": math.degrees(
                float(delta_roll_x[index].detach().cpu())
            ),
            "delta_roll_y_deg": math.degrees(
                float(delta_roll_y[index].detach().cpu())
            ),
            "joint_type": joint_types[index],
            "initial_iou": float(initial_iou[index].cpu()),
            "final_iou": float(final_iou[index].cpu()),
            "final_loss": float(final_instance_losses[index].cpu()),
            "writeback_max_error_m": writeback_error,
            "method": (
                "joint_differentiable_mask_edge_refinement"
                if loss_mode == "mask_edge" else
                "joint_differentiable_edge_refinement"
                if loss_mode == "edge_drawing" else
                "joint_differentiable_mask_refinement"
            ),
            "object_optimization_mode": object_optimization_mode,
            "optimization_schedule": optimization_schedule,
            "initial_yaw_offset_deg": float(initial_yaw_offset_deg),
            "yaw_delta_bounds_deg": (
                [float(yaw_delta_bounds[index][0]), float(yaw_delta_bounds[index][1])]
                if yaw_bounds_enabled else None
            ),
        }
        if bool(z_active[index]):
            invalid_supports = [
                parent_id for parent_id in support_parents_by_object[index]
                if (object_id, parent_id) in support_contact_fallbacks
            ]
            contact_values = [
                float(value.detach().cpu())
                for value in contact_deltas_by_object[index]
            ]
            spread = float(contact_spreads[index].detach().cpu())
            report["support_contact"] = {
                "support_parents": list(support_parents_by_object[index]),
                "contact_delta_z_m": contact_values,
                "contact_spread_m": spread,
                "contact_tolerance_m": CONTACT_TOLERANCE_M,
                "all_supports_contact": (
                    not invalid_supports
                    and spread <= CONTACT_TOLERANCE_M + 1e-8
                ),
                "invalid_support_parents": invalid_supports,
                "fallback": (
                    "keep_base_pose_when_no_valid_support"
                    if invalid_supports else None
                ),
            }
        if scale_mode == "xy":
            report["delta_log_xy_scale"] = delta_log_scale[index].detach().cpu().tolist()
            report["xy_scale_axes"] = (
                "target_obb_major_minor"
                if config.bbox_geometry == "obb" else "object_local_xy"
            )
            report["xy_scale_local_axes_swapped"] = bool(swap_xy_scale[index].detach().cpu())
            report["xy_scale_tied"] = bool(tie_xy_mask[index].detach().cpu())
        elif scale_mode == "xy_z":
            report["delta_log_shared_xy_scale"] = float(delta_log_scale[index, 0].detach().cpu())
            report["delta_log_z_scale"] = float(delta_log_scale[index, 1].detach().cpu())
        elif scale_mode == "xyz":
            report["delta_log_xyz_scale"] = delta_log_scale[index].detach().cpu().tolist()
            report["xy_scale_tied"] = bool(tie_xy_mask[index].detach().cpu())
        else:
            report["delta_log_uniform_scale"] = float(
                delta_log_scale[index].detach().cpu()
            )
        final_object = {
            **top_objects[object_id],
            "translation_world_m": [
                float(translation_xy[0].cpu()), float(translation_xy[1].cpu()),
                float(zs[index].cpu()),
            ],
            "yaw_deg": float(yaw_degrees),
            "roll_x_deg": math.degrees(float(rolls_x[index].cpu())),
            "roll_y_deg": math.degrees(float(rolls_y[index].cpu())),
            "uniform_scale": float(
                scale_values[2] if scale_mode in ("xy", "xy_z", "xyz") else scale_values[0]
            ),
            f"{view_name}_refinement": report,
        }
        if scale_mode in ("xy", "xy_z", "xyz"):
            final_object["scale_xyz"] = [float(value) for value in scale_values]
        else:
            final_object.pop("scale_xyz", None)
        if float(joint_active[index].detach().cpu()) > 0.0:
            joint = top_objects[object_id]["articulation"]["joint"]
            position = float(joints[index].detach().cpu())
            final_object["joint_state"] = {
                "name": joint["name"],
                "type": joint_types[index],
            }
            if joint_types[index] == "revolute":
                initial_angle = float(joint.get("initial_angle_deg", 0.0))
                final_object["joint_state"].update({
                    "relative_angle_rad": position,
                    "absolute_angle_deg": initial_angle + math.degrees(position),
                })
                report["delta_joint_rad"] = float(delta_joint[index].detach().cpu())
            else:
                final_object["joint_state"]["scene_current_q"] = position
                report["delta_joint_displacement"] = float(
                    delta_joint[index].detach().cpu()
                )
            final_object["joint_states"] = [final_object["joint_state"]]
            for extra_joint in top_objects[object_id]["articulation"].get("joints", [])[1:]:
                extra_state = {
                    "name": extra_joint["name"], "type": extra_joint["type"],
                }
                value = float(extra_joint.get(
                    "scene_current_q",
                    extra_joint.get("requested_scene_q", 0.0),
                ))
                if extra_joint["type"] == "revolute":
                    extra_state["relative_angle_rad"] = value
                else:
                    extra_state["scene_current_q"] = value
                final_object["joint_states"].append(extra_state)
        final_objects[object_id] = final_object
        object_reports[object_id] = report

    invalid_contacts = [
        object_id for index, object_id in enumerate(active_ids)
        if bool(z_active[index])
        and float(contact_spreads[index].detach().cpu())
        > CONTACT_TOLERANCE_M + 1e-8
    ]
    if invalid_contacts:
        raise ValueError(
            "multi-support contact did not converge within "
            f"{CONTACT_TOLERANCE_M:g} m: {', '.join(invalid_contacts)}"
        )

    report = {
        "method": (
            "nvdiffrast_soft_visible_masks_plus_normal_discontinuity_edges"
            if loss_mode == "mask_edge" else
            "nvdiffrast_soft_normal_discontinuity_edges"
            if loss_mode == "edge_drawing" else
            "nvdiffrast_soft_visible_instance_masks"
        ),
        "model_edge_method": (
            "balanced_external_mask_boundary_and_internal_normal_discontinuity"
            if loss_mode in ("edge_drawing", "mask_edge") else None
        ),
        "edge_component_weights": (
            {"external": 0.5, "internal": 0.5}
            if loss_mode == "mask_edge" else None
        ),
        "loss_mode": loss_mode,
        "object_optimization_mode": object_optimization_mode,
        "optimization_schedule": optimization_schedule,
        "joint_optimization_enabled": bool(optimize_internal_joints),
        "edge_parameter_scope": config.edge_parameter_scope,
        "initial_yaw_offset_deg": float(initial_yaw_offset_deg),
        "view": str(view_name),
        "front_uses_top_loss": False,
        "supported_occlusions_restored": bool(children_by_parent),
        "support_parent_ids": [
            active_ids[index] for index in children_by_parent
        ],
        "object_ids": list(active_ids),
        "table_occluder": True,
        "scale_mode": {
            "xy": "local_xy_z_fixed",
            "xy_z": "shared_xy_independent_z_after_xy_yaw_joint",
            "xyz": "independent_xyz_after_xy_yaw_joint",
            "uniform": "uniform_xyz_after_xy_yaw_joint",
        }[scale_mode],
        "tie_xy_scale_ids": sorted(tie_xy_scale_ids),
        "optimization_phases": [name for name, _ in parameter_groups],
        "phase_iterations": {
            name: int(settings[1])
            for (name, _), settings in zip(parameter_groups, phase_settings)
        },
        "phase_learning_rate_schedules": {
            name: "constant" if settings[3] is None else "cosine_annealing"
            for (name, _), settings in zip(parameter_groups, phase_settings)
        },
        "phase_actual_iterations": phase_actual_iterations,
        "early_stopped_phases": early_stopped_phases,
        "phase_object_stop_steps": phase_object_stop_steps,
        "config": asdict(config),
        "steps": global_step,
        "initial_mean_iou": float(initial_iou.mean().cpu()),
        "final_mean_iou": float(final_iou.mean().cpu()),
        "objects": object_reports,
        "texture_loss_object_ids": [
            object_id for object_id, colors in zip(active_ids, object_vertex_rgb)
            if colors is not None
        ],
        "texture_loss_skips": texture_loss_skips,
    }
    debug_parameters = {
        "report": report,
        "raw_xy": raw_xy,
        "delta_yaw_rad": delta_yaw,
        "raw_joint": raw_joint,
        "raw_log_scale": raw_log_scale,
    }
    _save_debug_outputs(
        output_dir,
        debug_initial_masks.detach().cpu().numpy(),
        debug_final_masks.detach().cpu().numpy(),
        debug_targets.detach().cpu().numpy(),
        history,
        debug_parameters,
        visualization_kind=(
            "edge_drawing"
            if loss_mode in ("edge_drawing", "mask_edge") else "mask"
        ),
        object_ids=active_ids,
        intermediate_masks=debug_intermediate_masks,
    )
    return final_objects, json_value(report)
