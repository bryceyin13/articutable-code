#!/usr/bin/env python3
"""Fixed two-camera tabletop alignment.

Top-only and front-only VGGT predictions are independently registered to one
table mesh.  After table calibration, exactly one vertical top camera and one
VGGT-derived front camera are immutable.  Every object refinement updates
world-space pose parameters and reprojects through those fixed cameras.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from itertools import product
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from threading import Condition, Lock, Semaphore
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psgsr.support_pivot import center_vertices_on_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINIMA_ROOT = PROJECT_ROOT.parent / "third_party" / "MINIMA"
ROOT = Path(
    os.environ.get("TABLETOP_ALIGNMENT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
OUTPUT = ROOT / "outputs/two_camera_alignment"
OBJECT_IDS = ()
_TOP_EDGE_FEATURE_LOCK = Lock()
ASSET_PATHS = {}
TABLE_SIZE_M = (0.9, 0.55, 0.72)
MIN_TABLETOP_PLANE_POINTS = 1000
TABLETOP_PLANE_EDGE_IGNORE_FRACTION = 0.05
TABLETOP_MIN_THICKNESS_M = 0.010
TABLETOP_MAX_LOWER_AREA_RATIO = 1.02
TABLETOP_ALIGNMENT_MAX_THICKNESS_RATIO = 0.10
TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO = 0.03
TABLETOP_DEPTH_MAX_THICKNESS_RATIO = 0.30
BLENDER_WORKER = "--blender-worker" in sys.argv
ALIGNMENT_COMPUTE_MODE = os.environ.get("ALIGNMENT_COMPUTE_MODE", "gpu")
EDGE_SAMPLE_SPACING_FRACTION = 0.005
ARTICULATION_VIDEO_FPS = 12
ARTICULATION_VIDEO_TRANSITION_FRAMES = 24
ARTICULATION_VIDEO_CYCLES_SAMPLES = 16


if not BLENDER_WORKER:
    import cv2
    from PIL import Image, ImageDraw, ImageOps
    from scipy.optimize import minimize, minimize_scalar
    from scipy.spatial import cKDTree

    from psgsr.table_alignment import (
        GLTF_TO_Z_UP,
        apply_similarity,
        fit_plane_ransac,
        gltf_y_up_to_z_up,
        robust_icp,
        rotation_z,
        table_frame_from_plane,
    )
    from psgsr.front_refinement import (
        FrontRefinementConfig,
        bottom_center_pivot,
        center_vertices_with_reference,
        early_stopping_update,
        mesh_vertex_texture_colors,
        render_normal_edge_buffers_batch_nvdiffrast,
        render_normal_edge_buffers_nvdiffrast,
        render_normal_edges_nvdiffrast,
        refine_front_masks_differentiable,
        signed_distance_field,
    )
    from psgsr.mask_refinement import (
        boundary,
        masked_edges,
        masked_full_frame_teed_edges,
        solid_external_silhouette,
        symmetric_contour_loss,
    )
    from psgsr.edge_detector import teed_edges
    from psgsr.unified_alignment import (
        blueprint_object_specs,
        generated_rigid_asset,
        load_scene_label_map,
        network_intrinsic_to_raw,
        repaired_articulation,
        search_mesh_z_up as _raw_search_mesh_z_up,
        vggt_crop_transform,
    )


def mesh_vertices_z_up(path):
    return _full_mesh_z_up(str(Path(path).resolve())).vertices


def mesh_extents_z_up(path):
    return np.ptp(mesh_vertices_z_up(path), axis=0)


@lru_cache(maxsize=None)
def _full_mesh_z_up(path):
    mesh = _raw_search_mesh_z_up(Path(path), face_count=None)
    mesh.metadata["file_path"] = path
    return mesh


@lru_cache(maxsize=None)
def _cached_mesh_z_up(path, face_count):
    mesh = _full_mesh_z_up(path)
    if face_count is None or len(mesh.faces) <= face_count:
        return mesh
    is_gltf = Path(path).suffix.lower() in {".glb", ".gltf"}
    mesh_to_simplify = mesh.copy() if is_gltf else mesh
    if is_gltf:
        mesh_to_simplify.vertices = np.asarray(mesh_to_simplify.vertices) @ GLTF_TO_Z_UP
    try:
        simplified = mesh_to_simplify.simplify_quadric_decimation(face_count=face_count)
        if is_gltf:
            normals = np.asarray(simplified.vertex_normals).copy()
            simplified.vertices = gltf_y_up_to_z_up(simplified.vertices)
            simplified.vertex_normals = gltf_y_up_to_z_up(normals)
        simplified.metadata["file_path"] = path
        return simplified
    except BaseException:
        return mesh


def search_mesh_z_up(path, face_count=12000):
    return _cached_mesh_z_up(
        str(Path(path).resolve()),
        None if face_count is None else int(face_count),
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def alignment_torch_device():
    """Return the selected alignment-loss device."""
    import torch

    if ALIGNMENT_COMPUTE_MODE == "cpu":
        return torch.device("cpu")
    if ALIGNMENT_COMPUTE_MODE != "gpu":
        raise ValueError("alignment compute mode must be 'cpu' or 'gpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("alignment compute mode 'gpu' requires CUDA")
    return torch.device("cuda")


def front_round_uses_joint_optimization(round_index, second_round_mode):
    return int(round_index) == 1 and second_round_mode == "joint"


def front_round_loss_mode(round_index, first_round_loss, second_round_loss=None):
    if int(round_index) == 1 and second_round_loss is not None:
        return second_round_loss
    return first_round_loss


def combine_front_round_reports(
    reports, second_round_mode="sequential", first_round_schedule="legacy_sequential",
):
    if not reports:
        raise ValueError("front refinement requires at least one round")
    if first_round_schedule == "bbox_center_then_joint":
        method = "first_round_bbox_bottom_center_xy_then_joint_pose_scale"
        if len(reports) == 2 and second_round_mode == "joint":
            method += "_second_round_joint_pose_scale"
    elif len(reports) == 2 and second_round_mode == "joint":
        method = "first_round_sequential_second_round_joint_xy_yaw_joint_scale"
    else:
        method = (
            "two_rounds_xy_yaw_joint_then_independent_xyz_scale"
            if len(reports) == 2
            else f"{len(reports)}_rounds_xy_yaw_joint_then_independent_xyz_scale"
        )
    return {
        **reports[-1],
        "method": method,
        "round_count": len(reports),
        "second_round_mode": second_round_mode,
        "first_round_schedule": first_round_schedule,
        "rounds": reports,
        "steps": sum(report["steps"] for report in reports),
        "initial_mean_iou": reports[0]["initial_mean_iou"],
        "final_mean_iou": reports[-1]["final_mean_iou"],
    }


def select_independent_yaw_branches(
    branches, view_name="front", allowed_offsets_by_object=None,
    rgb_losses_by_object=None, rgb_weight=0.0,
    direction_losses_by_object=None,
):
    """Select each yaw branch by its full optimized loss and optional RGB loss."""
    if not branches:
        raise ValueError("yaw branch selection requires candidates")
    rgb_losses_by_object = rgb_losses_by_object or {}
    direction_losses_by_object = direction_losses_by_object or {}
    rgb_weight = _nonnegative_finite_float("front yaw RGB weight", rgb_weight)
    object_ids = branches[0][2]["object_ids"]
    selected_objects = {}
    selected_reports = {}
    selected_offsets = {}
    for object_id in object_ids:
        candidates = list(branches)
        use_rgb = object_id in rgb_losses_by_object and rgb_weight > 0.0
        use_direction = object_id in direction_losses_by_object
        if allowed_offsets_by_object is not None:
            allowed = allowed_offsets_by_object[object_id]
            candidates = [
                branch for branch in candidates
                if any(math.isclose(branch[0], value) for value in allowed)
            ]
        metrics = []
        for candidate_offset, candidate_objects, candidate_report in candidates:
            object_metrics = candidate_report["objects"][object_id]
            iou = float(object_metrics["final_iou"])
            optimization_loss = float(object_metrics.get("final_loss", 1.0 - iou))
            rgb_loss = (
                float(rgb_losses_by_object[object_id][candidate_offset])
                if use_rgb else 0.0
            )
            direction_value = (
                direction_losses_by_object[object_id][candidate_offset]
                if use_direction else None
            )
            direction_loss = (
                float(direction_value["loss"])
                if isinstance(direction_value, dict) else
                float(direction_value) if direction_value is not None else None
            )
            score = (
                direction_loss if use_direction
                else optimization_loss + rgb_weight * rgb_loss
            )
            metrics.append((score, iou, optimization_loss, rgb_loss, direction_loss))
        scores = [score for score, _, _, _, _ in metrics]
        selected_index = int(np.argmin(scores))
        offset, objects, report = candidates[selected_index]
        selected = objects[object_id]
        object_report = selected[f"{view_name}_refinement"]
        object_report["yaw_branch_candidates"] = [
            {
                "initial_offset_deg": float(candidate_offset),
                "final_iou": float(candidate_metrics[1]),
                "optimization_loss": float(candidate_metrics[2]),
                "rgb_loss": float(candidate_metrics[3]) if use_rgb else None,
                "direction_loss": candidate_metrics[4],
                "direction_details": (
                    direction_losses_by_object[object_id][candidate_offset]
                    if use_direction else None
                ),
                "selection_score": float(candidate_metrics[0]),
                "final_yaw_deg": float(candidate_objects[object_id]["yaw_deg"]),
            }
            for (candidate_offset, candidate_objects, _), candidate_metrics
            in zip(candidates, metrics)
        ]
        object_report["selected_yaw_branch_offset_deg"] = float(offset)
        object_report["yaw_branch_rgb_weight"] = float(rgb_weight)
        object_report["yaw_branch_uses_rgb"] = bool(use_rgb)
        selected_direction = (
            direction_losses_by_object[object_id][offset]
            if use_direction else {}
        )
        object_report["yaw_branch_uses_dino"] = bool(use_direction)
        object_report["yaw_branch_selection"] = (
            "minimum_dino_fixed_grid_direction_loss"
            if use_direction else
            "minimum_full_optimization_loss"
        )
        selected_objects[object_id] = selected
        selected_reports[object_id] = object_report
        selected_offsets[object_id] = float(offset)
    first_report = branches[0][2]
    return selected_objects, {
        **first_report,
        "method": "independent_parallel_bidirectional_yaw",
        "steps": sum(report["steps"] for _, _, report in branches),
        "final_mean_iou": float(np.mean([
            report["final_iou"] for report in selected_reports.values()
        ])),
        "objects": selected_reports,
        "selected_yaw_branch_offsets_deg": selected_offsets,
        "yaw_branch_rgb_weight": float(rgb_weight),
        "yaw_branch_rgb_object_ids": sorted(rgb_losses_by_object),
        "yaw_branches": [
            {"initial_offset_deg": float(offset), "report": report}
            for offset, _, report in branches
        ],
    }


def yaw_delta_bounds_from_origin(origin_yaw_deg, current_yaw_deg, limit_deg):
    accumulated = (
        float(current_yaw_deg) - float(origin_yaw_deg) + 180.0
    ) % 360.0 - 180.0
    limit = abs(float(limit_deg))
    return -limit - accumulated, limit - accumulated


def front_yaw_branch_specs(search_mode, opposite_offset_deg):
    if search_mode == "bidirectional":
        return ((0.0, 30.0), (float(opposite_offset_deg), 30.0))
    return ((0.0, 30.0),)


def top_iou_allowed_front_yaw_offsets(
    zero_iou, opposite_iou, ambiguity_threshold=0.02, opposite_offset_deg=180.0,
):
    zero_iou = _unit_interval_float("top-view 0-degree IoU", zero_iou)
    opposite_iou = _unit_interval_float("top-view 180-degree IoU", opposite_iou)
    threshold = _unit_interval_float(
        "top-view yaw IoU ambiguity threshold", ambiguity_threshold,
    )
    if abs(zero_iou - opposite_iou) <= threshold:
        return (0.0, float(opposite_offset_deg))
    return (0.0,) if zero_iou > opposite_iou else (float(opposite_offset_deg),)


def top_view_yaw_ambiguity_gate(
    objects, top_camera, top_labels, top_ids, ambiguity_threshold,
    opposite_offset_deg=180.0, front_camera=None, front_edge_targets=None,
    front_edge_override_margin=0.01,
):
    front_edge_override_margin = _unit_interval_float(
        "front yaw edge override margin", front_edge_override_margin,
    )
    allowed_offsets = {}
    report = {}
    for object_id, item in objects.items():
        if object_id not in top_ids:
            allowed = (0.0, float(opposite_offset_deg))
            report[object_id] = {
                "reason": "missing_top_mask_treated_as_ambiguous",
                "allowed_front_yaw_offsets_deg": list(allowed),
            }
            allowed_offsets[object_id] = allowed
            continue
        mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=3000)
        reference_mesh = search_mesh_z_up(
            Path(item["geometry_asset"]), face_count=None,
        )
        vertices = center_vertices_with_reference(
            mesh.vertices, reference_mesh.vertices, center_vertices_on_table,
        )
        target = target_mask(top_labels, top_ids, object_id)
        ious = []
        for offset in (0.0, float(opposite_offset_deg)):
            posed = pose_vertices(
                vertices, item["uniform_scale"], item["translation_world_m"][:2],
                item["yaw_deg"] + offset, scale_xyz=item.get("scale_xyz"),
            )
            ious.append(binary_mask_iou(render_mesh_mask(mesh, posed, top_camera), target))
        allowed = top_iou_allowed_front_yaw_offsets(
            *ious, ambiguity_threshold, opposite_offset_deg,
        )
        front_edge_losses = None
        front_edge_override = False
        if front_camera is not None and object_id in (front_edge_targets or {}):
            front_edge_losses = []
            for offset in (0.0, float(opposite_offset_deg)):
                posed = pose_vertices(
                    vertices, item["uniform_scale"], item["translation_world_m"][:2],
                    item["yaw_deg"] + offset, scale_xyz=item.get("scale_xyz"),
                )
                rendered_mask = render_mesh_mask(mesh, posed, front_camera)
                rendered_edges = render_mesh_edges(
                    mesh, posed, front_camera,
                    silhouette=rendered_mask, visible_only=True,
                )
                front_edge_losses.append(symmetric_contour_loss(
                    rendered_edges.astype(np.uint8) * 255,
                    np.asarray(front_edge_targets[object_id], dtype=np.uint8),
                ))
            locked_index = 0 if allowed == (0.0,) else 1
            other_index = 1 - locked_index
            edge_improvement = (
                front_edge_losses[locked_index] - front_edge_losses[other_index]
            ) / max(front_edge_losses[locked_index], 1e-6)
            front_edge_override = len(allowed) == 1 and (
                edge_improvement >= front_edge_override_margin
            )
            if front_edge_override:
                allowed = (0.0, float(opposite_offset_deg))
        allowed_offsets[object_id] = allowed
        report[object_id] = {
            "top_iou_0_deg": float(ious[0]),
            "top_iou_opposite_deg": float(ious[1]),
            "absolute_iou_gap": float(abs(ious[0] - ious[1])),
            "ambiguity_threshold": float(ambiguity_threshold),
            "top_iou_ambiguous": abs(ious[0] - ious[1]) <= ambiguity_threshold,
            "bidirectional_allowed": len(allowed) == 2,
            "ambiguous": len(allowed) == 2,
            "allowed_front_yaw_offsets_deg": list(allowed),
            "front_edge_loss_0_deg": (
                float(front_edge_losses[0]) if front_edge_losses is not None else None
            ),
            "front_edge_loss_opposite_deg": (
                float(front_edge_losses[1]) if front_edge_losses is not None else None
            ),
            "front_edge_override_margin": float(front_edge_override_margin),
            "front_edge_opposite_relative_improvement": (
                float(edge_improvement) if front_edge_losses is not None else None
            ),
            "front_edge_override": bool(front_edge_override),
        }
    return allowed_offsets, report


def run_object_refinements(items, refine_fn, mode):
    """Run per-object refinement tasks sequentially or in a thread/GPU batch."""
    if not items:
        return {}
    if mode == "independent_parallel":
        device = alignment_torch_device()
        if device.type == "cuda":
            import torch

            # Initialize the CUDA eigensolver on the main thread before workers
            # reach sample_edge_features() concurrently.
            torch.linalg.eigh(
                torch.eye(2, dtype=torch.float64, device=device),
            )[0].cpu()
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            values = list(executor.map(refine_fn, items))
    else:
        values = [refine_fn(item) for item in items]
    return {object_id: value for (object_id, _), value in zip(items, values)}


def two_camera_asset(object_id):
    return ASSET_PATHS[object_id]


def alignment_object_specs(blueprint, root=ROOT, asset_source="processed"):
    if asset_source == "processed":
        return blueprint_object_specs(blueprint, root)
    if asset_source == "articulated":
        specs = alignment_object_specs(blueprint, root, "raw")
        blueprint_objects = {item["object_id"]: item for item in blueprint["objects"]}
        manifest = json.loads(
            (Path(root) / "data/articulation_manifest.json").read_text(encoding="utf-8")
        )
        for item in manifest["items"]:
            object_id = item["object_id"]
            if object_id not in specs:
                raise ValueError(f"articulation manifest has unknown object: {object_id}")
            specs[object_id].update({
                "asset_type": "articulated_urdf",
                "asset": item["package"],
            })
            try:
                articulation = repaired_articulation(
                    item["package"], blueprint_objects[object_id],
                )
            except ValueError as exc:
                if str(exc) != "alignment requires exactly one revolute or prismatic joint":
                    raise
            else:
                specs[object_id]["articulation"] = articulation
        return specs
    if asset_source != "raw":
        raise ValueError(f"unknown alignment asset source: {asset_source}")
    specs = {
        item["object_id"]: {
            "asset_type": "rigid_glb",
            "asset": str(two_camera_asset(item["object_id"])),
            "geometry_asset": str(two_camera_asset(item["object_id"])),
        }
        for item in blueprint["objects"]
    }
    manifest_path = Path(root) / "data/articulation_manifest.json"
    if not manifest_path.is_file():
        return specs
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["items"]:
        object_id = item["object_id"]
        if object_id not in specs:
            raise ValueError(f"articulation manifest has unknown object: {object_id}")
        requires_open_state = item.get("requires_open_state", False)
        if not isinstance(requires_open_state, bool):
            raise ValueError(
                f"requires_open_state must be boolean for {object_id}"
            )
        rest_mesh = item.get("rest_mesh")
        if requires_open_state and rest_mesh and Path(rest_mesh).is_file():
            specs[object_id].update({
                "asset": rest_mesh,
                "geometry_asset": rest_mesh,
            })
    return specs


def precomputed_mesh_orientation(path, blueprint, asset_source):
    """Load the exact, already-oriented assets selected before assembly."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "mesh_orientation_v1":
        raise ValueError("unsupported mesh orientation manifest format")
    if document.get("asset_source") != asset_source:
        raise ValueError(
            "mesh orientation asset source does not match scene assembly: "
            f"{document.get('asset_source')} != {asset_source}"
        )
    expected = {item["object_id"] for item in blueprint["objects"]}
    items = {item["object_id"]: item for item in document.get("items", [])}
    if set(items) != expected:
        raise ValueError("mesh orientation manifest object set does not match blueprint")
    specs = {}
    reports = {}
    templates = {}
    template_cache = document.get("top_template_cache") or {}
    for object_id, item in items.items():
        spec = dict(item["spec"])
        for field in ("asset", "geometry_asset"):
            if not Path(spec[field]).exists():
                raise FileNotFoundError(
                    f"mesh orientation {field} is missing for {object_id}: "
                    f"{spec[field]}"
                )
        specs[object_id] = spec
        reports[object_id] = dict(item["report"])
        if item.get("top_template") is not None:
            templates[object_id] = {
                **template_cache,
                **dict(item["top_template"]),
            }
    return specs, reports, templates


def top_template_cache_compatible(templates, object_ids, camera, render_scale):
    if set(templates) != set(object_ids):
        return False
    for value in templates.values():
        cached_camera = value.get("camera") or {}
        if (
            value.get("version") != 1
            or int(value.get("render_scale", 0)) != int(render_scale)
            or not Path(value.get("path", "")).is_file()
            or cached_camera.get("projection") != camera.get("projection")
        ):
            return False
        for field in (
            "image_size", "intrinsic", "rotation_world_from_camera",
        ):
            if field not in cached_camera or not np.allclose(
                cached_camera[field], camera[field], rtol=1e-7, atol=1e-8,
            ):
                return False
    return True


def rebase_top_template_pose(pose, cached_camera, camera):
    """Preserve cached camera coordinates after a world-space camera shift."""
    result = dict(pose)
    translation = np.asarray(result["translation_world_m"], dtype=float)
    camera_delta = (
        np.asarray(camera["center_world_m"], dtype=float)
        - np.asarray(cached_camera["center_world_m"], dtype=float)
    )
    result["translation_world_m"] = (translation + camera_delta).tolist()
    return result


def uniform_scale(native, target):
    ratios = np.asarray(target, dtype=float) / np.asarray(native, dtype=float)
    return float(np.exp(np.mean(np.log(ratios))))


def signed_up_axis_candidates():
    return (
        ("+X", np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])),
        ("-X", np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])),
        ("+Y", np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])),
        ("-Y", np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])),
        ("+Z", np.eye(3)),
        ("-Z", np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])),
    )


def obvious_up_axis_error(top_geometry, mesh_extents, threshold=1.5):
    spans = np.asarray(top_geometry["pca_footprint_px"], dtype=float)
    target_aspect = float(spans[0] / spans[1])
    extents = np.asarray(mesh_extents, dtype=float)
    candidates = []
    for label, indices in (("+Z", (0, 1)), ("+X", (2, 1)), ("+Y", (0, 2))):
        footprint = extents[list(indices)]
        aspect = float(max(footprint) / min(footprint))
        mismatch = float(max(aspect / target_aspect, target_aspect / aspect))
        candidates.append({
            "up_axis": label,
            "footprint_aspect_ratio": aspect,
            "aspect_mismatch": mismatch,
        })
    identity = candidates[0]
    alternative = min(candidates[1:], key=lambda item: item["aspect_mismatch"])
    triggered = bool(
        identity["aspect_mismatch"] > threshold
        and alternative["aspect_mismatch"] <= threshold
    )
    return triggered, {
        "method": "identity_gated_top_mask_pca_footprint_aspect",
        "maximum_aspect_mismatch": float(threshold),
        "top_pca_spans_px": spans.tolist(),
        "top_footprint_aspect_ratio": target_aspect,
        "candidates": candidates,
    }


def select_mesh_orientation_candidate(candidates):
    identity_candidates = [
        candidate for candidate in candidates
        if candidate["up_axis"] == "+Z"
    ]
    if not identity_candidates:
        raise ValueError("mesh orientation candidates must include +Z")
    identity_view_losses = {
        view: min(candidate[f"{view}_loss"] for candidate in identity_candidates)
        for view in ("front", "top")
    }
    return min(candidates, key=lambda item: item["loss"]), identity_view_losses


def mesh_orientation_scorer_for_run(
    direction_scorer, precomputed_orientation_reports, device,
):
    if precomputed_orientation_reports is not None:
        return direction_scorer
    from psgsr.foundpose_loss import FoundPoseDirectionScorer

    return FoundPoseDirectionScorer(device=device)


def normalized_shape_mask(mask, size=192, margin=12):
    mask = np.asarray(mask) > 0
    rows, columns = np.nonzero(mask)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if not len(columns):
        return canvas
    crop = mask[
        rows.min():rows.max() + 1, columns.min():columns.max() + 1,
    ].astype(np.uint8)
    height, width = crop.shape
    scale = min((size - 2 * margin) / width, (size - 2 * margin) / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST,
    )
    x = (size - resized_width) // 2
    y = (size - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def binary_mask_iou(first, second):
    first = np.asarray(first) > 0
    second = np.asarray(second) > 0
    return float(
        np.count_nonzero(first & second)
        / max(np.count_nonzero(first | second), 1)
    )


def project_points(points, camera):
    points = np.asarray(points, dtype=float)
    center = np.asarray(camera["center_world_m"], dtype=float)
    world_from_camera = np.asarray(camera["rotation_world_from_camera"], dtype=float)
    intrinsic = np.asarray(camera["intrinsic"], dtype=float)
    camera_points = (points - center) @ world_from_camera
    depth = camera_points[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=float)
    valid = depth > 1e-8
    if camera.get("projection") == "orthographic":
        pixels[valid, 0] = intrinsic[0, 0] * camera_points[valid, 0] + intrinsic[0, 2]
        pixels[valid, 1] = intrinsic[1, 1] * camera_points[valid, 1] + intrinsic[1, 2]
    else:
        pixels[valid, 0] = intrinsic[0, 0] * camera_points[valid, 0] / depth[valid] + intrinsic[0, 2]
        pixels[valid, 1] = intrinsic[1, 1] * camera_points[valid, 1] / depth[valid] + intrinsic[1, 2]
    return pixels, depth


def mask_boundary_on_world_plane(mask, camera, plane_z=0.0):
    """Back-project a complete image-mask boundary onto a horizontal plane."""
    solid = solid_external_silhouette(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
    )
    contours, _ = cv2.findContours(
        solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("cannot back-project an empty mask boundary")
    pixel_centers = np.concatenate([
        contour[:, 0, :] for contour in contours
    ]).astype(float)
    pixel_corners = np.concatenate([
        pixel_centers + offset
        for offset in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
    ])
    intrinsic = np.asarray(camera["intrinsic"], dtype=float)
    camera_xy = np.column_stack((
        (pixel_corners[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
        (pixel_corners[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
    ))
    center = np.asarray(camera["center_world_m"], dtype=float)
    world_from_camera = np.asarray(
        camera["rotation_world_from_camera"], dtype=float,
    )
    if camera.get("projection") == "orthographic":
        origins = center + np.column_stack((
            camera_xy, np.zeros(len(camera_xy)),
        )) @ world_from_camera.T
        directions = np.broadcast_to(
            np.array([0.0, 0.0, 1.0]) @ world_from_camera.T,
            origins.shape,
        )
    else:
        origins = np.broadcast_to(center, (len(camera_xy), 3))
        directions = np.column_stack((
            camera_xy, np.ones(len(camera_xy)),
        )) @ world_from_camera.T
    valid = np.abs(directions[:, 2]) > 1e-8
    if not np.any(valid):
        raise ValueError("camera rays are parallel to the tabletop plane")
    origins = origins[valid]
    directions = directions[valid]
    distance = (float(plane_z) - origins[:, 2]) / directions[:, 2]
    return origins + distance[:, None] * directions


def scaled_camera(camera, factor):
    result = dict(camera)
    intrinsic = np.asarray(camera["intrinsic"], dtype=float).copy()
    intrinsic[0, :] *= factor
    intrinsic[1, :] *= factor
    result["intrinsic"] = intrinsic.tolist()
    result["image_size"] = [
        max(1, int(round(camera["image_size"][0] * factor))),
        max(1, int(round(camera["image_size"][1] * factor))),
    ]
    if camera.get("projection") == "orthographic":
        result["pixels_per_meter"] = float(camera["pixels_per_meter"] * factor)
        result["meters_per_pixel"] = float(camera["meters_per_pixel"] / factor)
    return result


def vertical_top_camera(
    intrinsic, image_size, target_center_px, table_width_m, table_width_px,
    distance_m=None, projection="orthographic",
):
    intrinsic = np.asarray(intrinsic, dtype=float)
    distance = float(distance_m or (intrinsic[0, 0] * table_width_m / table_width_px))
    width, height = int(image_size[0]), int(image_size[1])
    pixels_per_meter = float(table_width_px / table_width_m)
    u, v = target_center_px
    if projection == "perspective":
        center_x = (intrinsic[0, 2] - u) * distance / intrinsic[0, 0]
        center_y = (v - intrinsic[1, 2]) * distance / intrinsic[1, 1]
        return {
            "name": "top_camera",
            "projection": "perspective",
            "image_size": [width, height],
            "intrinsic": intrinsic.tolist(),
            "center_world_m": [float(center_x), float(center_y), distance],
            "rotation_world_from_camera": [
                [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
            ],
            "fixed": True,
            "source": "vertical perspective camera metrically anchored by table width",
        }
    if projection != "orthographic":
        raise ValueError(f"unknown top camera projection: {projection}")
    orthographic_intrinsic = np.array([
        [pixels_per_meter, 0.0, width / 2.0],
        [0.0, pixels_per_meter, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    center_x = (width / 2.0 - u) / pixels_per_meter
    center_y = (v - height / 2.0) / pixels_per_meter
    return {
        "name": "top_camera",
        "projection": "orthographic",
        "image_size": [width, height],
        "intrinsic": orthographic_intrinsic.tolist(),
        "pixels_per_meter": pixels_per_meter,
        "meters_per_pixel": 1.0 / pixels_per_meter,
        "ortho_scale": width / pixels_per_meter,
        "center_world_m": [float(center_x), float(center_y), distance],
        "rotation_world_from_camera": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        "fixed": True,
        "source": "vertical orthographic camera metrically anchored by table width",
    }


def initial_table_similarity(observed, normal, camera_center, support_width_m):
    """Initialize the VGGT-to-table transform from robust support bounds."""
    observed = np.asarray(observed, dtype=float)
    plane_center = np.median(observed, axis=0)
    rotation, _ = table_frame_from_plane(normal, plane_center, camera_center)
    local = observed @ rotation.T
    low, high = np.percentile(local, [1, 99], axis=0)
    scale = float(support_width_m / (high[0] - low[0]))
    origin = (low + high) * 0.5
    return scale, rotation, -scale * origin


def robust_bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def instance_geometry(labels, instance_id):
    mask = labels == instance_id
    bbox = robust_bbox(mask)
    ys, xs = np.where(mask)
    coords = np.column_stack((xs - xs.mean(), ys - ys.mean()))
    covariance = coords.T @ coords / max(1, len(coords))
    values, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(values)[::-1]]
    axis = axes[:, 0]
    if axis[0] < 0:
        axis = -axis
    pca_footprint = np.sort(np.ptp(coords @ axes, axis=0))[::-1]
    return {
        "bbox_xyxy": bbox,
        "center_px": [float(xs.mean()), float(ys.mean())],
        "major_axis_image": axis.tolist(),
        "pca_footprint_px": pca_footprint.tolist(),
        "mask": mask,
    }


def resize_nearest(array, size):
    array = np.asarray(array)
    boolean = array.dtype == np.bool_
    resized = cv2.resize(
        array.astype(np.uint8) if boolean else array,
        tuple(size), interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0 if boolean else resized


def inset_support_for_plane_fit(mask, fraction):
    """Exclude the outer support band without treating occlusions as edges."""
    support = np.asarray(mask) > 0
    if not support.any() or float(fraction) <= 0.0:
        return support, 0.0
    ys, xs = np.nonzero(support)
    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    solid = np.zeros(support.shape, dtype=np.uint8)
    cv2.fillConvexPoly(solid, hull, 1)
    short_side = min(np.ptp(xs) + 1, np.ptp(ys) + 1)
    inset = float(fraction) * float(short_side)
    padded = np.pad(solid.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return support & (distance > inset), inset


def load_vggt_view(
    directory, labels, ids, support_id="table_0", support_mask=None,
):
    directory = Path(directory)
    with np.load(directory / "predictions.npz") as data:
        points = np.asarray(data["world_points"][0], dtype=float)
        retained = np.asarray(data["retained_mask"][0], dtype=bool)
        intrinsic = np.asarray(data["intrinsic"][0], dtype=float)
        extrinsic = np.asarray(data["extrinsic"][0], dtype=float)
    transform = vggt_crop_transform((labels.shape[1], labels.shape[0]))
    if transform["network_size"] != [points.shape[1], points.shape[0]]:
        raise ValueError(f"VGGT size mismatch for {directory}: {transform['network_size']} vs {points.shape[1::-1]}")
    network_labels = resize_nearest(labels, (points.shape[1], points.shape[0]))
    finite = np.isfinite(points).all(axis=-1)
    network_support = (
        network_labels == ids[support_id]
        if support_mask is None else
        resize_nearest(np.asarray(support_mask) > 0, (points.shape[1], points.shape[0]))
    )
    valid = network_support & retained & finite
    observed = points[valid]
    if len(observed) < 100:
        raise RuntimeError(f"too few VGGT table points in {directory}: {len(observed)}")
    fit_support, edge_inset_px = inset_support_for_plane_fit(
        network_support, TABLETOP_PLANE_EDGE_IGNORE_FRACTION,
    )
    fit_valid = fit_support & retained & finite
    fit_observed = points[fit_valid]
    edge_ignore_applied = len(fit_observed) >= 100
    if not edge_ignore_applied:
        fit_observed = observed
        edge_inset_px = 0.0
    span = np.linalg.norm(
        np.percentile(fit_observed, 95, axis=0)
        - np.percentile(fit_observed, 5, axis=0)
    )
    threshold = max(1e-6, 0.005 * span)
    normal, offset, _ = fit_plane_ransac(
        fit_observed, threshold, iterations=1500, seed=0,
    )
    inliers = np.abs(observed @ normal + offset) <= threshold
    plane_points = observed[inliers]
    tabletop_plane_mask = None
    if len(plane_points) >= MIN_TABLETOP_PLANE_POINTS:
        network_mask = np.zeros(valid.shape, dtype=np.uint8)
        network_mask[valid] = np.asarray(inliers, dtype=np.uint8) * 255
        network_mask = cv2.morphologyEx(
            network_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8),
        )
        tabletop_plane_mask = resize_nearest(
            solid_external_silhouette(network_mask),
            (labels.shape[1], labels.shape[0]),
        )
    plane_center = np.median(plane_points, axis=0)
    camera_rotation = extrinsic[:, :3]
    camera_center = -camera_rotation.T @ extrinsic[:, 3]
    if np.dot(normal, camera_center - plane_center) < 0:
        normal = -normal
    metric_scale, rotation_world_from_vggt, translation = initial_table_similarity(
        plane_points, normal, camera_center, TABLE_SIZE_M[0]
    )
    world_from_camera = rotation_world_from_vggt @ camera_rotation.T
    center_world = camera_center @ rotation_world_from_vggt.T * metric_scale + translation
    object_clouds = {}
    object_clouds_vggt = {}
    for object_id, instance_id in ids.items():
        if object_id == support_id:
            continue
        object_valid = (network_labels == instance_id) & retained & finite
        world = points[object_valid] @ rotation_world_from_vggt.T * metric_scale + translation
        if len(world):
            object_clouds[object_id] = world
            object_clouds_vggt[object_id] = points[object_valid]
    raw_intrinsic = network_intrinsic_to_raw(intrinsic, transform)
    return {
        "points": points,
        "observed_table_vggt": observed,
        "intrinsic_network": intrinsic,
        "extrinsic": extrinsic,
        "network_image_size": [int(points.shape[0]), int(points.shape[1])],
        "network_labels": network_labels,
        "metric_scale": metric_scale,
        "rotation_world_from_vggt": rotation_world_from_vggt,
        "translation_world": translation,
        "camera": {
            "projection": "perspective",
            "image_size": transform["raw_size"],
            "intrinsic": raw_intrinsic.tolist(),
            "center_world_m": center_world.tolist(),
            "rotation_world_from_camera": world_from_camera.tolist(),
            "fixed": True,
        },
        "object_clouds": object_clouds,
        "object_clouds_vggt": object_clouds_vggt,
        "plane_point_count": int(len(plane_points)),
        "table_plane_fit": {
            "edge_ignore_fraction": TABLETOP_PLANE_EDGE_IGNORE_FRACTION,
            "edge_ignore_applied": bool(edge_ignore_applied),
            "edge_inset_network_px": float(edge_inset_px),
            "full_candidate_count": int(len(observed)),
            "fit_candidate_count": int(len(fit_observed)),
        },
        "tabletop_plane_mask": tabletop_plane_mask,
    }


def refine_front_table_registration(
    view, table_mesh_path, table_scale_xyz, sample_count=30000,
    table_yaw_deg=0.0,
):
    """Reuse the established visible-surface robust ICP for front registration."""
    import trimesh

    source_mesh = search_mesh_z_up(table_mesh_path)
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(source_mesh.vertices), table_yaw_deg,
    )
    vertices *= np.asarray(table_scale_xyz, dtype=float)
    metric_mesh = trimesh.Trimesh(vertices=vertices, faces=source_mesh.faces, process=False)
    mesh_points, face_indices = trimesh.sample.sample_surface(metric_mesh, sample_count, seed=0)
    mesh_normals = metric_mesh.face_normals[face_indices]
    rotation, translation, _, _, _, _, history = robust_icp(
        view["observed_table_vggt"],
        mesh_points,
        mesh_normals,
        view["intrinsic_network"],
        view["extrinsic"],
        tuple(view["network_image_size"]),
        view["metric_scale"],
        view["rotation_world_from_vggt"],
        view["translation_world"],
    )
    camera_rotation = view["extrinsic"][:, :3]
    camera_center_vggt = -camera_rotation.T @ view["extrinsic"][:, 3]
    view["rotation_world_from_vggt"] = rotation
    view["translation_world"] = translation
    view["camera"]["rotation_world_from_camera"] = (rotation @ camera_rotation.T).tolist()
    view["camera"]["center_world_m"] = apply_similarity(
        camera_center_vggt[None], view["metric_scale"], rotation, translation
    )[0].tolist()
    view["object_clouds"] = {
        object_id: apply_similarity(points, view["metric_scale"], rotation, translation)
        for object_id, points in view["object_clouds_vggt"].items()
    }
    view["table_icp_history"] = history
    return view


def center_vertices_at_tabletop(vertices):
    vertices = np.asarray(vertices, dtype=float).copy()
    vertices[:, :2] -= (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) * 0.5
    vertices[:, 2] -= vertices[:, 2].max()
    return vertices


def table_yaw_rotation(yaw_deg):
    yaw = math.radians(float(yaw_deg))
    return np.array((
        (math.cos(yaw), -math.sin(yaw), 0.0),
        (math.sin(yaw), math.cos(yaw), 0.0),
        (0.0, 0.0, 1.0),
    ))


def orient_table_vertices(vertices, yaw_deg):
    return np.asarray(vertices, dtype=float) @ table_yaw_rotation(yaw_deg).T


def table_scale_for_dimensions(vertices, yaw_deg, dimensions_xyz):
    oriented = orient_table_vertices(center_vertices_at_tabletop(vertices), yaw_deg)
    return (np.asarray(dimensions_xyz, dtype=float) / np.ptp(oriented, axis=0)).tolist()


def table_blender_scale(scale_xyz, yaw_deg):
    """Convert world-axis scale to Blender local scale for a cardinal table yaw."""
    scale = list(map(float, scale_xyz))
    if int(round(float(yaw_deg) / 90.0)) % 2:
        scale[0], scale[1] = scale[1], scale[0]
    return scale


def centered_mesh_vertices(path):
    return center_vertices_at_tabletop(mesh_vertices_z_up(path))


def center_vertices_on_table(vertices):
    return center_vertices_on_support(vertices)


def pose_vertices(
    vertices, scale, xy, yaw_deg, scale_xyz=None, z=0.0, roll_x_deg=0.0,
    roll_y_deg=0.0,
):
    posed = np.asarray(vertices, dtype=float).copy()
    if scale_xyz is None:
        posed *= float(scale)
    else:
        posed *= np.asarray(scale_xyz, dtype=float)
    roll = math.radians(roll_x_deg)
    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])
    posed = posed @ rotation_x.T
    roll_y = math.radians(roll_y_deg)
    rotation_y = np.array([
        [math.cos(roll_y), 0.0, math.sin(roll_y)],
        [0.0, 1.0, 0.0],
        [-math.sin(roll_y), 0.0, math.cos(roll_y)],
    ])
    posed = posed @ rotation_y.T
    angle = math.radians(yaw_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    posed[:, :2] = posed[:, :2] @ rotation.T
    posed[:, :2] += np.asarray(xy, dtype=float)
    posed[:, 2] += float(z)
    return posed


def loaded_mesh_vertex_texture_colors(mesh):
    """Sample vertex colors from an already-loaded textured mesh."""
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if uv is None or image is None or len(uv) != len(mesh.vertices):
        return None
    image = np.asarray(
        image.convert("RGB") if hasattr(image, "convert") else image,
    )[..., :3]
    uv = np.asarray(uv, dtype=np.float32)
    x = np.rint(
        np.clip(uv[:, 0], 0.0, 1.0) * (image.shape[1] - 1),
    ).astype(int)
    y = np.rint(
        (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (image.shape[0] - 1),
    ).astype(int)
    return image[y, x].astype(np.float32) / 255.0


def item_vertex_texture_colors(mesh, item):
    cache_key = str(Path(item["geometry_asset"]).resolve())
    cached = mesh.metadata.get("direction_vertex_texture_colors")
    if isinstance(cached, dict) and cached.get("asset") == cache_key:
        return cached.get("colors")
    try:
        colors = loaded_mesh_vertex_texture_colors(mesh)
        loaded_asset = mesh.metadata.get("file_path")
        same_loaded_asset = (
            loaded_asset is not None
            and str(Path(loaded_asset).resolve()) == cache_key
        )
        if colors is None and not same_loaded_asset:
            colors = mesh_vertex_texture_colors(
                item["geometry_asset"], mesh.vertices,
            )
    except Exception:
        colors = None
    mesh.metadata["direction_vertex_texture_colors"] = {
        "asset": cache_key,
        "colors": colors,
    }
    return colors


def render_item_camera_direction_rgb(mesh, item, camera, vertex_rgb):
    """Render the textured RGB direction input, with normals as a safe fallback."""
    mask, rgb, _, modality = render_item_camera_direction_inputs(
        mesh, item, camera, vertex_rgb,
    )
    return mask, rgb, modality


def render_item_camera_direction_inputs(
    mesh, item, camera, vertex_rgb, *, return_object_coordinates=False,
):
    """Render RGB plus camera normals for one direction candidate."""
    spec, local = direction_candidate_render_spec(
        mesh, item, camera, vertex_rgb,
        return_object_coordinates=return_object_coordinates,
    )
    rendered = render_normal_edge_buffers_nvdiffrast(
        spec["vertices"], spec["faces"], camera,
        vertex_rgb=vertex_rgb, return_normal_map=True,
        vertex_coordinates=(local if return_object_coordinates else None),
    )
    rgb = rendered[2] if vertex_rgb is not None else rendered[3]
    result = (rendered[0], rgb, rendered[3], (
        "rgb" if vertex_rgb is not None else "normal_fallback"
    ))
    return (*result, rendered[4]) if return_object_coordinates else result


def direction_candidate_render_spec(
    mesh, item, camera, vertex_rgb, *, return_object_coordinates=False,
):
    translation = item["translation_world_m"]
    local = pose_vertices(
        center_vertices_on_table(mesh.vertices),
        item["uniform_scale"], (0.0, 0.0), 0.0,
        scale_xyz=item.get("scale_xyz"),
        z=0.0,
        roll_x_deg=item.get("roll_x_deg", 0.0),
        roll_y_deg=item.get("roll_y_deg", 0.0),
    )
    posed = pose_vertices(
        local, 1.0, translation[:2], item["yaw_deg"],
        z=translation[2] if len(translation) > 2 else 0.0,
    )
    return {
        "vertices": posed,
        "faces": mesh.faces,
        "vertex_rgb": vertex_rgb,
        "vertex_coordinates": local if return_object_coordinates else None,
    }, local


def direction_candidate_roi(reference_mask, specs, camera, padding=8):
    """Return a camera-coordinate ROI containing the reference and candidates."""
    reference_mask = np.asarray(reference_mask) > 0
    rows, columns = np.nonzero(reference_mask)
    if not len(rows):
        raise ValueError("direction candidate ROI requires a non-empty reference mask")
    reference_width = int(columns.max() - columns.min() + 1)
    reference_height = int(rows.max() - rows.min() + 1)
    reference_side = int(math.ceil(
        1.4 * max(reference_width, reference_height),
    ))
    center_x = 0.5 * float(columns.min() + columns.max())
    center_y = 0.5 * float(rows.min() + rows.max())
    x0 = int(math.floor(center_x - 0.5 * reference_side))
    y0 = int(math.floor(center_y - 0.5 * reference_side))
    x1, y1 = x0 + reference_side, y0 + reference_side
    width, height = map(int, camera["image_size"])
    for spec in specs:
        pixels, depth = project_points(spec["vertices"], camera)
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        if not np.any(valid):
            continue
        visible = pixels[valid]
        x0 = min(x0, int(math.floor(np.clip(visible[:, 0].min(), 0, width - 1))))
        x1 = max(x1, int(math.ceil(np.clip(visible[:, 0].max(), 0, width - 1))) + 1)
        y0 = min(y0, int(math.floor(np.clip(visible[:, 1].min(), 0, height - 1))))
        y1 = max(y1, int(math.ceil(np.clip(visible[:, 1].max(), 0, height - 1))) + 1)
    padding = max(0, int(padding))
    return (
        max(0, x0 - padding), max(0, y0 - padding),
        min(width, x1 + padding), min(height, y1 + padding),
    )


def camera_roi_array(value, roi, image_size, fill=0):
    """Copy a full-camera array into a padded fixed-size ROI canvas."""
    value = np.asarray(value)
    x0, y0, _, _ = map(int, roi)
    width, height = map(int, image_size)
    output = np.full((height, width, *value.shape[2:]), fill, dtype=value.dtype)
    copy_height = min(height, max(0, value.shape[0] - y0))
    copy_width = min(width, max(0, value.shape[1] - x0))
    if copy_height and copy_width:
        output[:copy_height, :copy_width] = value[
            y0:y0 + copy_height, x0:x0 + copy_width,
        ]
    return output


def direction_roi_camera(camera, roi, image_size):
    intrinsic = np.asarray(camera["intrinsic"], dtype=float).copy()
    intrinsic[0, 2] -= int(roi[0])
    intrinsic[1, 2] -= int(roi[1])
    return {
        **camera,
        "image_size": list(map(int, image_size)),
        "intrinsic": intrinsic.tolist(),
    }


def batch_render_direction_candidates(
    entries, camera, scorer, chunk_size=4, reference_masks=None,
):
    """Render fixed yaw candidates across objects in CUDA range-mode chunks."""
    entries = list(entries)
    needs_coordinates = getattr(scorer, "requires_object_coordinates", False)
    specs = []
    modalities = []
    for _, mesh, item, vertex_rgb in entries:
        spec, _ = direction_candidate_render_spec(
            mesh, item, camera, vertex_rgb,
            return_object_coordinates=needs_coordinates,
        )
        specs.append(spec)
        modalities.append("rgb" if vertex_rgb is not None else "normal_fallback")
    rois = {}
    if reference_masks:
        grouped_indices = {key: [] for key in reference_masks}
        for index, (key, _, _, _) in enumerate(entries):
            grouped_indices[key[0]].append(index)
        rois = {
            key: direction_candidate_roi(
                reference_masks[key], [specs[index] for index in indices], camera,
            )
            for key, indices in grouped_indices.items()
        }
        bins = []
        for key in sorted(
            grouped_indices,
            key=lambda value: (rois[value][2] - rois[value][0])
            * (rois[value][3] - rois[value][1]),
            reverse=True,
        ):
            if (
                bins and len(grouped_indices[key]) <= chunk_size
                and sum(len(grouped_indices[value]) for value in bins[-1])
                + len(grouped_indices[key]) <= chunk_size
            ):
                bins[-1].append(key)
            else:
                bins.append([key])
        rendered = [None] * len(specs)
        for keys in bins:
            indices = [index for key in keys for index in grouped_indices[key]]
            raster_size = [
                max(rois[key][2] - rois[key][0] for key in keys),
                max(rois[key][3] - rois[key][1] for key in keys),
            ]
            for index in indices:
                key = entries[index][0][0]
                specs[index]["camera"] = direction_roi_camera(
                    camera, rois[key], raster_size,
                )
            outputs = render_normal_edge_buffers_batch_nvdiffrast(
                [specs[index] for index in indices],
                {**camera, "image_size": raster_size},
                chunk_size=chunk_size,
            )
            for index, output in zip(indices, outputs):
                rendered[index] = output
    else:
        rendered = render_normal_edge_buffers_batch_nvdiffrast(
            specs, camera, chunk_size=chunk_size,
        )
    results = {}
    for index, ((key, _, _, _), output, modality) in enumerate(zip(
        entries, rendered, modalities,
    )):
        if rois:
            x0, y0 = rois[key[0]][:2]
            full_width, full_height = map(int, camera["image_size"])
            valid_width = min(output[0].shape[1], full_width - x0)
            valid_height = min(output[0].shape[0], full_height - y0)
            clipped = []
            for buffer_index, value in enumerate(output):
                if value is None:
                    clipped.append(None)
                    continue
                value = np.asarray(value).copy()
                fill = 127 if buffer_index == 3 else 0
                value[valid_height:] = fill
                value[:, valid_width:] = fill
                clipped.append(value)
            output = tuple(clipped)
        candidate = {
            "mask": output[0],
            "rgb": output[2] if output[2] is not None else output[3],
            "normal_map": output[3],
            "candidate_modality": modality,
            "_rendered_buffers": output,
        }
        if rois:
            candidate["_render_roi"] = rois[key[0]]
            candidate["_render_camera"] = specs[index]["camera"]
        if needs_coordinates:
            candidate["object_coordinates"] = output[4]
        results[key] = candidate
    return results


def rendered_direction_candidate(mesh, item, camera, vertex_rgb, scorer):
    needs_coordinates = getattr(scorer, "requires_object_coordinates", False)
    rendered = render_item_camera_direction_inputs(
        mesh, item, camera, vertex_rgb,
        return_object_coordinates=needs_coordinates,
    )
    mask, rgb, normal_map, modality = rendered[:4]
    candidate = {
        "mask": mask, "rgb": rgb, "normal_map": normal_map,
        "candidate_modality": modality,
    }
    if needs_coordinates:
        candidate["object_coordinates"] = rendered[4]
    return candidate


def score_direction_candidates(
    scorer, reference_key, reference_rgb, reference_mask, camera, candidates,
):
    """Score rendered direction candidates."""
    if hasattr(scorer, "score_candidates"):
        return scorer.score_candidates(
            reference_key, reference_rgb, reference_mask, camera, candidates,
        )
    return {
        key: scorer.score(
            reference_rgb, reference_mask,
            candidate["rgb"], candidate["mask"],
            candidate_modality=candidate["candidate_modality"],
        )
        for key, candidate in candidates.items()
    }


def score_multiview_direction_candidates(
    scorer, reference_key, reference_rgbs, reference_masks, cameras, candidates,
):
    if hasattr(scorer, "score_multiview_candidates"):
        return scorer.score_multiview_candidates(
            reference_key, reference_rgbs, reference_masks, cameras, candidates,
        )
    return score_direction_candidates(
        scorer, reference_key,
        reference_rgbs[reference_key], reference_masks[reference_key],
        cameras[reference_key], candidates[reference_key],
    )


def precompute_batched_direction_groups(groups, camera, scorer, chunk_size=4):
    """Render and score fixed candidates across object boundaries."""
    entries = [
        ((group_key, yaw), mesh, item, colors)
        for group_key, group in groups.items()
        for yaw, mesh, item, colors in group["entries"]
    ]
    render_start = time.perf_counter()
    rendered = batch_render_direction_candidates(
        entries, camera, scorer, chunk_size=chunk_size,
        reference_masks={
            key: group["reference_mask"] for key, group in groups.items()
        },
    )
    render_seconds = time.perf_counter() - render_start
    rendered_by_group = {key: {} for key in groups}
    score_groups = {}
    empty_scores = {key: {} for key in groups}
    for group_key, group in groups.items():
        candidates = {}
        for yaw, _, _, _ in group["entries"]:
            candidate = rendered[(group_key, yaw)]
            rendered_by_group[group_key][yaw] = candidate
            if np.any(candidate["mask"]):
                candidates[yaw] = {
                    key: value for key, value in candidate.items()
                    if not key.startswith("_")
                }
            else:
                empty_scores[group_key][yaw] = {
                    "loss": 1.0,
                    "foreground_mean_confidence": 0.0,
                    "backend": getattr(scorer, "backend", "direction_loss"),
                    "selection_source": "empty_render_penalty",
                }
        first_candidate = next(iter(rendered_by_group[group_key].values()), None)
        roi = first_candidate.get("_render_roi") if first_candidate else None
        render_size = (
            list(first_candidate["mask"].shape[::-1])
            if first_candidate is not None else camera["image_size"]
        )
        score_groups[group_key] = {
            "reference_key": group["reference_key"],
            "reference_rgb": (
                camera_roi_array(group["reference_rgb"], roi, render_size, 238)
                if roi is not None else group["reference_rgb"]
            ),
            "reference_mask": (
                camera_roi_array(group["reference_mask"], roi, render_size, False)
                if roi is not None else group["reference_mask"]
            ),
            "camera": (
                direction_roi_camera(camera, roi, render_size)
                if roi is not None else camera
            ),
            "candidates": candidates,
        }
    score_start = time.perf_counter()
    scores = scorer.score_candidate_groups(score_groups)
    score_seconds = time.perf_counter() - score_start
    for group_key, penalties in empty_scores.items():
        scores[group_key].update(penalties)
    print(
        "[direction batch timing] "
        f"render={render_seconds:.3f}s score={score_seconds:.3f}s "
        f"objects={len(groups)} candidates={len(entries)}",
        flush=True,
    )
    return rendered_by_group, scores


class DirectionBatchCoordinator:
    """Collect per-object candidate sets and score them in one GPU batch."""

    def __init__(self, expected_groups, camera, scorer):
        self.expected_groups = int(expected_groups)
        self.camera = camera
        self.scorer = scorer
        self.condition = Condition()
        self.groups = {}
        self.results = None
        self.error = None

    def score(self, key, group):
        with self.condition:
            if key in self.groups:
                raise ValueError(f"duplicate direction batch group: {key}")
            self.groups[key] = group
            producer = len(self.groups) == self.expected_groups
            if not producer:
                self.condition.wait_for(
                    lambda: self.results is not None or self.error is not None,
                )
                if self.error is not None:
                    raise self.error
                return self.results[key]
        try:
            start = time.perf_counter()
            _, results = precompute_batched_direction_groups(
                self.groups, self.camera, self.scorer,
            )
            print(
                "[top timing] cross_object_direction_batch="
                f"{time.perf_counter() - start:.3f}s "
                f"objects={len(self.groups)} candidates="
                f"{sum(len(group['entries']) for group in self.groups.values())}",
                flush=True,
            )
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
            raise
        with self.condition:
            self.results = results
            self.condition.notify_all()
            return self.results[key]




def minima_direction_camera(camera, max_size=1024):
    factor = min(1.0, float(max_size) / max(camera["image_size"]))
    return scaled_camera(camera, factor)


def direction_scoring_camera(scorer, camera):
    return (
        camera
        if getattr(scorer, "requires_full_resolution", False)
        else minima_direction_camera(camera)
    )


def support_pose_mode(support_parents, enabled=True):
    parents = tuple(support_parents or ("table_0",))
    if not enabled or not any(parent != "table_0" for parent in parents):
        return "disabled"
    return "z_only" if len(parents) == 1 else "z_and_roll_xy"


def top_camera_surface_height(vertices_world, top_camera, fraction=0.05):
    vertices = np.asarray(vertices_world, dtype=float)
    _, depth = project_points(vertices, top_camera)
    valid = np.isfinite(vertices).all(axis=1) & np.isfinite(depth) & (depth > 0)
    vertices, depth = vertices[valid], depth[valid]
    if not len(vertices):
        raise ValueError("support surface has no points visible to the top camera")
    count = max(1, int(math.ceil(len(vertices) * float(fraction))))
    nearest = np.argpartition(depth, count - 1)[:count]
    return float(vertices[nearest, 2].mean())


def initialize_support_pose_heights(
    blueprint, objects, top_camera, *, enabled=True,
):
    blueprint_objects = {
        item["object_id"]: item for item in blueprint["objects"]
    }
    pending = set(objects)
    surface_heights = {"table_0": 0.0}
    while pending:
        progressed = False
        for object_id in tuple(pending):
            placement = blueprint_objects[object_id].get("placement") or {}
            parents = tuple(placement.get("support_parents") or ("table_0",))
            if any(parent not in surface_heights for parent in parents):
                continue
            item = objects[object_id]
            mode = support_pose_mode(parents, enabled)
            item["support_parents"] = list(parents)
            item["support_pose_mode"] = mode
            if mode != "disabled":
                item["translation_world_m"][2] = float(np.mean([
                    surface_heights[parent] for parent in parents
                ]))
                item.setdefault("roll_x_deg", 0.0)
                item.setdefault("roll_y_deg", 0.0)
            vertices = center_vertices_on_table(
                search_mesh_z_up(
                    Path(item["geometry_asset"]), face_count=None,
                ).vertices
            )
            world = pose_vertices(
                vertices, item["uniform_scale"],
                item["translation_world_m"][:2], item["yaw_deg"],
                scale_xyz=item.get("scale_xyz"),
                z=item["translation_world_m"][2],
                roll_x_deg=item.get("roll_x_deg", 0.0),
                roll_y_deg=item.get("roll_y_deg", 0.0),
            )
            surface_heights[object_id] = top_camera_surface_height(
                world, top_camera,
            )
            pending.remove(object_id)
            progressed = True
        if not progressed:
            raise ValueError("support hierarchy is cyclic or references a missing object")
    return objects


def translation_for_yaw_about_local_pivot(
    translation_xy, pivot, scale_xyz, source_yaw_deg, target_yaw_deg,
):
    """Keep a local pivot fixed in world XY while changing object yaw."""
    pivot_xy = np.asarray(pivot, dtype=float)[:2] * np.asarray(
        scale_xyz, dtype=float,
    )[:2]

    def rotated(yaw_deg):
        angle = math.radians(float(yaw_deg))
        rotation = np.array((
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        ))
        return rotation @ pivot_xy

    anchor = np.asarray(translation_xy, dtype=float) + rotated(source_yaw_deg)
    return anchor - rotated(target_yaw_deg)


def fit_top_template_pose(vertices, pose, camera, margin_fraction=0.03):
    """Shrink an off-screen template about the endpoint opposite the overflow."""
    posed = pose_vertices(
        vertices, pose["uniform_scale"], pose["translation_world_m"][:2],
        pose["yaw_deg"], scale_xyz=pose.get("scale_xyz"),
    )
    pixels, depth = project_points(posed, camera)
    pixels = pixels[(depth > 1e-8) & np.isfinite(pixels).all(axis=1)]
    if not len(pixels):
        return dict(pose), {"applied": False, "reason": "no_visible_vertices"}

    width, height = map(float, camera["image_size"])
    margin = float(margin_fraction) * min(width, height)
    bounds = ((margin, width - margin), (margin, height - margin))
    low = pixels.min(axis=0)
    high = pixels.max(axis=0)
    overflow = [
        bool(low[axis] < bounds[axis][0] or high[axis] > bounds[axis][1])
        for axis in range(2)
    ]
    if not any(overflow):
        return dict(pose), {
            "applied": False,
            "reason": "already_inside_frame",
            "projected_bbox_xyxy": [*low.tolist(), *high.tolist()],
        }

    origin = project_points(
        [[*pose["translation_world_m"][:2], 0.0]], camera,
    )[0][0]
    opposite = np.zeros(2)
    for axis, (minimum, maximum) in enumerate(bounds):
        over_low = low[axis] < minimum
        over_high = high[axis] > maximum
        if over_low and not over_high:
            opposite[axis] = 1.0
        elif over_high and not over_low:
            opposite[axis] = -1.0
    spans = np.maximum(high - low, 1e-8)
    anchor = (
        pixels[np.argmax(((pixels - (low + high) * 0.5) / spans) @ opposite)]
        if np.any(opposite) else origin.copy()
    )
    for axis, (minimum, maximum) in enumerate(bounds):
        if not minimum <= anchor[axis] <= maximum:
            anchor[axis] = float(np.clip(origin[axis], minimum, maximum))

    scale_factor = 1.0
    for axis, (minimum, maximum) in enumerate(bounds):
        if low[axis] < minimum:
            scale_factor = min(
                scale_factor,
                (anchor[axis] - minimum) / max(anchor[axis] - low[axis], 1e-8),
            )
        if high[axis] > maximum:
            scale_factor = min(
                scale_factor,
                (maximum - anchor[axis]) / max(high[axis] - anchor[axis], 1e-8),
            )
    scale_factor = float(np.clip(scale_factor * 0.98, 1e-3, 1.0))
    pixel_delta = (1.0 - scale_factor) * (anchor - origin)
    world_delta = _pixel_delta_to_world(
        pixel_delta, pose["translation_world_m"][:2], camera,
    )
    fitted = dict(pose)
    fitted["translation_world_m"] = [
        float(pose["translation_world_m"][0] + world_delta[0]),
        float(pose["translation_world_m"][1] + world_delta[1]),
        float(pose["translation_world_m"][2]),
    ]
    fitted["uniform_scale"] = float(pose["uniform_scale"] * scale_factor)
    if pose.get("scale_xyz") is not None:
        fitted["scale_xyz"] = (
            np.asarray(pose["scale_xyz"], dtype=float) * scale_factor
        ).tolist()
    return fitted, {
        "applied": True,
        "reason": "projected_bbox_outside_frame",
        "scale_factor": scale_factor,
        "anchor_pixel": anchor.tolist(),
        "projected_bbox_xyxy": [*low.tolist(), *high.tolist()],
        "safe_frame_xyxy": [margin, margin, width - margin, height - margin],
        "overflow_left_top_right_bottom": [
            bool(low[0] < bounds[0][0]), bool(low[1] < bounds[1][0]),
            bool(high[0] > bounds[0][1]), bool(high[1] > bounds[1][1]),
        ],
    }


def render_mesh_mask(mesh, vertices, camera, fill_silhouette=True):
    width, height = camera["image_size"]
    pixels, depth = project_points(vertices, camera)
    faces = np.asarray(mesh.faces)
    valid = np.all(depth[faces] > 1e-8, axis=1) & np.isfinite(pixels[faces]).all(axis=(1, 2))
    polygons = np.rint(pixels[faces[valid]]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygons):
        cv2.fillPoly(mask, polygons, 255)
    return solid_external_silhouette(mask) if fill_silhouette else mask


def render_mesh_edges(
    mesh, vertices, camera, silhouette=None, visible_only=False, crease_angle_deg=25.0,
):
    width, height = camera["image_size"]
    pixels, depth = project_points(vertices, camera)
    edges = []
    if len(mesh.face_adjacency):
        sharp = np.asarray(mesh.face_adjacency_angles) > math.radians(crease_angle_deg)
        if visible_only:
            triangles = np.asarray(vertices)[np.asarray(mesh.faces)]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0],
            )
            view = np.asarray(camera["center_world_m"], dtype=float) - triangles.mean(axis=1)
            front_facing = np.einsum("ij,ij->i", normals, view) > 0.0
            sharp &= np.any(front_facing[np.asarray(mesh.face_adjacency)], axis=1)
        edges.extend(np.asarray(mesh.face_adjacency_edges)[sharp].tolist())
    output = np.zeros((height, width), dtype=np.uint8)
    for first, second in edges:
        if depth[first] <= 0 or depth[second] <= 0 or not np.isfinite(pixels[[first, second]]).all():
            continue
        a, b = np.rint(pixels[[first, second]]).astype(int)
        cv2.line(output, tuple(a), tuple(b), 255, 1, cv2.LINE_8)
    if silhouette is not None:
        output |= thin_external_boundary(silhouette).astype(np.uint8) * 255
    return output > 0


def table_scale_pivot_offset(table):
    pivot = table.get("scale_pivot")
    if not pivot:
        return np.zeros(3, dtype=float)
    offset = np.zeros(3, dtype=float)
    axis = {"x": 0, "y": 1, "z": 2}[pivot["axis"]]
    offset[axis] = (
        float(pivot["offset_m"])
        if "offset_m" in pivot else
        float(pivot["local_coordinate_m"]) * (
            float(pivot["reference_scale"]) - float(table["scale_xyz"][axis])
        )
    )
    return offset


def transform_table_vertices(vertices, table):
    """Scale the tabletop normally and change only the leg length in Z."""
    centered = orient_table_vertices(
        center_vertices_at_tabletop(vertices), table.get("yaw_deg", 0.0),
    )
    scale = np.asarray(table["scale_xyz"], dtype=float)
    transformed = centered * scale
    pivot = table.get("scale_pivot") or {}
    if "leg_scale_z" in pivot:
        anchor = float(pivot["leg_anchor_local_z"])
        below = centered[:, 2] < anchor
        transformed[below, 2] = (
            anchor * scale[2]
            + (centered[below, 2] - anchor) * float(pivot["leg_scale_z"])
        )
    return transformed + table_scale_pivot_offset(table)


def scene_render_mesh(item, face_count=3000):
    geometry_asset = item.get("geometry_asset")
    if geometry_asset is None:
        package = Path(item["asset"])
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        geometry_asset = package / manifest["scene_rigid_mesh"]
    mesh = search_mesh_z_up(Path(geometry_asset), face_count=face_count)
    if item.get("asset_orientation_matrix") is not None:
        mesh = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(item["asset_orientation_matrix"], dtype=float)
        mesh.apply_transform(transform)
    return mesh


def scene_render_maps(document, camera):
    """Render the table and every object into one diagnostic mask/edge map."""
    width, height = camera["image_size"]
    scene_mask = np.zeros((height, width), dtype=np.uint8)
    scene_edges = np.zeros((height, width), dtype=bool)

    table = document["table"]
    table_mesh = search_mesh_z_up(Path(table["mesh"]))
    table_vertices = transform_table_vertices(table_mesh.vertices, table)
    table_mask = render_mesh_mask(table_mesh, table_vertices, camera)
    scene_mask |= np.asarray(table_mask, dtype=np.uint8)
    scene_edges |= render_mesh_edges(
        table_mesh, table_vertices, camera, table_mask,
        visible_only=True, crease_angle_deg=25.0,
    )

    for item in document["objects"].values():
        mesh = scene_render_mesh(item)
        vertices = center_vertices_on_table(mesh.vertices)
        scale_xyz = item.get("scale_xyz", [item["uniform_scale"]] * 3)
        vertices = pose_vertices(
            vertices, item["uniform_scale"], item["translation_world_m"][:2],
            item["yaw_deg"], scale_xyz=scale_xyz,
            z=item["translation_world_m"][2],
            roll_x_deg=item.get("roll_x_deg", 0.0),
            roll_y_deg=item.get("roll_y_deg", 0.0),
        )
        mask = render_mesh_mask(mesh, vertices, camera)
        scene_mask |= np.asarray(mask, dtype=np.uint8)
        scene_edges |= render_mesh_edges(
            mesh, vertices, camera, mask,
            visible_only=True, crease_angle_deg=25.0,
        )
    return scene_mask, scene_edges


def blend_rendered_scene(reference, rendered, rendered_mask, alpha=0.5):
    reference = np.asarray(reference, dtype=np.uint8)
    rendered = np.asarray(rendered, dtype=np.uint8)
    mask = np.asarray(rendered_mask) > 0
    if reference.shape != rendered.shape or reference.shape[:2] != mask.shape:
        raise ValueError("reference, render, and rendered mask sizes must match")
    output = reference.copy()
    output[mask] = np.rint(
        (1.0 - alpha) * reference[mask] + alpha * rendered[mask]
    ).astype(np.uint8)
    return output


def color_alignment_edges(image, reference_edges, rendered_edges):
    output = np.asarray(image, dtype=np.float32).copy()
    reference = edge_visualization_values(reference_edges).astype(np.float32) / 255.0
    rendered = edge_visualization_values(rendered_edges).astype(np.float32) / 255.0
    for strength, color in (
        (reference, np.array([40, 220, 60], dtype=np.float32)),
        (rendered, np.array([255, 60, 60], dtype=np.float32)),
        (np.minimum(reference, rendered), np.full(3, 255, dtype=np.float32)),
    ):
        output = output * (1.0 - strength[..., None]) + color * strength[..., None]
    return np.rint(output).clip(0, 255).astype(np.uint8)


def render_scene_normal_edges(objects, camera, table=None):
    """Render optimizer-style normal edges for all final object poses."""
    vertices, normals, faces, foreground = [], [], [], []
    offset = 0
    for item in objects.values():
        mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=None)
        scale_xyz = np.asarray(
            item.get("scale_xyz") or [item["uniform_scale"]] * 3, dtype=float,
        )
        roll_x = np.deg2rad(float(item.get("roll_x_deg", 0.0)))
        rotation_x = np.array((
            (1.0, 0.0, 0.0),
            (0.0, np.cos(roll_x), -np.sin(roll_x)),
            (0.0, np.sin(roll_x), np.cos(roll_x)),
        ))
        roll_y = np.deg2rad(float(item.get("roll_y_deg", 0.0)))
        rotation_y = np.array((
            (np.cos(roll_y), 0.0, np.sin(roll_y)),
            (0.0, 1.0, 0.0),
            (-np.sin(roll_y), 0.0, np.cos(roll_y)),
        ))
        yaw = np.deg2rad(float(item["yaw_deg"]))
        rotation_z = np.array((
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ))
        rotation = rotation_z @ rotation_y @ rotation_x
        posed = pose_vertices(
            center_vertices_on_table(mesh.vertices),
            item["uniform_scale"],
            item["translation_world_m"][:2],
            item["yaw_deg"],
            scale_xyz=scale_xyz,
            z=item["translation_world_m"][2],
            roll_x_deg=item.get("roll_x_deg", 0.0),
            roll_y_deg=item.get("roll_y_deg", 0.0),
        )
        vertices.append(posed)
        posed_normals = (np.asarray(mesh.vertex_normals) / scale_xyz) @ rotation.T
        normals.append(posed_normals / np.linalg.norm(
            posed_normals, axis=1, keepdims=True,
        ).clip(min=1e-12))
        faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
        foreground.append(np.ones(len(posed), dtype=np.float32))
        offset += len(posed)
    if not vertices:
        width, height = camera["image_size"]
        return np.zeros((height, width), dtype=bool)
    if table is not None:
        table_mesh = search_mesh_z_up(Path(table["mesh"]), face_count=12000)
        table_vertices = transform_table_vertices(table_mesh.vertices, table)
        vertices.append(table_vertices)
        table_normals = orient_table_vertices(
            table_mesh.vertex_normals, table.get("yaw_deg", 0.0),
        ) / np.asarray(table["scale_xyz"], dtype=float)
        normals.append(table_normals / np.linalg.norm(
            table_normals, axis=1, keepdims=True,
        ).clip(min=1e-12))
        faces.append(np.asarray(table_mesh.faces, dtype=np.int32) + offset)
        foreground.append(np.zeros(len(table_vertices), dtype=np.float32))
    return render_normal_edges_nvdiffrast(
        np.concatenate(vertices), np.concatenate(faces), camera,
        vertex_foreground=np.concatenate(foreground),
        vertex_normals=np.concatenate(normals),
    )


def thin_external_boundary(mask):
    output = np.zeros_like(np.asarray(mask, dtype=np.uint8))
    contours, _ = cv2.findContours(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(output, contours, -1, 255, 1, cv2.LINE_8)
    return output > 0


def full_table_alignment_edges(table, camera, reference_mask):
    reference_mask = resize_nearest(
        np.asarray(reference_mask) > 0, tuple(camera["image_size"]),
    )
    mesh = search_mesh_z_up(Path(table["mesh"]), face_count=12000)
    vertices = transform_table_vertices(mesh.vertices, table)
    rendered_mask = render_mesh_mask(mesh, vertices, camera)
    return (
        thin_external_boundary(reference_mask).astype(np.uint8) * 255,
        thin_external_boundary(rendered_mask).astype(np.uint8) * 255,
    )


def sample_edge_points(edges, max_points=192):
    values = np.asarray(edges)
    weights = values.astype(float)
    if not is_soft_edge_map(values):
        weights = values.astype(bool).astype(float)
    dense = np.column_stack(np.where(weights > 0))[:, ::-1].astype(float)
    dense_weights = weights[weights > 0]
    points = dense
    if len(points) > max_points:
        cumulative = np.cumsum(dense_weights)
        indices = np.searchsorted(
            cumulative,
            np.linspace(0.0, cumulative[-1], max_points, endpoint=False)
            + 0.5 * cumulative[-1] / max_points,
        )
        points = dense[np.unique(np.minimum(indices, len(dense) - 1))]
    return points


def _spaced_edge_sample_indices(
    edges, mask, spacing_fraction=EDGE_SAMPLE_SPACING_FRACTION,
):
    values = np.asarray(edges)
    occupied = values > 0
    dense = np.column_stack(np.where(occupied))[:, ::-1].astype(float)
    if not len(dense):
        return np.empty(0, dtype=int)
    spacing = max(
        1.0,
        float(spacing_fraction) * math.sqrt(max(np.count_nonzero(mask), 1)),
    )
    cells = np.floor(dense / spacing).astype(np.int64)
    strongest_first = np.argsort(-values[occupied].astype(float), kind="stable")
    _, first = np.unique(
        cells[strongest_first], axis=0, return_index=True,
    )
    return np.sort(strongest_first[first])


def _sample_edge_features_gpu(
    edges, max_points, neighborhood_radius, min_neighbors, confidence_threshold,
    sampling_mask=None, sampling_spacing_fraction=EDGE_SAMPLE_SPACING_FRACTION,
):
    import torch

    values = np.asarray(edges)
    weighted = is_soft_edge_map(values)
    device = alignment_torch_device()
    weights = torch.as_tensor(values, dtype=torch.float64, device=device)
    if not weighted:
        weights = (weights > 0).to(torch.float64)
    occupied = weights > 0
    dense = torch.nonzero(occupied, as_tuple=False).flip(1).to(torch.float64)
    dense_weights = weights[occupied]
    if not len(dense):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    spaced_indices = (
        _spaced_edge_sample_indices(
            values, sampling_mask, sampling_spacing_fraction,
        )
        if sampling_mask is not None else None
    )
    if spaced_indices is not None:
        indices = torch.as_tensor(
            spaced_indices, dtype=torch.int64, device=device,
        )
    elif weighted and len(dense) > max_points:
        cumulative = torch.cumsum(dense_weights, dim=0)
        samples = torch.arange(
            max_points, dtype=torch.float64, device=device,
        ) * cumulative[-1] / max_points + 0.5 * cumulative[-1] / max_points
        indices = torch.searchsorted(cumulative, samples).clamp_max(
            len(dense) - 1,
        )
        indices = torch.unique(indices)
    else:
        indices = (
            torch.linspace(
                0, len(dense) - 1, max_points, device=device,
            ).to(torch.int64)
            if len(dense) > max_points else torch.arange(len(dense), device=device)
        )
    points = dense[indices]
    tangents = torch.zeros_like(points)
    confidence = torch.zeros(len(points), dtype=torch.float64, device=device)
    radius_squared = float(neighborhood_radius) ** 2
    for start in range(0, len(points), 64):
        stop = min(start + 64, len(points))
        offsets = dense[None] - points[start:stop, None]
        neighborhood = offsets.square().sum(dim=2) <= radius_squared
        local_weights = neighborhood * dense_weights[None]
        covariance = torch.einsum(
            "bni,bnj,bn->bij", offsets, offsets, local_weights,
        ) / local_weights.sum(dim=1).clamp_min(1e-12)[:, None, None]
        values, vectors = torch.linalg.eigh(covariance)
        score = (values[:, -1] - values[:, 0]) / (
            values[:, -1] + values[:, 0]
        ).clamp_min(1e-12)
        valid = (neighborhood.sum(dim=1) >= int(min_neighbors)) & (
            score >= float(confidence_threshold)
        )
        tangents[start:stop] = torch.where(
            valid[:, None], vectors[:, :, -1], tangents[start:stop],
        )
        confidence[start:stop] = torch.where(
            valid, score, confidence[start:stop],
        )
    return (
        points.cpu().numpy(), tangents.cpu().numpy(), confidence.cpu().numpy(),
    )


def _sample_edge_features_unlocked(
    edges, max_points=192, neighborhood_radius=6.0,
    min_neighbors=5, confidence_threshold=0.5,
    sampling_mask=None, sampling_spacing_fraction=EDGE_SAMPLE_SPACING_FRACTION,
):
    """Sample edge points with local PCA tangents and straightness confidence."""
    if alignment_torch_device().type == "cuda":
        return _sample_edge_features_gpu(
            edges, max_points, neighborhood_radius,
            min_neighbors, confidence_threshold,
            sampling_mask, sampling_spacing_fraction,
        )
    values = np.asarray(edges)
    weights = values.astype(float)
    weighted = is_soft_edge_map(values)
    if not weighted:
        weights = values.astype(bool).astype(float)
    occupied = weights > 0
    dense = np.column_stack(np.where(occupied))[:, ::-1].astype(float)
    dense_weights = weights[occupied]
    if not len(dense):
        return dense, np.empty((0, 2)), np.empty(0)
    spaced_indices = (
        _spaced_edge_sample_indices(
            values, sampling_mask, sampling_spacing_fraction,
        )
        if sampling_mask is not None else None
    )
    if spaced_indices is not None:
        indices = spaced_indices
    elif weighted and len(dense) > max_points:
        cumulative = np.cumsum(dense_weights)
        indices = np.searchsorted(
            cumulative,
            np.linspace(0.0, cumulative[-1], max_points, endpoint=False)
            + 0.5 * cumulative[-1] / max_points,
        )
        indices = np.unique(np.minimum(indices, len(dense) - 1))
    else:
        indices = (
            np.linspace(0, len(dense) - 1, max_points, dtype=int)
            if len(dense) > max_points else np.arange(len(dense))
        )
    points = dense[indices]
    tangents = np.zeros((len(points), 2), dtype=float)
    confidence = np.zeros(len(points), dtype=float)
    radius_squared = float(neighborhood_radius) ** 2
    for index, point in enumerate(points):
        neighborhood = np.sum((dense - point) ** 2, axis=1) <= radius_squared
        offsets = dense[neighborhood] - point
        if len(offsets) < int(min_neighbors):
            continue
        local_weights = dense_weights[neighborhood]
        covariance = (
            (offsets * local_weights[:, None]).T @ offsets
            / max(float(local_weights.sum()), 1e-12)
        )
        values, vectors = np.linalg.eigh(covariance)
        largest, smallest = float(values[-1]), float(values[0])
        score = (largest - smallest) / max(largest + smallest, 1e-12)
        if score < float(confidence_threshold):
            continue
        tangents[index] = vectors[:, -1]
        confidence[index] = score
    return points, tangents, confidence


def sample_edge_features(
    edges, max_points=192, neighborhood_radius=6.0,
    min_neighbors=5, confidence_threshold=0.5,
    sampling_mask=None, sampling_spacing_fraction=EDGE_SAMPLE_SPACING_FRACTION,
):
    # ponytail: one shared GPU/CPU edge-PCA lane avoids cross-object memory
    # thrashing; replace with a true cross-object batch only if profiling asks.
    with _TOP_EDGE_FEATURE_LOCK:
        return _sample_edge_features_unlocked(
            edges, max_points, neighborhood_radius,
            min_neighbors, confidence_threshold,
            sampling_mask, sampling_spacing_fraction,
        )


def sample_edge_features_batch(
    items, max_points=192, neighborhood_radius=6.0,
    min_neighbors=5, confidence_threshold=0.5,
    sampling_spacing_fraction=EDGE_SAMPLE_SPACING_FRACTION,
):
    """Sample independent edge maps with one chunked GPU PCA pass."""
    items = list(items)
    if alignment_torch_device().type != "cuda" or len(items) < 2:
        return [
            sample_edge_features(
                edges, max_points, neighborhood_radius,
                min_neighbors, confidence_threshold,
                sampling_mask=mask,
                sampling_spacing_fraction=sampling_spacing_fraction,
            )
            for edges, mask in items
        ]
    import torch

    prepared = []
    for edges, mask in items:
        values = np.asarray(edges)
        occupied = values > 0
        dense = np.column_stack(np.where(occupied))[:, ::-1].astype(float)
        weights = values[occupied].astype(float)
        if not is_soft_edge_map(values):
            weights = np.ones(len(dense), dtype=float)
        indices = _spaced_edge_sample_indices(
            values, mask, sampling_spacing_fraction,
        )
        prepared.append((dense, weights, dense[indices]))

    device = alignment_torch_device()
    results = [None] * len(prepared)
    nonempty = [index for index, value in enumerate(prepared) if len(value[2])]
    for index, value in enumerate(prepared):
        if not len(value[2]):
            results[index] = (
                np.empty((0, 2)), np.empty((0, 2)), np.empty(0),
            )
    while nonempty:
        max_dense = max(len(prepared[index][0]) for index in nonempty)
        batch_size = max(1, min(
            4, len(nonempty), 4_000_000 // max(1, 64 * max_dense),
        ))
        batch_indices, nonempty = nonempty[:batch_size], nonempty[batch_size:]
        dense_count = max(len(prepared[index][0]) for index in batch_indices)
        point_count = max(len(prepared[index][2]) for index in batch_indices)
        dense = torch.zeros(
            (len(batch_indices), dense_count, 2),
            dtype=torch.float64, device=device,
        )
        dense_weights = torch.zeros(
            (len(batch_indices), dense_count),
            dtype=torch.float64, device=device,
        )
        points = torch.zeros(
            (len(batch_indices), point_count, 2),
            dtype=torch.float64, device=device,
        )
        dense_valid = torch.zeros(
            (len(batch_indices), dense_count), dtype=torch.bool, device=device,
        )
        point_valid = torch.zeros(
            (len(batch_indices), point_count), dtype=torch.bool, device=device,
        )
        for local_index, source_index in enumerate(batch_indices):
            dense_value, weight_value, point_value = prepared[source_index]
            dense[local_index, :len(dense_value)] = torch.as_tensor(
                dense_value, dtype=torch.float64, device=device,
            )
            dense_weights[local_index, :len(weight_value)] = torch.as_tensor(
                weight_value, dtype=torch.float64, device=device,
            )
            points[local_index, :len(point_value)] = torch.as_tensor(
                point_value, dtype=torch.float64, device=device,
            )
            dense_valid[local_index, :len(dense_value)] = True
            point_valid[local_index, :len(point_value)] = True
        tangents = torch.zeros_like(points)
        confidence = torch.zeros(
            (len(batch_indices), point_count),
            dtype=torch.float64, device=device,
        )
        for start in range(0, point_count, 64):
            stop = min(start + 64, point_count)
            offsets = dense[:, None] - points[:, start:stop, None]
            neighborhood = (
                offsets.square().sum(dim=3) <= float(neighborhood_radius) ** 2
            ) & dense_valid[:, None]
            local_weights = neighborhood * dense_weights[:, None]
            covariance = torch.einsum(
                "bpni,bpnj,bpn->bpij",
                offsets, offsets, local_weights,
            ) / local_weights.sum(dim=2).clamp_min(1e-12)[..., None, None]
            values, vectors = torch.linalg.eigh(covariance)
            score = (values[..., -1] - values[..., 0]) / (
                values[..., -1] + values[..., 0]
            ).clamp_min(1e-12)
            valid = (
                (neighborhood.sum(dim=2) >= int(min_neighbors))
                & (score >= float(confidence_threshold))
                & point_valid[:, start:stop]
            )
            tangents[:, start:stop] = torch.where(
                valid[..., None], vectors[..., -1],
                tangents[:, start:stop],
            )
            confidence[:, start:stop] = torch.where(
                valid, score, confidence[:, start:stop],
            )
        tangents = tangents.cpu().numpy()
        confidence = confidence.cpu().numpy()
        for local_index, source_index in enumerate(batch_indices):
            value_points = prepared[source_index][2]
            count = len(value_points)
            results[source_index] = (
                value_points,
                tangents[local_index, :count],
                confidence[local_index, :count],
            )
    return results


def unbalanced_ot_loss(source, target, normalization=None, epsilon=0.015, tau=0.15, iterations=40):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if not len(source) or not len(target):
        return float("inf")
    if normalization is None:
        combined = np.vstack((source, target))
        normalization = max(float(np.ptp(combined, axis=0).max()), 1.0)
    cost = np.sum(((source[:, None] - target[None, :]) / normalization) ** 2, axis=2)
    kernel = np.exp(-cost / epsilon).clip(1e-100, None)
    source_mass = np.full(len(source), 1.0 / len(source))
    target_mass = np.full(len(target), 1.0 / len(target))
    exponent = tau / (tau + epsilon)
    left = np.ones_like(source_mass)
    right = np.ones_like(target_mass)
    for _ in range(iterations):
        left = (source_mass / (kernel @ right + 1e-100)) ** exponent
        right = (target_mass / (kernel.T @ left + 1e-100)) ** exponent
    transport = left[:, None] * kernel * right[None, :]

    def kl(first, second):
        first = np.asarray(first)
        second = np.asarray(second)
        return float(np.sum(first * np.log((first + 1e-100) / (second + 1e-100)) - first + second))

    reference = source_mass[:, None] * target_mass[None, :]
    return float(
        np.sum(transport * cost)
        + tau * kl(transport.sum(axis=1), source_mass)
        + tau * kl(transport.sum(axis=0), target_mass)
        + epsilon * kl(transport, reference)
    )


def balanced_sinkhorn_cost(source, target, normalization, epsilon=0.01, iterations=60):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if not len(source) or not len(target):
        return float("inf")
    cost = np.sum(((source[:, None] - target[None, :]) / normalization) ** 2, axis=2)
    kernel = np.exp(-cost / epsilon).clip(1e-100, None)
    source_mass = np.full(len(source), 1.0 / len(source))
    target_mass = np.full(len(target), 1.0 / len(target))
    left = np.ones_like(source_mass)
    right = np.ones_like(target_mass)
    for _ in range(iterations):
        left = source_mass / (kernel @ right + 1e-100)
        right = target_mass / (kernel.T @ left + 1e-100)
    transport = left[:, None] * kernel * right[None, :]
    reference = source_mass[:, None] * target_mass[None, :]
    entropy = np.sum(
        transport * np.log((transport + 1e-100) / (reference + 1e-100))
        - transport + reference
    )
    return float(np.sum(transport * cost) + epsilon * entropy)


def sinkhorn_divergence(source, target, normalization=None):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if not len(source) or not len(target):
        return float("inf")
    if normalization is None:
        combined = np.vstack((source, target))
        normalization = max(float(np.ptp(combined, axis=0).max()), 1.0)
    cross = balanced_sinkhorn_cost(source, target, normalization)
    source_self = balanced_sinkhorn_cost(source, source, normalization)
    target_self = balanced_sinkhorn_cost(target, target, normalization)
    return float(max(0.0, cross - 0.5 * source_self - 0.5 * target_self))


def sample_external_contour(mask, count=192):
    contours, _ = cv2.findContours(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.empty((0, 2), dtype=float)
    points = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 0:
        return points[:1]
    distances = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    segments = np.minimum(np.searchsorted(cumulative, distances, side="right") - 1, len(points) - 1)
    fraction = (distances - cumulative[segments]) / np.maximum(lengths[segments], 1e-12)
    return closed[segments] + fraction[:, None] * (closed[segments + 1] - closed[segments])


def normalized_contour(points):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    centered = points - center
    radius = math.sqrt(float(np.mean(np.sum(centered * centered, axis=1))))
    return centered / max(radius, 1e-12), center, radius


def reference_edge_detector(edge_detector):
    return "edge_drawing" if edge_detector == "geometry" else edge_detector


def scaled_edge_camera(camera, scale):
    scale = int(scale)
    if scale < 1:
        raise ValueError("top template render scale must be at least 1")
    intrinsic = np.asarray(camera["intrinsic"], dtype=float).copy()
    intrinsic[0, :] *= scale
    intrinsic[1, :] *= scale
    width, height = camera["image_size"]
    return {
        **camera,
        "image_size": [int(width) * scale, int(height) * scale],
        "intrinsic": intrinsic.tolist(),
    }


def load_rendered_edge_template(
    path, edge_detector="canny", mesh=None, pose=None, camera=None,
):
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    if camera is not None:
        expected_size = tuple(int(value) for value in camera["image_size"])
        if rgba.shape[1::-1] != expected_size:
            rgba = np.asarray(Image.fromarray(rgba).resize(
                expected_size, Image.Resampling.LANCZOS,
            ))
    mask = rgba[:, :, 3] > 16
    if not np.any(mask) and mesh is not None and pose is not None and camera is not None:
        vertices = center_vertices_on_table(mesh.vertices)
        scale_xyz = pose.get("scale_xyz", [pose["uniform_scale"]] * 3)
        posed = pose_vertices(
            vertices, pose["uniform_scale"], pose["translation_world_m"][:2],
            pose["yaw_deg"], scale_xyz=scale_xyz,
        )
        mask = render_mesh_mask(mesh, posed, camera) > 0
    if edge_detector == "mask_only":
        edges = np.zeros(mask.shape, dtype=np.uint8)
        metadata = {"method": "mask_only"}
    elif edge_detector == "geometry":
        if mesh is None or pose is None or camera is None:
            raise ValueError("geometry edges require mesh, pose, and camera")
        vertices = center_vertices_on_table(mesh.vertices)
        scale_xyz = pose.get("scale_xyz", [pose["uniform_scale"]] * 3)
        posed = pose_vertices(
            vertices, pose["uniform_scale"], pose["translation_world_m"][:2],
            pose["yaw_deg"], scale_xyz=scale_xyz,
        )
        edges = render_mesh_edges(
            mesh, posed, camera, mask, visible_only=True, crease_angle_deg=45.0,
        )
        metadata = {"method": "geometry", "crease_angle_deg": 45.0}
    else:
        edges, metadata = masked_edges(rgba[:, :, :3], mask, method=edge_detector)
    return mask.astype(np.uint8) * 255, edges, metadata


def warp_rendered_template(template, baseline, candidate, camera):
    template = np.asarray(template, dtype=np.uint8)
    if candidate.get("top_template_transform") is not None:
        matrix = np.asarray(candidate["top_template_transform"], dtype=float)
        transform_size = candidate.get("top_template_transform_image_size")
        if transform_size is not None:
            source_width, source_height = map(float, transform_size)
            target_height, target_width = template.shape[:2]
            scale = np.diag([
                target_width / source_width,
                target_height / source_height,
                1.0,
            ])
            matrix = scale @ matrix @ np.linalg.inv(scale)
        return cv2.warpAffine(
            template, matrix[:2], (template.shape[1], template.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    # The vertical top camera maps a positive world-Z yaw to the same visual
    # rotation used by OpenCV's affine transform.
    angle = float(candidate["yaw_deg"] - baseline["yaw_deg"])
    scale = float(candidate["uniform_scale"] / baseline["uniform_scale"])
    baseline_xy = np.array([[*baseline["translation_world_m"][:2], 0.0]])
    candidate_xy = np.array([[*candidate["translation_world_m"][:2], 0.0]])
    baseline_pixel = project_points(baseline_xy, camera)[0][0]
    candidate_pixel = project_points(candidate_xy, camera)[0][0]
    # Replay the physical Blender transform around the model's support pivot.
    # Rotating around each bitmap's foreground centroid makes the diagnostic
    # mask and edge maps disagree with the final rendered object.
    matrix = cv2.getRotationMatrix2D(tuple(baseline_pixel), angle, scale)
    matrix[:, 2] += candidate_pixel - baseline_pixel
    return cv2.warpAffine(
        template, matrix, (template.shape[1], template.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def ot_edge_loss(first, second):
    shape = np.broadcast_shapes(np.asarray(first).shape, np.asarray(second).shape)
    return unbalanced_ot_loss(
        sample_edge_points(first), sample_edge_points(second),
        normalization=float(max(shape)),
    )


def instance_mask(labels, ids, object_id):
    return np.where(labels == ids[object_id], 255, 0).astype(np.uint8)


def support_aware_reference_masks(labels, ids, objects):
    """Restore supported-object holes inside non-table support silhouettes."""
    raw = {
        object_id: instance_mask(labels, ids, object_id)
        for object_id in objects
        if object_id in ids
    }
    restored = {object_id: mask.copy() for object_id, mask in raw.items()}
    children_by_parent = {}
    for child_id, item in objects.items():
        if child_id not in raw:
            continue
        for parent_id in item.get("support_parents") or ():
            if parent_id != "table_0" and parent_id in raw:
                children_by_parent.setdefault(parent_id, []).append(child_id)
    for parent_id, child_ids in children_by_parent.items():
        inside_parent = solid_external_silhouette(raw[parent_id]) > 0
        supported = np.logical_or.reduce([raw[child_id] > 0 for child_id in child_ids])
        restored[parent_id][inside_parent & supported] = 255
    return restored


def target_mask(labels, ids, object_id):
    return solid_external_silhouette(instance_mask(labels, ids, object_id))


def filled_instance_appearance_mask(labels, ids, object_id):
    """Fill background holes for appearance losses without covering occluders."""
    object_label = ids[object_id]
    solid = solid_external_silhouette(instance_mask(labels, ids, object_id))
    occluded = (np.asarray(labels) != 0) & (np.asarray(labels) != object_label)
    return np.where(occluded, 0, solid).astype(np.uint8)


def tabletop_front_mask(labels, ids, min_width_fraction=0.5):
    """Keep the wide tabletop band of the front table mask and discard its legs."""
    raw = np.where(labels == ids["table_0"], 255, 0).astype(np.uint8)
    row_widths = np.count_nonzero(raw, axis=1)
    if not row_widths.any():
        return raw
    wide = row_widths >= float(min_width_fraction) * float(row_widths.max())
    wide_rows = np.flatnonzero(wide)
    if not len(wide_rows):
        return solid_external_silhouette(raw)
    gap_limit = max(2, int(round(0.01 * raw.shape[0])))
    segments = np.split(wide_rows, np.flatnonzero(np.diff(wide_rows) > gap_limit) + 1)
    peak_row = int(np.argmax(row_widths))
    tabletop_rows = min(
        segments,
        key=lambda rows: min(abs(int(rows[0]) - peak_row), abs(int(rows[-1]) - peak_row)),
    )
    clipped = raw.copy()
    clipped[int(tabletop_rows[-1]) + 1:] = 0
    return solid_external_silhouette(clipped)


def tabletop_refinement_target(labels, ids, plane_mask=None):
    if plane_mask is not None and np.any(plane_mask):
        return solid_external_silhouette(np.asarray(plane_mask, dtype=np.uint8))
    return tabletop_front_mask(labels, ids)


def tabletop_plane_geometry(mesh):
    """Return the dominant upper plane and similarly sized plane below it."""
    if not all(hasattr(mesh, name) for name in (
        "facets_normal", "facets_area", "facets", "triangles_center",
    )):
        vertices = np.asarray(mesh.vertices, dtype=float)
        top_z = float(vertices[:, 2].max())
        bottom_z = float(vertices[:, 2].min())
        return {
            "top_facet": None,
            "top_faces": None,
            "top_z": top_z,
            "lower_facet": -1 if top_z - bottom_z >= TABLETOP_MIN_THICKNESS_M else None,
            "bottom_z": bottom_z,
            "minimum_thickness_m": TABLETOP_MIN_THICKNESS_M,
            "maximum_lower_area_ratio": TABLETOP_MAX_LOWER_AREA_RATIO,
        }
    normals = np.asarray(mesh.facets_normal)
    areas = np.asarray(mesh.facets_area)
    upward = np.flatnonzero(normals[:, 2] > 0.9)
    if not len(upward):
        raise RuntimeError("table mesh has no upward-facing planar surface")
    facet_z = np.array([
        np.asarray(mesh.triangles_center)[faces, 2].mean()
        for faces in mesh.facets
    ])
    largest_upward_area = float(areas[upward].max())
    top_candidates = upward[areas[upward] >= 0.25 * largest_upward_area]
    top_facet = int(top_candidates[np.argmax(facet_z[top_candidates])])
    top_z = float(np.median(
        np.asarray(mesh.vertices)[np.unique(np.asarray(mesh.faces)[mesh.facets[top_facet]])][:, 2],
    ))
    z_span = float(np.ptp(np.asarray(mesh.vertices)[:, 2]))
    minimum_gap = max(TABLETOP_MIN_THICKNESS_M, 0.001 * z_span)
    lower = np.flatnonzero(
        (np.abs(normals[:, 2]) > 0.9)
        & (areas >= 0.25 * areas[top_facet])
        & (areas <= TABLETOP_MAX_LOWER_AREA_RATIO * areas[top_facet])
        & (facet_z <= top_z - minimum_gap)
    )
    lower_facet = int(lower[np.argmax(facet_z[lower])]) if len(lower) else None
    return {
        "top_facet": top_facet,
        "top_faces": np.asarray(mesh.facets[top_facet], dtype=int),
        "top_z": top_z,
        "lower_facet": lower_facet,
        "bottom_z": float(facet_z[lower_facet]) if lower_facet is not None else top_z,
        "minimum_thickness_m": TABLETOP_MIN_THICKNESS_M,
        "maximum_lower_area_ratio": TABLETOP_MAX_LOWER_AREA_RATIO,
    }


def source_tabletop_plane_geometry(mesh):
    """Measure tabletop planes on the original mesh when a source path exists."""
    metadata = getattr(mesh, "metadata", {}) or {}
    source = metadata.get("file_path")
    original = (
        search_mesh_z_up(Path(source), face_count=None)
        if source else mesh
    )
    return tabletop_plane_geometry(original)


def tabletop_surface_mesh(mesh, planes=None):
    """Return the tabletop slab bounded by its dominant top and lower planes."""
    planes = planes or source_tabletop_plane_geometry(mesh)
    if planes["top_faces"] is None:
        return mesh.copy() if hasattr(mesh, "copy") else mesh
    if planes["lower_facet"] is None:
        top_z = float(planes["top_z"])
        z_span = float(np.ptp(np.asarray(mesh.vertices)[:, 2]))
        tolerance = max(1e-5, 0.002 * z_span)
        face_z = np.asarray(mesh.vertices)[np.asarray(mesh.faces), 2]
        keep = np.all(np.abs(face_z - top_z) <= tolerance, axis=1)
        tabletop = mesh.copy()
        tabletop.update_faces(keep)
        tabletop.remove_unreferenced_vertices()
        return tabletop
    top_z = planes["top_z"]
    bottom_z = planes["bottom_z"]
    z_span = float(np.ptp(np.asarray(mesh.vertices)[:, 2]))
    tolerance = max(1e-5, 0.002 * z_span)
    face_z = np.asarray(mesh.vertices)[np.asarray(mesh.faces), 2]
    keep = np.all(
        (face_z >= bottom_z - tolerance) & (face_z <= top_z + tolerance),
        axis=1,
    )
    tabletop = mesh.copy()
    tabletop.update_faces(keep)
    tabletop.remove_unreferenced_vertices()
    return tabletop


def tabletop_alignment_mesh(mesh):
    """Return the tabletop proxy used to match the front lower edge.

    A reconstructed table can have an implausibly thick tabletop.  Its lower
    edge would then pull the stage-1 camera fit away from the scene.  Preserve
    the real mesh everywhere else, but cap this *alignment-only* proxy at 3%
    of the full table height when its slab exceeds 10%.
    """
    planes = source_tabletop_plane_geometry(mesh)
    tabletop = tabletop_surface_mesh(mesh, planes)
    full_vertices = np.asarray(mesh.vertices, dtype=float)
    slab_vertices = np.asarray(tabletop.vertices, dtype=float)
    full_height = float(np.ptp(full_vertices[:, 2]))
    top_z = (
        planes["top_z"] if planes["top_faces"] is not None
        else float(slab_vertices[:, 2].max())
    )
    bottom_z = (
        planes["bottom_z"] if planes["top_faces"] is not None
        else float(slab_vertices[:, 2].min())
    )
    thickness = top_z - bottom_z
    thickness_ratio = thickness / max(full_height, 1e-12)
    lower_plane_fallback = (
        planes["lower_facet"] is None
        if planes["top_faces"] is not None else
        thickness <= max(1e-12, 1e-6 * full_height)
    )
    report = {
        "alignment_only": True,
        "actual_tabletop_thickness_m": thickness,
        "full_table_height_m": full_height,
        "actual_thickness_height_ratio": thickness_ratio,
        "max_thickness_height_ratio": TABLETOP_ALIGNMENT_MAX_THICKNESS_RATIO,
        "replacement_thickness_height_ratio": (
            TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO
        ),
        "lower_plane_fallback": lower_plane_fallback,
        "applied": False,
    }
    if lower_plane_fallback:
        effective_thickness = max(
            TABLETOP_MIN_THICKNESS_M,
            full_height * TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO,
        )
        proxy = mesh.copy()
        proxy.vertices[:, 2] = np.clip(
            proxy.vertices[:, 2], top_z - effective_thickness, top_z,
        )
        report.update({
            "applied": True,
            "effective_tabletop_thickness_m": effective_thickness,
        })
        return proxy, report
    if (
        full_height <= 1e-12
        or thickness_ratio <= TABLETOP_ALIGNMENT_MAX_THICKNESS_RATIO
    ):
        report["effective_tabletop_thickness_m"] = thickness
        return tabletop, report

    effective_thickness = max(
        TABLETOP_MIN_THICKNESS_M,
        full_height * TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO,
    )
    proxy = tabletop.copy()
    relative_depth = np.clip(
        (top_z - slab_vertices[:, 2]) / max(thickness, 1e-12), 0.0, 1.0,
    )
    proxy.vertices[:, 2] = top_z - relative_depth * effective_thickness
    report.update({
        "applied": True,
        "effective_tabletop_thickness_m": effective_thickness,
    })
    return proxy, report


def tabletop_depth_thickness(mesh, planes):
    """Return a sane tabletop thickness for front-edge depth alignment."""
    full_height = float(np.ptp(np.asarray(mesh.vertices, dtype=float)[:, 2]))
    detected = max(
        TABLETOP_MIN_THICKNESS_M,
        float(planes["top_z"] - planes["bottom_z"]),
    )
    ratio = detected / max(full_height, 1e-12)
    missing_lower_plane = (
        "lower_facet" in planes and planes["lower_facet"] is None
    )
    fallback = full_height > 1e-12 and (
        missing_lower_plane or ratio > TABLETOP_DEPTH_MAX_THICKNESS_RATIO
    )
    effective = (
        max(
            TABLETOP_MIN_THICKNESS_M,
            full_height * TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO,
        )
        if fallback else detected
    )
    return effective, {
        "detected_tabletop_thickness_local_m": detected,
        "full_table_height_local_m": full_height,
        "detected_thickness_height_ratio": ratio,
        "maximum_thickness_height_ratio": TABLETOP_DEPTH_MAX_THICKNESS_RATIO,
        "fallback_thickness_height_ratio": (
            TABLETOP_ALIGNMENT_REPLACEMENT_THICKNESS_RATIO
        ),
        "lower_plane_fallback": missing_lower_plane,
        "tabletop_thickness_fallback_applied": fallback,
    }


def edge_drawing_silhouette_targets(
    image, labels, ids, object_ids, include_internal=False,
    edge_detector="teed",
):
    targets = {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_method = reference_edge_detector(edge_detector)
    full_frame_teed = (
        teed_edges(image)[0]
        if reference_method == "teed" else None
    )
    for object_id in object_ids:
        raw_mask = np.where(labels == ids[object_id], 255, 0).astype(np.uint8)
        support = solid_external_silhouette(raw_mask)
        detection_support = (
            cv2.erode(
                filled_instance_appearance_mask(labels, ids, object_id), kernel,
            ) if include_internal else support
        )
        detected, _ = (
            masked_full_frame_teed_edges(
                full_frame_teed, detection_support > 0,
            )
            if full_frame_teed is not None else
            masked_edges(image, detection_support > 0, method=reference_method)
        )
        silhouette_boundary = thin_external_boundary(support)
        boundary_band = cv2.dilate(
            silhouette_boundary.astype(np.uint8), kernel,
        ).astype(bool)
        detected = np.asarray(detected)
        if detected.dtype == bool:
            detected = detected.astype(np.uint8) * 255
        if include_internal and reference_method == "teed":
            detected = np.where(
                detected >= TEED_TARGET_INTERNAL_EDGE_MIN_CONFIDENCE,
                detected, 0,
            ).astype(np.uint8)
        target = (
            detected
            if include_internal else
            np.where(boundary_band, detected, 0)
        )
        targets[object_id] = np.maximum(
            target.astype(np.uint8),
            silhouette_boundary.astype(np.uint8) * 255,
        )
    return targets


def rotated_bbox_square_ratio(mask):
    """Return min/max side ratio of the mask's minimum-area rotated box."""
    y, x = np.nonzero(np.asarray(mask) > 0)
    if len(x) < 3:
        return 0.0
    (_, _), (width, height), _ = cv2.minAreaRect(
        np.column_stack((x, y)).astype(np.float32)
    )
    longest = max(float(width), float(height))
    return 0.0 if longest <= 0.0 else min(float(width), float(height)) / longest


def rendered_model_square_ratio(path):
    """Measure top-view squareness from the rendered model alpha mask."""
    alpha = np.asarray(Image.open(path).convert("RGBA"))[:, :, 3]
    return rotated_bbox_square_ratio(alpha > 16)


def model_mask_rotational_information(mask):
    """Return model-only directional information from 180-degree asymmetry."""
    mask = np.asarray(mask) > 0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rotated = cropped[::-1, ::-1]
    union = np.count_nonzero(cropped | rotated)
    return float(
        np.clip(1.0 - np.count_nonzero(cropped & rotated) / max(union, 1), 0.0, 1.0)
    )


FRONT_FOUR_WAY_SQUARE_RATIO_THRESHOLD = 0.8
NEAR_SQUARE_FOOTPRINT_ASPECT_RATIO = 1.1
CONTACT_BBOX_SQUARE_RATIO_THRESHOLD = 0.9
BODY_CIRCLE_SECTION_FRACTIONS = tuple(value / 10.0 for value in range(1, 10))
BODY_CIRCLE_MIN_STABLE_SECTIONS = 3
BODY_CIRCLE_MIN_AREA_RATIO = 0.25


def model_top_yaw_offsets(
    square_ratio,
    square_ratio_threshold=FRONT_FOUR_WAY_SQUARE_RATIO_THRESHOLD,
):
    """Use quarter turns when the model top view does not reveal a long axis."""
    if float(square_ratio) >= float(square_ratio_threshold):
        return (0.0, 90.0, 180.0, 270.0)
    return (0.0, 180.0)


def contour_circle_score(contour):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (_, _), radius = cv2.minEnclosingCircle(contour)
    if area <= 0.0 or perimeter <= 0.0 or radius <= 0.0:
        return 0.0
    circularity = 4.0 * math.pi * area / perimeter ** 2
    enclosing_circle_fill = area / (math.pi * float(radius) ** 2)
    return float(np.clip(min(circularity, enclosing_circle_fill), 0.0, 1.0))


def mask_circle_score(mask):
    """Return a circle-likeness score that rejects square and elongated masks."""
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    return contour_circle_score(max(contours, key=cv2.contourArea)) if contours else 0.0


def rendered_model_circle_score(path):
    alpha = np.asarray(Image.open(path).convert("RGBA"))[:, :, 3]
    return mask_circle_score(alpha > 16)


def bottom_contact_shape(mesh):
    """Measure circularity and square-bbox ratio of the lowest contact face."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if not len(vertices) or not len(faces):
        return {"circle_score": 0.0, "bbox_square_ratio": 0.0}
    z_min = float(vertices[:, 2].min())
    z_span = float(np.ptp(vertices[:, 2]))
    face_z = (
        vertices[faces[:, 0], 2]
        + vertices[faces[:, 1], 2]
        + vertices[faces[:, 2], 2]
    ) / 3.0
    low_faces = np.flatnonzero(
        face_z <= z_min + max(1e-5, 0.02 * z_span)
    )
    triangles = vertices[faces[low_faces]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    contact_faces = low_faces[
        normals[:, 2] < -0.9 * np.maximum(lengths, 1e-12)
    ]
    if not len(contact_faces):
        return {"circle_score": 0.0, "bbox_square_ratio": 0.0}
    points = vertices[np.unique(faces[contact_faces])][:, :2]
    if len(points) < 3:
        return {"circle_score": 0.0, "bbox_square_ratio": 0.0}
    contour = cv2.convexHull(points.astype(np.float32))
    width, height = cv2.minAreaRect(contour)[1]
    return {
        "circle_score": contour_circle_score(contour),
        "bbox_square_ratio": min(width, height) / max(width, height, 1e-12),
    }


def bottom_contact_circle_score(mesh):
    return float(bottom_contact_shape(mesh)["circle_score"])


def body_cross_section_shape(
    mesh, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
):
    """Measure stable radial symmetry across substantial horizontal sections."""
    vertices = np.asarray(mesh.vertices)
    z_min = float(vertices[:, 2].min()) if len(vertices) else 0.0
    z_span = float(np.ptp(vertices[:, 2])) if len(vertices) else 0.0
    slices = []
    if z_span > 0.0 and hasattr(mesh, "section"):
        for fraction in BODY_CIRCLE_SECTION_FRACTIONS:
            try:
                section = mesh.section(
                    plane_origin=[0.0, 0.0, z_min + fraction * z_span],
                    plane_normal=[0.0, 0.0, 1.0],
                )
            except (TypeError, ValueError):
                section = None
            candidates = []
            if section is not None:
                for points in section.discrete:
                    points = np.asarray(points, dtype=np.float32)
                    if len(points) < 3:
                        continue
                    contour = cv2.convexHull(
                        np.ascontiguousarray(points[:, :2]),
                    )
                    area = float(cv2.contourArea(contour))
                    width, height = cv2.minAreaRect(contour)[1]
                    if area <= 0.0 or width <= 0.0 or height <= 0.0:
                        continue
                    candidates.append({
                        "fraction": float(fraction),
                        "area": area,
                        "circle_score": contour_circle_score(contour),
                        "bbox_square_ratio": min(width, height) / max(width, height),
                    })
            slices.append(
                max(candidates, key=lambda item: item["area"])
                if candidates else {
                    "fraction": float(fraction), "area": 0.0,
                    "circle_score": 0.0, "bbox_square_ratio": 0.0,
                }
            )

    maximum_area = max((item["area"] for item in slices), default=0.0)
    best_run = current_run = []
    for item in slices:
        item["area_ratio"] = item["area"] / max(maximum_area, 1e-12)
        item["qualifies"] = bool(
            item["area_ratio"] >= BODY_CIRCLE_MIN_AREA_RATIO
            and item["circle_score"] >= circle_score_threshold
            and item["bbox_square_ratio"] >= square_ratio_threshold
        )
        current_run = current_run + [item] if item["qualifies"] else []
        if len(current_run) > len(best_run):
            best_run = current_run
    return {
        "circle_score": min(
            (item["circle_score"] for item in best_run), default=0.0,
        ),
        "bbox_square_ratio": min(
            (item["bbox_square_ratio"] for item in best_run), default=0.0,
        ),
        "stable_slice_count": len(best_run),
        "required_stable_slice_count": BODY_CIRCLE_MIN_STABLE_SECTIONS,
        "minimum_area_ratio": BODY_CIRCLE_MIN_AREA_RATIO,
        "slices": slices,
    }


def body_cross_section_xy_scale_tied(
    shape, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
):
    run = longest_run = 0
    for item in shape.get("slices", ()):
        qualifies = (
            item.get("area_ratio", 0.0) >= shape.get(
                "minimum_area_ratio", BODY_CIRCLE_MIN_AREA_RATIO,
            )
            and item.get("circle_score", 0.0) >= circle_score_threshold
            and item.get("bbox_square_ratio", 0.0) >= square_ratio_threshold
        )
        run = run + 1 if qualifies else 0
        longest_run = max(longest_run, run)
    return longest_run >= shape.get(
        "required_stable_slice_count", BODY_CIRCLE_MIN_STABLE_SECTIONS,
    )


def contact_xy_scale_tied(
    shape, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
):
    return bool(
        shape["circle_score"] >= circle_score_threshold
        and shape["bbox_square_ratio"] >= square_ratio_threshold
    )


def object_xy_scale_tie_sources(
    top_circle_score, contact_shape, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
    body_shape=None,
):
    return {
        "top_view_circle": bool(
            float(top_circle_score) >= circle_score_threshold
        ),
        "contact_circle_and_square": contact_xy_scale_tied(
            contact_shape, circle_score_threshold, square_ratio_threshold,
        ),
        "stable_body_cross_section": bool(
            body_shape and body_cross_section_xy_scale_tied(
                body_shape, circle_score_threshold, square_ratio_threshold,
            )
        ),
    }


def object_xy_scale_tied(
    top_circle_score, contact_shape, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
    body_shape=None,
):
    return any(object_xy_scale_tie_sources(
        top_circle_score, contact_shape,
        circle_score_threshold, square_ratio_threshold, body_shape,
    ).values())


def object_contact_shape(item):
    if "contact_circle_score" in item and "contact_bbox_square_ratio" in item:
        return {
            "circle_score": float(item["contact_circle_score"]),
            "bbox_square_ratio": float(item["contact_bbox_square_ratio"]),
        }
    asset = Path(item.get("geometry_asset", item["asset"]))
    return bottom_contact_shape(search_mesh_z_up(asset, face_count=None))


def object_body_cross_section_shape(
    item, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
):
    if "body_cross_section_shape" in item:
        return item["body_cross_section_shape"]
    asset = Path(item.get("geometry_asset", item["asset"]))
    return body_cross_section_shape(
        search_mesh_z_up(asset, face_count=None),
        circle_score_threshold, square_ratio_threshold,
    )


def rotated_bbox_area(mask):
    y, x = np.nonzero(np.asarray(mask) > 0)
    if len(x) < 3:
        return float(max(len(x), 1))
    (_, _), (width, height), _ = cv2.minAreaRect(
        np.column_stack((x, y)).astype(np.float32)
    )
    return max(float(width) * float(height), 1.0)


def camera_with_pitch_delta(camera, pitch_delta_deg):
    """Rotate one camera around its own right axis without moving world geometry."""
    rotation = np.asarray(camera["rotation_world_from_camera"], dtype=float)
    axis = rotation[:, 0]
    axis /= np.linalg.norm(axis)
    angle = math.radians(float(pitch_delta_deg))
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    delta = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    result = dict(camera)
    result["rotation_world_from_camera"] = (delta @ rotation).tolist()
    return result


def camera_with_roll_delta(camera, roll_delta_deg):
    """Rotate one camera around its optical axis without moving its center."""
    rotation = np.asarray(camera["rotation_world_from_camera"], dtype=float)
    axis = rotation[:, 2]
    axis /= np.linalg.norm(axis)
    angle = math.radians(float(roll_delta_deg))
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    delta = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    result = dict(camera)
    result["rotation_world_from_camera"] = (delta @ rotation).tolist()
    return result


def camera_with_pitch_roll_delta(camera, pitch_delta_deg, roll_delta_deg):
    pitched = camera_with_pitch_delta(camera, pitch_delta_deg)
    return camera_with_roll_delta(pitched, roll_delta_deg)


def camera_with_distance_delta(camera, distance_delta_m):
    """Move a camera along its optical axis; positive values move it farther away."""
    result = dict(camera)
    rotation = np.asarray(camera["rotation_world_from_camera"], dtype=float)
    center = np.asarray(camera["center_world_m"], dtype=float)
    result["center_world_m"] = (
        center - float(distance_delta_m) * rotation[:, 2]
    ).tolist()
    return result


def camera_with_world_translation(camera, translation_world_m):
    """Translate only the camera without changing the fixed world frame."""
    result = dict(camera)
    center = np.asarray(camera["center_world_m"], dtype=float).copy()
    center += np.asarray(translation_world_m, dtype=float)
    result["center_world_m"] = center.tolist()
    return result


def camera_with_world_yaw_delta(camera, yaw_delta_deg):
    """Rotate the camera registration around the fixed tabletop normal."""
    delta = rotation_z(math.radians(float(yaw_delta_deg)))
    result = dict(camera)
    result["rotation_world_from_camera"] = (
        delta @ np.asarray(camera["rotation_world_from_camera"], dtype=float)
    ).tolist()
    result["center_world_m"] = (
        delta @ np.asarray(camera["center_world_m"], dtype=float)
    ).tolist()
    return result


def vggt_registration_from_camera(camera, extrinsic, metric_scale):
    """Recover the VGGT-to-world similarity implied by a refined camera."""
    camera_rotation = np.asarray(extrinsic, dtype=float)[:, :3]
    camera_center_vggt = -camera_rotation.T @ np.asarray(extrinsic, dtype=float)[:, 3]
    rotation = (
        np.asarray(camera["rotation_world_from_camera"], dtype=float)
        @ camera_rotation
    )
    center_world = np.asarray(camera["center_world_m"], dtype=float)
    translation = (
        center_world
        - camera_center_vggt @ rotation.T * float(metric_scale)
    )
    return rotation, translation


def apply_camera_registration_to_view(view):
    """Keep VGGT object clouds consistent with a refined front camera."""
    rotation, translation = vggt_registration_from_camera(
        view["camera"], view["extrinsic"], view["metric_scale"],
    )
    view["rotation_world_from_vggt"] = rotation
    view["translation_world"] = translation
    view["object_clouds"] = {
        object_id: apply_similarity(
            points, view["metric_scale"], rotation, translation,
        )
        for object_id, points in view["object_clouds_vggt"].items()
    }
    return view


def bottom_bbox_edge(mask):
    left, _, right, bottom = robust_bbox(np.asarray(mask) > 0)
    return np.array([(left + right) * 0.5, bottom], dtype=float), float(right - left + 1)


def front_silhouette_contour(mask):
    """Return the lower image-space envelope of a tabletop silhouette."""
    solid = solid_external_silhouette(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
    ) > 0
    rows = np.where(solid, np.arange(solid.shape[0])[:, None], -1).max(axis=0)
    valid = rows >= 0
    contour = np.zeros(solid.shape, dtype=np.uint8)
    contour[rows[valid], np.flatnonzero(valid)] = 255
    return contour


def front_silhouette_profile(mask, sample_count=161, trim_fraction=0.05):
    """Normalize the front contour so yaw cannot compensate for shift or scale."""
    rows, columns = np.nonzero(front_silhouette_contour(mask))
    if len(columns) < 2:
        return None
    order = np.argsort(columns)
    columns = columns[order].astype(float)
    rows = rows[order].astype(float)
    width = float(columns[-1] - columns[0])
    if width <= 0.0:
        return None
    positions = (columns - columns[0]) / width
    query = np.linspace(
        float(trim_fraction), 1.0 - float(trim_fraction), int(sample_count),
    )
    profile = np.interp(query, positions, rows)
    return (profile - np.median(profile)) / width


def front_silhouette_profile_loss(rendered, target):
    rendered_profile = front_silhouette_profile(rendered)
    target_profile = front_silhouette_profile(target)
    if rendered_profile is None or target_profile is None:
        return float("inf")
    return float(
        np.mean(np.abs(rendered_profile - target_profile))
        * max(np.asarray(target).shape)
    )


def bottom_bbox_edge_loss(rendered, target):
    if not np.any(rendered):
        return 1e6, 5e5, 5e5
    predicted_center, predicted_length = bottom_bbox_edge(rendered)
    target_center, target_length = bottom_bbox_edge(target)
    height, width = np.asarray(target).shape
    position_loss = float(np.sum(
        ((predicted_center - target_center) / np.array([width, height], dtype=float)) ** 2
    ))
    length_loss = float(((predicted_length - target_length) / target_length) ** 2)
    return position_loss + length_loss, position_loss, length_loss


def _camera_refinement_accepted(baseline_loss, candidate_loss, baseline_iou, candidate_iou):
    improvement = baseline_loss - candidate_loss
    return bool(
        improvement > 1e-4
        and candidate_iou + 0.005 >= baseline_iou
    )


def _pointcloud_refinement_not_worse(enabled, baseline_rmse, candidate_rmse):
    if not enabled:
        return True
    if baseline_rmse is None or candidate_rmse is None:
        return False
    tolerance_m = max(1e-4, 0.01 * float(baseline_rmse))
    return bool(candidate_rmse <= baseline_rmse + tolerance_m)


def refine_front_camera_pitch(
    camera, table_mesh_path, table_scale_xyz, labels, ids, limit_degrees=5.0,
    tabletop_plane_mask=None, table_yaw_deg=0.0,
):
    """Refine only front-camera pitch against the segmented table silhouette."""
    optimization_scale = 0.25
    search_camera = scaled_camera(camera, optimization_scale)
    target = cv2.resize(
        tabletop_refinement_target(labels, ids, tabletop_plane_mask),
        tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST,
    )
    mesh, tabletop_proxy_report = tabletop_alignment_mesh(
        search_mesh_z_up(Path(table_mesh_path), face_count=12000),
    )
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(mesh.vertices), table_yaw_deg,
    )
    vertices *= np.asarray(table_scale_xyz, dtype=float)

    def evaluate(delta_deg):
        candidate = camera_with_pitch_delta(search_camera, delta_deg)
        rendered = render_mesh_mask(mesh, vertices, candidate)
        loss = symmetric_contour_loss(rendered, target)
        rendered = rendered > 0
        expected = target > 0
        union = np.count_nonzero(rendered | expected)
        iou = float(np.count_nonzero(rendered & expected) / max(union, 1))
        return float(loss), iou

    baseline_loss, baseline_iou = evaluate(0.0)
    grid = np.linspace(-float(limit_degrees), float(limit_degrees), 41)
    measured = [(evaluate(value)[0], float(value)) for value in grid]
    _, coarse_delta = min(measured)
    step = float(grid[1] - grid[0])
    low = max(-float(limit_degrees), coarse_delta - step)
    high = min(float(limit_degrees), coarse_delta + step)
    candidates = [(evaluate(coarse_delta)[0], coarse_delta)]
    if high > low:
        refined = minimize_scalar(
            lambda value: evaluate(value)[0], bounds=(low, high), method="bounded",
            options={"xatol": 0.01, "maxiter": 24},
        )
        candidates.append((float(refined.fun), float(refined.x)))
    candidate_loss, candidate_delta = min(candidates)
    _, candidate_iou = evaluate(candidate_delta)
    accepted = _camera_refinement_accepted(
        baseline_loss, candidate_loss, baseline_iou, candidate_iou,
    )
    applied_delta = candidate_delta if accepted else 0.0
    return camera_with_pitch_delta(camera, applied_delta), {
        "method": "bounded_table_silhouette_pitch_refinement",
        "search_range_deg": [-float(limit_degrees), float(limit_degrees)],
        "pitch_delta_deg": float(applied_delta),
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "baseline_iou": baseline_iou,
        "candidate_iou": candidate_iou,
        "accepted": accepted,
        "tabletop_lower_edge_proxy": tabletop_proxy_report,
    }


def refine_front_camera_pitch_roll(
    camera, table_mesh_path, table_scale_xyz, labels, ids,
    pitch_limit_degrees=30.0, roll_limit_degrees=5.0,
    tabletop_plane_mask=None, table_yaw_deg=0.0,
):
    """Jointly refine front-camera pitch and roll against the table silhouette."""
    optimization_scale = 0.25
    search_camera = scaled_camera(camera, optimization_scale)
    target = cv2.resize(
        tabletop_refinement_target(labels, ids, tabletop_plane_mask),
        tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST,
    )
    mesh, tabletop_proxy_report = tabletop_alignment_mesh(
        search_mesh_z_up(Path(table_mesh_path), face_count=12000),
    )
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(mesh.vertices), table_yaw_deg,
    )
    vertices *= np.asarray(table_scale_xyz, dtype=float)

    def evaluate(values):
        pitch_deg, roll_deg = np.asarray(values, dtype=float)
        candidate = camera_with_pitch_roll_delta(search_camera, pitch_deg, roll_deg)
        rendered = render_mesh_mask(mesh, vertices, candidate)
        loss = symmetric_contour_loss(rendered, target)
        rendered = rendered > 0
        expected = target > 0
        union = np.count_nonzero(rendered | expected)
        iou = float(np.count_nonzero(rendered & expected) / max(union, 1))
        return float(loss), iou

    baseline_loss, baseline_iou = evaluate((0.0, 0.0))
    pitch_grid = np.linspace(-float(pitch_limit_degrees), float(pitch_limit_degrees), 5)
    roll_grid = np.linspace(-float(roll_limit_degrees), float(roll_limit_degrees), 5)
    measured = [
        (evaluate((pitch, roll))[0], float(pitch), float(roll))
        for pitch in pitch_grid for roll in roll_grid
    ]
    coarse_loss, coarse_pitch, coarse_roll = min(measured)
    refined = minimize(
        lambda values: evaluate(values)[0],
        x0=np.array([coarse_pitch, coarse_roll]),
        method="Powell",
        bounds=[
            (-float(pitch_limit_degrees), float(pitch_limit_degrees)),
            (-float(roll_limit_degrees), float(roll_limit_degrees)),
        ],
        options={"xtol": 0.01, "ftol": 1e-4, "maxiter": 24},
    )
    candidates = [
        (coarse_loss, coarse_pitch, coarse_roll),
        (float(refined.fun), float(refined.x[0]), float(refined.x[1])),
    ]
    candidate_loss, candidate_pitch, candidate_roll = min(candidates)
    _, candidate_iou = evaluate((candidate_pitch, candidate_roll))
    accepted = _camera_refinement_accepted(
        baseline_loss, candidate_loss, baseline_iou, candidate_iou,
    )
    applied_pitch = candidate_pitch if accepted else 0.0
    applied_roll = candidate_roll if accepted else 0.0
    updated_camera = camera_with_pitch_roll_delta(camera, applied_pitch, applied_roll)
    return updated_camera, {
        "method": "bounded_table_silhouette_pitch_roll_refinement",
        "pitch_search_range_deg": [-float(pitch_limit_degrees), float(pitch_limit_degrees)],
        "roll_search_range_deg": [-float(roll_limit_degrees), float(roll_limit_degrees)],
        "pitch_delta_deg": float(applied_pitch),
        "roll_delta_deg": float(applied_roll),
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "baseline_iou": baseline_iou,
        "candidate_iou": candidate_iou,
        "accepted": accepted,
        "tabletop_lower_edge_proxy": tabletop_proxy_report,
    }


def refine_front_camera_pitch_roll_distance(
    camera, table_mesh_path, table_scale_xyz, labels, ids,
    pitch_limit_degrees=30.0, roll_limit_degrees=5.0,
    translation_limit_fraction=0.25, bottom_edge_position_weight=1.0,
    bottom_edge_length_weight=0.5,
    translation_regularization_weight=0.01,
    tabletop_plane_mask=None, table_points_vggt=None,
    vggt_extrinsic=None, metric_scale=None, pointcloud_loss_weight=1.0,
    silhouette_loss_weight=0.0, table_yaw_deg=0.0,
):
    """Refine observable world yaw and translation without tilting the table plane."""
    import trimesh

    optimization_scale = 0.25
    search_camera = scaled_camera(camera, optimization_scale)
    target = cv2.resize(
        tabletop_refinement_target(labels, ids, tabletop_plane_mask),
        tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST,
    )
    full_mesh = search_mesh_z_up(Path(table_mesh_path), face_count=12000)
    mesh, tabletop_proxy_report = tabletop_alignment_mesh(full_mesh)
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(mesh.vertices), table_yaw_deg,
    )
    vertices *= np.asarray(table_scale_xyz, dtype=float)

    def yaw_contour_loss(yaw_deg):
        candidate = camera_with_world_yaw_delta(search_camera, yaw_deg)
        rendered = render_mesh_mask(mesh, vertices, candidate)
        return front_silhouette_profile_loss(rendered, target)

    yaw_limit = float(roll_limit_degrees)
    yaw_baseline_loss = yaw_contour_loss(0.0)
    yaw_grid = np.linspace(-yaw_limit, yaw_limit, 41) if yaw_limit > 0.0 else np.zeros(1)
    yaw_losses = np.array([yaw_contour_loss(value) for value in yaw_grid])
    yaw_index = int(np.argmin(yaw_losses))
    yaw_candidate = float(yaw_grid[yaw_index])
    yaw_candidate_loss = float(yaw_losses[yaw_index])
    yaw_observability_threshold = max(
        0.1, 0.02 * abs(yaw_baseline_loss),
    )
    yaw_observable = bool(
        yaw_baseline_loss - yaw_candidate_loss
        > yaw_observability_threshold
    )
    if yaw_observable and len(yaw_grid) > 1:
        step = float(yaw_grid[1] - yaw_grid[0])
        refined_yaw = minimize_scalar(
            yaw_contour_loss,
            bounds=(
                max(-yaw_limit, yaw_candidate - step),
                min(yaw_limit, yaw_candidate + step),
            ),
            method="bounded",
            options={"xatol": 0.01, "maxiter": 24},
        )
        if float(refined_yaw.fun) < yaw_candidate_loss:
            yaw_candidate = float(refined_yaw.x)
            yaw_candidate_loss = float(refined_yaw.fun)
    yaw_delta = yaw_candidate if yaw_observable else 0.0
    search_camera = camera_with_world_yaw_delta(search_camera, yaw_delta)

    pointcloud_inputs = (
        table_points_vggt is not None
        and vggt_extrinsic is not None
        and metric_scale is not None
        and float(pointcloud_loss_weight) > 0.0
    )
    if pointcloud_inputs:
        observed = np.asarray(table_points_vggt, dtype=float)
        if len(observed) > 5000:
            observed = observed[np.linspace(
                0, len(observed) - 1, 5000, dtype=int,
            )]
        tabletop_mesh = tabletop_surface_mesh(full_mesh)
        tabletop_vertices = orient_table_vertices(
            center_vertices_at_tabletop(tabletop_mesh.vertices), table_yaw_deg,
        )
        tabletop_vertices *= np.asarray(table_scale_xyz, dtype=float)
        metric_mesh = trimesh.Trimesh(
            vertices=tabletop_vertices, faces=tabletop_mesh.faces, process=False,
        )
        surface_points, _ = trimesh.sample.sample_surface(
            metric_mesh, 12000, seed=0,
        )
        front_tabletop_mask = resize_nearest(
            np.asarray(tabletop_plane_mask) > 0,
            tuple(search_camera["image_size"]),
        )
        pointcloud_normalizer = max(
            0.01 * float(np.ptp(surface_points, axis=0).max()), 0.005,
        )

        def pointcloud_loss(candidate_camera):
            pixels, depth = project_points(surface_points, candidate_camera)
            finite_pixels = np.isfinite(pixels).all(axis=1)
            rounded = np.zeros_like(pixels, dtype=np.int64)
            rounded[finite_pixels] = np.rint(
                pixels[finite_pixels],
            ).astype(np.int64)
            visible = (
                (depth > 0)
                & finite_pixels
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < front_tabletop_mask.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < front_tabletop_mask.shape[0])
            )
            visible_indices = np.flatnonzero(visible)
            visible_indices = visible_indices[
                front_tabletop_mask[
                    rounded[visible_indices, 1],
                    rounded[visible_indices, 0],
                ]
            ]
            if len(visible_indices) < 100:
                return 1e12, None
            surface_tree = cKDTree(surface_points[visible_indices])
            rotation, translation = vggt_registration_from_camera(
                candidate_camera, vggt_extrinsic, metric_scale,
            )
            world = apply_similarity(
                observed, metric_scale, rotation, translation,
            )
            distances = surface_tree.query(world, workers=-1)[0]
            rmse = float(np.sqrt(np.mean(distances ** 2)))
            return (rmse / pointcloud_normalizer) ** 2, rmse
    else:
        pointcloud_normalizer = None

        def pointcloud_loss(_candidate_camera):
            return 0.0, None

    center = np.asarray(search_camera["center_world_m"], dtype=float)
    forward = np.asarray(search_camera["rotation_world_from_camera"], dtype=float)[:, 2]
    initial_distance = abs(float(np.dot(-center, forward)))
    translation_limit = max(
        0.05,
        float(translation_limit_fraction) * initial_distance,
    )
    pixel_scale = float(max(search_camera["image_size"]))

    def evaluate(values):
        translation = np.asarray(values, dtype=float)
        candidate = search_camera
        candidate = camera_with_world_translation(candidate, translation)
        rendered = render_mesh_mask(
            mesh, vertices, candidate,
        )
        silhouette_loss = (
            float(symmetric_contour_loss(rendered, target))
            if float(silhouette_loss_weight) > 0.0 else 0.0
        )
        if not np.isfinite(silhouette_loss):
            silhouette_loss = 1e12
        _, edge_position_loss, edge_length_loss = bottom_bbox_edge_loss(
            rendered, target,
        )
        edge_loss = (
            float(bottom_edge_position_weight) * edge_position_loss
            + float(bottom_edge_length_weight) * edge_length_loss
        )
        translation_regularization = float(np.sum(
            (translation / translation_limit) ** 2,
        ))
        cloud_loss, cloud_rmse = pointcloud_loss(candidate)
        total_loss = (
            float(silhouette_loss_weight) * silhouette_loss
            + pixel_scale * edge_loss
            + float(translation_regularization_weight) * pixel_scale
            * translation_regularization
            + float(pointcloud_loss_weight) * cloud_loss
        )
        if not np.isfinite(total_loss):
            total_loss = 1e12
        iou = binary_mask_iou(rendered, target)
        return (
            float(total_loss), silhouette_loss, edge_loss,
            edge_position_loss, edge_length_loss, translation_regularization,
            cloud_loss, cloud_rmse, iou,
        )

    baseline = evaluate(np.zeros(3))
    refined = minimize(
        lambda values: evaluate(values)[0],
        x0=np.zeros(3),
        method="Powell",
        bounds=[(-translation_limit, translation_limit)] * 3,
        options={"xtol": 0.01, "ftol": 1e-4, "maxiter": 30},
    )
    candidate_translation = np.asarray(refined.x, dtype=float)
    candidate = evaluate(candidate_translation)
    silhouette_not_worse = bool(
        float(silhouette_loss_weight) <= 0.0
        or candidate[1] <= baseline[1] + max(0.1, 0.005 * baseline[1])
    )
    pointcloud_not_worse = _pointcloud_refinement_not_worse(
        pointcloud_inputs, baseline[7], candidate[7],
    )
    translation_accepted = bool(
        _camera_refinement_accepted(baseline[0], candidate[0], baseline[8], candidate[8])
        and silhouette_not_worse
        and pointcloud_not_worse
    )
    proposed_translation = candidate_translation.copy()
    if not translation_accepted:
        candidate_translation[:] = 0.0
    updated = camera_with_world_yaw_delta(camera, yaw_delta)
    updated = camera_with_world_translation(updated, candidate_translation)
    accepted = bool(yaw_observable or translation_accepted)
    return updated, {
        "method": (
            "normalized_table_front_contour_world_yaw_then_pointcloud_"
            "world_xyz_translation_refinement"
        ),
        "pitch_search_range_deg": [0.0, 0.0],
        "roll_search_range_deg": [0.0, 0.0],
        "requested_pitch_search_range_deg": [
            -float(pitch_limit_degrees), float(pitch_limit_degrees),
        ],
        "requested_roll_search_range_deg": [
            -float(roll_limit_degrees), float(roll_limit_degrees),
        ],
        "world_yaw_search_range_deg": [-yaw_limit, yaw_limit],
        "world_yaw_delta_deg": float(yaw_delta),
        "world_yaw_proposed_delta_deg": float(yaw_candidate),
        "world_yaw_baseline_contour_loss": yaw_baseline_loss,
        "world_yaw_candidate_contour_loss": yaw_candidate_loss,
        "world_yaw_observability_threshold": yaw_observability_threshold,
        "world_yaw_contour_scope": (
            "translation_and_scale_normalized_front_silhouette_profile"
        ),
        "world_yaw_observable": yaw_observable,
        "world_yaw_accepted": yaw_observable,
        "translation_xyz_search_range_m": [
            -float(translation_limit), float(translation_limit),
        ],
        "pitch_delta_deg": 0.0,
        "roll_delta_deg": 0.0,
        "translation_delta_world_m": candidate_translation.tolist(),
        "proposed_pitch_delta_deg": 0.0,
        "proposed_roll_delta_deg": 0.0,
        "proposed_translation_delta_world_m": proposed_translation.tolist(),
        "baseline_loss": baseline[0],
        "candidate_loss": candidate[0],
        "baseline_silhouette_loss": baseline[1],
        "candidate_silhouette_loss": candidate[1],
        "baseline_bottom_edge_loss": baseline[2],
        "candidate_bottom_edge_loss": candidate[2],
        "baseline_bottom_edge_position_loss": baseline[3],
        "candidate_bottom_edge_position_loss": candidate[3],
        "baseline_bottom_edge_length_loss": baseline[4],
        "candidate_bottom_edge_length_loss": candidate[4],
        "baseline_pointcloud_loss": baseline[6],
        "candidate_pointcloud_loss": candidate[6],
        "baseline_pointcloud_rmse_m": baseline[7],
        "candidate_pointcloud_rmse_m": candidate[7],
        "baseline_iou": baseline[8],
        "candidate_iou": candidate[8],
        "bottom_edge_position_weight": float(bottom_edge_position_weight),
        "bottom_edge_length_weight": float(bottom_edge_length_weight),
        "camera_alignment_source": (
            "registered_table_plane_with_normalized_front_contour_world_yaw_"
            "and_rendered_tabletop_bbox_lower_edge_translation"
        ),
        "camera_rotation_locked": not yaw_observable,
        "camera_pitch_roll_locked": True,
        "camera_rotation_source": (
            "vggt_table_plane_with_normalized_front_contour_world_yaw"
        ),
        "silhouette_loss_weight": float(silhouette_loss_weight),
        "translation_regularization_weight": float(
            translation_regularization_weight,
        ),
        "pointcloud_loss_weight": float(pointcloud_loss_weight),
        "pointcloud_normalizer_m": pointcloud_normalizer,
        "pointcloud_enabled": bool(pointcloud_inputs),
        "pointcloud_source": "valid_vggt_points_inside_front_tabletop_mask",
        "pointcloud_target": "tabletop_slab_between_dominant_top_and_lower_planes",
        "pointcloud_visibility": "model_samples_projected_inside_front_tabletop_mask",
        "pointcloud_parameter_scope": "world_xyz_translation",
        "tabletop_lower_edge_proxy": tabletop_proxy_report,
        "world_origin_unchanged": True,
        "table_transform_persisted": False,
        "silhouette_not_worse": silhouette_not_worse,
        "pointcloud_not_worse": pointcloud_not_worse,
        "translation_accepted": translation_accepted,
        "accepted": accepted,
        "fallback": "unchanged_camera" if not accepted else None,
    }


def _table_edge_chamfer(first, second):
    if first is None or second is None or not np.any(first) or not np.any(second):
        return 1e12
    first_distance = cv2.distanceTransform(
        (~np.asarray(first, dtype=bool)).astype(np.uint8), cv2.DIST_L2, 3,
    )
    second_distance = cv2.distanceTransform(
        (~np.asarray(second, dtype=bool)).astype(np.uint8), cv2.DIST_L2, 3,
    )
    return float(second_distance[first].mean() + first_distance[second].mean())


def table_cardinal_yaw_candidates(
    native_xy, world_xy,
    square_ratio_threshold=FRONT_FOUR_WAY_SQUARE_RATIO_THRESHOLD,
):
    native_xy = np.asarray(native_xy, dtype=float)
    world_xy = np.asarray(world_xy, dtype=float)
    native_square_ratio = float(native_xy.min() / native_xy.max())
    world_square_ratio = float(world_xy.min() / world_xy.max())
    has_reliable_long_axis = bool(
        native_square_ratio < square_ratio_threshold
        and world_square_ratio < square_ratio_threshold
    )
    if has_reliable_long_axis:
        base_yaw = (
            0.0 if int(np.argmax(native_xy)) == int(np.argmax(world_xy))
            else 90.0
        )
        candidates = (base_yaw, (base_yaw + 180.0) % 360.0)
    else:
        base_yaw = 0.0
        candidates = (0.0, 90.0, 180.0, 270.0)
    return candidates, {
        "method": "long_axis_family_then_opposite_or_four_way",
        "square_ratio_threshold": float(square_ratio_threshold),
        "native_xy_extent": native_xy.tolist(),
        "world_xy_extent": world_xy.tolist(),
        "native_square_ratio": native_square_ratio,
        "world_square_ratio": world_square_ratio,
        "has_reliable_long_axis": has_reliable_long_axis,
        "base_yaw_deg": base_yaw,
        "candidate_yaws_deg": list(candidates),
    }


def select_table_yaw(
    table_mesh_path, front_camera, front_rgb, front_labels, front_ids,
    front_tabletop_mask, aligned_scale_xyz, table_scale_pivot,
    aligned_yaw_deg=0.0,
):
    """Choose yaw after camera/size alignment without refitting each candidate."""
    mesh = search_mesh_z_up(Path(table_mesh_path), face_count=12000)
    centered = center_vertices_at_tabletop(mesh.vertices)
    raw_target = front_labels == front_ids["table_0"]
    filled_tabletop = solid_external_silhouette(
        np.where(front_tabletop_mask, 255, 0).astype(np.uint8),
    ) > 0
    target_mask = raw_target | filled_tabletop
    tabletop_bottom = robust_bbox(filled_tabletop)[3]
    mask_split_row = min(target_mask.shape[0], tabletop_bottom + 1)
    edge_split_row = max(0, tabletop_bottom - 3)
    target_lower = np.zeros_like(target_mask)
    target_lower[mask_split_row:] = target_mask[mask_split_row:]
    target_edge_region = np.zeros_like(target_mask)
    target_edge_region[edge_split_row:] = target_mask[edge_split_row:]
    target_edges, _ = masked_edges(front_rgb, target_edge_region, method="canny")
    target_edges = np.asarray(target_edges) > 0
    aligned_table = {
        "scale_xyz": list(map(float, aligned_scale_xyz)),
        "scale_pivot": table_scale_pivot,
        "yaw_deg": float(aligned_yaw_deg),
    }
    fixed_dimensions = np.ptp(
        transform_table_vertices(mesh.vertices, aligned_table), axis=0,
    )
    candidate_yaws, axis_family_report = table_cardinal_yaw_candidates(
        np.ptp(centered, axis=0)[:2], fixed_dimensions[:2],
    )

    candidates = []
    for yaw_deg in candidate_yaws:
        oriented = orient_table_vertices(centered, yaw_deg)
        scale = np.asarray(aligned_scale_xyz, dtype=float).copy()
        scale[:2] = fixed_dimensions[:2] / np.ptp(oriented, axis=0)[:2]
        candidate_table = {
            "scale_xyz": scale.tolist(),
            "scale_pivot": table_scale_pivot,
            "yaw_deg": yaw_deg,
        }
        vertices = transform_table_vertices(mesh.vertices, candidate_table)
        rendered_mask = render_mesh_mask(
            mesh, vertices, front_camera, fill_silhouette=False,
        ) > 0
        rendered_edges = render_mesh_edges(
            mesh, vertices, front_camera, silhouette=rendered_mask,
            visible_only=True,
        )
        candidate_lower = np.zeros_like(rendered_mask)
        candidate_lower[mask_split_row:] = rendered_mask[mask_split_row:]
        candidate_edges = np.zeros_like(rendered_edges)
        candidate_edges[edge_split_row:] = rendered_edges[edge_split_row:]
        union = np.count_nonzero(candidate_lower | target_lower)
        intersection = np.count_nonzero(candidate_lower & target_lower)
        mask_iou = float(intersection / union) if union else 0.0
        mask_loss = 1.0 - mask_iou
        edge_loss_px = _table_edge_chamfer(candidate_edges, target_edges)
        candidates.append({
            "yaw_deg": yaw_deg,
            "scale_xyz": scale.tolist(),
            "world_dimensions_m": np.ptp(vertices, axis=0).tolist(),
            "mask_iou": mask_iou,
            "mask_loss": mask_loss,
            "edge_loss_px": edge_loss_px,
        })
    mask_normalizer = max(
        np.finfo(float).eps,
        float(np.mean([item["mask_loss"] for item in candidates])),
    )
    edge_normalizer = max(
        np.finfo(float).eps,
        float(np.mean([item["edge_loss_px"] for item in candidates])),
    )
    for candidate in candidates:
        candidate["normalized_mask_loss"] = (
            candidate["mask_loss"] / mask_normalizer
        )
        candidate["edge_loss"] = candidate["edge_loss_px"] / edge_normalizer
        candidate["loss"] = (
            candidate["normalized_mask_loss"] + candidate["edge_loss"]
        )
    selected = min(candidates, key=lambda item: (item["loss"], item["yaw_deg"]))
    return float(selected["yaw_deg"]), list(selected["scale_xyz"]), {
        "method": "post_alignment_axis_gated_mask_and_lower_structure_edge",
        "candidate_yaws_deg": list(candidate_yaws),
        "axis_family_selection": axis_family_report,
        "aligned_yaw_deg": float(aligned_yaw_deg),
        "scale_fit": "fixed_aligned_world_dimensions",
        "fixed_world_dimensions_m": fixed_dimensions.tolist(),
        "reference_tabletop_filled_pixels": int(np.count_nonzero(
            filled_tabletop & ~raw_target,
        )),
        "reference_lower_frame_holes_preserved": True,
        "mask_scope": "strictly_below_tabletop",
        "loss_formula": (
            "mean_normalized_(1 - below_tabletop_mask_iou) + "
            "mean_normalized_lower_edge_chamfer"
        ),
        "mask_loss_normalizer": mask_normalizer,
        "edge_loss_normalizer_px": float(edge_normalizer),
        "selected_yaw_deg": float(selected["yaw_deg"]),
        "candidates": candidates,
    }


def table_alignment(
    table_mesh_path, top_geometry, top_camera, front_camera, front_labels,
    front_ids, table_yaw_deg=0.0,
):
    mesh = search_mesh_z_up(table_mesh_path)
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(mesh.vertices), table_yaw_deg,
    )
    native = np.ptp(vertices, axis=0)
    bbox = top_geometry["bbox_xyxy"]
    depth_m = (bbox[3] - bbox[1] + 1) * top_camera["meters_per_pixel"]
    scale_xy = np.array([TABLE_SIZE_M[0] / native[0], depth_m / native[1]])
    target_bbox = robust_bbox(target_mask(front_labels, front_ids, "table_0"))

    def projected_vertical_bounds(scale_z):
        points = vertices * np.array([scale_xy[0], scale_xy[1], scale_z])
        pixels, depth = project_points(points, front_camera)
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        return float(pixels[valid, 1].min()), float(pixels[valid, 1].max())

    def objective(scale_z):
        top, bottom = projected_vertical_bounds(scale_z)
        return (top - target_bbox[1]) ** 2 + (bottom - target_bbox[3]) ** 2

    result = minimize_scalar(objective, bounds=(0.25, 3.0), method="bounded")
    return [float(scale_xy[0]), float(scale_xy[1]), float(result.x)], float(math.sqrt(result.fun))


def refine_table_depth_from_upper_edge(
    table_mesh_path, table_scale, top_geometry,
    front_camera, front_labels, front_ids,
    top_camera, top_labels, top_ids, square_ratio_threshold=0.9,
    front_tabletop_mask=None, table_yaw_deg=0.0,
):
    """Fit Y from anchored front edges, then fit X at the final depth."""
    left, top, right, bottom = top_geometry["bbox_xyxy"]
    width, height = right - left + 1, bottom - top + 1
    square_ratio = min(width, height) / max(width, height)
    report = {
        "top_mask_bbox_xyxy": [left, top, right, bottom],
        "top_mask_bbox_square_ratio": float(square_ratio),
        "square_ratio_threshold": float(square_ratio_threshold),
        "optimized_parameters": ["table_scale_x", "table_scale_y"],
        "world_origin_unchanged": True,
    }

    full_mesh = search_mesh_z_up(Path(table_mesh_path), face_count=None)
    planes = tabletop_plane_geometry(full_mesh)
    mesh = tabletop_surface_mesh(full_mesh, planes)
    centered_vertices = center_vertices_at_tabletop(mesh.vertices)
    vertices = orient_table_vertices(centered_vertices, table_yaw_deg)
    front_target_bbox = robust_bbox(
        np.asarray(front_tabletop_mask) > 0
        if front_tabletop_mask is not None else
        target_mask(front_labels, front_ids, "table_0")
    )
    target_upper = float(front_target_bbox[1])
    target_lower = float(front_target_bbox[3])
    target_width = float(front_target_bbox[2] - front_target_bbox[0] + 1)
    target_lower_center_x = 0.5 * (
        float(front_target_bbox[0]) + float(front_target_bbox[2])
    )
    base_scale = np.asarray(table_scale, dtype=float)
    model_top_mask = render_mesh_mask(
        mesh, vertices * base_scale, top_camera,
    )
    upper_vertices = mask_boundary_on_world_plane(
        model_top_mask, top_camera,
    ) / base_scale
    tabletop_thickness, tabletop_thickness_report = tabletop_depth_thickness(
        full_mesh, planes,
    )
    lower_vertices = upper_vertices.copy()
    lower_vertices[:, 2] -= tabletop_thickness
    local_y_min = float(upper_vertices[:, 1].min())
    local_y_max = float(upper_vertices[:, 1].max())

    def posed_vertices(scale_x, scale_y, offset_y):
        posed = vertices * np.array([
            float(scale_x), float(scale_y), base_scale[2],
        ])
        posed[:, 1] += float(offset_y)
        return posed

    def rendered_bbox(scale_x, scale_y, offset_y, camera):
        posed = posed_vertices(scale_x, scale_y, offset_y)
        rendered = render_mesh_mask(mesh, posed, camera)
        return robust_bbox(rendered > 0) if np.any(rendered) else None

    def projected_vertical_bounds(scale_y, offset_y):
        scale = np.array([base_scale[0], float(scale_y), base_scale[2]])
        upper = upper_vertices * scale
        lower = lower_vertices * scale
        upper[:, 1] += float(offset_y)
        lower[:, 1] += float(offset_y)
        upper_pixels, upper_depth = project_points(upper, front_camera)
        lower_pixels, lower_depth = project_points(lower, front_camera)
        upper_valid = (upper_depth > 0) & np.isfinite(upper_pixels).all(axis=1)
        lower_valid = (lower_depth > 0) & np.isfinite(lower_pixels).all(axis=1)
        if not np.any(upper_valid) or not np.any(lower_valid):
            return None
        return (
            float(upper_pixels[upper_valid, 1].min()),
            float(lower_pixels[lower_valid, 1].max()),
        )

    def projected_edge_mean_y(local_y):
        tolerance = max(1e-8, 1e-6 * (local_y_max - local_y_min))
        edge_vertices = upper_vertices[
            np.abs(upper_vertices[:, 1] - float(local_y)) <= tolerance
        ]
        pixels, depth = project_points(
            edge_vertices * base_scale, front_camera,
        )
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        if not np.any(valid):
            raise RuntimeError("tabletop Y edge is behind the front camera")
        return float(pixels[valid, 1].mean())

    y_edges = (local_y_min, local_y_max)
    lower_anchor_y = max(y_edges, key=projected_edge_mean_y)
    upper_anchor_y = min(y_edges, key=projected_edge_mean_y)

    def lower_anchored_offset(scale_y):
        return lower_anchor_y * (base_scale[1] - float(scale_y))

    def upper_edge_objective(scale_y):
        bounds = projected_vertical_bounds(
            float(scale_y), lower_anchored_offset(scale_y),
        )
        if bounds is None:
            return 1e12
        return ((bounds[0] - target_upper) / max(target_lower - target_upper, 1.0)) ** 2

    upper_baseline_loss = upper_edge_objective(base_scale[1])
    upper_result = minimize_scalar(
        upper_edge_objective,
        bounds=(0.5 * base_scale[1], 1.5 * base_scale[1]),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 40},
    )
    upper_accepted = bool(
        np.isfinite(upper_result.fun)
        and upper_result.fun + 1e-8 < upper_baseline_loss
    )
    upper_scale_y = (
        float(upper_result.x) if upper_accepted else float(base_scale[1])
    )
    upper_offset_y = lower_anchored_offset(upper_scale_y)

    def upper_anchored_offset(scale_y):
        return (
            upper_offset_y
            + upper_anchor_y * (upper_scale_y - float(scale_y))
        )

    def lower_edge_objective(scale_y):
        bounds = projected_vertical_bounds(
            float(scale_y), upper_anchored_offset(scale_y),
        )
        if bounds is None:
            return 1e12
        return ((bounds[1] - target_lower) / max(target_lower - target_upper, 1.0)) ** 2

    lower_baseline_loss = lower_edge_objective(upper_scale_y)
    lower_result = minimize_scalar(
        lower_edge_objective,
        bounds=(0.5 * upper_scale_y, 1.5 * upper_scale_y),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 40},
    )
    lower_accepted = bool(
        np.isfinite(lower_result.fun)
        and lower_result.fun + 1e-8 < lower_baseline_loss
    )
    final_scale = base_scale.copy()
    final_scale[1] = (
        float(lower_result.x) if lower_accepted else upper_scale_y
    )
    final_offset_y = upper_anchored_offset(final_scale[1])

    target_left = float(front_target_bbox[0])
    target_right = float(front_target_bbox[2])

    def width_objective(scale_x):
        # Keep off-frame geometry in the loss; rendered masks clip it away.
        pixels, depth = project_points(
            posed_vertices(
                float(scale_x), final_scale[1], final_offset_y,
            ),
            front_camera,
        )
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        if not np.any(valid):
            return 1e12
        left = float(pixels[valid, 0].min())
        right = float(pixels[valid, 0].max())
        return (
            ((left - target_left) / target_width) ** 2
            + ((right - target_right) / target_width) ** 2
        )

    width_baseline_loss = width_objective(base_scale[0])
    width_result = minimize_scalar(
        width_objective,
        bounds=(0.5 * base_scale[0], 1.5 * base_scale[0]),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 40},
    )
    width_accepted = bool(
        np.isfinite(width_result.fun)
        and width_result.fun + 1e-8 < width_baseline_loss
    )
    if width_accepted:
        final_scale[0] = float(width_result.x)

    final_upper, final_lower = projected_vertical_bounds(
        final_scale[1], final_offset_y,
    )
    final_bbox = rendered_bbox(
        final_scale[0], final_scale[1], final_offset_y, top_camera,
    )
    top_target_bbox = robust_bbox(
        target_mask(top_labels, top_ids, "table_0"),
    )
    target_top_center_y = 0.5 * (
        float(top_target_bbox[1]) + float(top_target_bbox[3])
    )
    pre_alignment_top_center_y = 0.5 * (
        float(final_bbox[1]) + float(final_bbox[3])
    )
    top_meters_per_pixel = float(top_camera.get(
        "meters_per_pixel",
        1.0 / float(np.asarray(top_camera["intrinsic"])[1, 1]),
    ))
    top_camera_center_y_delta = (
        target_top_center_y - pre_alignment_top_center_y
    ) * top_meters_per_pixel
    aligned_top_camera = dict(top_camera)
    aligned_top_camera["center_world_m"] = list(top_camera["center_world_m"])
    aligned_top_camera["center_world_m"][1] += top_camera_center_y_delta
    aligned_top_bbox = rendered_bbox(
        final_scale[0], final_scale[1], final_offset_y, aligned_top_camera,
    )
    final_front_bbox = rendered_bbox(
        final_scale[0], final_scale[1], final_offset_y, front_camera,
    )
    initial_front_bbox = rendered_bbox(
        base_scale[0], base_scale[1], 0.0, front_camera,
    )
    table_scale_pivot = {
        "axis": "y",
        "offset_m": float(final_offset_y),
    }
    return final_scale.tolist(), table_scale_pivot, aligned_top_camera, {
        **report,
        "accepted": bool(upper_accepted or lower_accepted or width_accepted),
        "optimization_view": "front",
        "vertical_alignment_mode": (
            "3d_upper_surface_edges_with_translated_lower_front_edge"
        ),
        "tabletop_minimum_thickness_m": TABLETOP_MIN_THICKNESS_M,
        "tabletop_thickness_local_m": tabletop_thickness,
        "tabletop_thickness_world_m": tabletop_thickness * float(base_scale[2]),
        **tabletop_thickness_report,
        "model_lower_front_edge_source": (
            "upper_surface_front_boundary_translated_in_3d_along_negative_z"
        ),
        "upper_surface_source": (
            "top_camera_rendered_model_tabletop_mask_boundary_back_projected_to_3d"
        ),
        "model_top_mask_pixel_count": int(np.count_nonzero(model_top_mask)),
        "upper_surface_boundary_point_count": int(len(upper_vertices)),
        "width_pivot": "front_tabletop_mask_bbox_lower_edge_center",
        "width_pivot_local_x_m": 0.0,
        "lower_edge_anchor_local_y_m": float(lower_anchor_y),
        "upper_edge_anchor_local_y_m": float(upper_anchor_y),
        "table_scale_pivot": table_scale_pivot,
        "final_front_rendered_bbox_xyxy": list(final_front_bbox),
        "target_front_mask_upper_edge_px": target_upper,
        "target_front_mask_lower_edge_px": target_lower,
        "target_front_mask_width_px": target_width,
        "target_front_mask_lower_edge_center_x_px": target_lower_center_x,
        "initial_front_rendered_lower_edge_center_x_px": 0.5 * (
            float(initial_front_bbox[0]) + float(initial_front_bbox[2])
        ),
        "final_front_rendered_lower_edge_center_x_px": 0.5 * (
            float(final_front_bbox[0]) + float(final_front_bbox[2])
        ),
        "target_top_bbox_center_y_px": target_top_center_y,
        "pre_alignment_top_bbox_center_y_px": pre_alignment_top_center_y,
        "final_rendered_center_y_px": 0.5 * (
            float(aligned_top_bbox[1]) + float(aligned_top_bbox[3])
        ),
        "top_camera_center_y_delta_m": float(top_camera_center_y_delta),
        "initial_scale_y": float(base_scale[1]),
        "lower_anchor_upper_edge_scale_y": float(upper_scale_y),
        "final_scale_y": float(final_scale[1]),
        "final_table_translation_world_m": [0.0, float(final_offset_y), 0.0],
        "world_origin_unchanged": True,
        "final_front_rendered_upper_edge_px": final_upper,
        "final_front_rendered_lower_edge_px": final_lower,
        "upper_edge_stage_accepted": upper_accepted,
        "upper_edge_stage_initial_loss": float(upper_baseline_loss),
        "upper_edge_stage_final_loss": float(
            upper_edge_objective(upper_scale_y)
        ),
        "lower_edge_stage_accepted": lower_accepted,
        "lower_edge_stage_initial_loss": float(lower_baseline_loss),
        "lower_edge_stage_final_loss": float(
            lower_edge_objective(final_scale[1])
        ),
        "width_accepted": width_accepted,
        "initial_scale_x": float(base_scale[0]),
        "final_scale_x": float(final_scale[0]),
        "initial_width_loss": float(width_baseline_loss),
        "final_width_loss": float(width_objective(final_scale[0])),
        "initial_loss": float(upper_baseline_loss + lower_baseline_loss),
        "final_loss": float(
            upper_edge_objective(upper_scale_y)
            + lower_edge_objective(final_scale[1])
        ),
    }


def refine_table_height_from_bottom_edge(
    table_mesh_path, table_scale, table_scale_pivot,
    front_camera, front_labels, front_ids, table_yaw_deg=0.0,
):
    """Optimize only the table legs in Z while keeping the tabletop fixed."""
    mesh = search_mesh_z_up(Path(table_mesh_path), face_count=12000)
    vertices = orient_table_vertices(
        center_vertices_at_tabletop(mesh.vertices), table_yaw_deg,
    )
    planes = source_tabletop_plane_geometry(mesh)
    leg_anchor_local_z = float(planes["bottom_z"] - planes["top_z"])
    base_scale = np.asarray(table_scale, dtype=float)
    pivot_offset_y = table_scale_pivot_offset({
        "scale_xyz": base_scale,
        "scale_pivot": table_scale_pivot,
    })[1]
    target_bottom = float(robust_bbox(
        target_mask(front_labels, front_ids, "table_0"),
    )[3])

    def projected_bottom(leg_scale_z):
        posed = vertices * base_scale
        below = vertices[:, 2] < leg_anchor_local_z
        posed[below, 2] = (
            leg_anchor_local_z * base_scale[2]
            + (vertices[below, 2] - leg_anchor_local_z) * float(leg_scale_z)
        )
        posed[:, 1] += pivot_offset_y
        pixels, depth = project_points(posed, front_camera)
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        return float(pixels[valid, 1].max())

    def objective(scale_z):
        return (projected_bottom(scale_z) - target_bottom) ** 2

    baseline_loss = objective(base_scale[2])
    result = minimize_scalar(
        objective,
        bounds=(0.5 * base_scale[2], 1.5 * base_scale[2]),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 40},
    )
    accepted = bool(np.isfinite(result.fun) and result.fun + 1e-8 < baseline_loss)
    final_leg_scale_z = float(result.x) if accepted else float(base_scale[2])
    final_pivot = dict(table_scale_pivot or {})
    final_pivot.update({
        "leg_anchor_local_z": leg_anchor_local_z,
        "leg_scale_z": final_leg_scale_z,
    })
    return base_scale.tolist(), final_pivot, {
        "method": "final_table_leg_length_from_front_bottom_edge",
        "accepted": accepted,
        "fixed_geometry": "tabletop_slab",
        "optimized_geometry": "vertices_below_tabletop_bottom",
        "target_bottom_edge_px": target_bottom,
        "initial_bottom_edge_px": projected_bottom(base_scale[2]),
        "final_bottom_edge_px": projected_bottom(final_leg_scale_z),
        "initial_scale_z": float(base_scale[2]),
        "final_scale_z": float(base_scale[2]),
        "leg_anchor_local_z": leg_anchor_local_z,
        "initial_leg_scale_z": float(base_scale[2]),
        "final_leg_scale_z": final_leg_scale_z,
        "initial_loss": float(baseline_loss),
        "final_loss": float(objective(final_leg_scale_z)),
    }


def coarse_object_pose_from_extents(native, top_geometry, top_camera):
    bbox = top_geometry["bbox_xyxy"]
    center = top_geometry["center_px"]
    k = np.asarray(top_camera["intrinsic"])
    camera_center = np.asarray(top_camera["center_world_m"])
    meters_per_pixel = top_camera["meters_per_pixel"]
    x = camera_center[0] + (center[0] - k[0, 2]) * meters_per_pixel
    y = camera_center[1] - (center[1] - k[1, 2]) * meters_per_pixel
    xy = np.array([x, y])
    target_xy = np.array([
        (bbox[2] - bbox[0] + 1) * meters_per_pixel,
        (bbox[3] - bbox[1] + 1) * meters_per_pixel,
    ])
    native = np.asarray(native, dtype=float)
    candidates = []
    for quarter_turns in (0, 1):
        native_xy = native[[1, 0]] if quarter_turns else native[:2]
        scale = uniform_scale(native_xy, target_xy)
        aspect_error = float(np.std(np.log(target_xy / native_xy)))
        candidates.append((aspect_error, quarter_turns, scale, native_xy))
    _, quarter_turns, scale, native_xy = min(candidates)
    axis = np.asarray(top_geometry["major_axis_image"])
    target_angle = math.degrees(math.atan2(-axis[1], axis[0]))
    native_major_angle = 0.0 if native_xy[0] >= native_xy[1] else 90.0
    yaw = target_angle - native_major_angle + 90.0 * quarter_turns
    yaw = float(((yaw + 180.0) % 360.0) - 180.0)
    return {
        "translation_world_m": [float(xy[0]), float(xy[1]), 0.0],
        "uniform_scale": scale,
        "yaw_deg": yaw,
        "axis_mapping_quarter_turns": quarter_turns,
    }


def coarse_object_pose(item, mesh_path, top_geometry, top_camera):
    return coarse_object_pose_from_extents(
        mesh_extents_z_up(mesh_path), top_geometry, top_camera,
    )


def align_local_footprint_axes(
    vertices, min_aspect_ratio=NEAR_SQUARE_FOOTPRINT_ASPECT_RATIO,
    axis_tolerance_deg=1.0,
):
    """Rotate a rigid mesh so its top-view OBB follows local X/Y axes.

    The top-view scale refinement is diagonal in mesh-local X/Y.  An elongated
    footprint that is diagonal in that frame (for example the fan) therefore
    shears visually under non-uniform scale.  Canonicalize only clearly
    elongated footprints and keep the nearest axis, avoiding needless 90°
    rotations for assets which are already valid.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return np.eye(3), {
            "applied": False,
            "method": "local_xy_minimum_area_bbox_axis_alignment",
            "reason": "insufficient_xy_vertices",
        }
    points = points[:, :2]
    if not np.isfinite(points).all():
        return np.eye(3), {
            "applied": False,
            "method": "local_xy_minimum_area_bbox_axis_alignment",
            "reason": "nonfinite_xy_vertices",
        }

    _, (width, height), angle_deg = cv2.minAreaRect(points.astype(np.float32))
    long_extent, short_extent = sorted((float(width), float(height)), reverse=True)
    if long_extent <= 1e-12:
        return np.eye(3), {
            "applied": False,
            "method": "local_xy_minimum_area_bbox_axis_alignment",
            "reason": "degenerate_xy_footprint",
        }
    if height > width:
        angle_deg += 90.0
    long_axis_deg = float((angle_deg + 90.0) % 180.0 - 90.0)
    aspect_ratio = long_extent / max(short_extent, 1e-12)
    nearest_axis_deg = 90.0 * round(long_axis_deg / 90.0)
    correction_deg = float(nearest_axis_deg - long_axis_deg)
    report = {
        "method": "local_xy_minimum_area_bbox_axis_alignment",
        "local_long_axis_deg": long_axis_deg,
        "long_extent": long_extent,
        "short_extent": short_extent,
        "aspect_ratio": aspect_ratio,
        "nearest_local_axis_deg": float(nearest_axis_deg),
        "correction_deg": correction_deg,
        "yaw_compensation_deg": -correction_deg,
        "min_aspect_ratio": float(min_aspect_ratio),
        "axis_tolerance_deg": float(axis_tolerance_deg),
    }
    if aspect_ratio < min_aspect_ratio:
        return np.eye(3), {
            **report,
            "applied": False,
            "reason": "near_square_footprint",
        }
    if abs(correction_deg) <= axis_tolerance_deg:
        return np.eye(3), {
            **report,
            "applied": False,
            "reason": "already_axis_aligned",
        }

    radians = math.radians(correction_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    matrix = np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return matrix, {**report, "applied": True, "reason": "diagonal_footprint"}


def infer_mesh_orientation(
    mesh_path, top_geometry, top_camera, front_geometry, front_camera,
    scorer, reference_rgbs, reference_key=None,
):
    """Gate obvious errors, then rank orientations with the direction scorer."""
    triggered, gate = obvious_up_axis_error(
        top_geometry, mesh_extents_z_up(mesh_path),
    )
    if not triggered:
        return "+Z", np.eye(3), {
            **gate,
            "selected_up_axis": "+Z",
            "orientation_changed": False,
            "decision": "identity_footprint_plausible_keep_identity",
            "shape_candidates": [],
        }

    mesh = search_mesh_z_up(mesh_path, face_count=3000)
    raw_vertices = np.asarray(mesh.vertices, dtype=float)
    targets = {
        "front": front_geometry["mask"],
        "top": top_geometry["mask"],
    }
    cameras = (
        {"front": front_camera, "top": top_camera}
        if getattr(scorer, "requires_full_resolution", False) else
        {
            "front": minima_direction_camera(front_camera),
            "top": minima_direction_camera(top_camera),
        }
    )
    rendered_candidates = {"front": {}, "top": {}}
    candidate_specs = []
    needs_coordinates = getattr(scorer, "requires_object_coordinates", False)
    try:
        vertex_rgb = mesh_vertex_texture_colors(mesh_path, raw_vertices)
    except Exception:
        vertex_rgb = None
    for label, matrix in signed_up_axis_candidates():
        oriented = raw_vertices @ matrix.T
        pose = coarse_object_pose_from_extents(
            np.ptp(oriented, axis=0), top_geometry, top_camera,
        )
        centered = center_vertices_on_table(oriented)
        yaw_candidates = (0, 90, 180, 270)
        for yaw_deg in yaw_candidates:
            key = (label, yaw_deg)
            local = pose_vertices(
                centered, pose["uniform_scale"], (0.0, 0.0), 0.0,
            )
            vertices = pose_vertices(
                local, 1.0, pose["translation_world_m"][:2], yaw_deg,
            )
            for view_name, camera in cameras.items():
                rendered = render_normal_edge_buffers_nvdiffrast(
                    vertices, mesh.faces, camera, vertex_rgb=vertex_rgb,
                    return_normal_map=True,
                    vertex_coordinates=local if needs_coordinates else None,
                )
                rendered_candidate = {
                    "mask": rendered[0],
                    "rgb": rendered[2] if vertex_rgb is not None else rendered[3],
                    "normal_map": rendered[3],
                    "candidate_modality": (
                        "rgb" if vertex_rgb is not None else "normal_fallback"
                    ),
                }
                if needs_coordinates:
                    rendered_candidate["object_coordinates"] = rendered[4]
                rendered_candidates[view_name][key] = rendered_candidate
            candidate_specs.append({
                "up_axis": label,
                "yaw_deg": yaw_deg,
                "matrix": matrix,
            })
    scores = {
        view_name: score_direction_candidates(
            scorer, view_name, reference_rgbs[view_name], targets[view_name],
            cameras[view_name], rendered_candidates[view_name],
        )
        for view_name in ("front", "top")
    }
    front_loss_weight, top_loss_weight = 1.0, 0.25
    candidates = []
    for candidate in candidate_specs:
        key = (candidate["up_axis"], candidate["yaw_deg"])
        front_score, top_score = scores["front"][key], scores["top"][key]
        loss = (
            front_loss_weight * front_score["loss"]
            + top_loss_weight * top_score["loss"]
        )
        candidates.append({
            **candidate,
            "front_loss": front_score["loss"],
            "top_loss": top_score["loss"],
            "loss": loss,
            "front_foreground_mean_confidence": front_score.get(
                "foreground_mean_confidence", 1.0 - front_score["loss"],
            ),
            "top_foreground_mean_confidence": top_score.get(
                "foreground_mean_confidence", 1.0 - top_score["loss"],
            ),
        })
    selected, identity_view_losses = select_mesh_orientation_candidate(candidates)
    orientation_changed = selected["up_axis"] != "+Z"
    return selected["up_axis"], selected["matrix"], {
        **gate,
        "method": (
            f"gated_{getattr(scorer, 'backend', 'direction_loss')}_front_top"
        ),
        "front_loss_weight": front_loss_weight,
        "top_loss_weight": top_loss_weight,
        "selected_up_axis": selected["up_axis"],
        "selected_comparison_yaw_deg": selected["yaw_deg"],
        "orientation_changed": orientation_changed,
        "decision": (
            "obvious_error_minimum_multiview_direction_loss"
            if orientation_changed
            else "obvious_error_minimum_multiview_direction_loss_is_identity"
        ),
        "identity_best_view_losses": identity_view_losses,
        "shape_candidates": [
            {key: value for key, value in candidate.items() if key != "matrix"}
            for candidate in sorted(
                candidates, key=lambda item: item["loss"],
            )
        ],
    }


def blueprint_object_pose(item, mesh_path):
    """Initialize an object when no unique top-view instance is available."""
    placement = item["placement"]
    center = placement["center_xy_norm"]
    bbox_cm = item.get("bbox_cm", placement.get("size_cm"))
    if (
        not isinstance(center, list) or len(center) != 2
        or not isinstance(bbox_cm, list) or len(bbox_cm) != 3
    ):
        raise ValueError(f"invalid blueprint placement for {item['object_id']}")
    native = mesh_extents_z_up(mesh_path)
    target = np.asarray(bbox_cm, dtype=float) / 100.0
    return {
        "translation_world_m": [
            float(center[0]) * TABLE_SIZE_M[0],
            float(center[1]) * TABLE_SIZE_M[1],
            0.0,
        ],
        "uniform_scale": uniform_scale(native, target),
        "yaw_deg": float(placement.get("yaw_deg", 0.0)),
        "axis_mapping_quarter_turns": 0,
        "initialization": "blueprint_front_only_fallback",
    }


def annotate_alignment_object_ids(image, object_regions):
    output = np.asarray(image, dtype=np.uint8).copy()
    height, width = output.shape[:2]
    font_scale = max(0.45, min(width, height) / 1400.0)
    for object_id, region in object_regions.items():
        rows, columns = np.nonzero(np.asarray(region) > 0)
        if not len(rows):
            continue
        (text_width, text_height), baseline = cv2.getTextSize(
            object_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        left = min(max(0, int(columns.min())), max(0, width - text_width - 6))
        bottom = int(rows.min()) - 5
        if bottom - text_height - baseline - 4 < 0:
            bottom = min(height - baseline - 3, int(rows.min()) + text_height + baseline + 5)
        cv2.rectangle(
            output,
            (left, bottom - text_height - baseline - 4),
            (left + text_width + 6, bottom + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output, object_id, (left + 3, bottom),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def save_edge_overlay(
    image_path, labels, ids, objects, camera, output, edge_detector="canny",
    rendered_image_path=None, model_edge_method="rendered_rgb", table=None,
    table_mask=None,
):
    image = np.asarray(Image.open(image_path).convert("RGB")).copy()
    source = image.copy()
    target = np.zeros(labels.shape, dtype=np.uint8)
    rendered = np.zeros(labels.shape, dtype=np.uint8)
    rendered_mask = np.zeros(labels.shape, dtype=np.uint8)
    object_regions = {}
    reference_method = reference_edge_detector(edge_detector)
    full_frame_reference_edges = (
        teed_edges(source)[0]
        if reference_method == "teed" and any(key in ids for key in objects)
        else None
    )
    for object_id, item in objects.items():
        object_region = np.zeros(labels.shape, dtype=bool)
        if object_id in ids:
            target_mask_value = target_mask(labels, ids, object_id)
            object_region |= target_mask_value > 0
            reference_edges, _ = (
                masked_full_frame_teed_edges(
                    full_frame_reference_edges, target_mask_value > 0,
                )
                if full_frame_reference_edges is not None else
                masked_edges(
                    source, target_mask_value > 0, method=reference_method,
                )
            )
            target = np.maximum(
                target,
                edge_visualization_values(
                    internal_edges(reference_edges, target_mask_value),
                ),
            )
            target = np.maximum(
                target,
                thin_external_boundary(target_mask_value).astype(np.uint8) * 255,
            )
        has_final_3d_pose = (
            item.get("top_refinement", {}).get("method")
            == "joint_differentiable_mask_refinement"
        )
        if (
            item.get("top_template_path")
            and camera.get("name") == "top_camera"
            and not has_final_3d_pose
            and rendered_image_path is None
        ):
            template_mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=3000)
            template_mask, template_edges, _ = load_rendered_edge_template(
                item["top_template_path"], edge_detector, template_mesh,
                item["top_template_pose"], camera,
            )
            baseline = item["top_template_pose"]
            candidate_mask = warp_rendered_template(template_mask, baseline, item, camera)
            candidate_edges = warp_rendered_template(
                template_edges.astype(np.uint8), baseline, item, camera,
            )
            object_region |= candidate_mask > 0
            rendered_mask |= np.asarray(candidate_mask, dtype=np.uint8)
            if rendered_image_path is None:
                rendered = np.maximum(
                    rendered,
                    edge_visualization_values(
                        internal_edges(candidate_edges, candidate_mask),
                    ),
                )
                rendered = np.maximum(
                    rendered,
                    thin_external_boundary(candidate_mask).astype(np.uint8) * 255,
                )
        else:
            mesh = search_mesh_z_up(
                Path(item["geometry_asset"]),
                face_count=None if has_final_3d_pose else 12000,
            )
            vertices = (
                center_vertices_on_table(mesh.vertices)
                if has_final_3d_pose else
                center_vertices_with_reference(
                    mesh.vertices,
                    search_mesh_z_up(
                        Path(item["geometry_asset"]), face_count=None,
                    ).vertices,
                    center_vertices_on_table,
                )
            )
            scale_xyz = item.get("scale_xyz", [item["uniform_scale"]] * 3)
            posed = pose_vertices(
                vertices, item["uniform_scale"], item["translation_world_m"][:2], item["yaw_deg"],
                scale_xyz=scale_xyz,
                z=item["translation_world_m"][2],
                roll_x_deg=item.get("roll_x_deg", 0.0),
                roll_y_deg=item.get("roll_y_deg", 0.0),
            )
            silhouette = render_mesh_mask(mesh, posed, camera)
            object_region |= silhouette > 0
            rendered_mask |= np.asarray(silhouette, dtype=np.uint8)
            if rendered_image_path is None:
                rendered = np.maximum(
                    rendered,
                    edge_visualization_values(render_mesh_edges(
                        mesh, posed, camera, silhouette, visible_only=True,
                    )),
                )
        object_regions[object_id] = object_region
    rendered_table_edges = np.zeros(labels.shape, dtype=np.uint8)
    if table_mask is not None:
        if table is None:
            raise ValueError("table is required when table_mask is provided")
        target_table_edges, rendered_table_edges = full_table_alignment_edges(
            table, camera, table_mask,
        )
        target = np.maximum(target, target_table_edges)
    if model_edge_method == "normal_discontinuity":
        rendered = edge_visualization_values(
            render_scene_normal_edges(objects, camera, table=table),
        )
    elif rendered_image_path is not None:
        rendered_image = np.asarray(
            Image.open(rendered_image_path).convert("RGB"),
        )
        if rendered_image.shape[:2] != rendered_mask.shape:
            raise ValueError("rendered image and projected masks must have the same size")
        detected, _ = masked_edges(
            rendered_image, rendered_mask > 0,
            method=reference_edge_detector(edge_detector),
        )
        rendered = edge_visualization_values(
            internal_edges(detected, rendered_mask),
        )
        rendered = np.maximum(
            rendered,
            thin_external_boundary(rendered_mask).astype(np.uint8) * 255,
        )
    rendered = np.maximum(rendered, rendered_table_edges)
    image = color_alignment_edges(image, target, rendered)
    image = annotate_alignment_object_ids(image, object_regions)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)


def save_direct_teed_edge_overlay(reference_image_path, rendered_image_path, output):
    """Compare full-frame TEED responses from the reference and final render."""
    reference = np.asarray(Image.open(reference_image_path).convert("RGB"))
    rendered = np.asarray(Image.open(rendered_image_path).convert("RGB"))
    if reference.shape != rendered.shape:
        raise ValueError("reference and rendered images must have the same size")
    reference_edges, _ = teed_edges(reference)
    rendered_edges, _ = teed_edges(rendered)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        color_alignment_edges(reference, reference_edges, rendered_edges),
    ).save(output)


def reference_scene_edges(image, labels, ids, object_ids, edge_detector="teed"):
    edges = np.zeros(np.asarray(labels).shape, dtype=np.uint8)
    reference_method = reference_edge_detector(edge_detector)
    full_frame_teed = (
        teed_edges(image)[0]
        if reference_method == "teed" and any(key in ids for key in object_ids)
        else None
    )
    for instance_id in ("table_0", *object_ids):
        if instance_id not in ids:
            continue
        mask = target_mask(labels, ids, instance_id)
        if instance_id != "table_0":
            detected, _ = (
                masked_full_frame_teed_edges(full_frame_teed, mask > 0)
                if full_frame_teed is not None else
                masked_edges(image, mask > 0, method=reference_method)
            )
            edges = np.maximum(
                edges,
                edge_visualization_values(internal_edges(detected, mask)),
            )
        edges = np.maximum(
            edges, thin_external_boundary(mask).astype(np.uint8) * 255,
        )
    return edges


def save_stage_alignment_visualizations(
    scene_json, render_dir, references=None, label_maps=None, alpha=0.5,
):
    document = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    render_dir = Path(render_dir)
    references = references or {
        "front": ROOT / "data/reference_image.png",
        "top": ROOT / "data/topview_image.png",
    }
    if label_maps is None:
        label_maps = {
            "front": load_scene_label_map(ROOT / "data/segmentation_results.json", ROOT),
            "top": load_scene_label_map(ROOT / "data/topview_segmentation_results.json", ROOT),
        }
    edge_detector = document.get("refinement", {}).get("edge_detector")
    if edge_detector is None:
        edge_detector = next(
            (item.get("edge_detector") for item in document["objects"].values() if item.get("edge_detector")),
            "edge_drawing",
        )
    for view in ("front", "top"):
        reference = np.asarray(Image.open(references[view]).convert("RGB"))
        rendered = np.asarray(Image.open(render_dir / f"{view}.png").convert("RGB"))
        labels, ids = label_maps[view]
        rendered_mask, rendered_edges = scene_render_maps(document, document["cameras"][view])
        target_edges = reference_scene_edges(
            reference, labels, ids, tuple(document["objects"]), edge_detector,
        )
        Image.fromarray(
            blend_rendered_scene(reference, rendered, rendered_mask, alpha),
        ).save(render_dir / f"{view}_overlay.png")
        Image.fromarray(
            color_alignment_edges(reference, target_edges, rendered_edges),
        ).save(render_dir / f"{view}_edges.png")


def is_soft_edge_map(edges):
    values = np.asarray(edges)
    maximum = float(values.max()) if values.size else 0.0
    return (
        values.dtype != bool
        and np.any((values > 0) & (values < maximum))
    )


def edge_support(edges, relative_level=0.5):
    """Binary support for visualization-only consumers of soft edge maps."""
    values = np.asarray(edges)
    if not is_soft_edge_map(values):
        return values.astype(bool)
    return values >= float(relative_level) * float(values.max())


def edge_visualization_values(edges):
    """Preserve TEED confidence; promote binary edge maps to full intensity."""
    values = np.asarray(edges)
    if is_soft_edge_map(values):
        return np.clip(values, 0, 255).astype(np.uint8)
    return values.astype(bool).astype(np.uint8) * 255


INTERNAL_EDGE_OUTER_BAND_FRACTION = 0.05
TEED_TARGET_INTERNAL_EDGE_MIN_CONFIDENCE = 64


def _internal_edge_support(mask):
    """Keep visible pixels farther than 5% of the outer short side."""
    visible = np.asarray(mask) > 0
    ys, xs = np.nonzero(visible)
    if not len(xs):
        return visible
    outer = solid_external_silhouette(visible.astype(np.uint8)) > 0
    _, (width, height), _ = cv2.minAreaRect(
        np.column_stack((xs, ys)).astype(np.float32),
    )
    short_side = max(min(float(width), float(height)), 1.0)
    distance = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 3)
    return visible & (
        distance > INTERNAL_EDGE_OUTER_BAND_FRACTION * short_side
    )


def internal_edges(edges, mask):
    support = _internal_edge_support(mask)
    values = np.asarray(edges)
    if is_soft_edge_map(values):
        return np.where(support, values, 0)
    raw = (np.asarray(edges, dtype=bool) & support).astype(np.uint8)
    if not raw.any():
        return raw.astype(bool)

    # Canny reports both intensity transitions of a narrow stripe.  Collapse
    # only elongated closed components (straps, seams, wheel marks) to their
    # medial line; compact details such as keys and circular rims stay as
    # ordinary contours.  The threshold is derived from the instance size,
    # not from an object-specific constant.
    mask_area = max(int(np.count_nonzero(mask)), 1)
    max_narrow_width = max(3.0, 0.12 * math.sqrt(mask_area))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw, connectivity=8,
    )
    output = np.zeros_like(raw)
    kernel = np.ones((3, 3), np.uint8)
    for component in range(1, component_count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        # One zero-valued pixel around the component preserves the full-frame
        # morphology boundary while avoiding full-image work per component.
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1 = min(raw.shape[1], x + width + 1)
        y1 = min(raw.shape[0], y + height + 1)
        component_mask = np.where(
            labels[y0:y1, x0:x1] == component, 255, 0,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        width, height = cv2.minAreaRect(contour)[1]
        minor, major = sorted((float(width), float(height)))
        if minor <= 0.0 or major / minor < 3.0 or minor > max_narrow_width:
            output[y0:y1, x0:x1] |= component_mask
            continue

        filled = np.zeros_like(component_mask)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        skeleton = np.zeros_like(component_mask)
        while filled.any():
            eroded = cv2.erode(filled, kernel)
            opened = cv2.dilate(eroded, kernel)
            skeleton |= filled & ~opened
            filled = eroded
        output[y0:y1, x0:x1] |= skeleton
    return output > 0


def internal_edge_length_density(edges, mask):
    """Measure internal one-pixel edge length relative to object size."""
    return float(np.count_nonzero(edges)) / math.sqrt(
        max(np.count_nonzero(mask), 1),
    )


def internal_edge_count_loss(candidate_density, target_density):
    epsilon = 1e-6
    return abs(math.log(
        (float(candidate_density) + epsilon)
        / (float(target_density) + epsilon)
    ))


def normalized_internal_edges(edges, mask, size=128):
    ys, xs = np.where(np.asarray(mask) > 0)
    if not len(xs):
        return np.zeros((size, size), dtype=bool)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    resized_mask = cv2.resize(
        np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8),
        (size, size), interpolation=cv2.INTER_NEAREST,
    )
    source_edges = np.asarray(edges[y0:y1, x0:x1])
    soft = is_soft_edge_map(source_edges)
    resized_edges = cv2.resize(
        source_edges.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_LINEAR if soft else cv2.INTER_NEAREST,
    )
    if not soft:
        resized_edges = resized_edges > 0
    return internal_edges(resized_edges, resized_mask)


def choose_180_direction(exterior, primary_interior, secondary_interior=None):
    exterior = [float(value) for value in exterior]
    primary_interior = [float(value) for value in primary_interior]
    has_secondary = secondary_interior is not None
    secondary_interior = (
        [float(value) for value in secondary_interior]
        if secondary_interior is not None else [float("inf"), float("inf")]
    )
    combined = []
    for primary, secondary in zip(primary_interior, secondary_interior):
        finite = [value for value in (primary, secondary) if np.isfinite(value)]
        combined.append(float(sum(finite)) if finite else float("inf"))
    tolerance = max(0.75, 0.10 * min(exterior))
    symmetric = abs(exterior[0] - exterior[1]) <= tolerance
    # Exterior contours determine pose, never front/back direction.  Even an
    # asymmetric projected silhouette may prefer the wrong 180-degree pose;
    # only masked internal features are allowed to disambiguate it.
    flipped = np.isfinite(combined).all() and combined[1] + 1e-6 < combined[0]
    report = {
        "roughly_180_symmetric": bool(symmetric),
        "exterior_loss_0_180": exterior,
        "internal_edge_loss_0_180": primary_interior,
        "combined_internal_edge_loss_0_180": combined,
        "kept_180_flipped_candidate": bool(flipped),
    }
    if has_secondary:
        report["secondary_internal_edge_loss_0_180"] = secondary_interior
    return bool(flipped), report


def resolve_180_direction(mesh, item, target, target_edges, camera, secondary_direction=None):
    vertices = center_vertices_on_table(mesh.vertices)

    def view_scores(candidate, view_camera, view_target, view_edges):
        search_camera = scaled_camera(view_camera, 0.25)
        search_target = cv2.resize(
            view_target, tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST,
        )
        soft = is_soft_edge_map(view_edges)
        search_edges = cv2.resize(
            np.asarray(view_edges, dtype=np.uint8), tuple(search_camera["image_size"]),
            interpolation=cv2.INTER_LINEAR if soft else cv2.INTER_NEAREST,
        )
        if not soft:
            search_edges = search_edges > 0
        scale_xyz = candidate.get("scale_xyz", [candidate["uniform_scale"]] * 3)
        posed = pose_vertices(
            vertices, 1.0, candidate["translation_world_m"][:2], candidate["yaw_deg"],
            scale_xyz=scale_xyz,
        )
        rendered_mask = render_mesh_mask(mesh, posed, search_camera)
        rendered_edges = render_mesh_edges(mesh, posed, search_camera, rendered_mask)
        exterior = symmetric_contour_loss(rendered_mask, search_target)
        interior = ot_edge_loss(
            normalized_internal_edges(rendered_edges, rendered_mask),
            normalized_internal_edges(search_edges, search_target),
        )
        return exterior, interior

    candidates = [dict(item), dict(item)]
    candidates[1]["yaw_deg"] = float(((item["yaw_deg"] + 360.0) % 360.0) - 180.0)
    primary = [view_scores(candidate, camera, target, target_edges) for candidate in candidates]
    secondary = None
    if secondary_direction is not None:
        second_camera, second_target, second_edges = secondary_direction
        secondary = [
            view_scores(candidate, second_camera, second_target, second_edges)[1]
            for candidate in candidates
        ]
    use_flipped, report = choose_180_direction(
        [value[0] for value in primary], [value[1] for value in primary], secondary,
    )
    result = candidates[int(use_flipped)]
    result["direction_disambiguation"] = report
    return result


def resolve_template_180_direction(item, target, target_edges, camera):
    mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=3000)
    template_mask, template_edges, metadata = load_rendered_edge_template(
        item["top_template_path"], item.get("edge_detector", "canny"), mesh,
        item["top_template_pose"], camera,
    )
    baseline = item["top_template_pose"]
    target_internal = normalized_internal_edges(target_edges, target)
    candidates = [dict(item), dict(item)]
    candidates[1]["yaw_deg"] = float(((item["yaw_deg"] + 360.0) % 360.0) - 180.0)
    exterior, interior = [], []
    for candidate in candidates:
        candidate_mask = warp_rendered_template(template_mask, baseline, candidate, camera)
        candidate_edges = warp_rendered_template(
            template_edges.astype(np.uint8), baseline, candidate, camera,
        )
        exterior.append(symmetric_contour_loss(candidate_mask, target))
        interior.append(ot_edge_loss(
            normalized_internal_edges(candidate_edges, candidate_mask), target_internal,
        ))
    use_flipped, report = choose_180_direction(exterior, interior)
    result = candidates[int(use_flipped)]
    result["direction_disambiguation"] = report
    result["top_template_edge_detector"] = metadata
    if metadata["method"] == "canny":
        result["top_template_canny_thresholds"] = [
            metadata["low_threshold"], metadata["high_threshold"],
        ]
    return result


def refine_object_top_staged(mesh, initial, target, target_edges, camera):
    vertices = center_vertices_on_table(mesh.vertices)
    search_camera = scaled_camera(camera, 0.25)
    search_target = cv2.resize(
        target, tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST,
    )
    target_points = sample_external_contour(search_target)
    target_normalized, target_center, target_radius = normalized_contour(target_points)
    item = {**initial}
    item.pop("scale_xyz", None)

    def rendered_mask(candidate):
        posed = pose_vertices(
            vertices, candidate["uniform_scale"], candidate["translation_world_m"][:2],
            candidate["yaw_deg"],
        )
        return render_mesh_mask(mesh, posed, search_camera)

    def align_center(candidate):
        source_points = sample_external_contour(rendered_mask(candidate))
        _, source_center, _ = normalized_contour(source_points)
        pixel_delta = target_center - source_center
        xy = np.asarray(candidate["translation_world_m"][:2], dtype=float)
        epsilon = 0.001
        origin = np.array([[xy[0], xy[1], 0.0]])
        base_pixel = project_points(origin, search_camera)[0][0]
        jacobian = np.column_stack((
            (project_points(origin + [epsilon, 0.0, 0.0], search_camera)[0][0] - base_pixel) / epsilon,
            (project_points(origin + [0.0, epsilon, 0.0], search_camera)[0][0] - base_pixel) / epsilon,
        ))
        xy += np.linalg.lstsq(jacobian, pixel_delta, rcond=None)[0]
        candidate["translation_world_m"] = [float(xy[0]), float(xy[1]), 0.0]

    align_center(item)

    def yaw_loss(yaw):
        candidate = {**item, "yaw_deg": float(yaw)}
        points = sample_external_contour(rendered_mask(candidate))
        normalized, _, _ = normalized_contour(points)
        return sinkhorn_divergence(normalized, target_normalized, normalization=2.0)

    initial_yaw = float(item["yaw_deg"])
    yaw_candidates = initial_yaw + np.arange(-90.0, 90.0, 10.0)
    yaw_losses = np.array([yaw_loss(yaw) for yaw in yaw_candidates])
    best_index = int(np.argmin(yaw_losses))
    loss_spread = float(np.percentile(yaw_losses, 90) - yaw_losses[best_index])
    yaw_observable = loss_spread > max(2e-4, 0.05 * float(np.median(yaw_losses)))
    if yaw_observable:
        anchor = float(yaw_candidates[best_index])
        result = minimize_scalar(
            yaw_loss, bounds=(anchor - 10.0, anchor + 10.0), method="bounded",
            options={"xatol": 0.25, "maxiter": 24},
        )
        item["yaw_deg"] = float(((result.x + 180.0) % 360.0) - 180.0)

    # Once yaw is fixed, uniform scale is the ratio of contour RMS radii.
    # Re-rendering makes this valid for the fixed perspective top camera too.
    scale_updates = []
    for _ in range(2):
        source_points = sample_external_contour(rendered_mask(item))
        _, _, source_radius = normalized_contour(source_points)
        ratio = float(target_radius / source_radius)
        item["uniform_scale"] *= ratio
        scale_updates.append(ratio)
        align_center(item)

    item["loss"] = float(yaw_loss(item["yaw_deg"]))
    item["success"] = True
    item["top_staged_refinement"] = {
        "method": "centroid_then_normalized_balanced_sinkhorn_yaw_then_rms_scale",
        "yaw_observable": bool(yaw_observable),
        "yaw_loss_spread": loss_spread,
        "scale_updates": scale_updates,
    }
    return resolve_template_180_direction(item, target, target_edges, camera)


def refine_object_uniform(mesh, initial, target, target_edges, camera, active_scale, top_constraint=None):
    vertices = center_vertices_on_table(mesh.vertices)
    optimization_scale = 0.25
    camera = scaled_camera(camera, optimization_scale)
    target = cv2.resize(
        target, tuple(camera["image_size"]), interpolation=cv2.INTER_NEAREST
    )
    target_edges = cv2.resize(
        np.asarray(target_edges, dtype=np.uint8), tuple(camera["image_size"]), interpolation=cv2.INTER_NEAREST
    ) > 0
    if top_constraint is not None:
        constraint_camera, constraint_target = top_constraint
        constraint_camera = scaled_camera(constraint_camera, optimization_scale)
        constraint_target = cv2.resize(
            constraint_target, tuple(constraint_camera["image_size"]), interpolation=cv2.INTER_NEAREST
        )
        top_constraint = (constraint_camera, constraint_target)
    p0 = np.array([
        initial["translation_world_m"][0], initial["translation_world_m"][1],
        initial["yaw_deg"], math.log(initial["uniform_scale"]),
    ])
    x0, y0 = p0[:2]

    fixed_scale_xyz = np.asarray(initial.get("scale_xyz", [initial["uniform_scale"]] * 3), dtype=float)
    target_outer = thin_external_boundary(target)

    def loss(parameters):
        scale = math.exp(parameters[3])
        posed = (
            pose_vertices(vertices, scale, parameters[:2], parameters[2])
            if active_scale
            else pose_vertices(vertices, 1.0, parameters[:2], parameters[2], scale_xyz=fixed_scale_xyz)
        )
        rendered = render_mesh_mask(mesh, posed, camera)
        # Coarse pose and scale are determined by the external silhouette only.
        # Internal edges are reserved for the final 0/180-degree disambiguation.
        value = ot_edge_loss(thin_external_boundary(rendered), target_outer)
        if top_constraint is not None:
            top_camera, top_target = top_constraint
            top_rendered = render_mesh_mask(mesh, posed, top_camera)
            value += 0.35 * symmetric_contour_loss(top_rendered, top_target)
        return value

    yaw_candidates = np.arange(-180.0, 180.0, 30.0)
    coarse = [(loss([x0, y0, yaw, p0[3]]), yaw) for yaw in yaw_candidates]
    p0[2] = min(coarse)[1]
    bounds = [(x0 - 0.08, x0 + 0.08), (y0 - 0.08, y0 + 0.08), (p0[2] - 20, p0[2] + 20)]
    bounds.append(
        (p0[3] + math.log(0.8), p0[3] + math.log(1.2))
        if active_scale else (p0[3], p0[3])
    )
    result = minimize(loss, p0, method="Powell", bounds=bounds, options={"maxiter": 30, "xtol": 5e-4, "ftol": 5e-4})
    parameters = result.x
    refined = {
        **initial,
        "translation_world_m": [float(parameters[0]), float(parameters[1]), 0.0],
        "yaw_deg": float(((parameters[2] + 180) % 360) - 180),
        "uniform_scale": float(math.exp(parameters[3])),
        "loss": float(result.fun),
        "success": bool(result.success),
    }
    if active_scale:
        refined.pop("scale_xyz", None)
    return refined


def refine_object(
    mesh, initial, target, target_edges, camera, scale_mode, top_constraint=None,
    direction_constraint=None,
):
    if scale_mode != "xy":
        positioned = refine_object_uniform(
            mesh, initial, target, target_edges, camera,
            active_scale=(scale_mode == "uniform"), top_constraint=top_constraint,
        )
    else:
        positioned = refine_object_uniform(
            mesh, initial, target, target_edges, camera, active_scale=False,
            top_constraint=top_constraint,
        )
        vertices = center_vertices_on_table(mesh.vertices)
        search_camera = scaled_camera(camera, 0.25)
        search_target = cv2.resize(target, tuple(search_camera["image_size"]), interpolation=cv2.INTER_NEAREST)
        target_bbox = robust_bbox(search_target > 0)
        target_size = np.array([target_bbox[2] - target_bbox[0] + 1, target_bbox[3] - target_bbox[1] + 1], dtype=float)
        base = float(positioned["uniform_scale"])

        def bbox_loss(log_scales):
            scale_xyz = [math.exp(log_scales[0]), math.exp(log_scales[1]), base]
            posed = pose_vertices(vertices, 1.0, positioned["translation_world_m"][:2], positioned["yaw_deg"], scale_xyz=scale_xyz)
            bbox = robust_bbox(render_mesh_mask(mesh, posed, search_camera))
            rendered_size = np.array([bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1], dtype=float)
            return float(np.sum(((rendered_size - target_size) / target_size) ** 2))

        scale0 = np.log([base, base])
        result = minimize(
            bbox_loss, scale0, method="Powell",
            bounds=[(scale0[0] + math.log(0.6), scale0[0] + math.log(1.4)), (scale0[1] + math.log(0.6), scale0[1] + math.log(1.4))],
            options={"maxiter": 30, "xtol": 1e-4, "ftol": 1e-5},
        )
        positioned["scale_xyz"] = [float(math.exp(result.x[0])), float(math.exp(result.x[1])), base]
        positioned["scale_mode"] = "xy"
    return resolve_180_direction(
        mesh, positioned, target, target_edges, camera,
        secondary_direction=direction_constraint,
    )


def camera_json(view):
    value = dict(view["camera"])
    value["metric_scale_world_per_vggt"] = float(view["metric_scale"])
    value["rotation_world_from_vggt"] = view["rotation_world_from_vggt"].tolist()
    value["translation_world"] = view["translation_world"].tolist()
    value["plane_point_count"] = view["plane_point_count"]
    value["table_plane_fit"] = view["table_plane_fit"]
    return value


def scene_document(
    table_mesh, table_scale, cameras, objects,
    table_scale_pivot=None, table_yaw_deg=0.0,
):
    return {
        "format": "fixed_two_camera_alignment_v1",
        "coordinate_system": {"origin": "tabletop center", "x_axis": "table width", "y_axis": "table depth", "z_axis": "up", "unit": "meter"},
        "table": {
            "mesh": str(table_mesh),
            "scale_xyz": list(table_scale),
            "blender_scale_xyz": table_blender_scale(
                table_scale, table_yaw_deg,
            ),
            "yaw_deg": float(table_yaw_deg),
            "scale_pivot": table_scale_pivot,
        },
        "cameras": cameras,
        "objects": objects,
    }


def blender_asset_signature(document):
    return (
        str(document["table"]["mesh"]),
        tuple(sorted(
            (
                str(object_id),
                str(item.get("asset_type", "rigid")),
                str(item.get("asset", item.get("geometry_asset", ""))),
                str(item.get("asset_orientation_matrix", "")),
            )
            for object_id, item in document["objects"].items()
        )),
    )


def blender_executable():
    configured = os.environ.get("BLENDER_BIN")
    if configured:
        return configured
    found = shutil.which("blender")
    if found:
        return found
    candidates = sorted((Path.home() / ".local/blender").glob("blender-*/blender"))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError("Blender not found; set BLENDER_BIN")


def blender_environment():
    environment = os.environ.copy()
    environment["TABLETOP_ALIGNMENT_ROOT"] = str(ROOT)
    blender_devices = environment.get("BLENDER_CUDA_VISIBLE_DEVICES")
    if blender_devices:
        environment["CUDA_VISIBLE_DEVICES"] = blender_devices
    if environment.get("TABLETOP_BLENDER_DISPLAY"):
        environment["DISPLAY"] = environment["TABLETOP_BLENDER_DISPLAY"]
    return environment


def render_stages(jobs):
    cmd = [
        blender_executable(), "--background", "--factory-startup", "--python", str(Path(__file__).resolve()), "--",
        "--blender-worker",
    ]
    for scene_json, output_dir, save_blend in jobs:
        cmd.extend(("--render-job", f"{Path(scene_json)}::{Path(output_dir)}::{int(save_blend)}"))
    subprocess.run(cmd, check=True, cwd=ROOT, env=blender_environment())


def render_stage(scene_json, output_dir, save_blend=False):
    render_stages([(scene_json, output_dir, save_blend)])


def render_front_stages(jobs):
    _persistent_blender_worker().request({
        "command": "render_jobs",
        "scene_mode": _blender_scene_mode,
        "views": ["front"],
        "samples": 8,
        "jobs": [
            [str(scene_json), str(output_dir), False]
            for scene_json, output_dir in jobs
        ],
    })


def render_top_edge_templates(scene_json, output_dir, render_scale=1):
    cmd = [
        blender_executable(), "--background", "--factory-startup", "--python", str(Path(__file__).resolve()), "--",
        "--blender-worker", "--top-template-job",
        f"{Path(scene_json)}::{Path(output_dir)}::{int(render_scale)}",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=blender_environment())


def blender_bake_mesh_orientation(source, destination, matrix):
    """Write the selected Z-up orientation into rigid mesh vertex coordinates."""
    import bpy
    from mathutils import Matrix
    from pipeline.scene_renderer import clear_scene, import_asset

    clear_scene()
    controller, objects = import_asset(Path(source), "normalized_asset")
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(matrix, dtype=float)
    world_rotation = Matrix(transform.tolist())
    bpy.context.view_layer.update()
    for obj in objects:
        if obj.parent == controller:
            obj.matrix_world = world_rotation @ obj.matrix_world
    bpy.context.view_layer.update()
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"asset contains no meshes: {source}")
    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.data = obj.data.copy()
        obj.data.transform(world)
        obj.matrix_world.identity()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(destination), export_format="GLB", use_selection=True,
    )




def bake_mesh_orientations(jobs):
    if jobs:
        worker = _persistent_blender_worker()
        try:
            worker.request({
                "command": "bake_mesh_orientations",
                "jobs": [
                    [str(source), str(destination), np.asarray(matrix).tolist()]
                    for source, destination, matrix in jobs
                ],
            })
        finally:
            # Exporting GLBs leaves Blender data/GPU state behind.  Template
            # rendering is more reliable in a clean worker.
            close_persistent_blender_worker()


def panel(path, title, size=(768, 432)):
    image = ImageOps.contain(Image.open(path).convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 32), "#202124")
    ImageDraw.Draw(canvas).text((8, 8), title, fill="white")
    canvas.paste(image, ((size[0] - image.width) // 2, 32 + (size[1] - image.height) // 2))
    return canvas


def compose_stage_comparison(
    current_dir,
    output,
    current_title,
    previous_dir=None,
    previous_title=None,
    references=None,
):
    references = references or {
        "front": ROOT / "data/reference_image.png",
        "top": ROOT / "data/topview_image.png",
    }
    columns = [(None, "reference")]
    if previous_dir is not None:
        columns.append((Path(previous_dir), previous_title or "previous stage"))
    columns.append((Path(current_dir), current_title))
    canvas = Image.new("RGB", (len(columns) * 768, 2 * 468), "#202124")
    for row, view in enumerate(("front", "top")):
        for column, (directory, title) in enumerate(columns):
            path = references[view] if directory is None else directory / f"{view}.png"
            canvas.paste(panel(path, f"{view.upper()} {title}"), (column * 768, row * 468))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def compose_report(stages, output):
    titles = [
        "reference",
        "stage 1: table+cameras+object coarse",
        "stage 2: top refined",
        "stage 3: front final",
    ]
    rows = []
    for view in ("front", "top"):
        items = []
        reference = ROOT / ("data/reference_image.png" if view == "front" else "data/topview_image.png")
        items.append(panel(reference, f"{view.upper()} {titles[0]}"))
        for title, stage in zip(titles[1:], stages):
            items.append(panel(Path(stage) / f"{view}.png", f"{view.upper()} {title}"))
        rows.append(items)
    width, height = len(rows[0]) * 768, 2 * 468
    canvas = Image.new("RGB", (width, height), "#202124")
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            canvas.paste(image, (column_index * 768, row_index * 468))
    canvas.save(output)


def output_paths(output):
    output = Path(output).resolve()
    return {
        "table_cameras": output / "01_table_cameras.json",
        "table_alignment": output / "02_table_alignment.json",
        "table_front": output / "02_table_alignment_render/front.png",
        "table_top": output / "02_table_alignment_render/top.png",
        "table_front_overlay": output / "02_table_alignment_render/front_overlay.png",
        "table_top_overlay": output / "02_table_alignment_render/top_overlay.png",
        "table_front_render_edges": output / "02_table_alignment_render/front_edges.png",
        "table_top_render_edges": output / "02_table_alignment_render/top_edges.png",
        "table_comparison": output / "02_table_alignment_comparison.png",
        "object_coarse_alignment": output / "03_object_coarse_alignment.json",
        "object_coarse_front": output / "03_object_coarse_render/front.png",
        "object_coarse_top": output / "03_object_coarse_render/top.png",
        "object_coarse_front_overlay": output / "03_object_coarse_render/front_overlay.png",
        "object_coarse_top_overlay": output / "03_object_coarse_render/top_overlay.png",
        "object_coarse_front_render_edges": output / "03_object_coarse_render/front_edges.png",
        "object_coarse_top_render_edges": output / "03_object_coarse_render/top_edges.png",
        "object_coarse_front_edges": output / "03_object_coarse_front_edges.png",
        "object_coarse_top_edges": output / "03_object_coarse_top_edges.png",
        "object_coarse_comparison": output / "03_object_coarse_comparison.png",
        "top_refined": output / "04_top_refined.json",
        "top_refined_front": output / "04_top_refined_render/front.png",
        "top_refined_top": output / "04_top_refined_render/top.png",
        "top_refined_front_overlay": output / "04_top_refined_render/front_overlay.png",
        "top_refined_top_overlay": output / "04_top_refined_render/top_overlay.png",
        "top_refined_front_render_edges": output / "04_top_refined_render/front_edges.png",
        "top_refined_top_render_edges": output / "04_top_refined_render/top_edges.png",
        "top_refined_edges": output / "04_top_refined_edges.png",
        "top_refined_comparison": output / "04_top_refined_comparison.png",
        "front_refined": output / "05_front_refined.json",
        "front_refinement_debug": output / "05_front_refinement_debug",
        "final_front": output / "05_front_refined_render/front.png",
        "final_top": output / "05_front_refined_render/top.png",
        "front_refined_edges": output / "05_front_refined_edges.png",
        "final_blend": output / "05_front_refined_render/final_scene.blend",
        "comparison": output / "scene_alignment_comparison.png",
    }


def printable_output_paths(output, names=None):
    paths = output_paths(output)
    selected = set(paths) if names is None else set(names)
    return {
        name: path for name, path in paths.items()
        if name in selected and path.suffix.lower() == ".png"
    }


def print_output_paths(output, planned=False, names=None):
    prefix = "planned output" if planned else "output"
    print(f"[{prefix} paths]", flush=True)
    for name, path in printable_output_paths(output, names).items():
        print(f"{name}='{path}'", flush=True)


def differentiable_refinement_config(args):
    return FrontRefinementConfig(
        resolution=args.front_refine_resolution,
        iterations=args.front_refine_iterations,
        fine_resolution=args.front_refine_fine_resolution,
        fine_iterations=args.front_refine_fine_iterations,
        lr_xy=args.front_refine_lr_xy,
        lr_yaw=args.front_refine_lr_yaw,
        lr_scale=args.front_refine_lr_scale,
        max_xy_fraction=args.front_refine_max_xy_fraction,
        scale_limit=args.front_refine_scale_limit,
        sdf_weight=args.front_loss_sdf_weight,
        iou_weight=args.front_loss_iou_weight,
        bbox_weight=args.bbox_loss_weight,
        bbox_center_weight=args.bbox_center_weight,
        edge_weight=args.front_edge_loss_weight,
        edge_parameter_scope=args.front_edge_parameter_scope,
        xy_weight=args.front_loss_xy_weight,
        scale_weight=args.front_loss_scale_weight,
        yaw_weight=args.front_loss_yaw_weight,
        loss_combination="geometric_product",
        rgb_weight=args.refinement_rgb_loss_weight,
        bbox_geometry="aabb",
        pose_update_mode="raster",
        gradient_clip=args.front_refine_gradient_clip,
        patience=args.front_refine_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )


def _top_3d_mask_loss_config(config):
    return replace(
        config,
        loss_combination="geometric_product",
        bbox_weight=0.0,
        edge_weight=0.0,
        rgb_weight=0.0,
    )


def refine_top_masks_differentiable(
    *, objects, top_camera, top_labels, top_ids, table_mesh_path,
    table_scale_xyz, output_dir, config, scale_mode="uniform",
    joint_pose_scale=False, circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
    object_optimization_mode="joint", top_rgb=None,
    table_scale_pivot=None, table_yaw_deg=0.0,
):
    matched = {
        object_id: item for object_id, item in objects.items()
        if object_id in top_ids
    }
    if not matched:
        return dict(objects), {
            "method": "skipped_no_matched_topview_objects",
            "object_ids": [],
            "objects": {},
        }
    top_circle_scores = {
        object_id: rendered_model_circle_score(item["top_template_path"])
        for object_id, item in matched.items()
    }
    contact_shapes = {
        object_id: object_contact_shape(item)
        for object_id, item in matched.items()
    }
    body_shapes = {
        object_id: object_body_cross_section_shape(
            item, circle_score_threshold, square_ratio_threshold,
        )
        for object_id, item in matched.items()
    }
    tie_sources = {
        object_id: object_xy_scale_tie_sources(
            top_circle_scores[object_id], contact_shapes[object_id],
            circle_score_threshold, square_ratio_threshold,
            body_shapes[object_id],
        )
        for object_id in matched
    }
    tied_ids = tuple(
        object_id for object_id in matched
        if any(tie_sources[object_id].values())
    ) if scale_mode == "xy" else ()
    rgb_loss_fns = {}
    rgb_loss_skips = {}
    if config.loss_combination == "geometric_product" and top_rgb is not None:
        width, height = map(int, top_camera["image_size"])
        for object_id, item in matched.items():
            if item.get("top_template_transform") is None:
                rgb_loss_skips[object_id] = "missing_top_template_transform"
                continue
            rgba = np.asarray(Image.open(item["top_template_path"]).convert("RGBA"))
            if rgba.shape[1::-1] != (width, height):
                rgba = np.asarray(Image.fromarray(rgba).resize(
                    (width, height), Image.Resampling.LANCZOS,
                ))
            baseline = item.get("top_template_pose", item)
            aligned_rgb = warp_rendered_template(
                rgba[:, :, :3], baseline, item, top_camera,
            )
            aligned_mask = warp_rendered_template(
                (rgba[:, :, 3] > 16).astype(np.uint8) * 255,
                baseline, item, top_camera,
            )
            if not np.any(aligned_mask):
                rgb_loss_skips[object_id] = "empty_aligned_template_mask"
                continue
            target = target_mask(top_labels, top_ids, object_id)
            source_center, _ = _mask_geometry(aligned_mask)
            target_center, _ = _mask_geometry(target)
            rgb_loss_fns[object_id] = _mask_rgb_product_objective(
                aligned_rgb, aligned_mask, top_rgb, target,
                source_center, target_center, 1.0,
            )
    refined, report = refine_front_masks_differentiable(
        top_objects=matched,
        front_camera=top_camera,
        front_labels=top_labels,
        front_ids=top_ids,
        object_ids=tuple(matched),
        table_mesh_path=table_mesh_path,
        table_scale_xyz=table_scale_xyz,
        mesh_loader=search_mesh_z_up,
        object_vertex_centerer=center_vertices_on_table,
        table_vertex_centerer=lambda vertices: (
            orient_table_vertices(
                center_vertices_at_tabletop(vertices), table_yaw_deg,
            )
            + table_scale_pivot_offset({
                "scale_xyz": table_scale_xyz,
                "scale_pivot": table_scale_pivot,
            })
            / np.asarray(table_scale_xyz, dtype=float)
        ),
        target_mask_fn=target_mask,
        output_dir=output_dir,
        config=config,
        view_name="top",
        scale_mode=scale_mode,
        joint_pose_scale=joint_pose_scale,
        tie_xy_scale_ids=tied_ids,
        object_optimization_mode=object_optimization_mode,
        rgb_loss_fns=rgb_loss_fns,
    )
    report["top_circle_score_threshold"] = float(circle_score_threshold)
    report["top_circle_scores"] = top_circle_scores
    report["top_circle_score_source"] = "model_top_render"
    report["contact_circle_score_threshold"] = float(circle_score_threshold)
    report["contact_bbox_square_ratio_threshold"] = float(square_ratio_threshold)
    report["contact_shapes"] = contact_shapes
    report["body_cross_section_shapes"] = body_shapes
    report["xy_scale_tie_rule"] = (
        "top_view_circle_or_contact_circle_and_square_or_stable_body_cross_section"
    )
    report["xy_scale_tie_sources"] = tie_sources
    report["contact_circle_score_source"] = "lowest_downward_mesh_facet"
    report["rgb_loss_object_ids"] = sorted(rgb_loss_fns)
    report["rgb_loss_skips"] = rgb_loss_skips
    for object_id, shape in contact_shapes.items():
        if object_id in refined:
            refined[object_id]["body_cross_section_shape"] = body_shapes[object_id]
        if object_id in report.get("objects", {}):
            report["objects"][object_id]["top_circle_score"] = float(
                top_circle_scores[object_id],
            )
            report["objects"][object_id]["xy_scale_tie_sources"] = tie_sources[object_id]
            report["objects"][object_id]["contact_circle_score"] = float(shape["circle_score"])
            report["objects"][object_id]["contact_bbox_square_ratio"] = float(shape["bbox_square_ratio"])
            report["objects"][object_id]["contact_circle_score_source"] = "lowest_downward_mesh_facet"
            report["objects"][object_id]["body_cross_section_shape"] = body_shapes[object_id]
            report["objects"][object_id]["xy_scale_tied"] = object_id in tied_ids
        refinement = refined.get(object_id, {}).get("top_refinement")
        if refinement is not None:
            refinement["top_circle_score"] = float(top_circle_scores[object_id])
            refinement["xy_scale_tie_sources"] = tie_sources[object_id]
            refinement["contact_circle_score"] = float(shape["circle_score"])
            refinement["contact_bbox_square_ratio"] = float(shape["bbox_square_ratio"])
            refinement["contact_circle_score_source"] = "lowest_downward_mesh_facet"
            refinement["body_cross_section_shape"] = body_shapes[object_id]
            refinement["xy_scale_tied"] = object_id in tied_ids
    return {**objects, **refined}, report


def _bbox_bottom_center(mask):
    left, _, right, bottom = robust_bbox(np.asarray(mask) > 0)
    return np.array([(left + right) * 0.5, bottom], dtype=float)


def _bbox_width(mask):
    left, _, right, _ = robust_bbox(np.asarray(mask) > 0)
    return max(float(right - left), 1.0)


def _bbox_height(mask):
    _, top, _, bottom = robust_bbox(np.asarray(mask) > 0)
    return max(float(bottom - top), 1.0)


def initialize_front_bbox_pose(
    mesh, item, target, camera,
):
    """Fit every local scale axis from front height before yaw selection."""
    vertices = center_vertices_on_table(mesh.vertices)
    fitted, frame_fit = fit_top_template_pose(vertices, item, camera)

    def rendered_mask(pose, view_camera):
        posed = pose_vertices(
            vertices, pose["uniform_scale"], pose["translation_world_m"][:2],
            pose["yaw_deg"], scale_xyz=pose.get("scale_xyz"),
            z=pose["translation_world_m"][2],
            roll_x_deg=pose.get("roll_x_deg", 0.0),
            roll_y_deg=pose.get("roll_y_deg", 0.0),
        )
        return render_mesh_mask(mesh, posed, view_camera)

    before_scale = rendered_mask(fitted, camera)
    if not np.any(before_scale) or not np.any(target):
        return fitted, {
            "frame_fit": frame_fit,
            "accepted": False,
            "reason": "empty_projected_or_target_mask",
        }
    source_front_height = _bbox_height(before_scale)
    target_front_height = _bbox_height(target)
    xyz_scale = target_front_height / source_front_height
    initialized = dict(fitted)
    scale_xyz = np.asarray(
        fitted.get("scale_xyz", [fitted["uniform_scale"]] * 3),
        dtype=float,
    )
    scale_xyz *= xyz_scale
    initialized["scale_xyz"] = scale_xyz.tolist()
    initialized["uniform_scale"] = float(scale_xyz[2])

    after_scale = rendered_mask(initialized, camera)
    pixel_delta = _bbox_bottom_center(target) - _bbox_bottom_center(after_scale)
    world_delta = _pixel_delta_to_world(
        pixel_delta, initialized["translation_world_m"][:2], camera,
    )
    initialized["translation_world_m"] = [
        float(initialized["translation_world_m"][0] + world_delta[0]),
        float(initialized["translation_world_m"][1] + world_delta[1]),
        float(initialized["translation_world_m"][2]),
    ]
    final_mask = rendered_mask(initialized, camera)
    return initialized, {
        "accepted": True,
        "frame_fit": frame_fit,
        "xyz_front_bbox_height_scale_ratio": float(xyz_scale),
        "scale_initialization_order": ["xyz_from_front_height_before_yaw_selection"],
        "bbox_bottom_center_pixel_delta": pixel_delta.tolist(),
        "source_front_bbox_height_px_before_xyz": source_front_height,
        "target_front_bbox_height_px": target_front_height,
        "final_bbox_bottom_center_error_px": (
            _bbox_bottom_center(final_mask) - _bbox_bottom_center(target)
        ).tolist(),
    }


def refine_selected_front_bbox_xy(
    mesh, item, target, camera, top_target=None, top_camera=None,
    xy_locked=False,
):
    """Fit local X/Y only after a yaw branch has been selected."""
    vertices = center_vertices_on_table(mesh.vertices)

    def rendered_mask(pose, view_camera):
        posed = pose_vertices(
            vertices, pose["uniform_scale"], pose["translation_world_m"][:2],
            pose["yaw_deg"], scale_xyz=pose.get("scale_xyz"),
            z=pose["translation_world_m"][2],
            roll_x_deg=pose.get("roll_x_deg", 0.0),
            roll_y_deg=pose.get("roll_y_deg", 0.0),
        )
        return render_mesh_mask(mesh, posed, view_camera)

    before_x = rendered_mask(item, camera)
    if not np.any(before_x) or not np.any(target):
        return dict(item), {
            "accepted": False,
            "reason": "empty_projected_or_target_mask",
        }
    source_front_width = _bbox_width(before_x)
    target_front_width = _bbox_width(target)
    x_scale = target_front_width / source_front_width
    refined = dict(item)
    scale_xyz = np.asarray(
        item.get("scale_xyz", [item["uniform_scale"]] * 3), dtype=float,
    )
    scale_xyz[0] *= x_scale
    if xy_locked:
        scale_xyz[1] *= x_scale
    refined["scale_xyz"] = scale_xyz.tolist()

    if not xy_locked and top_target is not None and top_camera is not None:
        after_x_top = rendered_mask(refined, top_camera)
        if not np.any(after_x_top) or not np.any(top_target):
            return dict(item), {
                "accepted": False,
                "reason": "empty_projected_or_top_target_mask",
            }
        source_top_length = _bbox_height(after_x_top)
        target_top_length = _bbox_height(top_target)
        y_scale = target_top_length / source_top_length
        scale_xyz[1] *= y_scale
        refined["scale_xyz"] = scale_xyz.tolist()
    else:
        source_top_length = target_top_length = None
        y_scale = 1.0

    after_xy = rendered_mask(refined, camera)
    pixel_delta = _bbox_bottom_center(target) - _bbox_bottom_center(after_xy)
    world_delta = _pixel_delta_to_world(
        pixel_delta, refined["translation_world_m"][:2], camera,
    )
    refined["translation_world_m"] = [
        float(refined["translation_world_m"][0] + world_delta[0]),
        float(refined["translation_world_m"][1] + world_delta[1]),
        float(refined["translation_world_m"][2]),
    ]
    final_mask = rendered_mask(refined, camera)
    return refined, {
        "accepted": True,
        "x_front_bbox_width_scale_ratio": float(x_scale),
        "y_top_bbox_length_scale_ratio": float(y_scale),
        "xy_scale_locked": bool(xy_locked),
        "y_scale_skipped_for_xy_lock": bool(xy_locked),
        "scale_initialization_order": [
            "x_from_front_width_after_yaw_selection",
            "y_from_top_length_after_yaw_selection_unless_xy_locked",
        ],
        "bbox_bottom_center_pixel_delta": pixel_delta.tolist(),
        "source_front_bbox_width_px_before_x": source_front_width,
        "target_front_bbox_width_px": target_front_width,
        "source_top_bbox_length_px_after_x": source_top_length,
        "target_top_bbox_length_px": target_top_length,
        "final_bbox_bottom_center_error_px": (
            _bbox_bottom_center(final_mask) - _bbox_bottom_center(target)
        ).tolist(),
    }


def initialize_front_bbox_yaw_branches(
    mesh, item, target, camera, yaw_offsets_deg=(0.0, 180.0),
):
    """Height-fit every yaw branch independently before direction selection."""
    vertices = center_vertices_on_table(mesh.vertices)
    pivot = bottom_center_pivot(vertices)
    scale_xyz = item.get("scale_xyz", [item["uniform_scale"]] * 3)
    branches = []
    for offset in yaw_offsets_deg:
        yaw_deg = float((item["yaw_deg"] + offset + 180.0) % 360.0 - 180.0)
        translation = translation_for_yaw_about_local_pivot(
            item["translation_world_m"][:2], pivot, scale_xyz,
            item["yaw_deg"], yaw_deg,
        )
        branch = {
            **item,
            "yaw_deg": yaw_deg,
            "translation_world_m": [
                float(translation[0]), float(translation[1]),
                float(item["translation_world_m"][2]),
            ],
        }
        fitted, report = initialize_front_bbox_pose(mesh, branch, target, camera)
        branches.append((
            math.radians(float(offset)), fitted,
            {**report, "initial_yaw_offset_deg": float(offset)},
        ))
    return tuple(branches)


def _rendered_model_edges(
    mask, normal_edges, rgb, model_edge_method, edge_detector,
):
    if model_edge_method == "normal_discontinuity":
        return normal_edges, {"method": "normal_discontinuity"}
    if model_edge_method not in ("teed", "reference_image"):
        raise ValueError(f"unknown model edge method: {model_edge_method}")
    if rgb is None:
        raise ValueError("rendered-RGB model edges require a textured model")
    return masked_edges(
        rgb, np.asarray(mask) > 0,
        method=(
            "teed" if model_edge_method == "teed"
            else reference_edge_detector(edge_detector)
        ),
    )


def _normalized_gradient_loss_gpu(source_rgb, target_rgb, source_mask, target_mask):
    import torch
    import torch.nn.functional as functional

    device = alignment_torch_device()
    support = torch.as_tensor(
        (np.asarray(source_mask) > 0) & (np.asarray(target_mask) > 0),
        device=device,
    )
    support = ~functional.max_pool2d(
        (~support)[None, None].to(torch.float32), 3, stride=1, padding=1,
    )[0, 0].bool()
    if int(support.sum()) < 16:
        return 1.0

    def gradients(image):
        rgb = torch.as_tensor(
            np.asarray(image, dtype=np.uint8).copy(),
            dtype=torch.float32, device=device,
        ).permute(2, 0, 1)[None]
        # Match cv2.COLOR_RGB2GRAY's fixed-point uint8 conversion before blur.
        gray = torch.round(
            (4899.0 * rgb[:, :1] + 9617.0 * rgb[:, 1:2] + 1868.0 * rgb[:, 2:])
            / 16384.0
        ) / 255.0
        kernel_x = torch.arange(-6, 7, dtype=torch.float32, device=device)
        gaussian = torch.exp(-0.5 * (kernel_x / 1.5).square())
        gaussian /= gaussian.sum()
        gaussian = gaussian.outer(gaussian)[None, None]
        gray = functional.conv2d(
            functional.pad(gray, (6, 6, 6, 6), mode="reflect"), gaussian,
        )
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=device,
        )[:, None]
        sobel_y = sobel_x.transpose(2, 3)
        gray = functional.pad(gray, (1, 1, 1, 1), mode="reflect")
        return torch.cat((
            functional.conv2d(gray, sobel_x), functional.conv2d(gray, sobel_y),
        ), dim=1)[0].permute(1, 2, 0)

    source = gradients(source_rgb)[support].reshape(-1)
    target = gradients(target_rgb)[support].reshape(-1)
    source = source - source.mean()
    target = target - target.mean()
    source_norm = torch.linalg.vector_norm(source)
    target_norm = torch.linalg.vector_norm(target)
    denominator = source_norm * target_norm
    if float(denominator) <= 1e-12:
        return 0.0 if torch.allclose(source_norm, target_norm) else 1.0
    correlation = torch.dot(source, target) / denominator
    return float((0.5 * (1.0 - correlation)).clamp(0.0, 1.0).cpu())


def normalized_gradient_loss(source_rgb, target_rgb, source_mask, target_mask):
    """Compare image structure while ignoring global brightness differences."""
    if alignment_torch_device().type == "cuda":
        return _normalized_gradient_loss_gpu(
            source_rgb, target_rgb, source_mask, target_mask,
        )
    support = (
        (np.asarray(source_mask) > 0)
        & (np.asarray(target_mask) > 0)
    ).astype(np.uint8)
    support = cv2.erode(support, np.ones((3, 3), dtype=np.uint8)) > 0
    if np.count_nonzero(support) < 16:
        return 1.0

    def gradients(image):
        gray = cv2.cvtColor(
            np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY,
        ).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
        return np.stack((
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        ), axis=-1)

    source = gradients(source_rgb)[support].reshape(-1)
    target = gradients(target_rgb)[support].reshape(-1)
    source -= source.mean()
    target -= target.mean()
    denominator = float(np.linalg.norm(source) * np.linalg.norm(target))
    if denominator <= 1e-12:
        return 0.0 if np.linalg.norm(source) == np.linalg.norm(target) else 1.0
    correlation = float(np.dot(source, target) / denominator)
    return float(np.clip(0.5 * (1.0 - correlation), 0.0, 1.0))


def precompute_front_direction_groups(
    objects, camera, labels, ids, target_rgb, target_mask_fn, scorer,
):
    if (
        scorer is None
        or not hasattr(scorer, "score_candidate_groups")
        or hasattr(scorer, "score_multiview_candidates")
    ):
        return {}, {}, {}
    start = time.perf_counter()
    groups = {}
    branch_cache = {}
    for object_id, item in objects.items():
        if object_id not in ids:
            continue
        mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=None)
        target = target_mask_fn(labels, ids, object_id)
        yaw_offsets = model_top_yaw_offsets(
            rendered_model_square_ratio(item["top_template_path"]),
        )
        branches = initialize_front_bbox_yaw_branches(
            mesh, item, target, camera, yaw_offsets_deg=yaw_offsets,
        )
        branch_cache[object_id] = branches
        colors = item_vertex_texture_colors(mesh, item)
        groups[object_id] = {
            "reference_key": "front",
            "reference_rgb": target_rgb,
            "reference_mask": target,
            "entries": [
                (float(yaw), mesh, branch, colors)
                for yaw, branch, _ in branches
            ],
        }
    rendered, scores = precompute_batched_direction_groups(
        groups, camera, scorer,
    )
    print(
        "[front timing] cross_object_direction_batch="
        f"{time.perf_counter() - start:.3f}s "
        f"objects={len(groups)} candidates="
        f"{sum(len(group['entries']) for group in groups.values())}",
        flush=True,
    )
    return rendered, scores, branch_cache


def precompute_front_candidate_metrics(
    objects, labels, ids, target_rgb, target_mask_fn, target_edges,
    direction_render_cache, *, edge_detector, model_edge_method,
    internal_orientation_weight, internal_edge_match_mode,
):
    """Preprocess every fixed front candidate, then batch its Chamfer losses."""
    started = time.perf_counter()
    prepared = {}
    sampling_items = []
    sampling_destinations = []
    for object_id in objects:
        candidates = direction_render_cache.get(object_id, {})
        if object_id not in ids or not candidates or not all(
            isinstance(candidate, dict) for candidate in candidates.values()
        ):
            continue
        first = next(iter(candidates.values()))
        roi = first.get("_render_roi")
        render_size = list(first["mask"].shape[::-1])
        full_target = target_mask_fn(labels, ids, object_id)
        full_appearance = filled_instance_appearance_mask(
            labels, ids, object_id,
        )
        target = camera_roi_array(full_target, roi, render_size, False)
        appearance = camera_roi_array(
            full_appearance, roi, render_size, False,
        )
        reference_rgb = camera_roi_array(
            target_rgb, roi, render_size, 238,
        )
        reference_edges = camera_roi_array(
            target_edges[object_id], roi, render_size, 0,
        )
        target_center, target_area = _mask_geometry(target)
        target_internal_edges = internal_edges(reference_edges, appearance)
        value = {
            "roi": roi,
            "render_size": render_size,
            "target": target,
            "appearance_target": appearance,
            "target_rgb": reference_rgb,
            "target_center": target_center,
            "target_area": target_area,
            "target_outer": _normalized_points(
                sample_external_contour(target, OT_POINT_COUNT),
                target_center, target_area,
            ),
            "target_internal_edges": target_internal_edges,
            "target_internal_edge_density": internal_edge_length_density(
                target_internal_edges, appearance,
            ),
            "candidates": {},
            "metrics": {},
        }
        sampling_items.append((target_internal_edges, appearance))
        sampling_destinations.append((value, "target"))
        for yaw, candidate in candidates.items():
            rendered = candidate["_rendered_buffers"]
            mask, edges, rgb = rendered[:3]
            if not np.any(mask):
                value["metrics"][float(yaw)] = (
                    1e6, 1e6, 1.0, 1.0, 0.0, 1e6,
                )
                continue
            edges, edge_metadata = _rendered_model_edges(
                mask, edges, rgb, model_edge_method, edge_detector,
            )
            center, area = _mask_geometry(mask)
            candidate_internal_edges = internal_edges(edges, mask)
            candidate_value = {
                "rendered": rendered,
                "mask": mask,
                "edges": edges,
                "rgb": rgb,
                "center": center,
                "area": area,
                "outer": _normalized_points(
                    sample_external_contour(mask, OT_POINT_COUNT), center, area,
                ),
                "internal_edges": candidate_internal_edges,
                "internal_edge_density": internal_edge_length_density(
                    candidate_internal_edges, mask,
                ),
                "edge_metadata": edge_metadata,
            }
            value["candidates"][float(yaw)] = candidate_value
            sampling_items.append((candidate_internal_edges, mask))
            sampling_destinations.append((candidate_value, "candidate"))
        prepared[object_id] = value

    if not prepared:
        return {}
    samples = sample_edge_features_batch(
        sampling_items, FRONT_YAW_EDGE_POINT_COUNT,
    )
    for (destination, kind), (points, tangents, confidence) in zip(
        sampling_destinations, samples,
    ):
        if kind == "target":
            center, area = destination["target_center"], destination["target_area"]
            destination["target_internal"] = _normalized_points(
                points, center, area,
            )
            destination["target_tangents"] = tangents
            destination["target_confidence"] = confidence
        else:
            destination["internal"] = _normalized_points(
                points, destination["center"], destination["area"],
            )
            destination["tangents"] = tangents
            destination["confidence"] = confidence

    outer_pairs, outer_destinations = [], []
    internal_pairs, internal_destinations = [], []
    for object_id, value in prepared.items():
        target_bool = value["target"] > 0
        for yaw, candidate in value["candidates"].items():
            outer_pairs.append((candidate["outer"], value["target_outer"]))
            outer_destinations.append((object_id, yaw))
            source_internal = candidate["internal"]
            target_internal = value["target_internal"]
            if not len(source_internal) or not len(target_internal):
                candidate["internal_components"] = (0.0, 0.0)
            elif internal_edge_match_mode == "centroid_offset":
                candidate["internal_components"] = (
                    _internal_centroid_offset_loss(
                        source_internal, target_internal, 0.0,
                    ),
                    0.0,
                )
            else:
                use_orientation = bool(
                    internal_orientation_weight > 0.0
                    and np.count_nonzero(candidate["confidence"])
                    and np.count_nonzero(value["target_confidence"])
                )
                internal_pairs.append(
                    (
                        source_internal, target_internal,
                        candidate["tangents"], value["target_tangents"],
                        candidate["confidence"], value["target_confidence"],
                    )
                    if use_orientation else
                    (source_internal, target_internal)
                )
                internal_destinations.append((object_id, yaw))
            mask = candidate["mask"] > 0
            union = np.count_nonzero(mask | target_bool)
            candidate["iou"] = np.count_nonzero(mask & target_bool) / max(union, 1)
            candidate["rgb_loss"] = (
                normalized_gradient_loss(
                    candidate["rgb"], value["target_rgb"], mask,
                    value["appearance_target"] > 0,
                )
                if candidate["rgb"] is not None else 0.0
            )

    for destination, result in zip(
        outer_destinations, _batched_zero_yaw_components(outer_pairs),
    ):
        prepared[destination[0]]["candidates"][destination[1]][
            "outer_loss"
        ] = result[0]
    tail_fraction = 0.10 if internal_edge_match_mode == "partial_hausdorff" else 0.0
    for destination, result in zip(
        internal_destinations,
        _batched_zero_yaw_components(
            internal_pairs, source_tail_fraction=tail_fraction,
        ),
    ):
        prepared[destination[0]]["candidates"][destination[1]][
            "internal_components"
        ] = result
    for value in prepared.values():
        for yaw, candidate in value["candidates"].items():
            internal_position, internal_direction = candidate[
                "internal_components"
            ]
            value["metrics"][yaw] = (
                float(candidate["outer_loss"]), float(internal_position),
                float(internal_direction), float(candidate["rgb_loss"]),
                float(candidate["iou"]),
                internal_edge_count_loss(
                    candidate["internal_edge_density"],
                    value["target_internal_edge_density"],
                ),
            )
    print(
        "[front timing] candidate_geometry_batch="
        f"{time.perf_counter() - started:.3f}s "
        f"objects={len(prepared)} candidates="
        f"{sum(len(value['candidates']) for value in prepared.values())}",
        flush=True,
    )
    return prepared


def initialize_front_bbox_and_normal_yaw(
    objects, camera, labels, ids, target_rgb, config, *,
    top_camera=None,
    top_target_masks=None,
    top_target_rgb=None,
    tie_xy_scale_ids=(),
    internal_orientation_weight=1.0,
    internal_edge_match_mode="chamfer",
    edge_detector="teed",
    model_edge_method="normal_discontinuity",
    target_edges=None,
    target_mask_fn=instance_mask,
    direction_scorer=None,
    direction_render_cache=None,
    direction_score_cache=None,
    branch_candidate_cache=None,
):
    """Run deterministic bbox initialization and stage-2-style yaw search."""
    tie_xy_scale_ids = frozenset(tie_xy_scale_ids)
    direction_render_cache = direction_render_cache or {}
    direction_score_cache = direction_score_cache or {}
    branch_candidate_cache = branch_candidate_cache or {}
    if target_edges is None:
        target_edges = edge_drawing_silhouette_targets(
            target_rgb, labels, ids,
            tuple(object_id for object_id in objects if object_id in ids),
            include_internal=True,
            edge_detector=edge_detector,
        )
    precomputed_metrics = precompute_front_candidate_metrics(
        objects, labels, ids, target_rgb, target_mask_fn, target_edges,
        direction_render_cache,
        edge_detector=edge_detector,
        model_edge_method=model_edge_method,
        internal_orientation_weight=internal_orientation_weight,
        internal_edge_match_mode=internal_edge_match_mode,
    ) if direction_render_cache else {}
    initialized = {}
    reports = {}
    for object_id, item in objects.items():
        object_start = time.perf_counter()
        if object_id not in ids:
            initialized[object_id] = item
            continue
        mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=None)
        full_target = target_mask_fn(labels, ids, object_id)
        full_appearance_target = filled_instance_appearance_mask(
            labels, ids, object_id,
        )
        model_top_square_ratio = rendered_model_square_ratio(
            item["top_template_path"],
        )
        yaw_offsets_deg = model_top_yaw_offsets(model_top_square_ratio)
        branch_candidates = branch_candidate_cache.get(object_id)
        if branch_candidates is None:
            branch_candidates = initialize_front_bbox_yaw_branches(
                mesh, item, full_target, camera,
                yaw_offsets_deg=yaw_offsets_deg,
            )
        initial_yaw_key = float(branch_candidates[0][0])
        object_render_cache = direction_render_cache.get(object_id, {})
        object_score_cache = direction_score_cache.get(object_id, {})
        initial_cached = object_render_cache.get(initial_yaw_key)
        render_roi = (
            initial_cached.get("_render_roi")
            if isinstance(initial_cached, dict) else None
        )
        render_size = (
            list(initial_cached["mask"].shape[::-1])
            if isinstance(initial_cached, dict) else camera["image_size"]
        )
        target = (
            camera_roi_array(full_target, render_roi, render_size, False)
            if render_roi is not None else full_target
        )
        appearance_target = (
            camera_roi_array(
                full_appearance_target, render_roi, render_size, False,
            )
            if render_roi is not None else full_appearance_target
        )
        metric_target_rgb = (
            camera_roi_array(target_rgb, render_roi, render_size, 238)
            if render_roi is not None else target_rgb
        )
        metric_target_edges = (
            camera_roi_array(
                target_edges[object_id], render_roi, render_size, 0,
            )
            if render_roi is not None else target_edges[object_id]
        )
        _, candidate, bbox_report = branch_candidates[0]
        vertices = center_vertices_on_table(mesh.vertices)
        candidate_scale_xyz = candidate.get(
            "scale_xyz", [candidate["uniform_scale"]] * 3,
        )

        candidate_local = pose_vertices(
            vertices, candidate["uniform_scale"], (0.0, 0.0), 0.0,
            scale_xyz=candidate_scale_xyz,
            z=0.0,
            roll_x_deg=candidate.get("roll_x_deg", 0.0),
            roll_y_deg=candidate.get("roll_y_deg", 0.0),
        )
        posed = pose_vertices(
            candidate_local, 1.0,
            candidate["translation_world_m"][:2], candidate["yaw_deg"],
            z=candidate["translation_world_m"][2],
        )
        colors = item_vertex_texture_colors(mesh, candidate)
        needs_coordinates = getattr(
            direction_scorer, "requires_object_coordinates", False,
        )
        initial_rendered = initial_cached
        if isinstance(initial_rendered, dict):
            initial_rendered = initial_rendered["_rendered_buffers"]
        if initial_rendered is None:
            initial_rendered = render_normal_edge_buffers_nvdiffrast(
                posed, mesh.faces, camera, vertex_rgb=colors,
                return_normal_map=direction_scorer is not None,
                vertex_coordinates=(candidate_local if needs_coordinates else None),
            )
        source_mask, source_edges, source_rgb = initial_rendered[:3]
        precomputed = precomputed_metrics.get(object_id)
        precomputed_source = (
            precomputed["candidates"].get(initial_yaw_key)
            if precomputed is not None else None
        )
        if precomputed_source is not None:
            source_edges = precomputed_source["edges"]
            source_edge_metadata = precomputed_source["edge_metadata"]
        else:
            source_edges, source_edge_metadata = _rendered_model_edges(
                source_mask, source_edges, source_rgb,
                model_edge_method, edge_detector,
            )
        if not np.any(source_mask) or not np.any(target):
            initialized[object_id] = candidate
            reports[object_id] = {
                "bbox_initialization": bbox_report,
                "yaw_search": {"accepted": False, "reason": "empty_mask"},
            }
            print(
                f"[front timing initializer] {object_id}="
                f"{time.perf_counter() - object_start:.3f}s empty_mask",
                flush=True,
            )
            continue
        if precomputed_source is not None:
            source_center = precomputed_source["center"]
            source_area = precomputed_source["area"]
            source_outer = precomputed_source["outer"]
            source_internal_edges = precomputed_source["internal_edges"]
            source_internal = precomputed_source["internal"]
            source_tangents = precomputed_source["tangents"]
            source_confidence = precomputed_source["confidence"]
            target_center = precomputed["target_center"]
            target_area = precomputed["target_area"]
            target_outer = precomputed["target_outer"]
            target_internal_edges = precomputed["target_internal_edges"]
            target_internal_edge_density = precomputed[
                "target_internal_edge_density"
            ]
            target_internal = precomputed["target_internal"]
            target_tangents = precomputed["target_tangents"]
            target_confidence = precomputed["target_confidence"]
        else:
            source_center, source_area = _mask_geometry(source_mask)
            target_center, target_area = _mask_geometry(target)
            source_outer = _normalized_points(
                sample_external_contour(source_mask, OT_POINT_COUNT),
                source_center, source_area,
            )
            target_outer = _normalized_points(
                sample_external_contour(target, OT_POINT_COUNT),
                target_center, target_area,
            )
            source_internal_edges = internal_edges(source_edges, source_mask)
            target_internal_edges = internal_edges(
                metric_target_edges, appearance_target,
            )
            target_internal_edge_density = internal_edge_length_density(
                target_internal_edges, appearance_target,
            )
            source_internal_pixels, source_tangents, source_confidence = (
                sample_edge_features(
                    source_internal_edges, FRONT_YAW_EDGE_POINT_COUNT,
                    sampling_mask=source_mask,
                )
            )
            target_internal_pixels, target_tangents, target_confidence = (
                sample_edge_features(
                    target_internal_edges, FRONT_YAW_EDGE_POINT_COUNT,
                    sampling_mask=appearance_target,
                )
            )
            source_internal = _normalized_points(
                source_internal_pixels, source_center, source_area,
            )
            target_internal = _normalized_points(
                target_internal_pixels, target_center, target_area,
            )

        metric_cache = dict(precomputed["metrics"]) if precomputed else {}
        direction_scores = {}
        uses_multiview_direction = bool(
            direction_scorer is not None
            and hasattr(direction_scorer, "score_multiview_candidates")
        )
        direction_inputs = (
            {"front": {}, "top": {}}
            if uses_multiview_direction else {}
        )
        def true_3d_metrics(yaw):
            key = float(yaw)
            if key in metric_cache:
                return metric_cache[key]
            _, branch, _ = min(
                branch_candidates,
                key=lambda value: abs(
                    (value[0] - key + math.pi) % (2.0 * math.pi) - math.pi
                ),
            )
            branch_scale_xyz = branch.get(
                "scale_xyz", [branch["uniform_scale"]] * 3,
            )
            candidate_local = pose_vertices(
                vertices, branch["uniform_scale"], (0.0, 0.0), 0.0,
                scale_xyz=branch_scale_xyz,
            )
            candidate_vertices = pose_vertices(
                candidate_local, 1.0, branch["translation_world_m"][:2],
                branch["yaw_deg"],
            )
            rendered = object_render_cache.get(key)
            if isinstance(rendered, dict):
                rendered = rendered["_rendered_buffers"]
            if rendered is None:
                rendered = (
                    initial_rendered
                    if math.isclose(key, initial_yaw_key) else
                    render_normal_edge_buffers_nvdiffrast(
                    candidate_vertices, mesh.faces, camera, vertex_rgb=colors,
                    return_normal_map=direction_scorer is not None,
                    vertex_coordinates=(
                        candidate_local if needs_coordinates else None
                    ),
                    )
                )
            rendered_mask, rendered_edges, rendered_rgb = rendered[:3]
            if not np.any(rendered_mask):
                metric_cache[key] = (1e6, 1e6, 1.0, 1.0, 0.0, 1e6)
                if direction_scorer is not None:
                    direction_scores[key] = {
                        "loss": 1.0,
                        "foreground_mean_confidence": 0.0,
                        "valid_correspondence_count": 0,
                        "structural_consistency_score": 0.0,
                        "structural_cycle_score": 0.0,
                        "structural_ransac_score": 0.0,
                        "inference_seconds": 0.0,
                        "candidate_modality": (
                            "rgb" if colors is not None else "normal_fallback"
                        ),
                        "backend": getattr(
                            direction_scorer, "backend", "direction_loss",
                        ),
                        "selection_source": "empty_render_penalty",
                    }
                return metric_cache[key]
            if direction_scorer is not None:
                front_direction_input = {
                    "mask": rendered_mask,
                    "rgb": rendered_rgb if rendered_rgb is not None else rendered[3],
                    "normal_map": rendered[3],
                    "candidate_modality": (
                        "rgb" if rendered_rgb is not None else "normal_fallback"
                    ),
                }
                if needs_coordinates:
                    front_direction_input["object_coordinates"] = rendered[4]
                if uses_multiview_direction:
                    direction_inputs["front"][key] = front_direction_input
                    direction_inputs["top"][key] = rendered_direction_candidate(
                        mesh, branch, top_camera, colors, direction_scorer,
                    )
                else:
                    direction_inputs[key] = front_direction_input
            rendered_edges, _ = _rendered_model_edges(
                rendered_mask, rendered_edges, rendered_rgb,
                model_edge_method, edge_detector,
            )
            rendered_center, rendered_area = _mask_geometry(rendered_mask)
            rendered_outer = _normalized_points(
                sample_external_contour(rendered_mask, OT_POINT_COUNT),
                rendered_center, rendered_area,
            )
            rendered_internal_edges = internal_edges(
                rendered_edges, rendered_mask,
            )
            rendered_internal_pixels, rendered_tangents, rendered_confidence = (
                sample_edge_features(
                    rendered_internal_edges, FRONT_YAW_EDGE_POINT_COUNT,
                    sampling_mask=rendered_mask,
                )
            )
            rendered_internal = _normalized_points(
                rendered_internal_pixels, rendered_center, rendered_area,
            )
            outer_loss = _combined_yaw_chamfer(
                [(rendered_outer, target_outer, 1.0)], 0.0,
            )
            if not len(rendered_internal) or not len(target_internal):
                internal_position_loss = internal_direction_loss = 0.0
            elif internal_edge_match_mode == "centroid_offset":
                internal_position_loss = _internal_centroid_offset_loss(
                    rendered_internal, target_internal, 0.0,
                )
                internal_direction_loss = 0.0
            elif (
                internal_orientation_weight > 0.0
                and np.count_nonzero(rendered_confidence)
                and np.count_nonzero(target_confidence)
            ):
                internal_position_loss, internal_direction_loss = (
                    _oriented_yaw_components(
                    rendered_internal, target_internal,
                    rendered_tangents, target_tangents,
                    rendered_confidence, target_confidence,
                    0.0,
                    source_tail_fraction=(
                        0.10
                        if internal_edge_match_mode == "partial_hausdorff"
                        else 0.0
                    ),
                    )
                )
            else:
                internal_position_loss = _combined_yaw_chamfer(
                    [(rendered_internal, target_internal, 1.0)], 0.0,
                    source_tail_fraction=(
                        0.10
                        if internal_edge_match_mode == "partial_hausdorff"
                        else 0.0
                    ),
                )
                internal_direction_loss = 0.0
            target_bool = np.asarray(target) > 0
            union = np.count_nonzero(rendered_mask | target_bool)
            iou = np.count_nonzero(rendered_mask & target_bool) / max(union, 1)
            rgb_loss = (
                normalized_gradient_loss(
                    rendered_rgb, metric_target_rgb, rendered_mask,
                    appearance_target > 0,
                )
                if rendered_rgb is not None
                else 0.0
            )
            metric_cache[key] = (
                float(outer_loss), float(internal_position_loss),
                float(internal_direction_loss),
                float(rgb_loss), float(iou),
                internal_edge_count_loss(
                    internal_edge_length_density(
                        rendered_internal_edges, rendered_mask,
                    ),
                    target_internal_edge_density,
                ),
            )
            return metric_cache[key]

        coarse_candidates = tuple(map(math.radians, yaw_offsets_deg))
        pca_report = {
            "used": False,
            "fallback": (
                "true_3d_0_90_180_270_direction_branches"
                if len(coarse_candidates) == 4 else
                "true_3d_0_180_direction_branches"
            ),
            "candidate_count": len(coarse_candidates),
            "model_top_square_ratio": float(model_top_square_ratio),
            "model_top_square_ratio_threshold": (
                FRONT_FOUR_WAY_SQUARE_RATIO_THRESHOLD
            ),
            "candidate_offsets_deg": list(yaw_offsets_deg),
        }
        top_yaw_report = item.get(
            "top_differentiable_edge_refinement", {},
        ).get("pca_direction_yaw", {})
        stage2_yaw_losses = (
            None if uses_multiview_direction or not getattr(
                direction_scorer, "stage3_uses_stage2_prior", True,
            ) else
            stage2_yaw_loss_multipliers(top_yaw_report, yaw_offsets_deg)
        )
        direction_loss_fn = None
        if direction_scorer is not None:
            def direction_loss_fn(yaw):
                return direction_scores[float(yaw)]["loss"]

            def prepare_direction_candidates(yaws):
                for yaw in yaws:
                    true_3d_metrics(yaw)
                if object_score_cache:
                    direction_scores.update({
                        float(yaw): object_score_cache[float(yaw)]
                        for yaw in yaws if float(yaw) in object_score_cache
                    })
                    return
                if uses_multiview_direction:
                    active = {
                        view: {
                            float(yaw): direction_inputs[view][float(yaw)]
                            for yaw in yaws
                            if float(yaw) in direction_inputs[view]
                        }
                        for view in ("front", "top")
                    }
                    direction_scores.update(score_multiview_direction_candidates(
                        direction_scorer, "front",
                        {"front": metric_target_rgb, "top": top_target_rgb},
                        {
                            "front": target,
                            "top": top_target_masks[object_id],
                        },
                        {"front": camera, "top": top_camera},
                        active,
                    ))
                else:
                    active = {
                        float(yaw): direction_inputs[float(yaw)]
                        for yaw in yaws if float(yaw) in direction_inputs
                    }
                    direction_scores.update(score_direction_candidates(
                        direction_scorer, "front", metric_target_rgb, target,
                        camera, active,
                    ))

            direction_loss_fn.prepare = prepare_direction_candidates
            direction_loss_fn.backend = (
                f"{getattr(direction_scorer, 'backend', 'dino_relative_position')}_"
                f"{'front_top' if uses_multiview_direction else 'front'}"
            )
            direction_loss_fn.loss_formula = getattr(
                direction_scorer, "loss_formula",
                "DINO_relative_position_consistency",
            )
        yaw, yaw_report = optimize_2d_edge_yaw_coarse_multistart(
            [
                (source_outer, target_outer, 0.5),
                (source_internal, target_internal, 0.5),
            ],
            _normalized_points(
                _sample_mask_points(source_mask, OT_POINT_COUNT),
                source_center, source_area,
            ),
            _normalized_points(
                _sample_mask_points(target, OT_POINT_COUNT),
                target_center, target_area,
            ),
            iterations=config.iterations,
            lr_yaw=config.lr_yaw,
            early_stop_patience=config.patience,
            early_stop_min_delta=config.early_stop_min_delta,
            rgb_loss_fn=(
                (lambda angle: true_3d_metrics(angle)[3])
                if source_rgb is not None else None
            ),
            mask_iou_fn=lambda angle: true_3d_metrics(angle)[4],
            coarse_yaw_candidates=coarse_candidates,
            internal_orientation_features=(
                source_tangents, target_tangents,
                source_confidence, target_confidence,
            ),
            internal_orientation_weight=internal_orientation_weight,
            internal_edge_match_mode=internal_edge_match_mode,
            coarse_metrics_fn=true_3d_metrics,
            rank_shape_component="internal",
            rank_rgb_component_weight=1.0,
            model_mask_information=model_mask_rotational_information(source_mask),
            coarse_loss_multipliers=(
                stage2_yaw_losses
            ),
            direction_loss_fn=direction_loss_fn,
        )
        yaw_report["pca_gate"] = pca_report
        yaw_report["rgb_loss_type"] = "normalized_sobel_gradient_correlation"
        if direction_scores:
            yaw_report["direction_candidates"] = [
                {"yaw_deg": float(math.degrees(angle)), **score}
                for angle, score in direction_scores.items()
            ]
        if stage2_yaw_losses is not None:
            yaw_report["stage2_direction_loss_prior"] = {
                "candidate_offsets_deg": list(yaw_offsets_deg),
                "loss_multiplier": list(stage2_yaw_losses),
                "selection_score": (
                    "front_direction_loss*stage2_direction_loss"
                    if direction_scorer is not None else
                    "front_internal_rgb_loss*stage2_yaw_loss"
                ),
            }
            if len(stage2_yaw_losses) == 2:
                yaw_report["stage2_direction_loss_prior"].update({
                    "current_direction_loss": stage2_yaw_losses[0],
                    "opposite_direction_loss": stage2_yaw_losses[1],
                })
        selected_offset, candidate, bbox_report = min(
            branch_candidates,
            key=lambda value: abs(
                (value[0] - yaw + math.pi) % (2.0 * math.pi) - math.pi
            ),
        )
        candidate, xy_report = refine_selected_front_bbox_xy(
            mesh, candidate, full_target, camera,
            top_target=(top_target_masks or {}).get(object_id),
            top_camera=top_camera,
            xy_locked=object_id in tie_xy_scale_ids,
        )
        initialized[object_id] = candidate
        reports[object_id] = {
            "bbox_initialization": {
                "height_xyz_before_yaw_selection": bbox_report,
                "xy_after_yaw_selection": xy_report,
            },
            "bbox_initialization_candidates": {
                f"{math.degrees(offset):g}": report
                for offset, _, report in branch_candidates
            },
            "selected_bbox_yaw_offset_deg": float(math.degrees(selected_offset)),
            "yaw_search": yaw_report,
            "model_edge_method": source_edge_metadata["method"],
        }
        print(
            f"[front timing initializer] {object_id}="
            f"{time.perf_counter() - object_start:.3f}s "
            f"candidates={len(coarse_candidates)}",
            flush=True,
        )
    return initialized, {
        "method": (
            "per_yaw_branch_fit_frame_front_height_xyz_bbox_bottom_xy_"
            "then_stage2_coarse_model_edge_yaw_then_selected_branch_"
            "front_width_x_top_length_y_bbox_bottom_xy"
        ),
        "objects": reports,
    }


def run(args):
    global ALIGNMENT_COMPUTE_MODE, _blender_scene_mode
    global OBJECT_IDS, ASSET_PATHS, TABLE_SIZE_M
    ALIGNMENT_COMPUTE_MODE = "gpu"
    _blender_scene_mode = args.blender_scene_mode
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[alignment] stage-2/3 matching device={alignment_torch_device().type}",
        flush=True,
    )
    print("[1/3] calibrating table/cameras and aligning rigid objects", flush=True)
    front_labels, front_ids = load_scene_label_map(ROOT / "data/segmentation_results.json", ROOT)
    top_labels, top_ids = load_scene_label_map(ROOT / "data/topview_segmentation_results.json", ROOT)
    front_manifest = json.loads(
        (ROOT / "data/segmentation_results.json").read_text(encoding="utf-8"),
    )
    front_tabletop_mask_path = ROOT / "data/front_tabletop_mask.png"
    if not front_tabletop_mask_path.exists():
        front_tabletop_mask_path = Path(front_manifest["tabletop_mask_path"])
    front_tabletop_mask = np.asarray(
        Image.open(front_tabletop_mask_path).convert("L"),
    ) > 127
    blueprint = json.loads((ROOT / "data/blueprint.json").read_text(encoding="utf-8"))
    image_to_3d = json.loads(
        (ROOT / "data/image_to_3d_manifest.json").read_text(encoding="utf-8")
    )
    ASSET_PATHS = {
        item["object_id"]: Path(item["mesh"]).resolve()
        for item in image_to_3d["items"]
    }
    explicitly_skipped = {
        item["object_id"] for item in image_to_3d.get("skipped_items", [])
        if item.get("reason") == "articulated_generation_disabled"
    }
    if "table_0" not in ASSET_PATHS:
        raise ValueError("image-to-3D manifest is missing object: table_0")
    missing = [
        item["object_id"] for item in blueprint["objects"]
        if item["object_id"] not in ASSET_PATHS
    ]
    unexpected_missing = [
        object_id for object_id in missing
        if object_id not in explicitly_skipped
    ]
    if unexpected_missing:
        raise ValueError(
            f"image-to-3D manifest is missing object: {unexpected_missing[0]}"
        )
    skipped_ids = [
        object_id for object_id in missing if object_id in explicitly_skipped
    ]
    if skipped_ids:
        print(
            "[alignment] skipping objects absent from image-to-3D manifest: "
            + ", ".join(skipped_ids),
            flush=True,
        )
    retained_ids = {"table_0", *ASSET_PATHS}
    retained_objects = []
    for item in blueprint["objects"]:
        if item["object_id"] not in ASSET_PATHS:
            continue
        item = dict(item)
        placement = dict(item.get("placement") or {})
        parents = [
            parent for parent in placement.get("support_parents", [])
            if parent in retained_ids
        ]
        if placement.get("support_parents") and not parents:
            parents = ["table_0"]
        if parents:
            placement["support_parents"] = parents
            item["placement"] = placement
        retained_objects.append(item)
    blueprint = {**blueprint, "objects": retained_objects}
    OBJECT_IDS = tuple(item["object_id"] for item in blueprint["objects"])
    dimensions = blueprint["table"].get("dimensions_cm", {})
    bbox = blueprint["table"].get("bbox_cm", blueprint["table"].get("size_cm"))
    TABLE_SIZE_M = tuple(
        float(value) / 100.0
        for value in (
            dimensions.get("width_x", bbox[0]),
            dimensions.get("depth_y", bbox[1]),
            dimensions.get("height_z", bbox[2]),
        )
    )
    orientation_manifest_path = ROOT / "data/mesh_orientation_manifest.json"
    precomputed_orientation_reports = None
    precomputed_top_templates = {}
    precomputed_table_orientation = {}
    if orientation_manifest_path.is_file():
        precomputed_table_orientation = dict(
            json.loads(orientation_manifest_path.read_text(encoding="utf-8")).get(
                "table",
            ) or {},
        )
        (
            specs, precomputed_orientation_reports, precomputed_top_templates,
        ) = precomputed_mesh_orientation(
            orientation_manifest_path, blueprint, args.asset_source,
        )
    else:
        specs = alignment_object_specs(blueprint, asset_source=args.asset_source)
    table_mesh = two_camera_asset("table_0")
    top_table = instance_geometry(top_labels, top_ids["table_0"])

    top_view = load_vggt_view(ROOT / "outputs/vggt_top_only", top_labels, top_ids)
    raw_top_k = np.asarray(top_view["camera"]["intrinsic"])
    bbox = top_table["bbox_xyxy"]
    table_bbox_center = [
        0.5 * (bbox[0] + bbox[2]),
        0.5 * (bbox[1] + bbox[3]),
    ]
    top_camera = vertical_top_camera(
        raw_top_k,
        top_view["camera"]["image_size"],
        table_bbox_center,
        TABLE_SIZE_M[0],
        bbox[2] - bbox[0] + 1,
    )
    front_view = load_vggt_view(
        ROOT / "outputs/vggt_front_only", front_labels, front_ids,
        support_mask=front_tabletop_mask,
    )
    top_depth_m = (bbox[3] - bbox[1] + 1) * top_camera["meters_per_pixel"]
    if "yaw_deg" in precomputed_table_orientation:
        table_yaw_deg = float(precomputed_table_orientation["yaw_deg"])
        table_orientation_report = dict(
            precomputed_table_orientation.get("report") or {
                "method": "precomputed_cardinal_table_yaw",
                "selected_yaw_deg": table_yaw_deg,
            },
        )
    else:
        table_yaw_deg = 0.0
        table_orientation_report = None
    initial_table_scale = table_scale_for_dimensions(
        mesh_vertices_z_up(table_mesh), table_yaw_deg,
        (TABLE_SIZE_M[0], top_depth_m, TABLE_SIZE_M[2]),
    )
    print(
        f"[table alignment] table_0: yaw={table_yaw_deg:.0f}deg "
        f"source={'precomputed' if table_orientation_report else 'provisional'}",
        flush=True,
    )
    front_view = refine_front_table_registration(
        front_view, table_mesh, initial_table_scale,
        table_yaw_deg=table_yaw_deg,
    )
    registered_front_camera = {
        **front_view["camera"],
        "name": "front_camera",
        "source": "front-only VGGT registered to table",
        "fixed": True,
    }
    preliminary_table_scale, _ = table_alignment(
        table_mesh, top_table, top_camera, registered_front_camera,
        front_labels, front_ids, table_yaw_deg=table_yaw_deg,
    )
    front_view["camera"], pitch_report = refine_front_camera_pitch_roll_distance(
        front_view["camera"], table_mesh, preliminary_table_scale,
        front_labels, front_ids,
        tabletop_plane_mask=front_tabletop_mask,
        table_points_vggt=front_view["observed_table_vggt"],
        vggt_extrinsic=front_view["extrinsic"],
        metric_scale=front_view["metric_scale"],
        pointcloud_loss_weight=float(os.environ.get(
            "TABLE_POINTCLOUD_LOSS_WEIGHT", "1.0",
        )),
        silhouette_loss_weight=float(os.environ.get(
            "TABLE_SILHOUETTE_LOSS_WEIGHT", "0.0",
        )),
        table_yaw_deg=table_yaw_deg,
    )
    apply_camera_registration_to_view(front_view)
    print(
        "[front camera registration] "
        f"delta={pitch_report['pitch_delta_deg']:.3f}deg "
        f"roll={pitch_report.get('roll_delta_deg', 0.0):.3f}deg "
        f"world_yaw={pitch_report.get('world_yaw_delta_deg', 0.0):.3f}deg "
        f"translation_xyz={pitch_report.get('translation_delta_world_m', [0.0] * 3)} "
        f"loss={pitch_report['baseline_loss']:.3f}->{pitch_report['candidate_loss']:.3f} "
        f"iou={pitch_report['baseline_iou']:.4f}->{pitch_report['candidate_iou']:.4f} "
        f"accepted={pitch_report['accepted']}",
        flush=True,
    )
    front_view["camera"]["table_silhouette_camera_refinement"] = pitch_report
    front_camera = {
        **front_view["camera"],
        "name": "front_camera",
        "source": "front-only VGGT registered to table and pitch-roll-distance-refined to its silhouette",
        "fixed": True,
    }

    table_scale, table_loss = table_alignment(
        table_mesh, top_table, top_camera, front_camera, front_labels, front_ids,
        table_yaw_deg=table_yaw_deg,
    )
    table_scale, table_scale_pivot, top_camera, table_depth_report = (
        refine_table_depth_from_upper_edge(
            table_mesh, table_scale, top_table,
            front_camera, front_labels, front_ids,
            top_camera, top_labels, top_ids,
            front_tabletop_mask=front_tabletop_mask,
            table_yaw_deg=table_yaw_deg,
        )
    )
    table_scale, table_scale_pivot, table_height_report = (
        refine_table_height_from_bottom_edge(
            table_mesh, table_scale, table_scale_pivot,
            front_camera, front_labels, front_ids,
            table_yaw_deg=table_yaw_deg,
        )
    )
    if table_orientation_report is None:
        table_yaw_deg, table_scale, table_orientation_report = select_table_yaw(
            table_mesh, front_camera,
            np.asarray(Image.open(ROOT / "data/reference_image.png").convert("RGB")),
            front_labels, front_ids, front_tabletop_mask,
            table_scale, table_scale_pivot,
            aligned_yaw_deg=0.0,
        )
    print(
        f"[mesh orientation] table_0: yaw={table_yaw_deg:.0f}deg "
        f"source={table_orientation_report['method']}",
        flush=True,
    )
    cameras = {"top": top_camera, "front": front_camera}
    top_refinement_camera = vertical_top_camera(
        raw_top_k,
        top_view["camera"]["image_size"],
        table_bbox_center,
        TABLE_SIZE_M[0],
        bbox[2] - bbox[0] + 1,
        projection=args.top_refinement_camera_projection,
    )
    top_refinement_camera["center_world_m"][1] += table_depth_report.get(
        "top_camera_center_y_delta_m", 0.0,
    )
    write_json(
        output / "01_table_cameras.json",
        {
            "top": {**camera_json(top_view), **top_camera},
            "front": camera_json(front_view),
        },
    )
    table_scene = scene_document(
        table_mesh, table_scale, cameras, {},
        table_scale_pivot=table_scale_pivot,
        table_yaw_deg=table_yaw_deg,
    )
    table_scene["table"]["mesh_orientation"] = table_orientation_report
    table_scene["table"]["front_bbox_error_px"] = table_loss
    table_scene["table"]["depth_upper_edge_refinement"] = table_depth_report
    table_scene["table"]["final_height_refinement"] = table_height_report
    write_json(output / "02_table_alignment.json", table_scene)
    if args.dry_run:
        print(json.dumps({"status": "validated", "output": str(output), "objects": list(OBJECT_IDS)}, indent=2))
        print_output_paths(output, planned=True)
        return
    if args.stop_after_table:
        render_stage(
            output / "02_table_alignment.json", output / "02_table_alignment_render",
            save_blend=True,
        )
        compose_stage_comparison(
            output / "02_table_alignment_render",
            output / "02_table_alignment_comparison.png",
            "stage 1 intermediate: table/cameras",
        )
        print_output_paths(output, names=(
            "table_front", "table_top", "table_front_overlay", "table_top_overlay",
            "table_front_render_edges", "table_top_render_edges", "table_comparison",
        ))
        return

    print(
        "[1/3] aligning rigid object point clouds with GPU refinement",
        flush=True,
    )
    orientation_reference_rgbs = {
        "front": np.asarray(
            Image.open(ROOT / "data/reference_image.png").convert("RGB"),
        ),
        "top": np.asarray(
            Image.open(ROOT / "data/topview_image.png").convert("RGB"),
        ),
    }
    from psgsr.dino_loss import DinoRelativePositionDirectionScorer

    direction_scorer = DinoRelativePositionDirectionScorer(
        args.minima_root, device=args.minima_device,
    )
    orientation_scorer = direction_scorer
    print(
        "[direction loss] shared-reference DINO relative-position loss; "
        "stage 2 uses top only, stage 3 uses front only; "
        f"root={args.minima_root}",
        flush=True,
    )
    orientation_scorer = mesh_orientation_scorer_for_run(
        direction_scorer, precomputed_orientation_reports, args.minima_device,
    )
    print(
        "[mesh orientation] using precomputed assets"
        if precomputed_orientation_reports is not None else
        "[mesh orientation] front-primary/top-reference scoring with "
        f"{getattr(orientation_scorer, 'backend', 'dino_relative_position')}",
        flush=True,
    )
    if (
        direction_scorer is not None
        and hasattr(direction_scorer, "preload_async")
        and not args.stop_after_mesh_orientation
    ):
        direction_scorer.preload_async()
    blender_warmup_executor = None
    blender_warmup_future = None
    if not args.stop_after_mesh_orientation:
        blender_warmup_executor = ThreadPoolExecutor(max_workers=1)
        blender_warmup_future = blender_warmup_executor.submit(
            _persistent_blender_worker,
        )
    coarse_objects = {}
    coarse_inputs = []
    orientation_inputs = []
    blueprint_objects = {
        item["object_id"]: item for item in blueprint["objects"]
    }
    for object_id in OBJECT_IDS:
        spec = specs[object_id]
        mesh_path = Path(spec["geometry_asset"])
        is_articulated = str(spec.get("asset_type", "")).startswith("articulated_")
        front_cloud = front_view["object_clouds"].get(object_id)
        top_geometry = (
            instance_geometry(top_labels, top_ids[object_id])
            if object_id in top_ids else None
        )
        front_geometry = (
            instance_geometry(front_labels, front_ids[object_id])
            if object_id in front_ids else None
        )
        if precomputed_orientation_reports is not None:
            orientation_report = dict(precomputed_orientation_reports[object_id])
            orientation_matrix = np.asarray(
                orientation_report.get("matrix", np.eye(3)), dtype=float,
            )
        else:
            orientation_matrix = np.eye(3)
            orientation_report = {
                "method": "identity_gated_top_mask_pca_footprint_aspect",
                "selected_up_axis": "+Z",
                "orientation_changed": False,
                "decision": (
                    "articulated_asset_keep_identity"
                    if is_articulated
                    else "missing_cross_view_mask_keep_identity"
                ),
                "candidates": [],
            }
            if (
                not is_articulated
                and top_geometry is not None
                and front_geometry is not None
            ):
                _, orientation_matrix, orientation_report = infer_mesh_orientation(
                    mesh_path, top_geometry, top_camera, front_geometry, front_camera,
                    orientation_scorer, orientation_reference_rgbs,
                    reference_key=object_id,
                )
            if not is_articulated and top_geometry is not None:
                up_oriented_vertices = (
                    mesh_vertices_z_up(mesh_path) @ orientation_matrix.T
                )
                local_xy_matrix, local_xy_report = align_local_footprint_axes(
                    up_oriented_vertices,
                )
                orientation_matrix = local_xy_matrix @ orientation_matrix
                orientation_report["local_xy_axis_alignment"] = local_xy_report
                orientation_report["orientation_changed"] = bool(
                    orientation_report["orientation_changed"]
                    or local_xy_report["applied"]
                )
            else:
                orientation_report["local_xy_axis_alignment"] = {
                    "applied": False,
                    "method": "local_xy_minimum_area_bbox_axis_alignment",
                    "reason": (
                        "articulated_asset_keep_identity"
                        if is_articulated else "missing_top_view_mask"
                    ),
                }
            orientation_report["matrix"] = np.asarray(orientation_matrix).tolist()
        local_xy_report = orientation_report["local_xy_axis_alignment"]
        print(
            f"[mesh orientation] {object_id}: "
            f"up={orientation_report['selected_up_axis']} "
            f"local_xy={local_xy_report.get('correction_deg', 0.0):+.1f}deg "
            f"decision={orientation_report['decision']}",
            flush=True,
        )
        orientation_inputs.append((
            object_id, spec, mesh_path, orientation_matrix,
            orientation_report, top_geometry, front_cloud,
        ))

    if precomputed_orientation_reports is None:
        bake_jobs = []
        normalized_mesh_dir = output / "00_normalized_meshes"
        for object_id, spec, mesh_path, matrix, report, _, _ in orientation_inputs:
            if not report["orientation_changed"]:
                continue
            normalized_path = normalized_mesh_dir / f"{object_id}.glb"
            bake_jobs.append((mesh_path, normalized_path, matrix))
            spec["asset"] = str(normalized_path)
            spec["geometry_asset"] = str(normalized_path)
        bake_mesh_orientations(bake_jobs)

    # A cache upgrade must preserve the already selected mesh axes exactly.
    # Rebuild only the missing top templates when an older manifest is supplied.
    if precomputed_orientation_reports is None or args.stop_after_mesh_orientation:
        orientation_template_dir = output / "00_top_edge_templates"
        orientation_template_objects = {}
        orientation_top_templates = {}
        for (
            object_id, spec, _, _, _, top_geometry, _
        ) in orientation_inputs:
            mesh_path = Path(spec["geometry_asset"])
            pose = (
                coarse_object_pose(
                    blueprint_objects[object_id], mesh_path,
                    top_geometry, top_camera,
                )
                if top_geometry is not None else
                blueprint_object_pose(blueprint_objects[object_id], mesh_path)
            )
            mesh = search_mesh_z_up(mesh_path, face_count=None)
            template_pose, frame_fit = fit_top_template_pose(
                center_vertices_on_table(mesh.vertices),
                pose, top_refinement_camera,
            )
            orientation_template_objects[object_id] = {
                **spec, **template_pose,
                "top_template_frame_fit": frame_fit,
            }
            orientation_top_templates[object_id] = {
                "path": str(orientation_template_dir / f"{object_id}.png"),
                "pose": {
                    "translation_world_m": list(
                        template_pose["translation_world_m"],
                    ),
                    "yaw_deg": float(template_pose["yaw_deg"]),
                    "uniform_scale": float(template_pose["uniform_scale"]),
                    **(
                        {"scale_xyz": list(template_pose["scale_xyz"])}
                        if template_pose.get("scale_xyz") is not None else {}
                    ),
                },
                "frame_fit": frame_fit,
            }
        orientation_template_scene = output / "00_top_template_scene.json"
        write_json(
            orientation_template_scene,
            scene_document(
                table_mesh, table_scale,
                {**cameras, "top": top_refinement_camera},
                orientation_template_objects,
                table_scale_pivot=table_scale_pivot,
                table_yaw_deg=table_yaw_deg,
            ),
        )
        if blender_warmup_future is not None:
            blender_warmup_future.result()
        render_top_edge_templates(
            orientation_template_scene,
            orientation_template_dir,
            args.top_template_render_scale,
        )
        precomputed_top_templates = {
            object_id: {
                "version": 1,
                "camera": top_refinement_camera,
                "render_scale": int(args.top_template_render_scale),
                **value,
            }
            for object_id, value in orientation_top_templates.items()
        }
        write_json(output / "00_mesh_orientation_manifest.json", {
            "format": "mesh_orientation_v1",
            "asset_source": args.asset_source,
            "table": {
                "mesh": str(table_mesh),
                "yaw_deg": table_yaw_deg,
                "report": table_orientation_report,
            },
            "vggt": {
                "front": str(ROOT / "outputs/vggt_front_only"),
                "top": str(ROOT / "outputs/vggt_top_only"),
            },
            "top_template_cache": {
                "version": 1,
                "camera": top_refinement_camera,
                "render_scale": int(args.top_template_render_scale),
            },
            "items": [
                {
                    "object_id": object_id,
                    "spec": spec,
                    "report": report,
                    "top_template": orientation_top_templates[object_id],
                }
                for object_id, spec, _, _, report, _, _ in orientation_inputs
            ],
        })
        if args.stop_after_mesh_orientation:
            print(
                f"[mesh orientation] manifest={output / '00_mesh_orientation_manifest.json'}",
                flush=True,
            )
            return

    for (
        object_id, spec, _, orientation_matrix, orientation_report,
        top_geometry, front_cloud,
    ) in orientation_inputs:
        mesh_path = Path(spec["geometry_asset"])
        if top_geometry is not None:
            pose = coarse_object_pose(
                blueprint_objects[object_id], mesh_path, top_geometry, top_camera,
            )
        else:
            pose = blueprint_object_pose(blueprint_objects[object_id], mesh_path)
        pose["mesh_orientation"] = orientation_report
        coarse_inputs.append((
            object_id, spec, mesh_path, pose,
            front_cloud,
        ))

    def prepare_top_static_input(item):
        object_id, spec, _, _, _ = item
        object_start = time.perf_counter()
        part_start = time.perf_counter()
        mesh = search_mesh_z_up(
            Path(spec["geometry_asset"]), face_count=None,
        )
        mesh_seconds = time.perf_counter() - part_start
        part_start = time.perf_counter()
        contact_shape = bottom_contact_shape(mesh)
        contact_seconds = time.perf_counter() - part_start
        part_start = time.perf_counter()
        body_shape = body_cross_section_shape(
            mesh,
            args.top_circle_score_threshold,
            args.contact_square_ratio_threshold,
        )
        body_seconds = time.perf_counter() - part_start
        values = {
            "contact_shape": contact_shape,
            "body_shape": body_shape,
        }
        texture_seconds = 0.0
        if direction_scorer is not None:
            part_start = time.perf_counter()
            values["direction_vertex_rgb"] = item_vertex_texture_colors(mesh, spec)
            texture_seconds = time.perf_counter() - part_start
        print(
            f"[top timing static] {object_id}="
            f"{time.perf_counter() - object_start:.3f}s "
            f"mesh={mesh_seconds:.3f}s contact={contact_seconds:.3f}s "
            f"body={body_seconds:.3f}s texture={texture_seconds:.3f}s",
            flush=True,
        )
        return mesh, values

    refined_poses = refine_poses_to_pointcloud_gpu(
        [
            (mesh_path, pose, cloud)
            for _, _, mesh_path, pose, cloud in coarse_inputs
        ],
        early_stop_patience=args.front_refine_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    for (object_id, spec, _, _, _), pose in zip(coarse_inputs, refined_poses):
        coarse_objects[object_id] = {**spec, **pose}
    if direction_scorer is not None and hasattr(
        direction_scorer, "_wait_for_preload",
    ):
        direction_scorer._wait_for_preload()
    static_preparation_start = time.perf_counter()
    static_items = [item for item in coarse_inputs if item[0] in top_ids]
    with ThreadPoolExecutor(
        max_workers=min(2, max(1, len(static_items))),
    ) as static_executor:
        static_futures = {
            item[0]: static_executor.submit(prepare_top_static_input, item)
            for item in static_items
        }
        top_static_preprocessing = {
            object_id: future.result()
            for object_id, future in static_futures.items()
        }
    print(
        "[top timing] static_preprocessing="
        f"{time.perf_counter() - static_preparation_start:.3f}s "
        f"workers={min(2, max(1, len(static_items)))} "
        f"objects={len(top_static_preprocessing)}",
        flush=True,
    )
    coarse_objects = initialize_support_pose_heights(
        blueprint, coarse_objects, top_camera,
        enabled=args.support_pose_refinement,
    )
    coarse_scene = scene_document(
        table_mesh, table_scale, cameras, coarse_objects,
        table_scale_pivot=table_scale_pivot,
        table_yaw_deg=table_yaw_deg,
    )
    write_json(output / "03_object_coarse_alignment.json", coarse_scene)
    refinement_cameras = {**cameras, "top": top_refinement_camera}
    top_template_dir = output / "03_top_edge_templates"
    if blender_warmup_future is not None:
        blender_warmup_future.result()
        blender_warmup_executor.shutdown(wait=True)
        blender_warmup_future = None
        blender_warmup_executor = None
    if top_template_cache_compatible(
        precomputed_top_templates, coarse_objects,
        top_refinement_camera, args.top_template_render_scale,
    ):
        for object_id, item in coarse_objects.items():
            cached = precomputed_top_templates[object_id]
            item["edge_detector"] = args.edge_detector
            item["top_template_path"] = cached["path"]
            item["top_template_pose"] = rebase_top_template_pose(
                cached["pose"], cached["camera"], top_refinement_camera,
            )
            item["top_template_frame_fit"] = dict(cached["frame_fit"])
        print(
            "[top timing] blender_templates=0.000s "
            f"objects={len(coarse_objects)} source=mesh_orientation_cache",
            flush=True,
        )
    else:
        template_objects = {}
        for object_id, item in coarse_objects.items():
            template_mesh = search_mesh_z_up(
                Path(item["geometry_asset"]), face_count=None,
            )
            template_pose, frame_fit = fit_top_template_pose(
                center_vertices_on_table(template_mesh.vertices),
                item, top_refinement_camera,
            )
            template_item = {
                **item, **template_pose, "top_template_frame_fit": frame_fit,
            }
            template_objects[object_id] = template_item
            if frame_fit["applied"]:
                print(
                    f"[top template] {object_id}: fit-to-frame "
                    f"scale={frame_fit['scale_factor']:.4f} "
                    f"anchor={frame_fit['anchor_pixel']}",
                    flush=True,
                )
        template_scene_path = output / "03_top_template_scene.json"
        write_json(
            template_scene_path,
            scene_document(
                table_mesh, table_scale, refinement_cameras, template_objects,
                table_scale_pivot=table_scale_pivot,
                table_yaw_deg=table_yaw_deg,
            ),
        )
        template_render_start = time.perf_counter()
        render_top_edge_templates(
            template_scene_path,
            top_template_dir,
            args.top_template_render_scale,
        )
        template_render_seconds = time.perf_counter() - template_render_start
        print(
            "[top timing] blender_templates="
            f"{template_render_seconds:.3f}s "
            f"objects={len(coarse_objects)} source=scene_render",
            flush=True,
        )
        for object_id, item in coarse_objects.items():
            template_item = template_objects[object_id]
            item["edge_detector"] = args.edge_detector
            item["top_template_path"] = str(
                top_template_dir / f"{object_id}.png"
            )
            item["top_template_pose"] = {
                "translation_world_m": list(
                    template_item["translation_world_m"],
                ),
                "yaw_deg": float(template_item["yaw_deg"]),
                "uniform_scale": float(template_item["uniform_scale"]),
            }
            if template_item.get("scale_xyz") is not None:
                item["top_template_pose"]["scale_xyz"] = list(
                    template_item["scale_xyz"],
                )
            item["top_template_frame_fit"] = template_item[
                "top_template_frame_fit"
            ]
    coarse_scene = scene_document(
        table_mesh, table_scale, cameras, coarse_objects,
        table_scale_pivot=table_scale_pivot,
        table_yaw_deg=table_yaw_deg,
    )
    write_json(output / "03_object_coarse_alignment.json", coarse_scene)
    save_edge_overlay(
        ROOT / "data/topview_image.png", top_labels, top_ids, coarse_objects, top_camera,
        output / "03_object_coarse_top_edges.png", args.edge_detector,
    )
    save_edge_overlay(
        ROOT / "data/reference_image.png", front_labels, front_ids, coarse_objects, front_camera,
        output / "03_object_coarse_front_edges.png", args.edge_detector,
    )
    if args.stop_after_object_coarse:
        render_stages([
            (output / "02_table_alignment.json", output / "02_table_alignment_render", True),
            (output / "03_object_coarse_alignment.json", output / "03_object_coarse_render", True),
        ])
        compose_stage_comparison(
            output / "02_table_alignment_render", output / "02_table_alignment_comparison.png",
            "stage 1 intermediate: table/cameras",
        )
        compose_stage_comparison(
            output / "03_object_coarse_render", output / "03_object_coarse_comparison.png",
            "stage 1 complete: table/cameras+object coarse",
            previous_dir=output / "02_table_alignment_render", previous_title="stage 1 intermediate: table/cameras",
        )
        print_output_paths(output, names=(
            "object_coarse_front", "object_coarse_top",
            "object_coarse_front_overlay", "object_coarse_top_overlay",
            "object_coarse_front_render_edges", "object_coarse_top_render_edges",
            "object_coarse_front_edges", "object_coarse_top_edges", "object_coarse_comparison",
        ))
        return

    front_direction_rgb = orientation_reference_rgbs["front"]
    top_camera = top_refinement_camera
    cameras = refinement_cameras
    print(
        f"[2/3] refining in the fixed {top_camera['projection']} top camera "
        f"with {args.top_refinement_method}",
        flush=True,
    )
    if (
        args.top_fine_refinement_method
        not in ("2d_affine", "2d_affine_search")
        and args.top_refinement_method != "differentiable_edge"
    ):
        raise ValueError(
            "--top-fine-refinement-method applies only to differentiable_edge"
        )
    top_report = None
    top_rgb_native = np.asarray(
        Image.open(ROOT / "data/topview_image.png").convert("RGB"),
    )
    if args.top_refinement_method in ("ot", "differentiable_edge"):
        top_rgb = top_rgb_native
        edge_camera = scaled_edge_camera(top_camera, args.top_template_render_scale)
        edge_width, edge_height = edge_camera["image_size"]
        if args.top_template_render_scale != 1:
            top_rgb = cv2.resize(
                top_rgb, (edge_width, edge_height), interpolation=cv2.INTER_LANCZOS4,
            )
            edge_labels = cv2.resize(
                top_labels, (edge_width, edge_height), interpolation=cv2.INTER_NEAREST,
            )
        else:
            edge_labels = top_labels
        top_reference_masks = support_aware_reference_masks(
            edge_labels, top_ids, coarse_objects,
        )
        top_full_frame_teed = (
            teed_edges(top_rgb)[0]
            if reference_edge_detector(args.edge_detector) == "teed" else None
        )
        input_preparation_start = time.perf_counter()
        top_inputs = []
        for object_id, initial in coarse_objects.items():
            if object_id not in top_ids:
                continue
            mesh, static_preprocessing = top_static_preprocessing[object_id]
            target = top_reference_masks[object_id]
            appearance_target = filled_instance_appearance_mask(
                edge_labels, top_ids, object_id,
            )
            edges, _ = (
                masked_full_frame_teed_edges(
                    top_full_frame_teed, appearance_target > 0,
                )
                if top_full_frame_teed is not None else
                masked_edges(
                    top_rgb, appearance_target > 0,
                    method=reference_edge_detector(args.edge_detector),
                )
            )
            top_inputs.append((
                object_id, (
                    mesh, initial, target, appearance_target, edges,
                    static_preprocessing,
                ),
            ))

        top_direction_batcher = (
            DirectionBatchCoordinator(
                len(top_inputs),
                direction_scoring_camera(direction_scorer, top_camera),
                direction_scorer,
            )
            if direction_scorer is not None
            and hasattr(direction_scorer, "score_candidate_groups")
            and not hasattr(direction_scorer, "score_multiview_candidates")
            else None
        )
        top_fine_semaphore = (
            Semaphore(2)
            if args.top_object_optimization_mode == "independent_parallel"
            else None
        )
        top_coarse_semaphore = (
            Semaphore(1)
            if args.top_object_optimization_mode == "independent_parallel"
            else None
        )

        def refine_top_input(item):
            object_id, (
                mesh, initial, target, appearance_target, edges,
                static_preprocessing,
            ) = item
            if args.top_refinement_method == "differentiable_edge":
                if args.top_scale_mode != "uniform":
                    raise ValueError("differentiable edge refinement requires uniform scale")
                return refine_object_top_differentiable_edges(
                    mesh, initial, target, appearance_target, edges, edge_camera,
                    differentiable_refinement_config(args),
                    fine_method=args.top_fine_refinement_method,
                    circle_score_threshold=args.top_circle_score_threshold,
                    square_ratio_threshold=args.contact_square_ratio_threshold,
                    scale_estimator="rotated_bbox",
                    yaw_pca_source="mask",
                    target_rgb=top_rgb,
                    yaw_candidate_selection="rank",
                    yaw_mask_iou_loss_weight=args.top_yaw_mask_iou_loss_weight,
                    yaw_direction_iou_ambiguity_threshold=(
                        args.front_yaw_top_iou_ambiguity_threshold
                    ),
                    internal_orientation_weight=args.top_internal_orientation_weight,
                    internal_edge_match_mode=args.top_internal_edge_match_mode,
                    model_edge_method=args.top_model_edge_method,
                    direction_scorer=direction_scorer,
                    direction_camera=direction_scoring_camera(
                        direction_scorer, top_camera,
                    ),
                    direction_reference_rgb=top_rgb_native,
                    direction_reference_mask=target_mask(
                        top_labels, top_ids, object_id,
                    ),
                    direction_views=(
                        {
                            "front": {
                                "camera": direction_scoring_camera(
                                    direction_scorer, front_camera,
                                ),
                                "reference_rgb": front_direction_rgb,
                                "reference_mask": target_mask(
                                    front_labels, front_ids, object_id,
                                ),
                            },
                            "top": {
                                "camera": direction_scoring_camera(
                                    direction_scorer, top_camera,
                                ),
                                "reference_rgb": top_rgb_native,
                                "reference_mask": target_mask(
                                    top_labels, top_ids, object_id,
                                ),
                            },
                        }
                        if direction_scorer is not None
                        and hasattr(
                            direction_scorer, "score_multiview_candidates",
                        ) else None
                    ),
                    timing_label=object_id,
                    static_preprocessing=static_preprocessing,
                    direction_batcher=top_direction_batcher,
                    coarse_refinement_semaphore=top_coarse_semaphore,
                    fine_refinement_semaphore=top_fine_semaphore,
                )
            return (
                refine_object_top_staged(mesh, initial, target, edges, edge_camera)
                if args.top_scale_mode == "uniform"
                else refine_object(mesh, initial, target, edges, edge_camera, scale_mode="xy")
            )

        print(
            "[top timing] input_preparation="
            f"{time.perf_counter() - input_preparation_start:.3f}s "
            f"objects={len(top_inputs)}",
            flush=True,
        )
        object_refinement_start = time.perf_counter()
        refined_top_objects = run_object_refinements(
            top_inputs, refine_top_input, args.top_object_optimization_mode,
        )
        print(
            "[top timing] object_refinements_wall="
            f"{time.perf_counter() - object_refinement_start:.3f}s "
            f"mode={args.top_object_optimization_mode}",
            flush=True,
        )
        top_objects = {**coarse_objects, **refined_top_objects}
        if (
            args.top_refinement_method == "differentiable_edge"
            and args.top_fine_refinement_method == "differentiable_3d_mask"
        ):
            top_objects, top_report = refine_top_masks_differentiable(
                objects=top_objects,
                top_camera=top_camera,
                top_labels=top_labels,
                top_ids=top_ids,
                table_mesh_path=table_mesh,
                table_scale_xyz=table_scale,
                output_dir=output / "04_top_3d_mask_refinement_debug",
                config=_top_3d_mask_loss_config(
                    differentiable_refinement_config(args),
                ),
                scale_mode="xy",
                joint_pose_scale=True,
                circle_score_threshold=args.top_circle_score_threshold,
                square_ratio_threshold=args.contact_square_ratio_threshold,
                object_optimization_mode=args.top_object_optimization_mode,
                top_rgb=top_rgb_native,
                table_scale_pivot=table_scale_pivot,
                table_yaw_deg=table_yaw_deg,
            )
    else:
        if args.top_scale_mode != "uniform":
            raise ValueError("differentiable top refinement requires uniform scale")
        top_objects, top_report = refine_top_masks_differentiable(
            objects=coarse_objects,
            top_camera=top_camera,
            top_labels=top_labels,
            top_ids=top_ids,
            table_mesh_path=table_mesh,
            table_scale_xyz=table_scale,
            output_dir=output / "04_top_refinement_debug",
            config=differentiable_refinement_config(args),
            circle_score_threshold=args.top_circle_score_threshold,
            square_ratio_threshold=args.contact_square_ratio_threshold,
            object_optimization_mode=args.top_object_optimization_mode,
            top_rgb=top_rgb_native,
            table_scale_pivot=table_scale_pivot,
            table_yaw_deg=table_yaw_deg,
        )
    top_objects = initialize_support_pose_heights(
        blueprint, top_objects, top_camera,
        enabled=args.support_pose_refinement,
    )
    top_scene = scene_document(
        table_mesh, table_scale, cameras, top_objects,
        table_scale_pivot=table_scale_pivot,
        table_yaw_deg=table_yaw_deg,
    )
    top_scene["refinement"] = {
        "asset_source": args.asset_source,
        "top_method": args.top_refinement_method,
        "top_fine_method": args.top_fine_refinement_method,
        "contact_circle_score_threshold": args.top_circle_score_threshold,
        "contact_bbox_square_ratio_threshold": args.contact_square_ratio_threshold,
        "top_object_optimization_mode": args.top_object_optimization_mode,
        "top_scale_mode": args.top_scale_mode,
        "direction_loss_backend": "dino_relative_position",
        "dino_relative_position_direction_loss": {
            "root": str(args.minima_root),
            "device": args.minima_device,
            "input_size": direction_scorer.input_size,
            "patch_grid": [direction_scorer.input_size // 14] * 2,
            "background_rgb": [238, 238, 238],
            "method": (
                "shared_reference_bbox_DINO_RGB_correspondence_"
                "normal_structure_weighted_relative_position"
            ),
            "stage_3_uses_stage_2_prior": False,
        },
        "direction_reference_views": {
            "stage_2": "top",
            "stage_3": "front",
        },
        "edge_detector": args.edge_detector,
        "top_report": top_report,
        "unmatched_topview_object_ids": [
            object_id for object_id in OBJECT_IDS if object_id not in top_ids
        ],
    }
    write_json(output / "04_top_refined.json", top_scene)
    if hasattr(direction_scorer, "close"):
        direction_scorer.close()
    render_stages([
        (output / "02_table_alignment.json", output / "02_table_alignment_render", False),
        (output / "03_object_coarse_alignment.json", output / "03_object_coarse_render", False),
        (
            output / "04_top_refined.json",
            output / "04_top_refined_render",
            args.stop_after_top_refined,
        ),
    ])
    save_edge_overlay(
        ROOT / "data/topview_image.png", top_labels, top_ids, top_objects, top_camera,
        output / "04_top_refined_edges.png", args.edge_detector,
        rendered_image_path=output / "04_top_refined_render" / "top.png",
        model_edge_method=args.top_model_edge_method,
    )
    compose_stage_comparison(
        output / "02_table_alignment_render", output / "02_table_alignment_comparison.png",
        "stage 1 intermediate: table/cameras",
    )
    compose_stage_comparison(
        output / "03_object_coarse_render", output / "03_object_coarse_comparison.png",
        "stage 1 complete: table/cameras+object coarse",
        previous_dir=output / "02_table_alignment_render", previous_title="stage 1 intermediate: table/cameras",
    )
    compose_stage_comparison(
        output / "04_top_refined_render", output / "04_top_refined_comparison.png",
        f"top {args.top_scale_mode}-scale refinement (stage 2/3)",
        previous_dir=output / "03_object_coarse_render", previous_title="stage 1: object coarse",
    )
    if args.stop_after_top_refined:
        print_output_paths(output, names=(
            "top_refined_front", "top_refined_top",
            "top_refined_front_overlay", "top_refined_top_overlay",
            "top_refined_front_render_edges", "top_refined_top_render_edges",
            "top_refined_edges", "top_refined_comparison",
        ))
        return
    close_persistent_blender_worker()

    first_round_description = (
        "fit frame, bbox-height XYZ scale, model-edge yaw, bbox XY, then joint fine refinement"
    )
    front_stage_start = time.perf_counter()
    print(f"[3/3] differentiable front-mask refinement: {first_round_description}", flush=True)
    if args.front_refine_rounds < 1:
        raise ValueError("front_refine_rounds must be at least 1")
    front_config = differentiable_refinement_config(args)
    front_rgb = front_direction_rgb
    front_edge_targets = {}
    specialized_bbox_schedule = True
    front_targets = None
    if specialized_bbox_schedule or any(
        mode in ("edge_drawing", "mask_edge")
        for mode in (args.front_refine_loss, args.front_refine_second_round_loss)
    ):
        front_targets = edge_drawing_silhouette_targets(
            front_rgb,
            front_labels, front_ids, OBJECT_IDS,
            include_internal=True,
            edge_detector=args.edge_detector,
        )
        front_edge_targets = {
            "edge_drawing": front_targets,
            "mask_edge": front_targets,
        }

    def target_edge_fn_for(loss_mode):
        targets = front_edge_targets.get(loss_mode)
        return (
            (lambda _labels, _ids, object_id: targets[object_id])
            if targets is not None else None
        )
    front_top_circle_scores = {
        object_id: rendered_model_circle_score(item["top_template_path"])
        for object_id, item in top_objects.items()
        if object_id in top_ids
    }
    front_contact_shapes = {
        object_id: object_contact_shape(item)
        for object_id, item in top_objects.items()
        if object_id in top_ids
    }
    front_body_shapes = {
        object_id: object_body_cross_section_shape(
            item, args.top_circle_score_threshold,
            args.contact_square_ratio_threshold,
        )
        for object_id, item in top_objects.items()
        if object_id in top_ids
    }
    front_tie_sources = {
        object_id: object_xy_scale_tie_sources(
            front_top_circle_scores[object_id], front_contact_shapes[object_id],
            args.top_circle_score_threshold,
            args.contact_square_ratio_threshold,
            front_body_shapes[object_id],
        )
        for object_id in front_contact_shapes
    }
    front_tied_xy_scale_ids = tuple(
        object_id for object_id, item in top_objects.items()
        if object_id in top_ids
        and any(front_tie_sources[object_id].values())
    )
    front_reference_masks = support_aware_reference_masks(
        front_labels, front_ids, top_objects,
    )
    front_top_reference_masks = support_aware_reference_masks(
        top_labels, top_ids, top_objects,
    )

    def front_reference_mask(labels, ids, object_id):
        restored = front_reference_masks.get(object_id)
        return (
            restored
            if restored is not None else instance_mask(labels, ids, object_id)
        )

    front_round_count = 1 if specialized_bbox_schedule else args.front_refine_rounds
    bbox_yaw_initialization_report = None
    if specialized_bbox_schedule:
        initializer_start = time.perf_counter()
        (
            front_direction_render_cache,
            front_direction_score_cache,
            front_branch_candidate_cache,
        ) = precompute_front_direction_groups(
            top_objects, front_camera, front_labels, front_ids,
            front_rgb, front_reference_mask, direction_scorer,
        )
        top_objects, bbox_yaw_initialization_report = (
            initialize_front_bbox_and_normal_yaw(
                top_objects, front_camera, front_labels, front_ids,
                front_rgb, front_config,
                top_camera=top_camera,
                top_target_masks=front_top_reference_masks,
                top_target_rgb=top_rgb_native,
                tie_xy_scale_ids=front_tied_xy_scale_ids,
                internal_orientation_weight=args.front_internal_orientation_weight,
                internal_edge_match_mode=args.top_internal_edge_match_mode,
                edge_detector=args.edge_detector,
                model_edge_method=args.front_model_edge_method,
                target_edges=front_targets,
                target_mask_fn=front_reference_mask,
                direction_scorer=direction_scorer,
                direction_render_cache=front_direction_render_cache,
                direction_score_cache=front_direction_score_cache,
                branch_candidate_cache=front_branch_candidate_cache,
            )
        )
        print(
            "[front timing] bbox_and_direction_initializer="
            f"{time.perf_counter() - initializer_start:.3f}s",
            flush=True,
        )
        if hasattr(direction_scorer, "close"):
            direction_scorer.close()
        front_config = replace(
            _top_3d_mask_loss_config(front_config),
            bbox_weight=front_config.bbox_weight,
            fine_resolution=args.front_refine_second_round_resolution,
        )
    front_yaw_origins = {
        object_id: float(item["yaw_deg"])
        for object_id, item in top_objects.items()
    }
    debug_root = output / "05_front_refinement_debug"
    branch_specs = (
        ((0.0, 30.0),)
        if specialized_bbox_schedule else
        front_yaw_branch_specs(
            args.front_yaw_search, args.front_yaw_branch_offset_deg,
        )
    )
    top_yaw_gate_report = {}
    if len(branch_specs) > 1:
        print(
            "[front yaw gate] disabled; every object evaluates both "
            f"0° and {args.front_yaw_branch_offset_deg:g}° branches",
            flush=True,
        )
    round_reports = []
    checkpoint_dir = (
        None if specialized_bbox_schedule
        else os.environ.get("FRONT_BRANCH_CHECKPOINT_DIR")
    )
    first_round_branches = (
        load_front_branch_checkpoints(checkpoint_dir, branch_specs, top_objects)
        if checkpoint_dir else []
    )
    if first_round_branches:
        print(f"[3/3] resumed first-round yaw branches from {checkpoint_dir}", flush=True)
    else:
        for yaw_offset, yaw_limit in branch_specs:
            branch_object_ids = tuple(OBJECT_IDS)
            branch_loss_mode = (
                "mask" if specialized_bbox_schedule else
                front_round_loss_mode(
                    0, args.front_refine_loss, args.front_refine_second_round_loss,
                )
            )
            print(
                f"[3/3 branch {yaw_offset:g}° round 1/{front_round_count}] "
                f"{len(branch_object_ids)}/{len(OBJECT_IDS)} objects; "
                f"{first_round_description}; "
                f"loss={branch_loss_mode}",
                flush=True,
            )
            branch_start = time.perf_counter()
            branch_objects, branch_report = refine_front_masks_differentiable(
                top_objects=top_objects,
                front_camera=front_camera,
                front_labels=front_labels,
                front_ids=front_ids,
                object_ids=branch_object_ids,
                table_mesh_path=table_mesh,
                table_scale_xyz=table_scale,
                mesh_loader=search_mesh_z_up,
                object_vertex_centerer=center_vertices_on_table,
                table_vertex_centerer=lambda vertices: (
                    transform_table_vertices(vertices, {
                        "scale_xyz": table_scale,
                        "scale_pivot": table_scale_pivot,
                        "yaw_deg": table_yaw_deg,
                    })
                    / np.asarray(table_scale, dtype=float)
                ),
                target_mask_fn=front_reference_mask,
                output_dir=debug_root / "round_1" / f"yaw_{yaw_offset:g}",
                config=front_config,
                scale_mode="xyz",
                tie_xy_scale_ids=front_tied_xy_scale_ids,
                joint_pose_scale=(
                    True if specialized_bbox_schedule else
                    front_round_uses_joint_optimization(
                        0, args.front_refine_second_round_mode,
                    )
                ),
                loss_mode=branch_loss_mode,
                target_edge_fn=target_edge_fn_for(branch_loss_mode),
                texture_target_rgb=(
                    None if specialized_bbox_schedule else front_rgb
                ),
                object_optimization_mode=args.front_object_optimization_mode,
                optimization_schedule="bbox_center_then_joint",
                support_pose_refinement=specialized_bbox_schedule,
                initial_yaw_offset_deg=yaw_offset,
                yaw_delta_bounds_deg_by_object={
                    object_id: (-yaw_limit, yaw_limit)
                    for object_id in branch_object_ids
                },
                restore_supported_occlusions=True,
            )
            print(
                f"[front timing] branch_{yaw_offset:g}_round_1="
                f"{time.perf_counter() - branch_start:.3f}s",
                flush=True,
            )
            first_round_branches.append((yaw_offset, branch_objects, branch_report))
    if len(first_round_branches) > 1:
        front_yaw_rgb_losses = {}
        if args.front_yaw_rgb_weight > 0.0:
            front_yaw_rgb_losses = rendered_front_rgb_branch_losses(
                first_round_branches, table_mesh, table_scale, cameras,
                front_camera, front_labels, front_ids,
                front_rgb,
                debug_root / "rgb_branch_renders",
                table_yaw_deg=table_yaw_deg,
            )
            print(
                f"[front yaw RGB] scored {len(front_yaw_rgb_losses)} objects, "
                f"weight={args.front_yaw_rgb_weight:g}",
                flush=True,
            )
        front_yaw_direction_losses = (
            direction_front_branch_losses(
                first_round_branches, direction_scorer,
                front_camera, front_rgb, front_reference_masks,
                top_camera=top_camera,
                top_rgb=top_rgb_native,
                top_masks=front_top_reference_masks,
            )
            if direction_scorer is not None else {}
        )
        final_objects, first_round_report = select_independent_yaw_branches(
            first_round_branches,
            rgb_losses_by_object=front_yaw_rgb_losses,
            rgb_weight=args.front_yaw_rgb_weight,
            direction_losses_by_object=front_yaw_direction_losses,
        )
        selected_offsets = first_round_report["selected_yaw_branch_offsets_deg"]
    else:
        yaw_offset, final_objects, first_round_report = first_round_branches[0]
        selected_offsets = {object_id: yaw_offset for object_id in final_objects}
    if hasattr(direction_scorer, "close"):
        direction_scorer.close()
    first_round_report["top_yaw_ambiguity_gate"] = top_yaw_gate_report
    for object_id, gate in top_yaw_gate_report.items():
        first_round_report["objects"][object_id]["top_yaw_ambiguity_gate"] = gate
    round_reports.append(first_round_report)

    for round_index in range(1, front_round_count):
        joint_round = front_round_uses_joint_optimization(
            round_index, args.front_refine_second_round_mode,
        )
        round_config = (
            replace(
                front_config,
                fine_resolution=args.front_refine_second_round_resolution,
            )
            if round_index == 1 else front_config
        )
        round_loss_mode = front_round_loss_mode(
            round_index, args.front_refine_loss,
            args.front_refine_second_round_loss,
        )
        round_description = (
            "joint XY/yaw/joint/shared-XY-scale/Z-scale"
            if joint_round
            else "XY/yaw/joint, then shared XY and independent Z scale"
        )
        print(
            f"[3/3 round {round_index + 1}/{args.front_refine_rounds}] "
            f"{round_description}; loss={round_loss_mode}",
            flush=True,
        )
        yaw_delta_bounds = {
            object_id: yaw_delta_bounds_from_origin(
                front_yaw_origins[object_id] + selected_offsets[object_id],
                item["yaw_deg"], 30.0,
            )
            for object_id, item in final_objects.items()
            if object_id in front_yaw_origins
        }
        final_objects, round_report = refine_front_masks_differentiable(
            top_objects=final_objects,
            front_camera=front_camera,
            front_labels=front_labels,
            front_ids=front_ids,
            object_ids=OBJECT_IDS,
            table_mesh_path=table_mesh,
            table_scale_xyz=table_scale,
            mesh_loader=search_mesh_z_up,
            object_vertex_centerer=center_vertices_on_table,
            table_vertex_centerer=lambda vertices: (
                transform_table_vertices(vertices, {
                    "scale_xyz": table_scale,
                    "scale_pivot": table_scale_pivot,
                    "yaw_deg": table_yaw_deg,
                })
                / np.asarray(table_scale, dtype=float)
            ),
            target_mask_fn=front_reference_mask,
            output_dir=debug_root,
            config=round_config,
            scale_mode="xyz",
            tie_xy_scale_ids=front_tied_xy_scale_ids,
            joint_pose_scale=joint_round,
            loss_mode=round_loss_mode,
            target_edge_fn=target_edge_fn_for(round_loss_mode),
            texture_target_rgb=front_rgb,
            object_optimization_mode=args.front_object_optimization_mode,
            optimization_schedule="bbox_center_then_joint",
            support_pose_refinement=round_index >= 1,
            initial_yaw_offset_deg=0.0,
            yaw_delta_bounds_deg_by_object=yaw_delta_bounds,
            restore_supported_occlusions=True,
        )
        round_reports.append(round_report)
    front_report = combine_front_round_reports(
        round_reports, args.front_refine_second_round_mode,
        "bbox_center_then_joint",
    )
    front_report["top_circle_score_threshold"] = args.top_circle_score_threshold
    front_report["top_circle_scores"] = front_top_circle_scores
    front_report["top_circle_score_source"] = "model_top_render"
    front_report["contact_circle_score_threshold"] = args.top_circle_score_threshold
    front_report["contact_bbox_square_ratio_threshold"] = args.contact_square_ratio_threshold
    front_report["contact_shapes"] = front_contact_shapes
    front_report["body_cross_section_shapes"] = front_body_shapes
    front_report["xy_scale_tie_rule"] = (
        "top_view_circle_or_contact_circle_and_square_or_stable_body_cross_section"
    )
    front_report["xy_scale_tie_sources"] = front_tie_sources
    front_report["contact_circle_score_source"] = "lowest_downward_mesh_facet"
    if specialized_bbox_schedule:
        front_report["method"] = (
            "frame_fit_front_height_xyz_bbox_xy_stage2_normal_yaw_"
            "selected_front_width_x_top_length_y_bbox_xy_"
            "then_joint_xy_yaw_joint_independent_xyz_scale"
        )
        front_report["bbox_yaw_initialization"] = bbox_yaw_initialization_report
    final_scene = scene_document(
        table_mesh, table_scale, cameras, final_objects,
        table_scale_pivot=table_scale_pivot,
        table_yaw_deg=table_yaw_deg,
    )
    final_scene["refinement"] = {
        **top_scene.get("refinement", {}),
        "front_method": front_report["method"],
        "front_loss_modes": [
            (
                "mask" if specialized_bbox_schedule else
                front_round_loss_mode(
                    index, args.front_refine_loss,
                    args.front_refine_second_round_loss,
                )
            )
            for index in range(front_round_count)
        ],
        "front_object_optimization_mode": args.front_object_optimization_mode,
        "front_yaw_search": args.front_yaw_search,
        "front_yaw_gate_enabled": False,
        "front_yaw_branch_offset_deg": args.front_yaw_branch_offset_deg,
        "front_yaw_search_limit_deg": branch_specs[0][1],
        "front_uses_top_loss": False,
        "front_report": front_report,
    }
    write_json(output / "05_front_refined.json", final_scene)
    print(
        "[front timing] optimization_total="
        f"{time.perf_counter() - front_stage_start:.3f}s",
        flush=True,
    )
    final_render_start = time.perf_counter()
    render_stage(
        output / "05_front_refined.json",
        output / "05_front_refined_render",
        save_blend=args.export_final_scene,
    )
    save_direct_teed_edge_overlay(
        ROOT / "data/reference_image.png",
        output / "05_front_refined_render/front.png",
        output / "05_front_refined_edges.png",
    )
    print("[comparison] composing fixed-camera stage comparison", flush=True)
    compose_report(
        [
            output / "03_object_coarse_render",
            output / "04_top_refined_render",
            output / "05_front_refined_render",
        ],
        output / "scene_alignment_comparison.png",
    )
    print(
        "[front timing] final_render_and_comparison="
        f"{time.perf_counter() - final_render_start:.3f}s",
        flush=True,
    )
    print_output_paths(output)


def suppress_blender_gltf_info(bpy_module):
    # Blender's glTF importer maps debug_value=1 to WARNING logging.
    bpy_module.app.debug_value = 1


_cached_blender_scene = None


def _reset_blender_controller(controller):
    controller.location = (0.0, 0.0, 0.0)
    controller.rotation_euler = (0.0, 0.0, 0.0)
    controller.scale = (1.0, 1.0, 1.0)


def _apply_cached_blender_scene(document, cache):
    from pipeline.scene_renderer import (
        apply_articulated_joint_state, place_asset, scale_asset_below_top,
    )

    for obj in cache["table"][1]:
        if obj.type == "MESH":
            obj.hide_render = False
    for object_id, entry in cache["objects"].items():
        hidden = object_id not in document["objects"]
        for obj in entry["objects"]:
            if obj.type == "MESH":
                obj.hide_render = hidden
    if cache.get("ground") is not None:
        cache["ground"].hide_render = False

    table_controller, table_objects = cache["table"]
    _reset_blender_controller(table_controller)
    table_pivot = document["table"].get("scale_pivot") or {}
    if "leg_scale_z" in table_pivot:
        scale_asset_below_top(
            table_controller, table_objects,
            table_pivot["leg_anchor_local_z"],
            float(table_pivot["leg_scale_z"])
            / float(document["table"]["scale_xyz"][2]),
        )
    place_asset(
        table_controller, table_objects,
        document["table"].get(
            "blender_scale_xyz", document["table"]["scale_xyz"],
        ),
        table_scale_pivot_offset(document["table"]),
        document["table"].get("yaw_deg", 0.0),
        z_anchor="top",
    )
    for object_id, item in document["objects"].items():
        entry = cache["objects"][object_id]
        controller, imported = entry["controller"], entry["objects"]
        _reset_blender_controller(controller)
        if item["asset_type"] == "articulated_urdf":
            apply_articulated_joint_state(imported, item)
        place_asset(
            controller, imported,
            item.get("scale_xyz", [item["uniform_scale"]] * 3),
            item["translation_world_m"], item["yaw_deg"],
            roll_x_deg=item.get("roll_x_deg", 0.0),
            roll_y_deg=item.get("roll_y_deg", 0.0),
        )


def _build_cached_blender_scene(document):
    import bpy
    from pipeline.scene_renderer import (
        add_shadow_ground, bounds, clear_scene,
        import_articulated_urdf_asset, import_asset,
        setup_ambient_occlusion_compositor, setup_world,
    )

    clear_scene()
    setup_world()
    table = document["table"]
    table_entry = import_asset(table["mesh"], "table_0")
    objects = {}
    all_meshes = [obj for obj in table_entry[1] if obj.type == "MESH"]
    for object_id, item in document["objects"].items():
        imported = (
            import_articulated_urdf_asset(
                item["asset"], object_id, item.get("asset_orientation_matrix"),
            )
            if item["asset_type"] == "articulated_urdf"
            else import_asset(item["asset"], object_id)
        )
        objects[object_id] = {
            "controller": imported[0], "objects": imported[1],
            "asset_type": item["asset_type"],
        }
        all_meshes.extend(obj for obj in imported[1] if obj.type == "MESH")
    cache = {
        "signature": blender_asset_signature(document),
        "table": table_entry,
        "objects": objects,
        "all_meshes": all_meshes,
        "cameras": {},
        "ground": None,
    }
    _apply_cached_blender_scene(document, cache)
    low, _ = bounds(list(table_entry[1]) + [
        obj for entry in objects.values() for obj in entry["objects"]
    ])
    cache["ground"] = add_shadow_ground(low.z - 0.002)
    setup_ambient_occlusion_compositor()
    return cache


def _ensure_cached_blender_scene(document, scene_mode):
    global _cached_blender_scene
    signature = blender_asset_signature(document)
    compatible = False
    if _cached_blender_scene is not None:
        cached_table, cached_objects = _cached_blender_scene["signature"]
        requested_table, requested_objects = signature
        compatible = (
            cached_table == requested_table
            and set(requested_objects).issubset(set(cached_objects))
        )
    if (
        scene_mode == "reload"
        or _cached_blender_scene is None
        or not compatible
    ):
        _cached_blender_scene = _build_cached_blender_scene(document)
    else:
        _apply_cached_blender_scene(document, _cached_blender_scene)
    return _cached_blender_scene


def _configure_cached_blender_camera(cache, view_name, info):
    import bpy
    from mathutils import Matrix

    camera = cache["cameras"].get(view_name)
    if camera is None:
        camera_data = bpy.data.cameras.new(f"{view_name}_camera")
        camera = bpy.data.objects.new(f"{view_name}_camera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
        cache["cameras"][view_name] = camera
    camera_data = camera.data
    width, height = info["image_size"]
    intrinsic = np.asarray(info["intrinsic"])
    if info.get("projection") == "orthographic":
        camera_data.type = "ORTHO"
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.ortho_scale = float(info["ortho_scale"])
    else:
        camera_data.type = "PERSP"
        camera_data.sensor_width = 36.0
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.lens = intrinsic[0, 0] * 36.0 / width
        camera_data.shift_x = (intrinsic[0, 2] - width / 2) / width
        camera_data.shift_y = (height / 2 - intrinsic[1, 2]) / width
    rotation_blender = np.asarray(info["rotation_world_from_camera"]) @ np.diag([1.0, -1.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_blender
    matrix[:3, 3] = info["center_world_m"]
    camera.matrix_world = Matrix(matrix.tolist())
    return camera, intrinsic, int(width), int(height)


def articulation_pose_value(pose, lower, upper, rest):
    lower, upper, rest = float(lower), float(upper), float(rest)
    if not all(math.isfinite(value) for value in (lower, upper, rest)):
        raise ValueError("articulation pose values must be finite")
    if lower > upper:
        raise ValueError("articulation joint lower limit exceeds upper limit")
    if pose == "lower":
        return lower
    if pose == "upper":
        return upper
    if pose == "rest":
        return min(max(rest, lower), upper)
    raise ValueError(f"unknown articulation pose: {pose}")


def _articulation_motion_entries(document, cache):
    replaced = document.get("replacement", {}).get(
        "articulated_object_ids", [],
    )
    entries = []
    for object_id in replaced:
        item = document["objects"][object_id]
        package_manifest = json.loads(
            (Path(item["asset"]) / "manifest.json").read_text(encoding="utf-8")
        )
        rest_by_joint = {
            state["name"]: float(state.get(
                "scene_current_q", state.get("relative_displacement", 0.0),
            ))
            for state in package_manifest.get("joint_states", [])
        }
        for motion in cache["objects"][object_id]["objects"]:
            if "joint_value" not in motion:
                continue
            joint_name = motion.get("joint_name")
            if not joint_name:
                prefix = f"{object_id}_joint_"
                if not motion.name.startswith(prefix):
                    raise ValueError(
                        f"cannot identify articulated joint object {motion.name}"
                    )
                joint_name = motion.name[len(prefix):]
            entries.append({
                "motion": motion,
                "lower": float(motion["joint_lower"]),
                "upper": float(motion["joint_upper"]),
                "rest": rest_by_joint.get(str(joint_name), 0.0),
            })
    if replaced and not entries:
        raise ValueError("replacement scene has no controllable articulated joints")
    return entries


def _set_articulation_pose(entries, pose):
    import bpy

    for entry in entries:
        entry["motion"]["joint_value"] = articulation_pose_value(
            pose, entry["lower"], entry["upper"], entry["rest"],
        )
        entry["motion"].update_tag()
    bpy.context.view_layer.update()


def _render_blender_still(scene):
    import bpy

    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError as error:
        if scene.render.engine != "CYCLES" or (
            scene.cycles.device != "GPU"
            or "DEVICE_OUT_OF_MEMORY" not in str(error)
        ):
            raise
        print("[Blender] GPU out of memory; retrying render on CPU", flush=True)
        scene.cycles.device = "CPU"
        bpy.ops.render.render(write_still=True)


def _render_articulation_video(scene, entries, output):
    import bpy

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required for articulation video output")
    poses = ("rest", "lower", "upper", "rest")
    transition_frames = ARTICULATION_VIDEO_TRANSITION_FRAMES
    original_engine = scene.render.engine
    original_samples = scene.cycles.samples
    original_denoiser = scene.cycles.denoiser
    original_persistent_data = scene.render.use_persistent_data
    scene.render.engine = "CYCLES"
    scene.cycles.samples = ARTICULATION_VIDEO_CYCLES_SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = (
        "OPTIX" if scene.cycles.device == "GPU" else "OPENIMAGEDENOISE"
    )
    scene.render.use_persistent_data = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".articulation_video_frames_", dir=output,
        ) as temporary:
            frames = Path(temporary)
            for index in range(3 * transition_frames + 1):
                segment = min(index // transition_frames, 2)
                fraction = (
                    index - segment * transition_frames
                ) / transition_frames
                for entry in entries:
                    start = articulation_pose_value(
                        poses[segment], entry["lower"], entry["upper"],
                        entry["rest"],
                    )
                    end = articulation_pose_value(
                        poses[segment + 1], entry["lower"], entry["upper"],
                        entry["rest"],
                    )
                    entry["motion"]["joint_value"] = (
                        start + (end - start) * fraction
                    )
                    entry["motion"].update_tag()
                bpy.context.view_layer.update()
                scene.render.filepath = str(frames / f"frame_{index + 1:04d}.png")
                bpy.ops.render.render(write_still=True)
            subprocess.run([
                ffmpeg, "-y", "-loglevel", "error",
                "-framerate", str(ARTICULATION_VIDEO_FPS),
                "-i", str(frames / "frame_%04d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output / "front_rest_lower_upper_rest.mp4"),
            ], check=True)
    finally:
        scene.render.use_persistent_data = original_persistent_data
        scene.cycles.denoiser = original_denoiser
        scene.cycles.samples = original_samples
        scene.render.engine = original_engine
        _set_articulation_pose(entries, "rest")


def _render_articulation_visualizations(document, cache, output):
    visualization = document.get("replacement", {}).get("visualization")
    if not visualization:
        return
    import bpy

    entries = _articulation_motion_entries(document, cache)
    info = document["cameras"]["front"]
    camera, intrinsic, width, height = _configure_cached_blender_camera(
        cache, "front", info,
    )
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(intrinsic[0, 0] / intrinsic[1, 1])
    destination = output / "articulation_states"
    destination.mkdir(parents=True, exist_ok=True)
    pose_files = {
        "rest": "front_rest.png",
        "lower": "front_lower.png",
        "upper": "front_upper.png",
    }
    for pose in visualization.get("front_pose_images", pose_files):
        if pose not in pose_files:
            raise ValueError(f"unknown articulation visualization pose: {pose}")
        _set_articulation_pose(entries, pose)
        scene.render.filepath = str(destination / pose_files[pose])
        _render_blender_still(scene)
    if visualization.get("video_enabled", False):
        _render_articulation_video(scene, entries, destination)
    _set_articulation_pose(entries, "rest")
    if visualization.get("rest_joint_ranges", False):
        from pipeline.blender_assembly import (
            add_joint_visualizations, remove_joint_visualizations,
        )

        project_root = Path(__file__).resolve().parents[1]
        try:
            for object_id in document["replacement"]["articulated_object_ids"]:
                add_joint_visualizations(
                    project_root, object_id,
                    Path(document["objects"][object_id]["asset"]),
                )
            for obj in bpy.data.objects:
                if obj.get("joint_visualization"):
                    obj.hide_render = False
            scene.render.filepath = str(destination / "front_rest_joint_ranges.png")
            _render_blender_still(scene)
        finally:
            remove_joint_visualizations()


def _interval_sample_centers(low, high, spacing):
    if high <= low:
        return np.empty(0, dtype=float)
    count = max(1, int(math.ceil((high - low) / spacing)))
    step = (high - low) / count
    return low + (np.arange(count, dtype=float) + 0.5) * step


def _extend_table_support_from_z0(document, cache):
    """Extend only the +world-Y table edge under near-z=0 object contact."""
    config = document.get("replacement", {}).get("table_support_extension") or {}
    if not config.get("enabled", False):
        return None
    if cache.get("table_support_extension") is not None:
        return cache["table_support_extension"]

    import bpy
    from mathutils import Vector
    from pipeline.scene_renderer import bounds

    contact_height = float(config.get("contact_height_m", 0.01))
    spacing = float(config.get("sample_spacing_m", 0.005))
    if contact_height <= 0.0 or spacing <= 0.0:
        raise ValueError("table support extension thresholds must be positive")

    table_meshes = [obj for obj in cache["table"][1] if obj.type == "MESH"]
    table_points = [
        obj.matrix_world @ vertex.co
        for obj in table_meshes
        for vertex in obj.data.vertices
    ]
    if not table_points:
        raise ValueError("table support extension requires a table mesh")
    table_top_z = max(point.z for point in table_points)
    top_band = [
        point for point in table_points
        if point.z >= table_top_z - contact_height
    ]
    front_y = min(point.y for point in top_band)
    back_y = max(point.y for point in top_band)
    min_x = min(point.x for point in top_band)
    max_x = max(point.x for point in top_band)

    candidate_max_y = back_y
    for entry in cache["objects"].values():
        meshes = [obj for obj in entry["objects"] if obj.type == "MESH"]
        if not meshes:
            continue
        low, high = bounds(meshes)
        if low.z <= contact_height and high.z >= 0.0 and high.y > back_y:
            candidate_max_y = max(candidate_max_y, float(high.y))

    report = {
        "method": config.get("method", "upward_raycast_from_z0"),
        "contact_height_m": contact_height,
        "sample_spacing_m": spacing,
        "front_y_before_m": float(front_y),
        "back_y_before_m": float(back_y),
        "back_y_after_m": float(back_y),
        "extension_m": 0.0,
        "supporting_object_ids": [],
        "applied": False,
    }
    if candidate_max_y <= back_y:
        cache["table_support_extension"] = report
        return report

    owner_by_name = {
        obj.name: object_id
        for object_id, entry in cache["objects"].items()
        for obj in entry["objects"]
        if obj.type == "MESH"
    }
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    epsilon = 1e-5
    farthest_hit_y = None
    supporting_ids = set()
    for y in _interval_sample_centers(back_y, candidate_max_y, spacing):
        for x in _interval_sample_centers(min_x, max_x, spacing):
            hit, location, _normal, _face, obj, _matrix = scene.ray_cast(
                depsgraph,
                Vector((float(x), float(y), -epsilon)),
                Vector((0.0, 0.0, 1.0)),
                distance=contact_height + epsilon,
            )
            object_id = owner_by_name.get(obj.name) if hit and obj else None
            if object_id is None or location.z < -epsilon:
                continue
            farthest_hit_y = float(y)
            supporting_ids.add(object_id)

    if farthest_hit_y is None:
        cache["table_support_extension"] = report
        return report

    target_back_y = min(candidate_max_y, farthest_hit_y + spacing * 0.5)
    extension = max(0.0, target_back_y - back_y)
    if extension <= epsilon:
        cache["table_support_extension"] = report
        return report

    scale = (back_y + extension - front_y) / (back_y - front_y)
    for obj in table_meshes:
        inverse = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            if point.y > front_y:
                point.y = front_y + (point.y - front_y) * scale
                vertex.co = inverse @ point
        obj.data.update()
    bpy.context.view_layer.update()

    report.update({
        "back_y_after_m": float(back_y + extension),
        "extension_m": float(extension),
        "supporting_object_ids": sorted(supporting_ids),
        "applied": True,
    })
    cache["table_support_extension"] = report
    return report


def blender_render_one(
    scene_json, output_dir, save_blend=False, scene_mode="reload",
    views=("front", "top"), samples=None,
):
    import bpy
    from pipeline.blender_assembly import remove_joint_visualizations

    suppress_blender_gltf_info(bpy)
    document = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = _ensure_cached_blender_scene(document, scene_mode)
    remove_joint_visualizations()
    support_report = _extend_table_support_from_z0(document, cache)
    if support_report is not None:
        write_json(output / "table_support_extension.json", support_report)
        print(
            "[table support] "
            f"extension={support_report['extension_m'] * 1000.0:.1f}mm, "
            f"objects={support_report['supporting_object_ids']}",
            flush=True,
        )
    scene = bpy.context.scene
    configure_blender_visualization(scene)
    if samples is not None:
        scene.cycles.samples = int(samples)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(
        os.environ.get("CYCLES_VISUALIZATION_EXPOSURE", "0.6")
    )
    for view_name in views:
        info = document["cameras"][view_name]
        camera, intrinsic, width, height = _configure_cached_blender_camera(
            cache, view_name, info,
        )
        scene.camera = camera
        scene.render.resolution_x = int(width)
        scene.render.resolution_y = int(height)
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = float(intrinsic[0, 0] / intrinsic[1, 1])
        scene.render.filepath = str(output / f"{view_name}.png")
        _render_blender_still(scene)
    _render_articulation_visualizations(document, cache, output)
    if save_blend:
        final_scene_format = os.environ.get(
            "BLENDER_FINAL_SCENE_FORMAT", "glb",
        ).lower()
        if final_scene_format == "glb":
            bpy.ops.export_scene.gltf(
                filepath=str(output / "final_scene.glb"), export_format="GLB"
            )
        elif final_scene_format == "usd":
            texture_state = _prepare_blender_usd_textures(output)
            try:
                _export_blender_usd(output / "final_scene.usd", False)
            finally:
                _restore_blender_usd_textures(texture_state)
        else:
            raise ValueError("BLENDER_FINAL_SCENE_FORMAT must be glb or usd")
        sim_export_mode = os.environ.get(
            "BLENDER_SIM_EXPORT_MODE", "full",
        ).lower()
        if sim_export_mode not in {"full", "isaac", "off"}:
            raise ValueError(
                "BLENDER_SIM_EXPORT_MODE must be full, isaac, or off"
            )
        if sim_export_mode != "off":
            export_blender_sim_assets(cache, output / "sim_export",
                include_redundant_assets=sim_export_mode == "full",
            )
        from pipeline.blender_assembly import add_joint_visualizations
        project_root = Path(__file__).resolve().parents[1]
        for object_id, item in document["objects"].items():
            if item.get("asset_type") == "articulated_urdf":
                add_joint_visualizations(project_root, object_id, Path(item["asset"]))
        bpy.ops.wm.save_as_mainfile(
            filepath=str(output / "final_scene.blend"), copy=True,
        )


def configure_blender_visualization(scene):
    import bpy
    from pipeline.render_settings import configure_specular_reflections

    configure_specular_reflections(bpy, scene)

    for name in ("Key_Area", "Fill_Area"):
        light = bpy.data.objects.get(name)
        if light is not None:
            light.hide_render = True
    softbox = bpy.data.objects.get("Studio_Diffuse_Softbox")
    if softbox is not None:
        softbox.hide_render = False

    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(os.environ.get("CYCLES_VISUALIZATION_SAMPLES", "64"))
    scene.cycles.use_denoising = True
    device = os.environ.get("CYCLES_VISUALIZATION_DEVICE", "GPU").upper()
    if device not in ("CPU", "GPU"):
        raise ValueError("CYCLES_VISUALIZATION_DEVICE must be CPU or GPU")
    if device == "CPU":
        scene.cycles.device = "CPU"
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
        except TypeError:
            continue
        devices = [device for device in preferences.devices if device.type == backend]
        if devices:
            for device in preferences.devices:
                device.use = device in devices
            scene.cycles.device = "GPU"
            break


def _export_blender_usd(destination, selected_objects_only):
    import bpy

    properties = {
        prop.identifier for prop in bpy.ops.wm.usd_export.get_rna_type().properties
    }
    options = {"filepath": str(destination)}
    if "selected_objects_only" in properties:
        options["selected_objects_only"] = selected_objects_only
    if "export_textures" in properties:
        options["export_textures"] = True
    bpy.ops.wm.usd_export(**options)


def _prepare_blender_usd_textures(output):
    import bpy

    texture_dir = Path(output) / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    state = []
    for image in bpy.data.images:
        source = Path(bpy.path.abspath(image.filepath))
        if image.source != "FILE" or (
            source.suffix.lower() != ".webp" and image.file_format != "WEBP"
        ):
            continue
        state.append((image, image.filepath, image.file_format))
        destination = texture_dir / f"{image.name}.png"
        image.file_format = "PNG"
        image.save_render(str(destination))
        image.filepath = str(destination)
    return state


def _restore_blender_usd_textures(state):
    for image, filepath, file_format in state:
        image.filepath = filepath
        image.file_format = file_format


def export_blender_sim_assets(cache, output, include_redundant_assets=True):
    import bpy

    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)

    def select(objects):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0] if objects else None

    articulated_objects = {
        obj
        for entry in cache["objects"].values()
        if entry["asset_type"] == "articulated_urdf"
        for obj in (entry["controller"], *entry["objects"])
    }
    root_offsets = {}
    for object_id, entry in cache["objects"].items():
        if entry["asset_type"] != "articulated_urdf":
            continue
        rig = next(
            obj for obj in entry["objects"]
            if obj.name == f"{object_id}_urdf_root"
        )
        relative = entry["controller"].matrix_world.inverted() @ rig.matrix_world
        root_offsets[object_id] = [float(value) for value in relative.translation]
    write_json(output / "articulated_root_offsets.json", root_offsets)

    select([
        obj for obj in bpy.context.scene.objects
        if obj not in articulated_objects and not obj.get("joint_visualization")
    ])
    if include_redundant_assets:
        bpy.ops.export_scene.gltf(
            filepath=str(output / "scene_visual.glb"), export_format="GLB",
            use_selection=True,
        )
    texture_state = _prepare_blender_usd_textures(output)
    try:
        _export_blender_usd(output / "scene_visual.usd", True)

        if not include_redundant_assets:
            return

        entries = {"table_0": cache["table"], **{
            object_id: (entry["controller"], entry["objects"])
            for object_id, entry in cache["objects"].items()
        }}
        for object_id, (controller, objects) in entries.items():
            if object_id in cache["objects"] and cache["objects"][object_id]["asset_type"] == "articulated_urdf":
                continue
            selected = [controller, *objects]
            select(list(dict.fromkeys(selected)))
            bpy.ops.export_scene.gltf(
                filepath=str(output / "assets" / f"{object_id}.glb"),
                export_format="GLB", use_selection=True,
            )
            _export_blender_usd(output / "assets" / f"{object_id}.usd", True)
    finally:
        _restore_blender_usd_textures(texture_state)


def blender_render_top_templates(
    scene_json, output_dir, render_scale=1, scene_mode="reload",
):
    import bpy

    suppress_blender_gltf_info(bpy)
    document = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = _ensure_cached_blender_scene(document, scene_mode)
    imported_by_id = {
        object_id: [obj for obj in entry["objects"] if obj.type == "MESH"]
        for object_id, entry in cache["objects"].items()
    }
    info = document["cameras"]["top"]
    camera, intrinsic, width, height = _configure_cached_blender_camera(
        cache, "top", info,
    )

    scene = bpy.context.scene
    from pipeline.render_settings import configure_specular_reflections
    configure_specular_reflections(bpy, scene)
    scene.compositing_node_group = None
    for name in ("Key_Area", "Fill_Area"):
        light = bpy.data.objects.get(name)
        if light is not None:
            light.hide_render = False
    softbox = bpy.data.objects.get("Studio_Diffuse_Softbox")
    if softbox is not None:
        softbox.hide_render = True
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    if int(render_scale) < 1:
        raise ValueError("top template render scale must be at least 1")
    scene.render.resolution_x = int(width) * int(render_scale)
    scene.render.resolution_y = int(height) * int(render_scale)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(intrinsic[0, 0] / intrinsic[1, 1])
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    cache["ground"].hide_render = True
    for object_id, visible in imported_by_id.items():
        visible = set(visible)
        for obj in cache["all_meshes"]:
            obj.hide_render = obj not in visible
        scene.render.filepath = str(output / f"{object_id}.png")
        bpy.ops.render.render(write_still=True)
    for obj in cache["all_meshes"]:
        obj.hide_render = False
    cache["ground"].hide_render = False


def blender_worker(args):
    for job in args.render_job or []:
        scene_json, output_dir, save_blend = job.rsplit("::", 2)
        blender_render_one(scene_json, output_dir, bool(int(save_blend)))
    if args.top_template_job:
        scene_json, output_dir, render_scale = args.top_template_job.rsplit("::", 2)
        blender_render_top_templates(scene_json, output_dir, int(render_scale))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--blender-scene-mode",
        choices=("persistent", "reload"), default="persistent",
    )
    parser.add_argument("--stop-after-table", action="store_true")
    parser.add_argument("--stop-after-mesh-orientation", action="store_true")
    parser.add_argument("--stop-after-object-coarse", action="store_true")
    parser.add_argument("--stop-after-top-refined", action="store_true")
    parser.add_argument("--export-final-scene", action="store_true")
    parser.add_argument(
        "--top-refinement-method",
        choices=("ot", "differentiable_mask", "differentiable_edge"),
        default="differentiable_edge",
    )
    parser.add_argument(
        "--top-refinement-camera-projection",
        choices=("perspective", "orthographic"),
        default=os.environ.get("TOP_REFINEMENT_CAMERA_PROJECTION", "orthographic"),
    )
    parser.add_argument(
        "--top-fine-refinement-method",
        choices=("2d_affine_search", "2d_affine", "differentiable_3d_mask"),
        default="2d_affine",
        help=(
            "Use batched direct affine search, Adam affine refinement with the "
            "stage-three loss, or defer to the 3D mask pass."
        ),
    )
    parser.add_argument(
        "--top-model-edge-method",
        choices=("teed", "reference_image", "normal_discontinuity"),
        default="normal_discontinuity",
    )
    parser.add_argument(
        "--front-model-edge-method",
        choices=("teed", "reference_image", "normal_discontinuity"),
        default="normal_discontinuity",
    )
    parser.add_argument(
        "--top-circle-score-threshold", type=float, default=0.85,
        help=(
            "Tie final X/Y scale when the top silhouette, contact bottom, or "
            "stable body cross-sections are near-circular."
        ),
    )
    parser.add_argument(
        "--contact-square-ratio-threshold", type=float,
        default=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
        help="Require the contact-bottom rotated bounding box to be near-square.",
    )
    parser.add_argument("--top-internal-orientation-weight", type=float, default=0.0)
    parser.add_argument("--front-internal-orientation-weight", type=float, default=0.0)
    parser.add_argument(
        "--top-internal-edge-match-mode",
        choices=("chamfer", "partial_hausdorff", "centroid_offset"),
        default=os.environ.get(
            "TOP_INTERNAL_EDGE_MATCH_MODE", "chamfer",
        ),
        help=(
            "Match internal edges with nearest-neighbor Chamfer (default), "
            "partial Hausdorff, or signed centroid offsets."
        ),
    )
    parser.add_argument(
        "--top-square-bbox-ratio", dest="top_circle_score_threshold",
        type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--top-yaw-mask-iou-loss-weight", type=float,
        default=float(os.environ.get("TOP_YAW_MASK_IOU_LOSS_WEIGHT", "1.0")),
    )
    parser.add_argument(
        "--top-object-optimization-mode",
        choices=("joint", "independent_parallel"), default="independent_parallel",
    )
    parser.add_argument(
        "--asset-source", choices=("processed", "raw", "articulated"), default="processed",
    )
    parser.add_argument("--top-scale-mode", choices=("uniform", "xy"), default="uniform")
    parser.add_argument("--top-template-render-scale", type=int, default=2)
    parser.add_argument(
        "--edge-detector",
        choices=("canny", "edge_drawing", "teed", "ffmpeg_edgedetect", "geometry"),
        default=os.environ.get("EDGE_DETECTOR", "teed"),
        help="Reference-image edge detector used by stage-2/3 internal-edge losses.",
    )
    parser.add_argument(
        "--minima-root", type=Path,
        default=Path(os.environ.get("MINIMA_ROOT", DEFAULT_MINIMA_ROOT)),
    )
    parser.add_argument(
        "--minima-device", default=os.environ.get("MINIMA_DEVICE"),
        required="MINIMA_DEVICE" not in os.environ,
    )
    parser.add_argument("--front-refine-resolution", type=int, default=256)
    parser.add_argument(
        "--front-refine-loss",
        choices=("mask", "edge_drawing", "mask_edge"), default="mask",
    )
    parser.add_argument(
        "--front-refine-second-round-loss",
        choices=("mask", "edge_drawing", "mask_edge"), default=None,
    )
    parser.add_argument("--front-refine-rounds", type=int, default=1)
    parser.add_argument(
        "--front-refine-second-round-mode",
        choices=("sequential", "joint"), default="joint",
    )
    support_pose = parser.add_mutually_exclusive_group()
    support_pose.add_argument(
        "--support-pose-refinement",
        dest="support_pose_refinement", action="store_true", default=True,
        help=(
            "Initialize supported-object Z after top refinement and optimize "
            "Z/roll-X only during the final front-refinement substage."
        ),
    )
    support_pose.add_argument(
        "--no-support-pose-refinement",
        dest="support_pose_refinement", action="store_false",
    )
    parser.add_argument(
        "--front-object-optimization-mode",
        choices=("joint", "independent_parallel"), default="independent_parallel",
    )
    parser.add_argument(
        "--front-yaw-search", choices=("single", "bidirectional"),
        default="bidirectional",
    )
    parser.add_argument("--front-yaw-branch-offset-deg", type=float, default=180.0)
    parser.add_argument(
        "--front-yaw-rgb-weight", type=float, default=0.0,
        help=(
            "Use low-sample front RGB renders only to rank optimized yaw branches; "
            "0 restores IoU-only branch selection."
        ),
    )
    parser.add_argument(
        "--front-yaw-top-iou-ambiguity-threshold", type=float, default=0.02,
        help=(
            "Allow both first-round front yaw branches only when the final top-view "
            "0/180 mask-IoU gap is at most this value."
        ),
    )
    parser.add_argument(
        "--front-yaw-edge-override-margin", type=float, default=0.01,
        help=(
            "Reopen the opposite first-round front yaw branch when its front-view "
            "geometry-edge loss improves by at least this relative margin."
        ),
    )
    parser.add_argument("--front-refine-iterations", type=int, default=200)
    parser.add_argument("--front-refine-fine-resolution", type=int, default=512)
    parser.add_argument(
        "--front-refine-second-round-resolution", type=int, default=1024,
    )
    parser.add_argument("--front-refine-fine-iterations", type=int, default=100)
    parser.add_argument("--front-refine-lr-xy", type=float, default=0.03)
    parser.add_argument("--front-refine-lr-yaw", type=float, default=0.03)
    parser.add_argument("--front-refine-lr-scale", type=float, default=0.02)
    parser.add_argument("--front-refine-max-xy-fraction", type=float, default=0.10)
    parser.add_argument("--front-refine-scale-limit", type=float, default=2.0)
    parser.add_argument("--front-refine-gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--early-stop-patience", "--front-refine-patience",
        dest="front_refine_patience", type=int, default=30,
        help="Stop after this many non-improving steps in GPU refinement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=1e-4,
        help="Minimum absolute loss decrease that resets early-stopping patience.",
    )
    parser.add_argument("--front-loss-sdf-weight", type=float, default=1.0)
    parser.add_argument("--front-loss-iou-weight", type=float, default=1.0)
    parser.add_argument("--front-edge-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--front-edge-parameter-scope", choices=("all", "yaw_only"),
        default=os.environ.get("FRONT_EDGE_PARAMETER_SCOPE", "all"),
        help="Apply the front edge loss to every pose parameter or yaw only.",
    )
    parser.add_argument(
        "--bbox-loss-weight", type=float, default=0.1,
        help="Projected bounding-box loss weight for top/front refinement; 0 restores the old loss.",
    )
    parser.add_argument(
        "--bbox-center-weight", type=float, default=1.0,
        help="Center term inside bbox loss; 0 keeps size-only bbox matching.",
    )
    parser.add_argument("--front-loss-xy-weight", type=float, default=0.005)
    parser.add_argument("--front-loss-scale-weight", type=float, default=0.005)
    parser.add_argument("--front-loss-yaw-weight", type=float, default=0.0001)
    parser.add_argument("--refinement-rgb-loss-weight", type=float, default=0.0)
    parser.add_argument("--blender-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scene-json", help=argparse.SUPPRESS)
    parser.add_argument("--save-blend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-job", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--top-template-job", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Standalone accelerated implementations
# ---------------------------------------------------------------------------

RANSAC_ITERATIONS = int(os.environ.get("FAST_RANSAC_ITERATIONS", "300"))
TABLE_SAMPLE_COUNT = int(os.environ.get("FAST_TABLE_SAMPLE_COUNT", "8000"))
OBJECT_SAMPLE_COUNT = int(os.environ.get("FAST_OBJECT_SAMPLE_COUNT", "5000"))
TARGET_POINT_LIMIT = int(os.environ.get("FAST_TARGET_POINT_LIMIT", "3000"))
POWELL_MAX_ITERATIONS = int(os.environ.get("FAST_POWELL_MAX_ITERATIONS", "30"))
GPU_POINTCLOUD_ITERATIONS = int(os.environ.get("GPU_POINTCLOUD_ITERATIONS", "120"))
GPU_POINTCLOUD_LR = float(os.environ.get("GPU_POINTCLOUD_LR", "0.05"))
OT_POINT_COUNT = int(os.environ.get("FAST_OT_POINT_COUNT", "1024"))
FRONT_YAW_EDGE_POINT_COUNT = int(
    os.environ.get("FAST_FRONT_YAW_EDGE_POINT_COUNT", "1024")
)
OT_SINKHORN_ITERATIONS = int(os.environ.get("FAST_OT_SINKHORN_ITERATIONS", "24"))
MASK_OT_ITERATIONS = int(os.environ.get("FAST_MASK_OT_ITERATIONS", "18"))
EDGE_OT_ITERATIONS = int(os.environ.get("FAST_EDGE_OT_ITERATIONS", "12"))


if not BLENDER_WORKER:
    _fit_plane_ransac = fit_plane_ransac
    _refine_front_table_registration = refine_front_table_registration
_blender_worker = None
_blender_scene_mode = "persistent"


class PersistentBlenderWorker:
    def __init__(self):
        self.socket_path = Path(f"/tmp/tabletop_blender_{os.getpid()}.sock")
        self.socket_path.unlink(missing_ok=True)
        self.process = subprocess.Popen(
            [
                blender_executable(), "--background", "--factory-startup",
                "--python", str(Path(__file__).with_name("blender_worker.py")),
                "--", "--socket", str(self.socket_path),
                "--parent-pid", str(os.getpid()),
            ],
            cwd=ROOT,
            env=blender_environment(),
            stdout=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 60.0
        while not self.socket_path.exists():
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"persistent Blender worker exited with code {return_code}")
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError("persistent Blender worker did not create its socket")
            time.sleep(0.05)

    def request(self, payload):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.socket_path))
            connection.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            chunks = []
            while True:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        response_bytes = b"".join(chunks).split(b"\n", 1)[0]
        if not response_bytes:
            raise RuntimeError(
                "persistent Blender worker exited without a response "
                f"(exit code {self.process.poll()}); "
                "check /tmp/blender.crash.txt"
            )
        response = json.loads(response_bytes)
        if not response.get("ok"):
            raise RuntimeError(
                f"persistent Blender worker failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({"command": "quit"})
                self.process.wait(timeout=30)
            except Exception:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.socket_path.unlink(missing_ok=True)


def _persistent_blender_worker():
    global _blender_worker
    if _blender_worker is None:
        _blender_worker = PersistentBlenderWorker()
    return _blender_worker


def close_persistent_blender_worker():
    global _blender_worker
    if _blender_worker is not None:
        _blender_worker.close()
        _blender_worker = None


def persistent_render_stages(jobs):
    if _blender_worker is not None and any(save_blend for _, _, save_blend in jobs):
        close_persistent_blender_worker()
    _persistent_blender_worker().request({
        "command": "render_jobs",
        "scene_mode": _blender_scene_mode,
        "jobs": [
            [str(scene_json), str(output_dir), bool(save_blend)]
            for scene_json, output_dir, save_blend in jobs
        ],
    })
    for scene_json, output_dir, _ in jobs:
        save_stage_alignment_visualizations(scene_json, output_dir)


def persistent_render_top_edge_templates(scene_json, output_dir, render_scale=1):
    _persistent_blender_worker().request({
        "command": "top_templates",
        "scene_mode": _blender_scene_mode,
        "scene_json": str(scene_json),
        "output_dir": str(output_dir),
        "render_scale": int(render_scale),
    })


def fast_fit_plane_ransac(points, threshold, iterations=1500, seed=0):
    return _fit_plane_ransac(
        points, threshold,
        iterations=min(int(iterations), RANSAC_ITERATIONS), seed=seed,
    )


def fast_refine_front_table_registration(
    view, table_mesh_path, table_scale_xyz, sample_count=30000,
    table_yaw_deg=0.0,
):
    return _refine_front_table_registration(
        view, table_mesh_path, table_scale_xyz,
        sample_count=min(int(sample_count), TABLE_SAMPLE_COUNT),
        table_yaw_deg=table_yaw_deg,
    )


def fast_refine_pose_to_pointcloud(mesh_path, initial, cloud):
    """Query one canonical KDTree instead of rebuilding it every evaluation."""
    if cloud is None or not len(cloud):
        return initial

    target = np.asarray(cloud, dtype=float)
    target = target[np.isfinite(target).all(axis=1)]
    if len(target) > TARGET_POINT_LIMIT:
        indices = np.linspace(0, len(target) - 1, TARGET_POINT_LIMIT, dtype=int)
        target = target[indices]

    import trimesh

    mesh = search_mesh_z_up(mesh_path, face_count=6000).copy()
    mesh.vertices = center_vertices_on_table(mesh.vertices)
    canonical, _ = trimesh.sample.sample_surface(
        mesh, OBJECT_SAMPLE_COUNT, seed=0,
    )
    model_tree = cKDTree(canonical)
    p0 = np.array([
        initial["translation_world_m"][0],
        initial["translation_world_m"][1],
        initial["yaw_deg"],
        math.log(initial["uniform_scale"]),
    ])

    def objective(parameters):
        scale = math.exp(parameters[3])
        angle = math.radians(parameters[2])
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        local_target = target.copy()
        local_target[:, :2] -= parameters[:2]
        local_target[:, :2] = local_target[:, :2] @ rotation
        local_target /= scale
        distances = model_tree.query(local_target, k=1, workers=-1)[0] * scale
        keep = max(1, int(0.9 * len(distances)))
        data_loss = float(np.partition(distances, keep - 1)[:keep].mean())
        xy_regularization = 0.25 * float(np.linalg.norm(parameters[:2] - p0[:2]))
        return data_loss + xy_regularization

    result = minimize(
        objective,
        p0,
        method="Powell",
        bounds=[
            (p0[0] - 0.025, p0[0] + 0.025),
            (p0[1] - 0.025, p0[1] + 0.025),
            (p0[2] - 45.0, p0[2] + 45.0),
            (p0[3] + math.log(0.85), p0[3] + math.log(1.15)),
        ],
        options={
            "maxiter": POWELL_MAX_ITERATIONS,
            "xtol": 1e-4,
            "ftol": 1e-4,
        },
    )
    value = result.x
    return {
        **initial,
        "translation_world_m": [float(value[0]), float(value[1]), 0.0],
        "yaw_deg": float(((value[2] + 180.0) % 360.0) - 180.0),
        "uniform_scale": float(math.exp(value[3])),
        "pointcloud_loss": float(result.fun),
        "pointcloud_refinement_backend": "scipy",
    }


def batched_pointcloud_losses(canonical, target, target_valid, xy, yaw_deg, log_scale):
    """One-way 90%-trimmed Chamfer loss matching the SciPy coarse objective."""
    import torch

    scale = torch.exp(log_scale)
    angle = torch.deg2rad(yaw_deg)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((
        torch.stack((cosine, -sine), dim=1),
        torch.stack((sine, cosine), dim=1),
    ), dim=1)
    local_target = target.clone()
    local_target[:, :, :2] = torch.bmm(
        local_target[:, :, :2] - xy[:, None, :], rotation,
    )
    local_target = local_target / scale[:, None, None]
    nearest = []
    for start in range(0, local_target.shape[1], 512):
        chunk = local_target[:, start:start + 512]
        nearest.append(torch.cdist(chunk, canonical).amin(dim=2))
    distances = torch.cat(nearest, dim=1) * scale[:, None]
    distances = torch.where(
        target_valid, distances, torch.full_like(distances, float("inf")),
    )
    ordered = distances.sort(dim=1).values
    counts = target_valid.sum(dim=1)
    keep = torch.clamp((counts.float() * 0.9).floor().long(), min=1)
    ranks = torch.arange(ordered.shape[1], device=ordered.device)[None, :]
    retained = ranks < keep[:, None]
    return torch.where(retained, ordered, torch.zeros_like(ordered)).sum(dim=1) / keep


def refine_poses_to_pointcloud_gpu(
    items, *, early_stop_patience=20, early_stop_min_delta=1e-4,
):
    """Refine all object poses together with one batched CUDA optimizer."""
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError("GPU point-cloud refinement requires CUDA")
    prepared = []
    results = [initial for _, initial, _ in items]
    for result_index, (mesh_path, initial, cloud) in enumerate(items):
        if cloud is None or not len(cloud):
            continue
        target = np.asarray(cloud, dtype=np.float32)
        target = target[np.isfinite(target).all(axis=1)]
        if not len(target):
            continue
        if len(target) > TARGET_POINT_LIMIT:
            indices = np.linspace(0, len(target) - 1, TARGET_POINT_LIMIT, dtype=int)
            target = target[indices]
        mesh = search_mesh_z_up(mesh_path, face_count=6000).copy()
        mesh.vertices = center_vertices_on_table(mesh.vertices)
        canonical, _ = trimesh.sample.sample_surface(
            mesh, OBJECT_SAMPLE_COUNT, seed=0,
        )
        prepared.append((result_index, initial, canonical.astype(np.float32), target))

    if not prepared:
        return results

    device = torch.device("cuda")
    batch_size = len(prepared)
    target_count = max(len(item[3]) for item in prepared)
    canonical = torch.as_tensor(
        np.stack([item[2] for item in prepared]), device=device,
    )
    target = torch.zeros((batch_size, target_count, 3), device=device)
    target_valid = torch.zeros((batch_size, target_count), dtype=torch.bool, device=device)
    p0 = torch.as_tensor(np.array([
        [
            item[1]["translation_world_m"][0],
            item[1]["translation_world_m"][1],
            item[1]["yaw_deg"],
            math.log(item[1]["uniform_scale"]),
        ]
        for item in prepared
    ], dtype=np.float32), device=device)
    for index, (_, _, _, points) in enumerate(prepared):
        count = len(points)
        target[index, :count] = torch.as_tensor(points, device=device)
        target_valid[index, :count] = True

    raw_xy = torch.nn.Parameter(torch.zeros((batch_size, 2), device=device))
    raw_yaw = torch.nn.Parameter(torch.zeros(batch_size, device=device))
    scale_low, scale_high = math.log(0.85), math.log(1.15)
    scale_fraction = -scale_low / (scale_high - scale_low)
    scale_initial = math.log(scale_fraction / (1.0 - scale_fraction))
    raw_scale = torch.nn.Parameter(torch.full((batch_size,), scale_initial, device=device))
    parameters = (raw_xy, raw_yaw, raw_scale)
    optimizer = torch.optim.Adam(parameters, lr=GPU_POINTCLOUD_LR)

    def state():
        xy = p0[:, :2] + 0.025 * torch.tanh(raw_xy)
        yaw = p0[:, 2] + 45.0 * torch.tanh(raw_yaw)
        scale_delta = scale_low + (scale_high - scale_low) * torch.sigmoid(raw_scale)
        return xy, yaw, p0[:, 3] + scale_delta

    best_loss = torch.full((batch_size,), float("inf"), device=device)
    best_state = None
    best_mean_loss = float("inf")
    stale_steps = 0
    completed_steps = 0
    for _ in range(GPU_POINTCLOUD_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        xy, yaw, log_scale = state()
        losses = batched_pointcloud_losses(
            canonical, target, target_valid, xy, yaw, log_scale,
        ) + 0.25 * torch.linalg.vector_norm(xy - p0[:, :2], dim=1)
        improved = losses.detach() < best_loss
        if torch.any(improved):
            candidate = torch.cat((xy, yaw[:, None], log_scale[:, None]), dim=1).detach()
            best_state = (
                candidate.clone() if best_state is None
                else torch.where(improved[:, None], candidate, best_state)
            )
            best_loss = torch.where(improved, losses.detach(), best_loss)
        best_mean_loss, stale_steps = early_stopping_update(
            float(losses.detach().mean().cpu()), best_mean_loss, stale_steps,
            early_stop_min_delta,
        )
        losses.mean().backward()
        optimizer.step()
        completed_steps += 1
        if early_stop_patience > 0 and stale_steps >= early_stop_patience:
            print(
                f"[gpu pointcloud] early stop at {completed_steps}/"
                f"{GPU_POINTCLOUD_ITERATIONS} steps",
                flush=True,
            )
            break

    values = best_state.cpu().numpy()
    losses = best_loss.cpu().numpy()
    for row, (result_index, initial, _, _) in enumerate(prepared):
        value = values[row]
        results[result_index] = {
            **initial,
            "translation_world_m": [float(value[0]), float(value[1]), 0.0],
            "yaw_deg": float(((value[2] + 180.0) % 360.0) - 180.0),
            "uniform_scale": float(math.exp(value[3])),
            "pointcloud_loss": float(losses[row]),
            "pointcloud_refinement_backend": "gpu",
            "pointcloud_refinement_steps": int(completed_steps),
        }
    return results


def _sample_mask_points(mask, count=OT_POINT_COUNT):
    points = np.column_stack(np.where(np.asarray(mask) > 0))[:, ::-1].astype(float)
    if len(points) > count:
        points = points[np.linspace(0, len(points) - 1, count, dtype=int)]
    return points


def _transform_points(points, transform):
    points = np.asarray(points, dtype=float)
    return points @ transform[:2, :2].T + transform[:2, 2]


def _balanced_sinkhorn_cost(source, target, normalization, epsilon=0.02):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    cost = np.sum(((source[:, None] - target[None, :]) / normalization) ** 2, axis=2)
    kernel = np.exp(-cost / epsilon).clip(1e-100, None)
    source_mass = np.full(len(source), 1.0 / len(source))
    target_mass = np.full(len(target), 1.0 / len(target))
    left = np.ones_like(source_mass)
    right = np.ones_like(target_mass)
    for _ in range(OT_SINKHORN_ITERATIONS):
        left = source_mass / (kernel @ right + 1e-100)
        right = target_mass / (kernel.T @ left + 1e-100)
    transport = left[:, None] * kernel * right[None, :]
    reference = source_mass[:, None] * target_mass[None, :]
    entropy = np.sum(
        transport * np.log((transport + 1e-100) / (reference + 1e-100))
        - transport + reference
    )
    return float(np.sum(transport * cost) + epsilon * entropy)


def _sinkhorn_divergence(source, target, normalization, target_self=None):
    if not len(source) or not len(target):
        return 0.0
    cross = _balanced_sinkhorn_cost(source, target, normalization)
    source_self = _balanced_sinkhorn_cost(source, source, normalization)
    if target_self is None:
        target_self = _balanced_sinkhorn_cost(target, target, normalization)
    return float(max(0.0, cross - 0.5 * source_self - 0.5 * target_self))


def _unit_interval_float(name, value, tolerance=1e-6):
    value = float(value)
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise FloatingPointError(f"{name} must be finite and in [0, 1], got {value}")
    return float(np.clip(value, 0.0, 1.0))


def _nonnegative_finite_float(name, value, tolerance=1e-6):
    value = float(value)
    if not math.isfinite(value) or value < -tolerance:
        raise FloatingPointError(f"{name} must be finite and non-negative, got {value}")
    return max(value, 0.0)


def stage2_yaw_loss_multipliers(stage2_yaw_report, yaw_offsets_deg):
    """Return the stage-2 loss for each relative stage-3 yaw branch.

    Stage 3 starts from the yaw selected by stage 2, so its 0/90/180/270-degree
    branches correspond to stage-2 landscape angles selected_yaw + offset.
    """
    offsets = tuple(float(offset) for offset in yaw_offsets_deg)
    if offsets == (0.0, 180.0):
        current_loss = stage2_yaw_report.get("current_direction_loss")
        opposite_loss = stage2_yaw_report.get("opposite_direction_loss")
        if current_loss is not None and opposite_loss is not None:
            return (
                _nonnegative_finite_float(
                    "stage-2 current direction loss", current_loss,
                ),
                _nonnegative_finite_float(
                    "stage-2 opposite direction loss", opposite_loss,
                ),
            )

    landscape = stage2_yaw_report.get("yaw_landscape", {})
    try:
        yaw_degrees = np.asarray(landscape["yaw_deg"], dtype=float)
        total_losses = np.asarray(landscape["total_loss"], dtype=float)
        selected_yaw = float(stage2_yaw_report["coarse_selected_yaw_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        yaw_degrees.ndim != 1
        or total_losses.shape != yaw_degrees.shape
        or not len(yaw_degrees)
        or not np.isfinite(yaw_degrees).all()
    ):
        return None

    multipliers = []
    for offset in offsets:
        target_yaw = selected_yaw + offset
        index = int(np.argmin(np.abs(
            (yaw_degrees - target_yaw + 180.0) % 360.0 - 180.0,
        )))
        multipliers.append(_nonnegative_finite_float(
            f"stage-2 yaw loss at offset {offset:g} degrees", total_losses[index],
        ))
    return tuple(multipliers)


def _mask_iou(template_mask, target, transform):
    warped = cv2.warpAffine(
        np.asarray(template_mask, dtype=np.uint8), transform[:2],
        (template_mask.shape[1], template_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    target = np.asarray(target) > 0
    union = np.count_nonzero(warped | target)
    return _unit_interval_float(
        "mask IoU", np.count_nonzero(warped & target) / max(union, 1),
    )


def _warped_normalized_gradient_loss(
    source_rgb, source_mask, target_rgb, target_mask, transform,
):
    height, width = np.asarray(target_mask).shape
    warped_rgb = cv2.warpAffine(
        np.asarray(source_rgb, dtype=np.uint8), transform[:2],
        (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_mask = cv2.warpAffine(
        np.asarray(source_mask, dtype=np.uint8), transform[:2],
        (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    return normalized_gradient_loss(
        warped_rgb, target_rgb, warped_mask, target_mask,
    )


def _mask_rgb_product_objective(
    source_rgb, source_mask, target_rgb, target_mask,
    source_center, target_center, scale_ratio, max_resolution=None,
):
    """Build a native-pixel ROI 1 - IoU * RGB-similarity yaw loss."""
    import torch
    import torch.nn.functional as functional

    source_mask = (np.asarray(source_mask) > 0).astype(np.float32)
    target_mask = (np.asarray(target_mask) > 0).astype(np.float32)
    source_center = np.asarray(source_center, dtype=np.float32)
    target_center = np.asarray(target_center, dtype=np.float32)
    source_y, source_x = np.nonzero(source_mask)
    target_y, target_x = np.nonzero(target_mask)
    source_radius = int(np.ceil(np.hypot(
        source_x - source_center[0], source_y - source_center[1],
    ).max())) + 2
    target_radius = int(np.ceil(np.hypot(
        target_x - target_center[0], target_y - target_center[1],
    ).max())) + 2
    output_radius = max(target_radius, int(np.ceil(source_radius * scale_ratio)) + 2)
    full_height, full_width = target_mask.shape
    left = max(0, int(np.floor(min(
        source_center[0] - source_radius, target_center[0] - output_radius,
    ))))
    top = max(0, int(np.floor(min(
        source_center[1] - source_radius, target_center[1] - output_radius,
    ))))
    right = min(full_width, int(np.ceil(max(
        source_center[0] + source_radius, target_center[0] + output_radius,
    ))) + 1)
    bottom = min(full_height, int(np.ceil(max(
        source_center[1] + source_radius, target_center[1] + output_radius,
    ))) + 1)
    source_mask = source_mask[top:bottom, left:right]
    target_mask = target_mask[top:bottom, left:right]
    source_rgb = np.asarray(source_rgb, dtype=np.float32)[top:bottom, left:right] / 255.0
    target_rgb = np.asarray(target_rgb, dtype=np.float32)[top:bottom, left:right] / 255.0
    source_center -= (left, top)
    target_center -= (left, top)
    height, width = target_mask.shape
    resize = (
        min(1.0, float(max_resolution) / max(height, width))
        if max_resolution is not None else 1.0
    )
    out_width = max(2, int(round(width * resize)))
    out_height = max(2, int(round(height * resize)))
    size = (out_width, out_height)
    source_mask = cv2.resize(source_mask, size, interpolation=cv2.INTER_AREA)
    target_mask = cv2.resize(target_mask, size, interpolation=cv2.INTER_AREA)
    source_rgb = cv2.resize(source_rgb, size, interpolation=cv2.INTER_AREA)
    target_rgb = cv2.resize(target_rgb, size, interpolation=cv2.INTER_AREA)
    source_center = (source_center + 0.5) * resize - 0.5
    target_center = (target_center + 0.5) * resize - 0.5
    cached = {}

    def batch_metrics(yaw):
        device = yaw.device
        if device not in cached:
            ys, xs = torch.meshgrid(
                torch.arange(out_height, device=device, dtype=torch.float32),
                torch.arange(out_width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            cached[device] = (
                torch.as_tensor(source_rgb, device=device).permute(2, 0, 1)[None],
                torch.as_tensor(source_mask, device=device)[None, None],
                torch.as_tensor(target_rgb, device=device).permute(2, 0, 1),
                torch.as_tensor(target_mask, device=device),
                xs - float(target_center[0]), ys - float(target_center[1]),
            )
        source_image, source_alpha, target_image, target_alpha, dx, dy = cached[device]
        scalar = yaw.ndim == 0
        yaw = yaw.reshape(-1, 1, 1)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        source_x = float(source_center[0]) + (cosine * dx - sine * dy) / scale_ratio
        source_y = float(source_center[1]) + (sine * dx + cosine * dy) / scale_ratio
        grid = torch.stack((
            2.0 * source_x / max(out_width - 1, 1) - 1.0,
            2.0 * source_y / max(out_height - 1, 1) - 1.0,
        ), dim=-1)
        batch_size = grid.shape[0]
        warped_mask = functional.grid_sample(
            source_alpha.expand(batch_size, -1, -1, -1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, 0]
        warped_rgb = functional.grid_sample(
            source_image.expand(batch_size, -1, -1, -1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        intersection = torch.sum(warped_mask * target_alpha, dim=(1, 2))
        union = (
            torch.sum(warped_mask, dim=(1, 2)) + torch.sum(target_alpha) - intersection
        )
        iou = intersection / union.clamp_min(1e-6)
        overlap = warped_mask * target_alpha
        rgb_loss = torch.sum(
            torch.abs(warped_rgb - target_image[None]) * overlap[:, None],
            dim=(1, 2, 3),
        ) / (3.0 * overlap.sum(dim=(1, 2))).clamp_min(1e-6)
        product_loss = 1.0 - iou * (1.0 - rgb_loss)
        values = (iou, rgb_loss, product_loss.clamp(0.0, 1.0))
        return tuple(value[0] if scalar else value for value in values)

    def objective(yaw):
        iou, rgb_loss, product_loss = batch_metrics(yaw)
        names = ("differentiable mask IoU", "differentiable RGB loss", "product loss")
        if yaw.ndim == 0:
            for name, value in zip(
                names, torch.stack((iou, rgb_loss, product_loss)).detach().cpu().tolist(),
            ):
                _unit_interval_float(name, value, tolerance=1e-5)
        return product_loss

    objective.batch_metrics = batch_metrics
    objective.batch_pixel_count = out_width * out_height
    return objective


def rendered_front_rgb_branch_losses(
    branches, table_mesh, table_scale, cameras, front_camera,
    front_labels, front_ids, target_rgb, output_dir, table_yaw_deg=0.0,
):
    """Render each final yaw branch once and compare true front-view object RGB."""
    output_dir = Path(output_dir)
    jobs = []
    for offset, objects, _ in branches:
        branch_dir = output_dir / f"yaw_{offset:g}"
        scene_json = branch_dir / "scene.json"
        write_json(scene_json, scene_document(
            table_mesh, table_scale, cameras, objects,
            table_yaw_deg=table_yaw_deg,
        ))
        jobs.append((scene_json, branch_dir))
    render_front_stages(jobs)

    expected_size = tuple(map(int, front_camera["image_size"]))
    target_rgb = np.asarray(target_rgb, dtype=np.float32)
    if target_rgb.shape[1::-1] != expected_size:
        target_rgb = cv2.resize(target_rgb, expected_size, interpolation=cv2.INTER_LANCZOS4)
    losses = {}
    for (offset, objects, _), (_, branch_dir) in zip(branches, jobs):
        rendered = np.asarray(
            Image.open(branch_dir / "front.png").convert("RGB"), dtype=np.float32,
        )
        for object_id, item in objects.items():
            if object_id not in front_ids:
                continue
            mesh = search_mesh_z_up(Path(item["geometry_asset"]), face_count=3000)
            vertices = center_vertices_on_table(np.asarray(mesh.vertices))
            posed = pose_vertices(
                vertices, item["uniform_scale"], item["translation_world_m"][:2],
                item["yaw_deg"], scale_xyz=item.get("scale_xyz"),
            )
            overlap = (
                render_mesh_mask(mesh, posed, front_camera)
                & (target_mask(front_labels, front_ids, object_id) > 0)
            )
            if np.count_nonzero(overlap) < 16:
                continue
            loss = np.mean(np.abs(rendered[overlap] - target_rgb[overlap])) / 255.0
            losses.setdefault(object_id, {})[offset] = _unit_interval_float(
                "front yaw branch RGB loss", loss,
            )
    return {
        object_id: values for object_id, values in losses.items()
        if len(values) == len(branches)
    }


def direction_front_branch_losses(
    branches, scorer, front_camera, front_rgb, front_masks, *,
    top_camera=None, top_rgb=None, top_masks=None,
):
    """Score final front-yaw branches with the selected direction backend."""
    uses_multiview = hasattr(scorer, "score_multiview_candidates")
    cameras = {"front": direction_scoring_camera(scorer, front_camera)}
    if uses_multiview:
        if top_camera is None or top_rgb is None or top_masks is None:
            raise ValueError("front branch multiview direction context is incomplete")
        cameras["top"] = direction_scoring_camera(scorer, top_camera)
    inputs = {}
    for offset, objects, _ in branches:
        for object_id, item in objects.items():
            if object_id not in front_masks:
                continue
            mesh = search_mesh_z_up(
                Path(item["geometry_asset"]), face_count=None,
            )
            colors = item_vertex_texture_colors(mesh, item)
            if uses_multiview:
                values = inputs.setdefault(
                    object_id, {"front": {}, "top": {}},
                )
                for view in ("front", "top"):
                    values[view][offset] = rendered_direction_candidate(
                        mesh, item, cameras[view], colors, scorer,
                    )
            else:
                inputs.setdefault(object_id, {})[offset] = (
                    rendered_direction_candidate(
                        mesh, item, cameras["front"], colors, scorer,
                    )
                )
    if uses_multiview:
        losses = {
            object_id: score_multiview_direction_candidates(
                scorer, "front", {"front": front_rgb, "top": top_rgb},
                {
                    "front": front_masks[object_id],
                    "top": top_masks[object_id],
                },
                cameras, candidates,
            )
            for object_id, candidates in inputs.items()
        }
    else:
        losses = {
            object_id: score_direction_candidates(
                scorer, "front", front_rgb, front_masks[object_id],
                cameras["front"], candidates,
            )
            for object_id, candidates in inputs.items()
        }
    return {
        object_id: values for object_id, values in losses.items()
        if len(values) == len(branches)
    }


def load_front_branch_checkpoints(root, branch_specs, expected_objects):
    root = Path(root)
    expected_ids = set(expected_objects)
    branches = []
    for offset, _ in branch_specs:
        name = f"yaw_{offset:g}"
        scene_path = root / "rgb_branch_renders" / name / "scene.json"
        parameters_path = root / "round_1" / name / "parameters.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
        objects = scene["objects"]
        report = parameters["report"]
        if set(objects) != expected_ids or set(report["objects"]) != expected_ids:
            raise ValueError(f"front branch checkpoint object IDs do not match: {root}")
        branches.append((offset, objects, report))
    return branches


def _mask_geometry(mask):
    ys, xs = np.where(np.asarray(mask) > 0)
    if not len(xs):
        raise ValueError("cannot align an empty object mask")
    return np.array([float(xs.mean()), float(ys.mean())]), float(len(xs))


def pca_gated_yaw_candidates(
    source_mask, target_mask, *, step_degrees=2.0,
    anisotropy_threshold=0.2, window_degrees=10.0,
):
    """Return two PCA-aligned yaw branches when both mask axes are reliable."""
    def principal_axis(mask):
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        if len(xs) < 3:
            return 0.0, 0.0
        (_, _), (width, height), angle = cv2.minAreaRect(
            np.column_stack((xs, ys)).astype(np.float32)
        )
        longest, shortest = max(width, height), min(width, height)
        axis_degrees = angle if width >= height else angle + 90.0
        confidence = 1.0 - shortest / longest if longest > 0.0 else 0.0
        return math.radians(axis_degrees), confidence

    source_angle, source_confidence = principal_axis(source_mask)
    target_angle, target_confidence = principal_axis(target_mask)
    reliable = min(source_confidence, target_confidence) >= anisotropy_threshold
    center = (source_angle - target_angle) % math.pi
    report = {
        "used": bool(reliable),
        "source_axis_angle_deg": float(math.degrees(source_angle) % 180.0),
        "target_axis_angle_deg": float(math.degrees(target_angle) % 180.0),
        "relative_axis_yaw_deg": float(math.degrees(center)),
        "source_anisotropy": float(source_confidence),
        "target_anisotropy": float(target_confidence),
        "anisotropy_threshold": float(anisotropy_threshold),
        "anisotropy_gate": "both",
        "axis_estimator": "minimum_area_rotated_bbox",
        "window_degrees": float(window_degrees),
        "fallback": None if reliable else "full_360_axis_not_reliable",
    }
    if not reliable:
        return None, report
    offsets = np.arange(
        -float(window_degrees),
        float(window_degrees) + 0.5 * float(step_degrees),
        float(step_degrees),
    )
    center_degrees = math.degrees(center)
    centers_degrees = tuple(
        float(value % 360.0) for value in (center_degrees, center_degrees + 180.0)
    )
    degrees = np.concatenate((
        center_degrees + offsets,
        center_degrees + 180.0 + offsets,
    )) % 360.0
    candidates = tuple(np.deg2rad(np.unique(np.round(degrees, 8))))
    report["candidate_count"] = len(candidates)
    report["candidate_window_centers_deg"] = list(centers_degrees)
    return candidates, report


def stage2_direction_yaw_candidates(source_mask, target_mask):
    return pca_gated_yaw_candidates(
        source_mask, target_mask, step_degrees=2.0,
    )


def _screen_rotation(yaw_radians, scale=1.0):
    cosine, sine = math.cos(yaw_radians), math.sin(yaw_radians)
    return float(scale) * np.array([[cosine, sine], [-sine, cosine]])


def _analytic_template_transform(
    source_centroid, target_centroid, yaw_radians, scale_ratio,
):
    linear = _screen_rotation(yaw_radians, scale_ratio)
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = target_centroid - linear @ source_centroid
    return transform


def _normalized_points(points, centroid, area):
    return (np.asarray(points, dtype=float) - centroid) / max(math.sqrt(area), 1.0)


def _yaw_measure_loss(source, target, yaw):
    if not len(source) or not len(target):
        return None
    normalization = max(float(np.ptp(np.vstack((source, target)), axis=0).max()), 0.1)
    return _sinkhorn_divergence(
        source @ _screen_rotation(float(yaw)).T,
        target,
        normalization,
    )


def _combined_yaw_loss(groups, yaw):
    active = [
        (source, target, weight)
        for source, target, weight in groups
        if len(source) and len(target) and weight > 0.0
    ]
    total_weight = sum(weight for _, _, weight in active)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(
        weight * _yaw_measure_loss(source, target, yaw)
        for source, target, weight in active
    ) / total_weight)


def _internal_direction_observable(source, target):
    if len(source) < 4 or len(target) < 4:
        return False
    # A centered ring or other centrosymmetric detail cannot decide a 180°
    # flip.  Directional cues such as a strap, wheel, keyboard or trackpad are
    # measurably offset from the instance center in at least one view.
    offset = max(
        float(np.linalg.norm(np.mean(source, axis=0))),
        float(np.linalg.norm(np.mean(target, axis=0))),
    )
    return bool(offset > 0.10)


def _optimize_yaw(groups, initial_yaw=0.0, limit_degrees=60.0, prior_weight=0.02, iterations=18):
    prepared = []
    for source, target, weight in groups:
        if not len(source) or not len(target):
            continue
        normalization = max(float(np.ptp(np.vstack((source, target)), axis=0).max()), 0.1)
        prepared.append((
            source, target, weight, normalization,
            _balanced_sinkhorn_cost(target, target, normalization),
        ))
    active_weight = sum(item[2] for item in prepared)
    if active_weight > 0.0:
        prepared = [
            (source, target, weight / active_weight, normalization, target_self)
            for source, target, weight, normalization, target_self in prepared
        ]

    def data_loss(yaw):
        rotation = _screen_rotation(float(yaw))
        return float(sum(
            weight * _sinkhorn_divergence(
                source @ rotation.T, target, normalization, target_self,
            )
            for source, target, weight, normalization, target_self in prepared
        ))

    def objective(value):
        yaw = float(value[0])
        residual = yaw - initial_yaw
        return data_loss(yaw) + prior_weight * residual * residual

    result = minimize(
        objective, np.array([initial_yaw]), method="L-BFGS-B",
        bounds=[(
            initial_yaw - math.radians(limit_degrees),
            initial_yaw + math.radians(limit_degrees),
        )],
        options={"maxiter": iterations, "ftol": 1e-9, "eps": 1e-3},
    )
    candidate = float(result.x[0])
    initial_loss = data_loss(initial_yaw)
    candidate_loss = data_loss(candidate)
    improvement = initial_loss - candidate_loss
    observable = bool(improvement > max(1e-6, 0.01 * max(initial_loss, 1e-6)))
    accepted = bool(observable and objective([candidate]) + 1e-8 < objective([initial_yaw]))
    return (candidate if accepted else initial_yaw), {
        "accepted": accepted,
        "observable": observable,
        "initial_loss": float(initial_loss),
        "candidate_loss": float(candidate_loss),
        "yaw_delta_deg": float(math.degrees(candidate - initial_yaw)) if accepted else 0.0,
        "optimizer_iterations": int(getattr(result, "nit", 0)),
    }


def _mask_linear_alignment_to_world(
    source_centroid, target_centroid, baseline, camera, linear,
):
    baseline_xy = np.asarray(baseline["translation_world_m"][:2], dtype=float)
    epsilon = 0.001
    origin = np.array([[baseline_xy[0], baseline_xy[1], 0.0]])
    base_pixel = project_points(origin, camera)[0][0]
    jacobian = np.column_stack((
        (project_points(origin + [epsilon, 0.0, 0.0], camera)[0][0] - base_pixel) / epsilon,
        (project_points(origin + [0.0, epsilon, 0.0], camera)[0][0] - base_pixel) / epsilon,
    ))
    target_origin_pixel = np.asarray(target_centroid, dtype=float) - linear @ (
        np.asarray(source_centroid, dtype=float) - base_pixel
    )
    pixel_delta = target_origin_pixel - base_pixel
    return baseline_xy + np.linalg.lstsq(jacobian, pixel_delta, rcond=None)[0]


def _mask_centroid_alignment_to_world(
    source_centroid, target_centroid, baseline, camera, yaw_radians, scale_ratio,
):
    return _mask_linear_alignment_to_world(
        source_centroid, target_centroid, baseline, camera,
        _screen_rotation(yaw_radians, scale_ratio),
    )


def _pixel_centroid_delta_to_world(source_centroid, target_centroid, baseline, camera):
    return _mask_centroid_alignment_to_world(
        source_centroid, target_centroid, baseline, camera, 0.0, 1.0,
    )


def _pixel_delta_to_world(pixel_delta, world_xy, camera):
    world_xy = np.asarray(world_xy, dtype=float)
    epsilon = 0.001
    origin = np.array([[world_xy[0], world_xy[1], 0.0]])
    base_pixel = project_points(origin, camera)[0][0]
    jacobian = np.column_stack((
        (project_points(origin + [epsilon, 0.0, 0.0], camera)[0][0] - base_pixel) / epsilon,
        (project_points(origin + [0.0, epsilon, 0.0], camera)[0][0] - base_pixel) / epsilon,
    ))
    return np.linalg.lstsq(jacobian, np.asarray(pixel_delta, dtype=float), rcond=None)[0]


def optimize_2d_edge_yaw_differentiable(
    groups, *, initial_yaw=0.0, iterations=200, lr_yaw=0.03,
    early_stop_patience=20, early_stop_min_delta=1e-4,
    objective_loss_fn=None, edge_loss_weight=1.0, objective_loss_weight=1.0,
    edge_loss_scale=1.0, objective_loss_scale=1.0,
    yaw_limit_degrees=None,
):
    """Fit only a 2D yaw to centered edge point sets with Adam."""
    import torch

    device = alignment_torch_device()
    active = [
        (np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32), float(weight))
        for source, target, weight in groups
        if len(source) and len(target) and weight > 0.0
    ]
    if not active:
        return 0.0, {
            "accepted": False,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "optimized_parameters": ["yaw"],
        }
    prepared = [
        (
            torch.as_tensor(source, device=device),
            torch.as_tensor(target, device=device),
            weight,
        )
        for source, target, weight in active
    ]
    total_weight = sum(item[2] for item in prepared)
    yaw = torch.nn.Parameter(torch.tensor(float(initial_yaw), device=device))

    def edge_loss_value(angle):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack((
            torch.stack((cosine, sine)),
            torch.stack((-sine, cosine)),
        ))
        loss = torch.zeros(())
        for source, target, weight in prepared:
            transformed = source @ rotation.T
            distances = torch.cdist(transformed, target).square()
            symmetric = distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
            loss = loss + (weight / total_weight) * symmetric
        return loss

    def loss_value(angle):
        edge_loss = edge_loss_value(angle)
        data_loss = edge_loss
        if objective_loss_fn is not None:
            total_weight = max(float(edge_loss_weight) + float(objective_loss_weight), 1e-6)
            data_loss = (
                float(edge_loss_weight) * edge_loss / float(edge_loss_scale)
                + float(objective_loss_weight) * objective_loss_fn(angle)
                / float(objective_loss_scale)
            ) / total_weight
        return data_loss

    baseline_loss = float(loss_value(torch.zeros_like(yaw)).detach())
    baseline_edge_loss = float(edge_loss_value(torch.zeros_like(yaw)).detach())
    initial_loss = float(loss_value(yaw).detach())
    optimizer = torch.optim.Adam([yaw], lr=lr_yaw)
    best_loss = float("inf")
    best_yaw = yaw.detach().clone()
    stale_steps = 0
    completed_steps = 0
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value(yaw)
        current_loss = float(loss.detach())
        previous_best = best_loss
        best_loss, stale_steps = early_stopping_update(
            current_loss, best_loss, stale_steps, early_stop_min_delta,
        )
        if best_loss < previous_best:
            best_yaw = yaw.detach().clone()
        loss.backward()
        optimizer.step()
        if yaw_limit_degrees is not None:
            limit = math.radians(abs(float(yaw_limit_degrees)))
            with torch.no_grad():
                yaw.clamp_(float(initial_yaw) - limit, float(initial_yaw) + limit)
        completed_steps += 1
        if early_stop_patience > 0 and stale_steps >= early_stop_patience:
            break

    candidate_loss = float(loss_value(yaw).detach())
    if candidate_loss < best_loss:
        best_loss = candidate_loss
        best_yaw = yaw.detach().clone()
    with torch.no_grad():
        yaw.copy_(best_yaw)

    final_loss = float(loss_value(yaw).detach())
    final_edge_loss = float(edge_loss_value(yaw).detach())
    accepted = bool(np.isfinite(final_loss) and final_loss + 1e-8 < initial_loss)
    optimized_yaw = float(yaw.detach()) if np.isfinite(final_loss) else float(initial_yaw)
    return optimized_yaw, {
        "accepted": accepted,
        "initial_yaw_deg": float(math.degrees(initial_yaw)),
        "baseline_loss": baseline_loss,
        "baseline_edge_loss": baseline_edge_loss,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "edge_final_loss": final_edge_loss,
        "edge_loss_scale": float(edge_loss_scale),
        "objective_loss_scale": float(objective_loss_scale),
        "yaw_limit_degrees": (
            float(yaw_limit_degrees) if yaw_limit_degrees is not None else None
        ),
        "steps": int(completed_steps),
        "early_stopped": bool(completed_steps < int(iterations)),
        "optimized_parameters": ["yaw"],
    }


def _nearest_cost_loss(costs, source_tail_fraction=0.0):
    source_nearest = np.asarray(costs).min(axis=1)
    if source_tail_fraction > 0.0:
        count = max(1, int(math.ceil(
            float(source_tail_fraction) * len(source_nearest),
        )))
        source_loss = np.partition(
            source_nearest, len(source_nearest) - count,
        )[-count:].mean()
    else:
        source_loss = source_nearest.mean()
    return float(source_loss + np.asarray(costs).min(axis=0).mean())


def _combined_yaw_chamfer(groups, yaw, source_tail_fraction=0.0):
    import torch

    rotation = _screen_rotation(float(yaw))
    active = [
        (np.asarray(source), np.asarray(target), float(weight))
        for source, target, weight in groups
        if len(source) and len(target) and weight > 0.0
    ]
    total_weight = sum(weight for _, _, weight in active)
    if total_weight <= 0.0:
        return 0.0
    device = alignment_torch_device()
    with torch.inference_mode():
        rotation = torch.as_tensor(rotation, dtype=torch.float64, device=device)
        loss = torch.zeros((), dtype=torch.float64, device=device)
        for source, target, weight in active:
            source = torch.as_tensor(source, dtype=torch.float64, device=device)
            target = torch.as_tensor(target, dtype=torch.float64, device=device)
            distances = torch.cdist(source @ rotation.T, target).square()
            source_nearest = distances.min(dim=1).values
            if source_tail_fraction > 0.0:
                count = max(1, int(math.ceil(
                    float(source_tail_fraction) * len(source_nearest),
                )))
                source_loss = source_nearest.topk(count).values.mean()
            else:
                source_loss = source_nearest.mean()
            loss += weight / total_weight * (
                source_loss + distances.min(dim=0).values.mean()
            )
    return float(loss.cpu())


def _oriented_yaw_components(
    source, target, source_tangents, target_tangents,
    source_confidence, target_confidence, yaw, source_tail_fraction=0.0,
):
    """Return separate position and tangent losses from spatial matches."""
    import torch

    device = alignment_torch_device()
    with torch.inference_mode():
        rotation = torch.as_tensor(
            _screen_rotation(float(yaw)), dtype=torch.float64, device=device,
        )
        source = torch.as_tensor(source, dtype=torch.float64, device=device)
        target = torch.as_tensor(target, dtype=torch.float64, device=device)
        source_tangents = torch.as_tensor(
            source_tangents, dtype=torch.float64, device=device,
        )
        target_tangents = torch.as_tensor(
            target_tangents, dtype=torch.float64, device=device,
        )
        source_confidence = torch.as_tensor(
            source_confidence, dtype=torch.float64, device=device,
        )
        target_confidence = torch.as_tensor(
            target_confidence, dtype=torch.float64, device=device,
        )
        distances = torch.cdist(source @ rotation.T, target).square()
        alignment = (source_tangents @ rotation.T @ target_tangents.T).clamp(-1.0, 1.0)
        orientation_costs = (
            source_confidence[:, None] * target_confidence[None, :]
            * (1.0 - alignment.square())
        )
        source_nearest, source_matches = distances.min(dim=1)
        _, target_matches = distances.min(dim=0)
        source_orientation = orientation_costs.gather(1, source_matches[:, None])[:, 0]
        if source_tail_fraction > 0.0:
            count = max(1, int(math.ceil(
                float(source_tail_fraction) * len(source_nearest),
            )))
            source_orientation = source_orientation.gather(
                0, source_nearest.topk(count).indices,
            )
            source_nearest = source_nearest.topk(count).values
        orientation_loss = (
            source_orientation.mean()
            + orientation_costs.gather(0, target_matches[None, :])[0].mean()
        )
        position_loss = source_nearest.mean() + distances.min(dim=0).values.mean()
    return float(position_loss.cpu()), float(orientation_loss.cpu())


def _batched_zero_yaw_components(pairs, source_tail_fraction=0.0):
    """Compute independent Chamfer/tangent pairs in chunked tensor batches."""
    import torch

    pairs = list(pairs)
    if not pairs:
        return []
    device = alignment_torch_device()
    outputs = []
    start = 0
    while start < len(pairs):
        remaining = pairs[start:]
        max_source = max(len(pair[0]) for pair in remaining)
        max_target = max(len(pair[1]) for pair in remaining)
        batch_size = max(1, min(
            8, len(remaining),
            8_000_000 // max(1, max_source * max_target),
        ))
        chunk = remaining[:batch_size]
        start += batch_size
        source_count = torch.as_tensor(
            [len(pair[0]) for pair in chunk], device=device,
        )
        target_count = torch.as_tensor(
            [len(pair[1]) for pair in chunk], device=device,
        )
        padded_source_count = int(source_count.max().item())
        padded_target_count = int(target_count.max().item())
        source = torch.zeros(
            (len(chunk), padded_source_count, 2),
            dtype=torch.float64, device=device,
        )
        target = torch.zeros(
            (len(chunk), padded_target_count, 2),
            dtype=torch.float64, device=device,
        )
        oriented = [len(pair) == 6 for pair in chunk]
        source_tangents = torch.zeros_like(source)
        target_tangents = torch.zeros_like(target)
        source_confidence = torch.zeros(
            (len(chunk), padded_source_count),
            dtype=torch.float64, device=device,
        )
        target_confidence = torch.zeros(
            (len(chunk), padded_target_count),
            dtype=torch.float64, device=device,
        )
        for index, pair in enumerate(chunk):
            source[index, :len(pair[0])] = torch.as_tensor(
                pair[0], dtype=torch.float64, device=device,
            )
            target[index, :len(pair[1])] = torch.as_tensor(
                pair[1], dtype=torch.float64, device=device,
            )
            if oriented[index]:
                source_tangents[index, :len(pair[2])] = torch.as_tensor(
                    pair[2], dtype=torch.float64, device=device,
                )
                target_tangents[index, :len(pair[3])] = torch.as_tensor(
                    pair[3], dtype=torch.float64, device=device,
                )
                source_confidence[index, :len(pair[4])] = torch.as_tensor(
                    pair[4], dtype=torch.float64, device=device,
                )
                target_confidence[index, :len(pair[5])] = torch.as_tensor(
                    pair[5], dtype=torch.float64, device=device,
                )
        source_valid = (
            torch.arange(padded_source_count, device=device)[None]
            < source_count[:, None]
        )
        target_valid = (
            torch.arange(padded_target_count, device=device)[None]
            < target_count[:, None]
        )
        with torch.inference_mode():
            # Keep only running nearest neighbours.  Materializing the full
            # BxSxT distance and tangent tensors can exceed a 24 GB card for
            # detailed internal edges, while this gives the identical minima.
            source_nearest = torch.full(
                (len(chunk), padded_source_count), torch.inf,
                dtype=torch.float64, device=device,
            )
            target_nearest = torch.full(
                (len(chunk), padded_target_count), torch.inf,
                dtype=torch.float64, device=device,
            )
            source_matches = torch.zeros(
                (len(chunk), padded_source_count),
                dtype=torch.int64, device=device,
            )
            target_matches = torch.zeros(
                (len(chunk), padded_target_count),
                dtype=torch.int64, device=device,
            )
            distance_block = 512
            for source_start in range(0, padded_source_count, distance_block):
                source_stop = min(
                    source_start + distance_block, padded_source_count,
                )
                source_block = source[:, source_start:source_stop]
                source_block_valid = source_valid[:, source_start:source_stop]
                for target_start in range(
                    0, padded_target_count, distance_block,
                ):
                    target_stop = min(
                        target_start + distance_block, padded_target_count,
                    )
                    distances = torch.cdist(
                        source_block, target[:, target_start:target_stop],
                    ).square()
                    source_values, source_indices = distances.masked_fill(
                        ~target_valid[:, None, target_start:target_stop],
                        torch.inf,
                    ).min(dim=2)
                    current_source = source_nearest[
                        :, source_start:source_stop
                    ]
                    improve_source = source_values < current_source
                    source_nearest[:, source_start:source_stop] = torch.where(
                        improve_source, source_values, current_source,
                    )
                    current_source_matches = source_matches[
                        :, source_start:source_stop
                    ]
                    source_matches[:, source_start:source_stop] = torch.where(
                        improve_source, source_indices + target_start,
                        current_source_matches,
                    )

                    target_values, target_indices = distances.masked_fill(
                        ~source_block_valid[:, :, None], torch.inf,
                    ).min(dim=1)
                    current_target = target_nearest[:, target_start:target_stop]
                    improve_target = target_values < current_target
                    target_nearest[:, target_start:target_stop] = torch.where(
                        improve_target, target_values, current_target,
                    )
                    current_target_matches = target_matches[
                        :, target_start:target_stop
                    ]
                    target_matches[:, target_start:target_stop] = torch.where(
                        improve_target, target_indices + source_start,
                        current_target_matches,
                    )

            matched_target_tangents = target_tangents.gather(
                1, source_matches[..., None].expand(-1, -1, 2),
            )
            matched_target_confidence = target_confidence.gather(
                1, source_matches,
            )
            source_alignment = (
                source_tangents * matched_target_tangents
            ).sum(dim=2).clamp(-1.0, 1.0)
            source_orientation = (
                source_confidence * matched_target_confidence
                * (1.0 - source_alignment.square())
            )
            matched_source_tangents = source_tangents.gather(
                1, target_matches[..., None].expand(-1, -1, 2),
            )
            matched_source_confidence = source_confidence.gather(
                1, target_matches,
            )
            target_alignment = (
                target_tangents * matched_source_tangents
            ).sum(dim=2).clamp(-1.0, 1.0)
            target_orientation = (
                target_confidence * matched_source_confidence
                * (1.0 - target_alignment.square())
            )
        for index, pair in enumerate(chunk):
            source_values = source_nearest[index, :len(pair[0])]
            if source_tail_fraction > 0.0:
                count = max(1, int(math.ceil(
                    float(source_tail_fraction) * len(source_values),
                )))
                tail = source_values.topk(count).indices
                source_loss = source_values[tail].mean()
                source_orientation_loss = source_orientation[
                    index, :len(pair[0])
                ][tail].mean()
            else:
                source_loss = source_values.mean()
                source_orientation_loss = source_orientation[
                    index, :len(pair[0])
                ].mean()
            position_loss = source_loss + target_nearest[
                index, :len(pair[1])
            ].mean()
            orientation_loss = (
                source_orientation_loss
                + target_orientation[index, :len(pair[1])].mean()
                if oriented[index] else torch.zeros((), device=device)
            )
            outputs.append((
                float(position_loss.cpu()), float(orientation_loss.cpu()),
            ))
    return outputs


def _oriented_yaw_chamfer(
    source, target, source_tangents, target_tangents,
    source_confidence, target_confidence, yaw, orientation_weight,
    source_tail_fraction=0.0,
):
    position, orientation = _oriented_yaw_components(
        source, target, source_tangents, target_tangents,
        source_confidence, target_confidence, yaw, source_tail_fraction,
    )
    return position + float(orientation_weight) * orientation


def _internal_centroid_offset_loss(source, target, yaw):
    """Compare signed internal-edge offsets from their object-mask centers."""
    import torch

    device = alignment_torch_device()
    with torch.inference_mode():
        source_offset = torch.as_tensor(
            source, dtype=torch.float64, device=device,
        ).mean(dim=0)
        target_offset = torch.as_tensor(
            target, dtype=torch.float64, device=device,
        ).mean(dim=0)
        rotation = torch.as_tensor(
            _screen_rotation(float(yaw)), dtype=torch.float64, device=device,
        )
        loss = (source_offset @ rotation.T - target_offset).square().sum()
    return float(loss.cpu())


def _candidate_max_normalize(values):
    """Normalize candidate losses by the largest loss in that component."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1 or not len(values)
        or not np.isfinite(values).all() or np.any(values < 0.0)
    ):
        raise ValueError(
            "candidate losses must be a non-empty finite non-negative vector",
        )
    maximum = float(values.max())
    return values / maximum if maximum > 1e-12 else np.zeros_like(values)


def _candidate_varying_max_normalize(values):
    """Max-normalize only losses that can distinguish the candidates."""
    values = np.asarray(values, dtype=float)
    normalized = _candidate_max_normalize(values)
    return (
        np.zeros_like(normalized)
        if float(np.ptp(values)) <= 1e-12 else normalized
    )


def _coarse_yaw_candidate_batch_size(candidate_count, pixel_count, edge_groups):
    max_batch_pixels = 2_000_000
    max_batch_edge_pairs = 32_000_000
    largest_edge_pair_count = max(
        (len(source) * len(target) for source, target in edge_groups),
        default=1,
    )
    return max(1, min(
        64,
        int(candidate_count),
        max_batch_pixels // max(1, int(pixel_count)),
        max_batch_edge_pairs // max(1, largest_edge_pair_count),
    ))


def optimize_2d_edge_yaw_coarse_multistart(
    groups, source_region, target_region, *, iterations=200, lr_yaw=0.03,
    early_stop_patience=20, early_stop_min_delta=1e-4,
    source_weights=None, target_weights=None,
    rgb_loss_fn=None, edge_loss_weight=1.0, rgb_loss_weight=0.0,
    mask_iou_fn=None, mask_iou_loss_weight=1.0,
    objective_loss_fn=None,
    candidate_selection="rank",
    direction_iou_ambiguity_threshold=0.02,
    coarse_step_degrees=2.0,
    coarse_yaw_candidates=None,
    internal_orientation_features=None,
    internal_orientation_weight=1.0,
    internal_edge_match_mode="chamfer",
    internal_edge_count_loss_fn=None,
    coarse_metrics_fn=None,
    rank_shape_component="outer",
    rank_rgb_component_weight=1.0,
    model_mask_information=None,
    coarse_loss_multipliers=None,
    direction_loss_fn=None,
    direction_candidate_window_centers_deg=None,
    use_zero_yaw_baseline=True,
    gpu_semaphore=None,
):
    del (
        source_region, target_region, source_weights, target_weights,
        iterations, lr_yaw, early_stop_patience, early_stop_min_delta,
    )
    coarse_step_degrees = float(coarse_step_degrees)
    if not 0.0 < coarse_step_degrees <= 45.0:
        raise ValueError("coarse yaw step must be in (0, 45] degrees")
    pca_local_search = coarse_yaw_candidates is not None
    coarse_yaws = (
        tuple(float(value) for value in coarse_yaw_candidates)
        if pca_local_search else
        tuple(np.deg2rad(np.arange(0.0, 360.0, coarse_step_degrees)))
    )
    if not coarse_yaws or not np.isfinite(coarse_yaws).all():
        raise ValueError("coarse yaw candidates must be non-empty and finite")
    has_outer_edges = bool(groups and len(groups[0][0]) and len(groups[0][1]))
    has_internal_edges = bool(
        len(groups) > 1 and len(groups[1][0]) and len(groups[1][1])
    )
    orientation_weight = _nonnegative_finite_float(
        "internal edge orientation weight", internal_orientation_weight,
    )
    rank_rgb_component_weight = _nonnegative_finite_float(
        "rank RGB component weight", rank_rgb_component_weight,
    )
    if internal_edge_match_mode not in (
        "chamfer", "partial_hausdorff", "centroid_offset",
    ):
        raise ValueError(
            "internal edge match mode must be 'chamfer', "
            "'partial_hausdorff', or 'centroid_offset'"
        )
    if rank_shape_component not in (
        "internal", "outer", "mask_iou", "outer_mask_iou", "paired_dynamic",
    ):
        raise ValueError(
            "rank shape component must be 'internal', 'outer', "
            "'mask_iou', 'outer_mask_iou', or 'paired_dynamic'"
        )
    has_internal_orientation = bool(
        has_internal_edges and internal_orientation_features is not None
        and orientation_weight > 0.0
        and np.count_nonzero(internal_orientation_features[2])
        and np.count_nonzero(internal_orientation_features[3])
    )
    has_internal_edge_count_loss = internal_edge_count_loss_fn is not None

    def coarse_metrics(yaw):
        outer = _nonnegative_finite_float(
            "coarse outer-edge loss",
            _combined_yaw_chamfer([groups[0]], yaw) if has_outer_edges else 0.0,
        )
        if not has_internal_edges:
            internal_position = internal_direction = 0.0
        elif internal_edge_match_mode == "centroid_offset":
            internal_position = _internal_centroid_offset_loss(
                groups[1][0], groups[1][1], yaw,
            )
            internal_direction = 0.0
        elif has_internal_orientation:
            internal_position, internal_direction = _oriented_yaw_components(
                groups[1][0], groups[1][1],
                *internal_orientation_features, yaw,
                source_tail_fraction=(
                    0.10
                    if internal_edge_match_mode == "partial_hausdorff"
                    else 0.0
                ),
            )
        else:
            internal_position = (
                _combined_yaw_chamfer(
                    [groups[1]], yaw, source_tail_fraction=0.10,
                )
                if internal_edge_match_mode == "partial_hausdorff"
                else _combined_yaw_chamfer([groups[1]], yaw)
            )
            internal_direction = 0.0
        internal_position = _nonnegative_finite_float(
            "coarse internal-position loss", internal_position,
        )
        internal_direction = _nonnegative_finite_float(
            "coarse internal-direction loss", internal_direction,
        )
        rgb = (
            _unit_interval_float("coarse RGB loss", rgb_loss_fn(yaw))
            if rgb_loss_fn is not None else 0.0
        )
        iou = (
            _unit_interval_float("coarse mask IoU", mask_iou_fn(yaw))
            if mask_iou_fn is not None else 1.0
        )
        count_loss = (
            _nonnegative_finite_float(
                "coarse internal-edge count loss",
                internal_edge_count_loss_fn(yaw),
            )
            if internal_edge_count_loss_fn is not None else 0.0
        )
        return outer, internal_position, internal_direction, rgb, iou, count_loss

    batch_metrics_fn = getattr(objective_loss_fn, "batch_metrics", None)
    if coarse_metrics_fn is not None:
        coarse_backend = "nvdiffrast_3d_candidates"
        coarse_metrics_by_yaw = []
        for yaw in coarse_yaws:
            metrics = tuple(coarse_metrics_fn(yaw))
            if len(metrics) == 4:
                outer, internal_position, rgb, iou = metrics
                metrics = (outer, internal_position, 0.0, rgb, iou)
            if len(metrics) == 5:
                metrics = (*metrics, 0.0)
            elif len(metrics) == 6:
                has_internal_edge_count_loss = True
            else:
                raise ValueError(
                    "coarse candidate metrics must contain 4, 5, or 6 values"
                )
            coarse_metrics_by_yaw.append(metrics)
    elif batch_metrics_fn is not None:
        import torch

        device = alignment_torch_device()
        coarse_backend = f"torch_{device.type}_batch"
        yaw_tensor = torch.as_tensor(coarse_yaws, dtype=torch.float32, device=device)
        cosine, sine = torch.cos(yaw_tensor), torch.sin(yaw_tensor)
        rotations = torch.stack((
            torch.stack((cosine, sine), dim=1),
            torch.stack((-sine, cosine), dim=1),
        ), dim=1)
        gpu_edge_groups = [
            (
                torch.as_tensor(source, dtype=torch.float32, device=device),
                torch.as_tensor(target, dtype=torch.float32, device=device),
            )
            for source, target, _ in groups
        ]
        gpu_orientation = (
            tuple(
                torch.as_tensor(value, dtype=torch.float32, device=device)
                for value in internal_orientation_features
            )
            if has_internal_orientation else None
        )

        def batch_edge_loss(group_index, batch_rotations):
            if group_index >= len(gpu_edge_groups):
                zeros = torch.zeros(len(batch_rotations), device=device)
                return zeros, zeros
            source, target = gpu_edge_groups[group_index]
            if not len(source) or not len(target):
                zeros = torch.zeros(len(batch_rotations), device=device)
                return zeros, zeros
            transformed = torch.einsum("sij,pj->spi", batch_rotations, source)
            if (
                group_index == 1
                and internal_edge_match_mode == "centroid_offset"
            ):
                position = (
                    transformed.mean(dim=1) - target.mean(dim=0)[None]
                ).square().sum(dim=1)
                return position, torch.zeros_like(position)
            distances = torch.cdist(
                transformed, target[None].expand(len(batch_rotations), -1, -1),
            ).square()
            orientation_costs = None
            if group_index == 1 and gpu_orientation is not None:
                source_tangents, target_tangents, source_confidence, target_confidence = (
                    gpu_orientation
                )
                rotated_tangents = torch.einsum(
                    "sij,pj->spi", batch_rotations, source_tangents,
                )
                alignment = torch.einsum(
                    "spi,ti->spt", rotated_tangents, target_tangents,
                ).clamp(-1.0, 1.0)
                confidence = (
                    source_confidence[None, :, None]
                    * target_confidence[None, None, :]
                )
                orientation_costs = confidence * (
                    1.0 - alignment.square()
                )
            source_nearest, source_matches = distances.min(dim=2)
            target_nearest, target_matches = distances.min(dim=1)
            source_loss = (
                source_nearest.topk(
                    max(1, int(math.ceil(0.10 * source.shape[0]))), dim=1,
                ).values.mean(dim=1)
                if (
                    group_index == 1
                    and internal_edge_match_mode == "partial_hausdorff"
                )
                else source_nearest.mean(dim=1)
            )
            position_loss = source_loss + target_nearest.mean(dim=1)
            if orientation_costs is None:
                return position_loss, torch.zeros_like(position_loss)
            source_orientation = orientation_costs.gather(
                2, source_matches[:, :, None],
            )[:, :, 0]
            if (
                group_index == 1
                and internal_edge_match_mode == "partial_hausdorff"
            ):
                count = max(1, int(math.ceil(0.10 * source.shape[0])))
                tail = source_nearest.topk(count, dim=1).indices
                source_orientation = source_orientation.gather(1, tail)
            target_orientation = orientation_costs.gather(
                1, target_matches[:, None, :],
            )[:, 0, :]
            orientation_loss = (
                source_orientation.mean(dim=1)
                + target_orientation.mean(dim=1)
            )
            return position_loss, orientation_loss

        pixel_count = max(1, int(getattr(objective_loss_fn, "batch_pixel_count", 1)))
        candidate_batch_size = _coarse_yaw_candidate_batch_size(
            len(coarse_yaws), pixel_count, gpu_edge_groups,
        )
        outer_parts, internal_position_parts = [], []
        internal_direction_parts, iou_parts, rgb_parts = [], [], []
        if gpu_semaphore is not None:
            gpu_semaphore.acquire()
        try:
            with torch.no_grad():
                for start in range(0, len(coarse_yaws), candidate_batch_size):
                    end = start + candidate_batch_size
                    batch_rotations = rotations[start:end]
                    outer_position, _ = batch_edge_loss(0, batch_rotations)
                    internal_position, internal_direction = batch_edge_loss(
                        1, batch_rotations,
                    )
                    outer_parts.append(outer_position)
                    internal_position_parts.append(internal_position)
                    internal_direction_parts.append(internal_direction)
                    batch_iou, batch_rgb, _ = batch_metrics_fn(
                        yaw_tensor[start:end],
                    )
                    iou_parts.append(batch_iou)
                    rgb_parts.append(batch_rgb)
                outer_tensor = torch.cat(outer_parts)
                internal_position_tensor = torch.cat(internal_position_parts)
                internal_direction_tensor = torch.cat(internal_direction_parts)
                iou_tensor = torch.cat(iou_parts)
                rgb_tensor = torch.cat(rgb_parts)
        finally:
            if gpu_semaphore is not None:
                gpu_semaphore.release()
        coarse_metrics_by_yaw = list(zip(
            outer_tensor.cpu().tolist(),
            internal_position_tensor.cpu().tolist(),
            internal_direction_tensor.cpu().tolist(),
            rgb_tensor.cpu().tolist(),
            iou_tensor.cpu().tolist(),
            [
                _nonnegative_finite_float(
                    "coarse internal-edge count loss",
                    internal_edge_count_loss_fn(yaw),
                )
                if internal_edge_count_loss_fn is not None else 0.0
                for yaw in coarse_yaws
            ],
        ))
    else:
        coarse_backend = "cpu_threaded_fallback"
        with ThreadPoolExecutor(
            max_workers=min(16, len(coarse_yaws), max(1, os.cpu_count() or 1)),
        ) as executor:
            coarse_metrics_by_yaw = list(executor.map(coarse_metrics, coarse_yaws))
    outer_losses = np.asarray([value[0] for value in coarse_metrics_by_yaw])
    internal_position_losses = np.asarray([
        value[1] for value in coarse_metrics_by_yaw
    ])
    internal_direction_losses = np.asarray([
        value[2] for value in coarse_metrics_by_yaw
    ])
    internal_losses = (
        internal_position_losses
        + orientation_weight * internal_direction_losses
    )
    rgb_losses = np.asarray([value[3] for value in coarse_metrics_by_yaw])
    mask_ious = np.asarray([value[4] for value in coarse_metrics_by_yaw])
    internal_edge_count_losses = np.asarray([
        value[5] for value in coarse_metrics_by_yaw
    ])
    outer_scale = max(float(np.median(outer_losses)), 1e-3)
    internal_scale = max(float(np.median(internal_losses)), 1e-3)
    outer_max_normalized = _candidate_max_normalize(outer_losses)
    mask_iou_max_normalized = _candidate_max_normalize(1.0 - mask_ious)
    internal_edge_count_max_normalized = _candidate_varying_max_normalize(
        internal_edge_count_losses,
    )
    rank_components = {}
    if rank_shape_component in (
        "mask_iou", "outer_mask_iou", "paired_dynamic",
    ):
        rank_components["mask_iou"] = mask_iou_max_normalized
    if rank_shape_component in (
        "outer", "outer_mask_iou", "paired_dynamic",
    ) and has_outer_edges:
        rank_components["outer"] = outer_max_normalized
    if has_internal_edges:
        rank_components["internal_position"] = _candidate_max_normalize(
            internal_position_losses,
        )
    if has_internal_orientation:
        rank_components["internal_direction"] = _candidate_max_normalize(
            internal_direction_losses,
        )
    if (
        has_internal_edge_count_loss
        and np.any(internal_edge_count_max_normalized > 0.0)
    ):
        rank_components["internal_count"] = internal_edge_count_max_normalized
    if rgb_loss_fn is not None or batch_metrics_fn is not None:
        rank_components["rgb"] = _candidate_max_normalize(rgb_losses)
    if not rank_components:
        raise ValueError("yaw search requires at least one active loss component")
    component_names = [
        "1-mask_iou"
        if name == "mask_iou" else
        (
            "internal_position_partial_hausdorff"
            if internal_edge_match_mode == "partial_hausdorff"
            else "internal_position_centroid_offset"
            if internal_edge_match_mode == "centroid_offset"
            else name
        )
        if name == "internal_position" else name
        for name in rank_components
    ]
    if candidate_selection in ("rank", "weighted", "top2"):
        if rank_shape_component == "paired_dynamic":
            information = (
                0.5 if model_mask_information is None else _unit_interval_float(
                    "model mask information", model_mask_information,
                )
            )
            model_mask_weight = information
            detail_weight = 1.0 - model_mask_weight
            shape_names = [
                name for name in ("mask_iou", "outer")
                if name in rank_components
            ]
            internal_names = [
                name for name in (
                    "internal_position", "internal_direction", "internal_count",
                )
                if name in rank_components
            ]
            detail_groups = []
            if internal_names:
                detail_groups.append(internal_names)
            if "rgb" in rank_components and rank_rgb_component_weight > 0.0:
                detail_groups.append(["rgb"])
            raw_component_weights = {}
            if shape_names and detail_groups:
                raw_component_weights.update({
                    name: model_mask_weight / len(shape_names)
                    for name in shape_names
                })
                for names in detail_groups:
                    group_weight = detail_weight / len(detail_groups)
                    relative_weights = {
                        name: (
                            orientation_weight
                            if name == "internal_direction" else
                            rank_rgb_component_weight
                            if name == "rgb" else 1.0
                        )
                        for name in names
                    }
                    total = sum(relative_weights.values())
                    raw_component_weights.update({
                        name: group_weight * weight / total
                        for name, weight in relative_weights.items()
                    })
            else:
                raw_component_weights = {
                    name: 1.0 / len(rank_components)
                    for name in rank_components
                }
        elif model_mask_information is None:
            raw_component_weights = {
                name: rank_rgb_component_weight if name == "rgb" else 1.0
                for name in rank_components
            }
            model_mask_weight = detail_weight = None
        else:
            information = _unit_interval_float(
                "model mask information", model_mask_information,
            )
            model_mask_weight = 0.1 + 0.8 * information
            detail_weight = 1.0 - model_mask_weight
            mask_names = [
                name for name in ("mask_iou",) if name in rank_components
            ]
            detail_names = [
                name for name in (
                    "outer", "internal_position", "internal_direction",
                    "internal_count", "rgb",
                )
                if name in rank_components
            ]
            raw_component_weights = {}
            if mask_names and detail_names:
                raw_component_weights.update({
                    name: model_mask_weight / len(mask_names)
                    for name in mask_names
                })
                detail_component_weights = {
                    name: (
                        orientation_weight
                        if name == "internal_direction" else
                        rank_rgb_component_weight
                        if name == "rgb" else
                        1.0
                    )
                    for name in detail_names
                }
                detail_component_total = sum(
                    detail_component_weights.values()
                )
                if detail_component_total > 0.0:
                    raw_component_weights.update({
                        name: detail_weight * weight / detail_component_total
                        for name, weight in detail_component_weights.items()
                    })
            else:
                active_names = mask_names or detail_names
                raw_component_weights = {
                    name: 1.0 / len(active_names) for name in active_names
                }
        total_component_weight = sum(raw_component_weights.values())
        if total_component_weight <= 0.0:
            raise ValueError("yaw rank selection requires a positive-weight component")
        component_weights = {
            name: weight / total_component_weight
            for name, weight in raw_component_weights.items()
        }
        outer_weight = component_weights.get("outer", 0.0)
        mask_iou_weight = component_weights.get("mask_iou", 0.0)
        internal_position_weight = component_weights.get(
            "internal_position", 0.0,
        )
        internal_direction_weight = component_weights.get(
            "internal_direction", 0.0,
        )
        internal_weight = internal_position_weight + internal_direction_weight
        internal_edge_count_weight = component_weights.get(
            "internal_count", 0.0,
        )
        rgb_weight = component_weights.get("rgb", 0.0)
        rgb_iou_direction_weight = 0.0
        coarse_losses = sum(
            component_weights[name] * values
            for name, values in rank_components.items()
        )
        edge_losses = (
            outer_weight * rank_components.get(
                "outer", np.zeros(len(coarse_yaws)),
            )
            + internal_position_weight * rank_components.get(
                "internal_position", np.zeros(len(coarse_yaws)),
            )
            + internal_direction_weight * rank_components.get(
                "internal_direction", np.zeros(len(coarse_yaws)),
            )
            + internal_edge_count_weight * rank_components.get(
                "internal_count", np.zeros(len(coarse_yaws)),
            )
        )
        internal_normalized_losses = (
            internal_position_weight * rank_components.get(
                "internal_position", np.zeros(len(coarse_yaws)),
            )
            + internal_direction_weight * rank_components.get(
                "internal_direction", np.zeros(len(coarse_yaws)),
            )
        ) / max(internal_weight, 1e-12)
        loss_normalization = "candidate_weighted_max_ratio_[0,1]"
        active_component_names = [
            name for name, weight in zip(component_names, component_weights.values())
            if weight > 0.0
        ]
        active_weights = [
            weight for weight in component_weights.values() if weight > 0.0
        ]
        loss_formula = "weighted_mean(" + ", ".join(
            f"max_norm({name})" for name in active_component_names
        ) + ")"
        if rank_shape_component == "paired_dynamic":
            shape_terms = [
                f"max_norm({name})"
                for name in component_names
                if name in ("1-mask_iou", "outer")
            ]
            internal_terms = [
                f"max_norm({name})"
                for name in component_names
                if name.startswith("internal_")
            ]
            detail_terms = [
                (
                    "weighted_mean(" + ", ".join(internal_terms) + ")"
                    if len(internal_terms) > 1 else internal_terms[0]
                )
                if internal_terms else None,
                "max_norm(rgb)" if "rgb" in component_names else None,
            ]
            loss_formula = (
                "dynamic_shape_weight*mean(" + ", ".join(shape_terms) + ") + "
                "dynamic_detail_weight*mean(" + ", ".join(
                    term for term in detail_terms if term is not None
                ) + ")"
            )
            loss_normalization = "candidate_dynamic_grouped_max_ratio_[0,1]"
        elif all(
            np.isclose(value, 1.0 / len(active_weights))
            for value in active_weights
        ):
            loss_formula = "mean(" + ", ".join(
                f"max_norm({name})" for name in active_component_names
            ) + ")"
            loss_normalization = "candidate_average_max_ratio_[0,1]"
    elif candidate_selection == "product":
        outer_weight = 0.5
        mask_iou_weight = 0.0
        internal_position_weight = 0.5
        internal_direction_weight = 0.0
        internal_weight = 0.5
        internal_edge_count_weight = 0.0
        rgb_weight = 0.0
        rgb_iou_direction_weight = 1.0
        edge_losses = (
            outer_weight * outer_losses / outer_scale
            if has_outer_edges else np.zeros(len(coarse_yaws))
        ) + (
            internal_weight * internal_losses / internal_scale
            if has_internal_edges else np.zeros(len(coarse_yaws))
        )
        internal_normalized_losses = (
            internal_losses / internal_scale
            if has_internal_edges else np.zeros(len(coarse_yaws))
        )
        coarse_losses = edge_losses + 1.0 - (1.0 - rgb_losses) * mask_ious
        loss_normalization = "outer_internal_median_scale"
        loss_formula = (
            "0.5*outer_norm + 0.5*"
            + (
                "internal_partial_hausdorff_norm"
                if internal_edge_match_mode == "partial_hausdorff"
                else "internal_oriented_norm"
                if has_internal_orientation
                and internal_edge_match_mode == "chamfer"
                else (
                    "internal_centroid_offset_norm"
                    if internal_edge_match_mode == "centroid_offset"
                    else "internal_norm"
                )
            )
            + " + 1-rgb_similarity*iou"
        )
    else:
        raise ValueError(
            f"unknown top-yaw candidate selection: {candidate_selection}"
        )
    direction_losses = None
    direction_candidate_report = None
    legacy_coarse_candidate_count = len(coarse_yaws)
    if direction_loss_fn is not None:
        legacy_yaws = tuple(coarse_yaws)
        legacy_losses = np.asarray(coarse_losses, dtype=float)
        candidate_sources = {}

        def add_candidate(index, source):
            candidate_sources.setdefault(int(index), []).append(source)

        if direction_candidate_window_centers_deg is not None:
            centers = np.asarray(
                direction_candidate_window_centers_deg, dtype=float,
            ) % 360.0
            if not len(centers) or not np.isfinite(centers).all():
                raise ValueError("direction candidate window centers must be finite")
            yaw_degrees = np.degrees(legacy_yaws) % 360.0
            distances = np.abs(
                (yaw_degrees[:, None] - centers[None, :] + 180.0) % 360.0
                - 180.0
            )
            assignments = distances.argmin(axis=1)
            for center_index, center in enumerate(centers):
                window = np.flatnonzero(assignments == center_index)
                if len(window):
                    best = window[np.argmin(legacy_losses[window])]
                    add_candidate(best, f"pca_window_total@{center:g}deg")
            generation_policy = "pca_window_total_minima"
        elif pca_local_search:
            for index in range(len(legacy_yaws)):
                add_candidate(index, "explicit_direction_candidate")
            generation_policy = "explicit_direction_candidates"
        else:
            yaw_degrees = np.degrees(legacy_yaws) % 360.0
            centers = np.arange(0.0, 360.0, 45.0)
            distances = np.abs(
                (yaw_degrees[:, None] - centers[None, :] + 180.0) % 360.0
                - 180.0
            )
            assignments = distances.argmin(axis=1)
            for center_index, center in enumerate(centers):
                window = np.flatnonzero(assignments == center_index)
                if len(window):
                    best = window[np.argmin(legacy_losses[window])]
                    add_candidate(best, f"total_loss_window@{center:g}deg")
            generation_policy = "full_search_total_loss_window_minima"

        candidate_indices = np.asarray(sorted(candidate_sources), dtype=int)
        if hasattr(direction_loss_fn, "prepare"):
            direction_loss_fn.prepare(tuple(
                legacy_yaws[index] for index in candidate_indices
            ))
        direction_losses = np.asarray([
            _unit_interval_float(
                "coarse direction loss", direction_loss_fn(legacy_yaws[index]),
            )
            for index in candidate_indices
        ])
        direction_candidate_report = {
            "generation_policy": generation_policy,
            "legacy_coarse_candidate_count": len(legacy_yaws),
            "candidate_count": len(candidate_indices),
            "candidate_window_degrees": (
                45.0
                if generation_policy == "full_search_total_loss_window_minima"
                else None
            ),
            "maximum_candidate_count": (
                8
                if generation_policy == "full_search_total_loss_window_minima"
                else None
            ),
            "candidates": [
                {
                    "yaw_deg": float(math.degrees(legacy_yaws[index])),
                    "legacy_total_loss": float(legacy_losses[index]),
                    "sources": candidate_sources[index],
                    "direction_loss": float(loss),
                }
                for index, loss in zip(candidate_indices, direction_losses)
            ],
        }
        coarse_yaws = tuple(legacy_yaws[index] for index in candidate_indices)
        coarse_metrics_by_yaw = [
            coarse_metrics_by_yaw[index] for index in candidate_indices
        ]
        outer_losses = outer_losses[candidate_indices]
        internal_position_losses = internal_position_losses[candidate_indices]
        internal_direction_losses = internal_direction_losses[candidate_indices]
        internal_losses = internal_losses[candidate_indices]
        rgb_losses = rgb_losses[candidate_indices]
        mask_ious = mask_ious[candidate_indices]
        internal_edge_count_losses = internal_edge_count_losses[candidate_indices]
        outer_max_normalized = outer_max_normalized[candidate_indices]
        mask_iou_max_normalized = mask_iou_max_normalized[candidate_indices]
        internal_edge_count_max_normalized = (
            internal_edge_count_max_normalized[candidate_indices]
        )
        rank_components = {
            name: values[candidate_indices]
            for name, values in rank_components.items()
        }
        edge_losses = edge_losses[candidate_indices]
        internal_normalized_losses = internal_normalized_losses[candidate_indices]
        coarse_losses = direction_losses
        loss_formula = getattr(
            direction_loss_fn, "loss_formula",
            "DINO_relative_position_consistency",
        )
        loss_normalization = "native_[0,1]"
    unweighted_coarse_losses = np.asarray(coarse_losses, dtype=float)
    if coarse_loss_multipliers is None:
        loss_multipliers = np.ones(len(coarse_yaws), dtype=float)
    else:
        loss_multipliers = np.asarray(coarse_loss_multipliers, dtype=float)
        if loss_multipliers.shape != unweighted_coarse_losses.shape:
            raise ValueError("coarse loss multipliers must match yaw candidates")
        if not np.isfinite(loss_multipliers).all() or np.any(loss_multipliers < 0.0):
            raise ValueError("coarse loss multipliers must be finite and non-negative")
        loss_formula = "(" + loss_formula + ")*stage2_direction_loss"
    coarse_losses = unweighted_coarse_losses * loss_multipliers
    yaw_degrees = np.degrees(coarse_yaws)
    def nearest_yaw_index(angle_degrees):
        return int(np.argmin(np.abs(
            (yaw_degrees - float(angle_degrees) + 180.0) % 360.0 - 180.0
        )))

    diagnostic_best_index = int(np.argmin(coarse_losses))
    angular_distance = np.abs(
        (yaw_degrees - yaw_degrees[diagnostic_best_index] + 180.0) % 360.0 - 180.0
    )
    independent_indices = np.flatnonzero(angular_distance >= 10.0)
    second_basin_index = (
        int(independent_indices[
            np.argmin(np.asarray(coarse_losses)[independent_indices])
        ])
        if len(independent_indices) else diagnostic_best_index
    )
    loss_p10, loss_median, loss_p90 = np.percentile(
        coarse_losses, (10, 50, 90),
    )
    diagnostic_best_loss = float(coarse_losses[diagnostic_best_index])
    second_basin_loss = float(coarse_losses[second_basin_index])
    coarse_selected = diagnostic_best_index
    yaw = coarse_yaws[coarse_selected]
    selected_metrics = coarse_metrics_by_yaw[coarse_selected]
    selected_loss = coarse_losses[coarse_selected]
    baseline_index = (
        nearest_yaw_index(0.0) if use_zero_yaw_baseline else None
    )
    baseline_loss = (
        coarse_losses[baseline_index] if baseline_index is not None else None
    )
    candidate_accepted_by_loss = bool(
        not use_zero_yaw_baseline
        or selected_loss + 1e-8 < baseline_loss
    )
    diagnostic_opposite_index = nearest_yaw_index(
        yaw_degrees[diagnostic_best_index] + 180.0,
    )
    opposite_metrics = coarse_metrics_by_yaw[diagnostic_opposite_index]
    direction_iou_gap = abs(
        coarse_metrics_by_yaw[diagnostic_best_index][4] - opposite_metrics[4]
    )
    direction_ambiguity_evaluated = mask_iou_fn is not None
    direction_ambiguous = bool(
        direction_ambiguity_evaluated
        and direction_iou_gap <= _unit_interval_float(
            "direction IoU ambiguity threshold", direction_iou_ambiguity_threshold,
        )
    )
    accepted = candidate_accepted_by_loss
    if not accepted:
        yaw = 0.0
        selected_metrics = coarse_metrics_by_yaw[baseline_index]
        selected_loss = baseline_loss
        coarse_selected = baseline_index
    applied_opposite_index = nearest_yaw_index(
        yaw_degrees[coarse_selected] + 180.0,
    )
    yaw_fallback_reason = (
        "candidate_not_better" if not candidate_accepted_by_loss else None
    )
    return yaw, {
        "accepted": accepted,
        "candidate_accepted_by_loss": candidate_accepted_by_loss,
        "yaw_update_applied": accepted,
        "yaw_fallback_reason": yaw_fallback_reason,
        "best_diagnostic_yaw_deg": float(yaw_degrees[diagnostic_best_index]),
        "best_diagnostic_loss": diagnostic_best_loss,
        "initial_loss": (
            float(baseline_loss) if baseline_loss is not None else None
        ),
        "final_loss": float(selected_loss),
        "edge_final_loss": float(edge_losses[coarse_selected]),
        "external_edge_loss": float(selected_metrics[0]),
        "internal_edge_loss": float(
            selected_metrics[1] + orientation_weight * selected_metrics[2]
        ),
        "internal_position_loss": float(selected_metrics[1]),
        "internal_direction_loss": float(selected_metrics[2]),
        "internal_edge_count_loss": float(selected_metrics[5]),
        "internal_edge_normalized_loss": float(
            internal_normalized_losses[coarse_selected]
        ),
        "rgb_pixel_loss": float(selected_metrics[3]),
        "mask_iou": float(selected_metrics[4]),
        "has_internal_edges": has_internal_edges,
        "has_internal_orientation": has_internal_orientation,
        "has_internal_edge_count_loss": has_internal_edge_count_loss,
        "internal_edge_loss_includes_orientation": bool(
            has_internal_orientation
            and internal_edge_match_mode in ("chamfer", "partial_hausdorff")
        ),
        "internal_edge_match_mode": internal_edge_match_mode,
        "internal_source_tail_fraction": (
            0.10 if internal_edge_match_mode == "partial_hausdorff" else 0.0
        ),
        "internal_orientation_weight": float(orientation_weight),
        "reliable_source_internal_tangent_count": (
            int(np.count_nonzero(internal_orientation_features[2]))
            if has_internal_orientation else 0
        ),
        "reliable_target_internal_tangent_count": (
            int(np.count_nonzero(internal_orientation_features[3]))
            if has_internal_orientation else 0
        ),
        "external_edge_loss_weight": outer_weight,
        "internal_edge_loss_weight": internal_weight,
        "internal_position_loss_weight": internal_position_weight,
        "internal_direction_loss_weight": internal_direction_weight,
        "internal_edge_count_loss_weight": internal_edge_count_weight,
        "rgb_loss_weight": rgb_weight,
        "mask_iou_loss_weight": mask_iou_weight,
        "rgb_iou_direction_loss_weight": rgb_iou_direction_weight,
        "model_mask_information": (
            None if model_mask_information is None
            else float(model_mask_information)
        ),
        "dynamic_mask_group_weight": (
            None if model_mask_information is None else float(model_mask_weight)
        ),
        "dynamic_detail_group_weight": (
            None if model_mask_information is None else float(detail_weight)
        ),
        "dynamic_weight_source": (
            None if model_mask_information is None else
            "model_mask_180_degree_rotational_asymmetry"
        ),
        "external_edge_loss_scale": outer_scale,
        "internal_edge_loss_scale": internal_scale,
        "loss_normalization": loss_normalization,
        "direction_loss_backend": (
            getattr(direction_loss_fn, "backend", "external_direction_loss")
            if direction_loss_fn is not None else "legacy_geometry_appearance"
        ),
        "yaw_initializer": (
            "two_branch_3d_coarse_search"
            if coarse_metrics_fn is not None else
            "pca_gated_local_2d_coarse_search"
            if pca_local_search else
            "parallel_full_circle_2d_coarse_search"
        ),
        "selection_policy": (
            "best_candidate"
            if not use_zero_yaw_baseline else
            "improve_zero_yaw_baseline"
        ),
        "coarse_step_degrees": coarse_step_degrees,
        "coarse_candidate_count": len(coarse_yaws),
        "legacy_coarse_candidate_count": legacy_coarse_candidate_count,
        "direction_candidate_generation": direction_candidate_report,
        "coarse_backend": coarse_backend,
        "coarse_candidate_batch_size": (
            candidate_batch_size
            if coarse_metrics_fn is None and batch_metrics_fn is not None else None
        ),
        "coarse_selected_yaw_deg": float(math.degrees(yaw)),
        "coarse_selected_loss": float(coarse_losses[coarse_selected]),
        "current_direction_loss": float(coarse_losses[coarse_selected]),
        "opposite_direction_loss": float(coarse_losses[applied_opposite_index]),
        "local_refinement_count": 0,
        "direction_iou_gap": float(direction_iou_gap),
        "direction_iou_ambiguity_threshold": float(direction_iou_ambiguity_threshold),
        "direction_ambiguity_evaluated": direction_ambiguity_evaluated,
        "direction_ambiguous": bool(direction_ambiguous),
        "direction_locked_by_top_iou": bool(
            direction_ambiguity_evaluated and not direction_ambiguous
        ),
        "opposite_yaw_loss": float(coarse_losses[diagnostic_opposite_index]),
        "yaw_landscape": {
            "yaw_deg": [float(value) for value in yaw_degrees],
            "total_loss": [float(value) for value in coarse_losses],
            "unweighted_total_loss": [
                float(value) for value in unweighted_coarse_losses
            ],
            "loss_multiplier": [float(value) for value in loss_multipliers],
            "direction_loss": (
                [float(value) for value in direction_losses]
                if direction_losses is not None else None
            ),
            "external_edge_loss": [float(value[0]) for value in coarse_metrics_by_yaw],
            "internal_edge_loss": [
                float(value[1] + orientation_weight * value[2])
                for value in coarse_metrics_by_yaw
            ],
            "internal_position_loss": [
                float(value[1]) for value in coarse_metrics_by_yaw
            ],
            "internal_direction_loss": [
                float(value[2]) for value in coarse_metrics_by_yaw
            ],
            "internal_edge_count_loss": [
                float(value[5]) for value in coarse_metrics_by_yaw
            ],
            "rgb_loss": [float(value[3]) for value in coarse_metrics_by_yaw],
            "mask_iou": [float(value[4]) for value in coarse_metrics_by_yaw],
            "external_edge_max_normalized": [
                float(value) for value in outer_max_normalized
            ],
            "mask_iou_loss_max_normalized": [
                float(value) for value in mask_iou_max_normalized
            ],
            "internal_edge_max_normalized": [
                float(value) for value in rank_components.get(
                    "internal_position", np.ones(len(coarse_yaws)),
                )
            ],
            "internal_direction_max_normalized": [
                float(value) for value in rank_components.get(
                    "internal_direction", np.ones(len(coarse_yaws)),
                )
            ],
            "internal_edge_count_max_normalized": [
                float(value) for value in internal_edge_count_max_normalized
            ],
            "rgb_max_normalized": [
                float(value) for value in rank_components.get(
                    "rgb", np.ones(len(coarse_yaws)),
                )
            ],
            "best_index": diagnostic_best_index,
            "applied_index": coarse_selected,
            "opposite_index": diagnostic_opposite_index,
            "applied_opposite_index": applied_opposite_index,
            "second_basin_exclusion_deg": 10.0,
            "second_basin_index": second_basin_index,
            "second_basin_yaw_deg": float(yaw_degrees[second_basin_index]),
            "second_basin_loss": second_basin_loss,
            "second_basin_loss_margin": second_basin_loss - diagnostic_best_loss,
            "second_basin_relative_margin": (
                (second_basin_loss - diagnostic_best_loss)
                / max(abs(diagnostic_best_loss), 1e-6)
            ),
            "loss_p10": float(loss_p10),
            "loss_median": float(loss_median),
            "loss_p90": float(loss_p90),
            "loss_p90_p10_span": float(loss_p90 - loss_p10),
            "relative_flatness": float(
                (loss_p90 - loss_p10) / max(abs(loss_median), 1e-6)
            ),
        },
        "loss_formula": loss_formula,
        "candidate_selection": candidate_selection,
        "legacy_additive_weights_ignored": {
            "edge": float(edge_loss_weight),
            "rgb": float(rgb_loss_weight),
            "mask_iou": float(mask_iou_loss_weight),
        },
        "legacy_candidate_selection_alias": (
            candidate_selection if candidate_selection in ("weighted", "top2")
            else None
        ),
    }


def optimize_2d_edge_xy_scale_yaw_differentiable(
    *, source_mask, target_mask, source_rgb, target_rgb, base_scale_ratio,
    initial_yaw, baseline_yaw, iterations=50,
    lr_xy=0.03, lr_yaw=0.03, lr_scale=0.02,
    max_xy_fraction=0.10, yaw_limit_degrees=30.0,
    scale_bounds=(0.7, 1.3), lock_xy_scale=False,
    early_stop_patience=20, early_stop_min_delta=1e-4,
    max_resolution=512, sdf_weight=1.0, iou_weight=1.0,
    bbox_weight=0.5, bbox_center_weight=1.0, rgb_weight=1.0,
    xy_weight=0.005, scale_weight=0.005, yaw_weight=0.0001,
):
    """Refine the 2D affine pose with the stage-three geometric-product loss."""
    import torch
    import torch.nn.functional as functional

    device = alignment_torch_device()
    source_mask = (np.asarray(source_mask) > 0).astype(np.float32)
    target_mask = (np.asarray(target_mask) > 0).astype(np.float32)
    if not np.any(source_mask) or not np.any(target_mask):
        return np.zeros(2), float(initial_yaw), 1.0, 1.0, {
            "accepted": False,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "steps": 0,
            "optimized_parameters": ["x", "y", "scale_x", "scale_y", "yaw"],
        }
    height, width = target_mask.shape
    if source_mask.shape != target_mask.shape:
        source_mask = cv2.resize(
            source_mask, (width, height), interpolation=cv2.INTER_NEAREST,
        )
    source_rgb = np.asarray(source_rgb)
    if source_rgb.shape[:2] != (height, width):
        source_rgb = cv2.resize(
            source_rgb, (width, height), interpolation=cv2.INTER_AREA,
        )
    has_rgb = target_rgb is not None
    target_rgb = (
        np.asarray(target_rgb)
        if has_rgb else np.zeros((height, width, 3), dtype=np.float32)
    )
    if target_rgb.shape[:2] != (height, width):
        target_rgb = cv2.resize(
            target_rgb, (width, height), interpolation=cv2.INTER_AREA,
        )
    resize_ratio = min(1.0, float(max_resolution) / max(height, width))
    if resize_ratio < 1.0:
        size = (
            max(1, int(round(width * resize_ratio))),
            max(1, int(round(height * resize_ratio))),
        )
        source_mask = cv2.resize(
            source_mask, size, interpolation=cv2.INTER_NEAREST,
        )
        target_mask = cv2.resize(
            target_mask, size, interpolation=cv2.INTER_NEAREST,
        )
        source_rgb = cv2.resize(source_rgb, size, interpolation=cv2.INTER_AREA)
        target_rgb = cv2.resize(target_rgb, size, interpolation=cv2.INTER_AREA)
        height, width = target_mask.shape

    source_center, _ = _mask_geometry(source_mask)
    target_center, target_area = _mask_geometry(target_mask)
    pixel_scale = math.sqrt(target_area)
    source_bbox = robust_bbox(source_mask > 0)
    target_bbox = robust_bbox(target_mask > 0)
    source_corners = np.array([
        [source_bbox[0], source_bbox[1]],
        [source_bbox[2], source_bbox[1]],
        [source_bbox[2], source_bbox[3]],
        [source_bbox[0], source_bbox[3]],
    ], dtype=np.float32)
    target_low = np.asarray(target_bbox[:2], dtype=np.float32)
    target_high = np.asarray(target_bbox[2:], dtype=np.float32)
    target_size = np.maximum(target_high - target_low, 1.0)
    target_bbox_center = (target_low + target_high) * 0.5

    def normalized_rgb(image):
        image = np.asarray(image, dtype=np.float32)
        return image / 255.0 if float(image.max(initial=0.0)) > 1.0 else image

    source_alpha = torch.as_tensor(
        source_mask, dtype=torch.float32, device=device,
    )[None, None]
    target_alpha = torch.as_tensor(
        target_mask, dtype=torch.float32, device=device,
    )
    source_rgb_tensor = torch.as_tensor(
        normalized_rgb(source_rgb), dtype=torch.float32, device=device,
    ).permute(2, 0, 1)[None]
    target_rgb_tensor = torch.as_tensor(
        normalized_rgb(target_rgb), dtype=torch.float32, device=device,
    )
    target_sdf = torch.as_tensor(
        signed_distance_field(target_mask), dtype=torch.float32, device=device,
    )
    source_center_tensor = torch.as_tensor(
        source_center, dtype=torch.float32, device=device,
    )
    target_center_tensor = torch.as_tensor(
        target_center, dtype=torch.float32, device=device,
    )
    source_corners_tensor = torch.as_tensor(
        source_corners, dtype=torch.float32, device=device,
    )
    target_size_tensor = torch.as_tensor(
        target_size, dtype=torch.float32, device=device,
    )
    target_bbox_center_tensor = torch.as_tensor(
        target_bbox_center, dtype=torch.float32, device=device,
    )
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    output_grid = torch.stack((xs, ys), dim=-1)

    translation = torch.nn.Parameter(torch.zeros(2, device=device))
    yaw = torch.nn.Parameter(torch.tensor(float(initial_yaw), device=device))
    log_scales = torch.nn.Parameter(torch.zeros(1 if lock_xy_scale else 2, device=device))
    baseline_yaw = torch.tensor(float(baseline_yaw), device=device)
    initial_yaw_tensor = torch.tensor(float(initial_yaw), device=device)

    def effective_log_scales():
        return log_scales.expand(2) if lock_xy_scale else log_scales

    def screen_rotation(angle):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        return torch.stack((
            torch.stack((cosine, sine)),
            torch.stack((-sine, cosine)),
        ))

    def loss_value():
        linear = (
            float(base_scale_ratio)
            * screen_rotation(baseline_yaw + yaw)
            @ torch.diag(torch.exp(effective_log_scales()))
            @ screen_rotation(-baseline_yaw)
        )
        inverse = (
            screen_rotation(baseline_yaw)
            @ torch.diag(torch.exp(-effective_log_scales()))
            @ screen_rotation(-(baseline_yaw + yaw))
            / float(base_scale_ratio)
        )
        translation_pixels = translation * pixel_scale
        source_coordinates = (
            (output_grid - target_center_tensor - translation_pixels)
            @ inverse.T
            + source_center_tensor
        )
        sampling_grid = torch.stack((
            2.0 * source_coordinates[..., 0] / max(width - 1, 1) - 1.0,
            2.0 * source_coordinates[..., 1] / max(height - 1, 1) - 1.0,
        ), dim=-1)[None]
        warped_mask = functional.grid_sample(
            source_alpha, sampling_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )[0, 0]
        warped_rgb = functional.grid_sample(
            source_rgb_tensor, sampling_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )[0].permute(1, 2, 0)

        intersection = torch.sum(warped_mask * target_alpha)
        union = torch.sum(warped_mask) + torch.sum(target_alpha) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        outside = torch.sum(warped_mask * torch.relu(target_sdf)) / (
            torch.sum(warped_mask) + 1e-6
        )
        missing = torch.sum(
            (1.0 - warped_mask) * target_alpha * torch.relu(-target_sdf)
        ) / (torch.sum(target_alpha) + 1e-6)
        sdf_loss = outside + missing

        transformed_corners = (
            (source_corners_tensor - source_center_tensor) @ linear.T
            + target_center_tensor + translation_pixels
        )
        predicted_low = transformed_corners.min(dim=0).values
        predicted_high = transformed_corners.max(dim=0).values
        predicted_size = (predicted_high - predicted_low).clamp_min(1e-6)
        predicted_center = (predicted_low + predicted_high) * 0.5
        bbox_size_loss = torch.log(
            predicted_size / target_size_tensor.clamp_min(1e-6)
        ).square().sum()
        bbox_center_loss = (
            (predicted_center - target_bbox_center_tensor)
            / target_size_tensor.clamp_min(1e-6)
        ).square().sum()
        bbox_loss = bbox_size_loss + float(bbox_center_weight) * bbox_center_loss

        overlap = warped_mask * target_alpha
        rgb_loss = (
            torch.sum(
                overlap
                * torch.mean(torch.abs(warped_rgb - target_rgb_tensor), dim=2)
            )
            / (torch.sum(overlap) + 1e-6)
            if has_rgb else torch.zeros((), device=device)
        )
        log_similarity = (
            float(iou_weight) * torch.log(iou.clamp_min(1e-6))
            - float(sdf_weight) * sdf_loss
            - float(rgb_weight) * rgb_loss
        )
        data_loss = 1.0 - torch.exp(log_similarity.clamp_max(0.0))
        return (
            data_loss
            + float(bbox_weight) * bbox_loss
            + float(xy_weight) * translation.square().sum()
            + float(scale_weight) * effective_log_scales().square().sum()
            + float(yaw_weight) * (1.0 - torch.cos(yaw - initial_yaw_tensor))
        )

    initial_loss = float(loss_value().detach())
    optimizer = torch.optim.Adam([
        {"params": [translation], "lr": float(lr_xy)},
        {"params": [yaw], "lr": float(lr_yaw)},
        {"params": [log_scales], "lr": float(lr_scale)},
    ])
    yaw_low = float(initial_yaw) - math.radians(float(yaw_limit_degrees))
    yaw_high = float(initial_yaw) + math.radians(float(yaw_limit_degrees))
    log_scale_low, log_scale_high = map(math.log, scale_bounds)
    best_loss = initial_loss
    best_state = (
        translation.detach().clone(), yaw.detach().clone(),
        log_scales.detach().clone(),
    )
    stale_steps = 0
    completed_steps = 0
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        current_loss = float(loss.detach())
        previous_best = best_loss
        best_loss, stale_steps = early_stopping_update(
            current_loss, best_loss, stale_steps, early_stop_min_delta,
        )
        if best_loss < previous_best:
            best_state = (
                translation.detach().clone(), yaw.detach().clone(),
                log_scales.detach().clone(),
            )
        loss.backward()
        optimizer.step()
        completed_steps += 1
        with torch.no_grad():
            translation.clamp_(-float(max_xy_fraction), float(max_xy_fraction))
            yaw.clamp_(yaw_low, yaw_high)
            log_scales.clamp_(log_scale_low, log_scale_high)
        if early_stop_patience > 0 and stale_steps >= early_stop_patience:
            break

    candidate_loss = float(loss_value().detach())
    if candidate_loss < best_loss:
        best_state = (
            translation.detach().clone(), yaw.detach().clone(),
            log_scales.detach().clone(),
        )
    with torch.no_grad():
        translation.copy_(best_state[0])
        yaw.copy_(best_state[1])
        log_scales.copy_(best_state[2])

    final_loss = float(loss_value().detach())
    accepted = bool(np.isfinite(final_loss) and final_loss + 1e-8 < initial_loss)
    if not accepted:
        return np.zeros(2), float(initial_yaw), 1.0, 1.0, {
            "accepted": False,
            "initial_loss": initial_loss,
            "final_loss": initial_loss,
            "steps": int(completed_steps),
            "early_stopped": bool(completed_steps < int(iterations)),
            "loss": "stage_3_geometric_product_2d_warp",
            "optimized_parameters": ["x", "y", "scale_x", "scale_y", "yaw"],
        }
    ratios = torch.exp(effective_log_scales()).detach().cpu().numpy()
    return (
        translation.detach().cpu().numpy(), float(yaw.detach()),
        float(ratios[0]), float(ratios[1]),
        {
            "accepted": True,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "steps": int(completed_steps),
            "early_stopped": bool(completed_steps < int(iterations)),
            "xy_limit_fraction": float(max_xy_fraction),
            "yaw_limit_degrees": float(yaw_limit_degrees),
            "scale_ratio_bounds": [float(scale_bounds[0]), float(scale_bounds[1])],
            "xy_scale_tied": bool(lock_xy_scale),
            "loss": "stage_3_geometric_product_2d_warp",
            "optimized_parameters": ["x", "y", "scale_x", "scale_y", "yaw"],
        },
    )


def search_2d_edge_xy_scale_yaw_batched(
    groups, source_mask, target_mask, source_center, target_center,
    base_scale_ratio, *, initial_yaw, baseline_yaw,
    max_xy_fraction=0.10, yaw_limit_degrees=30.0,
    scale_bounds=(0.7, 1.3), lock_xy_scale=False, beam_size=4,
):
    """Coarse-to-fine GPU search over the local 2D affine transform."""
    import torch
    import torch.nn.functional as functional

    device = alignment_torch_device()
    source_mask = (np.asarray(source_mask) > 0).astype(np.float32)
    target_mask = (np.asarray(target_mask) > 0).astype(np.float32)
    source_center = np.asarray(source_center, dtype=np.float32)
    target_center = np.asarray(target_center, dtype=np.float32)
    target_area = max(float(target_mask.sum()), 1.0)
    pixel_scale = math.sqrt(target_area)
    source_y, source_x = np.nonzero(source_mask)
    target_y, target_x = np.nonzero(target_mask)
    if not len(source_x) or not len(target_x):
        return np.zeros(2), float(initial_yaw), 1.0, 1.0, {
            "accepted": False, "initial_loss": 0.0, "final_loss": 0.0,
            "steps": 0, "method": "batched_coarse_to_fine_affine_search",
        }

    source_radius = float(np.hypot(
        source_x - source_center[0], source_y - source_center[1],
    ).max())
    target_radius = float(np.hypot(
        target_x - target_center[0], target_y - target_center[1],
    ).max())
    translation_radius = float(max_xy_fraction) * pixel_scale
    output_radius = int(math.ceil(max(
        target_radius,
        float(base_scale_ratio) * float(scale_bounds[1]) * source_radius,
    ) + translation_radius + 4))
    height, width = target_mask.shape
    left = max(0, int(math.floor(target_center[0] - output_radius)))
    right = min(width, int(math.ceil(target_center[0] + output_radius)) + 1)
    top = max(0, int(math.floor(target_center[1] - output_radius)))
    bottom = min(height, int(math.ceil(target_center[1] + output_radius)) + 1)
    source_margin = 3
    source_left = max(0, int(source_x.min()) - source_margin)
    source_right = min(width, int(source_x.max()) + source_margin + 1)
    source_top = max(0, int(source_y.min()) - source_margin)
    source_bottom = min(height, int(source_y.max()) + source_margin + 1)
    source_alpha = torch.as_tensor(
        source_mask[source_top:source_bottom, source_left:source_right],
        device=device,
    )[None, None]
    target_alpha = torch.as_tensor(
        target_mask[top:bottom, left:right], device=device,
    )
    ys, xs = torch.meshgrid(
        torch.arange(top, bottom, device=device, dtype=torch.float32),
        torch.arange(left, right, device=device, dtype=torch.float32),
        indexing="ij",
    )
    output_grid = torch.stack((
        xs - float(target_center[0]), ys - float(target_center[1]),
    ), dim=-1)
    prepared_groups = [
        (
            torch.as_tensor(source, dtype=torch.float32, device=device),
            torch.as_tensor(target, dtype=torch.float32, device=device),
        )
        for source, target, _ in groups
    ]
    baseline_yaw = float(baseline_yaw)

    def rotations(angles):
        cosine, sine = torch.cos(angles), torch.sin(angles)
        return torch.stack((
            torch.stack((cosine, sine), dim=1),
            torch.stack((-sine, cosine), dim=1),
        ), dim=1)

    def metrics(parameters):
        parameters = torch.as_tensor(
            parameters, dtype=torch.float32, device=device,
        )
        outputs = [[], [], []]
        roi_pixels = max(1, int(target_alpha.numel()))
        batch_size = max(1, min(64, 2_000_000 // roi_pixels))
        for start in range(0, len(parameters), batch_size):
            current = parameters[start:start + batch_size]
            translation = current[:, :2]
            yaw = current[:, 2]
            scales = torch.exp(current[:, 3:5])
            linear = (
                rotations(yaw + baseline_yaw)
                @ torch.diag_embed(scales)
                @ rotations(torch.full_like(yaw, -baseline_yaw))
            )
            pixel_linear = float(base_scale_ratio) * linear
            determinant = (
                pixel_linear[:, 0, 0] * pixel_linear[:, 1, 1]
                - pixel_linear[:, 0, 1] * pixel_linear[:, 1, 0]
            )
            inverse = torch.stack((
                torch.stack((
                    pixel_linear[:, 1, 1], -pixel_linear[:, 0, 1],
                ), dim=1),
                torch.stack((
                    -pixel_linear[:, 1, 0], pixel_linear[:, 0, 0],
                ), dim=1),
            ), dim=1) / determinant[:, None, None]
            centered = output_grid[None] - (
                translation * pixel_scale
            )[:, None, None, :]
            source_coordinates = torch.einsum(
                "bhwj,bkj->bhwk", centered, inverse,
            ) + torch.as_tensor(source_center, device=device)
            grid_x = (
                2.0 * (source_coordinates[..., 0] - source_left)
                / max(source_right - source_left - 1, 1) - 1.0
            )
            grid_y = (
                2.0 * (source_coordinates[..., 1] - source_top)
                / max(source_bottom - source_top - 1, 1) - 1.0
            )
            warped = functional.grid_sample(
                source_alpha.expand(len(current), -1, -1, -1),
                torch.stack((grid_x, grid_y), dim=-1),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )[:, 0]
            intersection = (warped * target_alpha).sum(dim=(1, 2))
            union = (
                warped.sum(dim=(1, 2)) + target_alpha.sum() - intersection
            )
            iou = intersection / union.clamp_min(1e-6)
            losses = []
            for source, target in prepared_groups:
                if not len(source) or not len(target):
                    losses.append(torch.zeros(len(current), device=device))
                    continue
                transformed = (
                    torch.einsum("pj,bij->bpi", source, linear)
                    + translation[:, None, :]
                )
                distances = torch.cdist(
                    transformed,
                    target[None].expand(len(current), -1, -1),
                ).square()
                losses.append(
                    distances.min(dim=2).values.mean(dim=1)
                    + distances.min(dim=1).values.mean(dim=1)
                )
            outputs[0].append(iou)
            outputs[1].append(losses[0])
            outputs[2].append(
                losses[1] if len(losses) > 1
                else torch.zeros(len(current), device=device)
            )
        return tuple(
            torch.cat(values).detach().cpu().numpy() for values in outputs
        )

    yaw_low = float(initial_yaw) - math.radians(float(yaw_limit_degrees))
    yaw_high = float(initial_yaw) + math.radians(float(yaw_limit_degrees))
    log_low, log_high = map(math.log, scale_bounds)
    initial = np.array([0.0, 0.0, float(initial_yaw), 0.0, 0.0])
    beams = initial[None]
    schedules = (
        (0.04, math.radians(2.0), 0.08),
        (0.02, math.radians(1.0), 0.04),
        (0.01, math.radians(0.5), 0.02),
        (0.005, math.radians(0.25), 0.01),
    )
    signs = tuple(product((-1.0, 0.0, 1.0), repeat=4 if lock_xy_scale else 5))
    outer_floor = outer_scale = shape_tolerance = None
    total_candidates = 0
    level_reports = []
    final_parameters = final_shape = final_internal = None
    initial_shape = initial_internal = None

    for level, (xy_step, yaw_step, scale_step) in enumerate(schedules):
        candidates = []
        for beam in beams:
            for delta in signs:
                candidate = beam.copy()
                candidate[:2] += np.asarray(delta[:2]) * xy_step
                candidate[2] += delta[2] * yaw_step
                if lock_xy_scale:
                    candidate[3:5] += delta[3] * scale_step
                else:
                    candidate[3:5] += np.asarray(delta[3:5]) * scale_step
                candidate[:2] = np.clip(
                    candidate[:2], -max_xy_fraction, max_xy_fraction,
                )
                candidate[2] = np.clip(candidate[2], yaw_low, yaw_high)
                candidate[3:5] = np.clip(candidate[3:5], log_low, log_high)
                if lock_xy_scale:
                    candidate[4] = candidate[3]
                candidates.append(candidate)
        candidates = np.unique(np.round(candidates, 8), axis=0)
        total_candidates += len(candidates)
        iou, outer, internal = metrics(candidates)
        if outer_floor is None:
            outer_floor = float(outer.min())
            outer_scale = max(
                float(np.percentile(outer, 90)) - outer_floor, 1e-6,
            )
        shape = (
            0.7 * (1.0 - iou)
            + 0.3 * (outer - outer_floor) / outer_scale
        )
        if shape_tolerance is None:
            p10, p90 = np.percentile(shape, (10, 90))
            shape_tolerance = 0.05 * float(p90 - p10)
        best_shape = float(shape.min())
        shape_best_index = int(np.argmin(shape))
        near = np.flatnonzero(shape <= best_shape + shape_tolerance + 1e-12)
        ordered = [
            shape_best_index,
            *(
                index for index in near[np.argsort(internal[near], kind="stable")]
                if index != shape_best_index
            ),
        ]
        if len(ordered) < beam_size:
            remaining = [
                index for index in np.argsort(shape, kind="stable")
                if index not in set(ordered)
            ]
            ordered.extend(remaining[:beam_size - len(ordered)])
        selected = np.asarray(ordered[:beam_size], dtype=int)
        beams = candidates[selected]
        final_parameters = candidates
        final_shape = shape
        final_internal = internal
        if level == 0:
            initial_index = int(np.argmin(np.linalg.norm(
                candidates - initial, axis=1,
            )))
            initial_shape = float(shape[initial_index])
            initial_internal = float(internal[initial_index])
        level_reports.append({
            "level": level + 1,
            "candidate_count": int(len(candidates)),
            "best_shape_loss": best_shape,
            "near_optimal_count": int(len(near)),
            "xy_step_fraction": float(xy_step),
            "yaw_step_degrees": float(math.degrees(yaw_step)),
            "log_scale_step": float(scale_step),
        })

    best_shape = float(final_shape.min())
    near = np.flatnonzero(
        final_shape <= best_shape + shape_tolerance + 1e-12
    )
    selected_index = int(near[np.argmin(final_internal[near])])
    selected = final_parameters[selected_index]
    final_loss = float(final_shape[selected_index])
    final_internal_loss = float(final_internal[selected_index])
    accepted = bool(
        final_loss + 1e-8 < initial_shape
        or (
            final_loss <= initial_shape + shape_tolerance
            and final_internal_loss + 1e-8 < initial_internal
        )
    )
    if not accepted:
        selected = initial
        final_loss = initial_shape
        final_internal_loss = initial_internal
    ratios = np.exp(selected[3:5])
    return selected[:2], float(selected[2]), float(ratios[0]), float(ratios[1]), {
        "accepted": accepted,
        "initial_loss": float(initial_shape),
        "final_loss": float(final_loss),
        "initial_internal_edge_loss": float(initial_internal),
        "final_internal_edge_loss": float(final_internal_loss),
        "steps": len(schedules),
        "candidate_count": int(total_candidates),
        "beam_size": int(beam_size),
        "shape_tie_tolerance": float(shape_tolerance),
        "xy_limit_fraction": float(max_xy_fraction),
        "yaw_limit_degrees": float(yaw_limit_degrees),
        "scale_ratio_bounds": [float(scale_bounds[0]), float(scale_bounds[1])],
        "xy_scale_tied": bool(lock_xy_scale),
        "optimized_parameters": ["x", "y", "scale_x", "scale_y", "yaw"],
        "method": "batched_coarse_to_fine_affine_search",
        "backend": f"torch_{device.type}_batch",
        "loss_formula": (
            "0.7*(1-mask_iou) + 0.3*outer_robust_norm; "
            "internal_edges_near_optimal_tiebreak"
        ),
        "levels": level_reports,
    }


def refine_object_top_differentiable_edges(
    mesh, initial, target, appearance_target, target_edges, camera, config,
    fine_method="2d_affine", circle_score_threshold=0.85,
    square_ratio_threshold=CONTACT_BBOX_SQUARE_RATIO_THRESHOLD,
    scale_estimator="pixel_area", yaw_pca_source="mask", target_rgb=None,
    yaw_candidate_selection="rank", yaw_mask_iou_loss_weight=1.0,
    yaw_direction_iou_ambiguity_threshold=0.02,
    internal_orientation_weight=1.0,
    internal_edge_match_mode="chamfer",
    model_edge_method="normal_discontinuity",
    direction_scorer=None, direction_camera=None,
    direction_reference_rgb=None, direction_reference_mask=None,
    direction_views=None,
    timing_label=None,
    static_preprocessing=None,
    direction_batcher=None,
    coarse_refinement_semaphore=None,
    fine_refinement_semaphore=None,
):
    static_preprocessing = static_preprocessing or {}
    total_start = time.perf_counter()
    preprocess_timing = {}
    part_start = time.perf_counter()
    template_mask, template_edges, metadata = load_rendered_edge_template(
        initial["top_template_path"],
        (
            "mask_only" if model_edge_method == "normal_discontinuity"
            else "teed" if model_edge_method == "teed"
            else reference_edge_detector(initial.get("edge_detector", "canny"))
        ),
        mesh, initial["top_template_pose"], camera,
    )
    preprocess_timing["template_load"] = time.perf_counter() - part_start
    part_start = time.perf_counter()
    if model_edge_method == "normal_discontinuity":
        template_vertices = pose_vertices(
            center_vertices_on_table(mesh.vertices),
            initial["top_template_pose"]["uniform_scale"],
            initial["top_template_pose"]["translation_world_m"][:2],
            initial["top_template_pose"]["yaw_deg"],
            scale_xyz=initial["top_template_pose"].get("scale_xyz"),
        )
        template_edges = render_normal_edges_nvdiffrast(
            template_vertices, mesh.faces, camera,
        )
        metadata = {"method": "normal_discontinuity"}
    elif model_edge_method not in ("teed", "reference_image"):
        raise ValueError(f"unknown top model edge method: {model_edge_method}")
    preprocess_timing["model_edge_render"] = time.perf_counter() - part_start
    part_start = time.perf_counter()
    baseline = initial["top_template_pose"]
    source_center, source_area = _mask_geometry(template_mask)
    target_center, target_area = _mask_geometry(target)
    measured_area_scale_ratio = math.sqrt(target_area / source_area)
    rotated_bbox_scale_ratio = math.sqrt(
        rotated_bbox_area(target) / rotated_bbox_area(template_mask)
    )
    scale_ratio = {
        "pixel_area": measured_area_scale_ratio,
        "rotated_bbox": rotated_bbox_scale_ratio,
        "none": 1.0,
    }[scale_estimator]
    source_mask_points = _normalized_points(
        _sample_mask_points(template_mask, OT_POINT_COUNT), source_center, source_area,
    )
    target_mask_points = _normalized_points(
        _sample_mask_points(target, OT_POINT_COUNT), target_center, target_area,
    )
    source_outer = _normalized_points(
        sample_external_contour(template_mask, OT_POINT_COUNT), source_center, source_area,
    )
    target_outer = _normalized_points(
        sample_external_contour(target, OT_POINT_COUNT), target_center, target_area,
    )
    source_internal_edges = internal_edges(template_edges, template_mask)
    target_internal_edges = internal_edges(target_edges, appearance_target)
    preprocess_timing["mask_and_edge_sampling"] = (
        time.perf_counter() - part_start
    )
    part_start = time.perf_counter()
    source_internal_pixels, source_internal_tangents, source_internal_confidence = (
        sample_edge_features(
            source_internal_edges, OT_POINT_COUNT,
            sampling_mask=template_mask,
        )
    )
    preprocess_timing["source_edge_features"] = time.perf_counter() - part_start
    part_start = time.perf_counter()
    target_internal_pixels, target_internal_tangents, target_internal_confidence = (
        sample_edge_features(
            target_internal_edges, OT_POINT_COUNT,
            sampling_mask=appearance_target,
        )
    )
    preprocess_timing["target_edge_features"] = time.perf_counter() - part_start
    part_start = time.perf_counter()
    stage2_internal_edge_count_loss = internal_edge_count_loss(
        internal_edge_length_density(source_internal_edges, template_mask),
        internal_edge_length_density(target_internal_edges, appearance_target),
    )
    source_internal = _normalized_points(
        source_internal_pixels, source_center, source_area,
    )
    target_internal = _normalized_points(
        target_internal_pixels, target_center, target_area,
    )
    edge_groups = [
        (source_outer, target_outer, 0.5),
        (source_internal, target_internal, 0.5),
    ]
    template_rgb = np.asarray(Image.open(initial["top_template_path"]).convert("RGB"))
    expected_size = tuple(int(value) for value in camera["image_size"])
    if template_rgb.shape[1::-1] != expected_size:
        template_rgb = np.asarray(Image.fromarray(template_rgb).resize(
            expected_size, Image.Resampling.LANCZOS,
        ))
    rgb_loss_weight = float(os.environ.get("TOP_YAW_RGB_LOSS_WEIGHT", "1.0"))
    edge_loss_weight = float(os.environ.get("TOP_YAW_EDGE_LOSS_WEIGHT", "1.0"))

    def rgb_loss(yaw):
        return _warped_normalized_gradient_loss(
            template_rgb, template_mask, target_rgb, appearance_target,
            _analytic_template_transform(source_center, target_center, yaw, scale_ratio),
        )

    def mask_iou(yaw):
        return _mask_iou(
            template_mask, target,
            _analytic_template_transform(source_center, target_center, yaw, scale_ratio),
        )

    product_objective = (
        _mask_rgb_product_objective(
            template_rgb, template_mask, target_rgb, target,
            source_center, target_center, scale_ratio,
        )
        if target_rgb is not None else None
    )
    preprocess_timing["edge_metrics_and_rgb"] = time.perf_counter() - part_start

    part_start = time.perf_counter()
    top_circle_score = mask_circle_score(template_mask)
    contact_shape = static_preprocessing.get("contact_shape")
    if contact_shape is None:
        contact_shape = bottom_contact_shape(mesh)
    body_shape = static_preprocessing.get("body_shape")
    if body_shape is None:
        body_shape = body_cross_section_shape(
            mesh, circle_score_threshold, square_ratio_threshold,
        )
    tie_sources = object_xy_scale_tie_sources(
        top_circle_score, contact_shape,
        circle_score_threshold, square_ratio_threshold, body_shape,
    )
    lock_xy_scale = any(tie_sources.values())
    coarse_yaw_candidates, pca_gate_report = stage2_direction_yaw_candidates(
        template_mask, target,
    )
    preprocess_timing["shape_analysis"] = time.perf_counter() - part_start
    direction_scores = {}
    direction_loss_fn = None
    direction_timing = {
        "candidate_render": 0.0,
        "direction_score": 0.0,
    }
    if direction_scorer is not None:
        uses_multiview = hasattr(
            direction_scorer, "score_multiview_candidates",
        )
        if uses_multiview and (
            not direction_views
            or set(direction_views) != {"front", "top"}
        ):
            raise ValueError("stage-2 multiview direction context is incomplete")
        if not uses_multiview and any(value is None for value in (
            direction_camera, direction_reference_rgb, direction_reference_mask,
        )):
            raise ValueError("stage-2 direction context is incomplete")
        part_start = time.perf_counter()
        direction_vertex_rgb = (
            static_preprocessing["direction_vertex_rgb"]
            if "direction_vertex_rgb" in static_preprocessing else
            item_vertex_texture_colors(mesh, initial)
        )
        preprocess_timing["vertex_texture_colors"] = (
            time.perf_counter() - part_start
        )

        def direction_loss_fn(yaw):
            return direction_scores[float(yaw)]["loss"]

        def prepare_direction_candidates(yaws):
            if direction_batcher is not None:
                entries = []
                for yaw in yaws:
                    key = float(yaw)
                    candidate = {
                        **initial, **baseline,
                        "yaw_deg": float(
                            baseline["yaw_deg"] + math.degrees(key)
                        ),
                    }
                    entries.append((
                        key, mesh, candidate, direction_vertex_rgb,
                    ))
                direction_scores.update(direction_batcher.score(
                    timing_label,
                    {
                        "reference_key": "top",
                        "reference_rgb": direction_reference_rgb,
                        "reference_mask": direction_reference_mask,
                        "entries": entries,
                    },
                ))
                return
            render_start = time.perf_counter()
            inputs = (
                {"front": {}, "top": {}} if uses_multiview else {}
            )
            for yaw in yaws:
                key = float(yaw)
                candidate = {
                    **initial, **baseline,
                    "yaw_deg": float(baseline["yaw_deg"] + math.degrees(key)),
                }
                if uses_multiview:
                    for view, context in direction_views.items():
                        inputs[view][key] = rendered_direction_candidate(
                            mesh, candidate, context["camera"],
                            direction_vertex_rgb, direction_scorer,
                        )
                else:
                    inputs[key] = rendered_direction_candidate(
                        mesh, candidate, direction_camera,
                        direction_vertex_rgb, direction_scorer,
                    )
            direction_timing["candidate_render"] += (
                time.perf_counter() - render_start
            )
            score_start = time.perf_counter()
            if uses_multiview:
                direction_scores.update(score_multiview_direction_candidates(
                    direction_scorer, "top",
                    {
                        view: context["reference_rgb"]
                        for view, context in direction_views.items()
                    },
                    {
                        view: context["reference_mask"]
                        for view, context in direction_views.items()
                    },
                    {
                        view: context["camera"]
                        for view, context in direction_views.items()
                    },
                    inputs,
                ))
            else:
                direction_scores.update(score_direction_candidates(
                    direction_scorer, "top", direction_reference_rgb,
                    direction_reference_mask, direction_camera, inputs,
                ))
            direction_timing["direction_score"] += (
                time.perf_counter() - score_start
            )

        direction_loss_fn.prepare = prepare_direction_candidates
        direction_loss_fn.backend = (
            f"{getattr(direction_scorer, 'backend', 'dino_relative_position')}_"
            f"{'front_top' if uses_multiview else 'top'}"
        )
        direction_loss_fn.loss_formula = getattr(
            direction_scorer, "loss_formula",
            "DINO_relative_position_consistency",
        )
    preprocess_seconds = time.perf_counter() - total_start
    preprocess_timing["unattributed"] = max(
        0.0, preprocess_seconds - sum(preprocess_timing.values()),
    )
    coarse_start = time.perf_counter()
    def run_coarse_refinement():
        return optimize_2d_edge_yaw_coarse_multistart(
            edge_groups,
            source_mask_points,
            target_mask_points,
            iterations=config.iterations,
            lr_yaw=config.lr_yaw,
            early_stop_patience=config.patience,
            early_stop_min_delta=config.early_stop_min_delta,
            rgb_loss_fn=rgb_loss if target_rgb is not None else None,
            edge_loss_weight=edge_loss_weight,
            rgb_loss_weight=rgb_loss_weight,
            mask_iou_fn=mask_iou,
            mask_iou_loss_weight=yaw_mask_iou_loss_weight,
            objective_loss_fn=product_objective,
            candidate_selection=yaw_candidate_selection,
            direction_iou_ambiguity_threshold=yaw_direction_iou_ambiguity_threshold,
            coarse_yaw_candidates=coarse_yaw_candidates,
            internal_orientation_features=(
                source_internal_tangents,
                target_internal_tangents,
                source_internal_confidence,
                target_internal_confidence,
            ),
            internal_orientation_weight=internal_orientation_weight,
            internal_edge_match_mode=internal_edge_match_mode,
            internal_edge_count_loss_fn=(
                lambda unused_yaw: stage2_internal_edge_count_loss
            ),
            rank_shape_component="paired_dynamic",
            rank_rgb_component_weight=1.0,
            model_mask_information=model_mask_rotational_information(template_mask),
            direction_loss_fn=direction_loss_fn,
            direction_candidate_window_centers_deg=(
                pca_gate_report.get("candidate_window_centers_deg")
            ),
            use_zero_yaw_baseline=(
                coarse_yaw_candidates is None and direction_scorer is None
            ),
            gpu_semaphore=coarse_refinement_semaphore,
        )

    yaw_radians, direction_report = run_coarse_refinement()
    coarse_total_seconds = time.perf_counter() - coarse_start
    legacy_coarse_seconds = max(
        0.0,
        coarse_total_seconds
        - direction_timing["candidate_render"]
        - direction_timing["direction_score"],
    )
    direction_report["pca_gate"] = pca_gate_report
    direction_report["rgb_loss_type"] = "normalized_sobel_gradient_correlation"
    if direction_scores:
        direction_report["direction_candidates"] = [
            {"yaw_deg": float(math.degrees(yaw)), **score}
            for yaw, score in direction_scores.items()
        ]
    fine_start = time.perf_counter()

    def run_fine_refinement(function):
        if fine_refinement_semaphore is None:
            return function()
        with fine_refinement_semaphore:
            return function()

    if fine_method == "2d_affine_search":
        normalized_xy, yaw_radians, scale_x_ratio, scale_y_ratio, fine_report = (
            run_fine_refinement(lambda: search_2d_edge_xy_scale_yaw_batched(
                edge_groups,
                template_mask, target, source_center, target_center, scale_ratio,
                initial_yaw=yaw_radians,
                baseline_yaw=math.radians(float(baseline["yaw_deg"])),
                max_xy_fraction=config.max_xy_fraction,
                yaw_limit_degrees=30.0,
                scale_bounds=(0.7, 1.3),
                lock_xy_scale=lock_xy_scale,
            ))
        )
    elif fine_method == "2d_affine":
        normalized_xy, yaw_radians, scale_x_ratio, scale_y_ratio, fine_report = (
            run_fine_refinement(lambda: optimize_2d_edge_xy_scale_yaw_differentiable(
                source_mask=template_mask,
                target_mask=target,
                source_rgb=template_rgb,
                target_rgb=target_rgb,
                base_scale_ratio=scale_ratio,
                initial_yaw=yaw_radians,
                baseline_yaw=math.radians(float(baseline["yaw_deg"])),
                iterations=config.fine_iterations,
                lr_xy=config.lr_xy,
                lr_yaw=config.lr_yaw,
                lr_scale=config.lr_scale,
                max_xy_fraction=config.max_xy_fraction,
                scale_bounds=(0.7, 1.3),
                lock_xy_scale=lock_xy_scale,
                early_stop_patience=config.patience,
                early_stop_min_delta=config.early_stop_min_delta,
                max_resolution=config.fine_resolution,
                sdf_weight=config.sdf_weight,
                iou_weight=config.iou_weight,
                bbox_weight=0.0,
                bbox_center_weight=config.bbox_center_weight,
                rgb_weight=0.0,
                xy_weight=config.xy_weight,
                scale_weight=config.scale_weight,
                yaw_weight=config.yaw_weight,
            ))
        )
    elif fine_method == "differentiable_3d_mask":
        normalized_xy = np.zeros(2)
        scale_x_ratio = scale_y_ratio = 1.0
        fine_report = {
            "accepted": False,
            "initial_loss": direction_report["final_loss"],
            "final_loss": direction_report["final_loss"],
            "steps": 0,
            "deferred_to": "differentiable_3d_mask",
        }
    else:
        raise ValueError(f"unknown top fine refinement method: {fine_method}")
    fine_seconds = time.perf_counter() - fine_start
    total_seconds = time.perf_counter() - total_start
    timing_seconds = {
        "preprocess": float(preprocess_seconds),
        "preprocess_breakdown": {
            name: float(seconds)
            for name, seconds in preprocess_timing.items()
        },
        "legacy_coarse": float(legacy_coarse_seconds),
        "candidate_render": float(direction_timing["candidate_render"]),
        "direction_score": float(direction_timing["direction_score"]),
        "fine_refinement": float(fine_seconds),
        "total": float(total_seconds),
    }
    print(
        "[top timing] "
        f"{timing_label or Path(initial['geometry_asset']).stem}: "
        + " ".join(
            f"{name}={seconds:.3f}s"
            for name, seconds in timing_seconds.items()
            if name != "preprocess_breakdown"
        ),
        flush=True,
    )
    print(
        "[top timing preprocess] "
        f"{timing_label or Path(initial['geometry_asset']).stem}: "
        + " ".join(
            f"{name}={seconds:.3f}s"
            for name, seconds in preprocess_timing.items()
        ),
        flush=True,
    )
    pixel_delta = np.asarray(normalized_xy, dtype=float) * math.sqrt(target_area)
    baseline_yaw = math.radians(float(baseline["yaw_deg"]))
    linear = (
        scale_ratio
        * _screen_rotation(baseline_yaw + yaw_radians)
        @ np.diag([scale_x_ratio, scale_y_ratio])
        @ _screen_rotation(-baseline_yaw)
    )
    world_xy = _mask_linear_alignment_to_world(
        source_center, np.asarray(target_center) + pixel_delta,
        baseline, camera, linear,
    )
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = target_center + pixel_delta - linear @ source_center
    uniform_scale = float(baseline["uniform_scale"] * scale_ratio)
    result = {
        **initial,
        "top_circle_score": float(top_circle_score),
        "contact_circle_score": float(contact_shape["circle_score"]),
        "contact_bbox_square_ratio": float(contact_shape["bbox_square_ratio"]),
        "body_cross_section_shape": body_shape,
        "translation_world_m": [float(world_xy[0]), float(world_xy[1]), 0.0],
        "yaw_deg": float(((baseline["yaw_deg"] + math.degrees(yaw_radians) + 180.0) % 360.0) - 180.0),
        "uniform_scale": uniform_scale,
        "top_template_transform": transform.tolist(),
        "top_template_transform_image_size": list(camera.get(
            "image_size", [template_mask.shape[1], template_mask.shape[0]],
        )),
        "loss": fine_report["final_loss"],
        "success": bool(direction_report["accepted"] or fine_report["accepted"]),
        "top_differentiable_edge_refinement": {
            **fine_report,
            "method": "parallel_gpu_dense_2d_yaw_then_fine_refinement",
            "fine_method": fine_method,
            "loss_weights": {"external_contour": 0.5, "internal_edges": 0.5},
            "joint_refinement_loss_weights": {
                "external_contour": 0.5, "internal_edges": 0.5,
            },
            "pca_direction_yaw": direction_report,
            "joint_xy_scale_yaw": fine_report,
            "timing_seconds": timing_seconds,
            "edge_detector": metadata,
            "model_edge_method": metadata.get("method", model_edge_method),
            "top_circle_score": float(top_circle_score),
            "top_circle_score_source": "model_top_render",
            "contact_circle_score": float(contact_shape["circle_score"]),
            "contact_bbox_square_ratio": float(contact_shape["bbox_square_ratio"]),
            "contact_circle_score_source": "lowest_downward_mesh_facet",
            "contact_circle_score_threshold": float(circle_score_threshold),
            "contact_bbox_square_ratio_threshold": float(square_ratio_threshold),
            "body_cross_section_shape": body_shape,
            "body_cross_section_score_source": "stable_horizontal_mesh_sections",
            "xy_scale_tie_rule": (
                "top_view_circle_or_contact_circle_and_square_or_"
                "stable_body_cross_section"
            ),
            "xy_scale_tie_sources": tie_sources,
            "measured_area_scale_ratio": float(measured_area_scale_ratio),
            "rotated_bbox_scale_ratio": float(rotated_bbox_scale_ratio),
            "applied_area_scale_ratio": float(scale_ratio),
            "scale_estimator": scale_estimator,
            "yaw_initializer": direction_report["yaw_initializer"],
            "yaw_pca_source": "mask",
            "requested_yaw_pca_source": yaw_pca_source,
            "xy_scale_tied": bool(lock_xy_scale),
        },
    }
    if fine_method in ("2d_affine_search", "2d_affine"):
        result["scale_xyz"] = [
            float(uniform_scale * scale_x_ratio),
            float(uniform_scale * scale_y_ratio),
            uniform_scale,
        ]
    else:
        result.pop("scale_xyz", None)
    return result


def fast_refine_object_top_staged(mesh, initial, target, target_edges, camera):
    """Solve scale and XY analytically; use OT only for the residual top-view yaw."""
    template_mask, template_edges, metadata = load_rendered_edge_template(
        initial["top_template_path"], initial.get("edge_detector", "canny"),
        mesh, initial["top_template_pose"], camera,
    )
    baseline = initial["top_template_pose"]
    source_centroid, source_area = _mask_geometry(template_mask)
    target_centroid, target_area = _mask_geometry(target)
    scale_ratio = math.sqrt(target_area / source_area)

    source_region = _normalized_points(_sample_mask_points(template_mask), source_centroid, source_area)
    target_region = _normalized_points(_sample_mask_points(target), target_centroid, target_area)
    source_contour = _normalized_points(
        sample_external_contour(template_mask, OT_POINT_COUNT), source_centroid, source_area,
    )
    target_contour = _normalized_points(
        sample_external_contour(target, OT_POINT_COUNT), target_centroid, target_area,
    )
    source_internal = _normalized_points(
        sample_edge_points(
            internal_edges(template_edges, template_mask), OT_POINT_COUNT,
        ), source_centroid, source_area,
    )
    target_internal = _normalized_points(
        sample_edge_points(
            internal_edges(target_edges, target), OT_POINT_COUNT,
        ), target_centroid, target_area,
    )
    yaw_groups = [
        (source_region, target_region, 0.15),
        (source_contour, target_contour, 0.25),
        (source_internal, target_internal, 0.60),
    ]
    primary_yaw, primary_report = _optimize_yaw(
        yaw_groups,
        limit_degrees=60.0,
        prior_weight=0.02,
        iterations=MASK_OT_ITERATIONS,
    )
    flipped_yaw, flipped_report = _optimize_yaw(
        yaw_groups,
        initial_yaw=math.pi,
        limit_degrees=60.0,
        prior_weight=0.02,
        iterations=MASK_OT_ITERATIONS,
    )
    primary_score = _combined_yaw_loss(yaw_groups, primary_yaw)
    flipped_score = _combined_yaw_loss(yaw_groups, flipped_yaw)
    direction_margin = max(1e-6, 0.01 * max(primary_score, 1e-6))
    direction_flipped = bool(flipped_score + direction_margin < primary_score)
    yaw_radians = flipped_yaw if direction_flipped else primary_yaw
    mask_report = dict(flipped_report if direction_flipped else primary_report)
    mask_report.update({
        "local_yaw_delta_deg": mask_report["yaw_delta_deg"],
        "yaw_delta_deg": float(math.degrees(yaw_radians)),
        "primary_direction_loss": float(primary_score),
        "flipped_direction_loss": float(flipped_score),
        "binary_180_direction_flipped": direction_flipped,
    })
    transform = _analytic_template_transform(
        source_centroid, target_centroid, yaw_radians, scale_ratio,
    )
    identity = np.eye(3)
    mask_report.update({
        "scale_ratio_from_area": float(scale_ratio),
        "initial_iou": _mask_iou(template_mask, target, identity),
        "candidate_iou": _mask_iou(template_mask, target, transform),
        "internal_edge_initial_loss": _yaw_measure_loss(source_internal, target_internal, 0.0),
        "internal_edge_candidate_loss": _yaw_measure_loss(
            source_internal, target_internal, yaw_radians,
        ),
    })
    internal_initial = mask_report["internal_edge_initial_loss"]
    internal_candidate = mask_report["internal_edge_candidate_loss"]
    internal_direction_observable = _internal_direction_observable(
        source_internal, target_internal,
    )
    internal_not_worse = bool(
        not internal_direction_observable
        or internal_initial is None
        or internal_candidate <= internal_initial + max(1e-6, 0.01 * internal_initial)
    )
    mask_report["internal_direction_observable"] = internal_direction_observable
    mask_report["internal_edge_not_worse"] = internal_not_worse
    mask_accepted = bool(
        internal_not_worse
        and mask_report["candidate_iou"] + 0.005 >= mask_report["initial_iou"]
    )
    if not mask_accepted:
        yaw_radians = 0.0
        scale_ratio = 1.0
        transform = identity
    mask_report["pose_accepted"] = bool(mask_accepted)

    refined_yaw, edge_report = _optimize_yaw([
        (source_contour, target_contour, 0.25),
        (source_internal, target_internal, 0.75),
    ], initial_yaw=yaw_radians, limit_degrees=30.0, prior_weight=0.06, iterations=EDGE_OT_ITERATIONS)
    edge_transform = _analytic_template_transform(
        source_centroid, target_centroid, refined_yaw, scale_ratio,
    )
    edge_report.update({
        "initial_iou": _mask_iou(template_mask, target, transform),
        "candidate_iou": _mask_iou(template_mask, target, edge_transform),
        "internal_edge_initial_loss": _yaw_measure_loss(
            source_internal, target_internal, yaw_radians,
        ),
        "internal_edge_candidate_loss": _yaw_measure_loss(
            source_internal, target_internal, refined_yaw,
        ),
    })
    edge_internal_initial = edge_report["internal_edge_initial_loss"]
    edge_internal_candidate = edge_report["internal_edge_candidate_loss"]
    edge_internal_not_worse = bool(
        not internal_direction_observable
        or edge_internal_initial is None
        or edge_internal_candidate
        <= edge_internal_initial + max(1e-6, 0.01 * edge_internal_initial)
    )
    edge_report["internal_direction_observable"] = internal_direction_observable
    edge_report["internal_edge_not_worse"] = edge_internal_not_worse
    edge_accepted = bool(
        edge_report["accepted"]
        and edge_internal_not_worse
        and edge_report["candidate_iou"] + 0.005 >= edge_report["initial_iou"]
    )
    if edge_accepted:
        yaw_radians = refined_yaw
        transform = edge_transform
    edge_report["pose_accepted"] = edge_accepted

    aligned_target_centroid = target_centroid if mask_accepted else source_centroid
    world_xy = _mask_centroid_alignment_to_world(
        source_centroid, aligned_target_centroid, baseline, camera,
        yaw_radians, scale_ratio,
    )
    result = {
        **initial,
        "translation_world_m": [float(world_xy[0]), float(world_xy[1]), 0.0],
        "yaw_deg": float(((baseline["yaw_deg"] + math.degrees(yaw_radians) + 180.0) % 360.0) - 180.0),
        "uniform_scale": float(baseline["uniform_scale"] * scale_ratio),
        "top_template_transform": transform.tolist(),
    }
    normalization = max(float(np.ptp(np.vstack((source_contour, target_contour)), axis=0).max()), 0.1)
    result.update({
        "loss": float(
            _sinkhorn_divergence(
                source_contour @ _screen_rotation(yaw_radians).T,
                target_contour,
                normalization,
            ) + 0.15 * (1.0 - _mask_iou(template_mask, target, transform))
        ),
        "success": True,
        "top_staged_refinement": {
            "method": "mask_centroid_analytic_scale_xy_mask_contour_internal_yaw_ot",
            "source_mask_centroid_pixel": [float(value) for value in source_centroid],
            "target_mask_centroid_pixel": [float(value) for value in target_centroid],
            "mask": mask_report,
            "edge": edge_report,
            "discrete_yaw_search": False,
            "binary_180_direction_check": True,
            "binary_180_direction_flipped": direction_flipped,
            "joint_xy_yaw_scale_optimization": False,
        },
        "top_template_edge_detector": metadata,
    })
    if metadata["method"] == "canny":
        result["top_template_canny_thresholds"] = [
            metadata["low_threshold"], metadata["high_threshold"],
        ]
    return result

# The fast implementation is selected locally; this file does not import or
# monkey-patch align_scene_two_camera.py.
fit_plane_ransac = fast_fit_plane_ransac
refine_front_table_registration = fast_refine_front_table_registration
refine_pose_to_pointcloud = fast_refine_pose_to_pointcloud
refine_object_top_staged = fast_refine_object_top_staged
render_stages = persistent_render_stages
render_top_edge_templates = persistent_render_top_edge_templates


def main():
    print(
        "[standalone fast pipeline] "
        f"ransac={RANSAC_ITERATIONS} "
        f"table_samples={TABLE_SAMPLE_COUNT} "
        f"object_samples={OBJECT_SAMPLE_COUNT} "
        f"target_points={TARGET_POINT_LIMIT} "
        f"powell_maxiter={POWELL_MAX_ITERATIONS} "
        f"ot_points={OT_POINT_COUNT} "
        f"mask_ot_iterations={MASK_OT_ITERATIONS} "
        f"edge_ot_iterations={EDGE_OT_ITERATIONS}",
        flush=True,
    )
    try:
        run(parse_args())
    finally:
        if _blender_worker is not None:
            _blender_worker.close()


if __name__ == "__main__":
    main()
