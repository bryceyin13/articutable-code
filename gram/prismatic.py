from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import trimesh

from .artifacts import KinematicSpec, read_json, write_json
from .parts import LinkGeometry, save_link_meshes


AUTO_OPEN_RATIO = 0.1
DRAWER_WALL_WIDTH_RATIO = 0.03
TOP_DRAWER_SAMPLE_GRID_SIZE = 64
TOP_DRAWER_SAMPLE_EDGE_MARGIN_RATIO = 0.05
TOP_DRAWER_SAMPLE_ORIGIN_OFFSET_RATIO = 0.05
TOP_DRAWER_LOWEST_SAMPLE_RATIO = 0.05
TOP_DRAWER_WALL_CLEARANCE_RATIO = 0.05
TOP_DRAWER_MIN_VALID_HITS = 32
TOP_DRAWER_HEIGHT_EPSILON_RATIO = 0.0001
NOTCH_SHAPE_IOU_THRESHOLD = 0.8
NOTCH_THICKNESS_RATIO_THRESHOLD = 0.35


def _span(mesh: trimesh.Trimesh, axis: np.ndarray) -> tuple[float, float]:
    values = np.asarray(mesh.vertices, dtype=float) @ axis
    if len(values) > 200:
        return tuple(np.quantile(values, (0.01, 0.99)))
    return float(values.min()), float(values.max())


def _box(axis, side, up, extents, coordinates):
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((axis, side, up))
    transform[:3, 3] = (
        axis * coordinates[0] + side * coordinates[1] + up * coordinates[2]
    )
    return trimesh.creation.box(extents=extents, transform=transform)


def _drawer_frame(mesh: trimesh.Trimesh, axis: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 1.0, 0.0])
    up = world_up - axis * float(world_up @ axis)
    if np.linalg.norm(up) < 0.5:
        candidates = [vector for vector in np.eye(3) if abs(vector @ axis) < 0.5]
        up = min(candidates, key=lambda vector: _span(mesh, vector)[1]
                 - _span(mesh, vector)[0])
    up /= np.linalg.norm(up)
    side = np.cross(up, axis)
    side /= np.linalg.norm(side)
    return side, up


def _sample_cabinet_top(
        parent: trimesh.Trimesh, axis: np.ndarray, up: np.ndarray,
        drawer_center: float,
) -> tuple[float | None, dict]:
    side = np.cross(up, axis)
    side /= np.linalg.norm(side)
    vertices = np.asarray(parent.vertices, dtype=float)
    axis_values = vertices @ axis
    side_values = vertices @ side
    up_values = vertices @ up
    axis_lo, axis_hi = float(axis_values.min()), float(axis_values.max())
    side_lo, side_hi = float(side_values.min()), float(side_values.max())
    up_lo, up_hi = float(up_values.min()), float(up_values.max())
    axis_margin = TOP_DRAWER_SAMPLE_EDGE_MARGIN_RATIO * (axis_hi - axis_lo)
    side_margin = TOP_DRAWER_SAMPLE_EDGE_MARGIN_RATIO * (side_hi - side_lo)
    axis_samples = np.linspace(
        axis_lo + axis_margin, axis_hi - axis_margin,
        TOP_DRAWER_SAMPLE_GRID_SIZE,
    )
    side_samples = np.linspace(
        side_lo + side_margin, side_hi - side_margin,
        TOP_DRAWER_SAMPLE_GRID_SIZE,
    )
    axis_grid, side_grid = np.meshgrid(axis_samples, side_samples, indexing="ij")
    ray_up = up_hi + TOP_DRAWER_SAMPLE_ORIGIN_OFFSET_RATIO * (up_hi - up_lo)
    origins = (
        np.outer(axis_grid.ravel(), axis)
        + np.outer(side_grid.ravel(), side)
        + ray_up * up
    )
    directions = np.tile(-up, (len(origins), 1))
    metadata = {
        "top_sample_grid_size": TOP_DRAWER_SAMPLE_GRID_SIZE,
        "top_sample_edge_margin_ratio": TOP_DRAWER_SAMPLE_EDGE_MARGIN_RATIO,
        "top_sample_origin_offset_ratio": TOP_DRAWER_SAMPLE_ORIGIN_OFFSET_RATIO,
        "top_sample_lowest_ratio": TOP_DRAWER_LOWEST_SAMPLE_RATIO,
        "top_sample_total_rays": len(origins),
    }
    try:
        locations, ray_indices, _ = parent.ray.intersects_location(
            origins, directions, multiple_hits=False,
        )
    except Exception as exc:
        return None, {
            **metadata,
            "top_sample_status": "unavailable",
            "top_sample_error": f"{type(exc).__name__}: {exc}",
        }

    first_heights = np.full(len(origins), -np.inf)
    np.maximum.at(first_heights, ray_indices, np.asarray(locations) @ up)
    first_heights = first_heights[np.isfinite(first_heights)]
    valid = first_heights[first_heights >= drawer_center]
    metadata.update({
        "top_sample_first_hits": len(first_heights),
        "top_sample_below_center_rejected": len(first_heights) - len(valid),
        "top_sample_valid_hits": len(valid),
    })
    if len(valid) < TOP_DRAWER_MIN_VALID_HITS:
        return None, {**metadata, "top_sample_status": "insufficient"}

    lowest_count = max(1, int(np.ceil(
        TOP_DRAWER_LOWEST_SAMPLE_RATIO * len(valid))))
    lowest = np.partition(valid, lowest_count - 1)[:lowest_count]
    limit = float(lowest.mean())
    return limit, {
        **metadata,
        "top_sample_status": "ok",
        "top_sample_lowest_count": lowest_count,
        "cabinet_top_limit": limit,
    }


def _remove_thin_notch_membrane(
        mesh: trimesh.Trimesh, axis: np.ndarray, side: np.ndarray,
        up: np.ndarray, resolution: int = 192,
) -> tuple[trimesh.Trimesh, dict]:
    """Remove a thin reconstructed membrane behind a top-edge finger pull."""
    from PIL import Image, ImageDraw

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    centers = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    depth = vertices @ axis
    side_values = vertices @ side
    up_values = vertices @ up
    face_depth = centers @ axis
    face_side = centers @ side
    face_up = centers @ up
    facing = normals @ axis
    panel_back, panel_front = np.quantile(depth, (0.01, 0.99))
    panel_thickness = float(panel_front - panel_back)
    metadata = {
        "detected": False,
        "removed_faces": 0,
        "shape_iou": 0.0,
        "candidate_thickness": None,
        "panel_thickness": panel_thickness,
        "thickness_ratio": None,
    }
    if panel_thickness <= 0 or len(faces) == 0:
        return mesh, metadata

    side_lo, side_hi = np.quantile(side_values, (0.01, 0.99))
    up_lo, up_hi = np.quantile(up_values, (0.01, 0.99))
    width, height = side_hi - side_lo, up_hi - up_lo
    if min(width, height) <= 0:
        return mesh, metadata

    def pixels(points):
        x = (points @ side - side_lo) / width * (resolution - 1)
        y = (up_hi - points @ up) / height * (resolution - 1)
        return list(zip(x, y))

    front_mask = Image.new("1", (resolution, resolution))
    draw = ImageDraw.Draw(front_mask)
    front_faces = (
        (facing > 0.65)
        & (face_depth > panel_front - 0.1 * panel_thickness)
    )
    for face in faces[front_faces]:
        draw.polygon(pixels(vertices[face]), fill=1)
    front_pixels = np.asarray(front_mask, dtype=bool)
    envelope = np.full(resolution, np.nan)
    for column in range(resolution):
        rows = np.flatnonzero(front_pixels[:, column])
        if len(rows):
            envelope[column] = rows[0]
    valid = np.isfinite(envelope)
    if np.count_nonzero(valid) < resolution // 2:
        return mesh, metadata
    envelope = np.interp(
        np.arange(resolution), np.flatnonzero(valid), envelope[valid],
    )
    quarter = max(resolution // 4, 1)
    top = float(np.median(np.r_[envelope[:quarter], envelope[-quarter:]]))
    gap = envelope > top + max(3, 0.04 * resolution)

    runs = []
    start = None
    for index, is_gap in enumerate(np.r_[gap, False]):
        if is_gap and start is None:
            start = index
        elif not is_gap and start is not None:
            if start > 0 and index < resolution:
                runs.append((start, index))
            start = None
    if not runs:
        return mesh, metadata
    gap_start, gap_stop = max(runs, key=lambda run: run[1] - run[0])
    if gap_stop - gap_start < max(4, int(0.03 * resolution)):
        return mesh, metadata

    notch = np.zeros((resolution, resolution), dtype=bool)
    top_row = max(int(np.floor(top)), 0)
    for column in range(gap_start, gap_stop):
        bottom_row = min(int(np.ceil(envelope[column])), resolution)
        notch[top_row:bottom_row, column] = True
    if not notch.any():
        return mesh, metadata

    center_x = (face_side - side_lo) / width * (resolution - 1)
    center_y = (up_hi - face_up) / height * (resolution - 1)
    ix = np.clip(np.rint(center_x).astype(int), 0, resolution - 1)
    iy = np.clip(np.rint(center_y).astype(int), 0, resolution - 1)
    behind_notch = (
        notch[iy, ix]
        & (face_depth < panel_front - 0.05 * panel_thickness)
    )
    candidate_faces = behind_notch & (np.abs(facing) > 0.65)
    if not candidate_faces.any():
        return mesh, metadata

    candidate_mask = Image.new("1", (resolution, resolution))
    draw = ImageDraw.Draw(candidate_mask)
    for face in faces[candidate_faces]:
        draw.polygon(pixels(vertices[face]), fill=1)
    candidate_pixels = np.asarray(candidate_mask, dtype=bool)
    intersection = int(np.count_nonzero(candidate_pixels & notch))
    union = int(np.count_nonzero(candidate_pixels | notch))
    shape_iou = intersection / union if union else 0.0

    front_depths = face_depth[candidate_faces & (facing > 0.65)]
    back_depths = face_depth[candidate_faces & (facing < -0.65)]
    candidate_thickness = (
        abs(float(np.median(front_depths) - np.median(back_depths)))
        if len(front_depths) and len(back_depths) else 0.0
    )
    thickness_ratio = candidate_thickness / panel_thickness
    metadata.update({
        "shape_iou": shape_iou,
        "candidate_thickness": candidate_thickness,
        "thickness_ratio": thickness_ratio,
    })
    if (
        shape_iou < NOTCH_SHAPE_IOU_THRESHOLD
        or thickness_ratio >= NOTCH_THICKNESS_RATIO_THRESHOLD
    ):
        return mesh, metadata

    cleaned = mesh.copy()
    keep = ~behind_notch
    cleaned.update_faces(keep)
    cleaned.remove_unreferenced_vertices()
    metadata.update({
        "detected": True,
        "removed_faces": int(np.count_nonzero(behind_notch)),
    })
    return cleaned, metadata


def _clean_texture_uv_bounds(
        material, source_uv: np.ndarray, source_faces: np.ndarray,
) -> tuple[float, float, float, float]:
    from PIL import Image, ImageDraw

    image = getattr(material, "baseColorTexture", None)
    if image is None:
        return 0.0, 1.0, 0.0, 1.0
    sample_size, stride = 256, 4
    pixels = np.asarray(
        image.convert("RGB").resize((sample_size, sample_size), Image.Resampling.BILINEAR),
        dtype=float,
    )
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722])
    mask_image = Image.new("L", (sample_size, sample_size))
    mask_draw = ImageDraw.Draw(mask_image)
    for face in source_faces:
        mask_draw.polygon([
            (
                float(source_uv[index, 0] * (sample_size - 1)),
                float((1.0 - source_uv[index, 1]) * (sample_size - 1)),
            )
            for index in face
        ], fill=255)
    mask = np.asarray(mask_image) > 0
    best = None
    for window in (32, 24, 16, 12, 8):
        for row in range(0, sample_size - window + 1, stride):
            for column in range(0, sample_size - window + 1, stride):
                patch_mask = mask[row:row + window, column:column + window]
                coverage = float(np.mean(patch_mask))
                if coverage < 0.85:
                    continue
                patch = luminance[row:row + window, column:column + window][patch_mask]
                dark_fraction = float(np.mean(patch < 45))
                score = (
                    50.0 * coverage
                    + float(np.percentile(patch, 5))
                    - 150.0 * dark_fraction
                    + 0.15 * float(np.std(patch))
                )
                if best is None or score > best[0]:
                    best = score, row, column, window
        if best is not None:
            break
    if best is None:
        raise ValueError("drawer texture UV islands contain no usable wood patch")
    _, row, column, window = best
    padding = 1
    u_min = (column + padding) / sample_size
    u_max = (column + window - padding) / sample_size
    v_min = 1.0 - (row + window - padding) / sample_size
    v_max = 1.0 - (row + padding) / sample_size
    return u_min, u_max, v_min, v_max


def _cabinet_collision(parent: trimesh.Trimesh, axis: np.ndarray, allowance: float
                       ) -> trimesh.Trimesh:
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    side = np.cross(axis, seed)
    side /= np.linalg.norm(side)
    up = np.cross(axis, side)
    lo, hi = _span(parent, axis)
    side_lo, side_hi = _span(parent, side)
    up_lo, up_hi = _span(parent, up)
    depth, width, height = hi - lo, side_hi - side_lo, up_hi - up_lo
    wall = max(0.02 * min(width, height), 1e-6)
    rear = max(allowance, 0.02 * depth, 1e-6)

    center_depth = (lo + hi) / 2
    center_side = (side_lo + side_hi) / 2
    center_up = (up_lo + up_hi) / 2
    return trimesh.util.concatenate([
        _box(axis, side, up, (rear, width, height),
             (lo + rear / 2, center_side, center_up)),
        _box(axis, side, up, (depth, wall, height),
             (center_depth, side_lo + wall / 2, center_up)),
        _box(axis, side, up, (depth, wall, height),
             (center_depth, side_hi - wall / 2, center_up)),
        _box(axis, side, up, (depth, max(width - 2 * wall, wall), wall),
             (center_depth, center_side, up_lo + wall / 2)),
        _box(axis, side, up, (depth, max(width - 2 * wall, wall), wall),
             (center_depth, center_side, up_hi - wall / 2)),
    ])


def _procedural_drawer(
        front_mesh: trimesh.Trimesh, axis: np.ndarray, closure_plane: float,
        depth: float, wall_top_limit: float | None = None,
) -> tuple[trimesh.Trimesh, float, dict, dict]:
    side, up = _drawer_frame(front_mesh, axis)
    front_mesh, notch_cleanup = _remove_thin_notch_membrane(
        front_mesh, axis, side, up,
    )
    side_lo, side_hi = _span(front_mesh, side)
    up_lo, up_hi = _span(front_mesh, up)
    width, original_height = side_hi - side_lo, up_hi - up_lo
    if min(depth, width, original_height) <= 0:
        raise ValueError("procedural drawer requires positive depth, width, and height")
    wall = min(
        max(DRAWER_WALL_WIDTH_RATIO * width, 1e-5),
        0.15 * min(width, original_height),
    )
    height = original_height
    height_state = {
        "original_wall_height": float(original_height),
        "final_wall_height": float(original_height),
        "height_clamped": False,
    }
    if wall_top_limit is not None:
        epsilon = max(1e-6, TOP_DRAWER_HEIGHT_EPSILON_RATIO * original_height)
        clearance = TOP_DRAWER_WALL_CLEARANCE_RATIO * original_height
        final_top = up_hi
        if up_hi > wall_top_limit + epsilon:
            final_top = max(up_lo + wall, wall_top_limit - clearance)
            height = final_top - up_lo
        height_state.update({
            "original_wall_top": float(up_hi),
            "final_wall_top": float(final_top),
            "wall_clearance_ratio": TOP_DRAWER_WALL_CLEARANCE_RATIO,
            "wall_clearance": float(clearance),
            "final_wall_height": float(height),
            "height_clamped": bool(height < original_height - epsilon),
        })
    center_axis = closure_plane - depth / 2
    center_side = (side_lo + side_hi) / 2
    center_up = up_lo + height / 2
    shell = trimesh.util.concatenate([
        _box(
            axis, side, up, (depth, wall, height),
            (center_axis, side_lo + wall / 2, center_up),
        ),
        _box(
            axis, side, up, (depth, wall, height),
            (center_axis, side_hi - wall / 2, center_up),
        ),
        _box(
            axis, side, up, (depth, max(width - 2 * wall, wall), wall),
            (center_axis, center_side, up_lo + wall / 2),
        ),
        _box(
            axis, side, up, (wall, width, height),
            (closure_plane - depth + wall / 2, center_side, center_up),
        ),
    ])
    source_uv = getattr(front_mesh.visual, "uv", None)
    source_material = getattr(front_mesh.visual, "material", None)
    if source_uv is not None and source_material is not None:
        shell.unmerge_vertices()
        basis = np.column_stack((axis, side, up))
        coordinates = np.asarray(shell.vertices) @ basis
        normalized = (
            (coordinates - coordinates.min(axis=0))
            / np.maximum(np.ptp(coordinates, axis=0), 1e-9)
        )
        uv = np.zeros((len(shell.vertices), 2))
        projection_axes = ((1, 2), (0, 2), (0, 1))
        dominant = np.argmax(np.abs(np.asarray(shell.face_normals) @ basis), axis=1)
        for face, normal_axis in zip(shell.faces, dominant):
            uv[face] = normalized[face][:, projection_axes[normal_axis]]
        u_min, u_max, v_min, v_max = _clean_texture_uv_bounds(
            source_material, np.asarray(source_uv), np.asarray(front_mesh.faces),
        )
        uv[:, 0] = u_min + uv[:, 0] * (u_max - u_min)
        uv[:, 1] = v_min + uv[:, 1] * (v_max - v_min)
        shell.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=source_material,
        )
    return (
        trimesh.util.concatenate([front_mesh, shell]), wall,
        notch_cleanup, height_state,
    )


def repair_prismatic_geometry(
        spec: KinematicSpec, links: dict[int, LinkGeometry],
        axes: Mapping[int, np.ndarray], selections: Mapping[int, Mapping],
        output_dir: Path,
        mode: str = "auto-3d", reuse_existing: bool = False,
) -> dict[int, dict]:
    if mode not in {"reconstructed-open", "procedural-closed", "auto-3d"}:
        raise ValueError(f"unknown drawer geometry mode: {mode}")
    state_path = output_dir / "prismatic_geometry.json"
    if mode in {"procedural-closed", "auto-3d"} and reuse_existing and state_path.is_file():
        payload = read_json(state_path)
        if payload.get("mode") == mode:
            return {int(child): state for child, state in payload["joints"].items()}
    top_drawers = {}
    for group in spec.drawer_groups:
        children = [child for child in group.children
                    if child in links and child in axes and child in selections]
        if not children:
            continue
        reference = children[0]
        reference_axis = np.asarray(axes[reference], dtype=float).copy()
        reference_axis *= int(selections[reference]["positive_direction"])
        reference_axis /= np.linalg.norm(reference_axis)
        _, up = _drawer_frame(links[reference].visual_mesh, reference_axis)
        top_drawers[group.parent] = max(
            children,
            key=lambda child: sum(_span(links[child].visual_mesh, up)) / 2,
        )
    states: dict[int, dict] = {}
    parent_proxies: dict[int, tuple[np.ndarray, float]] = {}
    for joint in spec.joints:
        if joint.type != "prismatic" or joint.child not in selections:
            continue
        selection = selections[joint.child]
        axis = np.asarray(axes[joint.child], dtype=float).copy()
        axis *= int(selection["positive_direction"])
        axis /= np.linalg.norm(axis)
        parent, child = links[joint.parent], links[joint.child]
        back, front = _span(parent.visual_mesh, axis)
        child_back, child_front = _span(child.visual_mesh, axis)
        depth = front - back
        projected_q = child_front - front
        resolved_mode = mode
        if mode == "auto-3d":
            resolved_mode = (
                "reconstructed-open"
                if depth > 0 and projected_q > AUTO_OPEN_RATIO * depth
                else "procedural-closed"
            )
        mesh_q = 0.0 if resolved_mode == "procedural-closed" else (
            projected_q if 0 <= projected_q <= depth
            else float(selection["mesh_current_q_ratio"]) * depth
        )
        allowance = float(selection["depth_allowance_ratio"]) * depth
        target_hidden = max(depth - mesh_q - allowance, 0.0)
        current_hidden = front - child_back
        scene_state = (
            {"scene_current_q": mesh_q} if mode == "auto-3d"
            else {"scene_current_q": 0.0}
            if resolved_mode == "procedural-closed" else {}
        )
        status = "skipped"
        wall = None
        notch_cleanup = None
        wall_height_state = {}
        top_sample_state = {"is_top_drawer": top_drawers.get(joint.parent) == joint.child}
        if resolved_mode == "procedural-closed" and target_hidden > 0:
            wall_top_limit = None
            if top_sample_state["is_top_drawer"]:
                _, up = _drawer_frame(child.visual_mesh, axis)
                child_up_lo, child_up_hi = _span(child.visual_mesh, up)
                wall_top_limit, sampling = _sample_cabinet_top(
                    parent.visual_mesh, axis, up,
                    (child_up_lo + child_up_hi) / 2,
                )
                top_sample_state.update(sampling)
            visual, wall, notch_cleanup, wall_height_state = _procedural_drawer(
                child.visual_mesh, axis, front, target_hidden, wall_top_limit,
            )
            links[joint.child] = LinkGeometry(
                child.id, child.primitive_ids, visual, visual.copy(),
                child.visual_path, child.collision_path,
            )
            status = "procedural"
        elif depth > 0 and current_hidden > 0.02 * depth and target_hidden > 0.02 * depth:
            vertices = np.asarray(child.visual_mesh.vertices, dtype=float).copy()
            projected = vertices @ axis
            hidden = projected <= front
            scale = target_hidden / current_hidden
            vertices[hidden] += np.outer(
                (front - projected[hidden]) * (1.0 - scale), axis
            )
            visual = child.visual_mesh.copy()
            visual.vertices = vertices
            collision = visual.copy()
            links[joint.child] = LinkGeometry(
                child.id, child.primitive_ids, visual, collision,
                child.visual_path, child.collision_path,
            )
            status = "repaired"
        states[joint.child] = {
            "child": joint.child,
            "parent": joint.parent,
            "axis": axis.tolist(),
            "cabinet_depth": depth,
            "depth_allowance": allowance,
            "mesh_current_q": mesh_q,
            **scene_state,
            "mesh_current_q_source": (
                "closed_primary" if resolved_mode == "procedural-closed"
                else "mesh_projection" if mesh_q == projected_q else "mllm_ratio"),
            "resolved_mode": resolved_mode,
            "current_hidden_length": current_hidden,
            "target_hidden_length": target_hidden,
            **({"wall_thickness": wall} if wall is not None else {}),
            **wall_height_state,
            **top_sample_state,
            **({"notch_cleanup": notch_cleanup} if notch_cleanup is not None else {}),
            "status": status,
            "confidence": float(selection["confidence"]),
        }
        previous = parent_proxies.get(joint.parent)
        if previous is None or allowance > previous[1]:
            parent_proxies[joint.parent] = (axis, allowance)

    for parent_id, (axis, allowance) in parent_proxies.items():
        parent = links[parent_id]
        links[parent_id] = LinkGeometry(
            parent.id, parent.primitive_ids, parent.visual_mesh,
            _cabinet_collision(parent.visual_mesh, axis, allowance),
            parent.visual_path, parent.collision_path,
        )
    save_link_meshes(links, output_dir / "parts")
    write_json(state_path, {
        "mode": mode,
        "joints": {str(child): state for child, state in states.items()}
    })
    return states
