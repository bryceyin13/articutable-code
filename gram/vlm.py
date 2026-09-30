from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import trimesh

from .artifacts import KinematicSpec, read_json, write_json
from .joints import JointEstimate
from .render import primitive_color

from pipeline.mllm import gram_runtime, run_mllm

VLM_EXECUTION_MODES = frozenset({"reasoning-only"})
VIEW_NAMES = ("front", "right", "back", "left", "top")
REVOLUTE_AXIS_MAX_CORRECTION_DEGREES = 10.0
REVOLUTE_AXIS_DIRECTION_EVIDENCE = frozenset({
    "motion_plane", "mechanical_centerline", "cross_joint_consistency",
})
JOINT_COORDINATE_CONVENTION = {
    "parent_is_fixed": True,
    "moving_link": "child",
    "q_is_full_child_relative_to_parent_motion": True,
    "world_or_camera_orientation_is_not_q": True,
}
WORKING_TO_Z_UP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def model_runtime(model: str | None = None,
                  execution_mode: str | None = None):
    runtime = gram_runtime().with_model(model).with_execution_mode(
        execution_mode)
    _validate_execution_mode(runtime.execution_mode)
    return runtime


def model_runtime_metadata(model: str | None = None,
                           execution_mode: str | None = None) -> dict:
    return model_runtime(model, execution_mode).metadata()


def _validate_execution_mode(execution_mode: str) -> str:
    if execution_mode not in VLM_EXECUTION_MODES:
        raise ValueError(f"unsupported VLM execution mode: {execution_mode}")
    return execution_mode


def _to_z_up(values) -> np.ndarray:
    return np.asarray(values, dtype=float) @ WORKING_TO_Z_UP.T


def _to_working(values) -> np.ndarray:
    return np.asarray(values, dtype=float) @ WORKING_TO_Z_UP


def _bounds_to_z_up(bounds: np.ndarray) -> dict[str, list[float]]:
    corners = _to_z_up(trimesh.bounds.corners(np.asarray(bounds, dtype=float)))
    return {"min": corners.min(axis=0).tolist(), "max": corners.max(axis=0).tolist()}


def _point_summary(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("mesh evidence requires non-empty 3D vertices")
    center = points.mean(axis=0)
    covariance = (points - center).T @ (points - center) / len(points)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order].T
    for axis in axes:
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis *= -1
    rounded = lambda value: np.round(np.asarray(value, dtype=float), 6).tolist()
    return {
        "bounds": {"min": rounded(points.min(axis=0)), "max": rounded(points.max(axis=0))},
        "centroid": rounded(center),
        "principal_axes": rounded(axes),
        "principal_variances": rounded(values[order]),
    }


def _mesh_geometry_summary(path: Path) -> dict:
    """Return fixed numeric evidence so the model never needs to open a mesh."""
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"mesh evidence must load as a scene: {path}")
    objects, aggregate = [], []
    for node in sorted(loaded.graph.nodes_geometry):
        transform, geometry_name = loaded.graph[node]
        geometry = loaded.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.vertices):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        points = _to_z_up(mesh.vertices)
        aggregate.append(points)
        objects.append({
            "name": str(node),
            "geometry": str(geometry_name),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "surface_area": round(float(mesh.area), 6),
            **_point_summary(points),
        })
    if not objects:
        raise ValueError(f"mesh evidence contains no geometry: {path}")
    return {
        "coordinate_frame": "normalized Z-up visual frame",
        "object_count": len(objects),
        "aggregate": _point_summary(np.concatenate(aggregate)),
        "objects": objects,
    }


def _relative_file(path: Path | None, root: Path) -> str | None:
    return os.path.relpath(path.resolve(), root.resolve()) if path else None


def _camera_paired_views(views_manifest: Path, paired_views: Sequence[Mapping]) -> list[dict]:
    views = read_json(views_manifest).get("views")
    if not isinstance(views, dict) or any(name not in views for name in VIEW_NAMES):
        raise ValueError("views.json must contain front/right/back/left/top camera data")
    result = []
    for name, pair in zip(VIEW_NAMES, paired_views, strict=True):
        item = views[name]
        matrix = np.asarray(item.get("camera_matrix"), dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"views.json {name} camera_matrix must be a finite 4x4 matrix")
        result.append({
            **pair, "name": name, "camera_matrix": matrix.tolist(),
            "tag_positions": item.get("tag_positions", {}),
            "visible_primitive_ids": item.get("visible_primitive_ids", []),
        })
    return result


def _revolute_joint_locator(
        rgb_view_paths: Sequence[Path], views_manifest: Path,
        estimates: Mapping[int, JointEstimate], spec: KinematicSpec,
        output_path: Path, axis_half_length: float) -> Path:
    """Compose the five calibrated RGB views with joint axes into one image."""
    from PIL import Image, ImageDraw, ImageFont

    if len(rgb_view_paths) != len(VIEW_NAMES):
        raise ValueError("revolute locator requires five RGB views")
    if not math.isfinite(axis_half_length) or axis_half_length <= 0:
        raise ValueError("revolute locator axis length must be positive")
    views = read_json(views_manifest).get("views")
    if not isinstance(views, dict) or any(name not in views for name in VIEW_NAMES):
        raise ValueError("revolute locator requires front/right/back/left/top camera data")
    joints = {joint.child: joint for joint in spec.joints}
    panel_size = 512
    canvas = Image.new("RGB", (3 * panel_size, 2 * panel_size), (32, 32, 32))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def project(point, direction, inverse, ortho_scale):
        camera_point = inverse @ np.r_[point, 1.0]
        camera_direction = inverse[:3, :3] @ direction
        center = np.array([
            (camera_point[0] / ortho_scale + 0.5) * panel_size,
            (0.5 - camera_point[1] / ortho_scale) * panel_size,
        ])
        delta = np.array([camera_direction[0], -camera_direction[1]])
        return center, delta

    for index, (name, path) in enumerate(zip(VIEW_NAMES, rgb_view_paths, strict=True)):
        with Image.open(path) as source:
            panel = source.convert("RGB").resize((panel_size, panel_size))
        draw = ImageDraw.Draw(panel)
        item = views[name]
        camera_matrix = np.asarray(item.get("camera_matrix"), dtype=float)
        if camera_matrix.shape != (4, 4) or not np.isfinite(camera_matrix).all():
            raise ValueError(f"views.json {name} camera_matrix must be a finite 4x4 matrix")
        try:
            inverse = np.linalg.inv(camera_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"views.json {name} camera_matrix must be invertible") from exc
        projection = item.get("projection") or {}
        ortho_scale = float(projection.get("ortho_scale", 1.35))
        if not math.isfinite(ortho_scale) or ortho_scale <= 0:
            raise ValueError(f"views.json {name} orthographic scale must be positive")
        draw.rectangle((0, 0, 116, 32), fill=(0, 0, 0))
        draw.text((8, 4), name.upper(), font=font, fill=(255, 255, 255))
        for child, estimate in sorted(estimates.items()):
            color = primitive_color(child % 256)
            center, delta = project(
                _to_z_up(estimate.pivot), _to_z_up(estimate.axis), inverse, ortho_scale)
            delta *= axis_half_length * panel_size / ortho_scale
            length = float(np.linalg.norm(delta))
            if length > 2.0:
                draw.line((*tuple(center - delta), *tuple(center + delta)), fill=color, width=5)
            x, y = map(float, center)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(0, 0, 0), width=2)
            label = f"J{child}"
            box = draw.textbbox((x + 10, y - 12), label, font=font)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((x + 10, y - 12), label, font=font, fill=color)
        canvas.paste(panel, ((index % 3) * panel_size, (index // 3) * panel_size))

    legend = ImageDraw.Draw(canvas)
    x, y = 2 * panel_size + 24, panel_size + 24
    legend.text((x, y), "JOINTS", font=font, fill=(255, 255, 255))
    for row, child in enumerate(sorted(estimates), start=1):
        color = primitive_color(child % 256)
        joint = joints[child]
        subtype = f"/{joint.subtype}" if joint.subtype else ""
        legend.text((x, y + row * 32), f"J{child}: {joint.type}{subtype}", font=font, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _revolute_lower_pose(
        spec: KinematicSpec, estimates: Mapping[int, JointEstimate],
        parent_mesh_paths: Mapping[int, Path], child_mesh_paths: Mapping[int, Path],
        hinge_states: Mapping[int, Mapping], spin_states: Mapping[int, Mapping],
        output_path: Path) -> Path:
    """Render all revolute joints at their deterministic canonical lower pose."""
    from .evidence_rendering import _mesh_preview

    link_paths = {}
    for joint in spec.joints:
        for link_id, path in (
                (joint.parent, parent_mesh_paths[joint.child]),
                (joint.child, child_mesh_paths[joint.child])):
            resolved = Path(path).resolve()
            if link_id in link_paths and link_paths[link_id] != resolved:
                raise ValueError(f"link {link_id} has inconsistent visual mesh paths")
            link_paths[link_id] = resolved

    states = {**hinge_states, **spin_states}
    children = {joint.child for joint in spec.joints}
    transforms = {
        link.id: np.eye(4) for link in spec.links if link.id not in children
    }
    pending = list(spec.joints)
    while pending:
        remaining = []
        for joint in pending:
            if joint.parent not in transforms:
                remaining.append(joint)
                continue
            relative = np.eye(4)
            if joint.child in states:
                state = states[joint.child]
                angle = float(state.get("closing_rotation", -float(state["mesh_current_q"])))
                if not math.isfinite(angle):
                    raise ValueError(f"joint {joint.child} closing rotation must be finite")
                estimate = estimates[joint.child]
                relative = trimesh.transformations.rotation_matrix(
                    angle, estimate.axis, estimate.pivot)
            transforms[joint.child] = transforms[joint.parent] @ relative
        if len(remaining) == len(pending):
            raise ValueError("kinematic tree contains an unreachable link")
        pending = remaining

    meshes, labels = [], []
    for link in sorted(spec.links, key=lambda item: item.id):
        mesh = trimesh.load(link_paths[link.id], force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"lower-pose visualization requires a non-empty link {link.id}")
        mesh.apply_transform(transforms[link.id])
        meshes.append(mesh)
        labels.append(np.full(len(mesh.faces), link.id, dtype=int))
    combined = trimesh.util.concatenate(meshes)
    legend = [
        (primitive_color(link.id % 256), f"L{link.id} {link.name}")
        for link in sorted(spec.links, key=lambda item: item.id)
    ] + [
        (None, f"J{joint.child}: L{joint.parent}->L{joint.child} q=0")
        for joint in sorted(spec.joints, key=lambda item: item.child)
        if joint.child in states
    ]
    return _mesh_preview(
        output_path, "Revolute canonical lower pose (q=0)", combined,
        np.concatenate(labels), legend)


def _file_evidence(evidence: dict, output_dir: Path, source_mesh: Path | None,
                   primitive_mesh: Path | None, views_manifest: Path,
                   execution_mode: str = "reasoning-only") -> dict:
    evidence = dict(evidence)
    if execution_mode == "agent":
        evidence["source_mesh"] = _relative_file(source_mesh, output_dir)
        evidence["primitive_mesh"] = _relative_file(primitive_mesh, output_dir)
    evidence["paired_views"] = _camera_paired_views(
        views_manifest, evidence["paired_views"])
    evidence["path_base"] = "."
    return evidence


def extract_json(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise ValueError("VLM response contains no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    return payload


JOINT_SELECTION_POLICIES = {
    "conservative": (
        "Return at least one joint: select the candidate with the strongest consistent geometric "
        "and mechanical evidence even when every candidate is imperfect. Beyond that single "
        "most-supported joint, include another joint only when its own rigid grouping, interface, "
        "and motion type are clear from consistent multi-view and mesh evidence. Do not force "
        "additional uncertain joints to complete a plausible mechanism. Assign primitives from "
        "rejected or unresolved candidates to the rigid link most strongly supported by continuity "
        "and attachment evidence so every primitive is still assigned exactly once."
    ),
    "high-recall": (
        "Return at least one joint and include every additional joint candidate supported by "
        "affirmative observable geometric or mechanical evidence. Do not restrict the result to "
        "only the most certain candidates. A candidate may be included when a separable rigid "
        "component and a localized movable interface are consistently supported, even if its exact "
        "partition or motion type is not perfectly certain; choose the best-supported valid "
        "partition and allowed motion type. Still reject candidates supported only by semantic "
        "expectation, appearance, or a desire to complete a plausible mechanism. Assign primitives "
        "from rejected candidates to the rigid link most strongly supported by continuity and "
        "attachment evidence so every primitive is still assigned exactly once."
    ),
    "high-recall-v2": (
        "Return at least one joint and optimize for recall among geometrically defensible "
        "candidates. Include every additional candidate when a separately traceable rigid "
        "component has at least one affirmative boundary, seam, nesting, or localized-interface "
        "cue and no stronger nonlocal continuity evidence shows that it is rigidly attached. Do "
        "not require the same cue to be visible in every view: occlusion, a closed rest pose, small "
        "displacement, or small component size is absence of evidence rather than contradictory "
        "evidence. When including the component and merging it into its parent are both plausible, "
        "include it unless positive geometric evidence supports the rigid merge. Evaluate repeated "
        "components individually and apply the same decision to geometrically equivalent instances "
        "with equivalent interfaces; do not impose an implicit joint-count cap. Choose the "
        "best-supported valid partition and allowed motion type when those details remain uncertain. "
        "Still reject candidates supported only by semantic expectation, appearance, or a desire to "
        "complete a plausible mechanism. Assign primitives from rejected candidates to the rigid "
        "link most strongly supported by continuity and attachment evidence so every primitive is "
        "still assigned exactly once."
    ),
}


def build_prompt(primitive_ids: set[int], evidence: Mapping | None = None,
                 joint_selection_mode: str = "conservative",
                 execution_mode: str = "reasoning-only") -> str:
    _validate_execution_mode(execution_mode)
    template = (Path(__file__).parents[1] / "prompts" / "13_gram_primitive_kinematics.txt").read_text(encoding="utf-8")
    try:
        policy = JOINT_SELECTION_POLICIES[joint_selection_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported joint selection mode: {joint_selection_mode}") from exc
    template = template.replace("{{JOINT_SELECTION_POLICY}}", policy)
    prompt = f"{template}\n\nThe complete primitive ID set is {sorted(primitive_ids)}."
    if evidence:
        inspection = (
            "Inspect both GLBs with read-only tools: source_mesh is the original textured mesh, "
            "and primitive_mesh contains separate objects named primitive_0, primitive_1, etc. "
            "Inspect every primitive object before grouping; do not rely only on visible labels."
            if execution_mode == "agent" else
            "Use the precomputed source_mesh_geometry and primitive_mesh_geometry summaries. "
            "They contain every mesh object in the normalized Z-up visual frame. No file access "
            "or external tools are available; reason only from the attached images and supplied data."
        )
        prompt += (
            "\n\nUse all evidence below. The source image shows the semantic appearance before "
            "image-to-3D. Each RGB render and tagged primitive render uses the same camera. "
            f"{inspection}\n" + json.dumps(evidence, indent=2)
        )
    return prompt


def build_persistent_spin_prompt(spec: KinematicSpec, evidence: Mapping) -> str:
    template = (Path(__file__).parents[1] / "prompts" / "14_gram_persistent_spin.txt").read_text(
        encoding="utf-8")
    children = {link.id: [] for link in spec.links}
    for joint in spec.joints:
        children[joint.parent].append(joint.child)

    def subtree(root: int) -> list[int]:
        result, stack = [], [root]
        while stack:
            node = stack.pop(); result.append(node); stack.extend(children[node])
        return sorted(result)

    candidates = [{
        "parent": joint.parent, "child": joint.child,
        "child_subtree": subtree(joint.child),
    } for joint in spec.joints
        if joint.type == "revolute" and joint.subtype == "spin"]
    request = {
        "validated_kinematics": spec.to_dict(),
        "spin_candidates": candidates,
        "evidence": evidence,
    }
    return template + "\n\nUse the validated data below.\n" + json.dumps(request, indent=2)


def _run_mllm(prompt: str, view_paths: Sequence[Path], response_path: Path,
               model: str | None, log_prefix: str, cwd: Path, runner,
               execution_mode: str | None = None) -> str:
    return run_mllm(
        model_runtime(model, execution_mode), prompt, view_paths, response_path,
        cwd,
        stdout_path=response_path.parent / f"{log_prefix}_stdout.txt",
        stderr_path=response_path.parent / f"{log_prefix}_stderr.txt",
    )


def _multimodal_evidence(
        view_paths: Sequence[Path], primitive_ids: set[int], source_image: Path | None,
        rgb_view_paths: Sequence[Path], source_mesh: Path | None,
        primitive_mesh: Path | None, primary_image: Path | None = None,
        redact_paths: bool = False,
        execution_mode: str = "reasoning-only",
        ) -> tuple[dict, list[Path]]:
    _validate_execution_mode(execution_mode)
    redact_paths = redact_paths or execution_mode == "reasoning-only"
    if rgb_view_paths and len(rgb_view_paths) != len(view_paths):
        raise ValueError("RGB and tagged view counts must match")
    optional_paths = [
        path for path in (source_image, primary_image, source_mesh, primitive_mesh)
        if path is not None
    ]
    if any(not path.is_file() for path in (*rgb_view_paths, *optional_paths)):
        raise FileNotFoundError("source image, RGB views, and mesh evidence must exist when provided")
    pairs = ([
        {"rgb": f"attached_rgb_view_{index}",
         "primitive_ids": f"attached_tagged_view_{index}"}
        for index, _ in enumerate(view_paths)
    ] if redact_paths else [
        {"rgb": str(rgb.resolve()), "primitive_ids": str(tagged.resolve())}
        for rgb, tagged in zip(rgb_view_paths, view_paths)
    ])
    reference = (lambda path, label: label if path else None) if redact_paths else (
        lambda path, _label: str(path.resolve()) if path else None)
    evidence = {
        "source_image": reference(source_image, "attached_source_image"),
        "primary_image": reference(primary_image, "attached_primary_image"),
        "paired_views": pairs,
        "source_mesh": reference(source_mesh, "attached_source_mesh"),
        "primitive_mesh": reference(primitive_mesh, "attached_primitive_mesh"),
        "primitive_objects": [f"primitive_{i}" for i in sorted(primitive_ids)],
    }
    if execution_mode == "reasoning-only":
        evidence.pop("source_mesh")
        evidence.pop("primitive_mesh")
        if source_mesh is not None:
            evidence["source_mesh_geometry"] = _mesh_geometry_summary(source_mesh)
        if primitive_mesh is not None:
            evidence["primitive_mesh_geometry"] = _mesh_geometry_summary(primitive_mesh)
    images = ([source_image] if source_image else []) + (
        [primary_image] if primary_image and primary_image != source_image else []) + (
        [path for pair in zip(rgb_view_paths, view_paths) for path in pair]
        if rgb_view_paths else list(view_paths))
    return evidence, images


def _axis_agent_evidence(
        view_paths: Sequence[Path], views_manifest: Path, primitive_ids: set[int],
        source_image: Path | None, rgb_view_paths: Sequence[Path],
        source_mesh: Path | None, primitive_mesh: Path | None,
        execution_mode: str = "reasoning-only",
        ) -> tuple[dict, list[Path]]:
    evidence, images = _multimodal_evidence(
        view_paths, primitive_ids, source_image, rgb_view_paths,
        source_mesh, primitive_mesh, execution_mode=execution_mode)
    if not rgb_view_paths:
        evidence["paired_views"] = [
            {
                "rgb": None,
                "primitive_ids": (
                    f"attached_tagged_view_{index}"
                    if execution_mode == "reasoning-only" else str(path.resolve())
                ),
            }
            for index, path in enumerate(view_paths)
        ]
    evidence["paired_views"] = _camera_paired_views(
        views_manifest, evidence["paired_views"])
    return {key: value for key, value in evidence.items() if value is not None}, images


def _generic_kinematics(spec: KinematicSpec) -> dict:
    """Prompt-only topology that cannot leak source object or asset names."""
    return {
        "links": [
            {"id": link.id, "primitive_ids": list(link.primitive_ids)}
            for link in spec.links
        ],
        "joints": [
            {"parent": joint.parent, "child": joint.child,
             "type": joint.type, "subtype": joint.subtype}
            for joint in spec.joints
        ],
    }


def _infer_persistent_spin_children(
        spec: KinematicSpec, primitive_ids: set[int], view_paths: Sequence[Path],
        rgb_view_paths: Sequence[Path], primary_image: Path | None,
        primitive_mesh: Path | None, model: str | None, output_dir: Path, runner,
        execution_mode: str = "reasoning-only") -> KinematicSpec:
    spin_children = {
        joint.child for joint in spec.joints
        if joint.type == "revolute" and joint.subtype == "spin"
    }
    if not spin_children:
        write_json(output_dir / "persistent_spin_selection.json", {
            "persistent_spin_children": [],
        })
        return spec
    if primary_image is None or not primary_image.is_file():
        raise FileNotFoundError(
            "persistent spin classification requires a primary image")
    evidence, images = _multimodal_evidence(
        view_paths, primitive_ids, None, rgb_view_paths, None, primitive_mesh,
        primary_image=primary_image, execution_mode=execution_mode)
    evidence = {key: value for key, value in evidence.items() if value is not None}
    prompt = build_persistent_spin_prompt(spec, evidence)
    write_json(output_dir / "persistent_spin_request.json", {
        **model_runtime_metadata(model, execution_mode),
        "view_paths": [str(path) for path in images], "evidence": evidence,
        "kinematics": spec.to_dict(), "prompt": prompt,
    })
    raw = _run_mllm(
        prompt, images, output_dir / "raw_persistent_spin_response.txt", model,
        "persistent_spin_mllm", Path(__file__).parents[1], runner,
        execution_mode)
    payload = extract_json(raw)
    if set(payload) != {"persistent_spin_children"}:
        raise ValueError(
            "persistent spin response must contain only persistent_spin_children")
    final = KinematicSpec.from_dict({
        **spec.to_dict(),
        "persistent_spin_children": payload["persistent_spin_children"],
    }, primitive_ids)
    write_json(output_dir / "persistent_spin_selection.json", {
        "persistent_spin_children": list(final.persistent_spin_children),
    })
    return final


def _primitive_adjacency_evidence(
        mesh_path: Path, face_ids_path: Path, primitive_ids: set[int]) -> list[dict]:
    if not mesh_path.is_file() or not face_ids_path.is_file():
        raise FileNotFoundError("working mesh and face primitive IDs are required")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    labels = np.asarray(np.load(face_ids_path))
    if labels.ndim != 1 or len(labels) != len(mesh.faces):
        raise ValueError("face primitive IDs must match working mesh faces")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("face primitive IDs must be integers")
    if set(map(int, np.unique(labels))) != primitive_ids:
        raise ValueError("face primitive IDs do not match the primitive manifest")
    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {}
    adjacency = np.asarray(mesh.face_adjacency, dtype=int)
    for faces, edge in zip(adjacency, np.asarray(mesh.face_adjacency_edges, dtype=int),
                           strict=True):
        pair = tuple(sorted(map(int, labels[faces])))
        if pair[0] != pair[1]:
            grouped.setdefault(pair, set()).add(tuple(sorted(map(int, edge))))

    def vector(value) -> list[float]:
        return np.round(np.asarray(value, dtype=float), 6).tolist()

    def summarize(edges: Sequence[tuple[int, int]], topology: bool = False) -> dict | None:
        edge_array = np.asarray(edges, dtype=int)
        coordinates = np.asarray(mesh.vertices[edge_array], dtype=float)
        lengths = np.linalg.norm(coordinates[:, 1] - coordinates[:, 0], axis=1)
        positive = lengths > np.finfo(float).eps
        edge_array, coordinates, lengths = (
            edge_array[positive], coordinates[positive], lengths[positive])
        if not len(edge_array):
            return None
        vertex_ids = np.unique(edge_array)
        points = np.asarray(mesh.vertices[vertex_ids], dtype=float)
        center = np.average(coordinates.mean(axis=1), axis=0, weights=lengths)
        covariance = np.cov(points - points.mean(axis=0), rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
        projection = points @ axis
        result = {
            "edge_count": int(len(edge_array)),
            "length": round(float(lengths.sum()), 6),
            "center": vector(center),
            "aabb_min": vector(points.min(axis=0)),
            "aabb_max": vector(points.max(axis=0)),
            "principal_axis": vector(axis),
            "principal_endpoints": [
                vector(points[int(np.argmin(projection))]),
                vector(points[int(np.argmax(projection))]),
            ],
        }
        if topology:
            degrees = {int(vertex): 0 for vertex in vertex_ids}
            for first, second in edge_array:
                degrees[int(first)] += 1
                degrees[int(second)] += 1
            result.update({
                "topological_endpoints": [
                    vector(mesh.vertices[vertex])
                    for vertex, degree in sorted(degrees.items()) if degree == 1
                ],
                "closed_loop": bool(degrees and all(degree == 2 for degree in degrees.values())),
            })
        return result

    result = []
    for pair, edge_set in sorted(grouped.items()):
        edges = sorted(edge_set)
        by_vertex: dict[int, list[int]] = {}
        for index, edge in enumerate(edges):
            for vertex in edge:
                by_vertex.setdefault(vertex, []).append(index)
        remaining = set(range(len(edges)))
        components = []
        while remaining:
            pending = [min(remaining)]
            component_indices = set()
            while pending:
                index = pending.pop()
                if index not in remaining:
                    continue
                remaining.remove(index)
                component_indices.add(index)
                for vertex in edges[index]:
                    pending.extend(by_vertex[vertex])
            summary = summarize(
                [edges[index] for index in sorted(component_indices)], topology=True)
            if summary is not None:
                components.append(summary)
        components.sort(key=lambda item: (-item["length"], item["center"]))
        aggregate = summarize(edges)
        if aggregate is None:
            continue
        kept = components[:4]
        result.append({
            "a": pair[0], "b": pair[1], "shared_edges": aggregate["edge_count"],
            "total_boundary_length": aggregate["length"],
            "boundary_center": aggregate["center"],
            "boundary_aabb_min": aggregate["aabb_min"],
            "boundary_aabb_max": aggregate["aabb_max"],
            "boundary_principal_axis": aggregate["principal_axis"],
            "boundary_principal_endpoints": aggregate["principal_endpoints"],
            "boundary_component_count": len(components),
            "boundary_components": kept,
            "omitted_boundary_component_count": len(components) - len(kept),
            "omitted_boundary_length": max(0.0, round(
                aggregate["length"] - sum(item["length"] for item in kept), 6)),
        })
    return result


def infer_kinematics(view_paths: Sequence[Path], primitive_ids: set[int], model: str | None,
                     output_dir: Path, source_image: Path | None = None,
                     rgb_view_paths: Sequence[Path] = (), source_mesh: Path | None = None,
                     primitive_mesh: Path | None = None, working_mesh: Path | None = None,
                     face_primitive_ids: Path | None = None,
                     primary_image: Path | None = None,
                     runner=subprocess.run,
                     joint_selection_mode: str = "conservative",
                     execution_mode: str = "reasoning-only") -> KinematicSpec:
    if not view_paths or any(not path.exists() for path in view_paths):
        raise FileNotFoundError("all tagged view images are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path(__file__).parents[1]
    evidence, images = _multimodal_evidence(
        view_paths, primitive_ids, source_image, rgb_view_paths, source_mesh,
        primitive_mesh, primary_image=primary_image,
        execution_mode=execution_mode)
    if (working_mesh is None) != (face_primitive_ids is None):
        raise ValueError("working mesh and face primitive IDs must be supplied together")
    if working_mesh is not None:
        evidence["primitive_adjacency"] = _primitive_adjacency_evidence(
            working_mesh, face_primitive_ids, primitive_ids)
    prompt = build_prompt(
        primitive_ids, evidence, joint_selection_mode, execution_mode)
    write_json(output_dir / "request.json", {
        **model_runtime_metadata(model, execution_mode),
        "joint_selection_mode": joint_selection_mode,
        "primitive_ids": sorted(primitive_ids),
        "view_paths": [str(path) for path in images], "evidence": evidence, "prompt": prompt,
    })
    raw = _run_mllm(
        prompt, images, output_dir / "raw_response.txt", model, "mllm", cwd,
        runner, execution_mode)
    try:
        topology = extract_json(raw); topology.pop("persistent_spin_children", None)
        spec = KinematicSpec.from_dict(topology, primitive_ids)
    except ValueError as exc:
        repair_prompt = (
            "Repair the JSON below. Return only corrected JSON; do not infer new semantics. "
            f"Validation error: {exc}. Required primitive IDs: {sorted(primitive_ids)}.\n{raw}"
        )
        repaired = _run_mllm(
            repair_prompt, (), output_dir / "raw_repair_response.txt", model,
            "mllm_repair", cwd, runner, execution_mode)
        topology = extract_json(repaired); topology.pop("persistent_spin_children", None)
        spec = KinematicSpec.from_dict(topology, primitive_ids)
    if len(spec.links) == 1 and len(primitive_ids) > 1:
        audit_prompt = (
            f"{prompt}\n\nAudit the valid kinematic JSON below for a missed separable moving "
            "assembly. Closed static components may still move when supported by geometric "
            "and mechanical evidence. Keep one link if the evidence is insufficient. Return "
            "only a complete JSON object in the same schema.\n"
            f"{json.dumps(spec.to_dict(), separators=(',', ':'))}"
        )
        try:
            audited = _run_mllm(
                audit_prompt, images, output_dir / "raw_audit_response.txt",
                model, "mllm_audit", cwd, runner, execution_mode)
            topology = extract_json(audited); topology.pop("persistent_spin_children", None)
            spec = KinematicSpec.from_dict(topology, primitive_ids)
        except (FileNotFoundError, RuntimeError, ValueError):
            pass
    spec = _infer_persistent_spin_children(
        spec, primitive_ids, view_paths, rgb_view_paths, primary_image,
        primitive_mesh, model, output_dir, runner, execution_mode)
    write_json(output_dir / "kinematics.json", spec.to_dict())
    return spec


def select_prismatic_axes(
        view_paths: Sequence[Path], views_manifest: Path, spec: KinematicSpec,
        candidates: Mapping[int, Mapping[str, Mapping]], mesh_paths: Mapping[int, Path],
        model: str | None, output_dir: Path, source_image: Path | None = None,
        rgb_view_paths: Sequence[Path] = (), source_mesh: Path | None = None,
        primitive_mesh: Path | None = None, runner=subprocess.run,
        execution_mode: str = "reasoning-only") -> dict[int, dict]:
    prismatic = {joint.child for joint in spec.joints if joint.type == "prismatic"}
    expected = set(candidates)
    if not expected <= prismatic or set(mesh_paths) != expected:
        raise ValueError("prismatic candidate and mesh sets must match prismatic joints")
    if not views_manifest.is_file() or any(not path.is_file() for path in view_paths):
        raise FileNotFoundError("prismatic axis selection requires the five views and views.json")
    if any(not path.is_file() for path in mesh_paths.values()):
        raise FileNotFoundError("prismatic child mesh is missing")

    primitive_ids = {primitive for link in spec.links for primitive in link.primitive_ids}
    evidence, images = _axis_agent_evidence(
        view_paths, views_manifest, primitive_ids, source_image, rgb_view_paths,
        source_mesh, primitive_mesh, execution_mode)
    model_candidates = {
        child: {
            name: {**candidate, "axis": _to_z_up(candidate["axis"]).tolist()}
            for name, candidate in candidates[child].items()
        }
        for child in expected
    }
    request = {
        "kinematics": spec.to_dict(),
        **({"views_manifest": str(views_manifest.resolve())}
           if execution_mode == "agent" else {}),
        "coordinate_frame": "normalized Z-up visual frame shared by candidate vectors and camera matrices",
        "mesh_coordinate_frame": (
            "mesh assets originate in normalized working Y-up, but Blender-imported "
            "world coordinates are already normalized Z-up; never transform Blender "
            "world coordinates again"
        ),
        "working_to_z_up": WORKING_TO_Z_UP.tolist(),
        "evidence": evidence,
        "joints": [{
            "child": child,
            **({"child_mesh": str(mesh_paths[child].resolve())}
               if execution_mode == "agent" else
               {"child_mesh_geometry": _mesh_geometry_summary(mesh_paths[child])}),
            "candidates": model_candidates[child],
        } for child in sorted(expected)],
    }
    inspection = (
        "Inspect source_mesh and every object in primitive_mesh to understand the complete object, "
        "then inspect every listed child_mesh in its relationship to the base using read-only tools. "
        "Mesh assets originate in normalized working Y-up, but inspection tools may convert them "
        "automatically. Blender-imported world coordinates are already normalized Z-up: do not apply "
        "working_to_z_up to Blender world coordinates. Apply working_to_z_up exactly once only when "
        "an inspection tool returns raw working Y-up coordinates."
        if execution_mode == "agent" else
        "Use the attached images and the precomputed source_mesh_geometry, "
        "primitive_mesh_geometry, and child_mesh_geometry summaries. All summary coordinates are "
        "already in normalized Z-up. No file access or external tools are available."
    )
    prompt = (
        "Select one exact PCA axis for every already-validated prismatic joint below. "
        "Use all supplied visual, mesh, and numeric evidence. The source image shows semantic appearance "
        "before image-to-3D, and each RGB render is camera-paired with its tagged primitive view. "
        f"{inspection} Compare all resulting Z-up coordinates "
        "with the Z-up candidate vectors and camera matrices. Infer the physical travel corridor from "
        "observable open/closed geometry and mechanical relationships, never from a category name "
        "alone. Select the PCA line most aligned with that travel corridor. Candidate names are "
        "scoped by child. pca_axis_0, pca_axis_1, and pca_axis_2 are ordered by increasing PCA "
        "variance only as a geometric property; do not assume the smallest or largest variance "
        "axis is the motion axis. Orient the selected line so positive motion goes from the "
        "semantic q=0 closed/retracted/rest state toward the open/extended/actuated state. "
        "All candidate vectors use the same Z-up visual frame as the camera matrices. "
        "positive_direction is 1 when the listed vector already points that way, otherwise -1. "
        "Do not change links, "
        "joint types, or numeric vectors. Return only JSON in this exact shape: "
        '{"selections":[{"child":1,"selected_axis":"pca_axis_0",'
        '"positive_direction":1,"confidence":0.9}]}. '
        "Include every listed child exactly once and choose only one of that child's names.\n\n"
        + json.dumps(request, indent=2)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "prismatic_axis_request.json", {
        **model_runtime_metadata(model, execution_mode),
        **request, "prompt": prompt,
    })
    raw = _run_mllm(
        prompt, images, output_dir / "raw_prismatic_axis_response.txt", model,
        "prismatic_axis_mllm", Path(__file__).parents[1], runner,
        execution_mode)
    payload = extract_json(raw)
    items = payload.get("selections")
    if not isinstance(items, list):
        raise ValueError("prismatic axis response must contain a selections list")
    normalized, selected = [], {}
    for item in items:
        try:
            child = int(item["child"])
            name = str(item["selected_axis"])
            direction = int(item["positive_direction"])
            confidence = float(item["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid prismatic axis selection") from exc
        if child not in expected or child in selected:
            raise ValueError(f"unexpected or duplicate prismatic child: {child}")
        if name not in candidates[child]:
            raise ValueError(f"unknown PCA axis for child {child}: {name}")
        if direction not in {-1, 1}:
            raise ValueError(f"invalid positive direction for child {child}")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence for child {child}")
        selected[child] = {
            "selected_axis": name,
            "positive_direction": direction,
            "confidence": confidence,
        }
        normalized.append({"child": child, **selected[child]})
    if set(selected) != expected:
        raise ValueError(f"missing prismatic children: {sorted(expected - set(selected))}")
    write_json(output_dir / "prismatic_axis_selection.json", {"selections": normalized})
    return selected


def select_drawer_cabinet_axes(
        view_paths: Sequence[Path], views_manifest: Path, spec: KinematicSpec,
        mesh_paths: Mapping[int, Path], model: str | None, output_dir: Path,
        source_image: Path | None = None, rgb_view_paths: Sequence[Path] = (),
        source_mesh: Path | None = None, primitive_mesh: Path | None = None,
        runner=subprocess.run, execution_mode: str = "reasoning-only") -> dict[int, dict]:
    expected = {child for group in spec.drawer_groups for child in group.children}
    required_meshes = expected | {group.parent for group in spec.drawer_groups}
    if not expected or set(mesh_paths) != required_meshes:
        raise ValueError("drawer mesh set must contain every grouped parent and child")
    if not views_manifest.is_file() or any(not path.is_file() for path in view_paths):
        raise FileNotFoundError("drawer axis selection requires the five views and views.json")
    if any(not path.is_file() for path in mesh_paths.values()):
        raise FileNotFoundError("drawer parent or child mesh is missing")

    primitive_ids = {primitive for link in spec.links for primitive in link.primitive_ids}
    evidence, images = _axis_agent_evidence(
        view_paths, views_manifest, primitive_ids, source_image, rgb_view_paths,
        source_mesh, primitive_mesh, execution_mode)
    groups = []
    for group in spec.drawer_groups:
        bounds = np.asarray(
            trimesh.load(mesh_paths[group.parent], force="mesh", process=False).bounds,
            dtype=float)
        groups.append({
            "parent": group.parent,
            "children": list(group.children),
            **({
                "parent_mesh": str(mesh_paths[group.parent].resolve()),
                "child_meshes": {
                    str(child): str(mesh_paths[child].resolve())
                    for child in group.children
                },
            } if execution_mode == "agent" else {
                "parent_mesh_geometry": _mesh_geometry_summary(
                    mesh_paths[group.parent]),
                "child_mesh_geometries": {
                    str(child): _mesh_geometry_summary(mesh_paths[child])
                    for child in group.children
                },
            }),
            "parent_aabb": _bounds_to_z_up(bounds),
            "candidate_axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        })
    request = {
        "kinematics": spec.to_dict(),
        **({"views_manifest": str(views_manifest.resolve())}
           if execution_mode == "agent" else {}),
        "coordinate_frame": "normalized Z-up visual frame shared by parent_aabb, candidate_axes, and camera matrices",
        "mesh_coordinate_frame": (
            "mesh assets originate in normalized working Y-up, but Blender-imported "
            "world coordinates are already normalized Z-up; never transform Blender "
            "world coordinates again"
        ),
        "working_to_z_up": WORKING_TO_Z_UP.tolist(),
        "evidence": evidence,
        "groups": groups,
    }
    inspection = (
        "Inspect the parent and every child mesh plus all image evidence using read-only tools. "
        "Mesh assets originate in normalized working Y-up, but inspection tools may convert them "
        "automatically. Blender-imported world coordinates are already normalized Z-up: do not apply "
        "working_to_z_up to Blender world coordinates. Apply working_to_z_up exactly once only when "
        "an inspection tool returns raw working Y-up coordinates."
        if execution_mode == "agent" else
        "Use all image evidence and the precomputed parent_mesh_geometry and "
        "child_mesh_geometries summaries. All summary coordinates are already in normalized Z-up. "
        "No file access or external tools are available."
    )
    prompt = (
        "For each already-validated drawer cabinet group, select the single exact axis-aligned "
        "normal of the cabinet opening/back plane. All child drawers in a group must share this "
        f"one axis; do not fit separate PCA axes to individual drawers. {inspection} "
        "Compare all resulting Z-up coordinates with the Z-up camera "
        "matrices, parent_aabb, and candidate_axes. Choose axis x, y, or z in the "
        "listed Z-up visual frame. "
        "positive_direction is +1 when that basis vector points from the cabinet interior outward "
        "through the drawer opening, otherwise -1. For each child estimate mesh_current_q_ratio as "
        "its visible extension in source_image divided by the parent's full AABB depth on the chosen "
        "axis. Estimate depth_allowance_ratio as rear-wall thickness plus minimal clearance divided "
        "by that depth. Ratios must be in [0,1] and sum to at most 1. Do not change topology or joint "
        "types. Return only JSON in this shape: "
        '{"groups":[{"parent":0,"axis":"z","positive_direction":1,"confidence":0.9,'
        '"children":[{"child":1,"mesh_current_q_ratio":0.25,'
        '"depth_allowance_ratio":0.05}]}]}. Include every group and child exactly once.\n\n'
        + json.dumps(request, indent=2)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "drawer_axis_request.json", {
        **model_runtime_metadata(model, execution_mode),
        **request, "prompt": prompt,
    })
    raw = _run_mllm(
        prompt, images, output_dir / "raw_drawer_axis_response.txt", model,
        "drawer_axis_mllm", Path(__file__).parents[1], runner,
        execution_mode)
    items = extract_json(raw).get("groups")
    if not isinstance(items, list):
        raise ValueError("drawer axis response must contain a groups list")

    expected_groups = {group.parent: set(group.children) for group in spec.drawer_groups}
    selected, normalized, seen_parents = {}, [], set()
    for item in items:
        try:
            parent = int(item["parent"])
            axis_name = str(item["axis"])
            direction = int(item["positive_direction"])
            confidence = float(item["confidence"])
            children = item["children"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid drawer axis selection") from exc
        if (parent not in expected_groups or parent in seen_parents
                or axis_name not in {"x", "y", "z"} or direction not in {-1, 1}
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or not isinstance(children, list)):
            raise ValueError(f"invalid or duplicate drawer group: {parent}")
        seen_parents.add(parent)
        model_axis_name = axis_name
        axis_name, frame_sign = {
            "x": ("x", 1),
            "y": ("z", -1),
            "z": ("y", 1),
        }[model_axis_name]
        direction *= frame_sign
        axis_index = "xyz".index(axis_name)
        parent_bounds = np.asarray(
            trimesh.load(mesh_paths[parent], force="mesh", process=False).bounds)
        child_bounds = [
            np.asarray(trimesh.load(
                mesh_paths[child], force="mesh", process=False).bounds)
            for child in expected_groups[parent]
        ]
        positive_extension = max(
            bounds[1, axis_index] - parent_bounds[1, axis_index]
            for bounds in child_bounds)
        negative_extension = max(
            parent_bounds[0, axis_index] - bounds[0, axis_index]
            for bounds in child_bounds)
        forward, reverse = (
            (positive_extension, negative_extension)
            if direction == 1 else (negative_extension, positive_extension)
        )
        depth = parent_bounds[1, axis_index] - parent_bounds[0, axis_index]
        if reverse > max(forward, 0.0) + 0.02 * depth:
            direction = -direction
        normalized_children, seen_children = [], set()
        for child_item in children:
            try:
                child = int(child_item["child"])
                mesh_ratio = float(child_item["mesh_current_q_ratio"])
                allowance_ratio = float(child_item["depth_allowance_ratio"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid drawer child selection") from exc
            if (child not in expected_groups[parent] or child in seen_children
                    or not np.isfinite([mesh_ratio, allowance_ratio]).all()
                    or not 0 <= mesh_ratio <= 1 or not 0 <= allowance_ratio <= 1
                    or mesh_ratio + allowance_ratio > 1):
                raise ValueError(f"invalid or duplicate drawer child: {child}")
            seen_children.add(child)
            selected[child] = {
                "axis": axis_name, "positive_direction": direction,
                "mesh_current_q_ratio": mesh_ratio,
                "depth_allowance_ratio": allowance_ratio, "confidence": confidence,
            }
            normalized_children.append({"child": child,
                                        "mesh_current_q_ratio": mesh_ratio,
                                        "depth_allowance_ratio": allowance_ratio})
        if seen_children != expected_groups[parent]:
            raise ValueError(f"missing drawer children for parent {parent}")
        normalized.append({
            "parent": parent, "axis": axis_name, "positive_direction": direction,
            "model_axis_z_up": model_axis_name,
            "confidence": confidence, "children": normalized_children,
        })
    if seen_parents != set(expected_groups):
        raise ValueError("missing drawer cabinet groups")
    write_json(output_dir / "drawer_axis_selection.json", {
        "coordinate_frame": "normalized working frame",
        "model_coordinate_frame": "normalized Z-up visual frame",
        "groups": normalized,
    })
    return selected


def correct_revolute_axes(
        view_paths: Sequence[Path], views_manifest: Path, spec: KinematicSpec,
        estimates: Mapping[int, JointEstimate], contacts: Mapping[int, np.ndarray],
        parent_mesh_paths: Mapping[int, Path], child_mesh_paths: Mapping[int, Path],
        diagonal: float, model: str | None, output_dir: Path,
        source_image: Path | None = None, rgb_view_paths: Sequence[Path] = (),
        source_mesh: Path | None = None, primitive_mesh: Path | None = None,
        runner=subprocess.run, execution_mode: str = "reasoning-only") -> dict[int, JointEstimate]:
    expected = set(estimates)
    if not expected or set(contacts) != expected or set(parent_mesh_paths) != expected or set(child_mesh_paths) != expected:
        raise ValueError("revolute estimate, contact, parent mesh, and child mesh sets must match")
    if any(estimate.type != "revolute" or estimate.subtype not in {"hinge", "spin"}
           for estimate in estimates.values()):
        raise ValueError("axis correction accepts only revolute hinge and spin estimates")
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("working mesh diagonal must be positive")
    if not views_manifest.is_file() or any(not path.is_file() for path in view_paths):
        raise FileNotFoundError("revolute correction requires the five views and views.json")
    if any(not path.is_file() for path in (*parent_mesh_paths.values(), *child_mesh_paths.values())):
        raise FileNotFoundError("revolute parent or child mesh is missing")

    primitive_ids = {primitive for link in spec.links for primitive in link.primitive_ids}
    evidence, images = _axis_agent_evidence(
        view_paths, views_manifest, primitive_ids, source_image, rgb_view_paths,
        source_mesh, primitive_mesh, execution_mode)
    parents = {joint.child: joint.parent for joint in spec.joints}
    request_joints = []
    for child in sorted(expected):
        estimate = estimates[child]
        center = np.asarray(contacts[child], dtype=float).mean(axis=0)
        request_joints.append({
            "parent": parents[child], "child": child, "subtype": estimate.subtype,
            **({
                "parent_mesh": str(parent_mesh_paths[child].resolve()),
                "child_mesh": str(child_mesh_paths[child].resolve()),
            } if execution_mode == "agent" else {
                "parent_mesh_geometry": _mesh_geometry_summary(
                    parent_mesh_paths[child]),
                "child_mesh_geometry": _mesh_geometry_summary(
                    child_mesh_paths[child]),
            }),
            "initial_axis_direction": _to_z_up(estimate.axis).tolist(),
            "initial_axis_point": _to_z_up(estimate.pivot).tolist(),
            "contact_center": _to_z_up(center).tolist(),
        })
    request = {
        "kinematics": spec.to_dict(),
        **({"views_manifest": str(views_manifest.resolve())}
           if execution_mode == "agent" else {}),
        "coordinate_frame": "normalized Z-up visual frame shared by all numeric 3-vectors and camera matrices",
        "mesh_coordinate_frame": (
            "mesh assets originate in normalized working Y-up, but Blender-imported "
            "world coordinates are already normalized Z-up; never transform Blender "
            "world coordinates again"
        ),
        "working_to_z_up": WORKING_TO_Z_UP.tolist(),
        "evidence": evidence,
        "joints": request_joints,
    }
    inspection = (
        "Inspect source_mesh, every object in primitive_mesh, and each listed parent_mesh and "
        "child_mesh with read-only tools before deciding. Mesh assets originate in the normalized "
        "working Y-up frame, but inspection tools may convert them automatically. Blender-imported "
        "world coordinates are already normalized Z-up: do not apply working_to_z_up to Blender "
        "world coordinates. Apply working_to_z_up exactly once only when an inspection tool returns "
        "raw working Y-up coordinates."
        if execution_mode == "agent" else
        "Use the attached images and the precomputed source_mesh_geometry, "
        "primitive_mesh_geometry, parent_mesh_geometry, and child_mesh_geometry summaries. All "
        "summary coordinates are already in normalized Z-up. No file access or external tools are available."
    )
    source_context = (
        "The source image and source_mesh are appearance evidence; never copy numeric coordinates "
        "directly from source_mesh."
        if execution_mode == "agent" else
        "The source image is appearance evidence; source_mesh_geometry is fixed numeric evidence."
    )
    numeric_geometry = "meshes" if execution_mode == "agent" else "mesh summaries"
    prompt = (
        "Correct the complete 3D axis line for every listed revolute hinge or spin joint. Use all supplied "
        f"visual, mesh, and numeric evidence. {inspection} All returned coordinates must use the normalized "
        f"Z-up visual frame shared by the listed numeric 3-vectors and camera matrices. {source_context} "
        "The paired-view entries "
        "contain the camera matrices, and "
        "kinematics is already validated; do not change topology or joint subtype. For hinge, the goal "
        "is visually correct articulated motion, not maximum agreement with reconstructed contact "
        "geometry. Judge direction and position separately. initial_axis_direction is the default geometric "
        "estimate. First decide direction_action independently for each joint. Use keep when no independent "
        "evidence contradicts the initial direction, refine for a local adjustment within "
        f"{REVOLUTE_AXIS_MAX_CORRECTION_DEGREES:g} degrees, and replace for a larger correction supported "
        "by at least one direction_evidence value. Allowed evidence values are motion_plane, "
        "mechanical_centerline, and cross_joint_consistency. motion_plane requires a consistently observed "
        "3D rotation plane across multiple views; mechanical_centerline requires directly observable coupled "
        "geometry in the meshes; cross_joint_consistency is valid only when the complete assembly visibly "
        "forms a common motion plane. Never use semantic category, confidence alone, or an unsupported claim "
        "as evidence. Treat opposite signs as the same axis line. Contact PCA "
        "is usually a strong proposal for axis_direction, but the contact centroid is only a weak proposal "
        "for axis_point. "
        "Reconstructed meshes may contain thickened, overlapping, sunken, or fused parent-child "
        "surfaces, so the center of the contact band may lie inside either part; initial_axis_point "
        "equaling contact_center is not independent supporting evidence. Independently locate the "
        "visible attachment or rotation seam from the source image and paired views, then use the "
        f"{numeric_geometry} and camera matrices to express that line numerically in Z-up. "
        "After determining axis_direction, solve axis_point as a separate 2D problem in the plane "
        "perpendicular to that direction, and independently determine both cross-section coordinates. "
        "Evaluate axis_point against the complete parent-child assembly and its global articulated "
        "function, not only geometry near a local contact or seam. Mentally rotate the entire child "
        "about the proposed line and require globally plausible open and closed configurations, including "
        "appropriate closure between the intended broad surfaces without orbiting, offset, detachment, or "
        "interpenetration. Local fragmented, fused, or noisy geometry must not dominate axis placement when "
        "it conflicts with the overall link shapes and observable motion semantics. "
        "Inspect the local parent and child profiles for mechanically coupled hinge evidence such as "
        "aligned cylindrical arcs, alternating protruding knuckles, matching recesses or gaps, and "
        "features sharing a common centerline. Use that shared centerline when the complementary "
        "features are clearly observable and mutually consistent, but never invent coupling from mesh "
        "overlap alone. If reliable complementary geometry is absent, use the visible root of the "
        "moving child and the parent's exterior attachment surface. For a thin moving part emerging "
        "from a thicker parent, prefer the apparent external attachment seam over the midpoint of the "
        "penetration band. Do not retain either cross-section coordinate from contact_center without "
        "independent visual support. "
        "Choose the line around which the child would rotate without visibly orbiting, translating, "
        "sinking into the parent, or detaching from it. Do not preserve the initial point merely because "
        "it lies inside the reconstructed contact band or overlapping bounds. Return the initial line "
        "unchanged only when visual evidence independently supports both its direction and position; "
        "when contact geometry conflicts with visually plausible motion, prefer the visually plausible "
        "rotation center. For spin, "
        "return the axis normal to the observable rotation plane and through the mechanical rotation "
        "center supported by parent-child contact or overlap evidence; do not substitute a bounding-box "
        "center or infer a center from rounded appearance alone. Return the physical axis line, not "
        "merely a nearby contact PCA line. axis_point may be any point on the intended line. "
        "Return only JSON in this exact shape: "
        '{"corrections":[{"child":1,"axis_direction":[0,1,0],"axis_point":[0.12,-0.03,0.25],'
        '"direction_action":"keep","direction_evidence":[]}]}. '
        "Include every listed child exactly once and use only finite numeric 3-vectors.\n\n"
        + json.dumps(request, indent=2)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "revolute_axis_request.json", {
        **model_runtime_metadata(model, execution_mode),
        **request, "prompt": prompt,
    })
    raw = _run_mllm(
        prompt, images, output_dir / "raw_revolute_axis_response.txt", model,
        "revolute_axis_mllm", Path(__file__).parents[1], runner,
        execution_mode)
    items = extract_json(raw).get("corrections")
    if not isinstance(items, list):
        raise ValueError("revolute axis response must contain a corrections list")
    corrected, normalized = {}, []
    for item in items:
        try:
            child = int(item["child"])
            axis_z_up = np.asarray(item["axis_direction"], dtype=float)
            point_z_up = np.asarray(item["axis_point"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid revolute axis correction") from exc
        if child not in expected or child in corrected:
            raise ValueError(f"unexpected or duplicate revolute child: {child}")
        if (axis_z_up.shape != (3,) or point_z_up.shape != (3,)
                or not np.isfinite(axis_z_up).all() or not np.isfinite(point_z_up).all()):
            raise ValueError(f"revolute correction for child {child} must contain finite 3-vectors")
        axis = _to_working(axis_z_up)
        point = _to_working(point_z_up)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-10:
            raise ValueError(f"revolute correction for child {child} has a zero axis")
        axis /= norm
        initial = estimates[child]
        if np.dot(axis, initial.axis) < 0:
            axis = -axis
        deviation = math.degrees(math.acos(float(np.clip(
            np.dot(axis, initial.axis), -1.0, 1.0))))
        action = item.get("direction_action")
        if action not in {"keep", "refine", "replace"}:
            action = "keep"
        reported_evidence = item.get("direction_evidence")
        evidence = sorted({
            value for value in reported_evidence
            if isinstance(value, str) and value in REVOLUTE_AXIS_DIRECTION_EVIDENCE
        }) if isinstance(reported_evidence, list) else []
        accepted = (
            action == "refine" and deviation <= REVOLUTE_AXIS_MAX_CORRECTION_DEGREES
        ) or (action == "replace" and bool(evidence))
        requested_axis = axis.copy()
        if not accepted:
            axis = initial.axis.copy()
        center = np.asarray(contacts[child], dtype=float).mean(axis=0)
        pivot = point + axis * float(np.dot(center - point, axis))
        if np.linalg.norm(pivot - center) > diagonal:
            raise ValueError(f"revolute axis for child {child} is farther than one model diagonal from contact")
        corrected[child] = JointEstimate("revolute", initial.subtype, axis, pivot)
        normalized.append({
            "child": child, "subtype": initial.subtype,
            "axis_direction": axis.tolist(), "axis_point": pivot.tolist(),
            "raw_axis_point": point.tolist(), "contact_center": center.tolist(),
            "requested_axis_direction": requested_axis.tolist(),
            "direction_action": action, "direction_evidence": evidence,
            "direction_deviation_degrees": deviation,
            "direction_correction_accepted": accepted,
        })
    if set(corrected) != expected:
        raise ValueError(f"missing revolute children: {sorted(expected - set(corrected))}")
    write_json(output_dir / "revolute_axis_correction.json", {
        "coordinate_frame": "normalized working frame",
        "model_coordinate_frame": "normalized Z-up visual frame",
        "corrections": normalized,
    })
    return corrected


def _fallback_joint_limit(child, estimate, prismatic_states, hinge_states,
                          spin_states, hinge_children):
    if estimate.type == "prismatic":
        state = prismatic_states.get(child, {})
        mesh_q = float(state.get("mesh_current_q", 0.0))
        procedural = (
            state.get("resolved_mode") == "procedural-closed"
            and state.get("status") == "procedural"
        )
        if procedural:
            upper = float(state.get("target_hidden_length", float("nan")))
            fallback = {
                "range_source": "procedural_drawer_depth_fallback",
                "target_hidden_length": upper,
            }
        else:
            scan_lower = float(estimate.lower)
            scan_upper = float(estimate.upper)
            if (not np.isfinite([scan_lower, scan_upper, mesh_q]).all()
                    or scan_lower > 0 or scan_upper < 0
                    or scan_lower > scan_upper or mesh_q < 0):
                raise ValueError(
                    f"invalid physical scan fallback for joint {child}")
            upper = max(mesh_q, mesh_q + scan_upper)
            fallback = {
                "range_source": "physical_scan_fallback",
                "physical_scan_axis_direction": estimate.axis.tolist(),
                "physical_scan_lower": scan_lower,
                "physical_scan_upper": scan_upper,
            }
    else:
        state = hinge_states[child] if child in hinge_children else spin_states[child]
        mesh_q = float(state["mesh_current_q"])
        upper = max(mesh_q, float(state.get("plane_closed_q", 0.0)))
        fallback = {"range_source": "geometric_state_fallback"}
    if not np.isfinite([mesh_q, upper]).all() or mesh_q < 0 or upper < mesh_q:
        raise ValueError(f"invalid range fallback for joint {child}")
    return estimate.with_limits(0.0, upper), {
        "child": child, "applicable": True,
        "semantic_applicable": False,
        "lower": 0.0, "upper": upper,
        "zero_q": -mesh_q, "mesh_current_q": mesh_q,
        "axis_direction": estimate.axis.tolist(),
        "axis_flipped_during_canonicalization": False,
        **fallback,
    }


def correct_joint_limits(
        view_paths: Sequence[Path], views_manifest: Path, spec: KinematicSpec,
        estimates: Mapping[int, JointEstimate], contacts: Mapping[int, np.ndarray],
        parent_mesh_paths: Mapping[int, Path], child_mesh_paths: Mapping[int, Path],
        diagonal: float, model: str | None, output_dir: Path,
        source_image: Path | None = None, rgb_view_paths: Sequence[Path] = (),
        source_mesh: Path | None = None, primitive_mesh: Path | None = None,
        prismatic_states: Mapping[int, Mapping] | None = None,
        hinge_states: Mapping[int, Mapping] | None = None,
        spin_states: Mapping[int, Mapping] | None = None,
        primary_image: Path | None = None,
        runner=subprocess.run,
        execution_mode: str = "reasoning-only") -> tuple[dict[int, JointEstimate], list[dict]]:
    """Infer canonical limits and reconstruction state with primary-image evidence."""
    expected = {joint.child for joint in spec.joints}
    if (set(estimates) != expected or set(contacts) != expected
            or set(parent_mesh_paths) != expected or set(child_mesh_paths) != expected):
        raise ValueError("semantic limit inputs must match the kinematic tree")
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("working mesh diagonal must be positive")
    hinge_states = hinge_states or {}
    hinge_children = {
        joint.child for joint in spec.joints
        if joint.type == "revolute" and joint.subtype == "hinge"
    }
    spin_states = spin_states or {}
    spin_children = {
        joint.child for joint in spec.joints
        if joint.type == "revolute" and joint.subtype == "spin"
    }
    if set(hinge_states) != hinge_children:
        raise ValueError("semantic limit correction requires one plane state per hinge")
    if set(spin_states) != spin_children:
        raise ValueError("semantic limit correction requires one closure state per spin")
    revolute_children = hinge_children | spin_children
    prismatic_children = expected - revolute_children
    if prismatic_children and (source_image is None or primary_image is None):
        raise ValueError("canonical limit inference requires source and primary images")
    if revolute_children and primary_image is None:
        raise ValueError("revolute limit inference requires a primary image")
    if any((estimate.lower is None or estimate.upper is None) and child not in revolute_children
           for child, estimate in estimates.items()):
        raise ValueError("non-deterministic semantic limit correction requires physical scan limits")
    if not views_manifest.is_file() or any(not path.is_file() for path in view_paths):
        raise FileNotFoundError("semantic limit correction requires the five views and views.json")
    if any(not path.is_file() for path in (*parent_mesh_paths.values(), *child_mesh_paths.values())):
        raise FileNotFoundError("semantic limit parent or child mesh is missing")

    primitive_ids = {primitive for link in spec.links for primitive in link.primitive_ids}
    evidence, images = _multimodal_evidence(
        view_paths, primitive_ids, source_image, rgb_view_paths, source_mesh, primitive_mesh,
        primary_image=primary_image, redact_paths=True,
        execution_mode=execution_mode)
    evidence = _file_evidence(
        evidence, output_dir, source_mesh, primitive_mesh, views_manifest,
        execution_mode)
    prismatic_states = prismatic_states or {}
    joints_by_child = {joint.child: joint for joint in spec.joints}
    request_joints = []
    for child in sorted(prismatic_children):
        joint, estimate = joints_by_child[child], estimates[child]
        item = {
            "parent": joint.parent, "child": child, "type": estimate.type,
            "subtype": estimate.subtype,
            **({
                "parent_mesh": _relative_file(parent_mesh_paths[child], output_dir),
                "child_mesh": _relative_file(child_mesh_paths[child], output_dir),
            } if execution_mode == "agent" else {
                "parent_mesh_geometry": _mesh_geometry_summary(
                    parent_mesh_paths[child]),
                "child_mesh_geometry": _mesh_geometry_summary(
                    child_mesh_paths[child]),
            }),
            "axis_direction": (
                estimate.axis if execution_mode == "agent"
                else _to_z_up(estimate.axis)
            ).tolist(),
            "axis_point": (
                estimate.pivot if execution_mode == "agent"
                else _to_z_up(estimate.pivot)
            ).tolist(),
            "contact_center": (
                np.asarray(contacts[child], dtype=float).mean(axis=0)
                if execution_mode == "agent" else
                _to_z_up(np.asarray(contacts[child], dtype=float).mean(axis=0))
            ).tolist(),
        }
        physical = {
            "lower": estimate.lower, "upper": estimate.upper,
            "reference_q": 0.0,
            "reference_pose": "reconstruction geometry before semantic rebasing",
            "units": "normalized working-frame distance",
        }
        known_mesh_q = prismatic_states.get(child, {}).get("mesh_current_q")
        if known_mesh_q is not None:
            physical["closed_frame_lower"] = max(
                0.0, float(estimate.lower) + float(known_mesh_q))
            physical["closed_frame_upper"] = max(
                0.0, float(estimate.upper) + float(known_mesh_q))
        item.update({
            "mesh_current_q": (
                float(prismatic_states[child]["mesh_current_q"])
                if child in prismatic_states else None),
            "drawer_geometry_canonicalized": child in prismatic_states,
            "physical_scan": physical,
        })
        request_joints.append(item)
    request = {
        "kinematics": _generic_kinematics(spec),
        "coordinate_frame": (
            "normalized working frame used by parent/child OBJ files"
            if execution_mode == "agent" else
            "normalized Z-up visual frame used by all mesh summaries and 3D vectors"
        ),
        "joint_coordinate_convention": JOINT_COORDINATE_CONVENTION,
        "working_mesh_diagonal": diagonal, "evidence": evidence, "joints": request_joints,
        "locked_revolute_children": [],
    }
    mesh_guidance = (
        "referenced meshes, topology, axis, and contact evidence. The source image shows the "
        "reconstruction state. The primary image is additional semantic evidence for the plausible "
        "prismatic range; its observed pose is not necessarily the maximum pose. Resolve every mesh "
        "path relative to this request file and inspect it with read-only tools."
        if execution_mode == "agent" else
        "precomputed mesh summaries, topology, axis, and contact evidence. The source image shows "
        "the reconstruction state. The primary image is additional semantic evidence for the plausible "
        "prismatic range; its observed pose is not necessarily the maximum pose. All summary coordinates "
        "are already in normalized Z-up. No file access or external tools are available."
    )
    prompt = (
        "Infer only the requested prismatic semantic range. The top-level joints array is "
        "the authoritative request set: return corrections for exactly its child IDs, and ignore kinematics "
        "joints absent from that array. Use the attached source image, primary image, camera-paired views, "
        f"embedded camera matrices, {mesh_guidance} Infer mechanics from "
        "consistent observable evidence, never from category conventions. Do not change topology, type, subtype, "
        "axis, or pivot. Hold the parent fixed; q is the child's full motion relative to that parent, never a "
        "world, camera, image, or object-centerline orientation. "
        "For a prismatic joint, direct parent-child axial displacement measured from the meshes has priority over "
        "physical-scan bounds when inferring reconstruction pose and canonical state. A physical scan is secondary "
        "feasibility evidence from reconstruction reference_q=0; its bounds are relative displacements, not "
        "absolute canonical values, and must not replace or clamp a direct mesh measurement. Use signed child "
        "position minus parent position along axis_direction, never either link's absolute position. q is "
        "nonnegative from canonical state toward actuated motion, and any supplied reconstruction state must be "
        "preserved exactly. Do not estimate or return a primary-image state, scene_current_q, or delta_q. Every "
        "range must contain zero and mesh_current_q. All listed joints are already validated and must be "
        "preserved. applicable=false means only that a semantic range cannot be inferred and requests the "
        "existing physical-scan range as fallback; it never invalidates a joint or changes topology. Return "
        "only JSON. Each false item must contain exactly child and applicable. Each true prismatic item must contain exactly child, "
        "applicable, mesh_current_q, and max_travel. Include every listed child exactly once. "
        "Example shape only: "
        '{"corrections":[{"child":1,"applicable":false}]}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "canonical_joint_request.json", {
        **model_runtime_metadata(model, execution_mode),
        **request, "prompt": prompt,
    })
    if request_joints:
        call_prompt = (
            "Read canonical_joint_request.json completely, inspect every referenced mesh, follow its "
            "prompt field exactly, and return only the requested JSON."
            if execution_mode == "agent" else
            prompt + "\n\nStructured request:\n" + json.dumps(request, indent=2)
        )
        raw = _run_mllm(
            call_prompt,
            images, (output_dir / "raw_canonical_joint_response.txt").resolve(), model,
            "canonical_joint_mllm", output_dir.resolve(), runner,
            execution_mode)
        items = extract_json(raw).get("corrections")
        if not isinstance(items, list):
            raise ValueError("semantic joint response must contain a corrections list")
    else:
        items = []

    if revolute_children:
        locator = _revolute_joint_locator(
            rgb_view_paths, views_manifest, estimates, spec,
            output_dir / "revolute_joint_locator.png", diagonal / 2)
        lower_pose = _revolute_lower_pose(
            spec, estimates, parent_mesh_paths, child_mesh_paths,
            hinge_states, spin_states,
            output_dir / "revolute_lower_pose.png")
        revolute_request = {
            "kinematic_structure": {
                "links": [
                    {"id": link.id, "name": link.name}
                    for link in sorted(spec.links, key=lambda value: value.id)
                ],
                "joints": [
                    {"parent": joint.parent, "child": joint.child,
                     "type": joint.type, "subtype": joint.subtype}
                    for joint in sorted(spec.joints, key=lambda value: value.child)
                ],
            },
            "requested_revolute_children": sorted(revolute_children),
            "evidence": {
                "primary_image": "attached_primary_image",
                "joint_locator": "attached_five_view_joint_locator",
                "lower_pose": "attached_revolute_lower_pose",
            },
            "prompt": (
                "Infer only the intrinsic maximum opening of each requested revolute hinge or spin joint. "
                "First infer the object's likely name, functional category, and normal operation from the "
                "primary image. Use the common maximum opening angle for that inferred name and category as "
                "prior knowledge, then combine that prior with the calibrated joint views to decide. Locate "
                "each exact J<number> pivot and axis in the calibrated five-view RGB locator and use the "
                "kinematic_structure to identify its parent, child, tree position, and complete moving "
                "subtree. For joint Jx, the moving assembly is its child link and every descendant. Combine "
                "the object category, supplied joint type, J-number and location, surrounding structure, "
                "and expected motion of that entire assembly to infer that joint's maximum independently. "
                "The lower-pose rendering shows the deterministic canonical q=0 geometry produced by the "
                "pipeline for all requested revolute joints; use it as the exact visual reference from which "
                "upper_degrees is measured. "
                "Topology, parent-child relations, and joint types are authoritative and must not be "
                "changed. Link names are weak hints only and must agree with the images and axis location. "
                "Do not assign the same range merely because joints share a subtype, and do not default any "
                "joint to 360 degrees. Return 360 only when this specific joint's functional role and "
                "complete moving subtree support continuous rotation without a visible or semantic limit "
                "such as wiring, linked members, enclosure geometry, or a mechanical stop. Do not infer or "
                "change axis, pivot, canonical zero, reconstruction state, or opening direction. The maximum "
                "is measured from the closed canonical pose and may be any finite value from 0 through 360 "
                "degrees; the observed primary pose need not be the maximum. Return exactly one item for every "
                "requested child and no others, using only JSON with a top-level corrections array. Every true "
                "item must contain exactly child, applicable, and upper_degrees; every false item must contain "
                "exactly child and applicable. applicable=false requests the existing geometric-state fallback; "
                "it never invalidates a joint or changes topology."
            ),
        }
        write_json(output_dir / "revolute_joint_request.json", {
            **model_runtime_metadata(model, execution_mode),
            **revolute_request,
        })
        revolute_call_prompt = (
            "Read revolute_joint_request.json completely, inspect all three attached images, follow its "
            "prompt field exactly, and return only the requested JSON."
            if execution_mode == "agent" else
            revolute_request["prompt"] + "\n\nStructured request:\n" + json.dumps({
                key: value for key, value in revolute_request.items() if key != "prompt"
            }, indent=2)
        )
        raw = _run_mllm(
            revolute_call_prompt,
            [primary_image, locator, lower_pose],
            (output_dir / "raw_revolute_joint_response.txt").resolve(), model,
            "revolute_joint_mllm", output_dir.resolve(), runner,
            execution_mode)
        revolute_items = extract_json(raw).get("corrections")
        if not isinstance(revolute_items, list):
            raise ValueError("revolute joint response must contain a corrections list")
        items.extend(revolute_items)
    corrected, normalized = dict(estimates), []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or type(item.get("applicable")) is not bool:
            raise ValueError("invalid semantic joint correction")
        try:
            child = int(item["child"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid semantic joint child") from exc
        if child not in expected or child in seen:
            raise ValueError(f"unexpected or duplicate semantic joint child: {child}")
        seen.add(child)
        if not item["applicable"]:
            if set(item) != {"child", "applicable"}:
                raise ValueError(f"inapplicable semantic joint {child} has unused fields")
            corrected[child], fallback = _fallback_joint_limit(
                child, estimates[child], prismatic_states, hinge_states,
                spin_states, hinge_children,
            )
            normalized.append(fallback)
            continue

        estimate = estimates[child]
        hinge = child in hinge_children
        spin = child in spin_children
        requested_upper_degrees = collision_upper_degrees = None
        upper_clamped_by_collision = False
        deterministic = hinge or spin
        revolute = estimate.type == "revolute"
        prismatic = estimate.type == "prismatic"
        known_mesh_q = prismatic_states.get(child, {}).get("mesh_current_q")
        if deterministic:
            required = {"child", "applicable", "upper_degrees"}
        elif prismatic:
            required = {"child", "applicable", "mesh_current_q", "max_travel"}
        else:
            required = {
                "child", "applicable", "mesh_current_q_degrees", "lower_degrees", "upper_degrees",
            }
        if set(item) != required:
            raise ValueError(f"semantic joint {child} fields do not match its type")
        try:
            if deterministic:
                state = hinge_states[child] if hinge else spin_states[child]
                mesh_q = math.degrees(float(state["mesh_current_q"]))
                plane_closed_q = (
                    math.degrees(float(state.get("plane_closed_q", 0.0))) if hinge else 0.0)
                requested_upper_degrees = float(item["upper_degrees"])
                lower = 0.0
                upper = min(requested_upper_degrees, 360.0)
                zero = -mesh_q
            elif prismatic:
                returned_mesh_q = float(item["mesh_current_q"])
                mesh_q = float(known_mesh_q if known_mesh_q is not None
                               else returned_mesh_q)
                lower, upper, zero = 0.0, float(item["max_travel"]), -mesh_q
            else:
                mesh_q = float(item["mesh_current_q_degrees"])
                lower = float(item["lower_degrees"])
                requested_upper_degrees = float(item["upper_degrees"])
                upper = min(requested_upper_degrees, 360.0)
                zero = -mesh_q
        except (TypeError, ValueError) as exc:
            raise ValueError(f"semantic joint {child} contains non-numeric values") from exc
        if revolute and (not math.isfinite(requested_upper_degrees)
                         or requested_upper_degrees < 0):
            raise ValueError(f"semantic revolute joint {child} has an invalid upper limit")
        if deterministic and np.isfinite([mesh_q, upper]).all():
            upper = max(upper, mesh_q)
        if (not np.isfinite([zero, mesh_q, lower, upper]).all()
                or lower > 0 or upper < 0 or lower > upper):
            raise ValueError(f"semantic joint {child} must have a finite ordered range containing zero")
        if deterministic:
            if (mesh_q < 0 or plane_closed_q < 0
                    or upper < max(mesh_q, plane_closed_q)):
                raise ValueError(
                    f"revolute joint {child} maximum opening must contain mesh_current_q")
            angular_values = [zero, mesh_q, lower, upper, upper - lower]
            if spin and max(map(abs, angular_values)) > 360:
                raise ValueError(f"semantic revolute joint {child} exceeds one revolution")
        elif prismatic:
            if (known_mesh_q is not None
                    and not math.isclose(float(known_mesh_q), returned_mesh_q,
                                         rel_tol=1e-6, abs_tol=1e-9)):
                raise ValueError(f"prismatic joint {child} did not preserve supplied mesh_current_q")
        physical_axis = estimate.axis.copy()
        axis_flipped = False
        if revolute and not deterministic:
            if not lower <= mesh_q <= upper:
                raise ValueError(
                    f"revolute joint {child} mesh_current_q must lie within its limits")
            negative_side = lower < 0 and math.isclose(upper, 0.0, abs_tol=1e-9)
            if negative_side:
                estimate = JointEstimate(
                    estimate.type, estimate.subtype, -estimate.axis, estimate.pivot)
                zero, mesh_q = -zero, -mesh_q
                lower, upper = -upper, -lower
                axis_flipped = True
            angular_values = [zero, mesh_q, lower, upper, upper - lower]
            if max(map(abs, angular_values)) > 360:
                raise ValueError(f"semantic revolute joint {child} exceeds one revolution")
            zero, mesh_q, lower, upper = map(math.radians, (zero, mesh_q, lower, upper))
        elif deterministic:
            zero, mesh_q, lower, upper = map(math.radians, (zero, mesh_q, lower, upper))
        elif (not np.isfinite([mesh_q]).all() or mesh_q < 0 or upper <= 0
              or max(mesh_q, upper) > 2 * diagonal or mesh_q > upper):
            raise ValueError(
                f"prismatic joint {child} requires nonnegative states within max_travel")
        corrected[child] = estimate.with_limits(lower, upper)
        normalized_item = {
            "child": child, "applicable": True, "lower": lower, "upper": upper,
            "zero_q": zero, "mesh_current_q": mesh_q,
            "axis_direction": estimate.axis.tolist(),
            "axis_flipped_during_canonicalization": axis_flipped,
        }
        if not deterministic:
            normalized_item["physical_scan_axis_direction"] = physical_axis.tolist()
        if deterministic:
            normalized_item.update({
                "mllm_upper": math.radians(requested_upper_degrees),
                **({"plane_closed_q": math.radians(plane_closed_q)} if hinge else {}),
                "collision_upper": (
                    None if collision_upper_degrees is None
                    else math.radians(collision_upper_degrees)),
                "upper_clamped_by_collision": upper_clamped_by_collision,
            })
        normalized.append(normalized_item)
    if seen != expected:
        raise ValueError(f"missing semantic joint children: {sorted(expected - seen)}")
    normalized.sort(key=lambda item: item["child"])
    write_json(output_dir / "canonical_joint_correction.json", {"corrections": normalized})
    return corrected, normalized
