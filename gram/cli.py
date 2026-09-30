from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .artifacts import KinematicSpec, PrimitiveManifest, RunPaths, read_json, write_json
from .hinge_planes import infer_hinge_plane_state
from .joints import (JointEstimate, extract_contact_points, initialize_prismatic,
                     prismatic_pca_candidates, require_contacts, revolute_hypotheses,
                     spin_axis_support)
from .parts import (LinkGeometry, build_link_meshes, merge_link_into_parent,
                    merge_tiny_leaf_links)
from .physics import (CollisionEvaluator, PhysicsConfig, find_prismatic_limits,
                      find_revolute_limits, refine_joint, trajectory_objective,
                      transform_prismatic)
from .render import render_primitive_views
from .prismatic import repair_prismatic_geometry
from .segment import run_p3sam
from .spin_closure import infer_spin_closure_state, scan_spin_upper_limit
from .urdf import export_urdf
from .vlm import (correct_joint_limits, correct_revolute_axes,
                  infer_kinematics, model_runtime_metadata, select_drawer_cabinet_axes,
                  select_prismatic_axes)

STAGES = (
    "segment", "render", "infer-structure", "build-parts",
    "fit-joints", "physics", "export",
)


@dataclass
class PipelineConfig:
    input: Path
    paths: RunPaths
    seed: int = 0
    p3sam_python: Path | None = None
    p3sam_checkout: Path | None = None
    p3sam_checkpoint: Path | None = None
    blender: Path | None = None
    mllm_model: str | None = None
    min_link_faces: int = 50
    contact_ratio: float = 0.015
    collision_clearance: float = 0.002
    trajectory_span: float = 1.0
    limit_step: float = 2.0
    density: float = 500.0
    source_image: Path | None = None
    primary_image: Path | None = None
    drawer_geometry_mode: str = "auto-3d"
    enable_revolute_subtype_override: bool = False
    enable_joint_refinement: bool = False
    p3sam_sonata_dir: Path | None = None
    enable_spin_upper_collision_scan: bool = False
    joint_selection_mode: str = "conservative"
    enable_spin_axis_guard: bool = False
    vlm_execution_mode: str = "reasoning-only"


def _require(path: Path, message: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(message)
    return path


def _primitive_manifest(config: PipelineConfig) -> PrimitiveManifest:
    path = _require(config.paths.primitives / "manifest.json",
                    "render requires 01_primitives/manifest.json; run gram segment first")
    return PrimitiveManifest.from_dict(read_json(path))


def _spec(config: PipelineConfig) -> KinematicSpec:
    manifest = _primitive_manifest(config)
    path = _require(config.paths.kinematics / "kinematics.json",
                    "build-parts requires 03_kinematics/kinematics.json; run infer-structure first")
    return KinematicSpec.from_dict(read_json(path), set(manifest.primitive_ids))


def _effective_spec(config: PipelineConfig) -> KinematicSpec:
    effective = config.paths.joints / "effective_kinematics.json"
    if effective.is_file():
        manifest = _primitive_manifest(config)
        return KinematicSpec.from_dict(
            read_json(effective), set(manifest.primitive_ids))
    fallback = config.paths.joints / "rigid_fallback.json"
    if not fallback.is_file():
        return _spec(config)
    manifest = _primitive_manifest(config)
    return KinematicSpec.from_dict(
        read_json(fallback)["effective_kinematics"],
        set(manifest.primitive_ids),
    )


def segment_stage(config: PipelineConfig) -> None:
    if not all((config.p3sam_python, config.p3sam_checkout, config.p3sam_checkpoint,
                config.p3sam_sonata_dir)):
        raise ValueError("segment requires --p3sam-python, --p3sam-checkout, "
                         "--p3sam-checkpoint, and --p3sam-sonata-dir")
    run_p3sam(config.input, config.paths.primitives, config.p3sam_python,
              config.p3sam_checkout, config.p3sam_checkpoint,
              config.p3sam_sonata_dir, config.seed)


def render_stage(config: PipelineConfig) -> None:
    if config.blender is None:
        raise ValueError("render requires --blender")
    manifest = _primitive_manifest(config)
    render_primitive_views(config.paths.primitives / manifest.mesh_path,
                           config.paths.primitives / manifest.face_ids_path,
                           config.paths.views, config.blender, source_mesh=config.input,
                           save_blend=False)


def infer_structure_stage(config: PipelineConfig) -> None:
    manifest = _primitive_manifest(config)
    views = [_require(config.paths.views / f"view_{name}.png", "infer-structure requires five rendered views")
             for name in ("front", "right", "back", "left", "top")]
    rgb_views = [_require(config.paths.views / f"view_{name}_rgb.png", "infer-structure requires five RGB views")
                 for name in ("front", "right", "back", "left", "top")]
    infer_kinematics(
        views, set(manifest.primitive_ids), config.mllm_model, config.paths.kinematics,
        source_image=config.source_image, rgb_view_paths=rgb_views, source_mesh=config.input,
        primitive_mesh=config.paths.views / "primitives.glb",
        working_mesh=config.paths.primitives / manifest.mesh_path,
        face_primitive_ids=config.paths.primitives / manifest.face_ids_path,
        primary_image=config.primary_image,
        joint_selection_mode=config.joint_selection_mode,
        execution_mode=config.vlm_execution_mode)


def build_parts_stage(config: PipelineConfig) -> None:
    manifest, spec = _primitive_manifest(config), _spec(config)
    mesh = trimesh.load(_require(config.paths.primitives / manifest.mesh_path, "working mesh is missing"), force="mesh", process=False)
    labels = np.load(_require(config.paths.primitives / manifest.face_ids_path, "face primitive IDs are missing"))
    spec, merges = merge_tiny_leaf_links(labels, spec, config.min_link_faces)
    if merges:
        write_json(config.paths.kinematics / "kinematics.json", spec.to_dict())
        write_json(config.paths.parts / "tiny_link_merges.json", {"merges": merges})
    build_link_meshes(mesh, labels, spec, config.paths.parts, config.min_link_faces)


def _load_links(
        config: PipelineConfig, spec: KinematicSpec,
        prefer_joint_parts: bool = False,
) -> dict[int, LinkGeometry]:
    parts_dir = config.paths.parts
    repaired_parts = config.paths.joints / "parts"
    if prefer_joint_parts and (repaired_parts / "parts.json").is_file():
        parts_dir = repaired_parts
    links = {}
    for item in read_json(_require(
            parts_dir / "parts.json",
            "required link geometry is missing; run build-parts and fit-joints first",
    ))["links"]:
        index = int(item["id"])
        visual, collision = Path(item["visual_path"]), Path(item["collision_path"])
        if not visual.is_absolute(): visual = parts_dir / visual
        if not collision.is_absolute(): collision = parts_dir / collision
        links[index] = LinkGeometry(index, tuple(item["primitive_ids"]),
                                    trimesh.load(visual, force="mesh", process=False),
                                    trimesh.load(collision, force="mesh", process=False), visual, collision)
    if set(links) != {link.id for link in spec.links}:
        raise ValueError("parts manifest does not match the validated kinematic tree")
    return links


def _descendants(spec: KinematicSpec, root: int) -> set[int]:
    children = {link.id: [] for link in spec.links}
    for joint in spec.joints: children[joint.parent].append(joint.child)
    result, stack = set(), [root]
    while stack:
        node = stack.pop(); result.add(node); stack.extend(children[node])
    return result


def _require_primary_image(config: PipelineConfig) -> Path:
    if config.primary_image is None:
        raise ValueError(
            "physics requires --primary-image; it must be the primary scene image, "
            "not an inferred reconstruction fallback"
        )
    if not config.primary_image.is_file():
        raise FileNotFoundError(
            f"physics primary image is not a regular file: {config.primary_image}"
        )
    return config.primary_image


def _select_revolute_estimate(
        contacts, parent, child, declared_subtype, seed, score, allow_override=False):
    hinge, spin, preference = revolute_hypotheses(contacts, parent, child, seed)
    declared = hinge if declared_subtype == "hinge" or spin is None else spin
    if not allow_override:
        return declared
    alternate = spin if preference == "spin" else hinge if preference == "hinge" else None
    if alternate is None or alternate.subtype == declared.subtype:
        return declared
    _, declared_collisions = score(declared)
    _, alternate_collisions = score(alternate)
    return alternate if alternate_collisions <= declared_collisions else declared


def _convert_to_rigid(
        config: PipelineConfig, spec: KinematicSpec, child: int,
        reason: str, **diagnostics) -> None:
    manifest = _primitive_manifest(config)
    rigid_spec = KinematicSpec.from_dict({
        "links": [{
            "id": 0,
            "name": "rigid_fallback",
            "primitive_ids": list(manifest.primitive_ids),
        }],
        "joints": [],
    }, set(manifest.primitive_ids))
    mesh = trimesh.load(
        _require(config.paths.primitives / manifest.mesh_path,
                 "working mesh is missing"),
        force="mesh", process=False,
    )
    labels = np.load(_require(
        config.paths.primitives / manifest.face_ids_path,
        "face primitive IDs are missing",
    ))
    parts = config.paths.joints / "parts"
    if parts.exists():
        shutil.rmtree(parts)
    links = build_link_meshes(mesh, labels, rigid_spec, parts)
    fallback = {
        "reason": reason,
        "rejected_joint_child": child,
        "original_kinematics": spec.to_dict(),
        "effective_kinematics": rigid_spec.to_dict(),
        **diagnostics,
    }
    write_json(config.paths.joints / "rigid_fallback.json", fallback)
    write_json(config.paths.joints / "initial_joints.json", {
        "joints": {}, "mllm_bounded": [],
    })
    print(
        f"[gram-stage] converted object to rigid at joint {child}: {reason}",
        flush=True,
    )


def _reject_joint(
        config: PipelineConfig, spec: KinematicSpec, child: int,
        reason: str, **diagnostics) -> KinematicSpec:
    effective, rejection = merge_link_into_parent(spec, child)
    rejection.update(reason=reason, **diagnostics)
    rejection_path = config.paths.joints / "rejected_joints.json"
    previous = (read_json(rejection_path).get("rejections", [])
                if rejection_path.is_file() else [])
    # A resumed/ablated run may share the completed fit-joints stage through a
    # directory symlink.  Rejection must fork that stage locally, not recurse
    # into (or try to rmtree) the shared source.
    if config.paths.joints.is_symlink():
        config.paths.joints.unlink()
    elif config.paths.joints.exists():
        shutil.rmtree(config.paths.joints)
    manifest = _primitive_manifest(config)
    mesh = trimesh.load(
        _require(config.paths.primitives / manifest.mesh_path,
                 "working mesh is missing"),
        force="mesh", process=False,
    )
    labels = np.load(_require(
        config.paths.primitives / manifest.face_ids_path,
        "face primitive IDs are missing",
    ))
    build_link_meshes(mesh, labels, effective, config.paths.joints / "parts")
    write_json(config.paths.joints / "effective_kinematics.json", effective.to_dict())
    write_json(rejection_path, {
        "rejections": [*previous, rejection],
        "effective_kinematics": effective.to_dict(),
    })
    print(
        f"[gram-stage] rejected joint {child} only: {reason}",
        flush=True,
    )
    return effective


def fit_joints_stage(
        config: PipelineConfig, effective_spec: KinematicSpec | None = None) -> None:
    if effective_spec is None:
        for name in (
                "rigid_fallback.json", "effective_kinematics.json",
                "rejected_joints.json"):
            path = config.paths.joints / name
            if path.is_file():
                path.unlink()
        repaired_parts = config.paths.joints / "parts"
        if repaired_parts.exists():
            shutil.rmtree(repaired_parts)
    spec = effective_spec or _spec(config)
    if config.drawer_geometry_mode == "procedural-closed" and not spec.drawer_groups:
        raise ValueError(
            "procedural-closed requires a validated drawer_groups structure"
        )
    links = _load_links(config, spec, prefer_joint_parts=effective_spec is not None)
    all_vertices = np.vstack([link.visual_mesh.vertices for link in links.values()])
    diagonal = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    config.paths.joints.mkdir(parents=True, exist_ok=True)
    view_paths = [config.paths.views / f"view_{name}.png"
                  for name in ("front", "right", "back", "left", "top")]
    rgb_view_paths = [config.paths.views / f"view_{name}_rgb.png"
                      for name in ("front", "right", "back", "left", "top")]
    drawer_children = {
        child for group in spec.drawer_groups for child in group.children
    }
    prismatic_candidates = {
        joint.child: prismatic_pca_candidates(links[joint.child].visual_mesh)
        for joint in spec.joints
        if joint.type == "prismatic" and joint.child not in drawer_children
    }
    selected_prismatic_axes = {}
    if prismatic_candidates:
        write_json(config.paths.joints / "prismatic_axis_candidates.json", {
            "joints": {str(child): {
                "child_mesh": str(links[child].visual_path), "candidates": candidates,
            } for child, candidates in prismatic_candidates.items()},
        })
        try:
            selected_prismatic_axes = select_prismatic_axes(
                view_paths, config.paths.views / "views.json", spec, prismatic_candidates,
                {child: links[child].visual_path for child in prismatic_candidates},
                config.mllm_model, config.paths.joints, source_image=config.source_image,
                rgb_view_paths=rgb_view_paths, source_mesh=config.input,
                primitive_mesh=config.paths.views / "primitives.glb",
                execution_mode=config.vlm_execution_mode)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            write_json(config.paths.joints / "prismatic_axis_selection.json", {
                "selections": [], "fallback": f"{type(exc).__name__}: {exc}",
            })
    else:
        write_json(config.paths.joints / "prismatic_axis_selection.json", {"selections": []})

    drawer_selections = {}
    if spec.drawer_groups:
        drawer_mesh_ids = drawer_children | {group.parent for group in spec.drawer_groups}
        drawer_selections = select_drawer_cabinet_axes(
            view_paths, config.paths.views / "views.json", spec,
            {link_id: links[link_id].visual_path for link_id in drawer_mesh_ids},
            config.mllm_model, config.paths.joints, source_image=config.source_image,
            rgb_view_paths=rgb_view_paths, source_mesh=config.input,
            primitive_mesh=config.paths.views / "primitives.glb",
            execution_mode=config.vlm_execution_mode)
    else:
        write_json(config.paths.joints / "drawer_axis_selection.json", {"groups": []})

    generic_axis_vectors = {
        child: (
            np.asarray(prismatic_candidates[child][selection["selected_axis"]]["axis"])
            * int(selection["positive_direction"])
        )
        for child, selection in selected_prismatic_axes.items()
    }
    basis = np.eye(3)
    drawer_axis_vectors = {
        child: basis["xyz".index(selection["axis"])]
        for child, selection in drawer_selections.items()
    }
    prismatic_states = repair_prismatic_geometry(
        spec, links, drawer_axis_vectors, drawer_selections, config.paths.joints,
        mode=config.drawer_geometry_mode)
    all_vertices = np.vstack([link.visual_mesh.vertices for link in links.values()])
    diagonal = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    initial, contexts = {}, {}
    for joint in spec.joints:
        child, parent = links[joint.child], links[joint.parent]
        contacts = extract_contact_points(child.visual_mesh, parent.visual_mesh, config.contact_ratio * diagonal)
        try:
            require_contacts(
                contacts, joint.child,
                config.paths.joints / "contacts" / f"joint_{joint.child}_diagnostic.ply",
            )
        except ValueError:
            if config.joint_selection_mode in {"high-recall", "high-recall-v2"}:
                effective = _reject_joint(
                    config, spec, joint.child, "insufficient_contact_samples",
                    contact_samples=len(contacts), minimum_contact_samples=32,
                )
                return fit_joints_stage(config, effective)
            _convert_to_rigid(
                config, spec, joint.child, "insufficient_contact_samples",
                contact_samples=len(contacts), minimum_contact_samples=32,
            )
            return
        contacts_path = config.paths.joints / "contacts" / f"joint_{joint.child}.npy"
        contacts_path.parent.mkdir(parents=True, exist_ok=True); np.save(contacts_path, contacts)
        subtree_ids = _descendants(spec, joint.child)
        moving = trimesh.util.concatenate([links[i].collision_mesh for i in subtree_ids])
        obstacles = [link.collision_mesh for i, link in links.items() if i not in subtree_ids]
        collision = CollisionEvaluator(obstacles)
        collision.calibrate(moving, config.collision_clearance)
        mllm_bounded = False
        if joint.type == "revolute":
            physics = PhysicsConfig(
                diagonal, trajectory_span=config.trajectory_span,
                collision_evaluator=collision)
            estimate = _select_revolute_estimate(
                contacts, parent.visual_mesh, child.visual_mesh, joint.subtype, config.seed,
                lambda candidate: trajectory_objective(
                    candidate, candidate, contacts, moving, obstacles, physics,
                    samples=np.deg2rad([-15, 15])),
                config.enable_revolute_subtype_override)
        else:
            axis_collision = CollisionEvaluator([parent.collision_mesh])
            axis_collision.calibrate(child.collision_mesh, config.collision_clearance)
            parent_tree = cKDTree(parent.visual_mesh.vertices)
            def score(axis, distance):
                moved = child.collision_mesh.copy(); moved.vertices = transform_prismatic(moved.vertices, axis, distance)
                _, collided = axis_collision.distance_and_collision(moved)
                moved_contacts = transform_prismatic(contacts, axis, distance)
                derailment = float(np.mean(parent_tree.query(moved_contacts)[0] > config.contact_ratio * diagonal))
                return float(collided), derailment
            selected_axis = generic_axis_vectors.get(joint.child)
            if joint.child in drawer_selections:
                selected_axis = (
                    drawer_axis_vectors[joint.child]
                    * int(drawer_selections[joint.child]["positive_direction"]))
            estimate = initialize_prismatic(
                child.visual_mesh, parent.visual_mesh, contacts, score, selected_axis)
            mllm_bounded = selected_axis is not None
        initial[joint.child] = estimate
        contexts[joint.child] = {
            "joint": joint, "contacts": contacts, "moving": moving, "obstacles": obstacles,
            "collision": collision, "mllm_bounded": mllm_bounded,
        }

    revolute_initial = {child: estimate for child, estimate in initial.items()
                        if estimate.type == "revolute"}
    if revolute_initial:
        try:
            corrected = correct_revolute_axes(
                view_paths, config.paths.views / "views.json", spec, revolute_initial,
                {child: contexts[child]["contacts"] for child in revolute_initial},
                {child: links[contexts[child]["joint"].parent].visual_path for child in revolute_initial},
                {child: links[child].visual_path for child in revolute_initial},
                diagonal, config.mllm_model, config.paths.joints,
                source_image=config.source_image, rgb_view_paths=rgb_view_paths,
                source_mesh=config.input, primitive_mesh=config.paths.views / "primitives.glb",
                execution_mode=config.vlm_execution_mode)
            for child, estimate in corrected.items():
                initial[child] = estimate
                contexts[child]["mllm_bounded"] = True
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            write_json(config.paths.joints / "revolute_axis_correction.json", {
                "corrections": [], "fallback": f"{type(exc).__name__}: {exc}",
            })
    if config.enable_spin_axis_guard:
        corrections = {
            int(item["child"]): item for item in read_json(
                config.paths.joints / "revolute_axis_correction.json",
            ).get("corrections", [])
        } if (config.paths.joints / "revolute_axis_correction.json").is_file() else {}
        for joint in spec.joints:
            if joint.type != "revolute" or joint.subtype != "spin":
                continue
            support = spin_axis_support(
                contexts[joint.child]["contacts"], initial[joint.child].axis,
            )
            correction = corrections.get(joint.child, {})
            independently_replaced = (
                correction.get("direction_action") == "replace"
                and bool(correction.get("direction_evidence"))
                and support["normal_alignment"] >= 0.85
            )
            if support["supported"] or independently_replaced:
                continue
            effective = _reject_joint(
                config, spec, joint.child, "unreliable_spin_axis", **support,
            )
            return fit_joints_stage(config, effective)

    write_json(config.paths.joints / "initial_joints.json", {
        "joints": {str(k): v.to_dict() for k, v in initial.items()},
        "mllm_bounded": [child for child, context in contexts.items()
                          if context["mllm_bounded"]],
    })


def _run_physics(config, spec, links, diagonal, view_paths, rgb_view_paths,
                 initial, contexts, prismatic_states=None) -> None:
    config.paths.physics.mkdir(parents=True, exist_ok=True)
    optimized, hinge_states, spin_states = {}, {}, {}
    for joint in spec.joints:
        context = contexts[joint.child]
        estimate = initial[joint.child]
        refined = estimate
        if config.enable_joint_refinement:
            refined = refine_joint(
                estimate, context["contacts"], context["moving"], context["obstacles"],
                PhysicsConfig(
                    diagonal, trajectory_span=config.trajectory_span,
                    trace_path=config.paths.physics / f"joint_{joint.child}_trace.json",
                    collision_evaluator=context["collision"],
                    axis_limit_degrees=10.0 if context["mllm_bounded"] else None))
        if joint.type == "revolute" and joint.subtype == "hinge":
            subtree = _descendants(spec, joint.child)
            try:
                state = infer_hinge_plane_state(
                    links[joint.parent].visual_mesh, links[joint.child].visual_mesh,
                    refined.axis, refined.pivot,
                    moving_meshes=[links[index].visual_mesh
                                   for index in sorted(subtree - {joint.child})],
                    stationary_meshes=[links[index].visual_mesh for index in sorted(links)
                                        if index not in subtree and index != joint.parent])
            except ValueError as exc:
                fallback_reason = {
                    "mechanical hinge scan found no collision-free pose":
                        "physics_no_collision_free_pose",
                    "hinge interface filtering removed an entire direct link":
                        "physics_unusable_hinge_geometry",
                }.get(str(exc))
                if fallback_reason is None:
                    raise
                if config.joint_selection_mode in {"high-recall", "high-recall-v2"}:
                    effective = _reject_joint(
                        config, spec, joint.child, fallback_reason,
                        physics_error=str(exc),
                    )
                    shutil.rmtree(config.paths.physics)
                    fit_joints_stage(config, effective)
                    return physics_stage(config)
                _convert_to_rigid(
                    config, spec, joint.child, fallback_reason,
                    physics_error=str(exc),
                )
                shutil.rmtree(config.paths.physics)
                rigid_spec = _effective_spec(config)
                rigid_links = _load_links(
                    config, rigid_spec, prefer_joint_parts=True)
                return _run_physics(
                    config, rigid_spec, rigid_links, diagonal,
                    view_paths, rgb_view_paths, {}, {},
                )
            refined = JointEstimate(
                refined.type, refined.subtype, state.axis, refined.pivot)
            hinge_states[joint.child] = state.to_dict()
            optimized[joint.child] = refined
        elif joint.type == "revolute" and joint.subtype == "spin":
            subtree = _descendants(spec, joint.child)
            state = infer_spin_closure_state(
                links[joint.parent].visual_mesh, links[joint.child].visual_mesh,
                refined.axis, refined.pivot,
                moving_meshes=[links[index].visual_mesh
                               for index in sorted(subtree - {joint.child})],
                stationary_meshes=[links[index].visual_mesh for index in sorted(links)
                                    if index not in subtree and index != joint.parent])
            refined = JointEstimate(
                refined.type, refined.subtype, state.axis, refined.pivot)
            spin_states[joint.child] = state.to_dict()
            optimized[joint.child] = refined
        else:
            state = (prismatic_states or {}).get(joint.child, {})
            procedural_travel = (
                state.get("target_hidden_length")
                if joint.type == "prismatic"
                and state.get("resolved_mode") == "procedural-closed"
                and state.get("status") == "procedural"
                else None
            )
            if procedural_travel is not None:
                procedural_travel = float(procedural_travel)
                if not np.isfinite(procedural_travel) or procedural_travel <= 0:
                    raise ValueError(
                        f"invalid procedural drawer travel for joint {joint.child}")
                limits = (0.0, procedural_travel)
            else:
                limits = (find_revolute_limits(
                    refined, context["moving"], context["obstacles"],
                    collision_evaluator=context["collision"], step_degrees=config.limit_step)
                    if joint.type == "revolute" else find_prismatic_limits(
                        refined, context["moving"], context["obstacles"], diagonal=diagonal,
                        contact_points=context["contacts"], contact_ratio=config.contact_ratio,
                        collision_evaluator=context["collision"],
                        step_ratio=config.limit_step / 100.0))
            optimized[joint.child] = refined.with_limits(*limits)
    write_json(config.paths.physics / "hinge_plane_states.json", {
        "joints": {str(child): state for child, state in sorted(hinge_states.items())},
    })
    write_json(config.paths.physics / "spin_closure_states.json", {
        "joints": {str(child): state for child, state in sorted(spin_states.items())},
    })
    if optimized:
        optimized, canonical_corrections = correct_joint_limits(
            view_paths, config.paths.views / "views.json", spec, optimized,
            {child: context["contacts"] for child, context in contexts.items()},
            {joint.child: links[joint.parent].visual_path for joint in spec.joints},
            {joint.child: links[joint.child].visual_path for joint in spec.joints},
            diagonal, config.mllm_model, config.paths.physics,
            source_image=config.source_image, primary_image=config.primary_image,
            rgb_view_paths=rgb_view_paths,
            source_mesh=config.input, primitive_mesh=config.paths.views / "primitives.glb",
            prismatic_states=prismatic_states, hinge_states=hinge_states,
            spin_states=spin_states, execution_mode=config.vlm_execution_mode)
        if config.enable_spin_upper_collision_scan:
            optimized, canonical_corrections, spin_upper_checks = (
                _apply_spin_upper_collision_checks(
                    spec, links, optimized, canonical_corrections))
        else:
            spin_upper_checks = []
        write_json(config.paths.physics / "canonical_joint_correction.json", {
            "corrections": canonical_corrections,
        })
        write_json(config.paths.physics / "spin_upper_collision_checks.json", {
            "joints": {str(item["child"]): item for item in spin_upper_checks},
        })
        same_image = _same_image_pixels(config.source_image, config.primary_image)
        correction_items = [{
            "child": int(item["child"]), "applicable": True,
            "lower": float(item["lower"]), "upper": float(item["upper"]),
            "zero_q": float(item["zero_q"]),
            "mesh_current_q": float(item["mesh_current_q"]),
            "scene_current_q": (float(item["mesh_current_q"]) if same_image else 0.0),
        } for item in canonical_corrections]
        write_json(config.paths.physics / "semantic_joint_correction.json", {
            "corrections": correction_items,
        })
    else:
        correction_items = []
        write_json(config.paths.physics / "semantic_joint_correction.json", {"corrections": []})
        write_json(config.paths.physics / "spin_upper_collision_checks.json", {"joints": {}})
    write_json(config.paths.physics / "optimized_joints.json", {"joints": {str(k): v.to_dict() for k, v in optimized.items()}})


def _apply_spin_upper_collision_checks(spec, links, optimized, corrections):
    corrections_by_child = {int(item["child"]): item for item in corrections}
    persistent = set(spec.persistent_spin_children)
    diagnostics = []
    for joint in spec.joints:
        if joint.type != "revolute" or joint.subtype != "spin":
            continue
        child, parent = joint.child, joint.parent
        estimate, correction = optimized[child], corrections_by_child[child]
        subtree = _descendants(spec, child)
        final_upper, item = scan_spin_upper_limit(
            links[parent].visual_mesh, links[child].visual_mesh,
            estimate.axis, estimate.pivot, float(estimate.upper),
            float(correction["mesh_current_q"]),
            persistent_interface=child in persistent,
            moving_meshes=[links[index].visual_mesh
                           for index in sorted(subtree - {child})],
            stationary_meshes=[links[index].visual_mesh for index in sorted(links)
                               if index not in subtree and index != parent])
        optimized[child] = estimate.with_limits(float(estimate.lower), final_upper)
        correction["upper"] = final_upper
        diagnostics.append({"child": child, "parent": parent, **item})
    return optimized, corrections, diagnostics


def _same_image_pixels(source_image: Path | None, primary_image: Path | None) -> bool:
    if source_image is None or primary_image is None:
        raise ValueError("semantic scene state requires source_image and primary_image")
    from PIL import Image
    with Image.open(source_image) as source, Image.open(primary_image) as primary:
        source_rgba = np.asarray(source.convert("RGBA"))
        primary_rgba = np.asarray(primary.convert("RGBA"))
    return source_rgba.shape == primary_rgba.shape and np.array_equal(source_rgba, primary_rgba)


def physics_stage(config: PipelineConfig) -> None:
    spec = _effective_spec(config)
    if spec.joints:
        _require_primary_image(config)
    links = _load_links(config, spec, prefer_joint_parts=True)
    all_vertices = np.vstack([link.visual_mesh.vertices for link in links.values()])
    diagonal = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    payload = read_json(_require(
        config.paths.joints / "initial_joints.json",
        "physics requires 05_joints/initial_joints.json; run fit-joints first"))
    initial = {int(key): JointEstimate.from_dict(value)
               for key, value in payload["joints"].items()}
    bounded = set(payload.get("mllm_bounded", ()))
    if "mllm_bounded" not in payload:
        for path, key in (
                (config.paths.joints / "revolute_axis_correction.json", "corrections"),
                (config.paths.joints / "prismatic_axis_selection.json", "selections")):
            if path.is_file():
                bounded.update(int(item["child"]) for item in read_json(path).get(key, ()))
        drawer_path = config.paths.joints / "drawer_axis_selection.json"
        if drawer_path.is_file():
            bounded.update(
                int(item["child"])
                for group in read_json(drawer_path).get("groups", ())
                for item in group.get("children", ()))
    selection_path = config.paths.joints / "drawer_axis_selection.json"
    selections = {}
    if selection_path.is_file():
        for group in read_json(selection_path).get("groups", []):
            for child_item in group["children"]:
                child = int(child_item["child"])
                selections[child] = {
                    **child_item,
                    "axis": group["axis"],
                    "positive_direction": group["positive_direction"],
                    "confidence": group["confidence"],
                }
    basis = np.eye(3)
    prismatic_axes = {
        child: basis["xyz".index(item["axis"])]
        for child, item in selections.items()
    }
    prismatic_states = repair_prismatic_geometry(
        spec, links, prismatic_axes, selections, config.paths.joints,
        mode=config.drawer_geometry_mode, reuse_existing=True)
    all_vertices = np.vstack([link.visual_mesh.vertices for link in links.values()])
    diagonal = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    contexts = {}
    for joint in spec.joints:
        contacts = extract_contact_points(
            links[joint.child].visual_mesh, links[joint.parent].visual_mesh,
            config.contact_ratio * diagonal)
        require_contacts(
            contacts, joint.child,
            config.paths.joints / "contacts" / f"joint_{joint.child}_diagnostic.ply")
        np.save(config.paths.joints / "contacts" / f"joint_{joint.child}.npy", contacts)
        subtree_ids = _descendants(spec, joint.child)
        moving = trimesh.util.concatenate([links[i].collision_mesh for i in subtree_ids])
        obstacles = [link.collision_mesh for i, link in links.items() if i not in subtree_ids]
        collision = CollisionEvaluator(obstacles)
        collision.calibrate(moving, config.collision_clearance)
        contexts[joint.child] = {
            "joint": joint, "contacts": contacts, "moving": moving,
            "obstacles": obstacles, "collision": collision,
            "mllm_bounded": joint.child in bounded,
        }
    view_paths = [config.paths.views / f"view_{name}.png"
                  for name in ("front", "right", "back", "left", "top")]
    rgb_view_paths = [config.paths.views / f"view_{name}_rgb.png"
                      for name in ("front", "right", "back", "left", "top")]
    _run_physics(config, spec, links, diagonal, view_paths, rgb_view_paths,
                 initial, contexts, prismatic_states)


def _scene_state_metadata(spec, corrections):
    by_child = {
        int(item["child"]): item
        for item in corrections if item.get("applicable")
    }
    result = {}
    for joint in spec.joints:
        item = by_child.get(joint.child)
        if item is None:
            result[joint.child] = {
                "zero_q": 0.0, "mesh_current_q": 0.0, "scene_current_q": 0.0,
            }
            continue
        required = {"zero_q", "mesh_current_q", "scene_current_q"}
        if not required <= set(item):
            raise ValueError(
                f"applicable semantic joint {joint.child} is missing state fields")
        zero, mesh, scene = (
            float(item["zero_q"]), float(item["mesh_current_q"]),
            float(item["scene_current_q"]),
        )
        if (not np.isfinite([zero, mesh, scene]).all()
                or not math.isclose(zero, -mesh, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(f"semantic joint {joint.child} has inconsistent state fields")
        result[joint.child] = {
            "zero_q": zero, "mesh_current_q": mesh, "scene_current_q": scene,
        }
    return result


def export_stage(config: PipelineConfig) -> None:
    spec = _effective_spec(config)
    links = _load_links(config, spec, prefer_joint_parts=True)
    payload = read_json(_require(config.paths.physics / "optimized_joints.json",
                                 "export requires 06_physics/optimized_joints.json; run physics first"))
    joints = {int(key): JointEstimate.from_dict(value) for key, value in payload["joints"].items()}
    correction_path = config.paths.physics / "semantic_joint_correction.json"
    corrections = read_json(correction_path).get("corrections", []) if correction_path.is_file() else []
    state_metadata = _scene_state_metadata(spec, corrections)
    zero_positions = {
        child: state["zero_q"] for child, state in state_metadata.items()
    }
    export_urdf(
        spec, links, joints, config.paths.urdf, config.density, zero_positions,
        state_metadata)
    fallback_path = config.paths.joints / "rigid_fallback.json"
    if fallback_path.is_file():
        manifest_path = config.paths.urdf / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["rigid_fallback"] = read_json(fallback_path)
        write_json(manifest_path, manifest)
    rejection_path = config.paths.joints / "rejected_joints.json"
    if rejection_path.is_file():
        manifest_path = config.paths.urdf / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["joint_rejections"] = read_json(rejection_path)
        write_json(manifest_path, manifest)


STAGE_FUNCTIONS = {
    "segment": segment_stage, "render": render_stage, "infer-structure": infer_structure_stage,
    "build-parts": build_parts_stage, "fit-joints": fit_joints_stage,
    "physics": physics_stage, "export": export_stage,
}


def _run_named_stage(config: PipelineConfig, name: str) -> None:
    print(f"[gram-stage] {name} started", flush=True)
    STAGE_FUNCTIONS[name](config)
    print(f"[gram-stage] {name} completed", flush=True)


def run_range(config: PipelineConfig, start: str, end: str) -> None:
    first, last = STAGES.index(start), STAGES.index(end)
    if first > last:
        raise ValueError(f"from-stage {start} follows to-stage {end}")
    for name in STAGES[first:last + 1]:
        _run_named_stage(config, name)


def _add_common(parser):
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p3sam-python", type=Path); parser.add_argument("--p3sam-checkout", type=Path); parser.add_argument("--p3sam-checkpoint", type=Path); parser.add_argument("--p3sam-sonata-dir", type=Path)
    parser.add_argument("--blender", type=Path); parser.add_argument("--mllm-model")
    parser.add_argument("--min-link-faces", type=int, default=50)
    parser.add_argument("--contact-ratio", type=float, default=.015)
    parser.add_argument("--collision-clearance", type=float, default=.002)
    parser.add_argument("--trajectory-span", type=float, default=1.0)
    parser.add_argument("--limit-step", type=float, default=2.0)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--primary-image", type=Path)
    parser.add_argument(
        "--drawer-geometry-mode",
        choices=("reconstructed-open", "procedural-closed", "auto-3d"),
        default="auto-3d",
    )
    parser.add_argument(
        "--enable-revolute-subtype-override", action="store_true",
        help="Allow geometry to replace the revolute subtype declared by infer-structure.",
    )
    parser.add_argument(
        "--enable-joint-refinement", action="store_true",
        help="Allow physics optimization to change the fitted joint axis and pivot.",
    )
    parser.add_argument(
        "--enable-spin-upper-collision-scan", action="store_true",
        help="Clamp semantic spin upper limits using visual-mesh collision scans.",
    )
    parser.add_argument(
        "--joint-selection-mode",
        choices=("conservative", "high-recall", "high-recall-v2"),
        default="conservative",
        help="Choose conservative precision or higher-recall joint topology inference.",
    )
    parser.add_argument(
        "--enable-spin-axis-guard", action="store_true",
        help="Reject spin joints whose contact geometry does not support the fitted axis.",
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="gram")
    commands = parser.add_subparsers(dest="command", required=True)
    for stage in (*STAGES, "all"):
        sub = commands.add_parser(stage); _add_common(sub)
        if stage == "all":
            sub.add_argument("--from-stage", choices=STAGES, default="segment")
            sub.add_argument("--to-stage", choices=STAGES, default="export")
            sub.add_argument("--saved-primitives", type=Path)
            sub.add_argument("--saved-kinematics", type=Path)
    return parser


def _seed_saved_artifacts(args, paths: RunPaths) -> None:
    if getattr(args, "saved_primitives", None):
        source = args.saved_primitives
        source = source.parent if source.is_file() else source
        _require(source / "manifest.json", "--saved-primitives must contain manifest.json")
        paths.primitives.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "working_mesh.glb", "face_primitive_ids.npy"):
            shutil.copy2(_require(source / name, f"saved primitives missing {name}"), paths.primitives / name)
    if getattr(args, "saved_kinematics", None):
        paths.kinematics.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_require(args.saved_kinematics, "--saved-kinematics file not found"), paths.kinematics / "kinematics.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = RunPaths.create(args.run_dir)
    config = PipelineConfig(args.input, paths, args.seed, args.p3sam_python, args.p3sam_checkout,
                            args.p3sam_checkpoint, args.blender, args.mllm_model, args.min_link_faces,
                            args.contact_ratio, args.collision_clearance, args.trajectory_span,
                            args.limit_step, args.density, args.source_image,
                            args.primary_image,
                            args.drawer_geometry_mode,
                            args.enable_revolute_subtype_override,
                            args.enable_joint_refinement,
                            args.p3sam_sonata_dir,
                            args.enable_spin_upper_collision_scan,
                            args.joint_selection_mode,
                            args.enable_spin_axis_guard,
                            "reasoning-only")
    serializable = asdict(config); serializable["paths"] = {key: str(value) for key, value in asdict(paths).items()}
    serializable["model_runtime"] = model_runtime_metadata(
        config.mllm_model, config.vlm_execution_mode,
    )
    serializable["input"] = str(config.input)
    for key in (
        "p3sam_python", "p3sam_checkout", "p3sam_checkpoint", "p3sam_sonata_dir", "blender",
        "source_image", "primary_image",
    ):
        serializable[key] = str(serializable[key]) if serializable[key] else None
    write_json(paths.root / "run_config.json", serializable)
    _seed_saved_artifacts(args, paths)
    if args.command == "all": run_range(config, args.from_stage, args.to_stage)
    else: _run_named_stage(config, args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
