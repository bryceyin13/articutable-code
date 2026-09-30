#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import log_step, path, read_json, write_json


STAGE = "blender_scene"


def rebase_alignment_asset_paths(document, temporary_root, final_root):
    """Replace attempt-local asset paths after the atomic attempt rename."""
    temporary_root = Path(temporary_root).resolve()
    final_root = Path(final_root).resolve()
    for item in document.get("objects", {}).values():
        for key in ("asset", "geometry_asset"):
            value = item.get(key)
            if not isinstance(value, str):
                continue
            try:
                relative = Path(value).relative_to(temporary_root)
            except ValueError:
                continue
            item[key] = str(final_root / relative)
    return document


def finalize_alignment_asset_paths(attempt):
    manifest = read_json(attempt.temp_root / "data/blender_scene_manifest.json")
    alignment = attempt.temp_root / manifest["alignment"]
    document = rebase_alignment_asset_paths(
        read_json(alignment), attempt.temp_root, attempt.final_root)
    write_json(alignment, document)


def object_ids_from_args(values):
    if values:
        return values
    env = os.environ.get("BLENDER_OBJECT_IDS")
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    blueprint = read_json(path("data/blueprint.json"))
    return [blueprint["table"]["object_id"]] + [
        item["object_id"] for item in blueprint["objects"]
    ]


def _artifact(context, attempt, dependency, name):
    try:
        return dependency_file(attempt, dependency, name)
    except ValueError as error:
        if "dependency was not recorded" not in str(error):
            raise
        root = context.stage_best(dependency)
        record = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
        return root / record["artifacts"][name]


def _blueprint_artifact(context, attempt):
    try:
        return _artifact(context, attempt, "segment_instances", "blueprint")
    except ValueError as error:
        if "dependency artifact is not published" not in str(error):
            raise
        return _artifact(context, attempt, "blueprint", "blueprint")


def _top_segmentation_artifact(context, attempt):
    return _artifact(context, attempt, "segment_instances", "topview_results")


def _image_to_3d_manifest(context, attempt):
    configured = os.environ.get("BLENDER_IMAGE_TO_3D_MANIFEST")
    if not configured:
        return _artifact(context, attempt, "image_to_3d", "manifest")
    manifest = Path(configured).expanduser().resolve(strict=True)
    image_to_3d_root = context.stage_best("image_to_3d")
    if image_to_3d_root not in manifest.parents:
        raise ValueError(
            "BLENDER_IMAGE_TO_3D_MANIFEST must be inside the best image_to_3d attempt"
        )
    return manifest


def _articulation_manifest(context, attempt):
    configured = os.environ.get("BLENDER_ARTICULATION_MANIFEST")
    if not configured:
        default = _artifact(context, attempt, "gram", "manifest")
        primary = default.parent / "articulation_manifest_primary.json"
        return primary if primary.is_file() else default
    manifest = Path(configured).expanduser().resolve(strict=True)
    gram_root = context.stage_best("gram")
    if gram_root not in manifest.parents:
        raise ValueError(
            "BLENDER_ARTICULATION_MANIFEST must be inside the best gram attempt"
        )
    return manifest


def _python_executable():
    configured = os.environ.get("VGGT_PYTHON")
    if configured:
        executable = Path(configured).expanduser()
        if executable.is_file():
            return str(executable.resolve())
        found = shutil.which(configured)
        if found:
            return found
        raise FileNotFoundError(f"VGGT_PYTHON is not executable: {configured}")
    env_name = os.environ.get("VGGT_CONDA_ENV")
    if not env_name:
        raise RuntimeError("set VGGT_PYTHON or VGGT_CONDA_ENV")
    for root in (Path.home() / ".conda/envs", Path.home() / "miniconda3/envs"):
        executable = root / env_name / "bin/python"
        if executable.is_file():
            return str(executable.resolve())
    raise FileNotFoundError(f"cannot find Python for VGGT_CONDA_ENV={env_name}")


def _export_final_scene():
    return os.environ.get("EXPORT_FINAL_SCENE", "").lower() in {
        "1", "true", "yes",
    }


def build_command(context, attempt, orientation_only=False):
    metadata = read_json(context.run_json)
    asset_source = metadata.get("assembly_asset_source", "raw")
    anonymous_results_path = _artifact(
        context, attempt, "segment_anonymous_instances", "results",
    )
    command = [
        _python_executable(), "-m", "psgsr.registration",
        "--front-image", str(_artifact(context, attempt, "reference_image", "image")),
        "--top-image", str(_artifact(context, attempt, "topview_image", "topview_image")),
        "--front-segmentation", str(_artifact(context, attempt, "segment_instances", "results")),
        "--top-segmentation", str(_top_segmentation_artifact(context, attempt)),
        "--blueprint", str(_blueprint_artifact(context, attempt)),
        "--image-to-3d-manifest", str(_image_to_3d_manifest(context, attempt)),
        "--output-dir", str(attempt.temp_root),
        "--asset-source", asset_source,
        "--top-object-optimization-mode", "independent_parallel",
        "--front-object-optimization-mode", "independent_parallel",
        "--front-yaw-search", "bidirectional",
        "--bbox-loss-weight", os.environ.get("BBOX_LOSS_WEIGHT", "0.1"),
        "--bbox-center-weight", "1.0",
        "--front-edge-parameter-scope", os.environ.get(
            "FRONT_EDGE_PARAMETER_SCOPE", "all",
        ),
        "--refinement-rgb-loss-weight", os.environ.get(
            "REFINEMENT_RGB_LOSS_WEIGHT", "0.0",
        ),
        "--early-stop-patience", "30",
        "--early-stop-min-delta", "1e-4",
    ]
    if anonymous_results_path.is_file():
        anonymous_results = read_json(anonymous_results_path)
        if anonymous_results.get("tabletop_mask_path"):
            tabletop_mask = (
                anonymous_results_path.parents[1]
                / anonymous_results["tabletop_mask_path"]
            )
            command.extend(("--front-tabletop-mask", str(tabletop_mask)))
    support_pose_enabled = os.environ.get(
        "SUPPORT_POSE_REFINEMENT", "1",
    ).lower() in {"1", "true", "yes"}
    command.append(
        "--support-pose-refinement"
        if support_pose_enabled else "--no-support-pose-refinement"
    )
    command.extend((
        "--articulation-manifest",
        str(_articulation_manifest(context, attempt)),
    ))
    if orientation_only:
        command.append("--stop-after-mesh-orientation")
    else:
        orientation_manifest = _artifact(
            context, attempt, "mesh_orientation", "manifest",
        )
        orientation = read_json(orientation_manifest)
        command.extend((
            "--mesh-orientation-manifest", str(orientation_manifest),
            "--front-vggt-dir", orientation["vggt"]["front"],
            "--top-vggt-dir", orientation["vggt"]["top"],
        ))
    if _export_final_scene():
        command.append("--export-final-scene")
    return command


def run(context=None, attempt=None, **_):
    if context is None or attempt is None:
        raise ValueError(
            "standalone use requires psgsr.registration"
        )
    command = build_command(context, attempt)
    log_step(STAGE, "running independent two-camera alignment and assembly")
    result = subprocess.run(
        command, cwd=context.project_root, env=os.environ.copy(), text=True
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    finalize_alignment_asset_paths(attempt)
    contract = STAGE_OUTPUTS[STAGE]
    if _export_final_scene():
        return completed_files(attempt, contract)
    return completed_files(
        attempt,
        {
            key: contract[key]
            for key in ("preview", "manifest")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="arguments for psgsr.registration",
    )
    values = parser.parse_args().args
    raise SystemExit(subprocess.call([
        _python_executable(), "-m", "psgsr.registration", *values,
    ]))
