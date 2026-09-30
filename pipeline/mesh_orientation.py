#!/usr/bin/env python3
"""Resolve and orient the exact assets consumed by scene assembly."""

import os
import shutil
import subprocess
from pathlib import Path

from pipeline import scene_assembly as assemble_scene_blender
from pipeline.stage_io import STAGE_OUTPUTS, completed_files
from pipeline.common import log_step, read_json, write_json


STAGE = "mesh_orientation"


def _rebase(value, temporary_root, final_root):
    path = Path(value)
    try:
        relative = path.relative_to(temporary_root)
    except ValueError:
        return str(path)
    return str(final_root / relative)


def finalize_manifest_paths(attempt):
    path = (
        attempt.temp_root
        / "outputs/two_camera_alignment/00_mesh_orientation_manifest.json"
    )
    document = read_json(path)
    for item in document["items"]:
        for field in ("asset", "geometry_asset"):
            item["spec"][field] = _rebase(
                item["spec"][field], attempt.temp_root, attempt.final_root,
            )
        if item.get("top_template") is not None:
            item["top_template"]["path"] = _rebase(
                item["top_template"]["path"],
                attempt.temp_root, attempt.final_root,
            )
    for view in ("front", "top"):
        document["vggt"][view] = _rebase(
            document["vggt"][view], attempt.temp_root, attempt.final_root,
        )
    write_json(path, document)


def reusable_vggt_args(context, attempt):
    """Reuse VGGT when inputs are identical; rebuild attempt-local oriented meshes."""
    try:
        root = context.stage_best(STAGE)
        record = read_json(root / "attempt.json")
        manifest = root / record["artifacts"]["manifest"]
        document = read_json(manifest)
        front = Path(document["vggt"]["front"])
        top = Path(document["vggt"]["top"])
        if record.get("upstream_best") != attempt.upstream_best:
            return []
        if not (front.is_dir() and top.is_dir()):
            return []
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return []
    return [
        "--front-vggt-dir", str(front),
        "--top-vggt-dir", str(top),
    ]


def materialize_reused_vggt(attempt):
    """Replace reused VGGT symlinks with attempt-local hardlinks or copies."""
    outputs = attempt.temp_root / ".two_camera_workspace/outputs"

    def link_or_copy(source, destination):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    for name in ("vggt_front_only", "vggt_top_only"):
        destination = outputs / name
        if not destination.is_symlink():
            continue
        source = destination.resolve(strict=True)
        materialized = outputs / f".{name}.materialized"
        shutil.copytree(source, materialized, copy_function=link_or_copy)
        destination.unlink()
        materialized.rename(destination)


def run(context=None, attempt=None, **_):
    if context is None or attempt is None:
        raise ValueError("mesh_orientation must run through the pipeline")
    command = assemble_scene_blender.build_command(
        context, attempt, orientation_only=True,
    )
    vggt_args = reusable_vggt_args(context, attempt)
    command.extend(vggt_args)
    log_step(STAGE, "resolving final assembly assets and normalizing mesh axes")
    if vggt_args:
        log_step(
            STAGE,
            "reusing VGGT from the previous input-equivalent attempt; "
            "rewriting oriented meshes locally",
        )
    environment = os.environ.copy()
    environment["DIRECTION_LOSS_BACKEND"] = environment.get(
        "MESH_ORIENTATION_DIRECTION_LOSS_BACKEND", "foundpose",
    )
    result = subprocess.run(
        command, cwd=context.project_root, env=environment, text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    if vggt_args:
        materialize_reused_vggt(attempt)
    finalize_manifest_paths(attempt)
    return completed_files(attempt, STAGE_OUTPUTS[STAGE])
