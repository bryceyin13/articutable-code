#!/usr/bin/env python3
"""Replace movable raw meshes after rigid two-camera assembly."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

from psgsr import registration as two_camera
from pipeline import scene_assembly as assemble_scene_blender
from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import log_step, read_json, write_json
from gram.urdf_assets import validate_rotation_matrix


STAGE = "replace_articulated"
FINAL_STAGE = "05_articulated_replaced"
ARTICULATION_VISUALIZATION_DIR = Path("outputs/articulation_states")
ARTICULATION_POSE_FILES = {
    "rest": "front_rest.png",
    "lower": "front_lower.png",
    "upper": "front_upper.png",
}
ARTICULATION_JOINT_RANGE_FILE = "front_rest_joint_ranges.png"
ARTICULATION_VIDEO_FILE = "front_rest_lower_upper_rest.mp4"


def articulation_video_enabled(environ=None):
    environment = os.environ if environ is None else environ
    value = environment.get(
        "REPLACE_ARTICULATED_VIDEO", "1",
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "REPLACE_ARTICULATED_VIDEO must be one of "
        "1/0, true/false, yes/no, or on/off"
    )


def articulation_sim_export_enabled(environ=None):
    environment = os.environ if environ is None else environ
    value = environment.get(
        "REPLACE_ARTICULATED_SIM_EXPORT", "1",
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "REPLACE_ARTICULATED_SIM_EXPORT must be one of "
        "1/0, true/false, yes/no, or on/off"
    )


def articulation_full_sim_export_enabled(environ=None):
    environment = os.environ if environ is None else environ
    value = environment.get(
        "REPLACE_ARTICULATED_FULL_SIM_EXPORT", "0",
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "REPLACE_ARTICULATED_FULL_SIM_EXPORT must be one of "
        "1/0, true/false, yes/no, or on/off"
    )


def articulation_final_scene_format(environ=None):
    environment = os.environ if environ is None else environ
    value = environment.get(
        "REPLACE_ARTICULATED_FINAL_SCENE_FORMAT", "usd",
    ).strip().lower()
    if value not in {"usd", "glb"}:
        raise ValueError(
            "REPLACE_ARTICULATED_FINAL_SCENE_FORMAT must be usd or glb"
        )
    return value


def articulation_manifest_path(attempt):
    default = dependency_file(attempt, "gram", "manifest")
    primary = default.with_name("articulation_manifest_primary.json")
    return primary if primary.is_file() else default


def baked_orientation_matrix(item):
    report = item.get("mesh_orientation") or {}
    if not report:
        return None
    if report.get("matrix") is not None:
        matrix = validate_rotation_matrix(report["matrix"])
    else:
        up_axis = report.get("selected_up_axis", "+Z")
        matrices = {
            "+X": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
            "-X": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            "+Y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "-Y": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            "+Z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "-Z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        }
        if up_axis not in matrices:
            raise ValueError(f"unknown baked mesh up axis: {up_axis}")
        matrix = matrices[up_axis]
        local = report.get("local_xy_axis_alignment") or {}
        if local.get("applied"):
            angle = math.radians(float(local["correction_deg"]))
            cosine, sine = math.cos(angle), math.sin(angle)
            rotation = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
            matrix = [
                [
                    sum(
                        rotation[row][axis] * matrix[axis][column]
                        for axis in range(3)
                    )
                    for column in range(3)
                ]
                for row in range(3)
            ]
        matrix = validate_rotation_matrix(matrix)
    if all(
        abs(matrix[row][column] - (row == column)) <= 1e-12
        for row in range(3) for column in range(3)
    ):
        return None
    return matrix


def gram_scale_compensation(attempt_root, record):
    if record.get("method") != "gram":
        return None
    rest_mesh = record.get("rest_mesh")
    if record.get("requires_open_state") and rest_mesh:
        rest_path = Path(rest_mesh)
        if not rest_path.is_absolute():
            rest_path = Path(attempt_root) / rest_path
        if rest_path.is_file():
            return None
    frame = read_json(
        Path(attempt_root) / "raw/gram"
        / Path(record.get("package", record["object_id"])).name
        / "01_primitives/manifest.json"
    )["frame"]
    source_to_working_scale = float(frame["scale"])
    if not math.isfinite(source_to_working_scale) or source_to_working_scale <= 0:
        raise ValueError(
            f"invalid gram normalization scale for {record['object_id']}"
        )
    return source_to_working_scale


def replacement_document(
    raw_document, articulation_manifest_path, video_enabled=True,
):
    document = dict(raw_document)
    document["objects"] = {
        object_id: dict(item)
        for object_id, item in raw_document["objects"].items()
    }
    manifest_path = Path(articulation_manifest_path).resolve(strict=True)
    package_root = manifest_path.parents[1]
    replaced = []
    for record in read_json(manifest_path).get("items", []):
        object_id = record["object_id"]
        if object_id not in document["objects"]:
            raise ValueError(f"articulation manifest has unknown object: {object_id}")
        package = (package_root / record["package"]).resolve(strict=True)
        package_manifest = read_json(package / "manifest.json")
        joint_states = package_manifest.get("joint_states", [])
        if not joint_states and package_manifest.get("rigid_fallback"):
            continue
        item = document["objects"][object_id]
        item.update({"asset_type": "articulated_urdf", "asset": str(package)})
        source_to_working_scale = gram_scale_compensation(
            package_root, record,
        )
        if source_to_working_scale is not None:
            compensation = 1.0 / source_to_working_scale
            if "uniform_scale" in item:
                item["uniform_scale"] = float(item["uniform_scale"]) * compensation
            if "scale_xyz" in item:
                item["scale_xyz"] = [
                    float(value) * compensation for value in item["scale_xyz"]
                ]
            item["articulation_scale_compensation"] = {
                "method": "inverse_gram_working_frame_scale",
                "source_to_working_scale": source_to_working_scale,
                "factor": compensation,
            }
        orientation = baked_orientation_matrix(item)
        if orientation is not None:
            item["asset_orientation_matrix"] = orientation
        else:
            item.pop("asset_orientation_matrix", None)
        item.pop("geometry_asset", None)
        item.pop("articulation", None)
        item.pop("joint_state", None)
        if joint_states:
            item["joint_states"] = joint_states
        else:
            item.pop("joint_states", None)
        replaced.append(object_id)
    document["replacement"] = {
        "method": "gram_after_raw_rigid_assembly",
        "joint_optimization": "disabled",
        "articulated_object_ids": replaced,
        "table_support_extension": {
            "enabled": True,
            "method": "upward_raycast_from_z0",
            "contact_height_m": 0.01,
            "sample_spacing_m": 0.005,
            "fixed_edge": "negative_world_y",
            "extended_edge": "positive_world_y",
        },
        "visualization": {
            "front_pose_images": list(ARTICULATION_POSE_FILES),
            "rest_joint_ranges": True,
            "video_enabled": bool(video_enabled),
            "video_sequence": ["rest", "lower", "upper", "rest"],
        },
    }
    return document


def raw_alignment_path(attempt):
    manifest_path = dependency_file(attempt, "blender_scene", "manifest")
    manifest = read_json(manifest_path)
    return (manifest_path.parents[1] / manifest["alignment"]).resolve(strict=True)


def raw_alignment_document(attempt):
    alignment = raw_alignment_path(attempt)
    final_root = alignment.parents[1]
    temporary_root = final_root.parent / f".{final_root.name}.tmp"
    return assemble_scene_blender.rebase_alignment_asset_paths(
        read_json(alignment), temporary_root, final_root)


def alignment_workspace(attempt):
    manifest_path = dependency_file(attempt, "blender_scene", "manifest")
    workspace = manifest_path.parents[1] / ".two_camera_workspace"
    if not (workspace / "data/segmentation_results.json").is_file():
        raise FileNotFoundError(f"blender scene alignment workspace is incomplete: {workspace}")
    return workspace


def publish_articulation_visualizations(
    output_root, alignment_root, video_enabled,
):
    source_root = alignment_root / f"{FINAL_STAGE}_render/articulation_states"
    destination_root = output_root / ARTICULATION_VISUALIZATION_DIR
    files = [
        *ARTICULATION_POSE_FILES.values(), ARTICULATION_JOINT_RANGE_FILE,
    ]
    if video_enabled:
        files.append(ARTICULATION_VIDEO_FILE)
    destination_root.mkdir(parents=True, exist_ok=True)
    for filename in files:
        source = source_root / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"articulation visualization did not produce {source}"
            )
        shutil.copy2(source, destination_root / filename)

    manifest_path = output_root / "data/blender_scene_manifest.json"
    manifest = read_json(manifest_path)
    visualization = {
        "front_pose_images": {
            pose: str(ARTICULATION_VISUALIZATION_DIR / filename)
            for pose, filename in ARTICULATION_POSE_FILES.items()
        },
        "rest_joint_ranges": str(
            ARTICULATION_VISUALIZATION_DIR / ARTICULATION_JOINT_RANGE_FILE
        ),
        "video_enabled": bool(video_enabled),
        "video_sequence": ["rest", "lower", "upper", "rest"],
    }
    if video_enabled:
        visualization["video"] = str(
            ARTICULATION_VISUALIZATION_DIR / ARTICULATION_VIDEO_FILE
        )
    manifest["articulation_visualization"] = visualization
    write_json(manifest_path, manifest)


def render_and_publish(scene_json, output_root, articulation_manifest):
    from psgsr import core

    output_root = Path(output_root)
    alignment_root = output_root / "outputs/two_camera_alignment"
    document = read_json(scene_json)
    video_enabled = bool(
        document.get("replacement", {}).get("visualization", {}).get(
            "video_enabled", True,
        )
    )
    sim_export_enabled = articulation_sim_export_enabled()
    full_sim_export_enabled = articulation_full_sim_export_enabled()
    sim_export_mode = (
        "full" if full_sim_export_enabled else
        "isaac" if sim_export_enabled else "off"
    )
    final_scene_format = articulation_final_scene_format()
    os.environ["BLENDER_SIM_EXPORT_MODE"] = sim_export_mode
    os.environ["BLENDER_FINAL_SCENE_FORMAT"] = final_scene_format
    core.render_stage(
        scene_json, alignment_root / f"{FINAL_STAGE}_render", save_blend=True,
    )
    two_camera.publish_outputs(
        output_root,
        alignment_root,
        articulation_manifest,
        core.blender_executable(),
        FINAL_STAGE,
        sim_export_mode=sim_export_mode,
        final_scene_format=final_scene_format,
    )
    publish_articulation_visualizations(
        output_root, alignment_root, video_enabled,
    )


def run(context=None, attempt=None):
    if context is None or attempt is None:
        raise ValueError("replace_articulated requires a pipeline context and attempt")
    video_enabled = articulation_video_enabled()
    sim_export_enabled = articulation_sim_export_enabled()
    full_sim_export_enabled = articulation_full_sim_export_enabled()
    final_scene_format = articulation_final_scene_format()
    articulation_manifest = articulation_manifest_path(attempt)
    document = replacement_document(
        raw_alignment_document(attempt), articulation_manifest,
        video_enabled=video_enabled,
    )
    alignment_root = attempt.temp_root / "outputs/two_camera_alignment"
    replacement_path = alignment_root / f"{FINAL_STAGE}.json"
    write_json(replacement_path, document)
    log_step(
        STAGE,
        "replacing movable raw meshes without joint optimization; "
        f"articulation video {'enabled' if video_enabled else 'disabled'}; "
        "simulation export "
        f"{'full' if full_sim_export_enabled else 'isaac-lightweight' if sim_export_enabled else 'disabled'}; "
        f"final scene {final_scene_format.upper()}",
    )
    command = (
        assemble_scene_blender._python_executable(),
        str(Path(__file__).resolve()),
        "--scene-json", str(replacement_path),
        "--output-root", str(attempt.temp_root),
        "--articulation-manifest", str(articulation_manifest),
    )
    environment = os.environ.copy()
    environment["TABLETOP_ALIGNMENT_ROOT"] = str(alignment_workspace(attempt))
    environment["REPLACE_ARTICULATED_SIM_EXPORT"] = (
        "1" if sim_export_enabled else "0"
    )
    environment["REPLACE_ARTICULATED_FULL_SIM_EXPORT"] = (
        "1" if full_sim_export_enabled else "0"
    )
    environment["REPLACE_ARTICULATED_FINAL_SCENE_FORMAT"] = final_scene_format
    subprocess.run(command, check=True, cwd=context.project_root, env=environment)
    outputs = dict(STAGE_OUTPUTS[STAGE])
    if final_scene_format == "glb":
        outputs["scene"] = STAGE_OUTPUTS["blender_scene"]["scene"]
    if not video_enabled:
        outputs.pop("motion_video")
    if not sim_export_enabled:
        outputs.pop("sim_manifest")
    if not full_sim_export_enabled:
        outputs.pop("sim_bundle")
    return completed_files(attempt, outputs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--articulation-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    render_and_publish(args.scene_json, args.output_root, args.articulation_manifest)


if __name__ == "__main__":
    main()
