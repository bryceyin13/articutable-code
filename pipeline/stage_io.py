import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.common import attempt_path


@dataclass(frozen=True)
class OutputFile:
    name: str
    relative_path: str


def _files(**paths):
    return {name: OutputFile(name, relative) for name, relative in paths.items()}


STAGE_OUTPUTS = {
    "reference_image": _files(image="data/reference_image.png"),
    "topview_image": _files(topview_image="data/topview_image.png"),
    "segment_anonymous_instances": _files(
        results="data/anonymous_segmentation_results.json",
        overlay="data/anonymous_segmentation_overlay.png",
    ),
    "select_items": _files(
        manifest="selected_items/manifest.json",
        overlay="selected_items/matching_overlay.svg",
    ),
    "condition_images": _files(
        manifest="data/condition_manifest.json",
        recognition="data/recognition.json",
    ),
    "blueprint": _files(blueprint="data/blueprint.json"),
    "segment_instances": _files(
        results="data/segmentation_results.json",
        overlay="data/segmentation_overlay.png",
        blueprint="data/blueprint.json",
    ),
    "image_to_3d": _files(manifest="data/image_to_3d_manifest.json"),
    "gram": _files(manifest="data/articulation_manifest.json"),
    "articulation": _files(manifest="data/articulation_manifest.json"),
    "mesh_orientation": _files(
        manifest="outputs/two_camera_alignment/00_mesh_orientation_manifest.json",
    ),
    "blender_scene": _files(
        scene="outputs/blender_scene.glb",
        blend="outputs/blender_scene.blend",
        preview="outputs/blender_scene_preview.png",
        manifest="data/blender_scene_manifest.json",
        sim_manifest="data/sim_asset_manifest.json",
        sim_bundle="outputs/sim_asset_bundle.zip",
    ),
    "replace_articulated": _files(
        scene="outputs/blender_scene.usd",
        blend="outputs/blender_scene.blend",
        preview="outputs/blender_scene_preview.png",
        manifest="data/blender_scene_manifest.json",
        sim_manifest="data/sim_asset_manifest.json",
        sim_bundle="outputs/sim_asset_bundle.zip",
        rest_preview="outputs/articulation_states/front_rest.png",
        lower_preview="outputs/articulation_states/front_lower.png",
        upper_preview="outputs/articulation_states/front_upper.png",
        joint_range_preview="outputs/articulation_states/front_rest_joint_ranges.png",
        motion_video="outputs/articulation_states/front_rest_lower_upper_rest.mp4",
    ),
    "simulation": _files(
        scene="outputs/isaac_loaded_scene.usd",
        bundle="outputs/isaac_asset_package.usdz",
        metadata="outputs/simulation_metadata.json",
        video="outputs/simulation.mp4",
        preview="outputs/simulation_preview.png",
    ),
}

MERGED_SEGMENTATION_OUTPUTS = {
    "segment_anonymous_instances": _files(
        results="front/data/anonymous_segmentation_results.json",
        overlay="front/data/anonymous_segmentation_overlay.png",
        topview_results="top/data/anonymous_topview_segmentation_results.json",
        topview_overlay="top/data/anonymous_topview_segmentation_overlay.png",
    ),
    "segment_instances": _files(
        results="front/data/segmentation_results.json",
        overlay="front/data/segmentation_overlay.png",
        blueprint="front/data/blueprint.json",
        topview_results="top/data/topview_segmentation_results.json",
        topview_overlay="top/data/topview_segmentation_overlay.png",
    ),
}

REAL_IMAGE_OUTPUTS = {
    "reference_image": _files(
        image="data/reference_image.png",
        overlay="data/reference_projection_overlay.png",
    ),
    "segment_anonymous_instances": MERGED_SEGMENTATION_OUTPUTS["segment_anonymous_instances"],
    "select_items": STAGE_OUTPUTS["select_items"],
    "condition_images": STAGE_OUTPUTS["condition_images"],
    "segment_instances": MERGED_SEGMENTATION_OUTPUTS["segment_instances"],
}


def stage_outputs(stage):
    return REAL_IMAGE_OUTPUTS.get(stage, STAGE_OUTPUTS[stage])


def dependency_file(attempt, dependency, name):
    try:
        attempt_id = attempt.upstream_best[dependency]
    except KeyError:
        raise ValueError(f"dependency was not recorded: {dependency}") from None
    matches = list(attempt.final_root.parents[1].glob(f"[0-9][0-9]_{dependency}"))
    if len(matches) != 1:
        raise ValueError(f"cannot locate dependency stage: {dependency}")
    root = matches[0] / "attempts" / attempt_id
    metadata = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    try:
        relative = metadata["artifacts"][name]
    except KeyError:
        raise ValueError(f"dependency file is unavailable: {dependency}.{name}") from None
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def completed_files(attempt, outputs):
    completed = {}
    for name, output in outputs.items():
        target = attempt_path(attempt, output.relative_path)
        if not target.is_file():
            raise FileNotFoundError(f"missing stage output: {output.relative_path}")
        completed[name] = output.relative_path
    return completed
