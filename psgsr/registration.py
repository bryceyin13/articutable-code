#!/usr/bin/env python3
"""Standalone and pipeline entry point for two-camera scene assembly."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.render_settings import (
    disabled_specular_materials,
    specular_reflections_enabled,
)
from gram.urdf_assets import namespace_urdf_assets, orient_urdf_root, urdf_root_pose


PROJECT = Path(__file__).resolve().parents[1]


def _replace_link(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _normalize_segmentation(source, destination):
    source = Path(source).resolve(strict=True)
    document = _read_json(source)
    source_root = source.parents[1]
    for item in document["objects"]:
        mask = Path(item["mask_path"])
        mask = mask if mask.is_absolute() else source_root / mask
        item["mask_path"] = str(mask.resolve(strict=True))
    if document.get("tabletop_mask_path"):
        mask = Path(document["tabletop_mask_path"])
        mask = mask if mask.is_absolute() else source_root / mask
        document["tabletop_mask_path"] = str(mask.resolve(strict=True))
    _write_json(destination, document)


def _normalize_mesh_manifest(source, destination):
    source = Path(source).resolve(strict=True)
    document = _read_json(source)
    source_root = source.parents[1]
    for item in document["items"]:
        mesh = Path(item["mesh"])
        mesh = mesh if mesh.is_absolute() else source_root / mesh
        item["mesh"] = str(mesh.resolve(strict=True))
    _write_json(destination, document)


def _normalize_articulation_manifest(source, destination):
    source = Path(source).resolve(strict=True)
    document = _read_json(source)
    source_root = source.parents[1]
    for item in document["items"]:
        for field in ("package", "urdf"):
            value = Path(item[field])
            item[field] = str(
                (value if value.is_absolute() else source_root / value).resolve(strict=True)
            )
        if item.get("rest_mesh"):
            value = Path(item["rest_mesh"])
            item["rest_mesh"] = str(
                (value if value.is_absolute() else source_root / value).resolve()
            )
    _write_json(destination, document)


def _normalize_mesh_orientation_manifest(source, destination):
    source = Path(source).resolve(strict=True)
    document = _read_json(source)
    source_root = source.parents[1]
    for item in document["items"]:
        for field in ("asset", "geometry_asset"):
            value = Path(item["spec"][field])
            item["spec"][field] = str(
                (value if value.is_absolute() else source_root / value).resolve(strict=True)
            )
    for view in ("front", "top"):
        value = Path(document["vggt"][view])
        document["vggt"][view] = str(
            (value if value.is_absolute() else source_root / value).resolve(strict=True)
        )
    _write_json(destination, document)


def prepare_workspace(args):
    output_root = Path(args.output_dir).resolve()
    workspace = output_root / ".two_camera_workspace"
    data = workspace / "data"
    outputs = workspace / "outputs"
    data.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    _replace_link(args.front_image, data / "reference_image.png")
    _replace_link(args.top_image, data / "topview_image.png")
    front_tabletop_mask = getattr(args, "front_tabletop_mask", None)
    if front_tabletop_mask is not None:
        _replace_link(front_tabletop_mask, data / "front_tabletop_mask.png")
    _replace_link(args.blueprint, data / "blueprint.json")
    _normalize_segmentation(
        args.front_segmentation, data / "segmentation_results.json"
    )
    _normalize_segmentation(
        args.top_segmentation, data / "topview_segmentation_results.json"
    )
    _normalize_mesh_manifest(
        args.image_to_3d_manifest, data / "image_to_3d_manifest.json"
    )
    articulation_manifest = getattr(args, "articulation_manifest", None)
    if isinstance(articulation_manifest, (str, os.PathLike)):
        _normalize_articulation_manifest(
            articulation_manifest, data / "articulation_manifest.json"
        )
    mesh_orientation_manifest = getattr(args, "mesh_orientation_manifest", None)
    if isinstance(mesh_orientation_manifest, (str, os.PathLike)):
        _normalize_mesh_orientation_manifest(
            mesh_orientation_manifest, data / "mesh_orientation_manifest.json"
        )
    return output_root, workspace


def prepare_vggt(args, workspace):
    front_output = workspace / "outputs/vggt_front_only"
    top_output = workspace / "outputs/vggt_top_only"
    supplied = (args.front_vggt_dir, args.top_vggt_dir)
    if any(supplied):
        if not all(supplied):
            raise ValueError("--front-vggt-dir and --top-vggt-dir must be provided together")
        _replace_link(args.front_vggt_dir, front_output)
        _replace_link(args.top_vggt_dir, top_output)
        return

    from psgsr.vggt import run as run_vggt

    common = {
        "device": args.vggt_device,
        "confidence_percentile": args.vggt_confidence_percentile,
        "checkpoint": args.vggt_checkpoint,
        "vggt_root": args.vggt_root,
    }
    run_vggt(
        images=[workspace / "data/reference_image.png"],
        output_dir=front_output,
        **common,
    )
    run_vggt(
        images=[workspace / "data/topview_image.png"],
        output_dir=top_output,
        **common,
    )
def _manifest(
    output_root, final_document, articulated_object_blends=(),
    root_offsets=None, final_stage="05_front_refined", final_scene_format="glb",
):
    root_offsets = root_offsets or {}
    items = [{
        "object_id": "table_0",
        "source_asset": final_document["table"]["mesh"],
        "location_m": [0.0, 0.0, 0.0],
        "yaw_deg": 0.0,
        "scale_xyz": final_document["table"]["scale_xyz"],
        "scale_factor": None,
    }]
    for object_id, item in final_document["objects"].items():
        scale_xyz = item.get("scale_xyz", [item["uniform_scale"]] * 3)
        entry = {
            "object_id": object_id,
            "source_asset": item["asset"],
            "location_m": item["translation_world_m"],
            "yaw_deg": item["yaw_deg"],
            "roll_x_deg": item.get("roll_x_deg", 0.0),
            "roll_y_deg": item.get("roll_y_deg", 0.0),
            "scale_xyz": scale_xyz,
            "scale_factor": item.get("uniform_scale"),
        }
        if object_id in root_offsets:
            entry["urdf_root_offset_local_m"] = root_offsets[object_id]
        if item.get("asset_orientation_matrix") is not None:
            entry["asset_orientation_matrix"] = item["asset_orientation_matrix"]
        items.append(entry)
    joint_items = [
        {"object_id": object_id, "enabled": True}
        for object_id, item in final_document["objects"].items()
        if item.get("asset_type") == "articulated_urdf"
    ]
    manifest = {
        "output_blend": "outputs/blender_scene.blend",
        "output_preview": "outputs/blender_scene_preview.png",
        "alignment": f"outputs/two_camera_alignment/{final_stage}.json",
        "camera": final_document["cameras"],
        "items": items,
        "joint_visualization": {"enabled": bool(joint_items), "items": joint_items},
        "articulated_object_blends": list(articulated_object_blends),
        "render_settings": {
            "specular_reflections": specular_reflections_enabled(),
            "specular_disabled_materials": list(disabled_specular_materials()),
        },
    }
    manifest[f"output_{final_scene_format}"] = (
        f"outputs/blender_scene.{final_scene_format}"
    )
    return manifest


def _publish_articulated_object_blends(output_root, articulated, blender):
    if not blender:
        return []
    results = []
    script = PROJECT / "gram/visualize_urdf_joint.py"
    for object_id, package in articulated.items():
        destination = output_root / "outputs/articulated_objects" / object_id
        destination.mkdir(parents=True, exist_ok=True)
        blend = destination / "joint_preview.blend"
        log = destination / "blender.log"
        try:
            with log.open("w", encoding="utf-8") as output:
                subprocess.run([
                    str(blender), "--background", "--python-exit-code", "1",
                    "--python", str(script), "--",
                    "--urdf", str(package / "model.urdf"), "--mesh-root", str(package),
                    "--output", str(blend), "--preserve-materials", "--no-pose-exports",
                ], check=True, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Blender joint preview failed; see {log}\n{log.read_text()[-4000:]}"
            ) from error
        results.append({
            "object_id": object_id,
            "blend": str(blend.relative_to(output_root)),
        })
    return results


def _gram_python():
    configured = os.environ.get("GRAM_PYTHON")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
        found = shutil.which(configured)
        if found:
            return Path(found)
        raise FileNotFoundError(f"GRAM_PYTHON is not executable: {configured}")
    environment = os.environ.get("GRAM_CONDA_ENV")
    if not environment:
        raise RuntimeError("set GRAM_PYTHON or GRAM_CONDA_ENV")
    for root in (
        Path.home() / "anaconda3/envs",
        Path.home() / ".conda/envs",
        Path.home() / "miniconda3/envs",
    ):
        executable = root / environment / "bin/python"
        if executable.is_file():
            return executable.resolve()
    raise FileNotFoundError(
        f"cannot find Python for GRAM_CONDA_ENV={environment}"
    )


def _add_scene_joint_ranges(manifest, alignment, sim_export):
    subprocess.run([
        str(_gram_python()),
        "-m", "gram.scene_joint_ranges",
        "--manifest", str(manifest),
        "--alignment", str(alignment),
        "--sim-export", str(sim_export),
        "--clearance-ratio", "0.002",
    ], check=True, cwd=PROJECT)


def publish_outputs(
    output_root, alignment_root, articulation_manifest=None, blender=None,
    final_stage="05_front_refined", export_final_scene=True,
    sim_export_mode="full", final_scene_format="glb",
):
    if sim_export_mode not in {"full", "isaac", "off"}:
        raise ValueError("sim_export_mode must be full, isaac, or off")
    if final_scene_format not in {"glb", "usd"}:
        raise ValueError("final_scene_format must be glb or usd")
    final_render = alignment_root / f"{final_stage}_render"
    sources = {
        output_root / "outputs/blender_scene_preview.png": final_render / "front.png",
    }
    if export_final_scene:
        sources.update({
            output_root / f"outputs/blender_scene.{final_scene_format}": (
                final_render / f"final_scene.{final_scene_format}"
            ),
            output_root / "outputs/blender_scene.blend": final_render / "final_scene.blend",
        })
    for destination, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"two-camera assembly did not produce {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if export_final_scene and final_scene_format == "usd":
        source_textures = final_render / "textures"
        if source_textures.is_dir():
            destination_textures = output_root / "outputs/textures"
            if destination_textures.exists():
                shutil.rmtree(destination_textures)
            shutil.copytree(source_textures, destination_textures)
    final_document = _read_json(alignment_root / f"{final_stage}.json")
    articulated = {}
    if export_final_scene and sim_export_mode != "off" and articulation_manifest:
        source = Path(articulation_manifest).resolve(strict=True)
        root = source.parents[1]
        articulated = {
            item["object_id"]: (root / item["package"]).resolve(strict=True)
            for item in _read_json(source)["items"]
            if final_document["objects"].get(item["object_id"], {}).get(
                "asset_type"
            ) == "articulated_urdf"
        }
    object_blends = (
        _publish_articulated_object_blends(output_root, articulated, blender)
        if sim_export_mode == "full" else []
    )
    root_offsets = (
        _read_json(final_render / "sim_export/articulated_root_offsets.json")
        if articulated else {}
    )
    blender_manifest = _manifest(
        output_root, final_document, object_blends,
        root_offsets=root_offsets, final_stage=final_stage,
        final_scene_format=final_scene_format,
    )
    _write_json(
        output_root / "data/blender_scene_manifest.json",
        blender_manifest,
    )
    if not export_final_scene or sim_export_mode == "off":
        return
    sim_export = output_root / "outputs/sim_export"
    if sim_export.exists():
        shutil.rmtree(sim_export)
    shutil.copytree(final_render / "sim_export", sim_export)
    objects = []
    for item in blender_manifest["items"]:
        object_id = item["object_id"]
        if object_id in articulated:
            destination = sim_export / "articulated" / object_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(articulated[object_id], destination)
            namespace_urdf_assets(destination, object_id)
            orient_urdf_root(
                destination, item.get("asset_orientation_matrix"), object_id,
            )
            objects.append({
                "object_id": object_id,
                "type": "articulated",
                "asset_format": "urdf",
                "asset": f"articulated/{object_id}/model.urdf",
                "scene_anchor_pose": {
                    "translation_m": item["location_m"],
                    "yaw_deg": item["yaw_deg"],
                    "roll_x_deg": item.get("roll_x_deg", 0.0),
                    "roll_y_deg": item.get("roll_y_deg", 0.0),
                    "scale": item["scale_xyz"],
                },
                "root_pose": urdf_root_pose(
                    item, item["urdf_root_offset_local_m"]
                ),
                "collision": "per_link_collision_mesh",
                "physics": {"mass_kg": 1.0, "static_friction": 0.8, "dynamic_friction": 0.6},
            })
            continue
        rigid = {
            "object_id": item["object_id"],
            "type": "static_rigid" if item["object_id"] == "table_0" else "dynamic_rigid",
            "pose_baked_in_asset": True,
            "source_asset": item.get("source_asset"),
            "collision": "convex_hull",
            "physics": {
                "mass_kg": 0.0 if item["object_id"] == "table_0" else 0.2,
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
            },
        }
        if sim_export_mode == "full":
            rigid.update({
                "asset_glb": f"assets/{item['object_id']}.glb",
                "asset_usd": f"assets/{item['object_id']}.usd",
            })
        objects.append(rigid)
    scene_visual = {"usd": "scene_visual.usd"}
    if sim_export_mode == "full":
        scene_visual["glb"] = "scene_visual.glb"
    sim_manifest = output_root / "data/sim_asset_manifest.json"
    _write_json(sim_manifest, {
        "format": "rigid_sim_export_v1",
        "unit": "meter",
        "up_axis": "Z",
        "source_blend": "outputs/blender_scene.blend",
        "scene_visual": scene_visual,
        "objects": objects,
    })
    if articulated and any(
        item.get("joint_state") for item in final_document.get("objects", {}).values()
    ):
        _add_scene_joint_ranges(
            sim_manifest,
            alignment_root / f"{final_stage}.json",
            sim_export,
        )
    if sim_export_mode == "full":
        shutil.make_archive(
            str(output_root / "outputs/sim_asset_bundle"), "zip", sim_export,
        )


def run(args):
    articulation_manifest = getattr(args, "articulation_manifest", None)
    if not isinstance(articulation_manifest, (str, os.PathLike)):
        articulation_manifest = None
    if args.asset_source == "articulated" and not articulation_manifest:
        raise ValueError("--articulation-manifest is required for articulated assembly")
    output_root, workspace = prepare_workspace(args)
    prepare_vggt(args, workspace)
    os.environ["TABLETOP_ALIGNMENT_ROOT"] = str(workspace)
    from psgsr.core import parse_args as parse_core_args, run as run_core
    from psgsr import core

    alignment_root = output_root / "outputs/two_camera_alignment"
    core_args = parse_core_args([
        "--output-dir", str(alignment_root),
        "--asset-source", args.asset_source,
        "--top-object-optimization-mode", args.top_object_optimization_mode,
        "--front-object-optimization-mode", args.front_object_optimization_mode,
        "--front-refine-rounds", str(args.front_refine_rounds),
        "--front-yaw-search", args.front_yaw_search,
        "--minima-root", str(args.minima_root),
        "--minima-device", args.minima_device,
        "--front-yaw-rgb-weight", str(args.front_yaw_rgb_weight),
        "--front-yaw-top-iou-ambiguity-threshold",
        str(args.front_yaw_top_iou_ambiguity_threshold),
        "--front-yaw-edge-override-margin", str(args.front_yaw_edge_override_margin),
        "--bbox-loss-weight", str(args.bbox_loss_weight),
        "--bbox-center-weight", str(args.bbox_center_weight),
        "--front-edge-loss-weight", str(args.front_edge_loss_weight),
        "--front-edge-parameter-scope", args.front_edge_parameter_scope,
        "--refinement-rgb-loss-weight", str(args.refinement_rgb_loss_weight),
        "--top-yaw-mask-iou-loss-weight", str(args.top_yaw_mask_iou_loss_weight),
        "--top-internal-edge-match-mode", args.top_internal_edge_match_mode,
        "--top-internal-orientation-weight", str(args.top_internal_orientation_weight),
        "--front-internal-orientation-weight", str(args.front_internal_orientation_weight),
        "--top-model-edge-method", args.top_model_edge_method,
        "--front-model-edge-method", args.front_model_edge_method,
        "--edge-detector", args.edge_detector,
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--front-refine-second-round-resolution",
        str(args.front_refine_second_round_resolution),
        (
            "--support-pose-refinement"
            if args.support_pose_refinement else
            "--no-support-pose-refinement"
        ),
    ] + (["--export-final-scene"] if args.export_final_scene else [])
    + (["--stop-after-mesh-orientation"] if args.stop_after_mesh_orientation else [])
    + (["--stop-after-top-refined"] if args.stop_after_top_refined else []))
    try:
        run_core(core_args)
    finally:
        if core._blender_worker is not None:
            core._blender_worker.close()
            core._blender_worker = None
    if args.stop_after_mesh_orientation:
        return alignment_root / "00_mesh_orientation_manifest.json"
    publish_outputs(
        output_root, alignment_root, articulation_manifest,
        core.blender_executable(),
        "04_top_refined" if args.stop_after_top_refined else "05_front_refined",
        export_final_scene=args.export_final_scene,
    )
    return output_root / "data/blender_scene_manifest.json"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-image", type=Path, required=True)
    parser.add_argument("--top-image", type=Path, required=True)
    parser.add_argument("--front-segmentation", type=Path, required=True)
    parser.add_argument("--front-tabletop-mask", type=Path)
    parser.add_argument("--top-segmentation", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--image-to-3d-manifest", type=Path, required=True)
    parser.add_argument("--articulation-manifest", type=Path)
    parser.add_argument("--mesh-orientation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--front-vggt-dir", type=Path)
    parser.add_argument("--top-vggt-dir", type=Path)
    parser.add_argument(
        "--vggt-root", type=Path,
        default=Path(os.environ.get("VGGT_ROOT", PROJECT.parent / "third_party/vggt")),
    )
    parser.add_argument(
        "--vggt-checkpoint", type=Path,
        default=Path(os.environ.get("VGGT_CHECKPOINT", PROJECT / "models/vggt/model.pt")),
    )
    parser.add_argument(
        "--vggt-device", default=os.environ.get("VGGT_DEVICE"),
        required="VGGT_DEVICE" not in os.environ,
    )
    parser.add_argument(
        "--minima-root", type=Path,
        default=Path(os.environ.get(
            "MINIMA_ROOT", PROJECT.parent / "third_party" / "MINIMA",
        )),
    )
    parser.add_argument(
        "--minima-device", default=os.environ.get("MINIMA_DEVICE"),
        required="MINIMA_DEVICE" not in os.environ,
    )
    parser.add_argument("--vggt-confidence-percentile", type=float, default=50.0)
    parser.add_argument(
        "--asset-source", choices=("raw", "articulated"), default="raw"
    )
    parser.add_argument(
        "--top-object-optimization-mode",
        choices=("joint", "independent_parallel"), default="independent_parallel",
    )
    parser.add_argument(
        "--front-object-optimization-mode",
        choices=("joint", "independent_parallel"), default="independent_parallel",
    )
    parser.add_argument(
        "--front-yaw-search", choices=("single", "bidirectional"),
        default="bidirectional",
    )
    parser.add_argument(
        "--front-yaw-rgb-weight", type=float,
        default=float(os.environ.get("FRONT_YAW_RGB_WEIGHT", "0")),
    )
    parser.add_argument(
        "--front-yaw-top-iou-ambiguity-threshold", type=float,
        default=float(os.environ.get("FRONT_YAW_TOP_IOU_AMBIGUITY_THRESHOLD", "0.02")),
    )
    parser.add_argument(
        "--front-yaw-edge-override-margin", type=float,
        default=float(os.environ.get("FRONT_YAW_EDGE_OVERRIDE_MARGIN", "0.01")),
    )
    parser.add_argument(
        "--bbox-loss-weight", type=float,
        default=float(os.environ.get("BBOX_LOSS_WEIGHT", "0.1")),
        help="Projected bbox loss weight in both refinement stages; set 0 to restore the old loss.",
    )
    parser.add_argument(
        "--bbox-center-weight", type=float,
        default=float(os.environ.get("BBOX_CENTER_WEIGHT", "1.0")),
        help="Center term inside bbox loss; set 0 to retain size-only behavior.",
    )
    parser.add_argument(
        "--front-edge-loss-weight", type=float,
        default=float(os.environ.get("FRONT_EDGE_LOSS_WEIGHT", "0.0")),
    )
    parser.add_argument(
        "--front-edge-parameter-scope", choices=("all", "yaw_only"),
        default=os.environ.get("FRONT_EDGE_PARAMETER_SCOPE", "all"),
    )
    parser.add_argument(
        "--refinement-rgb-loss-weight", type=float,
        default=float(os.environ.get("REFINEMENT_RGB_LOSS_WEIGHT", "0.0")),
    )
    parser.add_argument(
        "--top-yaw-mask-iou-loss-weight", type=float,
        default=float(os.environ.get("TOP_YAW_MASK_IOU_LOSS_WEIGHT", "1.0")),
        help="Weight of (1 - mask IoU) in weighted four-way yaw selection.",
    )
    parser.add_argument(
        "--top-internal-edge-match-mode",
        choices=("chamfer", "partial_hausdorff", "centroid_offset"),
        default=os.environ.get(
            "TOP_INTERNAL_EDGE_MATCH_MODE", "chamfer",
        ),
        help="Use Chamfer (default), partial Hausdorff, or centroid offsets for top yaw.",
    )
    parser.add_argument(
        "--top-internal-orientation-weight", type=float,
        default=float(os.environ.get("TOP_INTERNAL_ORIENTATION_WEIGHT", "0.0")),
    )
    parser.add_argument(
        "--front-internal-orientation-weight", type=float,
        default=float(os.environ.get("FRONT_INTERNAL_ORIENTATION_WEIGHT", "0.0")),
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
        "--edge-detector",
        choices=("canny", "edge_drawing", "teed", "ffmpeg_edgedetect", "geometry"),
        default=os.environ.get("EDGE_DETECTOR", "teed"),
        help="Reference-image edge detector used by stage-2/3 internal-edge losses.",
    )
    parser.add_argument(
        "--early-stop-patience", type=int,
        default=int(os.environ.get("EARLY_STOP_PATIENCE", "30")),
        help="Non-improving GPU steps before stopping; set 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float,
        default=float(os.environ.get("EARLY_STOP_MIN_DELTA", "1e-4")),
        help="Minimum loss decrease that resets early-stopping patience.",
    )
    parser.add_argument(
        "--front-refine-second-round-resolution", type=int,
        default=int(os.environ.get("FRONT_REFINE_SECOND_ROUND_RESOLUTION", "1024")),
        help="Maximum image dimension for the second front-refinement round.",
    )
    parser.add_argument(
        "--front-refine-rounds", type=int,
        default=int(os.environ.get("FRONT_REFINE_ROUNDS", "1")),
        help="Number of front-refinement rounds; set 1 to disable the second round.",
    )
    parser.add_argument(
        "--stop-after-mesh-orientation", action="store_true",
        help="Publish resolved, normalized assembly assets before object placement.",
    )
    parser.add_argument(
        "--stop-after-top-refined", action="store_true",
        default=os.environ.get("STOP_AFTER_TOP_REFINED", "").lower()
        in {"1", "true", "yes"},
        help="Publish stage 2 (04_top_refined) without running stage 3 front refinement.",
    )
    parser.add_argument(
        "--export-final-scene", action="store_true",
        default=os.environ.get("EXPORT_FINAL_SCENE", "").lower()
        in {"1", "true", "yes"},
        help="Export final_scene.blend/glb and simulation assets (temporarily off by default).",
    )
    support_pose = parser.add_mutually_exclusive_group()
    support_pose.add_argument(
        "--support-pose-refinement",
        dest="support_pose_refinement", action="store_true", default=True,
    )
    support_pose.add_argument(
        "--no-support-pose-refinement",
        dest="support_pose_refinement", action="store_false",
    )
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
