#!/usr/bin/env python3
import argparse
import json
import math
import os
from pipeline.common import ensure_dirs, log_step, path, write_json
from gram.repair_math import PROFILE_FIELDS, profile_axes_and_dimensions
from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import attempt_path, project_path
from pipeline.mllm import pipeline_runtime, run_mllm

AXIS_SEMANTICS = {"x": "left_to_right", "y": "front_to_back", "z": "bottom_to_top"}
FREEFORM_LOCAL_AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


def named_dimensions(size):
    return {"width_x": size[0], "depth_y": size[1], "height_z": size[2]}


def rigid_object(object_id, category, xy, yaw, size, clearance):
    return {
        "object_id": object_id,
        "category": category,
        "operability_type": "rigid_non_articulated",
        "bbox_cm": size,
        "dimensions_cm": named_dimensions(size),
        "axis_semantics": dict(AXIS_SEMANTICS),
        "placement": {
            "center_xy_norm": xy,
            "yaw_deg": yaw,
            "size_cm": size,
            "z_policy": "on_top_of_parent",
            "support_parents": ["table_0"],
            "clearance_cm": clearance,
        },
        "visibility": {"must_be_fully_visible": True, "allowed_occlusion": "none"},
        "articulation": None,
    }


def build_blueprint():
    return {
        "scene_id": "mvp_mixed_operability_tabletop",
        "scene_text": "A clean study desk with an open laptop, a mouse, a coffee mug, a closed notebook, and a small potted plant.",
        "image_prompt": "Fallback blueprint fixture; production image generation is scene-conditioned.",
        "coordinate_system": {
            "frame": "tabletop_local_frame",
            "origin": "center of tabletop",
            "x_axis": "right side of table",
            "y_axis": "back side of table",
            "z_axis": "upward",
            "xy_normalized_range": [-0.5, 0.5],
            "yaw_convention": "degrees counterclockwise around +z",
        },
        "camera": {
            "view": "front-oblique",
            "azimuth_deg": 0,
            "elevation_deg": 55,
            "distance_level": "medium",
            "require_full_table_visible": True,
        },
        "table": {
            "object_id": "table_0",
            "category": "table",
            "operability_type": "static_support",
            "size_cm": [90, 55, 72],
            "bbox_cm": [90, 55, 72],
            "dimensions_cm": named_dimensions([90, 55, 72]),
            "axis_semantics": dict(AXIS_SEMANTICS),
            "placement": {
                "center_xy_norm": [0.0, 0.0],
                "yaw_deg": 0,
                "z_policy": "floor_supported",
            },
        },
        "objects": [
            {
                "object_id": "laptop_0",
                "category": "open laptop",
                "operability_type": "articulated_operable",
                "bbox_cm": [32, 24, 22],
                "dimensions_cm": named_dimensions([32, 24, 22]),
                "axis_semantics": dict(AXIS_SEMANTICS),
                "placement": {
                    "center_xy_norm": [-0.10, 0.03],
                    "yaw_deg": -8,
                    "size_cm": [32, 24, 22],
                    "z_policy": "on_top_of_parent",
                    "support_parents": ["table_0"],
                    "clearance_cm": 3,
                },
                "visibility": {
                    "must_be_fully_visible": True,
                    "allowed_occlusion": "none",
                    "visible_parts": ["base", "screen"],
                },
                "articulation": {
                    "parts": [
                        {
                            "part_id": "base",
                            "part_name": "keyboard_base",
                            "part_role": "parent_link",
                            "geometry_type": "panel",
                            "bbox_cm": [32, 24, 1.8],
                            "dimensions_cm": {"width": 32, "length": 24, "thickness": 1.8},
                            "local_axes": {"width": [1, 0, 0], "length": [0, 1, 0], "thickness": [0, 0, 1]},
                            "repair_mode": "axis_aligned",
                        },
                        {
                            "part_id": "screen",
                            "part_name": "display_screen",
                            "part_role": "child_link",
                            "geometry_type": "panel",
                            "bbox_cm": [32, 24, 1.0],
                            "dimensions_cm": {"width": 32, "length": 24, "thickness": 1.0},
                            "local_axes": {"width": [1, 0, 0], "length": [0, 1, 0], "thickness": [0, 0, 1]},
                            "repair_mode": "axis_aligned",
                        },
                    ],
                    "joint": {
                        "joint_name": "screen_hinge",
                        "joint_type": "revolute",
                        "parent_part": "base",
                        "child_part": "screen",
                        "parent_anchor_cm": [0, 12, 0.9],
                        "child_anchor_cm": [0, -12, 0],
                        "axis": [1, 0, 0],
                        "initial_angle_deg": 105,
                        "joint_limits_deg": [0, 140],
                    },
                },
            },
            rigid_object("mouse_0", "mouse", [0.23, -0.02], 5, [11, 6, 3], 3),
            rigid_object("mug_0", "coffee mug", [0.28, 0.22], 0, [8, 8, 10], 4),
            rigid_object("notebook_0", "closed notebook", [-0.32, -0.22], 12, [21, 15, 2], 3),
            rigid_object("plant_0", "small potted plant", [-0.35, 0.25], 0, [10, 10, 16], 4),
        ],
        "scene_topology": {"relations": []},
        "layout_constraints": [
            {"type": "non_overlap", "objects": ["laptop_0", "mouse_0", "mug_0", "notebook_0", "plant_0"]},
            {"type": "articulation_visibility", "object_id": "laptop_0", "require_visible_parts": ["base", "screen"]},
        ],
    }


def build_prompt(context=None, recognition=None):
    resolve = (lambda *parts: project_path(context, *parts)) if context else path
    template = resolve("prompts/05_blueprint_generator.txt").read_text(encoding="utf-8").strip()
    manifest = json.dumps(recognition, indent=2, ensure_ascii=False) if recognition else "<provided at runtime>"
    return f"{template}\n\nRecognition manifest:\n{manifest}\n"


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("MLLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def validate_blueprint(data):
    required = ["scene_id", "scene_text", "image_prompt", "coordinate_system", "camera", "table", "objects", "scene_topology", "layout_constraints"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"blueprint missing fields: {missing}")
    _validate_english_only_json(data)
    if not _valid_bbox(data["table"].get("bbox_cm")):
        raise ValueError("table must define positive bbox_cm=[x, y, z]")
    if {"components", "parts", "articulation"} & set(data["table"]):
        raise ValueError("table must remain one complete object without components")
    _validate_named_dimensions(data["table"], "table")
    objects = data["objects"]
    ids = [obj["object_id"] for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("object_id values must be unique")
    _validate_scene_topology(data)
    for obj in objects:
        if not _valid_bbox(obj.get("bbox_cm")):
            raise ValueError(f"{obj.get('object_id')} must define positive bbox_cm=[x, y, z]")
        _validate_named_dimensions(obj, obj.get("object_id"))
        _validate_object_placement(obj)
        if obj.get("operability_type") not in {"articulated_operable", "rigid_non_articulated"}:
            raise ValueError(f"{obj.get('object_id')} has invalid operability_type: {obj.get('operability_type')}")
        if obj.get("operability_type") == "articulated_operable":
            articulation = obj.get("articulation") or {}
            parts = articulation.get("parts", [])
            if not parts:
                raise ValueError(f"{obj.get('object_id')} articulation must define parts")
            for part in parts:
                if not _valid_bbox(part.get("bbox_cm")):
                    raise ValueError(
                        f"{obj.get('object_id')} part {part.get('part_id')} must define positive bbox_cm=[x, y, z]"
                    )
                try:
                    profile_axes_and_dimensions(part)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{obj.get('object_id')} part {part.get('part_id')} has invalid geometry profile: {exc}"
                    ) from exc
            joints = articulation.get("joints")
            if joints is None:
                joints = [articulation["joint"]] if articulation.get("joint") else []
            if not joints:
                raise ValueError(f"{obj.get('object_id')} articulation must define joints")
            part_ids = {part.get("part_id") for part in parts}
            for joint in joints:
                vectors = [joint.get("parent_anchor_cm"), joint.get("child_anchor_cm"), joint.get("axis")]
                if any(
                    not isinstance(vector, list)
                    or len(vector) != 3
                    or any(not isinstance(value, (int, float)) for value in vector)
                    for vector in vectors
                ):
                    raise ValueError(
                        f"{obj.get('object_id')} joint must define numeric parent_anchor_cm, child_anchor_cm, and axis"
                    )
                if joint.get("joint_type") not in {"revolute", "prismatic"}:
                    raise ValueError(f"{obj.get('object_id')} has unsupported joint_type")
                if joint.get("parent_part") not in part_ids or joint.get("child_part") not in part_ids:
                    raise ValueError(f"{obj.get('object_id')} joint references unknown part")
    _validate_support_hierarchy(objects)
def validate_against_recognition(data, recognition):
    expected_table = recognition["table"]
    actual_table = data["table"]
    if actual_table.get("object_id") != expected_table.get("object_id"):
        raise ValueError("blueprint table object_id disagrees with recognition")
    actual_table["category"] = expected_table["category"]
    expected = {item["object_id"]: item for item in recognition["objects"]}
    actual = {item["object_id"]: item for item in data["objects"]}
    if set(actual) != set(expected):
        raise ValueError("blueprint object IDs disagree with recognition")
    for object_id, item in expected.items():
        actual[object_id]["category"] = item["category"]
        candidate = item.get("operability_candidate")
        if (
            candidate in {"rigid_non_articulated", "articulated_operable"}
            and actual[object_id].get("operability_type") != candidate
        ):
            raise ValueError("blueprint operability disagrees with recognition")


def attach_recognition_evidence(data, recognition):
    """Carry image-grounded identity evidence with the Blueprint object IDs."""
    fields = (
        "visual_evidence",
        "visible_components",
        "recognition_confidence",
        "alternative_categories",
        "front_segment_id",
        "topview_segment_id",
    )
    pairs = [(data["table"], recognition["table"])]
    recognized = {item["object_id"]: item for item in recognition["objects"]}
    pairs.extend((item, recognized[item["object_id"]]) for item in data["objects"])
    for blueprint_item, recognition_item in pairs:
        for field in fields:
            if field in recognition_item:
                blueprint_item[field] = recognition_item[field]
    return data


def _valid_bbox(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item > 0
            for item in value
        )
    )


def _validate_object_placement(item):
    object_id = item.get("object_id")
    placement = item.get("placement")
    if not isinstance(placement, dict):
        raise ValueError(f"{object_id} must define placement")
    center = placement.get("center_xy_norm")
    if (
        not isinstance(center, list)
        or len(center) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in center
        )
    ):
        raise ValueError(
            f"{object_id} must define numeric center_xy_norm=[x, y]"
        )
    if any(value < -0.5 or value > 0.5 for value in center):
        raise ValueError(f"{object_id} center_xy_norm out of range")
    yaw = placement.get("yaw_deg")
    if (
        not isinstance(yaw, (int, float))
        or isinstance(yaw, bool)
        or not math.isfinite(yaw)
    ):
        raise ValueError(f"{object_id} must define finite numeric yaw_deg")


def _validate_support_hierarchy(objects):
    parents = {
        item["object_id"]: item["placement"].get("support_parents")
        for item in objects
    }
    valid_ids = {"table_0", *parents}
    for object_id, direct_parents in parents.items():
        if (
            not isinstance(direct_parents, list)
            or not direct_parents
            or len(direct_parents) != len(set(direct_parents))
            or any(
                parent not in valid_ids or parent == object_id
                for parent in direct_parents
            )
        ):
            raise ValueError(f"{object_id} must define valid unique support_parents")

    visiting = set()
    visited = set()

    def visit(object_id):
        if object_id == "table_0" or object_id in visited:
            return
        if object_id in visiting:
            raise ValueError("support_parents hierarchy must be acyclic")
        visiting.add(object_id)
        for parent in parents[object_id]:
            visit(parent)
        visiting.remove(object_id)
        visited.add(object_id)

    for object_id in parents:
        visit(object_id)


def _validate_english_only_json(data):
    serialized = json.dumps(data, ensure_ascii=False)
    if any("\u3400" <= char <= "\u9fff" for char in serialized):
        raise ValueError("blueprint JSON must not contain Chinese characters; use English")


def _validate_scene_topology(data):
    topology = data.get("scene_topology")
    relations = topology.get("relations") if isinstance(topology, dict) else None
    if not isinstance(relations, list):
        raise ValueError("scene_topology must define a relations list")

    if relations:
        raise ValueError("scene_topology relations are temporarily disabled")


def _validate_named_dimensions(item, label):
    dimensions = item.get("dimensions_cm")
    expected = item.get("bbox_cm")
    if not isinstance(dimensions, dict):
        raise ValueError(f"{label} must define dimensions_cm")
    actual = [dimensions.get("width_x"), dimensions.get("depth_y"), dimensions.get("height_z")]
    if actual != expected:
        raise ValueError(f"{label} dimensions_cm disagrees with bbox_cm")
    if item.get("axis_semantics") != AXIS_SEMANTICS:
        raise ValueError(f"{label} must define canonical axis_semantics")


def _normalize_local_axes(part):
    geometry_type = part.get("geometry_type")
    if geometry_type not in PROFILE_FIELDS:
        return
    axis_names = PROFILE_FIELDS[geometry_type][0]
    local_axes = part.get("local_axes") or {}
    axes = [local_axes.get(name) for name in axis_names]
    if not all(
        isinstance(axis, (list, tuple))
        and len(axis) == 3
        and all(isinstance(value, (int, float)) for value in axis)
        for axis in axes
    ):
        return
    first, second, third = ([float(value) for value in axis] for axis in axes)
    first_length = math.sqrt(sum(value * value for value in first))
    if first_length <= 1e-9:
        return
    first = [value / first_length for value in first]
    projection = sum(first[index] * second[index] for index in range(3))
    second = [second[index] - projection * first[index] for index in range(3)]
    second_length = math.sqrt(sum(value * value for value in second))
    if second_length <= 1e-9:
        return
    second = [value / second_length for value in second]
    normalized_third = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
    if sum(normalized_third[index] * third[index] for index in range(3)) < 0:
        second = [-value for value in second]
        normalized_third = [-value for value in normalized_third]
    for name, axis in zip(axis_names, (first, second, normalized_third)):
        local_axes[name] = axis


def normalize_blueprint(data):
    aliases = {
        "rigid_nonoperable": "rigid_non_articulated",
        "rigid_non_operable": "rigid_non_articulated",
        "rigid": "rigid_non_articulated",
    }
    items = [data.get("table", {})] + data.get("objects", [])
    for item in items:
        bbox = item.get("bbox_cm")
        if _valid_bbox(bbox):
            item.setdefault("dimensions_cm", named_dimensions(bbox))
        item["axis_semantics"] = dict(AXIS_SEMANTICS)
    for obj in data.get("objects", []):
        placement = obj.get("placement") or {}
        if "support_parents" not in placement and "support_parent" in placement:
            placement["support_parents"] = [placement.pop("support_parent")]
        op = obj.get("operability_type")
        obj["operability_type"] = aliases.get(op, op)
        if obj["operability_type"] == "rigid_non_articulated":
            obj.setdefault("articulation", None)
        articulation = obj.get("articulation") or {}
        legacy_joints = articulation.get("joints")
        if "joint" not in articulation and isinstance(legacy_joints, list) and len(legacy_joints) == 1:
            articulation["joint"] = legacy_joints[0]
            del articulation["joints"]
        for part in articulation.get("parts", []):
            geometry_type = part.get("geometry_type")
            if isinstance(geometry_type, str) and geometry_type not in PROFILE_FIELDS and geometry_type != "none":
                part["geometry_type"] = "freeform"
                part["local_axes"] = dict(FREEFORM_LOCAL_AXES)
                part["repair_mode"] = "preserve_proportions"
            _normalize_local_axes(part)
    return data


def call_mllm(
    model, context=None, attempt=None, front_image_path=None,
    topview_image_path=None, recognition_path=None,
):
    out_dir = attempt_path(attempt, "logs") if attempt else path("data")
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "blueprint_raw_response.txt"
    log_step("blueprint", "building prompt")
    recognition = json.loads(recognition_path.read_text(encoding="utf-8")) if recognition_path else None
    prompt = build_prompt(context, recognition)
    runtime = pipeline_runtime().with_model(model)
    log_step("blueprint", f"calling {runtime.provider} model={runtime.model}")
    cwd = context.project_root if context else path()
    raw = run_mllm(
        runtime,
        prompt,
        [
            image_path for image_path in (front_image_path, topview_image_path)
            if image_path is not None
        ],
        raw_path,
        cwd,
    )
    log_step("blueprint", f"raw response saved: {raw_path}")
    log_step("blueprint", "parsing JSON response")
    data = normalize_blueprint(extract_json(raw))
    log_step("blueprint", "validating blueprint")
    validate_blueprint(data)
    if recognition is not None:
        validate_against_recognition(data, recognition)
        attach_recognition_evidence(data, recognition)
    write_json(out_dir / "blueprint_raw_response.json", data)
    log_step("blueprint", "wrote data/blueprint_raw_response.json")
    return data


def run(context, attempt):
    log_step("blueprint", "creating output directories")
    attempt.temp_root.mkdir(parents=True, exist_ok=True)
    runtime = pipeline_runtime()
    log_step("blueprint", f"provider={runtime.provider}")
    front_image = dependency_file(attempt, "reference_image", "image")
    topview_image = dependency_file(attempt, "topview_image", "topview_image")
    recognition = dependency_file(attempt, "condition_images", "recognition")
    blueprint = call_mllm(
        runtime.model, context, attempt, front_image, topview_image, recognition,
    )
    write_json(attempt_path(attempt, "data/blueprint.json"), blueprint)
    log_step("blueprint", "wrote data/blueprint.json")
    return completed_files(attempt, STAGE_OUTPUTS["blueprint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.model:
        os.environ["ARTICUTABLE_MLLM_MODEL"] = args.model
    run()
