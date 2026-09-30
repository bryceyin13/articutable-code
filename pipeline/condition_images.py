#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.stage_io import (
    dependency_file,
    completed_files,
    stage_outputs,
)
from pipeline.image_generation import run_image_generation
from pipeline.common import (
    attempt_path, env_positive_int, log_step, project_path, read_json, write_json,
)
from pipeline.segmentation import box_iou
from pipeline.select_items import match_anonymous_segments_by_horizontal_range
from pipeline.context import AttemptContext, RunContext
from pipeline.mllm import image_generation_runtime, pipeline_runtime, run_mllm


RIGID_TYPES = {"static_support", "rigid_non_articulated"}
ARTICULATED_TYPE = "articulated_operable"
CONDITION_IMAGE_RESOLUTION = "native"
CONDITION_IMAGE_PROMPT = "07_trellis2_condition_image.txt"
CONDITION_PLANNER_PROMPT = "06_trellis2_view_planner.txt"
REAL_CONDITION_PROMPT = "08_real_segment_condition_image.txt"
REAL_RENDER_PROMPT = "09_real_render_condition_image.txt"
REAL_TABLE_RENDER_PROMPT = "10_real_render_table_condition_image.txt"
ARTICULATION_OPEN_PROMPT = "11_articulation_open_image.txt"
REAL_OPEN_VIEW_PROMPT = "12_real_articulation_view_planner.txt"
ARTICULATION_RECONSTRUCTION = "articulation_reconstruction.json"
REAL_REQUEST_FINGERPRINT = "request_fingerprint.txt"
OBJECT_OPERABILITY_TYPES = {"rigid_non_articulated", "articulated_operable"}
OBJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[0-9]+$")
REFERENCE_DESCRIPTION = """The references are provided in this order:
1. the complete front source scene;
2. the complete generated orthographic top-view hypothesis;
3. an unmasked local crop around the current front segment;
4. the current front segment cut out on white;
5. the geometrically matched top-view segment cutout when available; otherwise an approximate front mask placeholder."""


def identity_evidence(label):
    category = " ".join(
        label["category"].replace("_", " ").replace("-", " ").split()
    )
    lines = [f"Category label: {category}"]
    operability = label.get("operability_type")
    if operability:
        lines.append(f"Operability: {operability}")
    if label.get("visual_evidence"):
        lines.append(f"Image-grounded visual evidence: {label['visual_evidence']}")
    if label.get("visible_components"):
        lines.append("Visible components: " + ", ".join(label["visible_components"]))
    return "\n".join(lines)


def extract_plan(text, target_role=None, operability=None):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("condition planner response did not contain JSON")
    plan = json.loads(text[start:end + 1])
    drawing_prompt = plan.get("drawing_prompt")
    articulation_view = plan.get("articulation_view")
    if target_role is None:
        if not _nonempty_string(drawing_prompt):
            raise ValueError("condition planner must return a nonempty drawing_prompt")
        plan["drawing_prompt"] = drawing_prompt.strip()
        if set(plan) != {"drawing_prompt", "articulation_view"}:
            raise ValueError("generated condition plan has invalid fields")
        plan["articulation_view"] = validate_planned_articulation_view(
            articulation_view, operability,
        )
    else:
        identity = validate_real_segment_identity(
            {
                key: value for key, value in plan.items()
                if key not in {"drawing_prompt", "articulation_view"}
            },
            target_role,
        )
        if identity["keep"] and not _nonempty_string(drawing_prompt):
            raise ValueError("condition planner must return a nonempty drawing_prompt")
        plan["drawing_prompt"] = drawing_prompt.strip() if _nonempty_string(drawing_prompt) else ""
        plan["articulation_view"] = validate_planned_articulation_view(
            articulation_view,
            identity["operability_candidate"] if identity["keep"] else None,
        )
    return plan


def _call_mllm_json(context, attempt, object_id, prompt, images, model, log_name):
    raw = attempt_path(attempt, "logs", f"{log_name}_{object_id}_raw.txt")
    raw.parent.mkdir(parents=True, exist_ok=True)
    runtime = pipeline_runtime().with_model(model)
    log_step(
        "condition_images",
        f"{log_name} {object_id} with provider={runtime.provider}, model={runtime.model}",
    )
    return run_mllm(
        runtime, prompt, images, raw, context.project_root,
    )


def call_planner(
    context, attempt, object_id, prompt, images, model, target_role=None,
    operability=None,
):
    return extract_plan(
        _call_mllm_json(
            context, attempt, object_id, prompt, images, model, "condition_plan",
        ),
        target_role, operability,
    )


def extract_real_camera_plan(text, target_role):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("real condition response did not contain JSON")
    plan = json.loads(text[start:end + 1])
    drawing_prompt = plan.pop("drawing_prompt", None)
    if target_role == "object" and plan.get("operability_candidate") == "static_support":
        plan["operability_candidate"] = "rigid_non_articulated"
    identity = validate_real_segment_identity(plan, target_role)
    if identity["keep"] and not _nonempty_string(drawing_prompt):
        raise ValueError("condition planner must return a nonempty drawing_prompt")
    if not identity["keep"] and drawing_prompt != "":
        raise ValueError("rejected condition plan requires an empty drawing_prompt")
    return {
        **identity,
        "drawing_prompt": drawing_prompt.strip() if identity["keep"] else "",
        "articulation_view": None,
    }


def real_open_view_prompt(context, identity, drawing_prompt):
    return (
        project_path(context, "prompts", REAL_OPEN_VIEW_PROMPT)
        .read_text(encoding="utf-8")
        .replace("{{TARGET_EVIDENCE}}", identity_evidence(identity))
        .replace("{{DRAWING_PROMPT}}", drawing_prompt)
    )


def extract_articulation_view_decision(text):
    decision = json.loads(text.strip()) if isinstance(text, str) else text
    fields = {
        "joint_type", "requires_open_state", "reason",
        "movable_components", "opening_amount", "open_state_instruction",
    }
    if not isinstance(decision, dict) or set(decision) != fields:
        raise ValueError("articulation-view decision has invalid fields")
    if decision["joint_type"] not in {"hinge", "spin", "prismatic"}:
        raise ValueError("joint_type must be hinge, spin, or prismatic")
    if not isinstance(decision["requires_open_state"], bool):
        raise ValueError("requires_open_state must be boolean")
    if not _nonempty_string(decision["reason"]):
        raise ValueError("articulation-view reason must be nonempty")
    motion_fields = (
        "movable_components", "opening_amount",
        "open_state_instruction",
    )
    if decision["requires_open_state"]:
        if decision["joint_type"] == "prismatic":
            raise ValueError("prismatic joints must use the primary state")
        if any(not _nonempty_string(decision[field]) for field in motion_fields):
            raise ValueError("positive articulation-view decision requires motion fields")
    elif any(decision[field] is not None for field in motion_fields):
        raise ValueError("negative articulation-view decision requires null motion fields")
    if not _english_only(decision):
        raise ValueError("articulation-view decision must use English strings")
    return decision


def validate_planned_articulation_view(decision, operability):
    if operability != ARTICULATED_TYPE:
        if decision is not None:
            raise ValueError("non-articulated condition plan requires null articulation_view")
        return None
    if decision is None:
        raise ValueError("articulated condition plan requires articulation_view")
    return extract_articulation_view_decision(decision)


def articulation_open_prompt(context, target_evidence, decision):
    return (
        project_path(context, "prompts", ARTICULATION_OPEN_PROMPT)
        .read_text(encoding="utf-8")
        .replace("{{TARGET_EVIDENCE}}", target_evidence)
        .replace("{{JOINT_TYPE}}", decision["joint_type"])
        .replace("{{OPENING_AMOUNT}}", decision["opening_amount"])
        .replace("{{OPEN_STATE_INSTRUCTION}}", decision["open_state_instruction"])
    )


def planner_prompt(
    context, target_evidence, branch_requirements="", include_open_view=True,
):
    template = project_path(context, "prompts", CONDITION_PLANNER_PROMPT).read_text(
        encoding="utf-8"
    )
    if not include_open_view:
        template = template.split("PHASE 3 — OPEN-VIEW ELIGIBILITY", 1)[0].rstrip()
    return (
        template
        .replace("{{BRANCH_REQUIREMENTS}}", branch_requirements.strip())
        .replace("{{REFERENCE_DESCRIPTION}}", REFERENCE_DESCRIPTION)
        .replace("{{TARGET_EVIDENCE}}", target_evidence)
    )


def execution_prompt(context, template_name, target_evidence, drawing_prompt):
    template = project_path(context, "prompts", template_name).read_text(encoding="utf-8")
    return (
        template
        .replace("{{BRANCH_REQUIREMENTS}}", "")
        .replace("{{REFERENCE_DESCRIPTION}}", REFERENCE_DESCRIPTION)
        .replace("{{TARGET_EVIDENCE}}", target_evidence)
        .replace("{{DRAWING_PROMPT}}", drawing_prompt)
    )


def real_planner_prompt(
    context, target_segment_id, target_role, topview_segment_id,
    fifth_reference_description,
):
    requirements = project_path(context, "prompts", REAL_CONDITION_PROMPT).read_text(
        encoding="utf-8"
    )
    replacements = {
        "{{TARGET_FRONT_SEGMENT_ID}}": target_segment_id,
        "{{TARGET_ROLE}}": target_role,
        "{{MATCHED_TOPVIEW_SEGMENT_ID}}": topview_segment_id or "none",
        "{{FIFTH_REFERENCE_DESCRIPTION}}": fifth_reference_description,
    }
    for placeholder, value in replacements.items():
        requirements = requirements.replace(placeholder, value)
    return planner_prompt(
        context,
        "Infer the category and operability from the supplied front evidence and "
        "optional top-view match.",
        requirements,
        include_open_view=False,
    )


def call_real_plan(
    context, attempt, segment_id, target_role, topview_segment_id,
    fifth_reference_description, images, model,
):
    plan = extract_real_camera_plan(
        _call_mllm_json(
            context, attempt, segment_id,
            real_planner_prompt(
                context, segment_id, target_role, topview_segment_id,
                fifth_reference_description,
            ),
            images, model, "condition_plan",
        ),
        target_role,
    )
    if (
        plan["keep"]
        and plan["operability_candidate"] == ARTICULATED_TYPE
    ):
        raw = _call_mllm_json(
            context, attempt, segment_id,
            real_open_view_prompt(context, plan, plan["drawing_prompt"]),
            images, model, "condition_open_view",
        )
        plan["articulation_view"] = extract_articulation_view_decision(raw)
    return plan


def _fifth_reference_description(topview_segment_id):
    if topview_segment_id is None:
        return "the approximate full-frame mask of the unmatched front segment"
    return "the matched top-view segment cutout"


def _real_request_fingerprint(
    context, segment_id, target_role, topview_segment_id,
    planner_model, image_model,
):
    payload = {
        "planner_model": planner_model,
        "image_model": image_model,
        "pipeline_mllm": pipeline_runtime().metadata(),
        "image_generation": image_generation_runtime().metadata(),
        "resolution": CONDITION_IMAGE_RESOLUTION,
        "planner_prompt": real_planner_prompt(
            context, segment_id, target_role, topview_segment_id,
            _fifth_reference_description(topview_segment_id),
        ),
        "open_view_prompt": project_path(
            context, "prompts", REAL_OPEN_VIEW_PROMPT,
        ).read_text(encoding="utf-8"),
        "renderer_prompt": project_path(
            context, "prompts",
            REAL_TABLE_RENDER_PROMPT if target_role == "table" else REAL_RENDER_PROMPT,
        ).read_text(encoding="utf-8"),
        "articulation_open_prompt": project_path(
            context, "prompts", ARTICULATION_OPEN_PROMPT,
        ).read_text(encoding="utf-8"),
    }
    mode = getattr(context, "drawer_geometry_mode", "auto-3d")
    if mode != "reconstructed-open":
        payload["drawer_geometry_mode"] = mode
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_image(context):
    root = context.stage_best("reference_image")
    record = read_json(root / "attempt.json")
    return root / record["artifacts"]["image"]


def topview_image(context):
    root = context.stage_best("topview_image")
    record = read_json(root / "attempt.json")
    return root / record["artifacts"]["topview_image"]


def _prepare_context_crop(context_image, bbox, output_path):
    if not context_image.is_file():
        raise FileNotFoundError("missing scene evidence: " + str(context_image))
    x0, y0, x1, y1 = map(int, bbox)
    padding = max(x1 - x0, y1 - y0) // 8
    crop = (
        f"crop=w='min(iw,{x1 - x0 + 2 * padding})':"
        f"h='min(ih,{y1 - y0 + 2 * padding})':"
        f"x='max(0,min(iw-ow,{x0 - padding}))':"
        f"y='max(0,min(ih-oh,{y0 - padding}))'"
    )
    result = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(context_image),
        "-vf", crop, "-frames:v", "1", str(output_path),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to prepare local context")
    return output_path


def _prepare_front_references(front_context, front, front_mask, bbox, output):
    paths = [front_context, front, front_mask]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing selected item evidence: " + ", ".join(missing))
    context_crop = _prepare_context_crop(
        front_context, bbox, output / "front_context_crop.png",
    )
    cutout = output / "front_cutout.png"
    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=white", "-i", str(front),
        "-filter_complex",
        "[0:v][1:v]scale2ref[bg][fg];[bg][fg]overlay=shortest=1:format=auto,format=rgb24",
        "-frames:v", "1", str(cutout),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to prepare front evidence")
    return [context_crop, front_mask, cutout]


def references(
    selected_root, object_id, front_context, top_context, bbox, output,
    include_topview=False,
):
    root = selected_root / object_id
    front = root / "front.png"
    front_mask = root / "front_mask.png"
    if not top_context.is_file():
        raise FileNotFoundError("missing top-view scene evidence: " + str(top_context))
    local = _prepare_front_references(front_context, front, front_mask, bbox, output)
    topview = root / "topview.png"
    if include_topview:
        if not topview.is_file():
            raise FileNotFoundError("missing selected item evidence: " + str(topview))
        fifth = topview
    else:
        fifth = front_mask
    return [front_context, top_context, local[0], local[2], fifth]


def condition_spec(operability):
    if operability == ARTICULATED_TYPE or operability in RIGID_TYPES:
        return CONDITION_PLANNER_PROMPT, CONDITION_IMAGE_PROMPT
    raise ValueError(f"unsupported operability_type: {operability}")


def _articulation_reconstruction_strategy(output, decision):
    path = output / ARTICULATION_RECONSTRUCTION
    strategy = (
        read_json(path)
        if path.is_file()
        else {
            "drawer_geometry_mode": "reconstructed-open",
            "source": "primary",
        }
    )
    if (
        set(strategy) != {"drawer_geometry_mode", "source"}
        or strategy["drawer_geometry_mode"] not in {
            "reconstructed-open", "procedural-closed", "auto-3d",
        }
        or strategy["source"] not in {"primary", "opened"}
    ):
        raise ValueError(f"invalid articulation reconstruction strategy: {path}")
    return strategy


def _articulation_view_selection(output, operability):
    primary = output / "primary.png"
    if operability != ARTICULATED_TYPE:
        return primary, None
    decision_path = output / "articulation_view.json"
    if not decision_path.is_file():
        raise FileNotFoundError(f"missing articulation-view decision: {decision_path}")
    decision = extract_articulation_view_decision(
        decision_path.read_text(encoding="utf-8")
    )
    strategy = _articulation_reconstruction_strategy(output, decision)
    selected = output / "reconstruction_input.png"
    source = output / f"{strategy['source']}.png"
    if (
        not selected.is_file()
        or selected.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
    ):
        raise FileNotFoundError(f"missing articulation reconstruction image: {selected}")
    if not source.is_file() or selected.read_bytes() != source.read_bytes():
        raise ValueError(f"articulation reconstruction image does not match source: {source}")
    return selected, decision


def _condition_image_fields(output, operability, attempt):
    selected, decision = _articulation_view_selection(output, operability)
    fields = {
        "views": {
            "primary": str(selected.relative_to(attempt.temp_root))
        },
    }
    if decision is not None:
        strategy = _articulation_reconstruction_strategy(output, decision)
        fields.update({
            "source_condition_image": str(
                (output / "primary.png").relative_to(attempt.temp_root)
            ),
            "articulation_view": {
                "decision": str(
                    (output / "articulation_view.json").relative_to(
                        attempt.temp_root
                    )
                ),
                "opened_image": (
                    str(
                        (output / "opened.png").relative_to(
                            attempt.temp_root
                        )
                    )
                    if strategy["source"] == "opened" else None
                ),
            },
            "drawer_geometry_mode": strategy["drawer_geometry_mode"],
            "reconstruction_input_source": strategy["source"],
        })
    return fields


def generate_articulation_view(
    context, attempt, object_id, output, target_evidence, operability,
    decision, image_model,
):
    if operability != ARTICULATED_TYPE:
        return
    primary = output / "primary.png"
    decision = validate_planned_articulation_view(decision, operability)
    write_json(output / "articulation_view.json", decision)
    write_json(output / ARTICULATION_RECONSTRUCTION, {
        "drawer_geometry_mode": getattr(
            context, "drawer_geometry_mode", "auto-3d",
        ),
        "source": "primary",
    })
    shutil.copy2(primary, output / "reconstruction_input.png")


def generate_one(
    context, attempt, selected_root, front_context, top_context, selected_item, label,
    planner_model, image_model,
):
    object_id = label["object_id"]
    operability = label["operability_type"]
    output = attempt_path(attempt, "condition_images", object_id)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    images = references(
        selected_root, object_id, front_context, top_context,
        selected_item["views"]["front"]["bbox_xyxy"], output,
        include_topview="topview" in selected_item["views"],
    )
    _, image_template = condition_spec(operability)
    target_evidence = identity_evidence(label)
    plan = call_planner(
        context, attempt, object_id,
        planner_prompt(context, target_evidence), images, planner_model,
        operability=operability,
    )
    plan_path = output / "plan.json"
    drawing_prompt_path = output / "drawing_prompt.txt"
    write_json(plan_path, plan)
    drawing_prompt_path.write_text(plan["drawing_prompt"] + "\n", encoding="utf-8")

    primary = output / "primary.png"
    run_image_generation(
        lambda _unused: execution_prompt(
            context, image_template, target_evidence, plan["drawing_prompt"],
        ),
        image_model, primary,
        f"condition_render_{object_id}",
        images=images,
        resolution=CONDITION_IMAGE_RESOLUTION,
        final_size=None,
        project_root=context.project_root, log_dir=attempt_path(attempt, "logs"),
    )
    generate_articulation_view(
        context, attempt, object_id, output, target_evidence, operability,
        plan["articulation_view"], image_model,
    )

    return {
        "object_id": object_id,
        "operability_type": operability,
        "plan": str(plan_path.relative_to(attempt.temp_root)),
        "drawing_prompt": str(drawing_prompt_path.relative_to(attempt.temp_root)),
        **_condition_image_fields(output, operability, attempt),
    }


def _identity_items(recognition):
    return [recognition["table"], *recognition["objects"]]


def duplicate_front_segment_ids(front_items):
    """Catch near-identical legacy proposals before per-segment generation."""
    kept = []
    suppressed = set()
    candidates = sorted(
        (
            item for item in front_items
            if (
                item["segment_id"] != "table_0"
                and item["views"]["front"].get("mask_area_px")
            )
        ),
        key=lambda item: (
            -float(item["views"]["front"]["score"]),
            item["segment_id"],
        ),
    )
    for item in candidates:
        view = item["views"]["front"]
        area = float(view["mask_area_px"])
        duplicate = any(
            box_iou(view["bbox_xyxy"], kept_view["bbox_xyxy"]) >= 0.95
            and min(area, kept_area) / max(area, kept_area) >= 0.90
            for kept_view, kept_area in kept
        )
        if duplicate:
            suppressed.add(item["segment_id"])
        else:
            kept.append((view, area))
    return suppressed


def _english_only(value):
    return not any(
        "\u3400" <= char <= "\u9fff"
        for char in json.dumps(value, ensure_ascii=False)
    )


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_string_list(value, field):
    if (
        not isinstance(value, list)
        or any(not _nonempty_string(item) for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"recognition {field} must contain unique non-empty strings")


def validate_real_recognition(recognition, front_segment_ids, topview_segment_ids):
    expected_top = {
        "table", "objects", "rejected_front_segment_ids",
        "unused_topview_segment_ids",
    }
    if not isinstance(recognition, dict) or set(recognition) != expected_top:
        raise ValueError("real recognition has invalid top-level fields")
    if not isinstance(recognition["table"], dict) or not isinstance(recognition["objects"], list):
        raise ValueError("real recognition must define one table and an objects list")
    if not _english_only(recognition):
        raise ValueError("real recognition JSON must use English strings")

    table_fields = {
        "object_id", "category", "front_segment_id", "topview_segment_id",
        "operability_type", "visual_evidence",
    }
    object_fields = {
        "object_id", "category", "front_segment_id", "topview_segment_id",
        "operability_candidate", "visual_evidence", "visible_components",
    }
    table = recognition["table"]
    if set(table) != table_fields:
        raise ValueError("real recognition table has invalid fields")
    if table["object_id"] != "table_0" or table["operability_type"] != "static_support":
        raise ValueError("real recognition must define table_0 as the static_support")

    for field in ("category", "front_segment_id", "topview_segment_id", "visual_evidence"):
        if not _nonempty_string(table.get(field)):
            raise ValueError(f"real recognition table must define {field}")

    for item in recognition["objects"]:
        if not isinstance(item, dict) or set(item) != object_fields:
            raise ValueError("real recognition object has invalid fields")
        if not OBJECT_ID_PATTERN.fullmatch(item.get("object_id", "")):
            raise ValueError("real recognition object_id must be stable snake_case with an index")
        if item["object_id"] == "table_0":
            raise ValueError("real recognition object_id values must not reuse table_0")
        for field in ("category", "front_segment_id", "visual_evidence"):
            if not _nonempty_string(item.get(field)):
                raise ValueError(f"{item.get('object_id')} must define {field}")
        if item.get("topview_segment_id") is not None and not _nonempty_string(
            item.get("topview_segment_id")
        ):
            raise ValueError(
                f"{item.get('object_id')} topview_segment_id must be a segment ID or null"
            )
        if item.get("operability_candidate") not in OBJECT_OPERABILITY_TYPES:
            raise ValueError(f"{item.get('object_id')} has invalid operability_candidate")
        _validate_string_list(
            item.get("visible_components"),
            f"{item.get('object_id')}.visible_components",
        )

    kept = _identity_items(recognition)
    object_ids = [item["object_id"] for item in kept]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("real recognition object_id values must be unique")

    rejected_front = recognition["rejected_front_segment_ids"]
    unused_top = recognition["unused_topview_segment_ids"]
    _validate_string_list(rejected_front, "rejected_front_segment_ids")
    _validate_string_list(unused_top, "unused_topview_segment_ids")

    kept_front = [item["front_segment_id"] for item in kept]
    kept_top = [
        item["topview_segment_id"]
        for item in kept
        if item["topview_segment_id"] is not None
    ]
    if len(kept_front) != len(set(kept_front)):
        raise ValueError("kept front segment IDs must be unique")
    if len(kept_top) != len(set(kept_top)):
        raise ValueError("kept top-view segment IDs must be one-to-one")
    if set(kept_front) & set(rejected_front):
        raise ValueError("kept and rejected front segment IDs must be disjoint")
    if set(kept_top) & set(unused_top):
        raise ValueError("used and unused top-view segment IDs must be disjoint")
    if set(kept_front) | set(rejected_front) != set(front_segment_ids):
        raise ValueError("recognition must cover every anonymous front segment exactly once")
    if set(kept_top) | set(unused_top) != set(topview_segment_ids):
        raise ValueError("recognition must cover every anonymous top-view segment exactly once")
    return recognition


def validate_real_segment_identity(identity, target_role):
    fields = {
        "keep", "category", "operability_candidate",
        "visual_evidence", "visible_components",
    }
    if not isinstance(identity, dict) or set(identity) != fields:
        raise ValueError("real segment identity has invalid fields")
    if not isinstance(identity["keep"], bool):
        raise ValueError("real segment identity keep must be boolean")
    if not _nonempty_string(identity.get("visual_evidence")):
        raise ValueError("real segment identity must define visual_evidence")
    _validate_string_list(identity.get("visible_components"), "visible_components")
    if target_role == "table" and not identity["keep"]:
        raise ValueError("the dedicated table segment cannot be rejected")
    if identity["keep"]:
        if not _nonempty_string(identity.get("category")):
            raise ValueError("kept real segment identity must define category")
        expected = (
            {"static_support"} if target_role == "table"
            else OBJECT_OPERABILITY_TYPES
        )
        if identity.get("operability_candidate") not in expected:
            raise ValueError("kept real segment identity has invalid operability_candidate")
    elif identity["category"] is not None or identity["operability_candidate"] is not None:
        raise ValueError("rejected real segment identity must use null category and operability")
    if not _english_only(identity):
        raise ValueError("real segment identity JSON must use English strings")
    return identity


def _object_id_base(category):
    base = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    if not base or not base[0].isalpha():
        base = f"object_{base}".rstrip("_")
    return base


def build_real_recognition(segment_identities, selected, segment_pairs, unused_top):
    front_items = selected["items"]
    topview_pool = selected["topview_pool"]
    if "table_0" not in segment_identities:
        raise ValueError("real recognition requires the dedicated front table_0")
    if not any(item["segment_id"] == "table_0" for item in topview_pool):
        raise ValueError("real recognition requires the dedicated top-view table_0")

    table_identity = segment_identities["table_0"]
    table = {
        "object_id": "table_0",
        "category": table_identity["category"],
        "front_segment_id": "table_0",
        "topview_segment_id": "table_0",
        "operability_type": "static_support",
        "visual_evidence": table_identity["visual_evidence"],
    }
    counts = {}
    objects = []
    rejected = []
    for selected_item in front_items:
        segment_id = selected_item["segment_id"]
        if segment_id == "table_0":
            continue
        identity = segment_identities[segment_id]
        if not identity["keep"]:
            rejected.append(segment_id)
            continue
        base = _object_id_base(identity["category"])
        index = counts.get(base, 0)
        counts[base] = index + 1
        objects.append({
            "object_id": f"{base}_{index}",
            "category": identity["category"],
            "front_segment_id": segment_id,
            "topview_segment_id": segment_pairs.get(segment_id),
            "operability_candidate": identity["operability_candidate"],
            "visual_evidence": identity["visual_evidence"],
            "visible_components": identity["visible_components"],
        })
    recognition = {
        "table": table,
        "objects": objects,
        "rejected_front_segment_ids": rejected,
        "unused_topview_segment_ids": [
            segment_id for segment_id in unused_top
            if segment_id not in {
                item["topview_segment_id"] for item in objects
                if item["topview_segment_id"] is not None
            }
        ],
    }
    for front_segment_id in rejected:
        top_segment_id = segment_pairs.get(front_segment_id)
        if (
            top_segment_id is not None
            and top_segment_id not in recognition["unused_topview_segment_ids"]
        ):
            recognition["unused_topview_segment_ids"].append(top_segment_id)
    validate_real_recognition(
        recognition,
        [item["segment_id"] for item in front_items],
        [item["segment_id"] for item in topview_pool],
    )
    return recognition


def _bootstrap_real_selection(selected, segment_pairs, object_ids):
    requested = [
        segment_id for segment_id in dict.fromkeys(object_ids)
        if segment_id != "table_0"
    ]
    front_ids = {item["segment_id"] for item in selected["items"]}
    unknown = [segment_id for segment_id in requested if segment_id not in front_ids]
    if unknown:
        raise KeyError(f"{unknown[0]} not found in selected items")
    if not requested:
        raise ValueError("partial real bootstrap requires at least one object segment")
    included = {"table_0", *requested}
    paired_top = {segment_pairs[segment_id] for segment_id in included}
    return {
        **selected,
        "items": [item for item in selected["items"] if item["segment_id"] in included],
        "topview_pool": [
            item for item in selected["topview_pool"]
            if item["segment_id"] in paired_top
        ],
        "segment_pairs": {
            segment_id: segment_pairs[segment_id] for segment_id in included
        },
        "matches": [
            item for item in selected.get("matches", [])
            if item["front_segment_id"] in included
        ],
        "unused_topview_segment_ids": [],
    }, set(requested)


def _selected_path(selected_root, relative):
    result = selected_root / relative
    if not result.is_file():
        raise FileNotFoundError(f"missing selected item evidence: {result}")
    return result


def real_references(
    selected_root, selected, selected_item, output, topview_segment_id,
):
    evidence = selected["evidence"]
    global_images = [
        _selected_path(selected_root, evidence[name])
        for name in ("front_image", "topview_image")
    ]
    front_view = selected_item["views"]["front"]
    front = _selected_path(selected_root, front_view["crop"])
    front_mask = _selected_path(selected_root, front_view["mask"])
    local_images = _prepare_front_references(
        global_images[0], front, front_mask, front_view["bbox_xyxy"], output,
    )
    if topview_segment_id is None:
        fifth = front_mask
        render_topview_context = local_images[0]
        render_topview_crop = front
    else:
        topview_item = next(
            item for item in selected["topview_pool"]
            if item["segment_id"] == topview_segment_id
        )
        topview_view = topview_item["views"]["topview"]
        fifth = _selected_path(selected_root, topview_view["crop"])
        render_topview_context = _prepare_context_crop(
            global_images[1], topview_view["bbox_xyxy"],
            output / "topview_context_crop.png",
        )
        render_topview_crop = fifth
    return (
        [global_images[0], global_images[1], local_images[0], local_images[2], fifth],
        [
            local_images[0], front, local_images[2], render_topview_context,
            render_topview_crop,
        ],
        _fifth_reference_description(topview_segment_id),
    )


def _real_operability(identity):
    return identity.get("operability_type") or identity["operability_candidate"]


def _real_manifest_item(identity, primary, attempt):
    operability = _real_operability(identity)
    condition_spec(operability)
    return {
        "object_id": identity["object_id"],
        "front_segment_id": identity["front_segment_id"],
        "topview_segment_id": identity["topview_segment_id"],
        "operability_type": operability,
        "plan": str((primary.parent / "plan.json").relative_to(attempt.temp_root)),
        "drawing_prompt": str(
            (primary.parent / "drawing_prompt.txt").relative_to(attempt.temp_root)
        ),
        **_condition_image_fields(primary.parent, operability, attempt),
    }


def _run_real_image_call(
    context, attempt, selected_root, selected, selected_item, planner_model,
    image_model, topview_segment_id, previous=None, render=True,
):
    segment_id = selected_item["segment_id"]
    output = attempt_path(attempt, "condition_images", segment_id)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    primary = output / "primary.png"
    identity_path = output / "identity.json"
    images, render_images, fifth_description = real_references(
        selected_root, selected, selected_item, output, topview_segment_id,
    )
    if len(images) != 5:
        raise RuntimeError(f"real condition generation requires exactly 5 images, got {len(images)}")
    target_role = "table" if segment_id == "table_0" else "object"
    request_fingerprint = _real_request_fingerprint(
        context, segment_id, target_role, topview_segment_id,
        planner_model, image_model,
    )
    (output / REAL_REQUEST_FINGERPRINT).write_text(
        request_fingerprint + "\n", encoding="utf-8",
    )
    previous_output = (
        previous / "condition_images" / segment_id if previous is not None else None
    )
    previous_plan = previous_output / "plan.json" if previous_output else None
    previous_fingerprint = (
        previous_output / REAL_REQUEST_FINGERPRINT if previous_output else None
    )
    if (
        previous_plan is not None
        and previous_plan.is_file()
        and previous_fingerprint.is_file()
        and previous_fingerprint.read_text(encoding="utf-8").strip()
        == request_fingerprint
    ):
        try:
            plan = extract_plan(
                previous_plan.read_text(encoding="utf-8"), target_role,
            )
            log_step("condition_images", f"reused completed plan for {segment_id}")
        except (ValueError, json.JSONDecodeError):
            plan = None
    else:
        plan = None
    if plan is None:
        plan = call_real_plan(
            context, attempt, segment_id, target_role,
            topview_segment_id, fifth_description, images, planner_model,
        )
    write_json(output / "plan.json", plan)
    (output / "drawing_prompt.txt").write_text(
        plan["drawing_prompt"] + "\n", encoding="utf-8"
    )
    identity = validate_real_segment_identity(
        {
            key: value for key, value in plan.items()
            if key not in {"drawing_prompt", "articulation_view"}
        },
        target_role,
    )
    write_json(identity_path, identity)
    target_evidence = identity_evidence(identity) if identity["keep"] else identity["visual_evidence"]
    if identity["keep"] and render:
        render_prompt = (
            REAL_TABLE_RENDER_PROMPT if target_role == "table" else REAL_RENDER_PROMPT
        )
        if target_role == "table":
            render_images = [render_images[2], render_images[4]]
        run_image_generation(
            lambda _unused: execution_prompt(
                context, render_prompt, target_evidence,
                plan["drawing_prompt"],
            ),
            image_model, primary,
            f"condition_render_{segment_id}",
            images=render_images,
            resolution=CONDITION_IMAGE_RESOLUTION,
            final_size=None,
            project_root=context.project_root, log_dir=attempt_path(attempt, "logs"),
        )
        generate_articulation_view(
            context, attempt, segment_id, output, target_evidence,
            _real_operability(identity), plan["articulation_view"], image_model,
        )
    else:
        log_step("condition_images", f"skipping rejected {segment_id}")
    return output, primary, identity


def _latest_reusable_real_attempt(attempt):
    """Return the newest failed attempt from the same upstream inputs."""
    for root in sorted(attempt.final_root.parent.glob("[!.]*"), reverse=True):
        record_path = root / "attempt.json"
        if not record_path.is_file():
            continue
        record = read_json(record_path)
        if (
            record.get("status") == "failed"
            and record.get("upstream_best") == attempt.upstream_best
        ):
            return root
    return None


def _real_segment_output(root, segment_id):
    direct = root / "condition_images" / segment_id
    if direct.is_dir():
        return direct
    recognition_path = root / "data/recognition.json"
    if recognition_path.is_file():
        recognition = read_json(recognition_path)
        for identity in _identity_items(recognition):
            if identity["front_segment_id"] == segment_id:
                return root / "condition_images" / identity["object_id"]
    return direct


def _checkpoint_link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _carry_real_gram_checkpoints(previous, attempt):
    source_root = previous / ".gram_stream/outputs"
    image_root = previous / ".image_to_3d_stream/outputs"
    records = []
    if not source_root.is_dir():
        return
    for source in sorted(source_root.iterdir()):
        job_id = source.name
        package = source / "packages" / job_id
        required = [
            package / "model.urdf",
            *(package / "joint_range_images" / f"{pose}.png"
              for pose in ("lower", "rest", "upper")),
        ]
        source_mesh = image_root / job_id / "model.glb"
        if (
            not source.is_dir()
            or any(not path.is_file() or path.stat().st_size == 0 for path in required)
            or not source_mesh.is_file()
        ):
            continue
        destination = attempt_path(
            attempt, ".gram_stream", "outputs", job_id,
        )
        shutil.copytree(
            source, destination, copy_function=_checkpoint_link_or_copy,
        )
        records.append({
            "job_id": job_id,
            "object_id": job_id,
            "gpu_id": None,
            "source_mesh_sha256": hashlib.sha256(
                source_mesh.read_bytes()
            ).hexdigest(),
            "metadata": {"checkpoint_attempt": previous.name},
        })
    if records:
        write_json(
            attempt_path(attempt, ".gram_stream", "manifest.json"),
            {"items": records},
        )
        log_step(
            "condition_images",
            f"carried forward {len(records)} completed gram checkpoint(s)",
        )


def _reuse_real_image_call(
    context, attempt, previous, selected_item, planner_model, image_model,
    topview_segment_id,
):
    segment_id = selected_item["segment_id"]
    target_role = "table" if segment_id == "table_0" else "object"
    request_fingerprint = _real_request_fingerprint(
        context, segment_id, target_role, topview_segment_id,
        planner_model, image_model,
    )
    source = _real_segment_output(previous, segment_id)
    primary = source / "primary.png"
    identity_path = source / "identity.json"
    plan_path = source / "plan.json"
    drawing_prompt_path = source / "drawing_prompt.txt"
    fingerprint_path = source / REAL_REQUEST_FINGERPRINT
    if (
        not primary.is_file()
        or primary.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
        or not identity_path.is_file()
        or not plan_path.is_file()
        or not drawing_prompt_path.is_file()
        or not fingerprint_path.is_file()
        or fingerprint_path.read_text(encoding="utf-8").strip()
        != request_fingerprint
    ):
        return None
    try:
        identity = validate_real_segment_identity(
            read_json(identity_path),
            "table" if segment_id == "table_0" else "object",
        )
    except (ValueError, json.JSONDecodeError):
        return None
    operability = _real_operability(identity)
    if operability == ARTICULATED_TYPE:
        try:
            _articulation_view_selection(source, operability)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return None
    output = attempt_path(attempt, "condition_images", segment_id)
    shutil.copytree(source, output)
    return output, output / "primary.png", identity


def _rerun_partial_real(
    context, attempt, selected_root, selected, segment_pairs,
    planner_model, image_model, object_ids,
):
    previous = latest_completed_attempt(context)
    recognition = read_json(previous / "data/recognition.json")
    manifest = read_json(previous / "data/condition_manifest.json")
    identities = {
        item["object_id"]: item for item in _identity_items(recognition)
    }
    ids = list(dict.fromkeys(object_ids))
    unknown = [object_id for object_id in ids if object_id not in identities]
    if unknown:
        raise KeyError(f"{unknown[0]} not found in previous real recognition")
    selected_by_segment = {
        item["segment_id"]: item for item in selected["items"]
    }
    shutil.copytree(
        previous / "condition_images",
        attempt_path(attempt, "condition_images"),
    )
    def regenerate(object_id):
        segment_id = identities[object_id]["front_segment_id"]
        selected_item = selected_by_segment[segment_id]
        destination = attempt_path(attempt, "condition_images", object_id)
        shutil.rmtree(destination)
        output, _, identity = _run_real_image_call(
            context, attempt, selected_root, selected, selected_item,
            planner_model, image_model, segment_pairs.get(segment_id), None,
        )
        if not identity["keep"]:
            raise ValueError(f"partial rerun rejected previously kept object {object_id}")
        output.rename(destination)
        updated_identity = {
            **identities[object_id],
            "category": identity["category"],
            "visual_evidence": identity["visual_evidence"],
        }
        if object_id != "table_0":
            updated_identity.update({
                "operability_candidate": identity["operability_candidate"],
                "visible_components": identity["visible_components"],
            })
        log_step("condition_images", f"regenerated {object_id} ({segment_id})")
        return object_id, updated_identity, _real_manifest_item(
            updated_identity, destination / "primary.png", attempt,
        )

    workers = min(len(ids), env_positive_int("CONDITION_MAX_WORKERS", 4))
    log_step("condition_images", f"object workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        regenerated = list(executor.map(regenerate, ids))
    recognition_updates = {
        object_id: identity for object_id, identity, _ in regenerated
    }
    replacements = {
        object_id: item for object_id, _, item in regenerated
    }
    recognition["table"] = recognition_updates.get(
        "table_0", recognition["table"],
    )
    recognition["objects"] = [
        recognition_updates.get(item["object_id"], item)
        for item in recognition["objects"]
    ]
    manifest["items"] = [
        replacements.get(item["object_id"], item) for item in manifest["items"]
    ]
    write_json(attempt_path(attempt, "data/recognition.json"), recognition)
    write_json(attempt_path(attempt, "data/condition_manifest.json"), manifest)
    return completed_files(
        attempt, stage_outputs("condition_images"),
    )


def run_real(
    context, attempt, planner_model, image_model, object_ids=None, item_ready=None,
    executor=None, submission_callback=None,
):
    selected_manifest = dependency_file(attempt, "select_items", "manifest")
    selected_root = selected_manifest.parent.parent
    selected = read_json(selected_manifest)
    front_items = selected.get("items", [])
    topview_pool = selected.get("topview_pool", [])
    if not front_items or not topview_pool:
        raise RuntimeError("real condition generation requires non-empty front and top-view pools")
    if len({item["segment_id"] for item in front_items}) != len(front_items):
        raise ValueError("anonymous front segment IDs must be unique")

    if "segment_pairs" in selected:
        segment_pairs = selected["segment_pairs"]
        unused_top = selected.get("unused_topview_segment_ids", [])
        matches = [
            (
                item["front_segment_id"],
                item["topview_segment_id"],
                item["cost"],
            )
            for item in selected.get("matches", [])
        ]
    else:
        segment_pairs, unused_top, matches = (
            match_anonymous_segments_by_horizontal_range(selected)
        )
    for front_segment_id, topview_segment_id, cost in matches:
        log_step(
            "condition_images",
            f"horizontal range paired {front_segment_id} -> "
            f"{topview_segment_id}, cost={cost:.6f}",
        )

    bootstrap_segments = None
    if object_ids:
        front_segment_ids = {item["segment_id"] for item in front_items}
        if all(object_id in front_segment_ids for object_id in object_ids):
            selected, bootstrap_segments = _bootstrap_real_selection(
                selected, segment_pairs, object_ids,
            )
            front_items = selected["items"]
            topview_pool = selected["topview_pool"]
            segment_pairs = selected["segment_pairs"]
            unused_top = selected["unused_topview_segment_ids"]
            matches = [
                (
                    item["front_segment_id"], item["topview_segment_id"],
                    item["cost"],
                )
                for item in selected["matches"]
            ]
            log_step(
                "condition_images",
                "bootstrapping selected segments: " + ", ".join(sorted(bootstrap_segments)),
            )
        else:
            return _rerun_partial_real(
                context, attempt, selected_root, selected, segment_pairs,
                planner_model, image_model, object_ids,
            )

    generated = {}
    segment_identities = {}
    downstream = []
    duplicate_front_ids = duplicate_front_segment_ids(front_items)
    for segment_id in sorted(duplicate_front_ids):
        segment_identities[segment_id] = {
            "keep": False,
            "category": None,
            "operability_candidate": None,
            "visual_evidence": (
                "Rejected deterministically as a near-duplicate mask proposal."
            ),
            "visible_components": [],
        }
        log_step(
            "condition_images",
            f"rejected near-duplicate front proposal {segment_id}",
        )
    previous = _latest_reusable_real_attempt(attempt)
    if previous is not None:
        log_step(
            "condition_images",
            f"checking reusable outputs from failed attempt {previous.name}",
        )
        if item_ready is None:
            _carry_real_gram_checkpoints(previous, attempt)

    def process(selected_item):
        segment_id = selected_item["segment_id"]
        topview_segment_id = segment_pairs.get(segment_id)
        reused = (
            _reuse_real_image_call(
                context, attempt, previous, selected_item,
                planner_model, image_model, topview_segment_id,
            )
            if previous is not None else None
        )
        if reused is not None:
            output, primary, identity = reused
            log_step("condition_images", f"reused completed {segment_id}")
        else:
            output, primary, identity = _run_real_image_call(
                context, attempt, selected_root, selected, selected_item,
                planner_model, image_model, topview_segment_id, previous,
                render=bootstrap_segments is None or segment_id != "table_0",
            )
        future = None
        if item_ready is not None and identity["keep"]:
            operability = _real_operability(identity)
            condition_spec(operability)
            future = item_ready({
                "object_id": segment_id,
                "front_segment_id": segment_id,
                "operability_type": operability,
                **_condition_image_fields(output, operability, attempt),
            }, job_id=segment_id)
        return segment_id, output, primary, identity, future

    pending_front_items = [
        item for item in front_items
        if item["segment_id"] not in duplicate_front_ids
    ]
    if pending_front_items:
        workers = min(
            len(pending_front_items), env_positive_int("CONDITION_MAX_WORKERS", 4),
        )
        log_step("condition_images", f"object workers={workers}")
        owns_executor = executor is None
        object_executor = executor or ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [
                object_executor.submit(process, selected_item)
                for selected_item in pending_front_items
            ]
            if submission_callback is not None:
                submission_callback()
            for task in futures:
                segment_id, output, primary, identity, future = task.result()
                generated[segment_id] = (output, primary)
                segment_identities[segment_id] = identity
                if future is not None:
                    downstream.append(future)
        finally:
            if owns_executor:
                object_executor.shutdown()
    elif submission_callback is not None:
        submission_callback()

    for future in downstream:
        future.result()

    recognition = build_real_recognition(
        segment_identities, selected, segment_pairs, unused_top,
    )
    recognition_path = attempt_path(attempt, "data/recognition.json")
    write_json(recognition_path, recognition)

    identities = _identity_items(recognition)
    items = []
    for identity in identities:
        segment_id = identity["front_segment_id"]
        if bootstrap_segments is not None and segment_id not in bootstrap_segments:
            continue
        object_id = identity["object_id"]
        output, primary = generated[segment_id]
        destination = attempt_path(attempt, "condition_images", object_id)
        if destination != output:
            if destination.exists():
                raise FileExistsError(destination)
            output.rename(destination)
            primary = destination / primary.name
        items.append(_real_manifest_item(identity, primary, attempt))
    kept_segments = {item["front_segment_id"] for item in identities}
    if bootstrap_segments is not None:
        kept_segments &= bootstrap_segments
    for segment_id, (output, _) in generated.items():
        if segment_id not in kept_segments:
            shutil.rmtree(output)

    write_json(attempt_path(attempt, "data/condition_manifest.json"), {"items": items})
    log_step("condition_images", 'manifest: "data/condition_manifest.json"')
    log_step("condition_images", 'recognition: "data/recognition.json"')
    return completed_files(
        attempt, stage_outputs("condition_images"),
    )


def latest_completed_attempt(context):
    metadata = read_json(context.run_json)
    attempts = context.run_path(
        "stages", metadata["stage_directories"]["condition_images"], "attempts"
    )
    for root in sorted(attempts.glob("[!.]*"), reverse=True):
        record = root / "attempt.json"
        manifest = root / "data/condition_manifest.json"
        if record.is_file() and manifest.is_file() and read_json(record).get("status") == "completed":
            return root
    raise FileNotFoundError("no completed condition_images attempt to carry forward")




def run(
    context, attempt, planner_model=None, image_model=None, object_ids=None,
    item_ready=None, executor=None, submission_callback=None,
):
    planner_model = planner_model or pipeline_runtime().model
    image_runtime = image_generation_runtime()
    image_model = image_model or image_runtime.model
    args = (context, attempt, planner_model, image_model, object_ids)
    return run_real(
        *args,
        item_ready=item_ready,
        executor=executor,
        submission_callback=submission_callback,
    )


def run_real_segment_test(
    project_root, scene_id, run_id, segment_id, output_root,
    planner_model=None, image_model=None,
):
    context = RunContext.resume(project_root, scene_id, run_id)
    selected_attempt = context.stage_best("select_items")
    record = read_json(selected_attempt / "attempt.json")
    selected_manifest = selected_attempt / record["artifacts"]["manifest"]
    selected_root = selected_manifest.parent.parent
    selected = read_json(selected_manifest)
    selected_by_segment = {
        item["segment_id"]: item for item in selected["items"]
    }
    if segment_id not in selected_by_segment:
        raise KeyError(f"{segment_id} not found in selected items")
    segment_pairs = selected.get("segment_pairs")
    if segment_pairs is None:
        segment_pairs, _, _ = match_anonymous_segments_by_horizontal_range(selected)

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = AttemptContext(
        "segment_test", output_root, output_root, {}, "condition_images",
    )
    output, _, _ = _run_real_image_call(
        context, attempt, selected_root, selected,
        selected_by_segment[segment_id],
        planner_model or pipeline_runtime().model,
        image_model or image_generation_runtime().model,
        segment_pairs.get(segment_id),
    )
    return output


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a non-publishing condition-image test for one anonymous segment."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planner-model")
    parser.add_argument("--image-model")
    args = parser.parse_args(argv)
    output = run_real_segment_test(
        args.project_root, args.scene, args.run_id, args.segment_id, args.output,
        args.planner_model, args.image_model,
    )
    print(output)


if __name__ == "__main__":
    _main()
