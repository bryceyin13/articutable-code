#!/usr/bin/env python3
from collections import deque
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from pipeline.common import log_step, path
from pipeline.stage_io import (
    MERGED_SEGMENTATION_OUTPUTS,
    STAGE_OUTPUTS,
    dependency_file,
    completed_files,
    stage_outputs,
)
from pipeline.common import attempt_path


DEFAULT_SAM3_CHECKPOINT = str(Path(__file__).resolve().parents[1] / "models/sam3.pt")
FALLBACK_PROMPT_PENALTY = 0.15
ANONYMOUS_OBJECT_PROMPT = os.environ.get(
    "SAM3_ANONYMOUS_OBJECT_PROMPT", "tabletop item",
)
ANONYMOUS_TABLE_PROMPT = "table"
TABLETOP_BBOX_PROMPT = "table top and apron"
_SAM3_DISPATCHER = None


def set_sam3_dispatcher(dispatcher):
    """Route SAM3 inference through a resident worker pool when configured."""
    global _SAM3_DISPATCHER
    previous = _SAM3_DISPATCHER
    _SAM3_DISPATCHER = dispatcher
    return previous


def segmentation_prompt(text):
    return " ".join(text.replace("_", " ").replace("-", " ").split()).strip(" .")


def segmentation_prompt_variants(label):
    """Use image-grounded identity first and the full category as fallback."""
    if isinstance(label, str):
        label = {"category": label}
    prompts = [label.get("visual_evidence"), label["category"]]
    return list(dict.fromkeys(
        segmentation_prompt(prompt) for prompt in prompts if prompt and prompt.strip()
    ))


def segmentation_component_prompt(label):
    """Build a specific fallback from multiple observed visible components."""
    components = list(dict.fromkeys(
        segmentation_prompt(component)
        for component in label.get("visible_components", [])
        if isinstance(component, str) and component.strip()
    ))
    return " and ".join(components) if len(components) >= 2 else None


def _prompt_penalty(variant_index):
    return FALLBACK_PROMPT_PENALTY if int(variant_index) else 0.0


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def select_complete_table(candidates):
    return max(
        candidates,
        key=lambda item: (box_area(item["bbox_xyxy"]), item["score"]),
        default=None,
    )


def select_tabletop_bbox(candidates, complete_table):
    """Select the compact, full-width tabletop proposal instead of the whole table."""
    x0, y0, x1, y1 = complete_table["bbox_xyxy"]
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    compatible = [
        item for item in candidates
        if item["bbox_xyxy"][2] - item["bbox_xyxy"][0] >= 0.7 * width
        and abs(item["bbox_xyxy"][1] - y0) <= 0.15 * height
        and item["bbox_xyxy"][3] < y0 + 0.9 * height
    ]
    return min(
        compatible,
        key=lambda item: (box_area(item["bbox_xyxy"]), -item["score"]),
        default=None,
    )


def box_iou(left, right):
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = box_area([x0, y0, x1, y1])
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union else 0.0


def topview_geometric_prior(label, support, support_box, image_size, padding=1.5):
    """Convert blueprint placement and size into a loose normalized box prompt."""
    x0, y0, x1, y1 = support_box
    image_width, image_height = image_size
    support_width = max(1.0, x1 - x0)
    support_height = max(1.0, y1 - y0)
    xy = label.get("center_xy_norm") or [0.0, 0.0]
    center_x = (x0 + x1) / 2 + float(xy[0]) * support_width
    center_y = (y0 + y1) / 2 - float(xy[1]) * support_height
    object_size = label.get("size_cm") or [1.0, 1.0]
    support_size = support.get("size_cm") or [100.0, 100.0]
    size_fraction = max(
        float(object_size[0]) / max(float(support_size[0]), 1e-6),
        float(object_size[1]) / max(float(support_size[1]), 1e-6),
    )
    expected_area = (
        max(float(object_size[0]) * float(object_size[1]), 1e-6)
        / max(float(support_size[0]) * float(support_size[1]), 1e-6)
        * support_width * support_height
    )
    box_width = min(float(image_width), padding * size_fraction * support_width)
    box_height = min(float(image_height), padding * size_fraction * support_height)
    return {
        "box": [center_x / image_width, center_y / image_height,
                box_width / image_width, box_height / image_height],
        "box_size_px": [box_width, box_height],
        "center": [center_x, center_y],
        "expected_area": expected_area,
    }


def select_geometric_candidate(candidates, prior):
    """Prefer the proposal whose position and footprint best match the blueprint."""
    if not candidates:
        return None
    center_x, center_y = prior["center"]
    prompt_width = max(1.0, prior["box_size_px"][0])
    prompt_height = max(1.0, prior["box_size_px"][1])

    def cost(item):
        x0, y0, x1, y1 = item["bbox_xyxy"]
        area_cost = abs(math.log(max(box_area(item["bbox_xyxy"]), 1.0) / prior["expected_area"]))
        distance = math.hypot(
            ((x0 + x1) / 2 - center_x) / prompt_width,
            ((y0 + y1) / 2 - center_y) / prompt_height,
        )
        return area_cost + 0.5 * distance

    return min(candidates, key=cost)


def candidate_overlap(left, right):
    """Measure whether two prompt results describe the same physical region."""
    left_mask = left.get("mask")
    right_mask = right.get("mask")
    if (
        left_mask is not None
        and right_mask is not None
        and getattr(left_mask, "shape", None) == getattr(right_mask, "shape", None)
    ):
        smaller = min(int(left_mask.sum()), int(right_mask.sum()))
        if smaller:
            return float((left_mask & right_mask).sum()) / smaller
    return box_iou(left["bbox_xyxy"], right["bbox_xyxy"])


def near_duplicate_masks(
    left, right, iou_threshold=0.90, containment_threshold=0.70,
):
    """Return whether the original foreground masks substantially overlap."""
    left_mask = left.get("mask")
    right_mask = right.get("mask")
    if (
        left_mask is None
        or right_mask is None
        or getattr(left_mask, "shape", None) != getattr(right_mask, "shape", None)
    ):
        return False
    left_area = int(left_mask.sum())
    right_area = int(right_mask.sum())
    if not left_area or not right_area:
        return False
    intersection = int((left_mask & right_mask).sum())
    union = left_area + right_area - intersection
    iou = intersection / union
    containment = intersection / min(left_area, right_area)
    return iou >= iou_threshold or containment >= containment_threshold


def merge_near_duplicate_masks(items):
    """Merge proposals whose original foreground masks overlap."""
    kept = []
    merged = []
    for item in sorted(
        items,
        key=lambda value: (
            -int(value["mask"].sum()),
            -value["score"],
            tuple(value["bbox_xyxy"]),
            value.get("object_id", value.get("segment_id", "")),
        ),
    ):
        target = next(
            (
                candidate for candidate in kept
                if near_duplicate_masks(item, candidate)
            ),
            None,
        )
        if target is not None:
            target["mask"] = target["mask"] | item["mask"]
            target["bbox_xyxy"] = [
                min(target["bbox_xyxy"][0], item["bbox_xyxy"][0]),
                min(target["bbox_xyxy"][1], item["bbox_xyxy"][1]),
                max(target["bbox_xyxy"][2], item["bbox_xyxy"][2]),
                max(target["bbox_xyxy"][3], item["bbox_xyxy"][3]),
            ]
            target["score"] = max(
                target.get("score", 0), item.get("score", 0),
            )
            merged.append(item)
        else:
            kept.append(item)
    return kept, merged


def suppress_near_duplicate_masks(items):
    """Compatibility alias; duplicate candidates are merged, not discarded."""
    return merge_near_duplicate_masks(items)


def merge_contained_masks(
    items, tolerance_ratio=0.01, minimum_tolerance_px=2,
    mask_containment_threshold=0.70,
):
    """Merge nested boxes when their original masks describe the same region."""
    kept = []
    merged = []
    for item in sorted(
        items, key=lambda value: box_area(value["bbox_xyxy"]), reverse=True,
    ):
        x0, y0, x1, y1 = item["bbox_xyxy"]
        target = None
        for outer in kept:
            ox0, oy0, ox1, oy1 = outer["_merge_anchor_bbox"]
            item_mask = item.get("mask")
            outer_mask = outer.get("mask")
            tolerance_x = max(
                minimum_tolerance_px, tolerance_ratio * max(ox1 - ox0, 1),
            )
            tolerance_y = max(
                minimum_tolerance_px, tolerance_ratio * max(oy1 - oy0, 1),
            )
            if (
                ox0 - tolerance_x <= x0
                and oy0 - tolerance_y <= y0
                and ox1 + tolerance_x >= x1
                and oy1 + tolerance_y >= y1
                and item_mask is not None
                and outer_mask is not None
                and getattr(item_mask, "shape", None)
                == getattr(outer_mask, "shape", None)
                and candidate_overlap(item, outer) >= mask_containment_threshold
            ):
                target = outer
                break
        if target is not None:
            target["mask"] = target["mask"] | item["mask"]
            target["bbox_xyxy"] = [
                min(target["bbox_xyxy"][0], x0),
                min(target["bbox_xyxy"][1], y0),
                max(target["bbox_xyxy"][2], x1),
                max(target["bbox_xyxy"][3], y1),
            ]
            target["score"] = max(target.get("score", 0), item.get("score", 0))
            merged.append(item)
        else:
            item["_merge_anchor_bbox"] = list(item["bbox_xyxy"])
            kept.append(item)
    for item in kept:
        item.pop("_merge_anchor_bbox", None)
    return kept, merged


def clean_small_disconnected_regions(mask, dominant_area_ratio=0.90):
    """Drop tiny 8-disconnected islands, or reject genuinely split masks."""
    ys, xs = mask.nonzero()
    if not len(xs):
        return mask
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    remaining = crop.copy()
    components = []
    while remaining.any():
        component = np.zeros_like(crop)
        start_y, start_x = np.argwhere(remaining)[0]
        component[start_y, start_x] = True
        remaining[start_y, start_x] = False
        queue = deque([(int(start_y), int(start_x))])
        while queue:
            y, x = queue.popleft()
            for next_y in range(max(0, y - 1), min(crop.shape[0], y + 2)):
                for next_x in range(max(0, x - 1), min(crop.shape[1], x + 2)):
                    if remaining[next_y, next_x]:
                        remaining[next_y, next_x] = False
                        component[next_y, next_x] = True
                        queue.append((next_y, next_x))
        components.append(component)
    if len(components) == 1:
        return mask
    largest = max(components, key=np.count_nonzero)
    if int(largest.sum()) < dominant_area_ratio * int(mask.sum()):
        return None
    cleaned = np.zeros_like(mask)
    cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = largest
    return cleaned


def filter_anonymous_candidates(
    items, table_bbox, tabletop_bbox, image_height, view,
    bottom_tolerance_ratio=0.08, table_width_ratio=0.90,
    min_mask_fill_ratio=0.05, below_tabletop_area_ratio=0.70,
):
    """Reject support-sized, implausibly sparse, and below-tabletop proposals."""
    table_area = box_area(table_bbox)
    table_width = max(table_bbox[2] - table_bbox[0], 1)
    max_bottom = (
        tabletop_bbox[3] + bottom_tolerance_ratio * image_height
        if view == "frontview" and tabletop_bbox is not None
        else None
    )
    tabletop_bottom = (
        int(tabletop_bbox[3])
        if tabletop_bbox is not None
        else None
    )
    kept = []
    oversized = []
    table_width_like = []
    below_tabletop = []
    sparse = []
    disconnected = []
    for item in items:
        item_box_area = box_area(item["bbox_xyxy"])
        mask = item.get("mask")
        mask_area = int(mask.sum()) if mask is not None else 0
        if (
            mask_area
            and tabletop_bottom is not None
            and int(mask[tabletop_bottom:, :].sum())
            >= below_tabletop_area_ratio * mask_area
        ):
            below_tabletop.append(item)
            continue
        if (
            mask is not None
            and mask_area < min_mask_fill_ratio * item_box_area
        ):
            sparse.append(item)
            continue
        if mask is not None:
            cleaned_mask = clean_small_disconnected_regions(mask)
            if cleaned_mask is None:
                disconnected.append(item)
                continue
            item["mask"] = cleaned_mask
        if item_box_area > table_area:
            oversized.append(item)
        elif item["bbox_xyxy"][2] - item["bbox_xyxy"][0] >= table_width_ratio * table_width:
            table_width_like.append(item)
        elif max_bottom is not None and item["bbox_xyxy"][3] > max_bottom:
            below_tabletop.append(item)
        else:
            kept.append(item)
    return (
        kept, oversized, table_width_like, below_tabletop, sparse, disconnected,
    )


def spatially_sorted_candidates(items):
    """Give anonymous proposals deterministic view-local reading order."""
    return sorted(items, key=lambda item: (
        item["bbox_xyxy"][1] + item["bbox_xyxy"][3],
        item["bbox_xyxy"][0] + item["bbox_xyxy"][2],
        -item["score"],
    ))


def merge_candidate_evidence(items, overlap=0.70):
    """Cluster duplicate masks while retaining each object's own detection."""
    clusters = []
    for item in sorted(items, key=lambda value: value["score"], reverse=True):
        cluster = next(
            (
                kept for kept in clusters
                if any(
                    candidate_overlap(item, evidence["candidate"]) >= overlap
                    for evidence in kept["label_evidence"].values()
                )
            ),
            None,
        )
        label_id = item["label_id"]
        evidence = {
            "prompt": item["prompt"],
            "variant_index": item["variant_index"],
            "score": item["score"],
            "candidate": item,
        }
        if cluster is None:
            clusters.append({
                "bbox_xyxy": item["bbox_xyxy"],
                "score": item["score"],
                "label_evidence": {label_id: evidence},
            })
            continue
        previous = cluster["label_evidence"].get(label_id)
        evidence_cost = (1.0 - evidence["score"]) + _prompt_penalty(evidence["variant_index"])
        previous_cost = (
            (1.0 - previous["score"]) + _prompt_penalty(previous["variant_index"])
            if previous else float("inf")
        )
        if evidence_cost < previous_cost:
            cluster["label_evidence"][label_id] = evidence
    return clusters


def _layout_cost(label, candidate, support_box, view):
    x0, y0, x1, y1 = support_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    xy = label.get("center_xy_norm") or [0.0, 0.0]
    target_x = (x0 + x1) / 2 + float(xy[0]) * width
    target_y = (y0 + y1) / 2 - float(xy[1]) * height
    bx0, by0, bx1, by1 = candidate["bbox_xyxy"]
    candidate_x = (bx0 + bx1) / 2
    candidate_y = (by0 + by1) / 2
    dx = (candidate_x - target_x) / width
    dy = (candidate_y - target_y) / height
    # Front-view object height makes vertical centers unreliable; horizontal
    # layout remains a useful low-weight identity tie-break.
    return abs(dx) + (abs(dy) if view == "topview" else 0.15 * abs(dy))


def match_global_candidates(
    labels,
    candidates,
    support_box,
    view,
    unmatched_cost=1.0,
):
    """Assign all object labels jointly from semantic evidence and layout."""
    if not labels or not candidates:
        return []
    costs = []
    components = {}
    for label_index, label in enumerate(labels):
        row = []
        for candidate_index, cluster in enumerate(candidates):
            evidence = cluster.get("label_evidence", {}).get(label["object_id"])
            if evidence is None:
                row.append(unmatched_cost + 10.0)
                continue
            detection = evidence.get("candidate", cluster)
            semantic = (1.0 - float(evidence["score"])) + _prompt_penalty(
                evidence.get("variant_index", 0)
            )
            layout = _layout_cost(label, detection, support_box, view)
            total = semantic + 0.10 * layout
            row.append(total)
            components[(label_index, candidate_index)] = {
                "semantic": semantic,
                "layout": layout,
                "total": total,
            }
        row.extend([unmatched_cost] * len(labels))
        costs.append(row)
    result = []
    for label_index, candidate_index in minimum_cost_assignment(costs):
        if candidate_index >= len(candidates):
            continue
        cost = costs[label_index][candidate_index]
        if cost >= unmatched_cost:
            continue
        label = labels[label_index]
        evidence = candidates[candidate_index]["label_evidence"].get(label["object_id"])
        if evidence is None:
            continue
        result.append((
            label,
            evidence.get("candidate", candidates[candidate_index]),
            cost,
            evidence,
            components[(label_index, candidate_index)],
        ))
    return result


def minimum_cost_assignment(costs):
    """Return an optimal one-to-one assignment for a rectangular cost matrix."""
    if not costs or not costs[0]:
        return []
    rows, columns = len(costs), len(costs[0])
    transposed = rows > columns
    matrix = (
        [[costs[row][column] for row in range(rows)] for column in range(columns)]
        if transposed else costs
    )
    n, m = len(matrix), len(matrix[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        p[0] = row
        column = 0
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[column] = True
            active_row = p[column]
            delta, next_column = float("inf"), 0
            for candidate_column in range(1, m + 1):
                if used[candidate_column]:
                    continue
                current = matrix[active_row - 1][candidate_column - 1] - u[active_row] - v[candidate_column]
                if current < minimum[candidate_column]:
                    minimum[candidate_column] = current
                    way[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta, next_column = minimum[candidate_column], candidate_column
            for candidate_column in range(m + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    pairs = [(p[column] - 1, column - 1) for column in range(1, m + 1) if p[column]]
    return [(column, row) for row, column in pairs] if transposed else pairs


def build_result(width, height, confidence, objects, image_artifact):
    for item in objects:
        for key in ("mask_path", "crop_path"):
            member = Path(item[key])
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"SAM result member must be attempt-relative: {item[key]}")
    return {
        "image_artifact": image_artifact,
        "img_width": width,
        "img_height": height,
        "model": "sam3",
        "confidence_threshold": confidence,
        "object_names": [item["object_id"] for item in objects],
        "objects": objects,
    }


def recognition_identity_mapping(
    recognition, segment_field, available_ids,
    allow_missing_object_segments=False,
):
    """Validate the formal-to-anonymous map emitted by the real-image VLM call."""
    table = recognition.get("table")
    objects = recognition.get("objects")
    if not isinstance(table, dict) or not isinstance(objects, list):
        raise ValueError("real recognition must define table and objects")
    pairs = []
    formal_ids = set()
    segment_ids = set()
    available_ids = set(available_ids)
    for index, item in enumerate([table, *objects]):
        operability_field = "operability_type" if index == 0 else "operability_candidate"
        missing = [
            field for field in ("object_id", "category", operability_field)
            if not isinstance(item.get(field), str) or not item[field].strip()
        ]
        if missing:
            raise ValueError(
                f"real recognition item is missing required field: {missing[0]}"
            )
        object_id = item["object_id"]
        if object_id in formal_ids:
            raise ValueError(f"duplicate formal object_id in real recognition: {object_id}")
        formal_ids.add(object_id)
        segment_id = item.get(segment_field)
        if allow_missing_object_segments and index > 0 and segment_id is None:
            continue
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(
                f"real recognition item is missing required field: {segment_field}"
            )
        if segment_id in segment_ids:
            raise ValueError(
                f"{segment_field} must be one-to-one; duplicate value: {segment_id}"
            )
        if segment_id not in available_ids:
            raise ValueError(
                f"{segment_field} references unknown anonymous segment: {segment_id}"
            )
        segment_ids.add(segment_id)
        pairs.append(({
            **item,
            "operability_type": item[operability_field],
        }, segment_id))
    return pairs


def rematch_topview_identity_mapping(recognition, front, topview):
    """Match top-view segments to recognized front IDs using position and area."""
    front_by_id = {item["object_id"]: item for item in front["objects"]}
    top_by_id = {item["object_id"]: item for item in topview["objects"]}
    if "table_0" not in front_by_id or "table_0" not in top_by_id:
        raise ValueError("top-view rematching requires table_0 in both views")
    if any(
        item.get("mask_area_px", 0) <= 0
        for item in [*front_by_id.values(), *top_by_id.values()]
    ):
        raise ValueError("top-view rematching requires mask_area_px from anonymous segmentation")

    def relative_values(item, table):
        box, support = item["bbox_xyxy"], table["bbox_xyxy"]
        width = max(float(support[2]) - float(support[0]), 1.0)
        return (
            (float(box[0]) - float(support[0])) / width,
            (float(box[2]) - float(support[0])) / width,
            float(item["mask_area_px"]) / float(table["mask_area_px"]),
        )

    front_ids = [item for item in front_by_id if item != "table_0"]
    top_ids = [item for item in top_by_id if item != "table_0"]

    def normalized_bottoms(items, item_ids):
        if not item_ids:
            return {}
        bottoms = {
            item_id: float(items[item_id]["bbox_xyxy"][3])
            for item_id in item_ids
        }
        low, high = min(bottoms.values()), max(bottoms.values())
        span = high - low
        return {
            item_id: (bottom - low) / span if span else 0.0
            for item_id, bottom in bottoms.items()
        }

    front_values = {
        item: relative_values(front_by_id[item], front_by_id["table_0"])
        for item in front_ids
    }
    top_values = {
        item: relative_values(top_by_id[item], top_by_id["table_0"])
        for item in top_ids
    }
    front_bottoms = normalized_bottoms(front_by_id, front_ids)
    top_bottoms = normalized_bottoms(top_by_id, top_ids)
    costs = [[
        abs(front_values[front_id][0] - top_values[top_id][0])
        + abs(front_values[front_id][1] - top_values[top_id][1])
        + 0.25 * abs(front_bottoms[front_id] - top_bottoms[top_id])
        + 0.1 * abs(math.log(front_values[front_id][2] / top_values[top_id][2]))
        for top_id in top_ids
    ] for front_id in front_ids]
    pairs = {"table_0": "table_0"}
    for front_index, top_index in minimum_cost_assignment(costs):
        pairs[front_ids[front_index]] = top_ids[top_index]

    mapping = []
    for index, item in enumerate([recognition["table"], *recognition["objects"]]):
        front_segment_id = item["front_segment_id"]
        top_segment_id = pairs.get(front_segment_id)
        if index == 0 and top_segment_id != "table_0":
            raise ValueError("table_0 was not preserved during top-view rematching")
        if top_segment_id is None:
            continue
        operability_field = "operability_type" if index == 0 else "operability_candidate"
        mapping.append(({
            **item,
            "operability_type": item[operability_field],
        }, top_segment_id))
        log_step(
            "segment_topview_instances",
            f"area-aware rematch {front_segment_id} -> {top_segment_id}",
        )
    return mapping


def _read_json(file_path):
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def _write_json(file_path, data):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def enrich_blueprint_with_tabletop_bbox(source, segmentation, destination):
    blueprint = _read_json(source)
    tabletop = segmentation.get("front_tabletop_bbox")
    if tabletop is not None:
        blueprint["table"]["front_tabletop_bbox"] = tabletop
    _write_json(destination, blueprint)


def _attempt_member(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"anonymous segmentation member is not attempt-relative: {relative}")
    return root / relative


def _safe_object_id(value):
    if Path(value).name != value or value in {"", ".", ".."} or "\\" in value:
        raise ValueError(f"unsafe formal object_id: {value}")
    return value


def _overlay_font(image_size):
    from PIL import ImageFont

    size = max(20, min(image_size) // 45)
    for name in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def mask_overlay_path(output_path):
    output_path = Path(output_path)
    suffix = "_overlay"
    if not output_path.stem.endswith(suffix):
        raise ValueError(f"expected an *_overlay output path: {output_path}")
    return output_path.with_name(
        f"{output_path.stem[:-len(suffix)]}_mask_overlay{output_path.suffix}"
    )


def draw_labeled_overlay(image, objects, output_path, attempt_root, annotations=True):
    from PIL import Image, ImageDraw

    colors = [
        (255, 70, 70, 115),
        (70, 255, 120, 115),
        (70, 170, 255, 115),
        (255, 220, 70, 115),
        (180, 100, 255, 115),
        (255, 140, 70, 115),
    ]
    overlay = (
        image.convert("RGBA")
        if isinstance(image, Image.Image)
        else Image.open(image).convert("RGBA")
    )
    line = ImageDraw.Draw(overlay)
    font = _overlay_font(overlay.size)
    line_width = max(3, min(overlay.size) // 500)
    occupied_labels = []

    def overlaps(first, second):
        return not (
            first[2] <= second[0] or second[2] <= first[0]
            or first[3] <= second[1] or second[3] <= first[1]
        )

    for index, item in enumerate(objects):
        color = colors[index % len(colors)]
        mask = Image.open(attempt_root / item["mask_path"]).convert("L")
        tint = Image.new("RGBA", overlay.size, color)
        transparent = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        overlay.alpha_composite(Image.composite(tint, transparent, mask))
        if not annotations:
            continue
        line.rectangle(
            item["bbox_xyxy"], outline=color[:3] + (255,), width=line_width,
        )
        label = item["object_id"]
        padding = max(3, line_width)
        natural = line.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = natural[2] - natural[0] + 2 * padding
        label_height = natural[3] - natural[1] + 2 * padding
        x0, y0, x1, y1 = [int(value) for value in item["bbox_xyxy"]]
        candidates = [
            (x0, y0),
            (x1 - label_width, y0),
            (x0, y1 - label_height),
            (x1 - label_width, y1 - label_height),
            (x1 + padding, (y0 + y1 - label_height) // 2),
            (x0 - label_width - padding, (y0 + y1 - label_height) // 2),
        ]
        candidates.extend(
            (padding, padding + row * (label_height + padding))
            for row in range(len(objects))
        )
        background = None
        for candidate_x, candidate_y in candidates:
            left = min(max(0, candidate_x), max(0, overlay.width - label_width))
            top = min(max(0, candidate_y), max(0, overlay.height - label_height))
            candidate = [left, top, left + label_width, top + label_height]
            if all(not overlaps(candidate, previous) for previous in occupied_labels):
                background = candidate
                break
        if background is None:
            background = candidate
        occupied_labels.append(background)
        x = background[0] + padding - natural[0]
        y = background[1] + padding - natural[1]
        line.rectangle(
            background,
            fill=(0, 0, 0, 225),
            outline=color[:3] + (255,),
            width=line_width,
        )
        line.text(
            (x, y),
            label,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path)


def _remap(
    context, attempt, stage, anonymous_stage, image_dependency, segment_field,
    output_prefix, allow_missing_object_segments=False, rematch_topview=False,
    anonymous_artifact="results", output_subdir=None, publish=True,
    front_anonymous_artifact="results",
):
    anonymous_path = dependency_file(
        attempt, anonymous_stage, anonymous_artifact,
    )
    source_image = dependency_file(attempt, *image_dependency)
    recognition_path = dependency_file(attempt, "condition_images", "recognition")
    anonymous = _read_json(anonymous_path)
    recognition = _read_json(recognition_path)
    anonymous_by_id = {item["object_id"]: item for item in anonymous["objects"]}
    if len(anonymous_by_id) != len(anonymous["objects"]):
        raise ValueError("anonymous segmentation contains duplicate segment IDs")
    if rematch_topview:
        front = _read_json(dependency_file(
            attempt, "segment_anonymous_instances", front_anonymous_artifact,
        ))
        mapping = rematch_topview_identity_mapping(recognition, front, anonymous)
    else:
        mapping = recognition_identity_mapping(
            recognition, segment_field, anonymous_by_id,
            allow_missing_object_segments=allow_missing_object_segments,
        )

    anonymous_root = anonymous_path.parents[1]
    output_root = (
        attempt.temp_root / output_subdir if output_subdir else attempt.temp_root
    )
    mask_root = output_root / "masks"
    crop_root = output_root / "crops"
    mask_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)
    objects = []
    for identity, segment_id in mapping:
        object_id = _safe_object_id(identity["object_id"])
        source = anonymous_by_id[segment_id]
        mask_file = mask_root / f"{output_prefix}{object_id}_mask.png"
        crop_file = crop_root / f"{output_prefix}{object_id}_crop.png"
        shutil.copy2(
            _attempt_member(anonymous_root, source["mask_path"]), mask_file,
        )
        shutil.copy2(
            _attempt_member(anonymous_root, source["crop_path"]), crop_file,
        )
        objects.append({
            **source,
            "source_segment_id": segment_id,
            "object_id": object_id,
            "class_name": identity["category"],
            "operability_type": identity["operability_type"],
            "mask_path": str(mask_file.relative_to(output_root)),
            "crop_path": str(crop_file.relative_to(output_root)),
        })

    result = build_result(
        anonymous["img_width"],
        anonymous["img_height"],
        anonymous["confidence_threshold"],
        objects,
        anonymous["image_artifact"],
    )
    result["segmentation_mode"] = "anonymous_segments_remapped_by_real_recognition"
    result["source_segmentation_mode"] = anonymous.get("segmentation_mode")
    if anonymous.get("tabletop_mask_path"):
        tabletop_mask = mask_root / f"{output_prefix}tabletop_support_mask.png"
        shutil.copy2(
            _attempt_member(anonymous_root, anonymous["tabletop_mask_path"]),
            tabletop_mask,
        )
        result["tabletop_mask_path"] = str(
            tabletop_mask.relative_to(output_root)
        )
        result["tabletop_bbox_xyxy"] = anonymous.get("tabletop_bbox_xyxy")
    mapped_ids = {identity["object_id"] for identity, _ in mapping}
    result["unmatched_object_ids"] = [
        item["object_id"]
        for item in [recognition["table"], *recognition["objects"]]
        if item["object_id"] not in mapped_ids
    ]
    contract = STAGE_OUTPUTS[stage]
    outputs = contract
    results_path = output_root / outputs["results"].relative_path
    _write_json(results_path, result)
    if stage == "segment_instances":
        enrich_blueprint_with_tabletop_bbox(
            dependency_file(attempt, "blueprint", "blueprint"),
            result,
            output_root / outputs["blueprint"].relative_path,
        )
    if not publish:
        return {}
    artifacts = completed_files(attempt, contract)
    for name, relative in artifacts.items():
        log_step(stage, f'{name}: "{Path(str(attempt.final_root)) / relative}"')
    return artifacts


def sam3_command():
    if os.environ.get("SAM3_PYTHON"):
        return [os.environ["SAM3_PYTHON"]]
    env_name = os.environ.get("SAM3_CONDA_ENV")
    if not env_name:
        raise RuntimeError("set SAM3_PYTHON or SAM3_CONDA_ENV")
    for root in [Path.home() / ".conda" / "envs", Path.home() / "miniconda3" / "envs"]:
        python = root / env_name / "bin" / "python"
        if python.exists():
            return [str(python)]
    if shutil.which("conda"):
        return ["conda", "run", "-n", env_name, "python"]
    raise FileNotFoundError(f"cannot find Python for SAM3_CONDA_ENV={env_name}")


def _run(
    context, attempt, stage, image_dependency, labels_dependency, output_prefix,
    view, anonymous=False, output_subdir=None, publish=True,
):
    log_step(stage, "using SAM3 segmentation")
    cmd = sam3_command() + ["-m", "pipeline.sam3_worker"]
    log_step(stage, "command: " + " ".join(f'"{part}"' for part in cmd))
    image_path = dependency_file(attempt, *image_dependency)
    output_root = (
        attempt.temp_root / output_subdir if output_subdir else attempt.temp_root
    )
    final_root = (
        attempt.final_root / output_subdir if output_subdir else attempt.final_root
    )
    results_path = output_root / f"data/{output_prefix}segmentation_results.json"
    env = {
        **os.environ,
        "SAM3_CHECKPOINT": os.environ.get("SAM3_CHECKPOINT", DEFAULT_SAM3_CHECKPOINT),
        "SAM3_IMAGE": str(image_path.resolve()),
        "SAM3_ANONYMOUS": "1" if anonymous else "0",
        "SAM3_OUTPUT_PREFIX": output_prefix,
        "SAM3_RESULTS": str(results_path),
        "SAM3_OVERLAY": str(output_root / f"data/{output_prefix}segmentation_overlay.png"),
        "SAM3_MASK_ROOT": str(output_root / "masks"),
        "SAM3_CROP_ROOT": str(output_root / "crops"),
        "SAM3_ATTEMPT_ROOT": str(output_root.resolve()),
        "SAM3_FINAL_ROOT": str(final_root.resolve()),
        "SAM3_IMAGE_ARTIFACT": ".".join(image_dependency),
        "SAM3_VIEW": view,
    }
    visible_devices = os.environ.get("SAM3_CUDA_VISIBLE_DEVICES")
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    if labels_dependency is not None:
        env["SAM3_LABELS"] = str(
            dependency_file(attempt, *labels_dependency).resolve()
        )
    else:
        env.pop("SAM3_LABELS", None)
    semantic_tag_images = (
        _SAM3_DISPATCHER.semantic_images(context)
        if anonymous
        and _SAM3_DISPATCHER is not None
        and hasattr(_SAM3_DISPATCHER, "semantic_images")
        else None
    )
    if anonymous and semantic_tag_images:
        env["SAM3_SEMANTIC_IMAGES"] = json.dumps(semantic_tag_images)
    if _SAM3_DISPATCHER is None:
        result = subprocess.run(cmd, cwd=context.project_root, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                "SAM3 segmentation failed; see the [sam3_segment] ERROR line above for the actual cause."
            )
    else:
        _SAM3_DISPATCHER.run(env)
    if stage == "segment_instances":
        contract = STAGE_OUTPUTS[stage]
        enrich_blueprint_with_tabletop_bbox(
            dependency_file(attempt, "blueprint", "blueprint"),
            _read_json(results_path),
            output_root / contract["blueprint"].relative_path,
        )
    if not publish:
        return {}
    artifacts = completed_files(attempt, STAGE_OUTPUTS[stage])
    for name, relative in artifacts.items():
        log_step(stage, f'{name}: "{Path(str(attempt.final_root)) / relative}"')
    return artifacts




def run_anonymous_views(context, attempt):
    """Segment front and top views, then publish them as one atomic stage."""
    _run(
        context, attempt, "segment_anonymous_instances",
        ("reference_image", "image"), None, "anonymous_", "frontview",
        anonymous=True, output_subdir="front", publish=False,
    )
    _run(
        context, attempt, "segment_anonymous_topview_instances",
        ("topview_image", "topview_image"), None, "anonymous_topview_",
        "topview", anonymous=True, output_subdir="top", publish=False,
    )
    return completed_files(
        attempt,
        MERGED_SEGMENTATION_OUTPUTS["segment_anonymous_instances"],
    )




def run_remapped_views(context, attempt):
    """Remap front and top masks, then publish them as one atomic stage."""
    _remap(
        context, attempt, "segment_instances", "segment_anonymous_instances",
        ("reference_image", "image"), "front_segment_id", "",
        output_subdir="front", publish=False,
    )
    _remap(
        context, attempt, "segment_topview_instances",
        "segment_anonymous_instances", ("topview_image", "topview_image"),
        "topview_segment_id", "topview_",
        allow_missing_object_segments=True,
        rematch_topview=True,
        anonymous_artifact="topview_results",
        front_anonymous_artifact="results",
        output_subdir="top",
        publish=False,
    )
    return completed_files(
        attempt, MERGED_SEGMENTATION_OUTPUTS["segment_instances"],
    )
