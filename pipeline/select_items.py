#!/usr/bin/env python3
import base64
import html
import math
import os
import struct
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from pipeline.common import log_step, path, read_json, write_json
from pipeline.stage_io import (
    STAGE_OUTPUTS,
    dependency_file,
    completed_files,
    stage_outputs,
)
from pipeline.common import attempt_path


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def view_items(results_path, view, require_complete=False):
    results_path = Path(results_path)
    data = read_json(results_path)
    unmatched = data.get("unmatched_object_ids", [])
    if require_complete and unmatched:
        raise RuntimeError(
            f"{view} segmentation is missing required objects: {', '.join(unmatched)}"
        )
    source_root = results_path.parent.parent
    return {
        item["object_id"]: {
            "view": view,
            "score": item["score"],
            "mask_area_px": item.get("mask_area_px"),
            "bbox_xyxy": item["bbox_xyxy"],
            "crop_path": source_root / item["crop_path"],
            "mask_path": source_root / item["mask_path"],
        }
        for item in data["objects"]
    }


def _linked_view(item, view, destination, attempt):
    crop_link = destination / f"{view}.png"
    mask_link = destination / f"{view}_mask.png"
    link(item["crop_path"], crop_link)
    link(item["mask_path"], mask_link)
    linked = {
        "score": item["score"],
        "bbox_xyxy": item["bbox_xyxy"],
        "crop": str(crop_link.relative_to(attempt.temp_root)),
        "mask": str(mask_link.relative_to(attempt.temp_root)),
    }
    if item.get("mask_area_px") is not None:
        linked["mask_area_px"] = item["mask_area_px"]
    return linked


def match_anonymous_segments_by_horizontal_range(selected):
    """Pair anonymous front/top segments by horizontal range, depth, and area."""
    front = {
        item["segment_id"]: item["views"]["front"]
        for item in selected["items"]
    }
    top = {
        item["segment_id"]: item["views"]["topview"]
        for item in selected["topview_pool"]
    }
    if "table_0" not in front or "table_0" not in top:
        raise ValueError("anonymous matching requires table_0 in both views")
    if any("bbox_xyxy" not in view for view in [*front.values(), *top.values()]):
        raise ValueError("anonymous matching requires every bbox")

    def normalized_range(box, support):
        width = max(float(support[2]) - float(support[0]), 1.0)
        return (
            (float(box[0]) - float(support[0])) / width,
            (float(box[2]) - float(support[0])) / width,
        )

    front_ids = [segment_id for segment_id in front if segment_id != "table_0"]
    top_ids = [segment_id for segment_id in top if segment_id != "table_0"]
    pairs = {"table_0": "table_0"}
    if not front_ids or not top_ids:
        return pairs, top_ids, []

    def normalized_bottoms(items, segment_ids):
        bottoms = {
            segment_id: float(items[segment_id]["bbox_xyxy"][3])
            for segment_id in segment_ids
        }
        low, high = min(bottoms.values()), max(bottoms.values())
        span = high - low
        return {
            segment_id: (bottom - low) / span if span else 0.0
            for segment_id, bottom in bottoms.items()
        }

    front_bottoms = normalized_bottoms(front, front_ids)
    top_bottoms = normalized_bottoms(top, top_ids)
    use_mask_area = all(
        view.get("mask_area_px", 0) > 0
        for view in [*front.values(), *top.values()]
    )
    costs = []
    for front_segment_id in front_ids:
        front_range = normalized_range(
            front[front_segment_id]["bbox_xyxy"],
            front["table_0"]["bbox_xyxy"],
        )
        row = []
        for top_segment_id in top_ids:
            cost = sum(abs(a - b) for a, b in zip(
                front_range,
                normalized_range(
                    top[top_segment_id]["bbox_xyxy"],
                    top["table_0"]["bbox_xyxy"],
                ),
            ))
            cost += 0.25 * abs(
                front_bottoms[front_segment_id] - top_bottoms[top_segment_id]
            )
            if use_mask_area:
                front_area = (
                    front[front_segment_id]["mask_area_px"]
                    / front["table_0"]["mask_area_px"]
                )
                top_area = (
                    top[top_segment_id]["mask_area_px"]
                    / top["table_0"]["mask_area_px"]
                )
                cost += abs(math.sqrt(front_area) - math.sqrt(top_area))
            row.append(cost)
        costs.append(row)

    rows, columns = linear_sum_assignment(costs)
    matches = []
    used = set()
    for row, column in zip(rows, columns):
        front_segment_id = front_ids[row]
        top_segment_id = top_ids[column]
        pairs[front_segment_id] = top_segment_id
        used.add(top_segment_id)
        matches.append((front_segment_id, top_segment_id, costs[row][column]))
    return pairs, [segment_id for segment_id in top_ids if segment_id not in used], matches


def write_matching_overlay(front_path, top_path, selected, matches, output):
    def png_size(path):
        header = Path(path).read_bytes()[:24]
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"matching overlay input is not PNG: {path}")
        return struct.unpack(">II", header[16:24])

    front_width, front_height = png_size(front_path)
    top_width, top_height = png_size(top_path)
    front_items = {
        item["segment_id"]: item["views"]["front"]
        for item in selected["items"]
    }
    top_items = {
        item["segment_id"]: item["views"]["topview"]
        for item in selected["topview_pool"]
    }
    colors = ("#00ff88", "#00bfff", "#ffcc00", "#ff66cc", "#aa66ff", "#ff6633")
    front_data = base64.b64encode(Path(front_path).read_bytes()).decode("ascii")
    top_data = base64.b64encode(Path(top_path).read_bytes()).decode("ascii")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{front_width + top_width}" '
        f'height="{max(front_height, top_height)}">',
        f'<image href="data:image/png;base64,{front_data}" width="{front_width}" '
        f'height="{front_height}"/>',
        f'<image href="data:image/png;base64,{top_data}" x="{front_width}" '
        f'width="{top_width}" height="{top_height}"/>',
    ]
    for index, (front_id, top_id, cost) in enumerate(matches):
        front_box = front_items[front_id]["bbox_xyxy"]
        top_box = top_items[top_id]["bbox_xyxy"]
        start = ((front_box[0] + front_box[2]) / 2, (front_box[1] + front_box[3]) / 2)
        end = (
            front_width + (top_box[0] + top_box[2]) / 2,
            (top_box[1] + top_box[3]) / 2,
        )
        color = colors[index % len(colors)]
        label = html.escape(f"{front_id} → {top_id} ({cost:.3f})")
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        elements.extend((
            f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" '
            f'y2="{end[1]}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{midpoint[0]}" y="{midpoint[1]}" fill="{color}" '
            f'stroke="black" stroke-width="3" paint-order="stroke" '
            f'font-family="sans-serif" font-size="16">{label}</text>',
        ))
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_real(context, attempt):
    stage = "select_items"
    log_step(stage, "collecting anonymous front proposals and top-view pool")
    front = view_items(
        dependency_file(attempt, "segment_anonymous_instances", "results"),
        "front",
    )
    top_results = dependency_file(
        attempt, "segment_anonymous_instances", "topview_results",
    )
    top_overlay = dependency_file(
        attempt, "segment_anonymous_instances", "topview_overlay",
    )
    top = view_items(top_results, "topview")
    if not front:
        raise RuntimeError("anonymous front segmentation produced no proposals")
    if not top:
        raise RuntimeError("anonymous top-view segmentation produced no proposals")

    out = attempt_path(attempt, "selected_items")
    evidence = out / "evidence"
    evidence_paths = {
        "front_image": dependency_file(attempt, "reference_image", "image"),
        "front_overlay": dependency_file(
            attempt, "segment_anonymous_instances", "overlay"
        ),
        "topview_image": dependency_file(attempt, "topview_image", "topview_image"),
        "topview_overlay": top_overlay,
    }
    linked_evidence = {}
    for name, source in evidence_paths.items():
        destination = evidence / f"{name}.png"
        link(source, destination)
        linked_evidence[name] = str(destination.relative_to(attempt.temp_root))

    items = []
    for segment_id in sorted(front):
        item = front[segment_id]
        views = {
            "front": _linked_view(
                item, "front", out / segment_id, attempt,
            )
        }
        items.append({"segment_id": segment_id, "views": views})

    topview_pool = []
    for segment_id in sorted(top):
        item = top[segment_id]
        views = {
            "topview": _linked_view(
                item, "topview", out / "topview_pool" / segment_id, attempt,
            )
        }
        topview_pool.append({"segment_id": segment_id, "views": views})

    manifest = {
        "evidence": linked_evidence,
        "items": items,
        "topview_pool": topview_pool,
    }
    segment_pairs, unused_top, matches = (
        match_anonymous_segments_by_horizontal_range(manifest)
    )
    manifest.update({
        "segment_pairs": segment_pairs,
        "matches": [
            {
                "front_segment_id": front_segment_id,
                "topview_segment_id": topview_segment_id,
                "cost": cost,
            }
            for front_segment_id, topview_segment_id, cost in matches
        ],
        "unmatched_front_segment_ids": sorted(
            set(front) - set(segment_pairs)
        ),
        "unused_topview_segment_ids": unused_top,
        "matching_overlay": "selected_items/matching_overlay.svg",
    })
    write_matching_overlay(
        evidence_paths["front_overlay"],
        evidence_paths["topview_overlay"],
        manifest,
        matches,
        attempt_path(attempt, manifest["matching_overlay"]),
    )
    write_json(out / "manifest.json", manifest)
    log_step(
        stage,
        f"matched {len(matches)} object pairs from {len(items)} front and "
        f"{len(topview_pool)} top-view proposals",
    )
    return completed_files(attempt, stage_outputs(stage))


def run(context, attempt):
    return run_real(context, attempt)


if __name__ == "__main__":
    run()
