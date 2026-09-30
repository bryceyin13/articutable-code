#!/usr/bin/env python3
import json
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pipeline.segmentation import (
    ANONYMOUS_OBJECT_PROMPT,
    ANONYMOUS_TABLE_PROMPT,
    TABLETOP_BBOX_PROMPT,
    box_area,
    build_result,
    draw_labeled_overlay,
    filter_anonymous_candidates,
    match_global_candidates,
    merge_candidate_evidence,
    segmentation_component_prompt,
    segmentation_prompt_variants,
    select_geometric_candidate,
    select_complete_table,
    select_tabletop_bbox,
    spatially_sorted_candidates,
    merge_contained_masks,
    merge_near_duplicate_masks,
    mask_overlay_path,
    topview_geometric_prior,
)
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


ROOT = Path(__file__).resolve().parents[1]
_MODEL_CACHE = {}
_SAM2_MODEL_CACHE = {}
DEFAULT_SAM2_CHECKPOINT = (
    ROOT.parent / "third_party/sam2.1/sam2.1_hiera_large.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
TABLETOP_SURFACE_PROMPT = "tabletop surface"
TOPVIEW_TABLE_FALLBACK_PROMPT = "work surface"


def log(message):
    print(f"[sam3_segment] {message}", flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def infer(processor, state, prompt, device):
    with inference_context(device):
        state = processor.set_text_prompt(prompt=prompt, state=state)
    return (
        state,
        state["masks"].detach().cpu().numpy(),
        state["boxes"].detach().cpu(),
        state["scores"].detach().cpu(),
    )


def candidate(mask, box, score):
    mask = mask.squeeze().astype(bool)
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    return {"mask": mask, "bbox_xyxy": bbox, "score": float(score)}


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_crop(image, mask, bbox, path):
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)
    rgba.crop(tuple(bbox)).save(path)


def draw_overlay(image, objects, out_path, attempt_root):
    draw_labeled_overlay(image, objects, out_path, attempt_root)
    draw_labeled_overlay(
        image, objects, mask_overlay_path(out_path), attempt_root, annotations=False,
    )


def local_checkpoint():
    checkpoint = os.environ.get("SAM3_CHECKPOINT")
    if checkpoint:
        return checkpoint
    checkpoint = ROOT / "models/sam3.pt"
    return str(checkpoint) if checkpoint.exists() else None


def load_model(device, checkpoint, enable_inst_interactivity=False):
    source = checkpoint if checkpoint else "facebook/sam3"
    key = (
        device,
        str(checkpoint) if checkpoint else None,
        bool(enable_inst_interactivity),
    )
    if key not in _MODEL_CACHE:
        log(
            f"loading model on {device}, checkpoint={source}, "
            f"instance_interactivity={enable_inst_interactivity}"
        )
        try:
            _MODEL_CACHE[key] = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint,
                load_from_HF=checkpoint is None,
                enable_inst_interactivity=enable_inst_interactivity,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to load SAM3 checkpoint. Run `hf auth login` with an account that has access to "
                "https://huggingface.co/facebook/sam3, put the checkpoint at models/sam3.pt, "
                f"or set SAM3_CHECKPOINT=/path/to/sam3.pt. Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
    return _MODEL_CACHE[key]


def load_sam2_model(device):
    from sam2.build_sam import build_sam2

    checkpoint = Path(os.environ.get(
        "SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT,
    )).resolve()
    config = os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint is missing: {checkpoint}")
    key = (device, str(checkpoint), config)
    if key not in _SAM2_MODEL_CACHE:
        log(
            f"loading SAM2.1 on {device}, checkpoint={checkpoint}, "
            f"config={config}"
        )
        _SAM2_MODEL_CACHE[key] = build_sam2(
            config,
            str(checkpoint),
            device=device,
            apply_postprocessing=False,
        )
    return _SAM2_MODEL_CACHE[key]


@contextmanager
def task_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def persistent_worker():
    """Serve JSONL requests while retaining the SAM3 model on this GPU."""
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            return
        response = Path(request["response"])
        try:
            with task_environment(request["environment"]):
                main()
            payload = {"ok": True}
        except BaseException:
            payload = {"error": traceback.format_exc()}
        response.parent.mkdir(parents=True, exist_ok=True)
        temporary = response.with_name(response.name + ".tmp")
        write_json(temporary, payload)
        os.replace(temporary, response)


def inference_context(device):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def detect_tabletop_bbox(
    processor, state, device, confidence, complete_table, image_size,
):
    tabletop_confidence = float(os.environ.get(
        "SAM3_TABLETOP_CONFIDENCE_THRESHOLD", confidence,
    ))
    log(f"segmenting tabletop bbox: prompt={TABLETOP_BBOX_PROMPT!r}")
    processor.set_confidence_threshold(tabletop_confidence)
    try:
        state, masks, boxes, scores = infer(
            processor, state, TABLETOP_BBOX_PROMPT, device,
        )
    finally:
        processor.set_confidence_threshold(confidence)
    selected = select_tabletop_bbox(
        [
            item for item in (
                candidate(mask, box, score)
                for mask, box, score in zip(masks, boxes, scores)
            )
            if item
        ],
        complete_table,
    )
    if selected is None:
        log("no compatible tabletop bbox proposal; keeping legacy alignment target")
        return state, None
    x0, y0, x1, y1 = selected["bbox_xyxy"]
    return state, {
        "prompt": TABLETOP_BBOX_PROMPT,
        "confidence_threshold": tabletop_confidence,
        "score": float(selected["score"]),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bottom_edge_xyxy": [[x0, y1], [x1, y1]],
        "image_size": list(image_size),
    }


def grid_object_candidates(model, state, image_size, table_candidate, threshold, device):
    width, height = image_size
    columns = int(os.environ.get("SAM3_ANONYMOUS_GRID_COLUMNS", "16"))
    rows = int(os.environ.get("SAM3_ANONYMOUS_GRID_ROWS", "9"))
    if columns < 1 or rows < 1:
        raise ValueError("SAM3 anonymous grid dimensions must be positive")
    points = np.asarray([
        [[(column + 0.5) * width / columns, (row + 0.5) * height / rows]]
        for row in range(rows)
        for column in range(columns)
    ], dtype=np.float32)
    labels = np.ones((len(points), 1), dtype=np.int64)
    batch_size = int(os.environ.get(
        "SAM3_ANONYMOUS_GRID_BATCH_SIZE", "32",
    ))
    if batch_size < 1:
        raise ValueError("SAM3 anonymous grid batch size must be positive")
    log(
        f"segmenting anonymous object proposals with official point prompts: "
        f"grid={columns}x{rows}, batch_size={batch_size}, "
        f"threshold={threshold:g}"
    )

    image_area = width * height
    min_area = float(os.environ.get(
        "SAM3_ANONYMOUS_GRID_MIN_AREA_RATIO", "0.0002",
    )) * image_area
    max_area = float(os.environ.get(
        "SAM3_ANONYMOUS_GRID_MAX_AREA_RATIO", "0.5",
    )) * max(int(table_candidate["mask"].sum()), 1)
    table_mask = table_candidate["mask"]
    candidates = []
    for start in range(0, len(points), batch_size):
        with inference_context(device):
            masks, scores, _ = model.predict_inst(
                state,
                point_coords=points[start:start + batch_size],
                point_labels=labels[start:start + batch_size],
                multimask_output=True,
            )
        for point_masks, point_scores in zip(masks, scores):
            index = int(np.argmax(point_scores))
            score = float(point_scores[index])
            mask = np.asarray(point_masks[index]).squeeze().astype(bool)
            area = int(mask.sum())
            if score < threshold or area < min_area or area > max_area:
                continue
            intersection = int((mask & table_mask).sum())
            if intersection / min(area, int(table_mask.sum())) >= 0.90:
                continue
            item = candidate(mask, None, score)
            if item:
                candidates.append(item)
    return candidates, columns, rows


def publish_anonymous_segmentation(
    image, confidence, output_prefix, results_path, overlay_path, mask_root,
    crop_root, attempt_root, final_root, proposals, result_metadata,
    overlay_extras=(),
):
    width, height = image.size
    objects = []
    for index, proposal in enumerate(proposals):
        object_id = proposal.get("object_id", f"segment_{index - 1:03d}")
        mask = proposal["mask"]
        bbox = mask_bbox(mask)
        mask_file = mask_root / f"{output_prefix}{object_id}_mask.png"
        crop_file = crop_root / f"{output_prefix}{object_id}_crop.png"
        mask_path = mask_file.relative_to(attempt_root)
        crop_path = crop_file.relative_to(attempt_root)
        save_mask(mask, mask_file)
        save_crop(image, mask, bbox, crop_file)
        objects.append({
            "object_id": object_id,
            "class_name": proposal.get("class_name"),
            "prompt": proposal["prompt"],
            "score": proposal["score"],
            "mask_area_px": int(np.count_nonzero(mask)),
            "bbox_xyxy": bbox,
            "bbox_norm": [
                bbox[0] / width, bbox[1] / height,
                bbox[2] / width, bbox[3] / height,
            ],
            "mask_path": str(mask_path),
            "crop_path": str(crop_path),
            "operability_type": proposal.get("operability_type"),
        })
        log(
            f'wrote "{final_root / mask_path}" and "{final_root / crop_path}", '
            f'score={proposal["score"]:.3f}, bbox={bbox}'
        )

    result = build_result(
        width,
        height,
        confidence,
        objects,
        os.environ.get("SAM3_IMAGE_ARTIFACT", "reference_image.image"),
    )
    result.update(result_metadata)
    write_json(results_path, result)
    draw_overlay(image, [*objects, *overlay_extras], overlay_path, attempt_root)
    log(
        f'wrote "{final_root / results_path.relative_to(attempt_root)}" and '
        f'"{final_root / overlay_path.relative_to(attempt_root)}"'
    )


def suppress_contained_sam2_masks(items, threshold=0.90):
    kept = []
    suppressed = []
    for item in sorted(items, key=lambda value: int(value["mask"].sum()), reverse=True):
        area = max(int(item["mask"].sum()), 1)
        if any(int((item["mask"] & outer["mask"]).sum()) / area >= threshold for outer in kept):
            suppressed.append(item)
        else:
            kept.append(item)
    return kept, suppressed


def detect_sam3_supports(
    processor, state, device, confidence, view, detect_tabletop=True,
):
    table_confidence = float(os.environ.get(
        "SAM3_TABLE_CONFIDENCE_THRESHOLD", confidence,
    ))
    tabletop_confidence = float(os.environ.get(
        "SAM3_TABLETOP_CONFIDENCE_THRESHOLD", confidence,
    ))
    log(f"segmenting the complete table with SAM3: prompt={ANONYMOUS_TABLE_PROMPT!r}")
    processor.set_confidence_threshold(table_confidence)
    try:
        state, masks, boxes, scores = infer(
            processor, state, ANONYMOUS_TABLE_PROMPT, device,
        )
    finally:
        processor.set_confidence_threshold(confidence)
    table_candidates = [
        item for item in (
            candidate(mask, box, score)
            for mask, box, score in zip(masks, boxes, scores)
        )
        if item
    ]
    table_candidate = select_complete_table(table_candidates)
    table_prompt = ANONYMOUS_TABLE_PROMPT
    if table_candidate is None and view == "topview":
        table_prompt = TOPVIEW_TABLE_FALLBACK_PROMPT
        log(
            "no top-view table proposal; retrying at the same confidence with "
            f"prompt={table_prompt!r}"
        )
        processor.set_confidence_threshold(table_confidence)
        try:
            state, masks, boxes, scores = infer(
                processor, state, table_prompt, device,
            )
        finally:
            processor.set_confidence_threshold(confidence)
        table_candidate = select_complete_table([
            item for item in (
                candidate(mask, box, score)
                for mask, box, score in zip(masks, boxes, scores)
            )
            if item
        ])
    if table_candidate is None:
        raise RuntimeError("SAM3 found no complete table proposal")
    table_candidate.update({
        "object_id": "table_0",
        "class_name": "table",
        "prompt": table_prompt,
        "operability_type": "static_support",
    })
    if not detect_tabletop:
        return state, table_candidate, None

    log(f"segmenting the tabletop with SAM3: prompt={TABLETOP_SURFACE_PROMPT!r}")
    processor.set_confidence_threshold(tabletop_confidence)
    try:
        state, masks, boxes, scores = infer(
            processor, state, TABLETOP_SURFACE_PROMPT, device,
        )
    finally:
        processor.set_confidence_threshold(confidence)
    tabletop_candidates = [
        item for item in (
            candidate(mask, box, score)
            for mask, box, score in zip(masks, boxes, scores)
        )
        if item
    ]
    tabletop_candidate = (
        select_tabletop_bbox(tabletop_candidates, table_candidate)
        if view == "frontview"
        else select_complete_table(tabletop_candidates)
    )
    if tabletop_candidate is None:
        raise RuntimeError("SAM3 found no tabletop surface proposal")
    return state, table_candidate, tabletop_candidate


def write_peeling_anonymous_segmentation(
    processor, state, image, device, confidence, output_prefix, results_path,
    overlay_path, mask_root, crop_root, attempt_root, final_root, view,
):
    object_confidence = float(os.environ.get(
        "SAM3_ANONYMOUS_OBJECT_CONFIDENCE", "0.07",
    ))
    rounds = int(os.environ.get("SAM3_PEEL_ROUNDS", "2"))
    if rounds < 1:
        raise ValueError("SAM3 peeling rounds must be positive")

    state, table_candidate, tabletop_candidate = detect_sam3_supports(
        processor, state, device, confidence, view,
    )
    tabletop_mask_file = mask_root / f"{output_prefix}tabletop_support_mask.png"
    save_mask(tabletop_candidate["mask"], tabletop_mask_file)

    peeled = table_candidate["mask"] | tabletop_candidate["mask"]
    original = np.asarray(image)
    debug_root = attempt_root / "debug/peeling"
    debug_root.mkdir(parents=True, exist_ok=True)
    accepted = []
    statistics = []
    for round_index in range(1, rounds + 1):
        round_root = debug_root / f"round_{round_index:02d}"
        round_root.mkdir(parents=True, exist_ok=True)
        peeled_mask_file = round_root / "peeled_mask.png"
        save_mask(peeled, peeled_mask_file)
        round_candidates = []
        background_statistics = {}
        for background, fill in (("black", 0), ("white", 255)):
            background_root = round_root / background
            background_root.mkdir(parents=True, exist_ok=True)
            residual_image = original.copy()
            residual_image[peeled] = fill
            residual_image = Image.fromarray(residual_image, mode="RGB")
            residual_image_file = background_root / "input.png"
            residual_image.save(residual_image_file)
            with inference_context(device):
                residual_state = processor.set_image(residual_image)
            processor.set_confidence_threshold(object_confidence)
            try:
                residual_state, masks, boxes, scores = infer(
                    processor, residual_state, ANONYMOUS_OBJECT_PROMPT, device,
                )
            finally:
                processor.set_confidence_threshold(confidence)
            raw = [
                item for item in (
                    candidate(mask, box, score)
                    for mask, box, score in zip(masks, boxes, scores)
                )
                if item
            ]
            candidate_statistics = {}
            for index, item in enumerate(raw):
                raw_mask_file = background_root / f"candidate_{index:03d}_raw.png"
                save_mask(item["mask"], raw_mask_file)
                candidate_statistics[id(item)] = {
                    "index": index,
                    "score": item["score"],
                    "raw_bbox_xyxy": item["bbox_xyxy"],
                    "raw_mask_area_px": int(item["mask"].sum()),
                    "raw_mask_path": str(raw_mask_file.relative_to(attempt_root)),
                }
            residual = []
            for index, item in enumerate(raw):
                item["mask"] &= ~peeled
                item["bbox_xyxy"] = mask_bbox(item["mask"])
                if item["bbox_xyxy"] is not None:
                    residual_mask_file = (
                        background_root / f"candidate_{index:03d}_residual.png"
                    )
                    save_mask(item["mask"], residual_mask_file)
                    candidate_statistics[id(item)].update({
                        "residual_bbox_xyxy": item["bbox_xyxy"],
                        "residual_mask_area_px": int(item["mask"].sum()),
                        "residual_mask_path": str(
                            residual_mask_file.relative_to(attempt_root)
                        ),
                    })
                    item["prompt"] = ANONYMOUS_OBJECT_PROMPT
                    residual.append(item)
                else:
                    candidate_statistics[id(item)]["filter_reason"] = "fully_peeled"
            (
                residual, oversized, table_width_like, below_tabletop,
                sparse, disconnected,
            ) = filter_anonymous_candidates(
                residual,
                table_candidate["bbox_xyxy"],
                tabletop_candidate["bbox_xyxy"],
                image.height,
                view,
            )
            for reason, items in (
                ("larger_than_table", oversized),
                ("near_table_width", table_width_like),
                ("below_tabletop_bottom", below_tabletop),
                ("sparse_mask", sparse),
                ("disconnected_mask", disconnected),
            ):
                for item in items:
                    candidate_statistics[id(item)]["filter_reason"] = reason
            for item in residual:
                candidate_statistics[id(item)]["filter_reason"] = "kept"
            round_candidates.extend(residual)
            background_statistics[background] = {
                "input_path": str(
                    residual_image_file.relative_to(attempt_root)
                ),
                "raw_candidate_count": len(raw),
                "residual_candidate_count": len(residual),
                "filtered_larger_than_table_count": len(oversized),
                "filtered_near_table_width_count": len(table_width_like),
                "filtered_below_tabletop_bottom_count": len(below_tabletop),
                "filtered_sparse_mask_count": len(sparse),
                "filtered_disconnected_mask_count": len(disconnected),
                "candidates": sorted(
                    candidate_statistics.values(),
                    key=lambda item: item["index"],
                ),
            }

        round_candidates, contained = merge_contained_masks(round_candidates)
        round_candidates, duplicates = merge_near_duplicate_masks(
            round_candidates,
        )
        accepted.extend(round_candidates)
        for item in round_candidates:
            peeled |= item["mask"]
        statistics.append({
            "round": round_index,
            "peeled_mask_path": str(
                peeled_mask_file.relative_to(attempt_root)
            ),
            "backgrounds": background_statistics,
            "new_candidate_count": len(round_candidates),
            "merged_contained_candidate_count": len(contained),
            "merged_duplicate_candidate_count": len(duplicates),
        })
        log(
            f"peeling round {round_index}: new={len(round_candidates)}, "
            f"merged_contained={len(contained)}, "
            f"merged_duplicate={len(duplicates)}"
        )
        if not round_candidates:
            break

    accepted, contained = merge_contained_masks(accepted)
    accepted, duplicates = merge_near_duplicate_masks(accepted)
    proposals = [table_candidate, *spatially_sorted_candidates(accepted)]
    publish_anonymous_segmentation(
        image, confidence, output_prefix, results_path, overlay_path,
        mask_root, crop_root, attempt_root, final_root, proposals,
        {
            "segmentation_mode": "anonymous_black_white_mask_peeling",
            "anonymous_method": "peel",
            "generic_prompts": [
                ANONYMOUS_OBJECT_PROMPT,
                ANONYMOUS_TABLE_PROMPT,
                TABLETOP_SURFACE_PROMPT,
            ],
            "object_confidence_threshold": object_confidence,
            "table_confidence_threshold": float(os.environ.get(
                "SAM3_TABLE_CONFIDENCE_THRESHOLD", confidence,
            )),
            "tabletop_confidence_threshold": float(os.environ.get(
                "SAM3_TABLETOP_CONFIDENCE_THRESHOLD", confidence,
            )),
            "candidate_count": len(proposals),
            "peeling_backgrounds": ["black", "white"],
            "peeling_round_limit": rounds,
            "peeling_rounds_completed": len(statistics),
            "peeling_statistics": statistics,
            "merged_contained_candidate_count": len(contained),
            "merged_duplicate_candidate_count": len(duplicates),
            "tabletop_prompt": TABLETOP_SURFACE_PROMPT,
            "tabletop_score": float(tabletop_candidate["score"]),
            "tabletop_bbox_xyxy": tabletop_candidate["bbox_xyxy"],
            "tabletop_mask_path": str(
                tabletop_mask_file.relative_to(attempt_root)
            ),
        },
        [{
            "object_id": "tabletop_0",
            "mask_path": str(tabletop_mask_file.relative_to(attempt_root)),
            "bbox_xyxy": tabletop_candidate["bbox_xyxy"],
        }],
    )


def supported_by_tabletop(item, tabletop_candidate, image_size, view):
    width, height = image_size
    x0, y0, x1, y1 = item["bbox_xyxy"]
    tx0, ty0, tx1, ty1 = tabletop_candidate["bbox_xyxy"]
    center_x = (x0 + x1) / 2
    if not tx0 <= center_x <= tx1:
        return False
    if view == "topview":
        return ty0 <= (y0 + y1) / 2 <= ty1
    tolerance = 0.08 * height
    return ty0 - tolerance <= y1 <= ty1 + tolerance


def write_sam2_anonymous_segmentation(
    model, processor, state, image, device, confidence, output_prefix,
    results_path, overlay_path, mask_root, crop_root, attempt_root, final_root,
    view,
):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    pred_iou = float(os.environ.get("SAM2_PRED_IOU_THRESHOLD", "0.8"))
    stability = float(os.environ.get("SAM2_STABILITY_SCORE_THRESHOLD", "0.95"))
    points_per_side = int(os.environ.get("SAM2_POINTS_PER_SIDE", "32"))
    points_per_batch = int(os.environ.get("SAM2_POINTS_PER_BATCH", "64"))
    crop_n_layers = int(os.environ.get("SAM2_CROP_N_LAYERS", "0"))
    image_array = np.array(image, copy=True)
    log(
        f"generating official SAM2 automatic masks: points_per_side="
        f"{points_per_side}, pred_iou={pred_iou:g}, stability={stability:g}, "
        f"crop_n_layers={crop_n_layers}"
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou,
        stability_score_thresh=stability,
        box_nms_thresh=0.7,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    with inference_context(device):
        annotations = generator.generate(image_array)

    width, height = image.size
    state, table_candidate, tabletop_candidate = detect_sam3_supports(
        processor, state, device, confidence, view,
    )
    tabletop_mask_file = mask_root / f"{output_prefix}tabletop_support_mask.png"
    save_mask(tabletop_candidate["mask"], tabletop_mask_file)

    table_mask = table_candidate["mask"]
    table_area = max(int(table_mask.sum()), 1)
    tx0, _ty0, tx1, _ty1 = table_candidate["bbox_xyxy"]
    table_width = max(tx1 - tx0, 1)
    raw_targets = []
    for annotation in annotations:
        mask = np.asarray(annotation["segmentation"]).astype(bool)
        item = candidate(mask, None, annotation["predicted_iou"])
        if item is None:
            continue
        x0, y0, x1, y1 = item["bbox_xyxy"]
        touches_image_edge = (
            x0 <= 2 or y0 <= 2 or x1 >= width - 2 or y1 >= height - 2
        )
        area = int(mask.sum())
        center_x = (x0 + x1) / 2
        table_overlap = int((mask & table_mask).sum()) / max(area, 1)
        if (
            touches_image_edge
            or not tx0 <= center_x <= tx1
            or table_overlap >= 0.90
            or not supported_by_tabletop(
                item, tabletop_candidate, image.size, view,
            )
            or area > 0.40 * table_area
            or x1 - x0 > 0.40 * table_width
            or (view == "frontview" and y0 > 0.85 * height)
        ):
            continue
        item["prompt"] = "automatic_mask"
        item["stability_score"] = float(annotation["stability_score"])
        raw_targets.append(item)

    deduplicated, duplicate_masks = suppress_near_duplicate_masks(raw_targets)
    targets, contained_masks = suppress_contained_sam2_masks(deduplicated)
    proposals = [table_candidate, *spatially_sorted_candidates(targets)]
    publish_anonymous_segmentation(
        image,
        pred_iou,
        output_prefix,
        results_path,
        overlay_path,
        mask_root,
        crop_root,
        attempt_root,
        final_root,
        proposals,
        {
            "model": "sam2.1_hiera_large",
            "segmentation_mode": "sam3_supports_with_sam2_automatic_objects",
            "anonymous_method": "sam2",
            "generic_prompts": [],
            "object_confidence_threshold": pred_iou,
            "sam2_pred_iou_threshold": pred_iou,
            "sam2_stability_score_threshold": stability,
            "sam2_points_per_side": points_per_side,
            "sam2_crop_n_layers": crop_n_layers,
            "candidate_count": len(proposals),
            "raw_candidate_count": len(annotations),
            "filtered_candidate_count": len(raw_targets),
            "suppressed_duplicate_candidate_count": len(duplicate_masks),
            "suppressed_contained_candidate_count": len(contained_masks),
            "table_model": "sam3",
            "tabletop_model": "sam3",
            "table_confidence_threshold": float(os.environ.get(
                "SAM3_TABLE_CONFIDENCE_THRESHOLD", confidence,
            )),
            "tabletop_confidence_threshold": float(os.environ.get(
                "SAM3_TABLETOP_CONFIDENCE_THRESHOLD", confidence,
            )),
            "tabletop_prompt": TABLETOP_SURFACE_PROMPT,
            "tabletop_score": float(tabletop_candidate["score"]),
            "tabletop_bbox_xyxy": tabletop_candidate["bbox_xyxy"],
            "tabletop_mask_path": str(
                tabletop_mask_file.relative_to(attempt_root)
            ),
        },
        [{
            "object_id": "tabletop_0",
            "mask_path": str(tabletop_mask_file.relative_to(attempt_root)),
            "bbox_xyxy": tabletop_candidate["bbox_xyxy"],
        }],
    )


def write_anonymous_segmentation(
    processor, state, image, device, confidence, output_prefix, results_path,
    overlay_path, mask_root, crop_root, attempt_root, final_root, view, method,
):
    table_confidence = float(os.environ.get(
        "SAM3_TABLE_CONFIDENCE_THRESHOLD", confidence,
    ))
    tabletop_confidence = float(os.environ.get(
        "SAM3_TABLETOP_CONFIDENCE_THRESHOLD", confidence,
    ))
    state, table_candidate, tabletop_candidate = detect_sam3_supports(
        processor, state, device, confidence, view,
    )
    tabletop_mask_file = None
    if tabletop_candidate is not None:
        tabletop_mask_file = (
            mask_root / f"{output_prefix}tabletop_support_mask.png"
        )
        save_mask(tabletop_candidate["mask"], tabletop_mask_file)

    object_confidence = float(os.environ.get(
        "SAM3_ANONYMOUS_OBJECT_CONFIDENCE", "0.07",
    ))
    grid_columns = grid_rows = None
    semantic_tags = []
    if method == "grid":
        raw_targets, grid_columns, grid_rows = grid_object_candidates(
            processor.model, state, image.size, table_candidate,
            object_confidence, device,
        )
        object_prompt = "point_grid"
    elif method == "semantic_tags":
        semantic_tags = json.loads(os.environ.get(
            "SAM3_SEMANTIC_TAGS", "[]",
        ))
        if not semantic_tags:
            raise RuntimeError("RAM++ returned no semantic tags")
        raw_targets = []
        for prompt in semantic_tags:
            log(f"segmenting semantic object proposals: prompt={prompt!r}")
            processor.set_confidence_threshold(object_confidence)
            try:
                state, masks, boxes, scores = infer(
                    processor, state, prompt, device,
                )
            finally:
                processor.set_confidence_threshold(confidence)
            for item in (
                candidate(mask, box, score)
                for mask, box, score in zip(masks, boxes, scores)
            ):
                if item:
                    item["prompt"] = prompt
                    raw_targets.append(item)
        object_prompt = None
    else:
        log(
            f"segmenting anonymous object proposals: "
            f"prompt={ANONYMOUS_OBJECT_PROMPT!r}"
        )
        processor.set_confidence_threshold(object_confidence)
        try:
            state, masks, boxes, scores = infer(
                processor, state, ANONYMOUS_OBJECT_PROMPT, device,
            )
        finally:
            processor.set_confidence_threshold(confidence)
        raw_targets = [
            item for item in (
                candidate(mask, box, score)
                for mask, box, score in zip(masks, boxes, scores)
            )
            if item
        ]
        object_prompt = ANONYMOUS_OBJECT_PROMPT
    raw_target_count = len(raw_targets)
    (
        raw_targets, oversized_targets, table_width_targets,
        below_tabletop_targets, sparse_targets, disconnected_targets,
    ) = (
        filter_anonymous_candidates(
            raw_targets,
            table_candidate["bbox_xyxy"],
            (
                tabletop_candidate["bbox_xyxy"]
                if tabletop_candidate is not None else None
            ),
            image.height,
            view,
        )
    )
    raw_targets, merged_nested_targets = merge_contained_masks(raw_targets)
    raw_targets, merged_duplicate_targets = merge_near_duplicate_masks(
        raw_targets,
    )
    if merged_duplicate_targets:
        log(
            f"merged {len(merged_duplicate_targets)} near-duplicate anonymous "
            "mask proposal(s)"
        )
    if (
        oversized_targets or table_width_targets
        or below_tabletop_targets or sparse_targets
        or disconnected_targets or merged_nested_targets
    ):
        log(
            "filtered anonymous proposals: "
            f"larger_than_table={len(oversized_targets)}, "
            f"near_table_width={len(table_width_targets)}, "
            f"below_tabletop_bottom={len(below_tabletop_targets)}, "
            f"sparse_mask={len(sparse_targets)}, "
            f"disconnected_mask={len(disconnected_targets)}, "
            f"merged_nested={len(merged_nested_targets)}"
        )
    if object_prompt is not None:
        for item in raw_targets:
            item["prompt"] = object_prompt
    proposals = [table_candidate, *spatially_sorted_candidates(raw_targets)]
    if not proposals:
        raise RuntimeError("SAM3 found no anonymous physical-asset proposals")

    result_metadata = {
        "segmentation_mode": (
            "anonymous_grid_points_with_dedicated_table"
            if method == "grid"
            else (
                "ram_plus_semantic_prompts_with_dedicated_table"
                if method == "semantic_tags"
                else "anonymous_objects_with_dedicated_table"
            )
        ),
        "anonymous_method": method,
        "generic_prompts": (
            [ANONYMOUS_TABLE_PROMPT]
            if method == "grid"
            else (
                [*semantic_tags, ANONYMOUS_TABLE_PROMPT]
                if method == "semantic_tags"
                else [ANONYMOUS_OBJECT_PROMPT, ANONYMOUS_TABLE_PROMPT]
            )
        ),
        "object_confidence_threshold": object_confidence,
        "table_confidence_threshold": table_confidence,
        "tabletop_confidence_threshold": tabletop_confidence,
        "candidate_count": len(proposals),
        "raw_candidate_count": 1 + raw_target_count,
        "filtered_larger_than_table_count": len(oversized_targets),
        "filtered_near_table_width_count": len(table_width_targets),
        "filtered_below_tabletop_bottom_count": len(below_tabletop_targets),
        "filtered_sparse_mask_count": len(sparse_targets),
        "filtered_disconnected_mask_count": len(disconnected_targets),
        "merged_nested_candidate_count": len(merged_nested_targets),
        # Retained so older result readers do not break. These candidates are
        # merged into their enclosing proposal now, rather than discarded.
        "suppressed_contained_candidate_count": len(merged_nested_targets),
        "merged_duplicate_candidate_count": len(merged_duplicate_targets),
        # Compatibility field: these candidates are merged now, not discarded.
        "suppressed_duplicate_candidate_count": len(merged_duplicate_targets),
    }
    overlay_extras = []
    if tabletop_candidate is not None:
        result_metadata["generic_prompts"].append(TABLETOP_SURFACE_PROMPT)
        result_metadata.update({
            "raw_candidate_count": 2 + raw_target_count,
            "tabletop_prompt": TABLETOP_SURFACE_PROMPT,
            "tabletop_score": float(tabletop_candidate["score"]),
            "tabletop_bbox_xyxy": tabletop_candidate["bbox_xyxy"],
            "tabletop_mask_path": str(
                tabletop_mask_file.relative_to(attempt_root)
            ),
        })
        overlay_extras.append({
            "object_id": "tabletop_0",
            "mask_path": str(tabletop_mask_file.relative_to(attempt_root)),
            "bbox_xyxy": tabletop_candidate["bbox_xyxy"],
        })
    if method == "grid":
        result_metadata["point_grid"] = {
            "columns": grid_columns,
            "rows": grid_rows,
        }
    elif method == "semantic_tags":
        result_metadata["semantic_tag_metadata"] = json.loads(os.environ.get(
            "SAM3_SEMANTIC_TAG_METADATA", "{}",
        ))
    publish_anonymous_segmentation(
        image, confidence, output_prefix, results_path, overlay_path,
        mask_root, crop_root, attempt_root, final_root, proposals,
        result_metadata, overlay_extras,
    )


def main():
    image_path = ROOT / os.environ.get("SAM3_IMAGE", "data/reference_image.png")
    labels_path = ROOT / os.environ.get("SAM3_LABELS", "data/labels.json")
    output_prefix = os.environ.get("SAM3_OUTPUT_PREFIX", "")
    view = os.environ.get("SAM3_VIEW", "topview" if output_prefix else "frontview")
    results_path = ROOT / os.environ.get("SAM3_RESULTS", f"data/{output_prefix}segmentation_results.json")
    overlay_path = ROOT / os.environ.get("SAM3_OVERLAY", f"data/{output_prefix}segmentation_overlay.png")
    mask_root = ROOT / os.environ.get("SAM3_MASK_ROOT", "masks")
    crop_root = ROOT / os.environ.get("SAM3_CROP_ROOT", "crops")
    attempt_root = Path(os.environ.get("SAM3_ATTEMPT_ROOT", ROOT)).resolve()
    final_root = Path(os.environ.get("SAM3_FINAL_ROOT", attempt_root)).resolve()
    confidence = float(os.environ.get("SAM3_CONFIDENCE_THRESHOLD", "0.35"))
    checkpoint = local_checkpoint()
    device = os.environ.get("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    anonymous = os.environ.get("SAM3_ANONYMOUS") == "1"
    anonymous_method = os.environ.get("SAM3_ANONYMOUS_METHOD", "text")
    if anonymous_method not in {
        "text", "grid", "sam2", "peel", "semantic_tags",
    }:
        raise ValueError(
            f"unsupported SAM3 anonymous method: {anonymous_method!r}"
        )

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not anonymous and not labels_path.exists():
        raise FileNotFoundError(labels_path)

    labels = []
    if not anonymous:
        label_data = read_json(labels_path)
        labels = [item for item in label_data["objects"] if item.get("segment", True)]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)
    if anonymous and anonymous_method == "sam2":
        model = load_model(device, checkpoint)
        processor = Sam3Processor(
            model, device=device, confidence_threshold=confidence,
        )
        with inference_context(device):
            state = processor.set_image(image)
        write_sam2_anonymous_segmentation(
            load_sam2_model(device),
            processor,
            state,
            image,
            device,
            confidence,
            output_prefix,
            results_path,
            overlay_path,
            mask_root,
            crop_root,
            attempt_root,
            final_root,
            view,
        )
        return

    model = load_model(
        device,
        checkpoint,
        enable_inst_interactivity=anonymous and anonymous_method == "grid",
    )
    log(f"segmenting on {device}, confidence={confidence}")
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence)
    with inference_context(device):
        state = processor.set_image(image)

    objects = []
    if anonymous:
        if anonymous_method == "peel":
            write_peeling_anonymous_segmentation(
                processor, state, image, device, confidence, output_prefix,
                results_path, overlay_path, mask_root, crop_root, attempt_root,
                final_root, view,
            )
            return
        write_anonymous_segmentation(
            processor, state, image, device, confidence, output_prefix,
            results_path, overlay_path, mask_root, crop_root, attempt_root,
            final_root, view, anonymous_method,
        )
        return

    supports = [item for item in labels if item.get("operability_type") == "static_support"]
    targets = [item for item in labels if item.get("operability_type") != "static_support"]
    if len(supports) != 1:
        raise RuntimeError(f"expected one static support, found {len(supports)}")

    support = supports[0]
    support_candidate = None
    support_prompt = None
    for prompt in segmentation_prompt_variants(support):
        log(f"segmenting static support role: prompt={prompt!r}")
        state, masks, boxes, scores = infer(processor, state, prompt, device)
        available = [candidate(mask, box, score) for mask, box, score in zip(masks, boxes, scores)]
        available = [item for item in available if item]
        if available:
            support_candidate = max(available, key=lambda item: box_area(item["bbox_xyxy"]))
            support_prompt = prompt
            break
    if support_candidate is None:
        raise RuntimeError("SAM3 found no instance for the static support role")
    tabletop_bbox = None
    if view == "frontview":
        state, tabletop_bbox = detect_tabletop_bbox(
            processor, state, device, confidence, support_candidate, image.size,
        )

    prompt_cache = {}
    raw_candidates = []
    for label in targets:
        for variant_index, prompt in enumerate(segmentation_prompt_variants(label)):
            if prompt not in prompt_cache:
                log(f"segmenting category role: prompt={prompt!r}")
                state, masks, boxes, scores = infer(processor, state, prompt, device)
                prompt_cache[prompt] = [
                    item
                    for item in (
                        candidate(mask, box, score)
                        for mask, box, score in zip(masks, boxes, scores)
                    )
                    if item
                ]
            for detected in prompt_cache[prompt]:
                raw_candidates.append({
                    **detected,
                    "label_id": label["object_id"],
                    "category": label["category"],
                    "prompt": prompt,
                    "variant_index": variant_index,
                })

    candidates = merge_candidate_evidence(raw_candidates)
    target_assignments = match_global_candidates(
        targets,
        candidates,
        support_candidate["bbox_xyxy"],
        view,
    )
    initially_matched = {label["object_id"] for label, *_ in target_assignments}
    retried = False
    for label in targets:
        if label["object_id"] in initially_matched:
            continue
        primary_prompts = segmentation_prompt_variants(label)
        prompt = segmentation_component_prompt(label)
        if not prompt or prompt in primary_prompts:
            continue
        retried = True
        if prompt not in prompt_cache:
            log(f"segmenting unmatched identity by visible components: prompt={prompt!r}")
            state, masks, boxes, scores = infer(processor, state, prompt, device)
            prompt_cache[prompt] = [
                item
                for item in (
                    candidate(mask, box, score)
                    for mask, box, score in zip(masks, boxes, scores)
                )
                if item
            ]
        for detected in prompt_cache[prompt]:
            raw_candidates.append({
                **detected,
                "label_id": label["object_id"],
                "category": label["category"],
                "prompt": prompt,
                "variant_index": len(primary_prompts),
            })
    if retried:
        candidates = merge_candidate_evidence(raw_candidates)
        target_assignments = match_global_candidates(
            targets,
            candidates,
            support_candidate["bbox_xyxy"],
            view,
        )
    geometric_attempted = []
    if view == "topview" and os.environ.get("SAM3_TOPVIEW_GEOMETRIC_FALLBACK", "1") != "0":
        matched_ids = {label["object_id"] for label, *_ in target_assignments}
        fallback_confidence = float(os.environ.get("SAM3_GEOMETRIC_FALLBACK_CONFIDENCE", "0.20"))
        for label in targets:
            if label["object_id"] in matched_ids:
                continue
            geometric_attempted.append(label["object_id"])
            prior = topview_geometric_prior(
                label, support, support_candidate["bbox_xyxy"], image.size,
            )
            prompt = segmentation_prompt_variants(label)[0]
            log(
                f"segmenting unmatched identity with geometric prompt: "
                f"object_id={label['object_id']}, box={prior['box']}"
            )
            processor.reset_all_prompts(state)
            processor.set_confidence_threshold(fallback_confidence)
            with inference_context(device):
                state = processor.set_text_prompt(prompt=prompt, state=state)
                state = processor.add_geometric_prompt(box=prior["box"], label=True, state=state)
            available = [
                item for item in (
                    candidate(mask, box, score)
                    for mask, box, score in zip(
                        state["masks"].detach().cpu().numpy(),
                        state["boxes"].detach().cpu(),
                        state["scores"].detach().cpu(),
                    )
                ) if item
            ]
            processor.set_confidence_threshold(confidence)
            detected = select_geometric_candidate(available, prior)
            if detected:
                raw_candidates.append({
                    **detected,
                    "label_id": label["object_id"],
                    "category": label["category"],
                    "prompt": prompt,
                    "variant_index": len(segmentation_prompt_variants(label)),
                })
        if geometric_attempted:
            candidates = merge_candidate_evidence(raw_candidates)
            target_assignments = match_global_candidates(
                targets, candidates, support_candidate["bbox_xyxy"], view,
            )
    assignments = [(
        support,
        support_candidate,
        0.0,
        support_prompt,
        {"semantic": 0.0, "shape": 0.0, "layout": 0.0, "total": 0.0},
    )]
    assignments.extend(
        (label, item, cost, evidence["prompt"], cost_components)
        for label, item, cost, evidence, cost_components in target_assignments
    )

    matched_ids = {label["object_id"] for label, _, _, _, _ in assignments}
    unmatched = [label["object_id"] for label in labels if label["object_id"] not in matched_ids]
    if unmatched:
        log(f"unmatched blueprint objects: {unmatched}")
    required_unmatched = [
        label["object_id"] for label in labels
        if label.get("reconstruct", True) and label["object_id"] in unmatched
    ]
    if view == "frontview" and required_unmatched:
        raise RuntimeError(
            "SAM3 could not match required front-view objects: "
            + ", ".join(required_unmatched)
        )

    for label, matched, match_cost, prompt, match_cost_components in assignments:
        object_id = label["object_id"]
        mask = matched["mask"]
        bbox = matched["bbox_xyxy"]
        score = matched["score"]
        mask_file = mask_root / f"{output_prefix}{object_id}_mask.png"
        crop_file = crop_root / f"{output_prefix}{object_id}_crop.png"
        mask_path = mask_file.relative_to(attempt_root)
        crop_path = crop_file.relative_to(attempt_root)
        save_mask(mask, mask_file)
        save_crop(image, mask, bbox, crop_file)
        objects.append(
            {
                "object_id": object_id,
                "class_name": label["category"],
                "prompt": prompt,
                "score": score,
                "mask_area_px": int(np.count_nonzero(mask)),
                "match_cost": match_cost,
                "match_cost_components": match_cost_components,
                "bbox_xyxy": bbox,
                "bbox_norm": [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height],
                "mask_path": str(mask_path),
                "crop_path": str(crop_path),
                "operability_type": label.get("operability_type"),
            }
        )
        log(
            f'wrote "{final_root / mask_path}" and "{final_root / crop_path}", '
            f"score={score:.3f}, bbox={bbox}"
        )

    result = build_result(
        width,
        height,
        confidence,
        objects,
        os.environ.get("SAM3_IMAGE_ARTIFACT", "reference_image.image"),
    )
    result["segmentation_mode"] = "object_labels_then_global_semantic_assignment"
    result["unmatched_object_ids"] = unmatched
    result["geometric_fallback_attempted_object_ids"] = geometric_attempted
    result["geometric_fallback_matched_object_ids"] = [
        object_id for object_id in geometric_attempted if object_id in matched_ids
    ]
    result["candidate_count"] = len(candidates)
    result["raw_candidate_count"] = len(raw_candidates)
    if tabletop_bbox is not None:
        result["front_tabletop_bbox"] = tabletop_bbox
    write_json(results_path, result)
    draw_overlay(image, objects, overlay_path, attempt_root)
    log(
        f'wrote "{final_root / results_path.relative_to(attempt_root)}" and '
        f'"{final_root / overlay_path.relative_to(attempt_root)}"'
    )


if __name__ == "__main__":
    try:
        persistent_worker() if "--persistent-worker" in sys.argv else main()
    except Exception as exc:
        print(f"[sam3_segment] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
