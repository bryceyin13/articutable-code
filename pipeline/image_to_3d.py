#!/usr/bin/env python3
import hashlib
import os
import random
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


from pipeline.stage_io import dependency_file
from pipeline.common import attempt_path, env_positive_int, log_step, read_json, write_json




def _input_image(item):
    views = item["views"]
    return views.get("primary") or views.get("right_45") or views["front"]


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _stream_records(stream_root):
    manifest = stream_root / "manifest.json"
    if manifest.is_file():
        return {item["job_id"]: item for item in read_json(manifest)["items"]}
    records = {}
    for job_manifest in stream_root.glob("jobs/*/condition_manifest.json"):
        items = read_json(job_manifest).get("items", [])
        if len(items) == 1:
            records[job_manifest.parent.name] = {
                "job_id": job_manifest.parent.name,
            }
    return records


def _stream_roots(manifest_path):
    current = manifest_path.parent.parent
    yield current / ".image_to_3d_stream"
    current_record = current / "attempt.json"
    if not current_record.is_file():
        return
    upstream = read_json(current_record).get("upstream_best")
    for candidate in sorted(current.parent.glob("[!.]*"), reverse=True):
        record = candidate / "attempt.json"
        if candidate == current or not record.is_file():
            continue
        metadata = read_json(record)
        if metadata.get("status") == "failed" and metadata.get("upstream_best") == upstream:
            yield candidate / ".image_to_3d_stream"


def _matching_stream_source(current_root, stream_root, job_id, source_item):
    record = _stream_records(stream_root).get(job_id)
    source = stream_root / "outputs" / job_id
    if record is None:
        return None
    if any(
        not (source / name).is_file() or (source / name).stat().st_size == 0
        for name in ("model.glb", "preview.png", "metadata.json")
    ):
        return None
    if stream_root.parent != current_root:
        current_fingerprint = (
            current_root / _input_image(source_item)
        ).parent / "request_fingerprint.txt"
        source_fingerprint = (
            _condition_output_for_segment(stream_root.parent, job_id)
            / "request_fingerprint.txt"
        )
        if (
            not current_fingerprint.is_file()
            or not source_fingerprint.is_file()
            or current_fingerprint.read_text(encoding="utf-8").strip()
            != source_fingerprint.read_text(encoding="utf-8").strip()
        ):
            return None
    return source


def _failed_stream_source(context, attempt, current_root, job_id, source_item):
    if context is None or attempt is None:
        return None
    attempts = context._stage_root(attempt.stage) / "attempts"
    for candidate in sorted(attempts.glob("*"), reverse=True):
        record = candidate / "attempt.json"
        if not record.is_file():
            continue
        metadata = read_json(record)
        if (
            metadata.get("status") == "failed"
            and metadata.get("upstream_best") == attempt.upstream_best
            and (source := _matching_stream_source(
                current_root, candidate / ".image_to_3d_stream", job_id,
                source_item,
            )) is not None
        ):
            return source
    return None


def _condition_output_for_segment(attempt_root, segment_id):
    direct = Path(attempt_root) / "condition_images" / segment_id
    if direct.is_dir():
        return direct
    recognition_path = Path(attempt_root) / "data/recognition.json"
    if recognition_path.is_file():
        recognition = read_json(recognition_path)
        identities = [recognition.get("table"), *recognition.get("objects", [])]
        for identity in identities:
            if (
                isinstance(identity, dict)
                and identity.get("front_segment_id") == segment_id
            ):
                return (
                    Path(attempt_root) / "condition_images"
                    / identity["object_id"]
                )
    return direct


def _streamed_results(manifest_path, manifest, output_root, ids):
    current_root = manifest_path.parent.parent
    source_items = {item["object_id"]: item for item in manifest["items"]}
    results = []
    for object_id in ids:
        source_item = source_items[object_id]
        job_id = source_item.get("front_segment_id", object_id)
        source = next((
            found for stream_root in _stream_roots(manifest_path)
            if (found := _matching_stream_source(
                current_root, stream_root, job_id, source_item,
            )) is not None
        ), None)
        if source is None:
            continue
        destination = Path(output_root) / object_id
        shutil.copytree(
            source, destination, dirs_exist_ok=True, copy_function=_link_or_copy,
        )
        log_step("image_to_3d", f"reused checkpoint {job_id} -> {object_id}")
        results.append({
            "object_id": object_id,
            "input_image": _input_image(source_item),
            "mesh": f"{object_id}/model.glb",
            "preview": f"{object_id}/preview.png",
            "metadata": f"{object_id}/metadata.json",
        })
    return results


def _reuse_completed_best(
    context, attempt, manifest_path, manifest, object_ids,
    allow_missing=False, physical_copy=False,
):
    if not object_ids:
        return []
    try:
        previous = context.stage_best("image_to_3d")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "selective image-to-3d generation requires a completed image_to_3d best"
        ) from exc
    record = read_json(previous / "attempt.json")
    if record.get("status") != "completed":
        raise RuntimeError("current image_to_3d best is not completed")
    previous_manifest = read_json(previous / "data/image_to_3d_manifest.json")
    previous_items = {
        item["object_id"]: item for item in previous_manifest.get("items", [])
    }
    previous_upstream = record.get("upstream_best", {})
    active_upstream = attempt.upstream_best
    source_stage = (
        "condition_images"
        if "condition_images" in previous_upstream
        and "condition_images" in active_upstream
        else None
    )
    changed_non_source = {
        stage for stage in set(previous_upstream) | set(active_upstream)
        if stage != source_stage
        and previous_upstream.get(stage) != active_upstream.get(stage)
    }
    if source_stage is None or changed_non_source:
        raise RuntimeError(
            "current image_to_3d best has incompatible upstream dependencies"
        )
    previous_source_root = (
        context._stage_root(source_stage) / "attempts"
        / previous_upstream[source_stage]
    )
    active_source_root = Path(manifest_path).parent.parent
    source_items = {item["object_id"]: item for item in manifest["items"]}

    invalid = []
    for object_id in object_ids:
        item = previous_items.get(object_id)
        source = previous / object_id
        reusable = not (
            item is None
            or any(
                not (source / name).is_file() or (source / name).stat().st_size == 0
                for name in ("model.glb", "preview.png", "metadata.json")
            )
        )
        if reusable and previous_upstream != active_upstream:
            old_input = previous_source_root / item["input_image"]
            active_input = active_source_root / _input_image(source_items[object_id])
            reusable = (
                old_input.is_file()
                and active_input.is_file()
                and old_input.stat().st_size == active_input.stat().st_size
                and hashlib.sha256(old_input.read_bytes()).digest()
                == hashlib.sha256(active_input.read_bytes()).digest()
            )
        if not reusable:
            invalid.append(object_id)
    if invalid and not allow_missing:
        raise RuntimeError(
            "selective image-to-3d generation cannot reuse completed object(s): "
            + ", ".join(invalid)
        )
    for object_id in invalid:
        log_step("image_to_3d", f"skipped missing completed object {object_id}")

    results = []
    for object_id in object_ids:
        if object_id in invalid:
            continue
        shutil.copytree(
            previous / object_id, Path(attempt.temp_root) / object_id,
            copy_function=shutil.copy2 if physical_copy else _link_or_copy,
        )
        source_item = next(
            item for item in manifest["items"] if item["object_id"] == object_id
        )
        results.append({
            "object_id": object_id,
            "input_image": _input_image(source_item),
            "mesh": f"{object_id}/model.glb",
            "preview": f"{object_id}/preview.png",
            "metadata": f"{object_id}/metadata.json",
        })
        log_step("image_to_3d", f"reused completed object {object_id}")
    return results


def _prepare_primary_regeneration(
    context, attempt, manifest_path, manifest, object_ids,
):
    """Copy opened results and build a temporary primary-input manifest."""
    if not object_ids:
        return None, {}
    previous = context.stage_best("image_to_3d")
    previous_record = read_json(previous / "attempt.json")
    if previous_record.get("status") != "completed":
        raise RuntimeError("current image_to_3d best is not completed")
    if previous_record.get("upstream_best") != attempt.upstream_best:
        raise RuntimeError(
            "primary regeneration requires image_to_3d best from current inputs"
        )
    previous_manifest = read_json(previous / "data/image_to_3d_manifest.json")
    previous_items = {
        item["object_id"]: item for item in previous_manifest.get("items", [])
    }
    source_items = {item["object_id"]: item for item in manifest["items"]}
    source_root = manifest_path.parent.parent.resolve()
    overlay_root = Path(attempt.temp_root) / ".primary_input_override"
    overlay_items = []
    variants = {}
    for object_id in object_ids:
        item = source_items[object_id]
        if item.get("reconstruction_input_source") != "opened":
            raise ValueError(
                f"primary regeneration requires opened source: {object_id}"
            )
        primary_relative = item.get("source_condition_image")
        if (
            not isinstance(primary_relative, str)
            or Path(primary_relative).is_absolute()
            or ".." in Path(primary_relative).parts
        ):
            raise ValueError(f"invalid primary input path for {object_id}")
        primary = (source_root / primary_relative).resolve()
        if source_root not in primary.parents or not primary.is_file():
            raise FileNotFoundError(f"missing primary input for {object_id}: {primary}")

        previous_item = previous_items.get(object_id)
        if previous_item is None:
            raise RuntimeError(f"missing completed opened result: {object_id}")
        expected_opened_input = _input_image(item)
        if previous_item.get("input_image") != expected_opened_input:
            raise RuntimeError(
                f"completed result is not the opened variant: {object_id}"
            )
        opened_source = previous / object_id
        if any(
            not (opened_source / name).is_file()
            for name in ("model.glb", "preview.png", "metadata.json")
        ):
            raise RuntimeError(f"incomplete completed opened result: {object_id}")
        opened_destination = Path(attempt.temp_root) / f"{object_id}_opened"
        shutil.copytree(
            opened_source, opened_destination, copy_function=shutil.copy2,
        )
        variants[object_id] = {
            "directory": f"{object_id}_opened",
            "source_attempt": previous.name,
            "input_image": previous_item["input_image"],
            "mesh": f"{object_id}_opened/model.glb",
            "preview": f"{object_id}_opened/preview.png",
            "metadata": f"{object_id}_opened/metadata.json",
        }

        copied_primary = overlay_root / primary_relative
        copied_primary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, copied_primary)
        overlay_item = {**item, "views": {"primary": primary_relative}}
        metadata = read_json(opened_source / "metadata.json")
        seed = metadata.get("trellis2_seed")
        if (
            not isinstance(seed, int) or isinstance(seed, bool)
            or not 0 <= seed < 2**31
        ):
            seed = random.randrange(2**31)
        overlay_item["trellis2_seed"] = seed
        overlay_items.append(overlay_item)

    overlay_manifest = overlay_root / "data/condition_manifest.json"
    write_json(overlay_manifest, {"items": overlay_items})
    return overlay_manifest, variants


def run(
    context, attempt, object_ids=None,
    generate_articulated_objects=True,
    only_generate_articulated_objects=False,
    regenerate_object_ids=None,
    trellis2_runner=None,
    primary_input_object_ids=None,
    preserve_previous_outputs=False,
):
    manifest_path = dependency_file(attempt, "condition_images", "manifest")
    manifest = read_json(manifest_path)
    available = [item["object_id"] for item in manifest["items"]]
    ids = list(available if object_ids is None else object_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("requested object IDs must be unique")
    missing = [object_id for object_id in ids if object_id not in available]
    if missing:
        raise KeyError(f"{missing[0]} not found in condition-image manifest")
    if only_generate_articulated_objects and not generate_articulated_objects:
        raise ValueError("articulated-only generation cannot disable articulated objects")
    if primary_input_object_ids or preserve_previous_outputs:
        raise ValueError("previous-output reuse is not supported")
    if regenerate_object_ids:
        requested = list(dict.fromkeys(regenerate_object_ids))
        unknown = [object_id for object_id in requested if object_id not in ids]
        if unknown:
            raise KeyError(f"{unknown[0]} not found in condition-image manifest")
        ids = requested
    skipped_ids = []
    if not generate_articulated_objects:
        articulated_ids = {
            item["object_id"] for item in manifest["items"]
            if item.get("operability_type") == "articulated_operable"
        }
        skipped_ids = [
            object_id for object_id in ids if object_id in articulated_ids
        ]
        ids = [object_id for object_id in ids if object_id not in articulated_ids]
        for object_id in skipped_ids:
            log_step("image_to_3d", f"skipped articulated object {object_id}")

    from pipeline import trellis2 as backend
    runner = trellis2_runner or backend.run_subprocess
    items = runner(manifest_path, attempt.temp_root, ids)
    required = ("object_id", "input_image", "mesh", "preview", "metadata")
    if [item.get("object_id") for item in items] != ids or len(set(ids)) != len(ids):
        raise ValueError("backend object IDs must exactly match requested objects in order")
    for item in items:
        missing_fields = [field for field in required if not item.get(field)]
        if missing_fields:
            raise ValueError("backend result missing normalized fields: " + ", ".join(missing_fields))
        source_item = next(source for source in manifest["items"] if source["object_id"] == item["object_id"])
        valid_inputs = set(source_item.get("views", {}).values())
        if item["input_image"] not in valid_inputs:
            raise ValueError(f"input_image does not match condition-image manifest: {item['input_image']!r}")
        input_target = (manifest_path.parent.parent / item["input_image"]).resolve()
        upstream_root = manifest_path.parent.parent.resolve()
        if upstream_root not in input_target.parents or not input_target.is_file():
            raise ValueError(f"invalid or missing input_image: {item['input_image']!r}")
        for field in ("mesh", "preview", "metadata"):
            relative = item[field]
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"{field} path must be attempt-relative: {relative!r}")
            target = (attempt.temp_root / relative).resolve()
            if attempt.temp_root.resolve() not in target.parents:
                raise ValueError(f"{field} path must be attempt-relative: {relative!r}")
            if not target.is_file():
                raise ValueError(f"missing normalized {field}: {relative}")
    normalized = {
        "items": items,
        "skipped_items": [
            {
                "object_id": object_id,
                "operability_type": "articulated_operable",
                "reason": "articulated_generation_disabled",
            }
            for object_id in skipped_ids
        ],
    }
    write_json(attempt_path(attempt, "data/image_to_3d_manifest.json"), normalized)
    return normalized
