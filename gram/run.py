#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import attempt_path, read_json, write_json
from gram.urdf_assets import validate_urdf_assets

GRAM_STAGES = (
    "segment", "render", "infer-structure", "build-parts",
    "fit-joints", "physics", "export",
)


def _gram(
    context, attempt, regenerate_object_ids=None,
    from_stage=None, to_stage=None,
):
    from gram.backend import run_backend

    return run_backend(
        context, attempt, regenerate_object_ids=regenerate_object_ids,
        from_stage=from_stage or "segment", to_stage=to_stage or "export",
    )


BACKENDS = MappingProxyType({
    "gram": _gram,
})


def _contained(root: Path, relative, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be an attempt-relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be an attempt-relative path: {relative!r}")
    root = Path(root).absolute()
    target = root / path
    resolved_root, resolved_target = root.resolve(), target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"{field} must remain beneath the attempt: {relative!r}")
    return target


def validate_manifest(
    attempt_root: Path,
    manifest: dict,
    requested_ids: list[str],
    image_to_3d_manifest_path: Path,
):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("method"), str):
        raise ValueError("articulation manifest requires a method")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("articulation manifest requires items")
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("requested object IDs must be unique")
    ids = [item.get("object_id") for item in items if isinstance(item, dict)]
    if len(ids) != len(items) or len(ids) != len(set(ids)):
        raise ValueError("articulation manifest object IDs must be unique")
    if set(ids) != set(requested_ids):
        raise ValueError("articulation manifest must exactly cover requested objects")

    root = Path(attempt_root)
    upstream_path = Path(image_to_3d_manifest_path)
    if not upstream_path.is_file():
        raise ValueError("image-to-3D dependency manifest is not a regular file")
    upstream = read_json(upstream_path)
    upstream_items = upstream.get("items") if isinstance(upstream, dict) else None
    if not isinstance(upstream_items, list):
        raise ValueError("image-to-3D dependency manifest requires items")
    upstream_ids = [
        item.get("object_id") for item in upstream_items if isinstance(item, dict)
    ]
    if len(upstream_ids) != len(upstream_items) or len(upstream_ids) != len(set(upstream_ids)):
        raise ValueError("image-to-3D dependency object IDs must be unique")
    upstream_by_id = {item["object_id"]: item for item in upstream_items}
    missing_upstream = [object_id for object_id in requested_ids if object_id not in upstream_by_id]
    if missing_upstream:
        raise ValueError(
            f"image-to-3D dependency is missing requested object: {missing_upstream[0]}"
        )
    upstream_root = upstream_path.parent.parent

    for item in items:
        if item.get("method") != manifest["method"]:
            raise ValueError("articulation item method does not match manifest")
        if not isinstance(item.get("metadata"), dict):
            raise ValueError("articulation item requires backend metadata")
        upstream_item = upstream_by_id[item["object_id"]]
        expected_sources = [upstream_item.get("mesh")]
        opened_variant = upstream_item.get("opened_variant")
        if isinstance(opened_variant, dict):
            expected_sources.append(opened_variant.get("mesh"))
        actual_source = item.get("source_mesh")
        if actual_source not in expected_sources:
            raise ValueError(
                f"source_mesh provenance does not match image-to-3D manifest: {item['object_id']}"
            )
        source = _contained(upstream_root, actual_source, "source_mesh")
        package = _contained(root, item.get("package"), "package")
        urdf = _contained(root, item.get("urdf"), "urdf")
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"source_mesh is not a regular file: {item.get('source_mesh')!r}")
        if not package.is_dir():
            raise ValueError(f"package is not a directory: {item.get('package')!r}")
        if package not in urdf.parents or not urdf.is_file() or urdf.stat().st_size == 0:
            raise ValueError(f"urdf is not a regular file in package: {item.get('urdf')!r}")
        if urdf != package / "model.urdf":
            raise ValueError("urdf must identify package/model.urdf")
        validate_urdf_assets(package, root)


def run(
    context, attempt, method: str, backends=BACKENDS,
    regenerate_object_ids=None, from_stage=None, to_stage=None,
):
    try:
        backend = backends[method]
    except KeyError:
        raise ValueError(f"unknown articulation method {method!r}") from None
    kwargs = {}
    if regenerate_object_ids is not None:
        kwargs["regenerate_object_ids"] = regenerate_object_ids
    if from_stage is not None:
        kwargs["from_stage"] = from_stage
    if to_stage is not None:
        kwargs["to_stage"] = to_stage
    manifest = backend(context, attempt, **kwargs)
    if manifest is None:
        return None

    blueprint = read_json(dependency_file(attempt, "blueprint", "blueprint"))
    image_to_3d_manifest = dependency_file(attempt, "image_to_3d", "manifest")
    image_document = read_json(image_to_3d_manifest)
    skipped_ids = {
        item["object_id"] for item in image_document.get("skipped_items", [])
    }
    requested_ids = [
        item["object_id"]
        for item in blueprint["objects"]
        if item.get("operability_type") == "articulated_operable"
        and item["object_id"] not in skipped_ids
    ]
    if manifest.get("method") != method:
        raise ValueError("backend method does not match requested articulation method")
    if manifest.get("completed_through") in GRAM_STAGES[:-1]:
        items = manifest.get("items")
        ids = [item.get("object_id") for item in items or ()]
        if set(ids) != set(requested_ids) or len(ids) != len(set(ids)):
            raise ValueError("partial gram manifest must cover requested objects")
        for item in items:
            if not _contained(
                attempt.temp_root, item.get("run_dir"), "run_dir",
            ).is_dir():
                raise ValueError("partial gram run_dir is not a directory")
    else:
        validate_manifest(
            attempt.temp_root, manifest, requested_ids, image_to_3d_manifest,
        )
    write_json(attempt_path(attempt, "data/articulation_manifest.json"), manifest)
    return completed_files(attempt, STAGE_OUTPUTS["articulation"])


__all__ = ["BACKENDS", "GRAM_STAGES", "run", "validate_manifest"]
