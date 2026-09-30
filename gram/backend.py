#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from pipeline.stage_io import dependency_file
from pipeline.common import env_positive_int, log_step, mesh_path, read_json, write_json
from pipeline.mllm import gram_runtime
from gram.urdf_assets import validate_urdf_assets


ROOT = Path(__file__).resolve().parents[1]
REUSED_DIRECTORIES = ("02_views", "03_kinematics", "05_joints")
STAGE_LABELS = {
    "segment": "P3-SAM primitive segmentation",
    "render": "multi-view rendering",
    "infer-structure": "MLLM kinematic inference",
    "build-parts": "rigid-part construction",
    "fit-joints": "joint fitting",
    "physics": "physics optimization",
    "export": "URDF export",
}
STAGE_NUMBERS = {
    "segment": 1,
    "render": 2,
    "infer-structure": 3,
    "build-parts": 4,
    "fit-joints": 5,
    "physics": 6,
    "export": 7,
}
STAGE_COUNT = 7
GRAM_FEATURE_FLAGS = (
    ("GRAM_ENABLE_REVOLUTE_SUBTYPE_OVERRIDE", "--enable-revolute-subtype-override"),
    ("GRAM_ENABLE_JOINT_REFINEMENT", "--enable-joint-refinement"),
    ("GRAM_ENABLE_SPIN_UPPER_COLLISION_SCAN", "--enable-spin-upper-collision-scan"),
)
GRAM_JOINT_SELECTION_MODES = {
    "conservative", "high-recall", "high-recall-v2",
}


def _enabled(name):
    return os.environ.get(name, "0").lower() in {"1", "true", "yes"}


def resolve_conda_environment(conda: Path, environment: str) -> Path:
    result = subprocess.run(
        [str(conda), "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    environments = json.loads(result.stdout).get("envs", [])
    matches = [Path(item).resolve() for item in environments if Path(item).name == environment]
    if len(matches) != 1 or not matches[0].is_dir():
        raise FileNotFoundError(f"GRAM conda environment is unavailable: {environment}")
    return matches[0]


def gram_vlm_execution_mode(environment=None):
    return gram_runtime(environment).execution_mode


def _saved_model_runtime_matches(config, expected):
    saved = config.get("model_runtime")
    if isinstance(saved, dict):
        return all(saved.get(key) == expected.get(key) for key in (
            "provider", "model", "reasoning_effort", "execution_mode",
            "api_base",
        ))
    return False


def gram_feature_flags(environment=None):
    environment = os.environ if environment is None else environment
    return [
        flag for name, flag in GRAM_FEATURE_FLAGS
        if environment.get(name, "0").lower() in {"1", "true", "yes"}
    ]


def gram_joint_selection_flags(environment=None):
    environment = os.environ if environment is None else environment
    mode = environment.get("GRAM_JOINT_SELECTION_MODE", "conservative")
    if mode not in GRAM_JOINT_SELECTION_MODES:
        raise ValueError(f"unsupported GRAM joint selection mode: {mode}")
    return ["--joint-selection-mode", mode]


def _reuse_stages(enable_subtype_override, drawer_geometry_mode="auto-3d"):
    if drawer_geometry_mode in {"procedural-closed", "auto-3d"}:
        return ("build-parts", "fit-joints", "physics", "export")
    return ("build-parts", "fit-joints", "physics", "export") if enable_subtype_override else (
        "build-parts", "physics", "export")


def _run_logged(command, cwd, output, env, object_id, runner):
    if runner is not subprocess.run:
        return runner(
            command, cwd=cwd, check=True, stdout=output,
            stderr=subprocess.STDOUT, env=env,
        )
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    current_stage = None
    last_line = ""
    for line in process.stdout:
        output.write(line.encode("utf-8"))
        output.flush()
        if line.strip():
            last_line = line.strip()
        if line.startswith("[gram-stage]"):
            stage, status = line.split("] ", 1)[1].rstrip().rsplit(" ", 1)
            if status not in {"started", "completed"} or stage not in STAGE_NUMBERS:
                continue
            current_stage = stage if status == "started" else None
            number = STAGE_NUMBERS[stage]
            log_step(
                "gram",
                f"[{object_id}][{number}/{STAGE_COUNT}]"
                f"[{stage}: {STAGE_LABELS[stage]}] {status}",
            )
    returncode = process.wait()
    if returncode:
        if current_stage:
            if current_stage == "segment" and "--run-dir" in command:
                p3sam_log = (
                    Path(command[command.index("--run-dir") + 1])
                    / "01_primitives/p3sam.log"
                )
                if p3sam_log.is_file():
                    lines = [
                        line for line in p3sam_log.read_text(
                            encoding="utf-8", errors="replace",
                        ).splitlines() if line.strip()
                    ]
                    if lines:
                        last_line = lines[-1]
            number = STAGE_NUMBERS[current_stage]
            log_step(
                "gram",
                f"[{object_id}][{number}/{STAGE_COUNT}]"
                f"[{current_stage}: {STAGE_LABELS[current_stage]}] "
                f"failed: {last_line}",
            )
        raise subprocess.CalledProcessError(returncode, command)


def normalize_visual_meshes(
    package, blender, runner=subprocess.run, output=None, env=None,
):
    urdf = package / "model.urdf"
    tree = ET.parse(urdf)
    for visual in tree.getroot().iter("visual"):
        mesh = visual.find("./geometry/mesh")
        if mesh is None or Path(mesh.attrib["filename"]).suffix.lower() == ".glb":
            continue
        source = package / mesh.attrib["filename"]
        target = source.with_suffix(".glb")
        runner([
            str(blender), "--background", "--python",
            str(Path(__file__).with_name("blender_obj_to_glb.py")), "--",
            "--input", str(source), "--output", str(target),
        ], check=True, stdout=output, stderr=subprocess.STDOUT, env=env)
        mesh.attrib["filename"] = str(target.relative_to(package))
    tree.write(urdf, encoding="utf-8", xml_declaration=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_item_path(manifest_path, item):
    root = Path(manifest_path).resolve().parent.parent
    relative = item.get("mesh")
    if not isinstance(relative, str):
        raise ValueError("image-to-3D item requires a mesh path")
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image-to-3D mesh must be attempt-relative: {relative}")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"image-to-3D mesh is missing: {path}")
    return path


def _preserved_checkpoint_source(manifest_path, item, checkpoint):
    """Match a checkpoint to its physical copy in the current 3D best."""
    saved_input = Path(read_json(checkpoint / "run_config.json")["input"])
    opened = item.get("opened_variant")
    source_attempt = next(
        (
            parent.name for parent in saved_input.parents
            if parent.parent.name == "attempts"
        ),
        None,
    )
    candidates = [("main", item)]
    if isinstance(opened, dict):
        candidates.append(("opened", {**item, **opened}))

    if isinstance(opened, dict) and opened.get("source_attempt") == source_attempt:
        selected = {**item, **opened}
        selected["_preserved_source"] = "opened"
        if saved_input.is_file() and _sha256(saved_input) != _sha256(
            _manifest_item_path(manifest_path, selected)
        ):
            raise RuntimeError(
                f"preserved opened source differs for {item['object_id']}"
            )
        return selected

    if saved_input.is_file():
        saved_hash = _sha256(saved_input)
        for name, candidate in candidates:
            if _sha256(_manifest_item_path(manifest_path, candidate)) == saved_hash:
                selected = dict(candidate)
                selected["_preserved_source"] = name
                return selected
        raise RuntimeError(
            f"no preserved source matches checkpoint for {item['object_id']}"
        )

    if not isinstance(opened, dict):
        selected = dict(item)
        selected["_preserved_source"] = "main"
        return selected
    raise RuntimeError(
        f"cannot identify preserved checkpoint source for {item['object_id']}"
    )


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _primary_image(condition_root, condition_item, object_id):
    relative = condition_item.get("source_condition_image") if condition_item else None
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"gram primary image is missing for {object_id}: "
            "condition manifest requires source_condition_image"
        )
    root = Path(condition_root).resolve()
    image = (root / relative).resolve()
    if root not in image.parents or not image.is_file():
        raise FileNotFoundError(
            f"gram primary image is not a regular condition asset for {object_id}: {relative}"
        )
    return image


def _requires_open_state(
    condition_root, condition_item, object_id, source_image, primary_image,
):
    articulation_view = (
        condition_item.get("articulation_view") if condition_item else None
    )
    if articulation_view is None:
        return False
    relative = articulation_view.get("decision")
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"gram articulation-view decision is missing for {object_id}"
        )
    root = Path(condition_root).resolve()
    decision_path = (root / relative).resolve()
    if root not in decision_path.parents or not decision_path.is_file():
        raise FileNotFoundError(
            f"gram articulation-view decision is not a regular condition "
            f"asset for {object_id}: {relative}"
        )
    requires_open_state = read_json(decision_path).get("requires_open_state")
    if not isinstance(requires_open_state, bool):
        raise ValueError(
            f"gram requires_open_state must be boolean for {object_id}"
        )
    reconstruction_source = condition_item.get("reconstruction_input_source")
    if reconstruction_source not in {"primary", "opened"}:
        raise ValueError(
            f"gram reconstruction source is invalid for {object_id}"
        )
    differs_from_primary = _sha256(source_image) != _sha256(primary_image)
    if reconstruction_source == "primary" and differs_from_primary:
        raise ValueError(
            f"gram primary reconstruction source differs from primary image "
            f"for {object_id}"
        )
    if reconstruction_source == "opened" and not requires_open_state:
        raise ValueError(
            f"gram opened reconstruction source is not required for {object_id}"
        )
    return reconstruction_source == "opened" and differs_from_primary


def _publish_rest_mesh(package, object_id):
    manifest = read_json(package / "manifest.json")
    states = manifest.get("joint_states")
    if not isinstance(states, list):
        raise ValueError(f"gram rest export has no joint states for {object_id}")
    if not states and not manifest.get("rigid_fallback"):
        raise ValueError(f"gram rest export has no joint states for {object_id}")
    if any(abs(float(state["scene_current_q"])) > 1e-8 for state in states):
        raise ValueError(
            f"gram rest export is not at lower q=0 for {object_id}"
        )
    source = package / manifest["scene_rigid_mesh"]
    if not source.is_file():
        raise FileNotFoundError(
            f"gram scene-rigid mesh is missing for {object_id}: {source}"
        )
    rest = package / "rest.glb"
    shutil.copy2(source, rest)
    return rest


def _attach_rest_exports(items, attempt, requires_open_state, partial):
    for item in items.values():
        object_id = item["object_id"]
        required = requires_open_state[object_id]
        item["requires_open_state"] = required
        item.pop("rest_mesh", None)
        if partial or not required:
            continue
        package = attempt.temp_root / item["package"]
        rest = _publish_rest_mesh(package, object_id)
        item["rest_mesh"] = str(rest.relative_to(attempt.temp_root))


def _reuse_failed_object(
    context, attempt, object_id, source_mesh, source_relative,
    drawer_geometry_mode, enable_subtype_override, enable_joint_refinement,
    enable_spin_upper_collision_scan, vlm_runtime=None,
):
    if (
        context is None
        or not hasattr(attempt, "stage")
        or not hasattr(attempt, "final_root")
    ):
        return None
    vlm_runtime = vlm_runtime or gram_runtime().metadata()
    attempts = context._stage_root(attempt.stage) / "attempts"
    for candidate in sorted(attempts.glob("*"), reverse=True):
        record = candidate / "attempt.json"
        if candidate == attempt.final_root or not record.is_file():
            continue
        try:
            if read_json(record).get("status") != "failed":
                continue
            raw = candidate / "raw/gram" / object_id
            config = read_json(raw / "run_config.json")
            old_input = Path(config["input"])
            package = candidate / "packages" / object_id
            if (
                config.get("drawer_geometry_mode", "auto-3d")
                != drawer_geometry_mode
                or bool(config.get("enable_revolute_subtype_override"))
                != enable_subtype_override
                or bool(config.get("enable_joint_refinement"))
                != enable_joint_refinement
                or bool(config.get("enable_spin_upper_collision_scan", True))
                != enable_spin_upper_collision_scan
                or not _saved_model_runtime_matches(config, vlm_runtime)
                or not old_input.is_file()
                or _sha256(old_input) != _sha256(source_mesh)
                or not all(
                    (package / f"joint_range_images/{pose}.png").is_file()
                    for pose in ("lower", "rest", "upper")
                )
            ):
                continue
            validate_urdf_assets(package, candidate)
        except (KeyError, OSError, TypeError, ValueError):
            continue
        shutil.copytree(
            package, attempt.temp_root / "packages" / object_id,
            dirs_exist_ok=True,
        )
        shutil.copytree(
            raw, attempt.temp_root / "raw/gram" / object_id,
            dirs_exist_ok=True,
        )
        log_step(
            "gram",
            f"[{object_id}] reused completed object from failed attempt "
            f"{candidate.name}",
        )
        return {
            "object_id": object_id,
            "method": "gram",
            "source_mesh": source_relative,
            "package": f"packages/{object_id}",
            "urdf": f"packages/{object_id}/model.urdf",
            "joint_range_images": {
                pose: f"packages/{object_id}/joint_range_images/{pose}.png"
                for pose in ("lower", "rest", "upper")
            },
            "metadata": {
                "backend": "subprocess",
                "reused_from_failed_attempt": candidate.name,
            },
        }
    return None


def _reuse_completed_best_objects(
    context, attempt, articulated_items, regenerate_object_ids,
    sources, image_manifest, partial=False, vlm_runtime=None,
):
    vlm_runtime = vlm_runtime or gram_runtime().metadata()
    if regenerate_object_ids is None:
        return {}, articulated_items
    requested = list(regenerate_object_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("gram object IDs must be unique")
    available = {item["object_id"] for item in articulated_items}
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"unknown articulated gram object: {sorted(unknown)[0]}")
    selected = set(requested)
    pending = [item for item in articulated_items if item["object_id"] in selected]
    reused_items = [item for item in articulated_items if item["object_id"] not in selected]
    if not reused_items:
        return {}, pending
    if context is None:
        raise ValueError("selective gram generation requires an existing run")

    best = context.stage_best("gram")
    best_record = read_json(best / "attempt.json")
    best_manifest = read_json(best / best_record["artifacts"]["manifest"])
    best_by_id = {item["object_id"]: item for item in best_manifest["items"]}
    image_attempt_id = best_record["upstream_best"]["image_to_3d"]
    old_image_root = context._stage_root("image_to_3d") / "attempts" / image_attempt_id
    old_image_record = read_json(old_image_root / "attempt.json")
    old_image_manifest = old_image_root / old_image_record["artifacts"]["manifest"]

    reused = {}
    for item in reused_items:
        object_id = item["object_id"]
        if object_id not in best_by_id:
            raise RuntimeError(
                f"selective gram cannot reuse missing completed object: {object_id}"
            )
        old_mesh = mesh_path(old_image_manifest, object_id)
        current_mesh = mesh_path(image_manifest, object_id)
        if (
            old_mesh.resolve() != current_mesh.resolve()
            and _sha256(old_mesh) != _sha256(current_mesh)
        ):
            raise RuntimeError(
                f"selective gram cannot reuse stale completed object: {object_id}"
            )
        if not partial:
            source_package = best / "packages" / object_id
            validate_urdf_assets(source_package, best)
            shutil.copytree(
                source_package, attempt.temp_root / "packages" / object_id,
                dirs_exist_ok=True,
            )
        source_raw = best / "raw/gram" / object_id
        run_config_path = source_raw / "run_config.json"
        source_config = read_json(run_config_path) if run_config_path.is_file() else {}
        if source_raw.is_dir() and not _saved_model_runtime_matches(
            source_config, vlm_runtime,
        ):
            raise RuntimeError(
                f"selective gram cannot mix model runtimes for {object_id}"
            )
        if source_raw.is_dir():
            shutil.copytree(
                source_raw, attempt.temp_root / "raw/gram" / object_id,
                dirs_exist_ok=True,
            )
        reused[object_id] = {
            **best_by_id[object_id],
            "source_mesh": sources[object_id]["mesh"],
            **({"run_dir": f"raw/gram/{object_id}"} if partial else {}),
            "metadata": {
                **best_by_id[object_id].get("metadata", {}),
                "reused_from_completed_attempt": best.name,
            },
        }
        log_step(
            "gram",
            f"[{object_id}] reused completed object from best attempt {best.name}",
        )
    return reused, pending


def run_backend(
    context,
    attempt,
    runner=subprocess.run,
    *,
    blueprint_data=None,
    image_manifest_path=None,
    condition_manifest_data=None,
    condition_root=None,
    visible_gpus=None,
    reuse_stream=True,
    regenerate_object_ids=None,
    from_stage="segment",
    to_stage="export",
    storage_suffix="",
    checkpoint_root=None,
):
    stages = tuple(STAGE_NUMBERS)
    if from_stage not in stages or to_stage not in stages:
        raise ValueError("unknown gram internal stage")
    if stages.index(from_stage) > stages.index(to_stage):
        raise ValueError("gram from_stage must not follow to_stage")
    runtime = gram_runtime()
    vlm_runtime = runtime.metadata()
    vlm_execution_mode = runtime.execution_mode
    if storage_suffix and not re.fullmatch(r"_[a-z0-9][a-z0-9_-]*", storage_suffix):
        raise ValueError("gram storage suffix must be a safe lowercase suffix")
    partial = to_stage != "export"
    blueprint = blueprint_data or read_json(
        dependency_file(attempt, "blueprint", "blueprint")
    )
    image_manifest = (
        Path(image_manifest_path)
        if image_manifest_path is not None
        else dependency_file(attempt, "image_to_3d", "manifest")
    )
    image_document = read_json(image_manifest)
    sources = {item["object_id"]: item for item in image_document["items"]}
    skipped_ids = {
        item["object_id"] for item in image_document.get("skipped_items", [])
    }
    articulated_items = [
        item for item in blueprint["objects"]
        if item.get("operability_type") == "articulated_operable"
        and item["object_id"] not in skipped_ids
    ]
    if not articulated_items:
        return {
            "method": "gram", "items": [],
            **({"completed_through": to_stage} if partial else {}),
        }
    if condition_manifest_data is not None:
        if condition_root is None:
            raise ValueError("gram condition manifest data requires condition_root")
        condition_items = {
            item["object_id"]: item
            for item in condition_manifest_data["items"]
        }
        condition_root = Path(condition_root)
    else:
        condition_manifest = dependency_file(
            attempt, "condition_images", "manifest"
        )
        condition_items = {
            item["object_id"]: item
            for item in read_json(condition_manifest)["items"]
        }
        condition_root = condition_manifest.parents[1]
    primary_images = {
        item["object_id"]: _primary_image(
            condition_root, condition_items.get(item["object_id"]), item["object_id"],
        )
        for item in articulated_items
    }
    articulated_ids = [item["object_id"] for item in articulated_items]
    checkpoint_root = (
        Path(checkpoint_root)
        if checkpoint_root is not None and from_stage != "segment"
        else context.stage_best("gram")
        if context is not None and from_stage != "segment"
        else None
    )
    use_preserved_source = (
        checkpoint_root is not None
        and _enabled("GRAM_USE_PRESERVED_SOURCE")
    )
    if use_preserved_source:
        for object_id in articulated_ids:
            sources[object_id] = _preserved_checkpoint_source(
                image_manifest,
                sources[object_id],
                checkpoint_root / "raw/gram" / f"{object_id}{storage_suffix}",
            )
    source_root = image_manifest.parent.parent
    if not all(
        (source_root / sources[object_id]["input_image"]).is_file()
        for object_id in articulated_ids
    ):
        upstream = read_json(source_root / "attempt.json")["upstream_best"]
        source_stage = "condition_images"
        if source_stage not in upstream:
            raise ValueError("GRAM requires condition_images provenance")
        stages_root = source_root.parents[1]
        source_root = next(stages_root.glob(f"[0-9][0-9]_{source_stage}")) \
            / "attempts" / upstream[source_stage]
    requires_open_state = {
        item["object_id"]: _requires_open_state(
            condition_root,
            condition_items.get(item["object_id"]),
            item["object_id"],
            source_root / sources[item["object_id"]]["input_image"],
            primary_images[item["object_id"]],
        )
        for item in articulated_items
    }

    enable_subtype_override = _enabled("GRAM_ENABLE_REVOLUTE_SUBTYPE_OVERRIDE")
    enable_joint_refinement = _enabled("GRAM_ENABLE_JOINT_REFINEMENT")
    enable_spin_upper_collision_scan = _enabled(
        "GRAM_ENABLE_SPIN_UPPER_COLLISION_SCAN"
    )
    streamed, articulated_items = _reuse_completed_best_objects(
        context, attempt, articulated_items, regenerate_object_ids,
        sources, image_manifest, partial=partial,
        vlm_runtime=vlm_runtime,
    )
    regenerate_ids = set(regenerate_object_ids or ())
    stream_root = (
        Path(condition_root) / ".gram_stream"
        if reuse_stream and condition_root is not None
        and (from_stage, to_stage) == ("segment", "export") else None
    )
    stream_manifest = stream_root / "manifest.json" if stream_root else None
    if stream_manifest is not None and stream_manifest.is_file():
        for record in read_json(stream_manifest).get("items", []):
            stream_object_id = record.get("object_id")
            job_id = record.get("job_id", stream_object_id)
            object_id = next(
                (
                    candidate_id
                    for candidate_id, candidate in condition_items.items()
                    if candidate.get("front_segment_id", candidate_id) == job_id
                ),
                stream_object_id,
            )
            if object_id not in sources or object_id in streamed or object_id in regenerate_ids:
                continue
            source_mesh = mesh_path(image_manifest, object_id)
            saved = stream_root / "outputs" / job_id
            try:
                saved_config = read_json(
                    saved / f"raw/gram/{stream_object_id}/run_config.json"
                )
            except (OSError, TypeError, ValueError):
                continue
            if (
                record.get("source_mesh_sha256") != _sha256(source_mesh)
                or bool(saved_config.get("enable_spin_upper_collision_scan", True))
                != enable_spin_upper_collision_scan
                or not _saved_model_runtime_matches(
                    saved_config, vlm_runtime,
                )
                or not (
                    saved / f"packages/{stream_object_id}/model.urdf"
                ).is_file()
            ):
                continue
            shutil.copytree(
                saved / "packages" / stream_object_id,
                attempt.temp_root / "packages" / object_id,
                dirs_exist_ok=True,
            )
            raw = saved / "raw/gram" / stream_object_id
            if raw.is_dir():
                shutil.copytree(
                    raw,
                    attempt.temp_root / "raw/gram" / object_id,
                    dirs_exist_ok=True,
                )
            streamed[object_id] = {
                "object_id": object_id,
                "method": "gram",
                "source_mesh": sources[object_id]["mesh"],
                "package": f"packages/{object_id}",
                "urdf": f"packages/{object_id}/model.urdf",
                "joint_range_images": {
                    pose: f"packages/{object_id}/joint_range_images/{pose}.png"
                    for pose in ("lower", "rest", "upper")
                },
                "metadata": {
                    **(record.get("metadata") or {}),
                    "streamed_from": job_id,
                },
            }
            log_step("gram", f"[{object_id}] reused streamed result")

    if (from_stage, to_stage) == ("segment", "export"):
        for item in articulated_items:
            object_id = item["object_id"]
            if object_id in streamed or object_id in regenerate_ids:
                continue
            drawer_geometry_mode = condition_items.get(object_id, {}).get(
                "drawer_geometry_mode", "auto-3d",
            )
            reused = _reuse_failed_object(
                context, attempt, object_id, mesh_path(image_manifest, object_id),
                sources[object_id]["mesh"], drawer_geometry_mode,
                enable_subtype_override, enable_joint_refinement,
                enable_spin_upper_collision_scan, vlm_runtime,
            )
            if reused is not None:
                streamed[object_id] = reused

    articulated_items = [
        item for item in articulated_items if item["object_id"] not in streamed
    ]
    if not articulated_items:
        _attach_rest_exports(
            streamed, attempt, requires_open_state, partial,
        )
        ordered = [
            streamed[item["object_id"]]
            for item in blueprint["objects"]
            if item.get("operability_type") == "articulated_operable"
            and item["object_id"] not in skipped_ids
        ]
        return {"method": "gram", "items": ordered}

    conda = Path(shutil.which("conda") or "conda").resolve()
    environment = os.environ.get("GRAM_CONDA_ENV")
    if not environment:
        raise RuntimeError("GRAM_CONDA_ENV must be configured")
    python = resolve_conda_environment(conda, environment) / "bin/python"
    blender = Path(
        os.environ.get("GRAM_BLENDER") or shutil.which("blender") or "blender"
    )
    checkout = Path(os.environ.get(
        "GRAM_P3SAM_CHECKOUT", ROOT / "external/Hunyuan3D-Part/P3-SAM",
    )).expanduser()
    checkpoint = Path(os.environ.get(
        "GRAM_P3SAM_CHECKPOINT", ROOT / "checkpoints/p3sam/p3sam.safetensors",
    )).expanduser()
    sonata_dir = Path(
        os.environ.get("GRAM_SONATA_DIR", "~/.cache/sonata")
    ).expanduser()
    for target, name in (
        (ROOT, "module"), (python, "Python"), (blender, "Blender"),
        (checkout, "P3-SAM checkout"), (checkpoint, "P3-SAM checkpoint"),
    ):
        if not target.exists():
            raise FileNotFoundError(f"gram {name} is missing: {target}")

    reuse_root = (
        context.stage_best("gram")
        if context is not None
        and regenerate_object_ids is None
        and (from_stage, to_stage) == ("segment", "export")
        and _enabled("GRAM_REUSE_BEST")
        else None
    )
    enable_flags = [
        *gram_feature_flags(),
        *gram_joint_selection_flags(),
        "--mllm-model", runtime.model,
    ]
    if visible_gpus is None:
        visible_gpus = [
            gpu.strip()
            for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if gpu.strip()
        ]
    else:
        visible_gpus = list(visible_gpus)
    gpu_limit = len(visible_gpus) or 1
    workers = min(
        len(articulated_items),
        gpu_limit,
        env_positive_int("GRAM_MAX_WORKERS", gpu_limit),
    )
    base_environment = os.environ.copy()
    log_step(
        "gram",
        f"objects={len(articulated_items)}, workers={workers}, "
        f"visible_gpus={gpu_limit}, "
        f"mllm={runtime.provider}/{runtime.model}/{runtime.reasoning_effort}/"
        f"{runtime.execution_mode}",
    )

    def process(index, item):
        object_id = item["object_id"]
        storage_id = f"{object_id}{storage_suffix}"
        source_mesh = _manifest_item_path(image_manifest, sources[object_id])
        environment_variables = base_environment.copy()
        if visible_gpus:
            environment_variables["CUDA_VISIBLE_DEVICES"] = visible_gpus[
                index % len(visible_gpus)
            ]
        gpu = environment_variables.get("CUDA_VISIBLE_DEVICES", "default")
        log_step(
            "gram",
            f"[{object_id}][GPU {gpu}] object started",
        )
        reconstruction_image = source_root / sources[object_id]["input_image"]
        condition_item = condition_items.get(object_id)
        primary_image = primary_images[object_id]
        drawer_geometry_mode = (condition_item or {}).get(
            "drawer_geometry_mode", "auto-3d",
        )
        run_dir = attempt.temp_root / "raw/gram" / storage_id
        package = attempt.temp_root / "packages" / storage_id
        log = attempt.temp_root / "logs" / f"{storage_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), "-m", "gram.cli", "all",
            "--input", str(source_mesh),
            "--source-image", str(reconstruction_image),
            "--primary-image", str(primary_image),
            "--drawer-geometry-mode", drawer_geometry_mode,
            "--run-dir", str(run_dir),
            "--p3sam-python", str(python),
            "--p3sam-checkout", str(checkout),
            "--p3sam-checkpoint", str(checkpoint),
            "--p3sam-sonata-dir", str(sonata_dir),
            "--blender", str(blender),
            *enable_flags,
        ]
        if (from_stage, to_stage) != ("segment", "export"):
            command.extend(["--from-stage", from_stage, "--to-stage", to_stage])
        with log.open("wb") as output:
            saved_checkpoint = (
                checkpoint_root / "raw/gram" / storage_id
                if checkpoint_root else None
            )
            if saved_checkpoint is not None:
                if not saved_checkpoint.is_dir():
                    raise FileNotFoundError(
                        f"gram checkpoint is missing for {object_id}: {saved_checkpoint}"
                    )
                saved_input = Path(read_json(
                    saved_checkpoint / "run_config.json"
                )["input"])
                if (
                    saved_input.is_file()
                    and _sha256(saved_input) != _sha256(source_mesh)
                ) or (not saved_input.is_file() and not use_preserved_source):
                    raise RuntimeError(
                        f"gram checkpoint source changed for {object_id}"
                    )
                if saved_checkpoint.resolve() != run_dir.resolve():
                    shutil.copytree(saved_checkpoint, run_dir, dirs_exist_ok=True)
                _run_logged(
                    command, ROOT, output, environment_variables, object_id, runner,
                )
            saved = reuse_root / "raw/gram" / object_id if reuse_root else None
            reused = (
                saved is not None
                and read_json(saved / "run_config.json").get(
                    "drawer_geometry_mode", "auto-3d",
                ) == drawer_geometry_mode
                and _saved_model_runtime_matches(
                    read_json(saved / "run_config.json"), vlm_runtime,
                )
            )
            if saved_checkpoint is not None:
                pass
            elif reused:
                reused_directories = (
                    REUSED_DIRECTORIES[:2]
                    if drawer_geometry_mode in {"procedural-closed", "auto-3d"}
                    else REUSED_DIRECTORIES
                )
                for name in reused_directories:
                    shutil.copytree(saved / name, run_dir / name)
                runner([
                    str(python), str(Path(__file__).with_name("restore_textured_primitives.py")),
                    "--input", str(source_mesh),
                    "--saved-primitives", str(saved / "01_primitives"),
                    "--output-dir", str(run_dir / "01_primitives"),
                ], cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT,
                    env=environment_variables)
                old_config = read_json(saved / "run_config.json")
                common = [
                    "--input", str(source_mesh),
                    "--run-dir", str(run_dir),
                    "--source-image", str(reconstruction_image),
                    "--primary-image", str(primary_image),
                    "--drawer-geometry-mode", drawer_geometry_mode,
                    "--min-link-faces", str(old_config["min_link_faces"]),
                    "--contact-ratio", str(old_config["contact_ratio"]),
                    "--collision-clearance", str(old_config["collision_clearance"]),
                    "--trajectory-span", str(old_config["trajectory_span"]),
                    "--limit-step", str(old_config["limit_step"]),
                    "--density", str(old_config["density"]),
                    *enable_flags,
                ]
                for stage in _reuse_stages(
                    enable_subtype_override, drawer_geometry_mode,
                ):
                    _run_logged(
                        [str(python), "-m", "gram.cli", stage, *common],
                        ROOT, output, environment_variables, object_id, runner,
                    )
            else:
                _run_logged(
                    command, ROOT, output, environment_variables, object_id, runner,
                )
        fallback_path = run_dir / "05_joints/rigid_fallback.json"
        rigid_fallback = read_json(fallback_path) if fallback_path.is_file() else None
        if partial:
            return {
                "object_id": object_id,
                "source_mesh": sources[object_id]["mesh"],
                "run_dir": str(run_dir.relative_to(attempt.temp_root)),
                "metadata": {
                    "backend": "subprocess", "environment": environment,
                    **({"rigid_fallback": rigid_fallback} if rigid_fallback else {}),
                },
            }
        shutil.copytree(run_dir / "07_urdf", package, dirs_exist_ok=True)
        with log.open("ab") as output:
            normalize_visual_meshes(
                package, blender, runner, output, environment_variables,
            )
        validate_urdf_assets(package, attempt.temp_root)
        preview = package / "joint_range_images"
        log_step("gram", f"[{object_id}][preview] rendering joint-range images")
        with log.open("ab") as output:
            runner([
                str(blender), "--background", "--python-exit-code", "1",
                "--python", str(Path(__file__).with_name("visualize_urdf_joint.py")), "--",
                "--urdf", str(package / "model.urdf"),
                "--mesh-root", str(package),
                "--image-output-dir", str(preview),
                "--preserve-materials",
            ], cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT,
                env=environment_variables)
        log_step(
            "gram",
            f"[{object_id}][completed] object completed",
        )
        return {
            "object_id": object_id,
            "method": "gram",
            "source_mesh": sources[object_id]["mesh"],
            "package": str(package.relative_to(attempt.temp_root)),
            "urdf": str((package / "model.urdf").relative_to(attempt.temp_root)),
            "joint_range_images": {
                pose: str((preview / f"{pose}.png").relative_to(attempt.temp_root))
                for pose in ("lower", "rest", "upper")
            },
            "metadata": {
                "backend": "subprocess", "environment": environment,
                **({"reused_from": reuse_root.name} if reused else {}),
                **({"rigid_fallback": rigid_fallback} if rigid_fallback else {}),
            },
        }

    items = [None] * len(articulated_items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process, index, item): index
            for index, item in enumerate(articulated_items)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                items[index] = future.result()
            except Exception:
                log_step(
                    "gram",
                    f"[{articulated_items[index]['object_id']}][failed] "
                    "object failed; inspect its log for the traceback",
                )
                raise
    generated = {
        item["object_id"]: item for item in items
    }
    generated.update(streamed)
    _attach_rest_exports(
        generated, attempt, requires_open_state, partial,
    )
    ordered = [
        generated[item["object_id"]]
        for item in blueprint["objects"]
        if item.get("operability_type") == "articulated_operable"
        and item["object_id"] not in skipped_ids
    ]
    return {
        "method": "gram", "items": ordered,
        **({"completed_through": to_stage} if partial else {}),
    }


def run_streamed_object(
    context,
    condition_root,
    item,
    image_output,
    stream_root,
    job_id,
    gpu_id=None,
    runner=subprocess.run,
):
    """Build one articulated object as soon as its Image-to-3D result exists."""
    object_id = item["object_id"]
    output_root = Path(stream_root) / "outputs" / job_id
    input_root = output_root / "_image_to_3d"
    source_root = Path(condition_root)
    source_image = source_root / (
        item["views"].get("front")
        or item["views"].get("primary")
        or next(iter(item["views"].values()))
    )
    _copy_file(Path(image_output) / "model.glb", input_root / object_id / "model.glb")
    _copy_file(source_image, input_root / "source.png")
    image_manifest = input_root / "data/image_to_3d_manifest.json"
    write_json(image_manifest, {"items": [{
        "object_id": object_id,
        "mesh": f"{object_id}/model.glb",
        "input_image": "source.png",
    }]})
    manifest = run_backend(
        context,
        SimpleNamespace(temp_root=output_root),
        runner=runner,
        blueprint_data={"objects": [{
            "object_id": object_id,
            "operability_type": "articulated_operable",
        }]},
        image_manifest_path=image_manifest,
        condition_manifest_data={"items": [item]},
        condition_root=source_root,
        visible_gpus=[gpu_id] if gpu_id is not None else [],
        reuse_stream=False,
    )
    result = manifest["items"][0]
    return {
        **result,
        "job_id": job_id,
        "source_mesh_sha256": _sha256(Path(image_output) / "model.glb"),
    }
