import json
import hashlib
import os
import re
import shutil
import tempfile
import traceback
import unicodedata
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from pipeline.mllm import runtime_snapshot
except ModuleNotFoundError:  # Imported as scripts.run_context.
    from scripts.mllm import runtime_snapshot


ARTICULATION_METHODS = {"gram"}
DEFAULT_ARTICULATION_METHOD = "gram"
ASSEMBLY_ASSET_SOURCES = {"raw", "articulated"}
DRAWER_GEOMETRY_MODES = {"reconstructed-open", "procedural-closed", "auto-3d"}
DEFAULT_DRAWER_GEOMETRY_MODE = "auto-3d"


def validate_articulation_method(value: str) -> str:
    if value not in ARTICULATION_METHODS:
        raise ValueError(f"unknown articulation method: {value}")
    return value


def articulation_method_from_metadata(metadata) -> str:
    return validate_articulation_method(
        metadata.get("articulation_method", DEFAULT_ARTICULATION_METHOD)
    )


def validate_assembly_asset_source(value: str) -> str:
    if value not in ASSEMBLY_ASSET_SOURCES:
        raise ValueError(f"unknown assembly asset source: {value}")
    return value


def validate_drawer_geometry_mode(value: str) -> str:
    if value not in DRAWER_GEOMETRY_MODES:
        raise ValueError(f"unknown drawer geometry mode: {value}")
    return value


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    result = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    if not result.strip("_"):
        raise ValueError("slug cannot be empty")
    return result


def build_run_id(
    started_at: datetime,
    method: str,
    label: Optional[str],
    articulation_method: str = DEFAULT_ARTICULATION_METHOD,
) -> str:
    parts = [started_at.strftime("%Y%m%d_%H%M%S")]
    if label is not None:
        if label in {"", ".", ".."} or "/" in label or "\\" in label:
            raise ValueError("invalid run label")
        normalized_label = slug(label)
        if normalized_label == "best":
            raise ValueError("run label 'best' is reserved")
        parts.append(normalized_label)
    parts.append(slug(method))
    parts.append(slug(articulation_method))
    return "_".join(parts)


def resolve_run_root(project_root, scene_id: str, run_id: str) -> Path:
    project_root = Path(project_root).resolve()
    normalized_scene_id = slug(scene_id)
    runs_root = project_root / "scenes" / normalized_scene_id / "runs"
    for name in (run_id, f"best_{run_id}"):
        candidate = runs_root / name
        if candidate.is_dir():
            return candidate.resolve()

    matches = []
    archive_root = project_root / "scenes" / "_final"
    for run_json in archive_root.glob(f"**/{run_id}/run.json"):
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
        if (metadata.get("scene_id"), metadata.get("run_id")) == (
            normalized_scene_id, run_id,
        ):
            matches.append(run_json.parent.resolve())
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"archived run is ambiguous: {run_id}")
    raise FileNotFoundError(f"run not found: {normalized_scene_id}/{run_id}")


def atomic_write_json(target: Path, value) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class AttemptContext:
    attempt_id: str
    temp_root: Path
    final_root: Path
    upstream_best: Dict[str, str]
    stage: str


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    scene_id: str
    run_id: str
    run_root: Path

    @classmethod
    def create(
        cls,
        project_root,
        scene_id: str,
        label: Optional[str] = None,
        now: Optional[datetime] = None,
        articulation_method: str = DEFAULT_ARTICULATION_METHOD,
        assembly_asset_source: str = "raw",
        input_image=None,
        drawer_geometry_mode: str = DEFAULT_DRAWER_GEOMETRY_MODE,
    ):
        project_root = Path(project_root).resolve()
        validate_articulation_method(articulation_method)
        validate_assembly_asset_source(assembly_asset_source)
        validate_drawer_geometry_mode(drawer_geometry_mode)
        if input_image is None:
            raise ValueError("input_image is required")
        source_image = Path(input_image).resolve(strict=True)
        if not source_image.is_file():
            raise ValueError(f"input image is not a regular file: {source_image}")
        run_id = build_run_id(
            now or datetime.now().astimezone(), "trellis2", label, articulation_method
        )
        normalized_scene_id = slug(scene_id)
        run_root = project_root / "scenes" / normalized_scene_id / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        context = cls(project_root, normalized_scene_id, run_id, run_root)
        for relative in ("inputs", "stages", "final", "logs/stages"):
            context.run_path(relative).mkdir(parents=True, exist_ok=True)
        metadata = {
            "scene_id": context.scene_id,
            "run_id": context.run_id,
            "status": "created",
            "method": "trellis2",
            "articulation_method": articulation_method,
            "assembly_asset_source": assembly_asset_source,
            "drawer_geometry_mode": drawer_geometry_mode,
            "model_runtime": runtime_snapshot(),
        }
        copied = context.run_path("inputs", f"source_image{source_image.suffix.lower()}")
        shutil.copyfile(source_image, copied)
        metadata["input_image"] = {
            "path": str(copied.relative_to(context.run_root)),
            "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        }
        atomic_write_json(context.run_json, metadata)
        return context

    @classmethod
    def resume(
        cls,
        project_root,
        scene_id: str,
        run_id: str,
        articulation_method: Optional[str] = None,
        assembly_asset_source: Optional[str] = None,
        drawer_geometry_mode: Optional[str] = None,
    ):
        project_root = Path(project_root).resolve()
        if articulation_method is not None:
            validate_articulation_method(articulation_method)
        if assembly_asset_source is not None:
            validate_assembly_asset_source(assembly_asset_source)
        if drawer_geometry_mode is not None:
            validate_drawer_geometry_mode(drawer_geometry_mode)
        normalized_scene_id = slug(scene_id)
        run_root = resolve_run_root(project_root, normalized_scene_id, run_id)
        metadata = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        if (metadata.get("scene_id"), metadata.get("run_id")) != (
            normalized_scene_id, run_id
        ):
            raise ValueError(f"invalid run metadata: {run_id}")
        frozen_method = articulation_method_from_metadata(metadata)
        if articulation_method is not None and frozen_method != articulation_method:
            raise ValueError(
                "articulation method conflicts with frozen run configuration"
            )
        frozen_assembly_source = metadata.get("assembly_asset_source", "raw")
        validate_assembly_asset_source(frozen_assembly_source)
        if (
            assembly_asset_source is not None
            and frozen_assembly_source != assembly_asset_source
        ):
            raise ValueError(
                "assembly asset source conflicts with frozen run configuration"
            )
        frozen_drawer_mode = validate_drawer_geometry_mode(metadata.get(
            "drawer_geometry_mode", DEFAULT_DRAWER_GEOMETRY_MODE,
        ))
        if drawer_geometry_mode is not None and frozen_drawer_mode != drawer_geometry_mode:
            raise ValueError(
                "drawer geometry mode conflicts with frozen run configuration"
            )
        return cls(project_root, normalized_scene_id, run_id, run_root)

    @property
    def run_json(self):
        return self.run_root / "run.json"

    def project_path(self, *parts):
        return self.project_root.joinpath(*parts)

    def run_path(self, *parts):
        return self.run_root.joinpath(*parts)

    def final_path(self, *parts):
        return self.run_root.joinpath("final", *parts)

    @property
    def articulation_method(self):
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        return articulation_method_from_metadata(metadata)

    @property
    def drawer_geometry_mode(self):
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        return validate_drawer_geometry_mode(metadata.get(
            "drawer_geometry_mode", DEFAULT_DRAWER_GEOMETRY_MODE,
        ))

    @property
    def source_image(self):
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        relative = Path(metadata["input_image"]["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input image path must be run-relative")
        source = (self.run_root / relative).resolve(strict=True)
        run_root = self.run_root.resolve()
        if run_root not in source.parents or not source.is_file():
            raise ValueError("input image must remain inside the run")
        return source

    @contextmanager
    def lock(self):
        lock_file = self.run_path(".lock").open("a+")
        try:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"run is already locked: {self.run_id}") from None
            yield
        finally:
            lock_file.close()

    def initialize_stages(self, registry):
        snapshot = registry.snapshot()
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        snapshot_keys = snapshot.keys()
        if any(key in metadata for key in snapshot_keys):
            existing = {key: metadata.get(key) for key in snapshot_keys}
            if existing != snapshot:
                raise ValueError("stages already initialized with a different snapshot")
        else:
            metadata.update(snapshot)
            atomic_write_json(self.run_json, metadata)

    def _stage_root(self, stage: str) -> Path:
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        try:
            directory = metadata["stage_directories"][stage]
        except KeyError:
            raise ValueError(f"unknown or uninitialized stage: {stage}") from None
        return self.run_path("stages", directory)

    def begin_attempt(self, stage: str, now=None, upstream_overrides=None) -> AttemptContext:
        metadata = json.loads(self.run_json.read_text(encoding="utf-8"))
        stage_root = self._stage_root(stage)
        upstream_overrides = upstream_overrides or {}
        upstream_best = {
            dependency: (
                upstream_overrides[dependency]
                if dependency in upstream_overrides
                else self.stage_best(dependency).name
            )
            for dependency in metadata["stage_dependencies"][stage]
        }
        base_id = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
        attempts_root = stage_root / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            attempt_id = base_id if suffix == 0 else f"{base_id}_{suffix:02d}"
            final_root = attempts_root / attempt_id
            if final_root.exists():
                suffix += 1
                continue
            temp_root = attempts_root / f".{attempt_id}.tmp"
            try:
                temp_root.mkdir()
            except FileExistsError:
                suffix += 1
                continue
            if final_root.exists():
                temp_root.rmdir()
                suffix += 1
                continue
            break
        return AttemptContext(
            attempt_id,
            temp_root,
            final_root,
            upstream_best,
            stage,
        )

    def complete_attempt(self, attempt: AttemptContext, artifacts: dict) -> None:
        metadata = {
            "attempt_id": attempt.attempt_id,
            "stage": attempt.stage,
            "status": "completed",
            "upstream_best": attempt.upstream_best,
            "artifacts": artifacts,
        }
        self._record_model_runtime(attempt, metadata)
        self._record_articulation_method(attempt, metadata)
        self._finalize_attempt(attempt, metadata)
        stage_root = self._stage_root(attempt.stage)
        best = stage_root / "best"
        temporary = stage_root / f".best.{attempt.attempt_id}.tmp"
        try:
            temporary.symlink_to(Path("attempts") / attempt.attempt_id)
            os.replace(temporary, best)
        finally:
            if temporary.is_symlink():
                temporary.unlink()

    def fail_attempt(self, attempt: AttemptContext, exc: BaseException) -> None:
        metadata = {
            "attempt_id": attempt.attempt_id,
            "stage": attempt.stage,
            "status": "failed",
            "upstream_best": attempt.upstream_best,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )),
            },
        }
        self._record_model_runtime(attempt, metadata)
        self._record_articulation_method(attempt, metadata)
        self._finalize_attempt(attempt, metadata)

    def _record_articulation_method(self, attempt: AttemptContext, metadata: dict) -> None:
        if attempt.stage == "gram":
            metadata["method"] = self.articulation_method

    def _record_model_runtime(self, attempt: AttemptContext, metadata: dict) -> None:
        grouped = {}
        marker_root = attempt.temp_root / "logs/model_runtime"
        for marker in sorted(marker_root.glob("*_model_runtime.json")):
            try:
                runtime = json.loads(marker.read_text(encoding="utf-8"))
                domain = runtime.pop("domain")
            except (KeyError, OSError, TypeError, ValueError):
                continue
            values = grouped.setdefault(domain, [])
            if runtime not in values:
                values.append(runtime)
        if grouped:
            metadata["model_runtime"] = {
                domain: values[0] if len(values) == 1 else values
                for domain, values in grouped.items()
            }

    @staticmethod
    def _finalize_attempt(attempt: AttemptContext, metadata: dict) -> None:
        atomic_write_json(attempt.temp_root / "attempt.json", metadata)
        attempt.temp_root.rename(attempt.final_root)

    def stage_best(self, stage: str) -> Path:
        return (self._stage_root(stage) / "best").resolve(strict=True)

    def mark_stage_best(self, stage: str, attempt_id: str) -> None:
        stage_root = self._stage_root(stage)
        attempt_root = stage_root / "attempts" / attempt_id
        metadata = json.loads(
            (attempt_root / "attempt.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "completed":
            raise ValueError("only successful attempts can be marked best")

        best = stage_root / "best"
        temporary = stage_root / f".best.{attempt_id}.tmp"
        try:
            temporary.symlink_to(Path("attempts") / attempt_id)
            os.replace(temporary, best)
        finally:
            if temporary.is_symlink():
                temporary.unlink()


def _run_timestamp(run_id: str) -> datetime:
    return datetime.strptime(_without_best(run_id)[:15], "%Y%m%d_%H%M%S")


def _without_best(value: str) -> str:
    return value[5:] if value.startswith("best_") else value


def _load_valid_run(run_root: Path, scene_id: str) -> dict:
    metadata = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    run_id = _without_best(run_root.name)
    if metadata.get("scene_id") != scene_id or metadata.get("run_id") != run_id:
        raise ValueError(f"invalid run metadata: {run_id}")
    _run_timestamp(run_id)
    return metadata


def _artifact_paths(metadata: dict):
    artifacts = metadata.get("final_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("completed run has no final artifacts")
    for name, value in artifacts.items():
        path = value.get("path") if isinstance(value, dict) else value
        if not isinstance(path, str):
            raise ValueError(f"invalid final artifact: {name}")
        yield name, path


def _validate_completed_run(run_root: Path, scene_id: str) -> dict:
    metadata = _load_valid_run(run_root, scene_id)
    if metadata.get("status") != "completed":
        raise ValueError("only completed runs can be marked best")
    for _, relative in _artifact_paths(metadata):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (run_root / path).is_file():
            raise ValueError(f"missing final artifact: {relative}")
    return metadata


@contextmanager
def _scene_lock(scene_root: Path):
    scene_root.mkdir(parents=True, exist_ok=True)
    lock_file = (scene_root / ".lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        lock_file.close()


def _write_scene_indexes(scene_root: Path, scene: dict, index: str) -> None:
    previous_scene = (
        (scene_root / "scene.json").read_bytes()
        if (scene_root / "scene.json").exists() else None
    )
    temporary = []
    try:
        for name, content in (
            ("scene.json", json.dumps(scene, indent=2, sort_keys=True) + "\n"),
            ("INDEX.md", index),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=scene_root, delete=False
            ) as output:
                output.write(content)
                temporary.append((Path(output.name), scene_root / name))
        os.replace(*temporary[0])
        try:
            os.replace(*temporary[1])
        except Exception:
            if previous_scene is None:
                temporary[0][1].unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=scene_root, delete=False
                ) as output:
                    output.write(previous_scene)
                    restore = Path(output.name)
                os.replace(restore, temporary[0][1])
            raise
    finally:
        for source, _ in temporary:
            if source.exists():
                source.unlink()


def update_scene_index(project_root, scene_id: str) -> None:
    project_root = Path(project_root).resolve()
    scene_id = slug(scene_id)
    scene_root = project_root / "scenes" / scene_id
    runs_root = scene_root / "runs"
    if not runs_root.is_dir():
        return
    runs = []
    for run_root in runs_root.iterdir() if runs_root.is_dir() else ():
        if not run_root.is_dir() or not (run_root / "run.json").is_file():
            continue
        metadata = _load_valid_run(run_root, scene_id)
        runs.append({
            "run_id": metadata["run_id"],
            "directory": run_root.name,
            "method": metadata.get("method"),
            "status": metadata.get("status"),
            "started_at": metadata.get("started_at"),
            "duration_seconds": metadata.get("duration_seconds"),
            "label": metadata.get("label"),
            "final_artifacts": metadata.get("final_artifacts", {}),
        })
    runs.sort(key=lambda run: _run_timestamp(run["run_id"]), reverse=True)
    best_runs = [run for run in runs if run["directory"].startswith("best_")]
    if len(best_runs) > 1:
        raise ValueError("multiple best run directories")
    best = best_runs[0] if best_runs else None
    if best:
        runs.remove(best)
        runs.insert(0, best)
    scene = {
        "scene_id": scene_id,
        "best_run_id": best["run_id"] if best else None,
        "runs": runs,
    }
    lines = [
        f"# Scene: {scene_id}",
        "",
        "| Run ID | Method | Status | Start time | Duration (s) | Label | Final artifacts |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for run in runs:
        links = []
        for name, relative in _artifact_paths(run) if run["final_artifacts"] else ():
            links.append(f"[{name}](runs/{run['directory']}/{relative})")
        lines.append("| " + " | ".join(str(value if value is not None else "") for value in (
            run["run_id"], run["method"], run["status"], run["started_at"],
            run["duration_seconds"], run["label"], ", ".join(links),
        )) + " |")
    _write_scene_indexes(scene_root, scene, "\n".join(lines) + "\n")


def mark_run_best(project_root, scene_id: str, run_id: str) -> str:
    project_root = Path(project_root).resolve()
    scene_id = slug(scene_id)
    run_id = _without_best(run_id)
    scene_root = project_root / "scenes" / scene_id
    runs_root = scene_root / "runs"
    with _scene_lock(scene_root):
        best_runs = [
            path for path in runs_root.iterdir()
            if path.is_dir() and path.name.startswith("best_")
        ]
        if len(best_runs) > 1:
            raise ValueError("multiple best run directories")
        selected = runs_root / run_id
        already_best = runs_root / f"best_{run_id}"
        if already_best.is_dir():
            _validate_completed_run(already_best, scene_id)
            update_scene_index(project_root, scene_id)
            return already_best.name
        _validate_completed_run(selected, scene_id)

        old_best = best_runs[0] if best_runs else None
        old_normal = runs_root / _without_best(old_best.name) if old_best else None
        new_best = runs_root / f"best_{run_id}"
        if old_normal and old_normal.exists():
            raise FileExistsError(old_normal)
        try:
            if old_best:
                old_best.rename(old_normal)
            selected.rename(new_best)
            update_scene_index(project_root, scene_id)
        except Exception:
            if new_best.exists() and not selected.exists():
                new_best.rename(selected)
            if old_best and old_normal.exists() and not old_best.exists():
                old_normal.rename(old_best)
            raise
        return new_best.name
