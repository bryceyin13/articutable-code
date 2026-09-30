from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pipeline.common import log_step, read_json, write_json
try:
    from pipeline.trellis2_runtime import MODEL_ROOT, TRELLIS_ROOT, load_pipeline, release_cuda_memory
except ModuleNotFoundError as exc:
    if exc.name != "trellis2_runtime":
        raise
    TRELLIS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "trellis2"
    MODEL_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "trellis2-model"
    release_cuda_memory = lambda: None

    def load_pipeline():
        raise RuntimeError("optional TRELLIS.2 runtime helper is not installed")


TARGET_FACES = 50_000
TEXTURE_SIZE = 4096
PREVIEW_FRAMES = 1
PRIMARY_PIPELINE_TYPE = "1024_cascade"
OOM_FALLBACK_PIPELINE_TYPE = "512"
REQUIRED_OUTPUTS = ("model.glb", "preview.png", "metadata.json")


def _input_relative(item):
    views = item["views"]
    return views.get("primary") or views.get("right_45") or views["front"]


def _input_hash(source):
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _is_complete(object_root, input_hash):
    required = [object_root / name for name in REQUIRED_OUTPUTS]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        metadata = read_json(object_root / "metadata.json")
    except (OSError, ValueError):
        return False
    timings = metadata.get("timings_seconds")
    positive_integer_fields = ("actual_faces", "actual_vertices")
    return all((
        metadata.get("backend") == "trellis2",
        metadata.get("input_sha256") == input_hash,
        metadata.get("model") == str(MODEL_ROOT),
        metadata.get("target_faces") == TARGET_FACES,
        metadata.get("texture_size") == TEXTURE_SIZE,
        metadata.get("preview_frames") == PREVIEW_FRAMES,
        all(
            isinstance(metadata.get(field), int)
            and not isinstance(metadata.get(field), bool)
            and metadata[field] > 0
            for field in positive_integer_fields
        ),
        isinstance(timings, dict),
        isinstance(timings, dict) and all(
            isinstance(timings.get(name), (int, float))
            and not isinstance(timings.get(name), bool)
            and timings[name] > 0
            for name in ("inference", "preview", "export")
        ),
        isinstance(metadata.get("elapsed_seconds"), (int, float))
        and not isinstance(metadata.get("elapsed_seconds"), bool)
        and metadata["elapsed_seconds"] > 0,
    ))


def carry_forward(context, attempt, condition_manifest, object_ids):
    """Copy strictly valid completed objects from failed attempts into a new attempt."""
    manifest_path = Path(condition_manifest)
    manifest = read_json(manifest_path)
    items = {item["object_id"]: item for item in manifest["items"]}
    attempts_root = attempt.final_root.parent
    candidates = []
    for root in attempts_root.iterdir():
        if root != attempt.temp_root and root.name.startswith(".") and root.name.endswith(".tmp"):
            candidates.append(root)
            continue
        metadata_path = root / "attempt.json"
        if root == attempt.temp_root or not metadata_path.is_file():
            continue
        try:
            metadata = read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if (metadata.get("status") == "failed"
                and metadata.get("stage") == "image_to_3d"
                and metadata.get("upstream_best") == attempt.upstream_best):
            candidates.append(root)

    for object_id in object_ids:
        input_relative = _input_relative(items[object_id])
        source = manifest_path.parent.parent / input_relative
        for candidate in reversed(sorted(candidates)):
            object_root = candidate / object_id
            if not _is_complete(object_root, _input_hash(source)):
                continue
            candidate_root = candidate.resolve()
            files = [object_root / name for name in REQUIRED_OUTPUTS]
            if any(path.is_symlink() or not path.is_file()
                   or candidate_root not in path.resolve().parents for path in files):
                continue
            destination = Path(attempt.temp_root) / object_id
            destination.mkdir(parents=True, exist_ok=True)
            for path in files:
                shutil.copy2(path, destination / path.name)
            break


def _result(object_id, input_relative):
    return {
        "object_id": object_id,
        "input_image": input_relative,
        "mesh": f"{object_id}/model.glb",
        "preview": f"{object_id}/preview.png",
        "metadata": f"{object_id}/metadata.json",
    }


def _geometry_counts(export):
    geometry = getattr(export, "geometry", None)
    meshes = geometry.values() if geometry is not None else (export,)
    meshes = list(meshes)
    return (
        sum(len(mesh.faces) for mesh in meshes),
        sum(len(mesh.vertices) for mesh in meshes),
    )


def _is_oom(error):
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda error: 700" in message
        or "illegal memory access" in message
    )


def _is_subtriangle_overflow(error):
    return "subtriangle count overflow" in str(error).lower()


def _render_glb_preview(model, preview, runner=subprocess.run):
    from pipeline.image_to_3d import _preview_blender

    runner([
        str(_preview_blender()), "--background", "--python-exit-code", "1",
        "--python", str(Path(__file__).with_name("model_previews.py")),
        "--", "--item", str(model), str(preview),
    ], check=True)
    if not preview.is_file() or preview.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"GLB preview is not a PNG: {preview}")


def _generate(
    source, object_root, pipeline, envmap, pipeline_type,
    trellis2_seed=None,
):
    import imageio
    import o_voxel
    from PIL import Image
    from trellis2.utils import render_utils

    started = time.monotonic()
    if trellis2_seed is None:
        trellis2_seed = random.randrange(2**31)
    if (
        not isinstance(trellis2_seed, int)
        or isinstance(trellis2_seed, bool)
        or not 0 <= trellis2_seed < 2**31
    ):
        raise ValueError(f"invalid TRELLIS.2 seed: {trellis2_seed!r}")
    with Image.open(source) as image:
        mesh = pipeline.run(
            image, seed=trellis2_seed, pipeline_type=pipeline_type,
        )[0]
    inference_seconds = time.monotonic() - started
    vertices = len(mesh.vertices)
    faces = len(mesh.faces)
    release_cuda_memory()

    preview_path = object_root / "preview.png"
    preview_started = time.monotonic()
    glb_preview = pipeline_type == OOM_FALLBACK_PIPELINE_TYPE
    if not glb_preview:
        try:
            frames = render_utils.make_pbr_vis_frames(
                render_utils.render_video(mesh, envmap=envmap, resolution=512, num_frames=PREVIEW_FRAMES)
            )
            imageio.imwrite(preview_path, frames[0])
            preview_seconds = time.monotonic() - preview_started
            del frames
        except RuntimeError as exc:
            if not _is_subtriangle_overflow(exc):
                raise
            glb_preview = True
    release_cuda_memory()

    export_started = time.monotonic()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=TARGET_FACES,
        texture_size=TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    actual_faces, actual_vertices = _geometry_counts(glb)
    del mesh
    release_cuda_memory()
    temporary = object_root / ".model.tmp.glb"
    try:
        glb.export(temporary, extension_webp=True)
        os.replace(temporary, object_root / "model.glb")
    finally:
        if temporary.exists():
            temporary.unlink()
    del glb
    release_cuda_memory()
    export_seconds = time.monotonic() - export_started
    if glb_preview:
        preview_started = time.monotonic()
        _render_glb_preview(object_root / "model.glb", preview_path)
        preview_seconds = time.monotonic() - preview_started
    return {
        "trellis2_seed": trellis2_seed,
        "source_faces": faces,
        "source_vertices": vertices,
        "actual_faces": actual_faces,
        "actual_vertices": actual_vertices,
        "timings_seconds": {
            "inference": inference_seconds,
            "preview": preview_seconds,
            "export": export_seconds,
        },
    }


def run(
    condition_manifest: Path, output_root: Path, object_ids: list[str],
    pipeline_resources=None, pipeline_type=None,
    deferred_oom=None,
) -> list[dict]:
    if pipeline_type is None:
        pipeline_type = PRIMARY_PIPELINE_TYPE
    condition_manifest = Path(condition_manifest)
    manifest = read_json(condition_manifest)
    items = {item["object_id"]: item for item in manifest["items"]}
    results = {}
    pending = []
    for object_id in object_ids:
        item = items[object_id]
        input_relative = _input_relative(item)
        source = condition_manifest.parent.parent / input_relative
        object_root = Path(output_root) / object_id
        object_root.mkdir(parents=True, exist_ok=True)
        if _is_complete(object_root, _input_hash(source)):
            results[object_id] = _result(object_id, input_relative)
        else:
            pending.append((object_id, input_relative, source, object_root))
    if not pending:
        return [results[object_id] for object_id in object_ids]

    pipeline, envmap = pipeline_resources or load_pipeline()
    for object_id, input_relative, source, object_root in pending:
        (object_root / "preview.mp4").unlink(missing_ok=True)
        started = time.monotonic()
        try:
            seed = items[object_id].get("trellis2_seed")
            generated = (
                _generate(
                    source, object_root, pipeline, envmap, pipeline_type,
                    trellis2_seed=seed,
                )
                if seed is not None else
                _generate(source, object_root, pipeline, envmap, pipeline_type)
            )
            release_cuda_memory()
        except BaseException as exc:
            if deferred_oom is None or not _is_oom(exc):
                raise
            deferred_oom[object_id] = traceback.format_exc()
            log_step(
                "trellis2",
                f"{object_id} OOM at {pipeline_type}; deferred to "
                f"{OOM_FALLBACK_PIPELINE_TYPE}",
            )
            continue
        metadata = {
            **generated,
            "backend": "trellis2",
            "pipeline_type": pipeline_type,
            "oom_fallback": pipeline_type == OOM_FALLBACK_PIPELINE_TYPE,
            "elapsed_seconds": time.monotonic() - started,
            "input_sha256": _input_hash(source),
            "model": str(MODEL_ROOT),
            "target_faces": TARGET_FACES,
            "texture_size": TEXTURE_SIZE,
            "preview_frames": PREVIEW_FRAMES,
        }
        write_json(object_root / "metadata.json", metadata)
        required = [object_root / name for name in REQUIRED_OUTPUTS]
        missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"incomplete TRELLIS.2 output for {object_id}: {', '.join(missing)}")
        results[object_id] = _result(object_id, input_relative)
    return [
        results[object_id] for object_id in object_ids
        if object_id in results
    ]


def _subprocess_environment(gpu_id):
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    existing_pythonpath = environment.get("PYTHONPATH")
    roots = [str(TRELLIS_ROOT)]
    if existing_pythonpath:
        roots.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def _python_and_gpus():
    python = Path(os.environ.get(
        "TRELLIS2_PYTHON", Path.home() / ".conda/envs/trellis2/bin/python"
    )).expanduser()
    if not python.is_file():
        raise FileNotFoundError(f"TRELLIS.2 Python not found: {python}; set TRELLIS2_PYTHON")
    gpu_ids = [
        item.strip()
        for item in os.environ.get("TRELLIS2_GPUS", "4,5,6,7").split(",")
        if item.strip()
    ]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("TRELLIS2_GPUS must contain unique GPU IDs")
    return python, gpu_ids or [None]


class PersistentWorkerPool:
    """One resident TRELLIS process per GPU, fed by a shared work queue."""

    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.python, self.gpu_ids = _python_and_gpus()
        self.tasks = queue.Queue()
        self.threads = []
        self.closed = False
        self.deferred_oom = []
        self.lock = threading.Lock()

    def _start(self):
        if self.threads:
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        log_step("trellis2", "persistent GPU workers=" + ",".join(
            "default" if gpu is None else str(gpu) for gpu in self.gpu_ids
        ))
        for gpu_id in self.gpu_ids:
            thread = threading.Thread(
                target=self._worker, args=(gpu_id,), daemon=True,
                name=f"trellis2-gpu-{gpu_id}",
            )
            thread.start()
            self.threads.append(thread)

    def submit(self, manifest, object_id, output_root=None):
        if self.closed:
            raise RuntimeError("TRELLIS.2 worker pool is closed")
        self._start()
        future = Future()
        pipeline_type = PRIMARY_PIPELINE_TYPE
        self.tasks.put((
            Path(manifest), object_id, Path(output_root or self.output_root),
            future, pipeline_type,
            pipeline_type != OOM_FALLBACK_PIPELINE_TYPE,
        ))
        return future

    def run(self, manifest, output_root, object_ids):
        futures = [
            self.submit(manifest, object_id, output_root)
            for object_id in object_ids
        ]
        for future in futures:
            future.result()
        self.retry_deferred_oom()
        items = {
            item["object_id"]: item
            for item in read_json(Path(manifest))["items"]
        }
        return [
            _result(object_id, _input_relative(items[object_id]))
            for object_id in object_ids
        ]

    def retry_deferred_oom(self):
        with self.lock:
            deferred = self.deferred_oom
            self.deferred_oom = []
        if not deferred:
            return
        log_step(
            "trellis2",
            "1024 phase complete; retrying OOM objects at 512: "
            + ",".join(object_id for _, object_id, _ in deferred),
        )
        futures = []
        for manifest, object_id, output_root in deferred:
            future = Future()
            self.tasks.put((
                manifest, object_id, output_root, future,
                OOM_FALLBACK_PIPELINE_TYPE, False,
            ))
            futures.append(future)
        for future in futures:
            future.result()

    def _worker(self, gpu_id):
        process = None
        while True:
            task = self.tasks.get()
            if task is None:
                if process is not None:
                    try:
                        process.stdin.write(json.dumps({"stop": True}) + "\n")
                        process.stdin.flush()
                        process.wait(timeout=30)
                    except (BrokenPipeError, subprocess.TimeoutExpired):
                        process.kill()
                        process.wait()
                self.tasks.task_done()
                return
            manifest, object_id, output_root, future, pipeline_type, defer_oom = task
            response = output_root / ".trellis_responses" / f"{object_id}.json"
            response.parent.mkdir(parents=True, exist_ok=True)
            response.unlink(missing_ok=True)
            try:
                if process is None or process.poll() is not None:
                    process = subprocess.Popen(
                        [
                            str(self.python), str(Path(__file__).resolve()),
                            "--persistent-worker", str(self.output_root.resolve()),
                        ],
                        stdin=subprocess.PIPE, text=True,
                        env=_subprocess_environment(gpu_id),
                    )
                process.stdin.write(json.dumps({
                    "manifest": str(manifest.resolve()),
                    "object_id": object_id,
                    "response": str(response.resolve()),
                    "output_root": str(output_root.resolve()),
                    "pipeline_type": pipeline_type,
                    "defer_oom": defer_oom,
                }) + "\n")
                process.stdin.flush()
                while not response.is_file():
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            f"TRELLIS.2 GPU {gpu_id} worker exited with code {return_code}"
                        )
                    time.sleep(0.2)
                payload = read_json(response)
                if "oom" in payload:
                    with self.lock:
                        self.deferred_oom.append((manifest, object_id, output_root))
                    log_step(
                        "trellis2",
                        f"{object_id} OOM at {pipeline_type}; deferred to "
                        f"{OOM_FALLBACK_PIPELINE_TYPE}",
                    )
                    future.set_result({"object_id": object_id, "deferred_oom": True})
                    process.kill()
                    process.wait()
                    process = None
                    continue
                if "error" in payload:
                    if defer_oom and _is_oom(RuntimeError(payload["error"])):
                        with self.lock:
                            self.deferred_oom.append((manifest, object_id, output_root))
                        log_step(
                            "trellis2",
                            f"{object_id} recoverable CUDA failure at "
                            f"{pipeline_type}; deferred to "
                            f"{OOM_FALLBACK_PIPELINE_TYPE}",
                        )
                        future.set_result({
                            "object_id": object_id,
                            "deferred_oom": True,
                        })
                        process.kill()
                        process.wait()
                        process = None
                        continue
                    raise RuntimeError(payload["error"])
                future.set_result(payload["result"])
            except BaseException as exc:
                future.set_exception(exc)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                process = None
            finally:
                self.tasks.task_done()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for _ in self.threads:
            self.tasks.put(None)
        for thread in self.threads:
            thread.join()


def persistent_worker(output_root):
    pipeline_resources = load_pipeline()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        response = Path(request["response"])
        try:
            deferred_oom = {} if request.get("defer_oom") else None
            results = run(
                Path(request["manifest"]),
                Path(request.get("output_root", output_root)),
                [request["object_id"]], pipeline_resources,
                request.get(
                    "pipeline_type", PRIMARY_PIPELINE_TYPE,
                ),
                deferred_oom,
            )
            payload = (
                {"oom": deferred_oom[request["object_id"]]}
                if deferred_oom else
                {"result": results[0]}
            )
        except BaseException:
            payload = {"error": traceback.format_exc()}
        temporary = response.with_name(response.name + ".tmp")
        write_json(temporary, payload)
        os.replace(temporary, response)
    del pipeline_resources
    release_cuda_memory()


def run_subprocess(
    condition_manifest: Path, output_root: Path, object_ids: list[str],
) -> list[dict]:
    if not object_ids:
        return []
    python, gpu_ids = _python_and_gpus()
    assignments = [(None, object_ids)]
    if gpu_ids:
        assignments = [
            (gpu_id, object_ids[index::len(gpu_ids)])
            for index, gpu_id in enumerate(gpu_ids)
            if object_ids[index::len(gpu_ids)]
        ]
        log_step(
            "trellis2",
            "GPU workers=" + ", ".join(
                f"{gpu_id}:[{','.join(ids)}]" for gpu_id, ids in assignments
            ),
        )

    def execute(assignment, pipeline_type, oom_root=None):
        gpu_id, assigned_ids = assignment
        command = [
            str(python), str(Path(__file__).resolve()),
            "--pipeline-type", pipeline_type,
            str(Path(condition_manifest).resolve()), str(Path(output_root).resolve()),
            *assigned_ids,
        ]
        oom_record = None
        if oom_root is not None:
            oom_record = oom_root / f"{gpu_id or 'default'}.json"
            command[2:2] = ["--deferred-oom-record", str(oom_record)]
        subprocess.run(
            command, check=True,
            env=_subprocess_environment(gpu_id),
        )
        return read_json(oom_record) if oom_record and oom_record.is_file() else {}

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    initial_pipeline_type = PRIMARY_PIPELINE_TYPE
    with tempfile.TemporaryDirectory(
        prefix=".trellis2-oom-", dir=output_root,
    ) as temporary:
        oom_root = Path(temporary)
        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            primary = list(executor.map(
                lambda assignment: execute(
                    assignment, initial_pipeline_type,
                    oom_root
                    if initial_pipeline_type != OOM_FALLBACK_PIPELINE_TYPE
                    else None,
                ),
                assignments,
            ))
        deferred_ids = [
            object_id for object_id in object_ids
            if any(object_id in deferred for deferred in primary)
        ]
        if deferred_ids:
            log_step(
                "trellis2",
                "1024 phase complete; retrying OOM objects at 512: "
                + ",".join(deferred_ids),
            )
            fallback_assignments = [
                (gpu_id, deferred_ids[index::len(gpu_ids)])
                for index, gpu_id in enumerate(gpu_ids)
                if deferred_ids[index::len(gpu_ids)]
            ]
            with ThreadPoolExecutor(
                max_workers=len(fallback_assignments),
            ) as executor:
                list(executor.map(
                    lambda assignment: execute(
                        assignment, OOM_FALLBACK_PIPELINE_TYPE,
                    ),
                    fallback_assignments,
                ))
    manifest = read_json(Path(condition_manifest))
    items = {item["object_id"]: item for item in manifest["items"]}
    return [
        _result(object_id, _input_relative(items[object_id]))
        for object_id in object_ids
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-worker", type=Path)
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default=None,
    )
    parser.add_argument("--deferred-oom-record", type=Path)
    parser.add_argument("condition_manifest", type=Path, nargs="?")
    parser.add_argument("output_root", type=Path, nargs="?")
    parser.add_argument("object_ids", nargs="*")
    args = parser.parse_args(argv)
    if args.persistent_worker:
        persistent_worker(args.persistent_worker)
        return
    if args.condition_manifest is None or args.output_root is None or not args.object_ids:
        parser.error("condition_manifest, output_root and object_ids are required")
    deferred_oom = {} if args.deferred_oom_record else None
    run(
        args.condition_manifest, args.output_root, args.object_ids,
        pipeline_type=args.pipeline_type or PRIMARY_PIPELINE_TYPE,
        deferred_oom=deferred_oom,
    )
    if args.deferred_oom_record:
        write_json(args.deferred_oom_record, deferred_oom)


if __name__ == "__main__":
    main()
