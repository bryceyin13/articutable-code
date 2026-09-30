#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from pipeline import blueprint as generate_blueprint
from pipeline import condition_images as generate_condition_images
from pipeline import reference_image as generate_reference_image
from pipeline import topview_image as generate_topview_image
from pipeline import image_to_3d as run_image_to_3d
from gram import run as run_articulation
from pipeline import scene_assembly as assemble_scene_blender
from pipeline import mesh_orientation as prepare_mesh_orientation
from pipeline import articulated_scene as replace_articulated_scene
from pipeline import segmentation as run_grounded_sam
from pipeline import simulation as run_joint_demo_isaac
from pipeline import select_items
from pipeline.stage_io import STAGE_OUTPUTS, completed_files
from pipeline.common import ensure_dirs, env_positive_int, read_json
from pipeline.context import (
    ARTICULATION_METHODS,
    DEFAULT_ARTICULATION_METHOD,
    DEFAULT_DRAWER_GEOMETRY_MODE,
    RunContext,
    mark_run_best,
    resolve_run_root,
    slug,
    update_scene_index,
)
from pipeline.stages import Stage, StageRegistry


def run_image_to_3d_stage(
    context, attempt, generate_articulated_objects=True,
    only_generate_articulated_objects=False,
    image_to_3d_object_ids=None,
    trellis2_runner=None,
    primary_input_object_ids=None,
    preserve_previous_outputs=False,
):
    run_image_to_3d.run(
        context, attempt,
        generate_articulated_objects=generate_articulated_objects,
        only_generate_articulated_objects=only_generate_articulated_objects,
        regenerate_object_ids=image_to_3d_object_ids,
        trellis2_runner=trellis2_runner,
        primary_input_object_ids=primary_input_object_ids,
        preserve_previous_outputs=preserve_previous_outputs,
    )
    return completed_files(attempt, STAGE_OUTPUTS["image_to_3d"])


def run_gram_stage(
    context, attempt, gram_object_ids=None,
    gram_from_stage=None, gram_to_stage=None,
):
    return run_articulation.run(
        context, attempt, context.articulation_method,
        regenerate_object_ids=gram_object_ids,
        from_stage=gram_from_stage,
        to_stage=gram_to_stage,
    )


def build_stage_registry():
    stages = [
    Stage("reference_image", generate_reference_image.run),
    Stage("topview_image", generate_topview_image.run, ("reference_image",)),
    Stage(
        "segment_anonymous_instances",
        run_grounded_sam.run_anonymous_views,
        ("reference_image", "topview_image"),
    ),
    Stage(
        "select_items",
        select_items.run,
        (
            "reference_image", "topview_image", "segment_anonymous_instances",
        ),
    ),
    Stage(
        "condition_images",
        generate_condition_images.run,
        (
            "reference_image", "topview_image", "segment_anonymous_instances",
            "select_items",
        ),
    ),
    Stage(
        "blueprint",
        generate_blueprint.run,
        ("reference_image", "topview_image", "condition_images"),
    ),
    Stage(
        "segment_instances",
        run_grounded_sam.run_remapped_views,
        (
            "reference_image", "topview_image", "segment_anonymous_instances",
            "condition_images", "blueprint",
        ),
    ),
    Stage(
        "image_to_3d",
        run_image_to_3d_stage,
        ("condition_images",),
    ),
    Stage("gram", run_gram_stage, ("image_to_3d", "blueprint", "condition_images")),
    Stage(
        "mesh_orientation", prepare_mesh_orientation.run,
        (
            "reference_image", "topview_image", "blueprint",
            "segment_instances", "image_to_3d", "gram",
        ),
    ),
    Stage(
        "blender_scene", assemble_scene_blender.run,
        (
            "reference_image", "topview_image", "blueprint",
            "segment_instances",
            "image_to_3d", "gram", "mesh_orientation",
        ),
    ),
    Stage(
        "replace_articulated", replace_articulated_scene.run,
        ("blender_scene", "gram"),
    ),
    Stage("simulation", run_joint_demo_isaac.run, ("replace_articulated",)),
    ]
    return StageRegistry(stages)


STAGE_REGISTRY = build_stage_registry()
STAGES = [(stage.name, stage.runner) for stage in STAGE_REGISTRY.ordered()]


def registry_for_assembly_asset_source(registry, asset_source):
    if asset_source == "raw":
        return StageRegistry(
            Stage(
                stage.name,
                stage.runner,
                tuple((*stage.depends_on, "gram"))
                if stage.name == "blender_scene"
                and "gram" not in stage.depends_on
                else stage.depends_on,
            )
            for stage in registry.ordered()
        )
    return StageRegistry(
        Stage(
            stage.name,
            stage.runner,
            (
                tuple((*stage.depends_on, "gram"))
                if stage.name == "blender_scene"
                and "gram" not in stage.depends_on
                else ("blender_scene",)
                if stage.name == "simulation"
                and "replace_articulated" in stage.depends_on
                else tuple(
                    dependency for dependency in stage.depends_on
                    if dependency != "replace_articulated"
                )
            ),
        )
        for stage in registry.ordered()
        if stage.name != "replace_articulated"
    )


def check_outputs(context):
    metadata = read_json(context.run_json)
    assert metadata.get("status") == "completed", "run is not completed"
    artifacts = metadata.get("final_artifacts") or {}
    relatives = [value.get("path") if isinstance(value, dict) else value
                 for value in artifacts.values()]
    invalid = [relative for relative in relatives
               if not isinstance(relative, str)
               or Path(relative).is_absolute() or ".." in Path(relative).parts]
    assert not invalid, "final artifact paths must be run-relative: " + ", ".join(map(str, invalid))
    run_root = context.run_root.resolve()
    invalid = [relative for relative in relatives
               if run_root not in context.run_path(relative).resolve().parents]
    assert not invalid, "final artifact paths must be run-relative: " + ", ".join(invalid)
    missing = [relative for relative in relatives
               if not context.run_path(relative).is_file()]
    assert not missing, "missing outputs: " + ", ".join(missing)
    assert artifacts, "run has no final artifacts"
    print("pipeline check passed")


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


class _FinalPathStream:
    _PATH = re.compile(
        r'(?<!["\w])((?:/|~/|(?:data|logs|outputs|inputs|masks|crops|selected_items|multiviews|'
        r'rigid_meshes|articulated_assets|completed_instances|objects|raw)/)'
        r'[A-Za-z0-9_./~*+:-]+)'
    )

    def __init__(self, stream, attempt):
        self.stream = stream
        self.lock = threading.Lock()
        attempts = attempt if isinstance(attempt, (list, tuple)) else (attempt,)
        self.paths = []
        for item in attempts:
            self.add(item)

    def add(self, attempt):
        with self.lock:
            self.paths.append((str(attempt.temp_root), str(attempt.final_root)))

    def write(self, value):
        with self.lock:
            for temporary, final in self.paths:
                value = value.replace(temporary, final)
            return self.stream.write(self._PATH.sub(r'"\1"', value))

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def print_final_attempt_paths(attempt):
    with redirect_stdout(_FinalPathStream(sys.stdout, attempt)), \
         redirect_stderr(_FinalPathStream(sys.stderr, attempt)):
        yield


def print_final_output_paths(attempt):
    alignment = attempt.final_root / "outputs/two_camera_alignment"
    manifest = alignment / "visualization_paths.json"
    if not manifest.is_file():
        return
    print("[final output paths]", flush=True)
    for name, relative in read_json(manifest).items():
        output = alignment / relative
        if output.is_file():
            print(f"{name}='{output}'", flush=True)


def run_stage(name, fn, index=None, total=None):
    label = f"[stage {index}/{total}] {name}" if index else f"[stage] {name}"
    start = time.monotonic()
    print(f"{label} started", flush=True)
    try:
        fn()
    except Exception:
        print(f"{label} failed after {format_duration(time.monotonic() - start)}", flush=True)
        raise
    print(f"{label} done in {format_duration(time.monotonic() - start)}", flush=True)


def run_context_stage(context, name, runner):
    """Execute one registered stage with the pipeline's attempt lifecycle."""
    attempt = context.begin_attempt(name)
    result = None

    def execute():
        nonlocal result
        result = runner(context, attempt) or {}

    try:
        run_stage(name, execute)
    except BaseException as exc:
        context.fail_attempt(attempt, exc)
        raise
    context.complete_attempt(attempt, result)
    return attempt


def run_all():
    run_selected(STAGE_REGISTRY.ordered(), check=True)


def split_stage_names(value):
    if not value:
        return []
    names = []
    for part in value:
        names.extend(name.strip() for name in part.split(",") if name.strip())
    return names


def require_known_stages(names, registry=STAGE_REGISTRY):
    known = {stage.name for stage in registry.ordered()}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit("unknown stage(s): " + ", ".join(unknown))


def selected_stages(args, registry=STAGE_REGISTRY):
    stages = registry.ordered()
    names = [stage.name for stage in stages]
    require_known_stages(
        [name for name in (args.from_stage, args.to_stage) if name],
        registry,
    )
    selected = stages
    if args.stages:
        wanted = split_stage_names(args.stages)
        require_known_stages(wanted, registry)
        wanted_set = set(wanted)
        selected = [stage for stage in stages if stage.name in wanted_set]
    else:
        start = names.index(args.from_stage) if args.from_stage else 0
        end = names.index(args.to_stage) + 1 if args.to_stage else len(stages)
        if start >= end:
            raise SystemExit("--from-stage must be before or equal to --to-stage")
        selected = stages[start:end]

    skip = split_stage_names(args.skip)
    require_known_stages(skip, registry)
    skip_set = set(skip)
    return [stage for stage in selected if stage.name not in skip_set]


def run_selected(stages, check=False):
    if not stages:
        raise SystemExit("no stages selected")
    ensure_dirs()
    start = time.monotonic()
    total = len(stages)
    for index, stage in enumerate(stages, 1):
        run_stage(stage.name, stage.runner, index, total)
    if check:
        check_outputs()
    print(f"[pipeline] done in {format_duration(time.monotonic() - start)}", flush=True)


def _run_name(value):
    try:
        normalized = slug(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if normalized == "best":
        raise argparse.ArgumentTypeError("run name 'best' is reserved")
    return value


def _run_id(value):
    if value.startswith("best_"):
        raise argparse.ArgumentTypeError("--run-id requires the immutable run ID, not a best_ directory name")
    if value == "latest" or not re.fullmatch(r"\d{8}_\d{6}_[a-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("--run-id requires an exact timestamped run ID")
    return value


def build_parser(registry=STAGE_REGISTRY):
    names = [stage.name for stage in registry.ordered()]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-name", type=_run_name)
    parser.add_argument("--run-id", type=_run_id)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument(
        "--articulation-method",
        choices=sorted(ARTICULATION_METHODS),
        default=None,
    )
    parser.add_argument(
        "--assembly-asset-source", choices=("raw", "articulated"), default=None,
        help="Use Image-to-3D meshes directly or the articulated modeling result during assembly.",
    )
    parser.add_argument(
        "--drawer-geometry-mode",
        choices=("reconstructed-open", "procedural-closed", "auto-3d"),
        default=None,
        help="Use opened reconstruction, procedural closed drawers, or select per drawer from 3D geometry.",
    )
    parser.add_argument("--rerun-stage", choices=names)
    parser.add_argument("--mark-stage-best", metavar="STAGE:ATTEMPT")
    parser.add_argument("--mark-best", type=_run_id, metavar="RUN_ID")
    parser.add_argument("--stage", choices=names)
    parser.add_argument("--stages", nargs="+", help="Run selected stages in pipeline order. Accepts comma or space separated names.")
    parser.add_argument("--from-stage", choices=names, help="Run from this stage through --to-stage or the end.")
    parser.add_argument("--to-stage", choices=names, help="Run through this stage.")
    parser.add_argument("--skip", nargs="+", help="Skip selected stages. Accepts comma or space separated names.")
    parser.add_argument(
        "--condition-object-id", action="append", dest="condition_object_ids",
        help="Regenerate only this object in condition_images; repeat for multiple objects.",
    )
    parser.add_argument(
        "--image-to-3d-object-id", action="append", dest="image_to_3d_object_ids",
        help=(
            "Regenerate only this object in image_to_3d, reuse available "
            "others from the current best, and skip unavailable ones; repeat "
            "for multiple objects."
        ),
    )
    parser.add_argument(
        "--gram-object-id", action="append", dest="gram_object_ids",
        help=(
            "Regenerate only this object in gram and reuse completed "
            "packages for all other articulated objects; repeat for multiple objects."
        ),
    )
    parser.add_argument(
        "--gram-from-stage", choices=run_articulation.GRAM_STAGES,
        help="Resume each gram object from this internal stage.",
    )
    parser.add_argument(
        "--gram-to-stage", choices=run_articulation.GRAM_STAGES,
        help="Stop each gram object after this internal stage.",
    )
    articulated_generation = parser.add_mutually_exclusive_group()
    articulated_generation.add_argument(
        "--generate-articulated-objects",
        dest="generate_articulated_objects",
        action="store_true",
        default=True,
        help="Generate articulated objects during image_to_3d (default).",
    )
    articulated_generation.add_argument(
        "--no-generate-articulated-objects",
        dest="generate_articulated_objects",
        action="store_false",
        help="Skip articulated objects during image_to_3d.",
    )
    articulated_generation.add_argument(
        "--only-generate-articulated-objects",
        action="store_true",
        help=(
            "Reuse rigid objects from the current image_to_3d best and generate "
            "only articulated objects."
        ),
    )
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def parse_args(argv=None, registry=STAGE_REGISTRY):
    parser = build_parser(registry)
    args = parser.parse_args(argv)
    selectors = (args.stage, args.stages, args.from_stage, args.to_stage, args.skip)
    if args.run_id and args.run_name:
        parser.error("--run-id cannot be combined with --run-name")
    if args.run_id and args.input_image:
        parser.error("--input-image can only be used when creating a run")
    if not args.run_id and not args.input_image and not args.list_stages and not args.mark_best:
        parser.error("--input-image is required when creating a run")
    if args.mark_best and (args.run_id or any(selectors) or args.rerun_stage or args.mark_stage_best):
        parser.error("--mark-best cannot be combined with run execution options")
    if args.rerun_stage and any(selectors):
        parser.error("--rerun-stage cannot be combined with stage selection options")
    if args.mark_stage_best and not args.run_id:
        parser.error("--mark-stage-best requires --run-id")
    if (args.rerun_stage or args.mark_stage_best) and not args.run_id:
        parser.error("existing-run operations require --run-id")
    if args.condition_object_ids and not args.run_id:
        parser.error("--condition-object-id requires --run-id")
    if args.image_to_3d_object_ids and not args.run_id:
        parser.error("--image-to-3d-object-id requires --run-id")
    if args.gram_object_ids and not args.run_id:
        parser.error("--gram-object-id requires --run-id")
    if (args.gram_from_stage or args.gram_to_stage) and not args.run_id:
        parser.error("gram stage range requires --run-id")
    gram_start = args.gram_from_stage or "segment"
    gram_end = args.gram_to_stage or "export"
    if run_articulation.GRAM_STAGES.index(gram_start) > \
            run_articulation.GRAM_STAGES.index(gram_end):
        parser.error("--gram-from-stage must not follow --gram-to-stage")
    if args.only_generate_articulated_objects and not args.run_id:
        parser.error("--only-generate-articulated-objects requires --run-id")
    if args.image_to_3d_object_ids and args.only_generate_articulated_objects:
        parser.error(
            "--image-to-3d-object-id cannot be combined with "
            "--only-generate-articulated-objects"
        )
    if args.stage and any((args.stages, args.from_stage, args.to_stage, args.skip)):
        parser.error("--stage cannot be combined with --stages/--from-stage/--to-stage/--skip")
    return args


def _load_run(project_root, scene_id, run_id):
    scene_id = slug(scene_id)
    project_root = Path(project_root).resolve()
    run_root = resolve_run_root(project_root, scene_id, run_id)
    metadata = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if metadata.get("scene_id") != scene_id or metadata.get("run_id") != run_id:
        raise ValueError(f"invalid run metadata: {run_id}")
    return RunContext(project_root, scene_id, run_id, run_root), metadata


def _registry_for_snapshot(registry, metadata):
    current = {stage.name: stage.runner for stage in registry.ordered()}
    frozen_names = metadata.get("stage_order", [])
    if frozen_names != list(current):
        raise ValueError("run stage topology does not match this review package")
    dependencies = metadata.get("stage_dependencies")
    if not isinstance(dependencies, dict) or any(
        name not in dependencies for name in frozen_names
    ):
        raise ValueError("invalid frozen stage dependencies")
    return StageRegistry(
        Stage(name, current[name], tuple(dependencies[name]))
        for name in frozen_names
    )


def registry_for_context(context, registry=STAGE_REGISTRY):
    metadata = read_json(context.run_json)
    active = registry_for_assembly_asset_source(
        registry, metadata.get("assembly_asset_source", "raw"),
    )
    return _registry_for_snapshot(active, metadata)




def _validate_real_attempt_lineage(context, attempt, allow_remap_reuse=False):
    run_json = getattr(context, "run_json", None)
    if not isinstance(run_json, Path):
        return
    metadata = read_json(run_json)
    if allow_remap_reuse and attempt.stage in {
        "segment_instances", "blender_scene",
        "replace_articulated",
        "simulation",
    }:
        return


def _allows_remap_reuse(selected_names):
    return (
        "condition_images" not in selected_names
        and bool({
            "segment_instances", "blender_scene",
            "replace_articulated",
            "simulation",
        } & selected_names)
    )


def ensure_mesh_orientation_for_blender(context, stages, registry):
    """Run mesh orientation before Blender only when no usable result exists."""
    names = {stage.name for stage in stages}
    if "blender_scene" not in names or "mesh_orientation" in names:
        return stages
    blender = next(stage for stage in stages if stage.name == "blender_scene")
    if "mesh_orientation" not in blender.depends_on:
        return stages
    orientation = next(
        stage for stage in registry.ordered()
        if stage.name == "mesh_orientation"
    )
    refresh = bool(names & set(orientation.depends_on))
    try:
        root = context.stage_best("mesh_orientation")
        record = read_json(root / "attempt.json")
        relative = record["artifacts"]["manifest"]
        manifest = (root / relative).resolve()
        manifest_document = read_json(manifest)
        current_inputs = {
            dependency: context.stage_best(dependency).name
            for dependency in orientation.depends_on
        }
        reusable = (
            not refresh
            and isinstance(relative, str)
            and root.resolve() in manifest.parents
            and manifest.is_file()
            and (manifest_document.get("top_template_cache") or {}).get(
                "version"
            ) == 1
            and isinstance(
                (manifest_document.get("table") or {}).get("yaw_deg"),
                (int, float),
            )
            and record.get("upstream_best") == current_inputs
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        reusable = False
    if reusable:
        print(
            f"[blender_scene] reusing mesh_orientation attempt {root.name}",
            flush=True,
        )
        return stages
    print(
        "[blender_scene] mesh_orientation is missing or stale; running it once",
        flush=True,
    )
    names.add("mesh_orientation")
    return [stage for stage in registry.ordered() if stage.name in names]


def _run_attempts(
    context, stages, condition_object_ids=None,
    generate_articulated_objects=True,
    only_generate_articulated_objects=False,
    image_to_3d_object_ids=None,
    gram_object_ids=None,
    gram_from_stage=None,
    gram_to_stage=None,
):
    if not stages:
        raise SystemExit("no stages selected")
    total = len(stages)
    selected_names = {stage.name for stage in stages}
    allow_remap_reuse = _allows_remap_reuse(selected_names)
    stage_numbers = {stage.name: index for index, stage in enumerate(stages, 1)}
    pending = {stage.name: stage for stage in stages}
    completed_in_run = {}
    running = {}
    errors = []
    sam_gpu_lock = threading.Lock()
    gpu_stage_lock = threading.Lock()
    sam_gpu_runners = {
        run_grounded_sam.run_anonymous_views,
        run_grounded_sam.run_remapped_views,
    }
    stdout = _FinalPathStream(sys.stdout, [])
    stderr = _FinalPathStream(sys.stderr, [])
    max_stage_workers = min(
        total, env_positive_int("PIPELINE_MAX_STAGE_WORKERS", 4),
    )

    def selected_dependencies(stage):
        dependencies = (
            stage.depends_on
            if isinstance(stage.depends_on, (tuple, list, set))
            else ()
        )
        return set(dependencies) & selected_names

    def execute_stage(stage, attempt):
        def execute():
            _validate_real_attempt_lineage(
                context, attempt, allow_remap_reuse=allow_remap_reuse,
            )
            parameters = inspect.signature(stage.runner).parameters
            if stage.name == "condition_images" and condition_object_ids:
                return stage.runner(context, attempt, object_ids=condition_object_ids) or {}
            args = () if not inspect.signature(stage.runner).parameters else (
                context, attempt,
            )
            kwargs = {}
            if stage.name == "image_to_3d":
                if "generate_articulated_objects" in parameters:
                    kwargs["generate_articulated_objects"] = generate_articulated_objects
                if "only_generate_articulated_objects" in parameters:
                    kwargs["only_generate_articulated_objects"] = (
                        only_generate_articulated_objects
                    )
                if "image_to_3d_object_ids" in parameters:
                    kwargs["image_to_3d_object_ids"] = image_to_3d_object_ids
            if stage.name == "gram":
                if "gram_object_ids" in parameters:
                    kwargs["gram_object_ids"] = gram_object_ids
                if "gram_from_stage" in parameters:
                    kwargs["gram_from_stage"] = gram_from_stage
                if "gram_to_stage" in parameters:
                    kwargs["gram_to_stage"] = gram_to_stage
            return stage.runner(*args, **kwargs) or {}

        result = None

        def capture():
            nonlocal result
            result = execute()

        if stage.runner in sam_gpu_runners:
            with sam_gpu_lock:
                run_stage(
                    stage.name, capture, stage_numbers[stage.name], total,
                )
        elif stage.name in {"gram", "mesh_orientation", "blender_scene"}:
            with gpu_stage_lock:
                run_stage(
                    stage.name, capture, stage_numbers[stage.name], total,
                )
        else:
            run_stage(stage.name, capture, stage_numbers[stage.name], total)
        return result

    with context.lock(), redirect_stdout(stdout), redirect_stderr(stderr), \
            ThreadPoolExecutor(
                max_workers=max_stage_workers
            ) as executor:
        while pending or running:
            if not errors:
                ready = [
                    stage for stage in pending.values()
                    if selected_dependencies(stage) <= completed_in_run.keys()
                ]
                for stage in ready:
                    if len(running) >= max_stage_workers:
                        break
                    attempt = context.begin_attempt(
                        stage.name, upstream_overrides=completed_in_run,
                    )
                    stdout.add(attempt)
                    stderr.add(attempt)
                    pending.pop(stage.name)
                    future = executor.submit(execute_stage, stage, attempt)
                    running[future] = (stage, attempt)

            if not running:
                if errors:
                    break
                raise RuntimeError("selected pipeline stages cannot make progress")

            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in sorted(
                finished, key=lambda item: stage_numbers[running[item][0].name],
            ):
                stage, attempt = running.pop(future)
                try:
                    artifacts = future.result()
                except BaseException as exc:
                    context.fail_attempt(attempt, exc)
                    errors.append(exc)
                else:
                    context.complete_attempt(attempt, artifacts)
                    print_final_output_paths(attempt)
                    completed_in_run[stage.name] = attempt.attempt_id
                    print(
                        f"{stage.name} attempt: {attempt.attempt_id}", flush=True,
                    )

        if errors:
            raise errors[0]
        if stages[-1].name == "simulation":
            simulation_root = context.stage_best("simulation")
            scene_stage = (
                "replace_articulated"
                if "replace_articulated" in read_json(context.run_json).get("stage_order", [])
                else "blender_scene"
            )
            scene_root = context.stage_best(scene_stage)
            scene_attempt = read_json(scene_root / "attempt.json")
            simulation_attempt = read_json(simulation_root / "attempt.json")
            simulation_metadata = read_json(
                simulation_root / simulation_attempt["artifacts"]["metadata"]
            )
            scene_path = scene_root / scene_attempt["artifacts"]["scene"]
            final_artifacts = {
                "scene_blend": scene_root / "outputs/blender_scene.blend",
                "scene_usd": simulation_root / "outputs/isaac_loaded_scene.usd",
                "preview": simulation_root / "outputs/simulation_preview.png",
            }
            if scene_path.suffix.lower() == ".glb":
                final_artifacts["scene_glb"] = scene_path
            asset_bundle = simulation_attempt["artifacts"].get("bundle")
            if asset_bundle:
                final_artifacts["asset_bundle"] = simulation_root / asset_bundle
            elif simulation_metadata.get("asset_package_published") is not False:
                sim_bundle = scene_attempt["artifacts"].get("sim_bundle")
                if sim_bundle:
                    final_artifacts["asset_bundle"] = scene_root / sim_bundle
            else:
                for name in ("isaac_asset_package.usdz", "sim_asset_bundle.zip"):
                    (context.run_root / "final" / name).unlink(missing_ok=True)
            video = simulation_root / "outputs/simulation.mp4"
            if video.is_file():
                final_artifacts["video"] = video
            else:
                (context.run_root / "final/simulation.mp4").unlink(missing_ok=True)
            sample_frames = simulation_metadata.get("sample_frames", [])
            if sample_frames or simulation_metadata.get("capture_mode") == "single_frame":
                final_root = context.run_root / "final"
                for stale in final_root.glob("frame_*.png"):
                    stale.unlink()
            if sample_frames:
                for sample in sample_frames:
                    frame = int(sample["frame_index"])
                    if frame == 0:
                        continue
                    relative = Path(sample["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"invalid sampled frame path: {relative}")
                    final_artifacts[f"motion_frame_{frame:06d}"] = (
                        simulation_root / relative
                    )
            update_scene_index(context.project_root, context.scene_id)


def dispatch(args, project_root=None, registry=STAGE_REGISTRY):
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    if args.list_stages:
        active_registry = registry_for_assembly_asset_source(
            registry, args.assembly_asset_source or "raw",
        )
        print("\n".join(stage.name for stage in active_registry.ordered()))
        return None
    if args.mark_best:
        mark_run_best(project_root, args.scene, args.mark_best)
        return None

    if args.run_id:
        context = RunContext.resume(
            project_root, args.scene, args.run_id,
            articulation_method=args.articulation_method,
            assembly_asset_source=args.assembly_asset_source,
            drawer_geometry_mode=args.drawer_geometry_mode,
        )
        active_registry = registry_for_context(context, registry)
    else:
        assembly_asset_source = args.assembly_asset_source or "raw"
        context = RunContext.create(
            project_root,
            args.scene,
            args.run_name,
            articulation_method=(
                args.articulation_method or DEFAULT_ARTICULATION_METHOD
            ),
            assembly_asset_source=assembly_asset_source,
            input_image=args.input_image,
            drawer_geometry_mode=(
                args.drawer_geometry_mode or DEFAULT_DRAWER_GEOMETRY_MODE
            ),
        )
        active_registry = registry_for_assembly_asset_source(
            registry, assembly_asset_source,
        )
        context.initialize_stages(active_registry)

    if args.mark_stage_best:
        try:
            stage, attempt_id = args.mark_stage_best.split(":", 1)
        except ValueError:
            raise SystemExit("--mark-stage-best must be STAGE:ATTEMPT") from None
        require_known_stages([stage], active_registry)
        with context.lock():
            context.mark_stage_best(stage, attempt_id)
        return context
    if args.check:
        check_outputs(context)
        return context
    if args.rerun_stage:
        require_known_stages([args.rerun_stage], active_registry)
        selected = [stage for stage in active_registry.ordered() if stage.name == args.rerun_stage]
    elif args.stage:
        require_known_stages([args.stage], active_registry)
        selected = [stage for stage in active_registry.ordered() if stage.name == args.stage]
    elif args.stages or args.from_stage or args.to_stage or args.skip:
        selected = selected_stages(args, active_registry)
    else:
        selected = active_registry.ordered()
    selected = ensure_mesh_orientation_for_blender(
        context, selected, active_registry,
    )
    if args.condition_object_ids and [stage.name for stage in selected] != ["condition_images"]:
        raise SystemExit("--condition-object-id requires running only condition_images")
    if (
        args.image_to_3d_object_ids
        and [stage.name for stage in selected] != ["image_to_3d"]
    ):
        raise SystemExit("--image-to-3d-object-id requires running only image_to_3d")
    if (
        args.gram_object_ids
        and [stage.name for stage in selected] != ["gram"]
    ):
        raise SystemExit("--gram-object-id requires running only gram")
    if (
        (args.gram_from_stage or args.gram_to_stage)
        and [stage.name for stage in selected] != ["gram"]
    ):
        raise SystemExit("gram stage range requires running only gram")
    if (
        args.only_generate_articulated_objects
        and [stage.name for stage in selected] != ["image_to_3d"]
    ):
        raise SystemExit(
            "--only-generate-articulated-objects requires running only image_to_3d"
        )
    _run_attempts(
        context, selected,
        condition_object_ids=args.condition_object_ids,
        generate_articulated_objects=args.generate_articulated_objects,
        only_generate_articulated_objects=args.only_generate_articulated_objects,
        image_to_3d_object_ids=args.image_to_3d_object_ids,
        gram_object_ids=args.gram_object_ids,
        gram_from_stage=args.gram_from_stage,
        gram_to_stage=args.gram_to_stage,
    )
    return context


def main(argv=None):
    dispatch(parse_args(argv))


if __name__ == "__main__":
    main()
