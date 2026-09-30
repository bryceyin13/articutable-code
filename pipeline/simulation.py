#!/usr/bin/env python3
import json
import math
import os
import shutil
import socket
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import log_step, path, write_json
from gram.urdf_assets import urdf_prismatic_scales


def joint_angle_degrees(frame, frame_count, lower_deg, upper_deg):
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    phase = 2.0 * math.pi * frame / (frame_count - 1)
    blend = 0.5 * (1.0 - math.cos(phase))
    return round(lower_deg + (upper_deg - lower_deg) * blend, 12)


def manifest_joint_ranges(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranges = {
        (obj["object_id"], joint["name"]): (
            *joint["video_range"],
            joint.get(
                "scene_current_q",
                joint.get("reference_q", joint["video_range"][0]),
            ),
        )
        for obj in manifest.get("objects", [])
        for joint in obj.get("scene_joint_ranges", [])
        if len(joint.get("video_range", [])) == 2
        and joint["video_range"][0] < joint["video_range"][1]
    }
    sim_export = manifest_path.parents[1] / "outputs/sim_export"
    for obj in manifest.get("objects", []):
        if obj.get("type") != "articulated":
            continue
        package_manifest = sim_export / Path(obj["asset"]).parent / "manifest.json"
        if not package_manifest.is_file():
            continue
        for state in json.loads(package_manifest.read_text(encoding="utf-8")).get(
            "joint_states", []
        ):
            if "scene_current_q" not in state:
                continue
            name = str(state["name"])
            prefix = f'{obj["object_id"]}_'
            if not name.startswith(prefix):
                name = prefix + name
            key = (obj["object_id"], name)
            lower, upper, _ = ranges.get(key, (None, None, None))
            ranges[key] = (lower, upper, float(state["scene_current_q"]))
        urdf_path = sim_export / obj["asset"]
        factors = urdf_prismatic_scales(
            urdf_path,
            obj.get("root_pose", {}).get("scale", (1.0, 1.0, 1.0)),
        ) if urdf_path.is_file() else {}
        for joint_name, factor in factors.items():
            key = (obj["object_id"], joint_name)
            if key not in ranges:
                continue
            lower, upper, reference = ranges[key]
            ranges[key] = tuple(
                None if value is None else value * factor
                for value in (lower, upper, reference)
            )
    return ranges


def video_config(env):
    if not env.get("ISAAC_GPU"):
        raise ValueError("ISAAC_GPU must be configured")
    fps = int(env.get("JOINT_VIDEO_FPS", "30"))
    seconds = float(env.get("JOINT_VIDEO_SECONDS", "5"))
    single_frame = env.get("JOINT_VIDEO_SINGLE_FRAME", "1").lower() in {
        "1", "true", "yes",
    }
    frame_count = 1 if single_frame else round(fps * seconds)
    sample_frames = [
        int(value.strip())
        for value in env.get("JOINT_VIDEO_SAMPLE_FRAMES", "").split(",")
        if value.strip()
    ]
    if sample_frames and (
        single_frame
        or sample_frames != sorted(set(sample_frames))
        or sample_frames[0] != 0
        or sample_frames[-1] >= frame_count
    ):
        raise ValueError(
            "JOINT_VIDEO_SAMPLE_FRAMES must be sorted unique timeline indices "
            "starting at 0 and inside the full multi-frame timeline"
        )
    config = {
        "width": int(env.get("ISAAC_STREAM_WIDTH", "3840")),
        "height": int(env.get("ISAAC_STREAM_HEIGHT", "2160")),
        "fps": fps,
        "seconds": seconds,
        "frame_count": frame_count,
        "sample_frames": sample_frames,
        "single_frame": single_frame,
        "gpu": int(env["ISAAC_GPU"]),
        "pathtracing_spp": int(env.get("ISAAC_PATHTRACING_SPP", "16")),
        "pathtracing_denoiser": env.get(
            "ISAAC_PATHTRACING_DENOISER", "1"
        ).lower() in {"1", "true", "yes"},
    }
    if env.get("ISAAC_TEXTURE_CACHE_PATH"):
        config["texture_cache_path"] = env["ISAAC_TEXTURE_CACHE_PATH"]
    return config


def asset_package_enabled(env):
    value = env.get("TABLETOP_PUBLISH_USDZ", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "TABLETOP_PUBLISH_USDZ must be one of "
        "1/0, true/false, yes/no, or on/off"
    )


def ffmpeg_command(frames_dir, output_path, fps):
    return [
        shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(Path(frames_dir) / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def isaac_command(isaac_root, worker, config):
    width = config["width"]
    height = config["height"]
    renderer_gpu = config["gpu"]
    portable_root = Path(
        config.get("portable_root", Path(worker).parent / ".joint_video_portable")
    ).resolve()
    return [
        str(isaac_root / "isaac-sim.sh"),
        "--no-window",
        "--portable",
        "--portable-root",
        str(portable_root),
        "--/app/file/ignoreUnsavedOnExit=true",
        "--/app/scripting/ignoreWarningDialog=true",
        "--/app/fastShutdown=true",
        f"--/renderer/activeGpu={renderer_gpu}",
        "--/physics/cudaDevice=0",
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        f"--/app/window/width={width}",
        f"--/app/window/height={height}",
        f"--/app/renderer/resolution/width={width}",
        f"--/app/renderer/resolution/height={height}",
        f"--/persistent/app/viewport/Viewport/Viewport0/resolution/0={width}",
        f"--/persistent/app/viewport/Viewport/Viewport0/resolution/1={height}",
        "--/persistent/app/viewport/Viewport/Viewport0/resolutionScale=1.0",
        "--/rtx/post/histogram/enabled=false",
        "--/rtx/post/aa/op=0",
        "--/ngx/enabled=false",
        *(
            [
                "--/rtx-transient/resourcemanager/localTextureCachePath="
                + str(config["texture_cache_path"])
            ]
            if config.get("texture_cache_path") else []
        ),
        "--exec",
        str(worker),
    ]


def run_isaac_capture(command, env, log_path):
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path(command[0]).parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_file.write(line)
            if "[joint-video]" in line:
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _persistent_request(socket_path, payload, timeout):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        connection.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        with connection.makefile("r", encoding="utf-8") as response_file:
            return json.loads(response_file.readline())


def run_persistent_isaac_capture(socket_path, request, log_path):
    config = json.loads(Path(request["config_path"]).read_text(encoding="utf-8"))
    gpu = str(config["gpu"])
    deadline = time.monotonic() + 3600.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"resident Isaac worker {gpu} remained busy")
        try:
            status = _persistent_request(
                socket_path, {"ping": True}, min(1.0, remaining),
            )
        except (OSError, json.JSONDecodeError):
            time.sleep(min(1.0, max(0.0, remaining)))
            continue
        if status.get("ok") and status.get("status") == "ready":
            response = _persistent_request(socket_path, request, remaining)
            if response.get("status") != "busy":
                break
        time.sleep(min(1.0, max(0.0, remaining)))
    log_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    if not response.get("ok"):
        raise RuntimeError(
            "persistent Isaac capture failed: "
            f"{response.get('error', 'unknown error')}\n"
            f"{response.get('traceback', '')}"
        )


def discover_persistent_isaac_socket(gpu, state_root=None):
    state_root = Path(state_root or path("logs/isaac_debug_worker"))
    try:
        state = json.loads(
            (state_root / f"gpu_{gpu}/state.json").read_text(encoding="utf-8")
        )
        if str(state.get("gpu")) != str(gpu):
            return None
        socket_path = state["socket"]
        response = _persistent_request(socket_path, {"ping": True}, 1.0)
        return (
            socket_path
            if response.get("ok") and response.get("status") == "ready"
            else None
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def probe_video(video_path):
    command = [
        shutil.which("ffprobe") or "/usr/bin/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def validate_probe(probe, config):
    stream = probe["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count = int(stream.get("nb_frames") or config["frame_count"])
    duration = float(probe["format"]["duration"])
    if (width, height) != (config["width"], config["height"]):
        raise RuntimeError(f"unexpected video resolution: {width}x{height}")
    if abs(fps - config["fps"]) > 0.01:
        raise RuntimeError(f"unexpected video fps: {fps}")
    if frame_count != config["frame_count"]:
        raise RuntimeError(f"unexpected video frame count: {frame_count}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def remove_capture_work_dir(work_dir):
    if not work_dir.exists():
        return
    subprocess.run(["chmod", "-R", "u+rwX", str(work_dir)], check=True)
    shutil.rmtree(work_dir)


def write_rest_preview(frames_dir, preview_path):
    rest_frame = Path(frames_dir) / "frame_000000.png"
    if not rest_frame.is_file():
        raise FileNotFoundError(f"rest frame is missing: {rest_frame}")
    shutil.copy2(rest_frame, preview_path)


def output_paths(output_root):
    output_root = Path(output_root)
    work_dir = output_root / "outputs/simulation_capture"
    return {
        "work_dir": work_dir,
        "frames_dir": work_dir / "frames",
        "config_path": work_dir / "config.json",
        "status_path": work_dir / "status.json",
        "log_path": work_dir / "isaac.log",
        "video": output_root / "outputs/simulation.mp4",
        "preview": output_root / "outputs/simulation_preview.png",
        "metadata": output_root / "outputs/simulation_metadata.json",
        "bundle": output_root / "outputs/isaac_asset_package.usdz",
        "portable_root": output_root / "outputs/simulation_kit_portable",
    }


def run(context=None, attempt=None):
    config = video_config(os.environ)
    publish_package = asset_package_enabled(os.environ)
    minimum_frames = 1 if config["single_frame"] else 2
    if (
        min(config["width"], config["height"], config["fps"]) <= 0
        or config["frame_count"] < minimum_frames
    ):
        raise ValueError("capture dimensions, fps, and frame count are invalid")

    isaac_root_value = os.environ.get("ISAAC_SIM_ROOT")
    if not isaac_root_value:
        raise RuntimeError("ISAAC_SIM_ROOT must be configured")
    isaac_root = Path(isaac_root_value).resolve()
    isaac_launcher = isaac_root / "isaac-sim.sh"
    if not isaac_launcher.is_file():
        raise FileNotFoundError(f"Isaac Sim launcher not found: {isaac_launcher}")

    output_root = attempt.temp_root if attempt else path()
    outputs = output_paths(output_root)
    work_dir = outputs["work_dir"]
    frames_dir = outputs["frames_dir"]
    config_path = outputs["config_path"]
    status_path = outputs["status_path"]
    log_path = outputs["log_path"]
    output_path = outputs["video"]
    preview_path = outputs["preview"]
    metadata_path = outputs["metadata"]

    remove_capture_work_dir(work_dir)
    frames_dir.mkdir(parents=True)
    portable_root = outputs["portable_root"]
    portable_root.mkdir(exist_ok=True)
    config["portable_root"] = str(portable_root)
    output_path.unlink(missing_ok=True)
    preview_path.unlink(missing_ok=True)

    frame_indices = (
        config["sample_frames"]
        or ([0] if config["single_frame"] else list(range(config["frame_count"])))
    )
    cycle_blend = [1.0] if config["single_frame"] else [
        joint_angle_degrees(frame, config["frame_count"], 0.0, 1.0)
        for frame in frame_indices
    ]
    if attempt:
        scene_stage = (
            "replace_articulated"
            if "replace_articulated" in attempt.upstream_best
            else "blender_scene"
        )
        sim_manifest = dependency_file(attempt, scene_stage, "sim_manifest")
    else:
        sim_manifest = path("data/sim_asset_manifest.json")
        if not sim_manifest.is_file():
            sim_manifest = path("outputs/sim_export/scene_manifest.json")
    joint_ranges = manifest_joint_ranges(sim_manifest)
    capture_config = {
        **config,
        "project_root": str(path()),
        "frames_dir": str(frames_dir),
        "status_path": str(status_path),
        "camera_path": "/World/BlenderScene/front_camera/front_camera",
        "capture_mode": (
            "sampled_frames" if config["sample_frames"] else
            "single_frame" if config["single_frame"] else "video"
        ),
        "asset_package_path": str(outputs["bundle"]),
        "frame_indices": frame_indices,
        "cycle_blend": cycle_blend,
        "scene_joint_ranges": [
            {
                "object_id": object_id,
                "joint_name": joint_name,
                "lower": limits[0],
                "upper": limits[1],
                "reference": limits[2],
            }
            for (object_id, joint_name), limits in joint_ranges.items()
        ],
    }
    write_json(config_path, capture_config)

    worker = Path(__file__).with_name("capture_joint_demo_isaac.py")
    command = isaac_command(isaac_root, worker, config)
    child_env = os.environ.copy()
    child_env.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PRIVACY_CONSENT": "N",
            "TABLETOP_ROOT": str(path()),
            "TABLETOP_SIM_EXPORT_ROOT": str(
                sim_manifest.parents[1] / "outputs/sim_export"
                if attempt else path("outputs/sim_export")
            ),
            "TABLETOP_SIM_MANIFEST_PATH": str(sim_manifest),
            "TABLETOP_SIM_OUTPUT_ROOT": str(output_root / "outputs"),
            "ISAAC_SIM_ROOT": str(isaac_root),
            "JOINT_VIDEO_CONFIG": str(config_path),
            "PYTHONPATH": child_env.get("PYTHONPATH", ""),
        }
    )
    child_env.setdefault("TABLETOP_FIX_ARTICULATION_BASE", "0")
    child_env["TABLETOP_PUBLISH_USDZ"] = "1" if publish_package else "0"
    child_env["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    child_env["NVIDIA_VISIBLE_DEVICES"] = str(config["gpu"])
    for name in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"):
        child_env.pop(name, None)

    log_step(
        "simulation",
        f"capturing timeline frame(s) {frame_indices} at "
        f"{config['width']}x{config['height']} on GPU {config['gpu']}",
    )
    server_socket = (
        os.environ.get("ISAAC_CAPTURE_SERVER_SOCKET")
        or discover_persistent_isaac_socket(config["gpu"])
    )
    if server_socket:
        log_step("simulation", f"using persistent Isaac worker {server_socket}")
        run_persistent_isaac_capture(
            server_socket,
            {
                "config_path": str(config_path),
                "environment": {
                    name: child_env[name]
                    for name in (
                        "TABLETOP_ROOT",
                        "TABLETOP_SIM_EXPORT_ROOT", "TABLETOP_SIM_MANIFEST_PATH",
                        "TABLETOP_SIM_OUTPUT_ROOT", "ISAAC_SIM_ROOT",
                        "TABLETOP_FIX_ARTICULATION_BASE", "TABLETOP_PUBLISH_USDZ",
                        "TABLETOP_LIGHTING_MODE", "TABLETOP_DOME_INTENSITY",
                        "TABLETOP_SOFTBOX_INTENSITY",
                        "TABLETOP_STUDIO_GROUND_SIZE",
                        "TABLETOP_BLENDER_BIN", "GRAM_BLENDER", "BLENDER_BIN",
                        "TABLETOP_TABLE_COLLISION_TARGET_FACES",
                        "TABLETOP_TABLE_COLLISION_MAX_ERROR_M",
                    )
                    if name in child_env
                },
            },
            log_path,
        )
    else:
        run_isaac_capture(command, child_env, log_path)
    if not status_path.is_file():
        raise RuntimeError(f"Isaac capture did not write {status_path}")
    capture_status = json.loads(status_path.read_text(encoding="utf-8"))
    if capture_status.get("status") != "complete":
        raise RuntimeError(f"Isaac capture failed: {capture_status.get('error', 'unknown error')}")

    if config["sample_frames"] and not publish_package:
        outputs["bundle"].unlink(missing_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    expected_frames = [frames_dir / f"frame_{frame:06d}.png" for frame in frame_indices]
    if frame_paths != expected_frames:
        raise RuntimeError(
            f"expected timeline frames {frame_indices}, found "
            f"{[int(item.stem.removeprefix('frame_')) for item in frame_paths]}"
        )

    video = None
    if config["single_frame"] or config["sample_frames"]:
        write_rest_preview(frames_dir, preview_path)
    else:
        log_step("simulation", "encoding outputs/simulation.mp4")
        subprocess.run(ffmpeg_command(frames_dir, output_path, config["fps"]), check=True)
        write_rest_preview(frames_dir, preview_path)
        video = validate_probe(probe_video(output_path), config)
    metadata = {
        **capture_status,
        "capture_mode": (
            "sampled_frames" if config["sample_frames"] else
            "single_frame" if config["single_frame"] else "video"
        ),
        "preview_pose": "reference_rest",
        "preview": str(preview_path.relative_to(output_root)),
        "isaac_log": str(log_path.relative_to(output_root)),
    }
    if config["sample_frames"]:
        metadata.update({
            "timeline_frame_count": config["frame_count"],
            "sample_frames": [
                {
                    "frame_index": frame,
                    "path": str(frame_path.relative_to(output_root)),
                }
                for frame, frame_path in zip(frame_indices, expected_frames)
            ],
            "video_generated": False,
        })
    if video is not None:
        metadata.update({
            "video": str(output_path.relative_to(path())),
            "video_properties": video,
        })
    write_json(metadata_path, metadata)

    keep_frames = os.environ.get("JOINT_VIDEO_KEEP_FRAMES", "0").lower() in {"1", "true", "yes"}
    if not keep_frames and not config["sample_frames"]:
        shutil.rmtree(frames_dir)
    if video is None:
        log_step("simulation", "wrote outputs/simulation_preview.png")
    else:
        log_step(
            "simulation",
            f"wrote outputs/simulation.mp4 ({video['width']}x{video['height']}, "
            f"{video['fps']:.0f} fps, {video['duration_seconds']:.1f}s)",
        )
    if attempt:
        outputs = dict(STAGE_OUTPUTS["simulation"])
        if config["single_frame"] or config["sample_frames"]:
            outputs = {key: value for key, value in outputs.items() if key != "video"}
        if not publish_package:
            outputs.pop("bundle")
        return completed_files(attempt, outputs)


if __name__ == "__main__":
    run()
