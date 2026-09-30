from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from pipeline.common import env_positive_int, path
from pipeline.mllm import (
    image_data_url,
    image_generation_runtime,
    request_responses,
    write_runtime_marker,
)


TOPVIEW_IMAGE_RESOLUTION = "1024x576"
REFERENCE_IMAGE_RESOLUTION = "2048x1152"
MULTIVIEW_SHEET_RESOLUTION = "2048x1024"
TRELLIS2_CONDITION_RESOLUTION = "1024x1024"
IMAGE_RESOLUTION = REFERENCE_IMAGE_RESOLUTION


def resize_png(in_path, size):
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required to normalize generated image size")
    width, height = size.split("x", 1)
    resized = in_path.with_name(in_path.stem + ".resized.png")
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(in_path),
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white"
            ),
            str(resized),
        ],
        text=True,
        capture_output=True,
        cwd=path(),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    shutil.move(str(resized), in_path)


def _run_image_generation_once(
    prompt, model, out_path, log_prefix, images=None, resolution=IMAGE_RESOLUTION,
    final_size=None, project_root=None, log_dir=None, opener=urllib.request.urlopen,
):
    runtime = image_generation_runtime().with_model(model)
    tool_model = os.environ.get(
        "IMAGE_GENERATION_MODEL", "YOUR_IMAGE_GENERATION_MODEL",
    )
    if not tool_model or tool_model.startswith("YOUR_"):
        raise ValueError("IMAGE_GENERATION_MODEL must be configured")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir or path("data"))
    log_dir.mkdir(parents=True, exist_ok=True)
    response_path = log_dir / f"{log_prefix}_image_response.json"
    content = [{"type": "input_text", "text": prompt(None) if callable(prompt) else prompt}]
    content.extend({"type": "input_image", "image_url": image_data_url(image)} for image in images or [])
    payload = {
        "model": runtime.model,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": runtime.reasoning_effort},
        "tools": [{"type": "image_generation", "model": tool_model}],
        "tool_choice": {"type": "image_generation"},
        "store": False,
    }
    write_runtime_marker(runtime, response_path)
    started = time.monotonic()
    document = request_responses(runtime, payload, timeout=1200, opener=opener)
    response_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    calls = [item for item in document.get("output", []) if item.get("type") == "image_generation_call"]
    if len(calls) != 1 or not calls[0].get("result"):
        raise RuntimeError(f"OpenAI Responses API returned {len(calls)} generated images")
    temporary = out_path.with_name(out_path.stem + ".api_tmp.png")
    temporary.write_bytes(base64.b64decode(calls[0]["result"], validate=True))
    if temporary.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        temporary.unlink(missing_ok=True)
        raise RuntimeError("OpenAI image generation returned a non-PNG artifact")
    if final_size:
        resize_png(temporary, final_size)
    shutil.move(str(temporary), out_path)
    print(
        f"[image-generation] {log_prefix} saved {out_path} in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )
    return calls[0].get("revised_prompt", "")


def run_image_generation(
    prompt, model, out_path, log_prefix, images=None, resolution=IMAGE_RESOLUTION,
    final_size=None, project_root=None, log_dir=None,
):
    attempts = env_positive_int("IMAGE_GENERATION_MAX_ATTEMPTS", 3)
    for attempt in range(1, attempts + 1):
        attempt_prefix = log_prefix if attempt == 1 else f"{log_prefix}_retry_{attempt}"
        try:
            return _run_image_generation_once(
                prompt, model, out_path, attempt_prefix, images, resolution,
                final_size, project_root, log_dir,
            )
        except RuntimeError:
            if attempt == attempts:
                raise
            print(
                f"[image-generation] {log_prefix} failed; retrying "
                f"({attempt + 1}/{attempts})",
                flush=True,
            )
