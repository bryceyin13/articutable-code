#!/usr/bin/env python3
import subprocess

from pipeline.image_generation import (
    REFERENCE_IMAGE_RESOLUTION,
    run_image_generation,
)
from pipeline.common import log_step
from pipeline.stage_io import (
    completed_files,
    stage_outputs,
)
from pipeline.common import attempt_path
from pipeline.mllm import image_generation_runtime


REAL_REFERENCE_PROMPT = """The highest-priority invariant is exact registration with the input image. Treat the input as a fixed, immutable canvas, not as a scene to recompose or redraw. Keep the original camera projection and the exact pixel coordinates, silhouettes, scale, orientation, and visible geometry of the complete load-bearing table and every retained subject. If any instruction appears to conflict with exact registration, exact registration wins.

Determine the tabletop-supported scene without changing its geometry. Retain another physical subject only when it has a continuous, visually supported gravity-load path that terminates at the tabletop: it either rests directly on the tabletop or rests on retained supporting subjects whose support chain ends at the tabletop. Apparent 2D overlap, proximity, or alignment is not physical support. Remove every visible subject outside this retained support graph. Remove it only by locally editing the pixels occupied by that subject and its cast shadow; do not regenerate, shift, rescale, reshape, or repaint the surrounding retained scene. Do not invent, move, or extend contacts to make an unsupported subject appear supported.

Replace the remaining background with a clean, uniform pure white background (#FFFFFF). Preserve each retained foreground/background boundary at exactly the same pixel location as the input.

All retained foreground content must remain exactly the same as in the original image, including its quantity, position, shape, proportions, pose, angle, color, material, texture, text, logos, intrinsic surface shading, and fine details. Do not add, move, cover, distort, redraw, enhance, retouch, repair, or replace any retained foreground content.

Remove visible cast shadows and shadow remnants from the replaced background and support surfaces as completely as possible. Prefer clean, evenly illuminated surfaces with no directional shadow patches, dark silhouettes, ambient shadow stains, or ghosted shadow boundaries. Preserve only subtle intrinsic shading that is necessary to describe the foreground subjects' three-dimensional shape and material; do not preserve strong cast shadows.

Output the final image at a fixed resolution of 2048 × 1152 pixels in a 16:9 landscape format. Preserve the correct proportions of all original content. Do not stretch or compress any subject. If the original image is not 16:9, scale it proportionally and preserve the full image without cropping; fill any extra canvas area with pure white.

Foreground edges must be precise, natural, and clean, with no white outlines, jagged edges, halos, ghosting, transparent gaps, missing details, or remnants of the original background.

Only the following four changes are allowed:

Remove physical subjects that do not belong to the tabletop-supported scene defined above.
Replace the background with pure white.
Remove or strongly suppress cast shadows while preserving necessary intrinsic object shading.
Export the final image at 2048 × 1152 pixels.

Do not modify anything else in the image.

Return only the requested image.
After generation, reply exactly DONE."""


def real_reference_prompt(context, _output_path=None):
    return REAL_REFERENCE_PROMPT


def _image_size(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"failed to read image dimensions: {result.stderr.strip()}"
        )
    try:
        return tuple(map(int, result.stdout.strip().splitlines()[0].split("x")))
    except (IndexError, ValueError):
        raise RuntimeError(
            f"invalid image dimensions reported for {path}: {result.stdout.strip()}"
        ) from None


def _write_reference_projection(source, reference, output):
    width, height = _image_size(source)
    reference_width, reference_height = _image_size(reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"[1:v]scale={width}:{height}:flags=lanczos,"
        "edgedetect=low=0.08:high=0.20:mode=wires:planes=y,"
        "format=gray,dilation,gblur=sigma=0.3[edge];"
        f"color=c=red:s={width}x{height},format=rgba[red];"
        "[red][edge]alphamerge,colorchannelmixer=aa=0.65[lines];"
        "[0:v][lines]overlay=shortest=1:format=auto[out]"
    )
    log_step(
        "reference_image",
        "writing projection overlay "
        f"{reference_width}x{reference_height} -> {width}x{height}",
    )
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-i", str(reference),
            "-filter_complex", filter_graph,
            "-map", "[out]", "-frames:v", "1", str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"failed to render reference projection: {result.stderr.strip()}"
        )


def run(context, attempt, model=None):
    log_step("reference_image", "creating output directories")
    runtime = image_generation_runtime()
    model = model or runtime.model
    out_path = attempt_path(attempt, "data/reference_image.png")
    log_step("reference_image", f"provider=openai-api, output={out_path}")
    log_step(
        "reference_image",
        f"generating the {REFERENCE_IMAGE_RESOLUTION} pure-white reference in one model call",
    )
    run_image_generation(
        lambda _unused: real_reference_prompt(context),
        model,
        out_path,
        "real_reference_image",
        images=[context.source_image],
        resolution=REFERENCE_IMAGE_RESOLUTION,
        final_size=REFERENCE_IMAGE_RESOLUTION,
        project_root=context.project_root,
        log_dir=attempt_path(attempt, "logs"),
    )
    _write_reference_projection(
        context.source_image,
        out_path,
        attempt_path(attempt, "data/reference_projection_overlay.png"),
    )
    return completed_files(attempt, stage_outputs("reference_image"))
