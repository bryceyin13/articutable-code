#!/usr/bin/env python3
from pipeline.image_generation import TOPVIEW_IMAGE_RESOLUTION, run_image_generation
from pipeline.common import log_step
from pipeline.stage_io import STAGE_OUTPUTS, completed_files, dependency_file
from pipeline.common import attempt_path, project_path
from pipeline.mllm import image_generation_runtime


def topview_prompt(context, _output_path=None):
    template = project_path(context, "prompts/04_topview_image.txt").read_text(
        encoding="utf-8"
    )
    return (
        template.replace("{{RESOLUTION}}", TOPVIEW_IMAGE_RESOLUTION)
    )


def run(context, attempt, model=None):
    log_step("topview_image", "creating output directories")
    runtime = image_generation_runtime()
    model = model or runtime.model
    front_image = dependency_file(attempt, "reference_image", "image")
    out_path = attempt_path(attempt, "data/topview_image.png")
    log_step("topview_image", f"provider=openai-api, output={out_path}")
    log_step("topview_image", "building top-view prompt from front-view image")
    run_image_generation(
        lambda _unused: topview_prompt(context),
        model,
        out_path,
        "topview_image",
        images=[front_image],
        resolution=TOPVIEW_IMAGE_RESOLUTION,
        final_size=TOPVIEW_IMAGE_RESOLUTION,
        project_root=context.project_root,
        log_dir=attempt_path(attempt, "logs"),
    )
    return completed_files(attempt, STAGE_OUTPUTS["topview_image"])
