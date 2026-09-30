from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from .joints import JointEstimate
from .render import primitive_color


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _mesh_preview(
    output: Path,
    title: str,
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    legend: Sequence[tuple[tuple[int, int, int] | None, str]],
    joints: Mapping[int, JointEstimate] | None = None,
    up_axis: str = "Y",
) -> Path:
    if joints:
        raise ValueError("joint overlays are not supported in MLLM evidence rendering")
    labels = np.asarray(face_labels, dtype=int)
    if labels.shape != (len(mesh.faces),):
        raise ValueError("face labels must match mesh faces")
    if up_axis != "Y":
        raise ValueError("MLLM evidence rendering expects a Y-up mesh")

    views = (
        ("FRONT", np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        ("RIGHT", np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([1, 0, 0])),
        ("TOP", np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])),
    )
    panel_size = 480
    canvas = Image.new("RGB", (1520, 650), (24, 27, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 20), title, font=_font(28), fill=(245, 247, 250))
    center = np.asarray(mesh.bounds).mean(axis=0)
    span = max(float(np.ptp(mesh.bounds, axis=0).max()), 1e-9)
    scale = panel_size * 0.78 / span
    centroids = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)

    for index, (name, horizontal, vertical, camera) in enumerate(views):
        pixels = np.full((panel_size, panel_size, 3), (43, 47, 54), dtype=np.uint8)
        relative = centroids - center
        x = np.rint(panel_size / 2 + relative @ horizontal * scale).astype(int)
        y = np.rint(panel_size / 2 - relative @ vertical * scale).astype(int)
        valid = (x >= 0) & (x < panel_size) & (y >= 0) & (y < panel_size)
        selected = np.flatnonzero(valid)
        depth = centroids[selected] @ camera
        order = np.argsort(depth)
        selected = selected[order]
        light = 0.62 + 0.38 * np.abs(normals[selected] @ camera)
        colors = np.asarray([
            primitive_color(int(labels[face]) % 256) for face in selected
        ], dtype=float)
        pixels[y[selected], x[selected]] = np.clip(
            colors * light[:, None], 0, 255,
        ).astype(np.uint8)
        left = 20 + index * 500
        canvas.paste(Image.fromarray(pixels), (left, 70))
        draw.text((left + 10, 78), name, font=_font(18), fill=(245, 247, 250))

    x = 40
    for color, text in legend:
        if color is not None:
            draw.rectangle((x, 580, x + 18, 598), fill=color)
            x += 24
        draw.text((x, 578), text, font=_font(16), fill=(225, 229, 235))
        x += int(draw.textlength(text, font=_font(16))) + 32
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
