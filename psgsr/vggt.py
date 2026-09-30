#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VGGT_ROOT = ROOT.parent / "third_party/vggt"
DEFAULT_CHECKPOINT = ROOT / "models/vggt/model.pt"
STATIC_OUTPUT_NAMES = (
    "predictions.npz",
    "cameras.json",
    "point_cloud.ply",
    "point_cloud_preview.png",
    "summary.json",
)


def resolve_path(value, root=ROOT):
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value


def select_inputs(images=None, front=None, top=None):
    if images is not None:
        if front or top:
            raise ValueError("--images cannot be combined with --front or --top")
        if not images:
            raise ValueError("--images requires at least one image")
        return (
            [resolve_path(path) for path in images],
            [f"view_{index:02d}" for index in range(len(images))],
        )
    paths = [front or "data/reference_image.png", top or "data/topview_image.png"]
    return [resolve_path(path) for path in paths], ["front", "top"]


def confidence_mask(confidence, percentile):
    confidence = np.asarray(confidence)
    masks = np.zeros(confidence.shape, dtype=bool)
    thresholds = []
    for index, view in enumerate(confidence):
        finite = view[np.isfinite(view)]
        if not finite.size:
            raise RuntimeError(f"view {index} has no finite confidence values")
        threshold = float(np.percentile(finite, percentile))
        thresholds.append(threshold)
        masks[index] = np.isfinite(view) & (view >= threshold)
    return masks, thresholds


def normalize_depth(depth):
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    for index, view in enumerate(depth):
        valid = np.isfinite(view) & (view > 0)
        if not valid.any():
            continue
        low = float(view[valid].min())
        high = float(view[valid].max())
        if high > low:
            scaled = (view[valid] - low) / (high - low)
            normalized[index][valid] = np.rint(scaled * 255).astype(np.uint8)
    return normalized


def robust_preview_points(points, colors, percentile=1.0, max_points=100000):
    points = np.asarray(points)
    colors = np.asarray(colors)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if not len(points):
        raise ValueError("point cloud has no finite points")
    low, high = np.percentile(points, [percentile, 100.0 - percentile], axis=0)
    inside = ((points >= low) & (points <= high)).all(axis=1)
    if inside.any():
        points = points[inside]
        colors = colors[inside]
    stride = max(1, len(points) // max_points)
    return points[::stride], colors[::stride]


def camera_to_display_points(points):
    points = np.asarray(points)
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


def preview_view_angles(view_count):
    return (0, -90) if view_count == 1 else (24, -62)


def write_binary_ply(output, points, colors):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != (len(points), 3) or colors.shape != (len(points), 3):
        raise ValueError("points and colors must both have shape (N, 3)")
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, channel]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment metric_scale false\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with output.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def build_camera_document(
    inputs, source_sizes, network_size, intrinsic, extrinsic, view_names=None
):
    view_names = view_names or ["front", "top"]
    height, width = (int(value) for value in network_size)
    return {
        "metric_scale": False,
        "coordinate_convention": "OpenCV camera axes: x-right, y-down, z-forward; extrinsics are world-to-camera.",
        "intrinsic_applies_to": "preprocessed network image only, not the original source image",
        "network_image_size": {"width": width, "height": height},
        "preprocessing": {
            "mode": "VGGT load_and_preprocess_images(mode='crop')",
            "raw_to_network_transform_exported": False,
            "warning": "Do not apply these intrinsics directly to original-image pixels.",
        },
        "views": {
            name: {
                "source_path": str(inputs[index]),
                "source_image_size": {
                    "width": int(source_sizes[index][0]),
                    "height": int(source_sizes[index][1]),
                },
                "intrinsic": intrinsic[index].tolist(),
                "extrinsic_world_to_camera": extrinsic[index].tolist(),
            }
            for index, name in enumerate(view_names)
        },
    }


def prepare_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in STATIC_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    for path in output_dir.glob("*_depth.png"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    return output_dir


def load_vggt_model(checkpoint=DEFAULT_CHECKPOINT, vggt_root=DEFAULT_VGGT_ROOT, *, device):
    checkpoint = resolve_path(checkpoint)
    vggt_root = Path(vggt_root).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing VGGT checkpoint: {checkpoint}")
    if not (vggt_root / "vggt/models/vggt.py").is_file():
        raise FileNotFoundError(f"missing VGGT source: {vggt_root}")

    sys.path.insert(0, str(vggt_root))
    import torch
    from vggt.models.vggt import VGGT

    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = selected_device.index if selected_device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index out of range: {index}")

    print(f"[vggt] loading checkpoint {checkpoint}", flush=True)
    model = VGGT(enable_track=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key: value for key, value in state.items() if not key.startswith("track_head.")}
    model.load_state_dict(state)
    return model.eval().to(selected_device)


def build_summary(
    inputs, checkpoint, device, depth_shape, thresholds, point_count, view_names=None
):
    view_names = view_names or ["front", "top"]
    warning = (
        "VGGT single-view geometry is not metric and cannot recover unseen surfaces."
        if len(view_names) == 1
        else "VGGT geometry from generated views is not metric and may reflect cross-view inconsistencies."
    )
    return {
        "views": list(view_names),
        "view_count": len(view_names),
        "inputs": [str(path) for path in inputs],
        "checkpoint": str(checkpoint),
        "device": device,
        "depth_shape": list(depth_shape),
        "confidence_thresholds": [float(value) for value in thresholds],
        "retained_point_count": int(point_count),
        "metric_scale": False,
        "warning": warning,
    }


def save_depth_visualizations(depth, output_dir, view_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = normalize_depth(depth)
    outputs = []
    for index, view_name in enumerate(view_names):
        output = Path(output_dir) / f"{view_name}_depth.png"
        plt.imsave(output, normalized[index], cmap="magma", vmin=0, vmax=255)
        outputs.append(output)
    return outputs


def save_point_cloud_preview(output, points, colors, view_count=2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown_points, shown_colors = robust_preview_points(points, colors)
    shown_points = camera_to_display_points(shown_points)
    shown_colors = np.asarray(shown_colors, dtype=np.float32) / 255.0
    center = np.median(shown_points, axis=0)
    centered = shown_points - center
    limits = np.stack([centered.min(axis=0), centered.max(axis=0)])

    figure = plt.figure(figsize=(10, 8), dpi=120)
    figure.patch.set_facecolor("#111318")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#111318")
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_pane_color((0.09, 0.10, 0.13, 1.0))
    axis.scatter(
        centered[:, 0],
        centered[:, 1],
        centered[:, 2],
        c=shown_colors,
        s=0.25,
        linewidths=0,
        depthshade=False,
    )
    axis.set_xlim(limits[:, 0])
    axis.set_ylim(limits[:, 1])
    axis.set_zlim(limits[:, 2])
    extent = np.maximum(limits[1] - limits[0], 1e-6)
    axis.set_box_aspect(extent)
    axis.set_xlabel("X (right)", color="white")
    axis.set_ylabel("Depth", color="white")
    axis.set_zlabel("Up", color="white")
    axis.tick_params(colors="#c8ccd4")
    title = "VGGT point cloud (single view, non-metric)" if view_count == 1 else "VGGT fused point cloud (non-metric)"
    axis.set_title(title, color="white")
    elev, azim = preview_view_angles(view_count)
    axis.view_init(elev=elev, azim=azim)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def run(
    front=None,
    top=None,
    output_dir="outputs/vggt_scene",
    *,
    device,
    confidence_percentile=50.0,
    checkpoint=DEFAULT_CHECKPOINT,
    vggt_root=DEFAULT_VGGT_ROOT,
    images=None,
    model=None,
):
    inputs, view_names = select_inputs(images, front, top)
    output_dir = resolve_path(output_dir)
    checkpoint = resolve_path(checkpoint)
    vggt_root = Path(vggt_root).resolve()
    for input_path in inputs:
        if not input_path.is_file():
            raise FileNotFoundError(f"missing VGGT input image: {input_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing VGGT checkpoint: {checkpoint}")
    if not (vggt_root / "vggt/models/vggt.py").is_file():
        raise FileNotFoundError(f"missing VGGT source: {vggt_root}")
    if not 0 <= confidence_percentile <= 100:
        raise ValueError("confidence percentile must be in [0, 100]")

    sys.path.insert(0, str(vggt_root))
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from PIL import Image

    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = selected_device.index if selected_device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index out of range: {index}")
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(selected_device)[0] >= 8
            else torch.float16
        )
        autocast = torch.cuda.amp.autocast(dtype=dtype)
    else:
        print("[vggt] warning: CPU inference will be slow", flush=True)
        dtype = torch.float32
        autocast = nullcontext()

    if model is None:
        print(f"[vggt] loading checkpoint {checkpoint}", flush=True)
        model = VGGT(enable_track=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith("track_head.")}
        model.load_state_dict(state)
        del state
        model.eval().to(selected_device)
    else:
        print("[vggt] reusing loaded model", flush=True)

    print(f"[vggt] preprocessing {', '.join(str(path) for path in inputs)}", flush=True)
    images = load_and_preprocess_images([str(path) for path in inputs]).to(selected_device)
    source_sizes = []
    for path in inputs:
        with Image.open(path) as image:
            source_sizes.append(image.size)
    print(f"[vggt] input tensor {tuple(images.shape)}, device={selected_device}, dtype={dtype}", flush=True)
    with torch.no_grad():
        with autocast:
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    depth = predictions["depth"].cpu().numpy().squeeze(0)
    depth_confidence = predictions["depth_conf"].cpu().numpy().squeeze(0)
    point_map = predictions["world_points"].cpu().numpy().squeeze(0)
    point_confidence = predictions["world_points_conf"].cpu().numpy().squeeze(0)
    extrinsic = extrinsic.cpu().numpy().squeeze(0)
    intrinsic = intrinsic.cpu().numpy().squeeze(0)
    colors_by_view = (
        predictions["images"]
        .cpu()
        .numpy()
        .squeeze(0)
        .transpose(0, 2, 3, 1)
    )
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    mask, thresholds = confidence_mask(depth_confidence, confidence_percentile)
    mask &= np.isfinite(world_points).all(axis=-1)
    points = world_points[mask].astype(np.float32)
    colors = np.clip(colors_by_view[mask] * 255.0, 0, 255).astype(np.uint8)
    if not len(points):
        raise RuntimeError("VGGT confidence filtering produced an empty point cloud")

    prepare_output_dir(output_dir)
    np.savez_compressed(
        output_dir / "predictions.npz",
        depth=depth,
        depth_confidence=depth_confidence,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        world_points=world_points,
        point_map=point_map,
        point_confidence=point_confidence,
        retained_mask=mask,
    )
    (output_dir / "cameras.json").write_text(
        json.dumps(
            build_camera_document(
                inputs,
                source_sizes,
                images.shape[-2:],
                intrinsic,
                extrinsic,
                view_names,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_binary_ply(output_dir / "point_cloud.ply", points, colors)
    depth_outputs = save_depth_visualizations(depth, output_dir, view_names)
    save_point_cloud_preview(
        output_dir / "point_cloud_preview.png", points, colors, len(view_names)
    )
    summary = build_summary(
        inputs,
        checkpoint,
        str(selected_device),
        depth.shape,
        thresholds,
        len(points),
        view_names,
    )
    summary.update(
        {
            "input_tensor_shape": list(images.shape),
            "intrinsic_shape": list(intrinsic.shape),
            "extrinsic_shape": list(extrinsic.shape),
            "confidence_percentile": float(confidence_percentile),
            "outputs": list(STATIC_OUTPUT_NAMES[:-1])
            + [path.name for path in depth_outputs]
            + ["summary.json"],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[vggt] retained {len(points):,} points", flush=True)
    print(f"[vggt] wrote {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--front")
    parser.add_argument("--top")
    parser.add_argument("--output-dir", default="outputs/vggt_scene")
    parser.add_argument(
        "--device", default=os.environ.get("VGGT_DEVICE"),
        required="VGGT_DEVICE" not in os.environ,
    )
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vggt-root", default=str(DEFAULT_VGGT_ROOT))
    args = parser.parse_args()
    if args.images and (args.front or args.top):
        parser.error("--images cannot be combined with --front or --top")
    run(
        front=args.front,
        top=args.top,
        output_dir=args.output_dir,
        device=args.device,
        confidence_percentile=args.confidence_percentile,
        checkpoint=args.checkpoint,
        vggt_root=args.vggt_root,
        images=args.images,
    )


if __name__ == "__main__":
    main()
