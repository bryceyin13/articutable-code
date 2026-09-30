#!/usr/bin/env python3
"""Small silhouette refinement initialized by table point-cloud alignment."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from psgsr.table_alignment import load_metric_mesh_samples, project_world_points
from psgsr.scene_coordinates import load_alignment


ROOT = Path(__file__).resolve().parents[1]
FFMPEG_EDGE_FILTER = (
    "edgedetect=low=0.08:high=0.20:mode=wires:planes=y,"
    "format=gray,dilation,gblur=sigma=0.3"
)


def boundary(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def thin_boundary(mask):
    output = np.zeros_like(np.asarray(mask, dtype=np.uint8))
    contours, _ = cv2.findContours(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(output, contours, -1, 255, 1, cv2.LINE_8)
    return output > 0


def masked_full_frame_teed_edges(full_frame_edges, mask):
    """Keep a full-frame TEED response only inside one instance mask."""
    response = np.asarray(full_frame_edges, dtype=np.uint8)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if response.shape != mask_u8.shape:
        raise ValueError("full-frame TEED response and mask must have the same size")
    interior = cv2.erode(mask_u8, np.ones((3, 3), np.uint8)).astype(bool)
    internal_edges = np.where(interior, response, 0).astype(np.uint8)
    return np.maximum(
        internal_edges, thin_boundary(mask_u8).astype(np.uint8) * 255,
    ), {
        "method": "teed",
        "source": "full_frame_then_instance_mask",
        "postprocess": "none",
    }


def ffmpeg_edgedetect(image):
    """Run the reference-projection FFmpeg edge filter on one RGB image."""
    rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("FFmpeg edgedetect expects an RGB image")
    height, width = rgb.shape[:2]
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-i", "pipe:0",
            "-filter:v", FFMPEG_EDGE_FILTER,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ],
        input=rgb.tobytes(), capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"FFmpeg edgedetect failed: {result.stderr.decode(errors='replace').strip()}"
        )
    response = np.frombuffer(result.stdout, dtype=np.uint8)
    if response.size != width * height:
        raise RuntimeError(
            "FFmpeg edgedetect returned "
            f"{response.size} pixels, expected {width * height}"
        )
    return response.reshape(height, width)


def masked_edges(image, mask, method="canny"):
    image = np.asarray(image)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    if not len(xs):
        return np.zeros(mask_u8.shape, dtype=bool), {"method": method}
    object_size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    padding = max(4, int(round(0.05 * object_size)))
    x0, x1 = max(0, xs.min() - padding), min(mask_u8.shape[1], xs.max() + padding + 1)
    y0, y1 = max(0, ys.min() - padding), min(mask_u8.shape[0], ys.max() + padding + 1)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask_u8[y0:y1, x0:x1]
    object_pixels = crop[crop_mask > 0]
    if method == "teed":
        median_rgb = np.median(object_pixels, axis=0)
        luminance = float(
            0.2126 * median_rgb[0]
            + 0.7152 * median_rgb[1]
            + 0.0722 * median_rgb[2]
        )
        fill = np.full(3, 255 if luminance < 127.5 else 0, dtype=crop.dtype)
    else:
        fill = np.median(object_pixels, axis=0).astype(crop.dtype)
    crop[crop_mask == 0] = fill
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    interior = cv2.erode(crop_mask, np.ones((3, 3), np.uint8)).astype(bool)

    if method == "canny":
        # Preserve the original edge style for every object.  Cropping isolates
        # objects; it must not silently change the detector's spatial scale.
        kernel, sigma = 7, 2.0
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        gradient = cv2.magnitude(
            cv2.Sobel(blurred, cv2.CV_32F, 1, 0),
            cv2.Sobel(blurred, cv2.CV_32F, 0, 1),
        )
        # Estimate the threshold away from the silhouette boundary.  Otherwise
        # the strong boundary spread by the blur removes weaker direction cues.
        threshold_region = cv2.distanceTransform(
            crop_mask, cv2.DIST_L2, 3,
        ) >= float(kernel // 2 + 1)
        if threshold_region.sum() < 16:
            threshold_region = interior
        local_gradient = np.clip(gradient[threshold_region], 0, 255).astype(np.uint8)
        if local_gradient.size and local_gradient.max() > 0:
            high, _ = cv2.threshold(
                local_gradient.reshape(-1, 1), 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            high = max(1, int(round(high)))
        else:
            high = 1
        low = max(1, int(round(0.4 * high)))
        crop_edges = cv2.Canny(blurred, low, high, L2gradient=True) > 0
        metadata = {"method": "canny", "low_threshold": low, "high_threshold": high}
    elif method == "edge_drawing":
        if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createEdgeDrawing"):
            raise RuntimeError(
                "Edge Drawing requires opencv-contrib-python; "
                "install the contrib build matching your OpenCV version"
            )
        detector = cv2.ximgproc.createEdgeDrawing()
        detector.detectEdges(gray)
        crop_edges = detector.getEdgeImage() > 0
        metadata = {"method": "edge_drawing"}
    elif method == "teed":
        from psgsr.edge_detector import teed_edges

        crop_edges, metadata = teed_edges(crop)
    elif method == "ffmpeg_edgedetect":
        crop_edges = ffmpeg_edgedetect(crop)
        metadata = {
            "method": "ffmpeg_edgedetect",
            "filter": FFMPEG_EDGE_FILTER,
        }
    else:
        raise ValueError(f"unknown edge detector: {method}")

    if method in ("teed", "ffmpeg_edgedetect"):
        crop_edges = np.where(interior, crop_edges, 0).astype(np.uint8)
        internal_edges = np.zeros(mask_u8.shape, dtype=np.uint8)
        internal_edges[y0:y1, x0:x1] = crop_edges
        return np.maximum(
            internal_edges, thin_boundary(mask_u8).astype(np.uint8) * 255,
        ), metadata

    crop_edges &= interior
    internal_edges = np.zeros(mask_u8.shape, dtype=bool)
    internal_edges[y0:y1, x0:x1] = crop_edges
    return internal_edges | thin_boundary(mask_u8), metadata


def masked_canny_edges(image, mask):
    edges, metadata = masked_edges(image, mask, method="canny")
    return edges, (metadata.get("low_threshold", 1), metadata.get("high_threshold", 1))


def solid_external_silhouette(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 255, cv2.FILLED)
    return solid


def symmetric_contour_loss(rendered, target, ignore=None):
    rendered_edge = boundary(rendered)
    target_edge = boundary(target)
    if ignore is not None:
        rendered_edge &= ~ignore
        target_edge &= ~ignore
    if not rendered_edge.any() or not target_edge.any():
        return float("inf")
    target_distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_distance = cv2.distanceTransform((~rendered_edge).astype(np.uint8), cv2.DIST_L2, 3)
    return float(target_distance[rendered_edge].mean() + rendered_distance[target_edge].mean())


def camera_geometry(alignment, extrinsic):
    scale = float(alignment["metric_scale_world_per_vggt"])
    rotation = np.asarray(alignment["rotation_world_from_vggt"])
    translation = np.asarray(alignment["translation_world"])
    camera_rotation = extrinsic[:, :3]
    center_vggt = -camera_rotation.T @ extrinsic[:, 3]
    center_world = center_vggt @ rotation.T * scale + translation
    world_from_camera = rotation @ camera_rotation.T
    forward = world_from_camera[:, 2]
    target = center_world - forward * (center_world[2] / forward[2])
    distance = float(np.linalg.norm(target - center_world))
    elevation = float(np.degrees(np.arcsin(-forward[2])))
    return center_vggt, target, distance, elevation


def alignment_for_camera(alignment, extrinsic, elevation_deg, distance_scale):
    scale = float(alignment["metric_scale_world_per_vggt"])
    camera_rotation = extrinsic[:, :3]
    center_vggt, target, initial_distance, _ = camera_geometry(alignment, extrinsic)
    current_center = center_vggt @ np.asarray(alignment["rotation_world_from_vggt"]).T * scale + np.asarray(alignment["translation_world"])
    horizontal = target - current_center
    horizontal[2] = 0
    horizontal /= np.linalg.norm(horizontal)
    elevation = np.radians(elevation_deg)
    forward = horizontal * np.cos(elevation) + np.array([0, 0, -np.sin(elevation)])
    right = np.cross(horizontal, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world_from_camera = np.column_stack((right, down, forward))
    rotation = world_from_camera @ camera_rotation
    center_world = target - forward * initial_distance * distance_scale
    translation = center_world - center_vggt @ rotation.T * scale
    result = dict(alignment)
    result["rotation_world_from_vggt"] = rotation.tolist()
    result["translation_world"] = translation.tolist()
    return result


def render_silhouette(mesh, alignment, intrinsic, extrinsic, image_size):
    height, width = image_size
    pixels, depth = project_world_points(
        mesh.vertices,
        alignment["metric_scale_world_per_vggt"],
        np.asarray(alignment["rotation_world_from_vggt"]),
        np.asarray(alignment["translation_world"]),
        intrinsic,
        extrinsic,
    )
    valid = np.all(depth[mesh.faces] > 0, axis=1)
    polygons = np.rint(pixels[mesh.faces[valid]]).astype(np.int32)
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, polygons, 255)
    return solid_external_silhouette(output)


def optimize_camera(mesh, target, ignore, alignment, intrinsic, extrinsic):
    image_size = target.shape
    _, _, _, initial_elevation = camera_geometry(alignment, extrinsic)

    def score(parameters):
        candidate = alignment_for_camera(alignment, extrinsic, parameters[0], parameters[1])
        silhouette = render_silhouette(mesh, candidate, intrinsic, extrinsic, image_size)
        data_loss = symmetric_contour_loss(silhouette, target, ignore)
        regularization = 0.01 * (parameters[0] - initial_elevation) ** 2 + 2.0 * (parameters[1] - 1.0) ** 2
        return data_loss + regularization

    parameters = np.array([initial_elevation, 1.0])
    best = score(parameters)
    for steps in (np.array([2.0, 0.05]), np.array([0.5, 0.01]), np.array([0.1, 0.002])):
        improved = True
        while improved:
            improved = False
            for index in range(2):
                for direction in (-1, 1):
                    candidate = parameters.copy()
                    candidate[index] += direction * steps[index]
                    if not 20 <= candidate[0] <= 55 or not 0.7 <= candidate[1] <= 1.3:
                        continue
                    value = score(candidate)
                    if value + 1e-6 < best:
                        parameters, best, improved = candidate, value, True
    return initial_elevation, parameters[0], parameters[1], best


def overlay(source, rendered, ignore, color):
    image = np.asarray(source).copy()
    rendered_edge = boundary(rendered)
    rendered_edge &= ~ignore
    image[rendered_edge] = color
    return Image.fromarray(image)


def run(table_alignment, predictions, cameras, output_dir):
    table_alignment = Path(table_alignment)
    predictions = Path(predictions)
    cameras = Path(cameras)
    output_dir = Path(output_dir)
    blueprint = json.loads((ROOT / "data/blueprint.json").read_text(encoding="utf-8"))
    alignment = load_alignment(table_alignment)
    camera_document = json.loads(cameras.read_text(encoding="utf-8"))
    view = camera_document["views"]["front"]
    with np.load(predictions) as data:
        intrinsic = data["intrinsic"][0]
        extrinsic = data["extrinsic"][0]
        height, width = data["world_points"].shape[1:3]
    table, *_ = load_metric_mesh_samples(
        ROOT / "trellis2_outputs/table_0/right_45_100k/table_0.glb",
        np.asarray(blueprint["table"]["size_cm"], dtype=float) / 100,
        1000,
        0,
    )
    if len(table.faces) > 25000:
        table = table.simplify_quadric_decimation(face_count=25000)
    target = np.asarray(
        Image.open(ROOT / "masks/table_0_mask.png").convert("L").resize((width, height), Image.Resampling.NEAREST)
    )
    target = np.where(target > 127, 255, 0).astype(np.uint8)
    target = solid_external_silhouette(target)
    ignore = np.zeros((height, width), dtype=np.uint8)
    for mask_path in sorted((ROOT / "masks").glob("*_0_mask.png")):
        if mask_path.name == "table_0_mask.png" or mask_path.name.startswith("topview_"):
            continue
        object_mask = np.asarray(
            Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        )
        ignore |= np.where(object_mask > 127, 255, 0).astype(np.uint8)
    ignore = cv2.dilate(ignore, np.ones((5, 5), np.uint8)) > 0
    before = render_silhouette(table, alignment, intrinsic, extrinsic, (height, width))
    before_loss = symmetric_contour_loss(before, target, ignore)
    initial_elevation, refined_elevation, distance_scale, objective = optimize_camera(
        table, target, ignore, alignment, intrinsic, extrinsic
    )
    refined = alignment_for_camera(alignment, extrinsic, refined_elevation, distance_scale)
    after = render_silhouette(table, refined, intrinsic, extrinsic, (height, width))
    after_loss = symmetric_contour_loss(after, target, ignore)
    source = Image.open(view["source_path"]).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    panels = [source, overlay(source, before, ignore, (255, 120, 20)), overlay(source, after, ignore, (80, 255, 80))]
    comparison = Image.new("RGB", (width * 3 + 8, height), "#202020")
    for index, panel in enumerate(panels):
        comparison.paste(panel, (index * (width + 4), 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.resize((comparison.width * 2, comparison.height * 2), Image.Resampling.LANCZOS).save(
        output_dir / "mask_refinement_comparison.png"
    )
    result = {
        "format": "table_mask_refinement_v1",
        "initial_alignment": str(table_alignment),
        "camera_elevation_initial_deg": initial_elevation,
        "camera_elevation_refined_deg": refined_elevation,
        "camera_distance_scale": distance_scale,
        "contour_loss_before_px": before_loss,
        "contour_loss_after_px": after_loss,
        "regularized_objective": objective,
        "refined_rotation_world_from_vggt": refined["rotation_world_from_vggt"],
        "refined_translation_world": refined["translation_world"],
    }
    refined_manifest = dict(alignment)
    refined_rotation = np.asarray(refined["rotation_world_from_vggt"])
    refined_translation = np.asarray(refined["translation_world"])
    matrix = np.eye(4)
    matrix[:3, :3] = alignment["metric_scale_world_per_vggt"] * refined_rotation
    matrix[:3, 3] = refined_translation
    camera_rotation = extrinsic[:, :3]
    center_vggt = -camera_rotation.T @ extrinsic[:, 3]
    refined_manifest["rotation_world_from_vggt"] = refined_rotation.tolist()
    refined_manifest["translation_world"] = refined_translation.tolist()
    refined_manifest["similarity_matrix_world_from_vggt"] = matrix.tolist()
    refined_manifest["rotation_determinant"] = float(np.linalg.det(refined_rotation))
    refined_manifest["camera_center_world_m"] = (
        center_vggt @ refined_rotation.T * alignment["metric_scale_world_per_vggt"]
        + refined_translation
    ).tolist()
    refined_manifest["mask_refinement"] = {
        "source": str(output_dir / "table_mask_refinement.json"),
        "camera_elevation_initial_deg": initial_elevation,
        "camera_elevation_refined_deg": refined_elevation,
        "camera_distance_scale": distance_scale,
    }
    (output_dir / "table_alignment_refined.json").write_text(
        json.dumps(refined_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "table_mask_refinement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-alignment", default=str(ROOT / "outputs/table_alignment/table_alignment.json"))
    parser.add_argument("--predictions", default=str(ROOT / "outputs/vggt_front_only/predictions.npz"))
    parser.add_argument("--cameras", default=str(ROOT / "outputs/vggt_front_only/cameras.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/table_mask_refinement"))
    run(**vars(parser.parse_args()))
