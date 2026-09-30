#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "outputs/vggt_front_only/predictions.npz"
DEFAULT_CAMERAS = ROOT / "outputs/vggt_front_only/cameras.json"
DEFAULT_MASK = ROOT / "masks/table_0_mask.png"
DEFAULT_MESH = ROOT / "trellis2_outputs/table_0/right_45_100k/table_0.glb"
DEFAULT_BLUEPRINT = ROOT / "data/blueprint.json"
DEFAULT_OUTPUT = ROOT / "outputs/table_alignment"
OUTPUT_NAMES = (
    "observed_table_world.ply",
    "sampled_table_mesh.ply",
    "alignment_preview.png",
    "image_overlay.png",
    "table_alignment.json",
)
GLTF_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
)


def resolve_path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def gltf_y_up_to_z_up(points):
    return np.asarray(points) @ GLTF_TO_Z_UP.T


def fit_plane_ransac(points, threshold, iterations=1000, seed=0):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape (N, 3) with N >= 3")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < 3:
        raise RuntimeError("RANSAC could not find a plane")
    inliers = best[1]
    center = points[inliers].mean(axis=0)
    _, _, vectors = np.linalg.svd(points[inliers] - center, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    inliers = np.abs(points @ normal + offset) <= threshold
    return normal, offset, inliers


def table_frame_from_plane(normal, center, camera_center):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    center = np.asarray(center, dtype=float)
    camera_center = np.asarray(camera_center, dtype=float)
    if normal @ (camera_center - center) < 0:
        normal = -normal
    forward = np.array([0.0, 0.0, 1.0])
    world_y_in_vggt = forward - (forward @ normal) * normal
    length = np.linalg.norm(world_y_in_vggt)
    if length < 1e-6:
        raise RuntimeError("camera forward is nearly normal to the tabletop")
    world_y_in_vggt /= length
    world_x_in_vggt = np.cross(world_y_in_vggt, normal)
    world_x_in_vggt /= np.linalg.norm(world_x_in_vggt)
    if world_x_in_vggt @ np.array([1.0, 0.0, 0.0]) < 0:
        world_x_in_vggt = -world_x_in_vggt
        world_y_in_vggt = -world_y_in_vggt
    rotation = np.stack((world_x_in_vggt, world_y_in_vggt, normal))
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("table frame is not a proper right-handed rotation")
    return rotation, normal


def apply_similarity(points, scale, rotation, translation):
    return np.asarray(points) @ np.asarray(rotation).T * float(scale) + np.asarray(
        translation
    )


def project_world_points(
    points_world, scale, rotation, translation, intrinsic, extrinsic
):
    points_vggt = (np.asarray(points_world) - translation) @ rotation / scale
    camera = points_vggt @ extrinsic[:, :3].T + extrinsic[:, 3]
    pixels = np.column_stack(
        (
            intrinsic[0, 0] * camera[:, 0] / camera[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * camera[:, 1] / camera[:, 2] + intrinsic[1, 2],
        )
    )
    return pixels, camera[:, 2]


def solve_yaw_translation_step(source, target, target_normals, weights):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    normals = np.asarray(target_normals, dtype=float)
    weights = np.sqrt(np.asarray(weights, dtype=float))
    yaw_column = -normals[:, 0] * source[:, 1] + normals[:, 1] * source[:, 0]
    matrix = np.column_stack((yaw_column, normals))
    rhs = np.einsum("ij,ij->i", normals, target - source)
    return np.linalg.lstsq(matrix * weights[:, None], rhs * weights, rcond=None)[0]


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_x(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )


def trimmed_mean(values, keep_fraction=0.9):
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(len(values) * keep_fraction))
    return float(values[:count].mean())


def metric_scale_from_tabletop(observed_top, metric_mesh_top, percentile=1.0):
    observed_limits = np.percentile(
        np.asarray(observed_top)[:, 0], [percentile, 100.0 - percentile]
    )
    mesh_limits = np.percentile(
        np.asarray(metric_mesh_top)[:, 0], [percentile, 100.0 - percentile]
    )
    observed_width = float(np.diff(observed_limits)[0])
    mesh_width = float(np.diff(mesh_limits)[0])
    if observed_width <= 0 or mesh_width <= 0:
        raise RuntimeError("tabletop X width is degenerate")
    return mesh_width / observed_width


def uniform_mesh_scale(native_extents, target_size):
    ratios = np.asarray(target_size, dtype=float)[:2] / np.asarray(
        native_extents, dtype=float
    )[:2]
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise RuntimeError("mesh XY extents are degenerate")
    return float(ratios.mean())


def blueprint_tabletop_origin(support_points):
    support_points = np.asarray(support_points, dtype=float)
    low = support_points.min(axis=0)
    high = support_points.max(axis=0)
    return np.array(
        [(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, np.median(support_points[:, 2])]
    )


def write_binary_ply(output, points, color):
    output = Path(output)
    points = np.asarray(points, dtype=np.float32)
    colors = np.broadcast_to(np.asarray(color, dtype=np.uint8), points.shape)
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("r", "g", "b")):
        vertices[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with output.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def load_metric_mesh_samples(mesh_path, target_size, sample_count, seed):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry()
    vertices = gltf_y_up_to_z_up(mesh.vertices)
    native_extents = np.ptp(vertices, axis=0)
    if np.any(native_extents < 1e-8):
        raise RuntimeError(f"degenerate mesh extents: {native_extents.tolist()}")
    mesh_scale = uniform_mesh_scale(native_extents, target_size)
    metric_mesh = trimesh.Trimesh(
        vertices=vertices * mesh_scale, faces=mesh.faces, process=False
    )
    points, face_indices = trimesh.sample.sample_surface(
        metric_mesh, sample_count, seed=seed
    )
    normals = metric_mesh.face_normals[face_indices]
    metric_extents = native_extents * mesh_scale
    top_cut = points[:, 2].max() - max(0.01, 0.04 * metric_extents[2])
    top = (points[:, 2] >= top_cut) & (normals[:, 2] > 0.7)
    if top.sum() < 100:
        raise RuntimeError("could not identify the mesh top support surface")
    support_center = blueprint_tabletop_origin(points[top])
    points -= support_center
    metric_mesh.vertices -= support_center
    return metric_mesh, points, normals, top, native_extents, mesh_scale, metric_extents


def visible_mesh_samples(
    points_world, normals_world, intrinsic, extrinsic, image_size, scale, rotation, translation
):
    points_vggt = (points_world - translation) @ rotation / scale
    camera = points_vggt @ extrinsic[:, :3].T + extrinsic[:, 3]
    depth = camera[:, 2]
    u = intrinsic[0, 0] * camera[:, 0] / depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / depth + intrinsic[1, 2]
    height, width = image_size
    x = np.rint(u).astype(int)
    y = np.rint(v).astype(int)
    valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = np.flatnonzero(valid)
    pixels = y[valid] * width + x[valid]
    order = np.lexsort((depth[valid], pixels))
    ordered_pixels = pixels[order]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    visible = indices[order[first]]
    return points_world[visible], normals_world[visible]


def robust_icp(
    observed_vggt,
    mesh_points,
    mesh_normals,
    intrinsic,
    extrinsic,
    image_size,
    scale,
    rotation,
    translation,
):
    from scipy.spatial import cKDTree

    history = []
    for max_distance in (0.10, 0.05, 0.025):
        for _ in range(8):
            visible_points, visible_normals = visible_mesh_samples(
                mesh_points,
                mesh_normals,
                intrinsic,
                extrinsic,
                image_size,
                scale,
                rotation,
                translation,
            )
            observed_world = apply_similarity(
                observed_vggt, scale, rotation, translation
            )
            distances, indices = cKDTree(visible_points).query(observed_world, workers=-1)
            keep = distances < max_distance
            if keep.sum() < 100:
                raise RuntimeError(
                    f"ICP has too few correspondences at {max_distance}m: {keep.sum()}"
                )
            source = observed_world[keep]
            target = visible_points[indices[keep]]
            normals = visible_normals[indices[keep]]
            residual = np.abs(np.einsum("ij,ij->i", normals, target - source))
            huber = max_distance * 0.25
            weights = np.minimum(1.0, huber / np.maximum(residual, 1e-8))
            update = solve_yaw_translation_step(source, target, normals, weights)
            update[0] = np.clip(update[0], -0.05, 0.05)
            update[1:] = np.clip(update[1:], -max_distance * 0.5, max_distance * 0.5)
            delta_rotation = rotation_z(update[0])
            rotation = delta_rotation @ rotation
            translation = delta_rotation @ translation + update[1:]
            history.append(
                {
                    "max_distance_m": max_distance,
                    "correspondences": int(keep.sum()),
                    "median_distance_m": float(np.median(distances[keep])),
                    "yaw_update_deg": float(np.degrees(update[0])),
                    "translation_update_m": update[1:].tolist(),
                }
            )
            if abs(update[0]) < 1e-5 and np.linalg.norm(update[1:]) < 1e-5:
                break
    observed_world = apply_similarity(observed_vggt, scale, rotation, translation)
    visible_points, visible_normals = visible_mesh_samples(
        mesh_points,
        mesh_normals,
        intrinsic,
        extrinsic,
        image_size,
        scale,
        rotation,
        translation,
    )
    distances, _ = cKDTree(visible_points).query(observed_world, workers=-1)
    return rotation, translation, observed_world, visible_points, visible_normals, distances, history


def refine_pitch_by_overall_error(
    observed_vggt,
    mesh_points,
    mesh_normals,
    intrinsic,
    extrinsic,
    image_size,
    scale,
    rotation,
    translation,
):
    from scipy.spatial import cKDTree

    candidates = []
    for angle_deg in np.arange(-0.5, 0.5001, 0.05):
        delta = rotation_x(np.radians(angle_deg))
        candidate_rotation = delta @ rotation
        candidate_translation = delta @ translation
        visible_points, visible_normals = visible_mesh_samples(
            mesh_points,
            mesh_normals,
            intrinsic,
            extrinsic,
            image_size,
            scale,
            candidate_rotation,
            candidate_translation,
        )
        observed_world = apply_similarity(
            observed_vggt, scale, candidate_rotation, candidate_translation
        )
        distances, _ = cKDTree(visible_points).query(observed_world, workers=-1)
        candidates.append(
            (
                trimmed_mean(distances, 0.9),
                float(angle_deg),
                candidate_rotation,
                candidate_translation,
                observed_world,
                visible_points,
                visible_normals,
                distances,
            )
        )
    return min(candidates, key=lambda item: item[0])


def save_preview(output, observed, metric_mesh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(10, 8), dpi=130)
    figure.patch.set_facecolor("#101319")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#101319")
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_pane_color((0.06, 0.075, 0.10, 1.0))
    obs_stride = max(1, len(observed) // 30000)
    preview_mesh = (
        metric_mesh.simplify_quadric_decimation(face_count=30000)
        if len(metric_mesh.faces) > 30000
        else metric_mesh
    )
    triangles = preview_mesh.vertices[preview_mesh.faces]
    axis.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor="#c87922",
            edgecolor="none",
            alpha=1.0,
            zsort="average",
        )
    )
    axis.scatter(
        observed[::obs_stride, 0],
        observed[::obs_stride, 1],
        observed[::obs_stride, 2],
        s=0.55,
        c="#36c8ff",
        alpha=1.0,
        linewidths=0,
    )
    combined = np.vstack((observed, metric_mesh.bounds))
    low, high = np.percentile(combined, [1, 99], axis=0)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_box_aspect(np.maximum(high - low, 1e-6))
    axis.set_xlabel("X", color="white")
    axis.set_ylabel("Y", color="white")
    axis.set_zlabel("Z", color="white")
    axis.tick_params(colors="#ccd2dc")
    axis.set_title("Table alignment: mesh vs VGGT", color="white")
    axis.legend(
        handles=[
            Patch(facecolor="#c87922", label="metric mesh"),
            Line2D(
                [],
                [],
                marker=".",
                linestyle="none",
                color="#36c8ff",
                label="VGGT table",
            ),
        ],
        facecolor="#20242c",
        labelcolor="white",
        loc="upper right",
    )
    axis.view_init(elev=24, azim=-62)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_image_overlay(
    output,
    source_image,
    table_mask,
    image_size,
    metric_mesh,
    scale,
    rotation,
    translation,
    intrinsic,
    extrinsic,
):
    from PIL import Image, ImageDraw, ImageFilter

    height, width = image_size
    background = Image.open(source_image).convert("RGB").resize(
        (width, height), Image.Resampling.BICUBIC
    )
    preview_mesh = (
        metric_mesh.simplify_quadric_decimation(face_count=30000)
        if len(metric_mesh.faces) > 30000
        else metric_mesh
    )
    pixels, depth = project_world_points(
        preview_mesh.vertices,
        scale,
        rotation,
        translation,
        intrinsic,
        extrinsic,
    )
    mesh_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mesh_mask)
    valid_faces = preview_mesh.faces[np.all(depth[preview_mesh.faces] > 0, axis=1)]
    for face in valid_faces:
        draw.polygon([tuple(value) for value in pixels[face]], fill=255)

    fill = Image.new("RGBA", (width, height), (224, 116, 25, 0))
    fill.putalpha(mesh_mask.point(lambda value: 72 if value else 0))
    composed = Image.alpha_composite(background.convert("RGBA"), fill)
    mesh_edge = mesh_mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    mesh_line = Image.new("RGBA", (width, height), (255, 120, 20, 0))
    mesh_line.putalpha(mesh_edge)
    composed = Image.alpha_composite(composed, mesh_line)

    sam = Image.open(table_mask).convert("L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    sam_edge = sam.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    sam_line = Image.new("RGBA", (width, height), (20, 220, 255, 0))
    sam_line.putalpha(sam_edge)
    composed = Image.alpha_composite(composed, sam_line)
    composed.resize((width * 2, height * 2), Image.Resampling.NEAREST).save(output)


def run(
    predictions=DEFAULT_PREDICTIONS,
    cameras=DEFAULT_CAMERAS,
    mask=DEFAULT_MASK,
    mesh=DEFAULT_MESH,
    blueprint=DEFAULT_BLUEPRINT,
    output_dir=DEFAULT_OUTPUT,
    samples=100000,
    seed=0,
):
    from PIL import Image

    predictions = resolve_path(predictions)
    cameras = resolve_path(cameras)
    mask = resolve_path(mask)
    mesh = resolve_path(mesh)
    blueprint = resolve_path(blueprint)
    output_dir = resolve_path(output_dir)
    for path in (predictions, cameras, mask, mesh, blueprint):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(predictions) as data:
        world_points = data["world_points"][0]
        retained = data["retained_mask"][0]
        intrinsic = data["intrinsic"][0]
        extrinsic = data["extrinsic"][0]
    height, width = world_points.shape[:2]
    resized_mask = np.asarray(
        Image.open(mask)
        .convert("L")
        .resize((width, height), Image.Resampling.NEAREST)
    ) > 127
    valid = resized_mask & retained & np.isfinite(world_points).all(axis=-1)
    observed = world_points[valid]
    if len(observed) < 1000:
        raise RuntimeError(f"too few masked VGGT table points: {len(observed)}")

    camera_document = json.loads(cameras.read_text(encoding="utf-8"))
    camera_view = next(iter(camera_document["views"].values()))
    source_image = Path(camera_view["source_path"])
    camera_extrinsic = np.asarray(camera_view["extrinsic_world_to_camera"])
    camera_center = -camera_extrinsic[:, :3].T @ camera_extrinsic[:, 3]
    blueprint_data = json.loads(blueprint.read_text(encoding="utf-8"))
    blueprint_origin = blueprint_data["coordinate_system"]["origin"]
    if blueprint_origin != "center of tabletop":
        raise RuntimeError(f"unsupported blueprint origin policy: {blueprint_origin}")
    target_size = np.asarray(blueprint_data["table"]["size_cm"], dtype=float) / 100.0

    span = np.linalg.norm(
        np.percentile(observed, 95, axis=0) - np.percentile(observed, 5, axis=0)
    )
    plane_threshold = 0.005 * span
    plane_normal, plane_offset, plane_inliers = fit_plane_ransac(
        observed, plane_threshold, iterations=1500, seed=seed
    )
    if plane_inliers.mean() < 0.25:
        raise RuntimeError(f"tabletop plane support is too small: {plane_inliers.mean():.3f}")
    plane_center = np.median(observed[plane_inliers], axis=0)
    initial_rotation, oriented_normal = table_frame_from_plane(
        plane_normal, plane_center, camera_center
    )

    (
        metric_mesh,
        mesh_points,
        mesh_normals,
        mesh_top,
        native_extents,
        mesh_scale,
        metric_mesh_extents,
    ) = load_metric_mesh_samples(mesh, target_size, samples, seed)
    observed_top_local = apply_similarity(
        observed[plane_inliers], 1.0, initial_rotation, -initial_rotation @ plane_center
    )
    metric_scale = metric_scale_from_tabletop(
        observed_top_local, mesh_points[mesh_top]
    )
    initial_translation = -metric_scale * (initial_rotation @ plane_center)

    (
        final_rotation,
        final_translation,
        observed_world,
        visible_mesh,
        visible_normals,
        final_distances,
        icp_history,
    ) = robust_icp(
        observed,
        mesh_points,
        mesh_normals,
        intrinsic,
        extrinsic,
        (height, width),
        metric_scale,
        initial_rotation,
        initial_translation,
    )
    (
        pitch_score,
        pitch_deg,
        final_rotation,
        final_translation,
        observed_world,
        visible_mesh,
        visible_normals,
        final_distances,
    ) = refine_pitch_by_overall_error(
        observed,
        mesh_points,
        mesh_normals,
        intrinsic,
        extrinsic,
        (height, width),
        metric_scale,
        final_rotation,
        final_translation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    write_binary_ply(output_dir / "observed_table_world.ply", observed_world, (70, 170, 255))
    write_binary_ply(output_dir / "sampled_table_mesh.ply", mesh_points, (245, 160, 65))
    save_preview(output_dir / "alignment_preview.png", observed_world, metric_mesh)
    save_image_overlay(
        output_dir / "image_overlay.png",
        source_image,
        mask,
        (height, width),
        metric_mesh,
        metric_scale,
        final_rotation,
        final_translation,
        intrinsic,
        extrinsic,
    )

    camera_center_world = apply_similarity(
        camera_center[None], metric_scale, final_rotation, final_translation
    )[0]
    observed_top_world = observed_world[plane_inliers]
    observed_top_extent = np.diff(
        np.percentile(observed_top_world, [1, 99], axis=0), axis=0
    )[0]
    mesh_top_extent = np.diff(
        np.percentile(mesh_points[mesh_top], [1, 99], axis=0), axis=0
    )[0]
    similarity_matrix = np.eye(4)
    similarity_matrix[:3, :3] = metric_scale * final_rotation
    similarity_matrix[:3, 3] = final_translation
    result = {
        "format": "table_alignment_v1",
        "unit": "meter",
        "up_axis": "Z",
        "coordinate_convention": {
            "origin": blueprint_origin,
            "origin_source": "data/blueprint.json:coordinate_system.origin",
            "origin_method": "center of tabletop support bounds at Z=0",
            "x_axis": "tabletop direction consistent with VGGT camera-right",
            "y_axis": "VGGT camera-forward projected onto the tabletop",
            "z_axis": "tabletop normal toward the camera/up side",
            "right_handed": True,
        },
        "metric_scale_world_per_vggt": float(metric_scale),
        "rotation_world_from_vggt": final_rotation.tolist(),
        "translation_world": final_translation.tolist(),
        "similarity_matrix_world_from_vggt": similarity_matrix.tolist(),
        "rotation_determinant": float(np.linalg.det(final_rotation)),
        "camera_center_world_m": camera_center_world.tolist(),
        "table_size_m": metric_mesh_extents.tolist(),
        "blueprint_table_size_m": target_size.tolist(),
        "mesh_native_extents_y_up": native_extents.tolist(),
        "mesh_uniform_scale": float(mesh_scale),
        "tabletop_plane_vggt": {
            "normal": oriented_normal.tolist(),
            "offset": float(-oriented_normal @ plane_center),
            "threshold": float(plane_threshold),
            "inlier_count": int(plane_inliers.sum()),
            "inlier_ratio": float(plane_inliers.mean()),
        },
        "counts": {
            "observed_table_points": int(len(observed)),
            "sampled_mesh_points": int(len(mesh_points)),
            "visible_mesh_points": int(len(visible_mesh)),
        },
        "alignment": {
            "median_nearest_distance_m": float(np.median(final_distances)),
            "p90_nearest_distance_m": float(np.percentile(final_distances, 90)),
            "tabletop_median_z_m": float(np.median(observed_top_world[:, 2])),
            "tabletop_p90_abs_z_m": float(
                np.percentile(np.abs(observed_top_world[:, 2]), 90)
            ),
            "observed_tabletop_extent_p01_p99_m": observed_top_extent.tolist(),
            "mesh_tabletop_extent_p01_p99_m": mesh_top_extent.tolist(),
            "pitch_refinement_deg": float(pitch_deg),
            "pitch_trimmed_mean_m": float(pitch_score),
            "iterations": icp_history,
        },
        "inputs": {
            "predictions": str(predictions),
            "cameras": str(cameras),
            "mask": str(mask),
            "mesh": str(mesh),
            "blueprint": str(blueprint),
        },
        "outputs": list(OUTPUT_NAMES),
    }
    (output_dir / "table_alignment.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[table_alignment] observed={len(observed):,} plane={plane_inliers.sum():,} "
        f"scale={metric_scale:.6f} median={np.median(final_distances):.4f}m",
        flush=True,
    )
    print(f"[table_alignment] wrote {output_dir}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--cameras", default=str(DEFAULT_CAMERAS))
    parser.add_argument("--mask", default=str(DEFAULT_MASK))
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    parser.add_argument("--blueprint", default=str(DEFAULT_BLUEPRINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
