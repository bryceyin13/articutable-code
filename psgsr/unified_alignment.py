#!/usr/bin/env python3
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from psgsr.table_alignment import fit_plane_ransac, gltf_y_up_to_z_up, project_world_points, rotation_z, table_frame_from_plane
from psgsr.mask_refinement import solid_external_silhouette, symmetric_contour_loss


ROOT = Path(__file__).resolve().parents[1]


def vggt_crop_transform(raw_size, target_width=518, patch_size=14):
    raw_width, raw_height = raw_size
    resized_height = round(raw_height * target_width / raw_width / patch_size) * patch_size
    crop_y = max(0, (resized_height - target_width) // 2)
    network_height = min(resized_height, target_width)
    return {
        "raw_size": [int(raw_width), int(raw_height)],
        "resized_size": [int(target_width), int(resized_height)],
        "network_size": [int(target_width), int(network_height)],
        "crop_xy": [0, int(crop_y)],
        "scale_xy": [target_width / raw_width, resized_height / raw_height],
    }


def network_intrinsic_to_raw(intrinsic, transform):
    intrinsic = np.asarray(intrinsic, dtype=float).copy()
    scale_x, scale_y = transform["scale_xy"]
    crop_x, crop_y = transform["crop_xy"]
    intrinsic[0, 0] /= scale_x
    intrinsic[1, 1] /= scale_y
    intrinsic[0, 2] = (intrinsic[0, 2] + crop_x) / scale_x
    intrinsic[1, 2] = (intrinsic[1, 2] + crop_y) / scale_y
    return intrinsic


def preprocess_label_crop(labels, transform):
    resized = Image.fromarray(labels).resize(tuple(transform["resized_size"]), Image.Resampling.NEAREST)
    crop_x, crop_y = transform["crop_xy"]
    width, height = transform["network_size"]
    return np.asarray(resized)[crop_y : crop_y + height, crop_x : crop_x + width]


def generated_rigid_asset(object_id, root=ROOT):
    root = Path(root)
    if object_id == "mouse_0":
        conditioned = root / "trellis2_condition_outputs" / object_id / f"{object_id}.glb"
        if conditioned.exists():
            return conditioned
    return root / "trellis2_outputs" / object_id / "right_45_100k" / f"{object_id}.glb"


def repaired_articulation(package, object_blueprint):
    package = Path(package)
    robot = ET.parse(package / "model.urdf").getroot()
    joints = robot.findall("joint")
    if not joints or any(
        joint.attrib.get("type") not in {"revolute", "prismatic"} for joint in joints
    ):
        raise ValueError("alignment requires revolute or prismatic joints")
    state_path = package / "manifest.json"
    package_states = {
        item["name"]: item
        for item in (
            json.loads(state_path.read_text(encoding="utf-8")).get("joint_states", [])
            if state_path.is_file() else []
        )
    }
    blueprint_articulation = object_blueprint.get("articulation") or {}
    blueprint_joints = blueprint_articulation.get("joints")
    if blueprint_joints is None:
        blueprint_joints = (
            [blueprint_articulation["joint"]]
            if blueprint_articulation.get("joint") else []
        )
    links = []
    for link in robot.findall("link"):
        mesh = link.find("./visual/geometry/mesh")
        if mesh is None:
            raise ValueError(f"articulated link {link.attrib.get('name')} has no visual mesh")
        relative = Path(mesh.attrib["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"articulated visual mesh escapes package: {relative}")
        links.append({
            "name": link.attrib["name"],
            "geometry_asset": str(package / relative),
        })
    joint_metadata = []
    for index, joint in enumerate(joints):
        joint_type = joint.attrib["type"]
        blueprint_joint = next((
            item for item in blueprint_joints
            if (item.get("urdf_joint") or item.get("joint_name")) == joint.attrib["name"]
        ), blueprint_joints[index] if len(blueprint_joints) == len(joints) else {})

        def vector(element, field, default):
            node = joint.find(element)
            return [float(value) for value in (
                node.attrib.get(field, default) if node is not None else default
            ).split()]

        limit = joint.find("limit")
        limits = [
            float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
            float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
        ]
        metadata = {
            "name": joint.attrib["name"],
            "type": joint_type,
            "parent": joint.find("parent").attrib["link"],
            "child": joint.find("child").attrib["link"],
            "origin": vector("origin", "xyz", "0 0 0"),
            "axis": vector("axis", "xyz", "1 0 0"),
            "limits": limits,
            "scene_current_q": float(package_states.get(
                joint.attrib["name"], {}).get(
                    "scene_current_q",
                    package_states.get(joint.attrib["name"], {}).get(
                        "requested_scene_q", 0.0),
                )),
            "blender_object":
                f"{object_blueprint['object_id']}_joint_{joint.attrib['name']}",
        }
        if joint_type == "revolute":
            metadata.update({
                "limits_rad": limits,
                "initial_angle_deg": float(blueprint_joint.get("initial_angle_deg", 0.0)),
            })
        else:
            metadata["initial_position"] = 0.0
        joint_metadata.append(metadata)
    return {
        "package": str(package),
        "links": links,
        "joints": joint_metadata,
        "joint": joint_metadata[0],
    }


def blueprint_object_specs(blueprint, root=ROOT):
    root = Path(root)
    specs = {}
    for item in blueprint["objects"]:
        object_id = item["object_id"]
        articulated = item.get("operability_type") == "articulated_operable"
        if articulated:
            asset = root / "repaired_models" / object_id / "repaired.blend"
            geometry_asset = root / "repaired_models" / object_id / "repaired_animated.glb"
            asset_type = "articulated_blend"
        else:
            repaired = root / "repaired_models" / object_id / "repaired.glb"
            asset = repaired if repaired.exists() else generated_rigid_asset(object_id, root)
            geometry_asset = asset
            asset_type = "rigid_glb"
        specs[object_id] = {
            "asset_type": asset_type,
            "asset": str(asset),
            "geometry_asset": str(geometry_asset),
        }
        model = root / "repaired_models" / object_id
        native_manifest = model / "articulation.json"
        package = model / "repaired_package"
        if articulated and (package / "model.urdf").is_file():
            articulation = repaired_articulation(package, item)
            if len(articulation["joints"]) == 1:
                child_link = articulation["joint"]["child"]
                child_asset = next(
                    link["geometry_asset"] for link in articulation["links"]
                    if link["name"] == child_link
                )
                articulation.update({
                    "geometry_asset": str(model / "repaired_initial.glb"),
                    "child_node": Path(child_asset).stem,
                })
            specs[object_id].update({
                "asset_type": "articulated_urdf",
                "asset": str(package),
                "geometry_asset": str(model / "repaired_initial.glb"),
                "articulation": articulation,
            })
        elif articulated and native_manifest.is_file():
            articulation = json.loads(native_manifest.read_text(encoding="utf-8"))
            specs[object_id].update({
                "asset_type": "articulated_glb",
                "asset": articulation["asset"],
                "geometry_asset": articulation["geometry_asset"],
                "articulation": articulation,
            })
    return specs


def load_scene_label_map(manifest_path, root=None):
    manifest_path = Path(manifest_path)
    root = Path(root) if root is not None else manifest_path.parents[1]
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = np.zeros((document["img_height"], document["img_width"]), dtype=np.uint16)
    ids = {}
    for instance_id, item in enumerate(document["objects"], 1):
        ids[item["object_id"]] = instance_id
        mask = np.asarray(Image.open(root / item["mask_path"]).convert("L")) > 127
        labels[mask] = instance_id
    return labels, ids


def table_axis_scales(native_extents, target_extents):
    return np.asarray(target_extents, dtype=float) / np.asarray(native_extents, dtype=float)


def uniform_object_scale(native_extents, target_extents):
    ratios = np.asarray(target_extents, dtype=float) / np.asarray(native_extents, dtype=float)
    return float(np.exp(np.mean(np.log(ratios))))


def world_pose_vertices(vertices, uniform_scale, translation_xy, yaw_deg):
    vertices = np.asarray(vertices, dtype=float)
    center = np.array(
        [
            np.mean([vertices[:, 0].min(), vertices[:, 0].max()]),
            np.mean([vertices[:, 1].min(), vertices[:, 1].max()]),
            vertices[:, 2].min(),
        ]
    )
    posed = (vertices - center) * float(uniform_scale)
    posed = posed @ rotation_z(np.radians(yaw_deg)).T
    posed[:, :2] += np.asarray(translation_xy, dtype=float)
    return posed


def render_top_silhouette(mesh, image_size, table_center_px, meters_per_pixel, scale, translation_xy, yaw_deg):
    width, height = image_size
    vertices = world_pose_vertices(mesh.vertices, scale, translation_xy, yaw_deg)
    pixels = np.column_stack(
        (
            table_center_px[0] + vertices[:, 0] / meters_per_pixel[0],
            table_center_px[1] - vertices[:, 1] / meters_per_pixel[1],
        )
    )
    polygons = np.rint(pixels[np.asarray(mesh.faces)]).astype(np.int32)
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, polygons, 255)
    return solid_external_silhouette(output)


def refine_top_pose(mesh, target, table_center_px, meters_per_pixel, scale, translation_xy, yaw_deg):
    initial = np.array([translation_xy[0], translation_xy[1], yaw_deg, scale], dtype=float)

    def score(parameters):
        rendered = render_top_silhouette(
            mesh,
            (target.shape[1], target.shape[0]),
            table_center_px,
            meters_per_pixel,
            parameters[3],
            parameters[:2],
            parameters[2],
        )
        return symmetric_contour_loss(rendered, target)

    parameters = initial.copy()
    before = best = score(parameters)
    for coarse_yaw in np.arange(-180.0, 180.0, 15.0):
        candidate = parameters.copy()
        candidate[2] = coarse_yaw
        value = score(candidate)
        if value < best:
            parameters, best = candidate, value
    yaw_anchor = parameters[2]
    for steps in (
        np.array([0.01, 0.01, 5.0, 0.05]),
        np.array([0.002, 0.002, 1.0, 0.01]),
        np.array([0.0005, 0.0005, 0.25, 0.002]),
    ):
        improved = True
        while improved:
            improved = False
            for index in range(4):
                for direction in (-1, 1):
                    candidate = parameters.copy()
                    candidate[index] += direction * steps[index]
                    if (
                        np.any(np.abs(candidate[:2] - initial[:2]) > 0.05)
                        or abs(candidate[3] - initial[3]) > 0.35
                        or abs(candidate[2] - yaw_anchor) > 20.0
                        or candidate[3] <= 0
                    ):
                        continue
                    value = score(candidate)
                    if value + 1e-6 < best:
                        parameters, best, improved = candidate, value, True
    return {
        "translation_world_m": parameters[:2].tolist(),
        "yaw_deg": float((parameters[2] + 180.0) % 360.0 - 180.0),
        "uniform_scale": float(parameters[3]),
        "loss_before_px": float(before),
        "loss_after_px": float(best),
    }


def render_front_silhouette(mesh, image_size, scale, translation_xy, yaw_deg, frame, intrinsic, extrinsic):
    width, height = image_size
    vertices = world_pose_vertices(mesh.vertices, scale, translation_xy, yaw_deg)
    pixels, depth = project_world_points(
        vertices,
        frame["scale_world_per_vggt"],
        np.asarray(frame["rotation_world_from_vggt"]),
        np.asarray(frame["translation_world"]),
        intrinsic,
        extrinsic,
    )
    faces = np.asarray(mesh.faces)
    valid = np.all(depth[faces] > 0, axis=1)
    polygons = np.rint(pixels[faces[valid]]).astype(np.int32)
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, polygons, 255)
    return solid_external_silhouette(output)


def refine_front_translation(mesh, target, scale, translation_xy, yaw_deg, frame, intrinsic, extrinsic):
    initial = np.asarray(translation_xy, dtype=float)

    def score(xy):
        rendered = render_front_silhouette(
            mesh,
            (target.shape[1], target.shape[0]),
            scale,
            xy,
            yaw_deg,
            frame,
            intrinsic,
            extrinsic,
        )
        return symmetric_contour_loss(rendered, target)

    translation = initial.copy()
    before = best = score(translation)
    for step in (0.005, 0.001, 0.00025):
        improved = True
        while improved:
            improved = False
            for axis in range(2):
                for direction in (-1, 1):
                    candidate = translation.copy()
                    candidate[axis] += direction * step
                    if np.any(np.abs(candidate - initial) > 0.025):
                        continue
                    value = score(candidate)
                    if value + 1e-6 < best:
                        translation, best, improved = candidate, value, True
    return {
        "translation_world_m": translation.tolist(),
        "uniform_scale": float(scale),
        "loss_before_px": float(before),
        "loss_after_px": float(best),
    }


def refine_joint_translation(
    mesh,
    top_target,
    front_target,
    table_center_px,
    meters_per_pixel,
    scale,
    translation_xy,
    yaw_deg,
    frame,
    intrinsic,
    extrinsic,
    top_weight=1.0,
    front_weight=2.0,
):
    initial = np.asarray(translation_xy, dtype=float)

    def losses(xy):
        top = render_top_silhouette(
            mesh,
            (top_target.shape[1], top_target.shape[0]),
            table_center_px,
            meters_per_pixel,
            scale,
            xy,
            yaw_deg,
        )
        front = render_front_silhouette(
            mesh,
            (front_target.shape[1], front_target.shape[0]),
            scale,
            xy,
            yaw_deg,
            frame,
            intrinsic,
            extrinsic,
        )
        top_loss = symmetric_contour_loss(top, top_target)
        front_loss = symmetric_contour_loss(front, front_target)
        return top_loss, front_loss, top_weight * top_loss + front_weight * front_loss

    translation = initial.copy()
    top_before, front_before, best = losses(translation)
    for step in (0.005, 0.001, 0.00025):
        improved = True
        while improved:
            improved = False
            for axis in range(2):
                for direction in (-1, 1):
                    candidate = translation.copy()
                    candidate[axis] += direction * step
                    if np.any(np.abs(candidate - initial) > 0.025):
                        continue
                    _, _, value = losses(candidate)
                    if value + 1e-6 < best:
                        translation, best, improved = candidate, value, True
    top_after, front_after, combined_after = losses(translation)
    return {
        "translation_world_m": translation.tolist(),
        "uniform_scale": float(scale),
        "top_weight": float(top_weight),
        "front_weight": float(front_weight),
        "top_loss_before_px": float(top_before),
        "top_loss_after_px": float(top_after),
        "front_loss_before_px": float(front_before),
        "front_loss_after_px": float(front_after),
        "combined_loss_before": float(top_weight * top_before + front_weight * front_before),
        "combined_loss_after": float(combined_after),
    }


def instance_geometry(labels, instance_id):
    mask = np.where(labels == instance_id, 255, 0).astype(np.uint8)
    contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    points = contour[:, 0].astype(float)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    axis = np.linalg.eigh(covariance)[1][:, -1]
    if axis[1] < 0:
        axis = -axis
    return {
        "bbox_xyxy": [x, y, x + width - 1, y + height - 1],
        "center_px": center.tolist(),
        "major_axis_image": axis.tolist(),
        "edge_points": int(len(points)),
    }


def mesh_extents_z_up(path):
    import trimesh

    mesh = trimesh.load(path, force="scene").to_geometry()
    vertices = mesh.vertices
    if Path(path).suffix.lower() in {".glb", ".gltf"}:
        vertices = gltf_y_up_to_z_up(vertices)
    return np.ptp(vertices, axis=0)


def mesh_vertices_z_up(path):
    import trimesh

    vertices = trimesh.load(path, force="scene").to_geometry().vertices
    if Path(path).suffix.lower() in {".glb", ".gltf"}:
        vertices = gltf_y_up_to_z_up(vertices)
    return vertices


def search_mesh_z_up(path, face_count=12000):
    import trimesh

    mesh = trimesh.load(path, force="scene").to_geometry()
    if face_count is not None and len(mesh.faces) > face_count:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=face_count)
        except BaseException:
            mesh = mesh.copy()
    if Path(path).suffix.lower() in {".glb", ".gltf"}:
        normals = np.asarray(mesh.vertex_normals).copy()
        mesh.vertices = gltf_y_up_to_z_up(mesh.vertices)
        mesh.vertex_normals = gltf_y_up_to_z_up(normals)
    return mesh


def front_vggt_frame(labels, ids, support_id, support_width_m):
    with np.load(ROOT / "outputs/vggt_front_only/predictions.npz") as data:
        points = data["world_points"][0]
        retained = data["retained_mask"][0]
        extrinsic = data["extrinsic"][0]
    height, width = points.shape[:2]
    resized_labels = np.asarray(
        Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST)
    )
    finite = np.isfinite(points).all(axis=-1)
    table_valid = (resized_labels == ids[support_id]) & retained & finite
    observed = points[table_valid]
    span = np.linalg.norm(np.percentile(observed, 95, axis=0) - np.percentile(observed, 5, axis=0))
    normal, _, inliers = fit_plane_ransac(observed, 0.005 * span, iterations=1500, seed=0)
    plane_center = np.median(observed[inliers], axis=0)
    camera_center = -extrinsic[:, :3].T @ extrinsic[:, 3]
    rotation, _ = table_frame_from_plane(normal, plane_center, camera_center)
    local_top = observed[inliers] @ rotation.T
    observed_width = np.diff(np.percentile(local_top[:, 0], [1, 99]))[0]
    scale = float(support_width_m / observed_width)
    origin = np.array(
        [
            np.mean(np.percentile(local_top[:, 0], [1, 99])),
            np.mean(np.percentile(local_top[:, 1], [1, 99])),
            np.median(local_top[:, 2]),
        ]
    )
    translation = -scale * origin
    object_pointclouds = {}
    for object_id, instance_id in ids.items():
        if object_id == support_id:
            continue
        valid = (resized_labels == instance_id) & retained & finite
        world = points[valid] @ rotation.T * scale + translation
        if len(world):
            object_pointclouds[object_id] = {
                "point_count": int(len(world)),
                "center_world_m": np.median(world, axis=0).tolist(),
                "extent_p01_p99_m": np.diff(np.percentile(world, [1, 99], axis=0), axis=0)[0].tolist(),
            }
    return {
        "scale_world_per_vggt": scale,
        "rotation_world_from_vggt": rotation.tolist(),
        "translation_world": translation.tolist(),
        "objects": object_pointclouds,
    }


def fit_table_z_scale(vertices, scale_xy, front_frame, target_bbox, intrinsic, extrinsic):
    from scipy.optimize import minimize_scalar

    vertices = np.asarray(vertices, dtype=float).copy()
    vertices[:, :2] -= (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) * 0.5
    vertices[:, 2] -= vertices[:, 2].max()

    def bounds(scale_z):
        points = vertices * np.array([scale_xy[0], scale_xy[1], scale_z])
        pixels, depth = project_world_points(
            points,
            front_frame["scale_world_per_vggt"],
            np.asarray(front_frame["rotation_world_from_vggt"]),
            np.asarray(front_frame["translation_world"]),
            intrinsic,
            extrinsic,
        )
        return float(pixels[depth > 0, 1].min()), float(pixels[depth > 0, 1].max())

    def objective(scale_z):
        top, bottom = bounds(scale_z)
        return (top - target_bbox[1]) ** 2 + (bottom - target_bbox[3]) ** 2

    result = minimize_scalar(objective, bounds=(0.25, 3.0), method="bounded")
    projected = bounds(result.x)
    return float(result.x), list(projected), float(np.sqrt(result.fun))


def run(output_dir=ROOT / "outputs/unified_scene_alignment", object_ids=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    front_labels, front_ids = load_scene_label_map(ROOT / "data/segmentation_results.json", ROOT)
    top_labels, top_ids = load_scene_label_map(ROOT / "data/topview_segmentation_results.json", ROOT)
    Image.fromarray(front_labels).save(output_dir / "front_scene_labels.png")
    Image.fromarray(top_labels).save(output_dir / "top_scene_labels.png")

    blueprint = json.loads((ROOT / "data/blueprint.json").read_text(encoding="utf-8"))
    object_specs = blueprint_object_specs(blueprint)
    selected_ids = set(object_ids) if object_ids else set(object_specs)
    table_mesh = generated_rigid_asset("table_0")
    front_table = instance_geometry(front_labels, front_ids["table_0"])
    top_table = instance_geometry(top_labels, top_ids["table_0"])
    table_native = mesh_extents_z_up(table_mesh)

    top_table_width = top_table["bbox_xyxy"][2] - top_table["bbox_xyxy"][0] + 1
    top_table_height = top_table["bbox_xyxy"][3] - top_table["bbox_xyxy"][1] + 1
    table_width_m = 0.9
    table_depth_m = table_width_m * top_table_height / top_table_width
    scale_xy = table_axis_scales(table_native[:2], [table_width_m, table_depth_m])
    front_frame = front_vggt_frame(front_labels, front_ids, "table_0", table_width_m)
    with np.load(ROOT / "outputs/vggt_front_only/predictions.npz") as data:
        front_intrinsic = data["intrinsic"][0]
        front_extrinsic = data["extrinsic"][0]
        network_height, network_width = data["world_points"].shape[1:3]
    front_transform = vggt_crop_transform((front_labels.shape[1], front_labels.shape[0]))
    if front_transform["network_size"] != [network_width, network_height]:
        raise ValueError(f"VGGT preprocessing size mismatch: {front_transform['network_size']} != {[network_width, network_height]}")
    raw_front_intrinsic = network_intrinsic_to_raw(front_intrinsic, front_transform)
    network_labels = preprocess_label_crop(front_labels, front_transform)
    network_table = instance_geometry(network_labels, front_ids["table_0"])
    scale_z, projected_vertical, vertical_error = fit_table_z_scale(
        mesh_vertices_z_up(table_mesh),
        scale_xy,
        front_frame,
        network_table["bbox_xyxy"],
        front_intrinsic,
        front_extrinsic,
    )
    table_scale = np.array([scale_xy[0], scale_xy[1], scale_z])
    table_height_m = table_native[2] * scale_z

    meters_per_top_x = table_width_m / top_table_width
    meters_per_top_y = table_depth_m / top_table_height
    table_center = np.asarray(top_table["center_px"])
    objects = {}
    for object_id, spec in object_specs.items():
        if object_id not in selected_ids:
            continue
        if object_id not in front_ids or object_id not in top_ids:
            continue
        asset = Path(spec["asset"])
        geometry_asset = Path(spec["geometry_asset"])
        if not asset.exists() or not geometry_asset.exists():
            raise FileNotFoundError(f"missing {spec['asset_type']} asset for {object_id}: {asset}")
        front_geometry = instance_geometry(front_labels, front_ids[object_id])
        top_geometry = instance_geometry(top_labels, top_ids[object_id])
        native = mesh_extents_z_up(geometry_asset)
        top_center = np.asarray(top_geometry["center_px"])
        top_xy = np.array(
            [
                (top_center[0] - table_center[0]) * meters_per_top_x,
                -(top_center[1] - table_center[1]) * meters_per_top_y,
            ]
        )
        pointcloud = front_frame["objects"].get(object_id)
        translation_xy = top_xy if pointcloud is None else 0.5 * (top_xy + np.asarray(pointcloud["center_world_m"][:2]))
        target_xy = np.array(
            [
                (top_geometry["bbox_xyxy"][2] - top_geometry["bbox_xyxy"][0] + 1) * meters_per_top_x,
                (top_geometry["bbox_xyxy"][3] - top_geometry["bbox_xyxy"][1] + 1) * meters_per_top_y,
            ]
        )
        candidates = []
        for quarter_turns in (0, 1):
            native_xy = native[[1, 0]] if quarter_turns else native[:2]
            scale = uniform_object_scale(native_xy, target_xy)
            error = float(np.std(np.log(target_xy / native_xy)))
            candidates.append((error, quarter_turns, scale))
        _, quarter_turns, uniform_scale = min(candidates)
        image_axis = np.asarray(top_geometry["major_axis_image"])
        world_axis = np.array([image_axis[0], -image_axis[1]])
        target_angle = np.degrees(np.arctan2(world_axis[1], world_axis[0]))
        native_xy = native[[1, 0]] if quarter_turns else native[:2]
        native_major_angle = 0.0 if native_xy[0] >= native_xy[1] else 90.0
        yaw = float(((target_angle - native_major_angle + 180) % 360) - 180)
        refinement = None
        if object_id in selected_ids:
            search_mesh = search_mesh_z_up(geometry_asset)
            target_top = np.where(top_labels == top_ids[object_id], 255, 0).astype(np.uint8)
            target_top = solid_external_silhouette(target_top)
            total_yaw = yaw + 90 * quarter_turns
            top_refined = refine_top_pose(
                search_mesh,
                target_top,
                table_center,
                np.array([meters_per_top_x, meters_per_top_y]),
                uniform_scale,
                top_xy,
                total_yaw,
            )
            uniform_scale = top_refined["uniform_scale"]
            translation_xy = np.asarray(top_refined["translation_world_m"])
            total_yaw = top_refined["yaw_deg"]
            target_front = np.where(network_labels == front_ids[object_id], 255, 0).astype(np.uint8)
            target_front = solid_external_silhouette(target_front)
            front_refined = refine_front_translation(
                search_mesh,
                target_front,
                uniform_scale,
                translation_xy,
                total_yaw,
                front_frame,
                front_intrinsic,
                front_extrinsic,
            )
            translation_xy = np.asarray(front_refined["translation_world_m"])
            yaw = float(total_yaw - 90 * quarter_turns)
            refinement = {"top_coarse": top_refined, "front_final": front_refined}
        objects[object_id] = {
            **spec,
            "uniform_scale": uniform_scale,
            "axis_mapping_quarter_turns": quarter_turns,
            "translation_world_m": [*translation_xy.tolist(), 0.0],
            "top_edge_center_world_m": top_xy.tolist(),
            "front_pointcloud": pointcloud,
            "yaw_deg": yaw,
            "top_instance_geometry": top_geometry,
            "front_instance_geometry": front_geometry,
            "scale_source": "top scene instance bbox with one XYZ-uniform scalar",
            "edge_refinement": refinement,
        }

    scene = {
        "format": "unified_scene_alignment_v1",
        "coordinate_system": {
            "origin": "top-view table instance bbox center on tabletop",
            "x_axis": "top image right",
            "y_axis": "top image up",
            "z_axis": "up",
            "unit": "meter",
        },
        "segmentation": {
            "front_manifest": str(ROOT / "data/segmentation_results.json"),
            "top_manifest": str(ROOT / "data/topview_segmentation_results.json"),
            "front_label_map": str(output_dir / "front_scene_labels.png"),
            "top_label_map": str(output_dir / "top_scene_labels.png"),
            "front_instance_ids": front_ids,
            "top_instance_ids": top_ids,
            "front_image_size": [int(front_labels.shape[1]), int(front_labels.shape[0])],
            "top_image_size": [int(top_labels.shape[1]), int(top_labels.shape[0])],
        },
        "table": {
            "mesh": str(table_mesh),
            "scale_xyz": table_scale.tolist(),
            "target_extents_m": [table_width_m, table_depth_m, table_height_m],
            "top_instance_geometry": top_table,
            "front_instance_geometry": front_table,
            "scale_sources": {"xy": "top scene table bbox", "z": "front scene table bbox"},
            "front_projection_fit": {
                "target_bbox_network_xyxy": network_table["bbox_xyxy"],
                "projected_vertical_bounds_px": projected_vertical,
                "vertical_rms_error_px": vertical_error,
            },
        },
        "objects": objects,
        "cameras": {
            "top": {"type": "centered_vertical", "right": "+X", "down": "-Y", "forward": "-Z"},
            "front": {
                "type": "front-only VGGT",
                "world_frame": front_frame,
                "network_intrinsic": front_intrinsic.tolist(),
                "raw_intrinsic": raw_front_intrinsic.tolist(),
                "raw_to_network": front_transform,
            },
        },
    }
    output = output_dir / "scene_alignment.json"
    output.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table": scene["table"], "objects": scene["objects"]}, indent=2))
    print(output)
    return scene


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", action="append")
    args = parser.parse_args()
    run(object_ids=args.object_id)
