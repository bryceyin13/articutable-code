#!/usr/bin/env python3
"""Limit URDF joints against the final Stage 12 scene with trimesh/FCL."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def _numbers(value, default):
    return np.array([float(item) for item in value.split()] if value else default)


def _origin(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = trimesh.transformations.euler_matrix(
        *_numbers(element.get("rpy"), [0, 0, 0]), axes="sxyz"
    )[:3, :3]
    matrix[:3, 3] = _numbers(element.get("xyz"), [0, 0, 0])
    return matrix


def _load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"no triangle mesh in {path}")
    return mesh


def _collision_hull(mesh, maximum_vertices=256):
    hull = mesh.convex_hull
    if len(hull.vertices) <= maximum_vertices:
        return hull
    count = maximum_vertices - 6
    index = np.arange(count)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    angle = index * math.pi * (3.0 - math.sqrt(5.0))
    directions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    selected = []
    for chunk in np.array_split(directions, 8):
        selected.extend(np.argmax(hull.vertices @ chunk.T, axis=0))
    selected.extend(np.argmin(hull.vertices, axis=0))
    selected.extend(np.argmax(hull.vertices, axis=0))
    return trimesh.convex.convex_hull(hull.vertices[np.unique(selected)])


def _mesh_path(package, filename):
    if filename.startswith("package://"):
        filename = filename[len("package://"):].split("/", 1)[-1]
    path = Path(filename)
    return path if path.is_absolute() else package / path


class UrdfModel:
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        package = self.path.parent
        self.collisions = {}
        for link in root.findall("link"):
            meshes = []
            for collision in link.findall("collision"):
                node = collision.find("./geometry/mesh")
                if node is None:
                    continue
                mesh = _collision_hull(
                    _load_mesh(_mesh_path(package, node.attrib["filename"]))
                )
                scale = _numbers(node.get("scale"), [1, 1, 1])
                mesh.apply_scale(scale)
                meshes.append((mesh, _origin(collision.find("origin"))))
            self.collisions[link.attrib["name"]] = meshes

        self.joints = {}
        self.children = {}
        child_links = set()
        for node in root.findall("joint"):
            limit = node.find("limit")
            joint = {
                "name": node.attrib["name"],
                "type": node.attrib.get("type", "fixed"),
                "parent": node.find("parent").attrib["link"],
                "child": node.find("child").attrib["link"],
                "origin": _origin(node.find("origin")),
                "axis": _numbers(
                    node.find("axis").get("xyz") if node.find("axis") is not None else None,
                    [1, 0, 0],
                ),
                "lower": float(limit.get("lower", "0")) if limit is not None else 0.0,
                "upper": float(limit.get("upper", "0")) if limit is not None else 0.0,
            }
            self.joints[joint["name"]] = joint
            self.children.setdefault(joint["parent"], []).append(joint)
            child_links.add(joint["child"])
        self.roots = set(self.collisions) - child_links

    def subtree(self, joint_name):
        links = set()
        pending = [self.joints[joint_name]["child"]]
        while pending:
            link = pending.pop()
            links.add(link)
            pending.extend(joint["child"] for joint in self.children.get(link, ()))
        return links

    def meshes(self, values, root_transform, links=None):
        transforms = {}

        def visit(link, transform):
            transforms[link] = transform
            for joint in self.children.get(link, ()):
                motion = np.eye(4)
                value = float(values.get(joint["name"], 0.0))
                axis = joint["axis"] / np.linalg.norm(joint["axis"])
                if joint["type"] in {"revolute", "continuous"}:
                    motion = trimesh.transformations.rotation_matrix(value, axis)
                elif joint["type"] == "prismatic":
                    motion[:3, 3] = axis * value
                visit(joint["child"], transform @ joint["origin"] @ motion)

        for root in self.roots:
            visit(root, np.eye(4))
        selected = set(transforms) if links is None else set(links)
        result = []
        for link in selected:
            for source, collision_origin in self.collisions.get(link, ()):
                mesh = source.copy()
                mesh.apply_transform(root_transform @ transforms[link] @ collision_origin)
                result.append(mesh)
        return result


def _root_transform(pose):
    yaw = trimesh.transformations.rotation_matrix(
        math.radians(float(pose.get("yaw_deg", 0.0))), [0, 0, 1]
    )
    roll_x = trimesh.transformations.rotation_matrix(
        math.radians(float(pose.get("roll_x_deg", 0.0))), [1, 0, 0]
    )
    roll_y = trimesh.transformations.rotation_matrix(
        math.radians(float(pose.get("roll_y_deg", 0.0))), [0, 1, 0]
    )
    matrix = yaw @ roll_y @ roll_x
    scale = pose.get("scale", 1.0)
    scale = [scale] * 3 if isinstance(scale, (int, float)) else scale
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(scale)
    matrix[:3, 3] = pose.get("translation_m", [0, 0, 0])
    return matrix


def _reference_states(alignment, models=None):
    states = {}
    for object_id, item in alignment.get("objects", {}).items():
        joint_states = item.get("joint_states") or (
            [item["joint_state"]] if item.get("joint_state") else [])
        for state in joint_states:
            value = (
                state.get("relative_angle_rad")
                if state.get("type") == "revolute"
                else state.get(
                    "scene_current_q", state.get("relative_displacement"))
            )
            if value is not None:
                name = state.get("name", "joint_1")
                model = (models or {}).get(object_id)
                namespaced = f"{object_id}_{name}"
                if model and name not in model.joints and namespaced in model.joints:
                    name = namespaced
                states.setdefault(object_id, {})[name] = float(value)
    return states


class SceneCollision:
    def __init__(self, obstacles, reference_mesh, clearance):
        self.entries = []
        for object_id, meshes in obstacles.items():
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(meshes):
                manager.add_object(f"{object_id}:{index}", mesh)
            _, contacts = manager.in_collision_single(
                reference_mesh, return_data=True
            )
            baseline = max((float(contact.depth) for contact in contacts), default=0.0)
            self.entries.append((object_id, manager, baseline + clearance))

    def blockers(self, mesh):
        blocked = []
        for object_id, manager, allowed_depth in self.entries:
            if not manager.in_collision_single(mesh):
                continue
            collision, contacts = manager.in_collision_single(mesh, return_data=True)
            depth = max((float(contact.depth) for contact in contacts), default=0.0)
            if collision and depth > allowed_depth:
                blocked.append(object_id)
        return blocked


def _scan_interval(lower, upper, reference, evaluate, samples=91):
    points = sorted(set(np.linspace(lower, upper, samples).tolist() + [reference]))
    results = [evaluate(value) for value in points]
    free = [not blockers for blockers in results]

    def boundary(blocked_value, free_value):
        for _ in range(20):
            middle = (blocked_value + free_value) / 2.0
            if evaluate(middle):
                blocked_value = middle
            else:
                free_value = middle
        return free_value

    intervals = []
    index = 0
    while index < len(points):
        if not free[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(points) and free[index + 1]:
            index += 1
        end = index
        lo = points[start] if start == 0 else boundary(points[start - 1], points[start])
        hi = points[end] if end == len(points) - 1 else boundary(points[end + 1], points[end])
        intervals.append([lo, hi])
        index += 1
    selected = next(
        (interval for interval in intervals if interval[0] <= reference <= interval[1]),
        [reference, reference],
    )
    blockers = sorted({name for result in results for name in result})
    return intervals, selected, blockers


def compute_ranges(manifest, alignment, sim_export, clearance_ratio=0.002):
    objects = {item["object_id"]: item for item in manifest["objects"]}
    models = {
        object_id: UrdfModel(sim_export / item["asset"])
        for object_id, item in objects.items()
        if item["type"] == "articulated"
    }
    references = _reference_states(alignment, models)
    roots = {
        object_id: _root_transform(objects[object_id].get("root_pose", {}))
        for object_id in models
    }

    scene_meshes = {}
    for object_id, item in objects.items():
        if item["type"] == "articulated":
            scene_meshes[object_id] = models[object_id].meshes(
                references.get(object_id, {}), roots[object_id]
            )
        else:
            scene_meshes[object_id] = [
                _collision_hull(_load_mesh(sim_export / item["asset_glb"]))
            ]
    bounds = np.array([mesh.bounds for meshes in scene_meshes.values() for mesh in meshes])
    scene_diagonal = float(np.linalg.norm(bounds[:, 1].max(axis=0) - bounds[:, 0].min(axis=0)))
    clearance = clearance_ratio * scene_diagonal

    for object_id, model in models.items():
        values = references.get(object_id, {})
        ranges = []
        for name, joint in model.joints.items():
            if (
                joint["type"] not in {"revolute", "prismatic"}
                or name not in values
                or joint["lower"] >= joint["upper"]
            ):
                continue
            reference = float(values[name])
            if not joint["lower"] <= reference <= joint["upper"]:
                raise ValueError(
                    f"{object_id}/{name} Stage 12 reference {reference} is outside "
                    f"URDF limits [{joint['lower']}, {joint['upper']}]"
                )
            moving_links = model.subtree(name)

            def moving(q):
                current = dict(values)
                current[name] = q
                meshes = model.meshes(current, roots[object_id], moving_links)
                return trimesh.util.concatenate(meshes)

            obstacles = {
                other_id: meshes
                for other_id, meshes in scene_meshes.items()
                if other_id != object_id and meshes
            }
            collision = SceneCollision(obstacles, moving(reference), clearance)
            cache = {}

            def evaluate(q):
                key = round(float(q), 12)
                if key not in cache:
                    cache[key] = collision.blockers(moving(q))
                return cache[key]

            intervals, video_range, blockers = _scan_interval(
                joint["lower"], joint["upper"], reference, evaluate
            )
            result = {
                "name": name,
                "type": joint["type"],
                "unit": "rad" if joint["type"] == "revolute" else "m",
                "urdf_limit": [joint["lower"], joint["upper"]],
                "scene_collision_free_intervals": intervals,
                "video_range": video_range,
                "blocking_objects": blockers,
            }
            result[
                "scene_current_q" if joint["type"] == "prismatic" else "reference_q"
            ] = reference
            ranges.append(result)
        objects[object_id]["scene_joint_ranges"] = ranges

    manifest["scene_joint_range_analysis"] = {
        "method": "fcl_convex_hull",
        "collision_mesh_preparation": "support_sampled_convex_hull_256",
        "collision_scope": "external_objects_fixed_at_stage12_pose",
        "clearance_ratio_of_scene_diagonal": clearance_ratio,
        "scene_collision_bbox_diagonal_m": scene_diagonal,
        "clearance_m": clearance,
    }
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--sim-export", type=Path, required=True)
    parser.add_argument("--clearance-ratio", type=float, default=0.002)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    result = compute_ranges(
        manifest, alignment, args.sim_export, args.clearance_ratio
    )
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
