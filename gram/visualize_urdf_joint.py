#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.render_settings import configure_specular_reflections


# --- config: edit these or pass CLI args after "--" ---
ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "outputs/gram/laptop_0/model.urdf"
MESH_ROOT = ROOT / "outputs/gram/laptop_0"
TARGET_JOINT_NAME = None
FRAME_START = 1
FRAME_END = 120
REVOLUTE_ANGLE_DEG = 45.0
PRISMATIC_DISTANCE = 0.2
OUTPUT_BLEND = ROOT / "outputs/urdf_joint_visualization.blend"
OUTPUT_PREVIEW = ROOT / "outputs/urdf_joint_visualization.png"


def parse_vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return Vector(default)
    return Vector(float(v) for v in text.split())


def make_transform(xyz, rpy):
    mat = Matrix.Translation(Vector(xyz))
    mat @= Euler((float(rpy[0]), float(rpy[1]), float(rpy[2])), "XYZ").to_matrix().to_4x4()
    return mat


def parse_origin(node):
    if node is None:
        return make_transform((0, 0, 0), (0, 0, 0))
    return make_transform(parse_vec(node.attrib.get("xyz")), parse_vec(node.attrib.get("rpy")))


def parse_urdf(urdf_path):
    xml = ET.parse(urdf_path).getroot()
    links = {}
    for link in xml.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            scale = parse_vec(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
            visuals.append(
                {
                    "filename": mesh.attrib["filename"],
                    "origin": parse_origin(visual.find("origin")),
                    "scale": scale,
                }
            )
        links[link.attrib["name"]] = {"name": link.attrib["name"], "visuals": visuals}

    joints = []
    child_links = set()
    for joint in xml.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        child_links.add(child)
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent,
                "child": child,
                "origin": parse_origin(joint.find("origin")),
                "axis": parse_vec((joint.find("axis").attrib.get("xyz") if joint.find("axis") is not None else None), (1.0, 0.0, 0.0)).normalized(),
                "lower": float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
                "upper": float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
            }
        )
    roots = [name for name in links if name not in child_links]
    return {"links": links, "joints": joints, "roots": roots}


def resolve_mesh_path(filename, mesh_root):
    filename = filename.replace("\\", "/")
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
        parts = filename.split("/", 1)
        filename = parts[1] if len(parts) == 2 else parts[0]
    path = Path(filename)
    if path.is_absolute() and path.exists():
        return path
    direct = Path(mesh_root) / filename
    if direct.exists():
        return direct
    matches = list(Path(mesh_root).rglob(Path(filename).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"mesh not found: {filename} under {mesh_root}")


def import_mesh(mesh_path):
    before = set(bpy.context.scene.objects)
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(
                filepath=str(mesh_path), forward_axis="Y", up_axis="Z",
            )
        else:
            bpy.ops.import_scene.obj(
                filepath=str(mesh_path), axis_forward="Y", axis_up="Z",
            )
    elif suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=str(mesh_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    else:
        raise ValueError(f"unsupported mesh format: {mesh_path}")
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def root_objects(imported):
    return [obj for obj in imported if obj.parent not in imported]


def parent_keep_world(obj, parent):
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()


def joint_frame_matrix(origin, axis):
    x_axis = axis.normalized()
    helper = Vector((0, 0, 1)) if abs(x_axis.z) < 0.95 else Vector((0, 1, 0))
    y_axis = helper.cross(x_axis).normalized()
    z_axis = x_axis.cross(y_axis).normalized()
    return Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x, origin.translation.x),
            (x_axis.y, y_axis.y, z_axis.y, origin.translation.y),
            (x_axis.z, y_axis.z, z_axis.z, origin.translation.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


def cylinder_between(name, start, end, radius, mat):
    mid = (start + end) * 0.5
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=direction.length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    return obj


def cone_at(name, location, direction, radius, depth, mat):
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.normalized().to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    return obj


def axis_basis(axis):
    axis = axis.normalized()
    helper = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((0, 1, 0))
    u = axis.cross(helper).normalized()
    v = axis.cross(u).normalized()
    return u, v


def create_axis_visual(name, origin_matrix, axis, length=0.28, radius=0.004, mat=None):
    origin = origin_matrix.translation
    direction = (origin_matrix.to_3x3() @ axis).normalized()
    start = origin - direction * length * 0.5
    end = origin + direction * length * 0.5
    objs = [cylinder_between(f"{name}_axis", start, end, radius, mat)]
    objs.append(cone_at(f"{name}_arrow", end, direction, radius * 4.0, radius * 12.0, mat))
    return objs


def create_revolute_arc(name, origin_matrix, axis, radius, angle_deg, mat):
    origin = origin_matrix.translation
    world_axis = (origin_matrix.to_3x3() @ axis).normalized()
    u, v = axis_basis(world_axis)
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = 0.003
    curve.materials.append(mat)
    poly = curve.splines.new("POLY")
    steps = 40
    poly.points.add(steps)
    start_a = math.radians(-angle_deg)
    end_a = math.radians(angle_deg)
    for i in range(steps + 1):
        t = i / steps
        a = start_a + (end_a - start_a) * t
        p = origin + radius * (math.cos(a) * u + math.sin(a) * v)
        poly.points[i].co = (p.x, p.y, p.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    end = origin + radius * (math.cos(end_a) * u + math.sin(end_a) * v)
    tangent = (-math.sin(end_a) * u + math.cos(end_a) * v).normalized()
    cone = cone_at(f"{name}_arrow", end, tangent, 0.015, 0.04, mat)
    return [obj, cone]


def create_prismatic_arrow(name, origin_matrix, axis, distance, mat):
    direction = (origin_matrix.to_3x3() @ axis).normalized()
    origin = origin_matrix.translation
    start = origin - direction * distance * 0.5
    end = origin + direction * distance * 0.5
    return [
        cylinder_between(f"{name}_slide", start, end, 0.004, mat),
        cone_at(f"{name}_arrow_a", end, direction, 0.016, 0.045, mat),
        cone_at(f"{name}_arrow_b", start, -direction, 0.016, 0.045, mat),
    ]


def package_joint_subtypes(mesh_root):
    manifest = Path(mesh_root) / "manifest.json"
    if not manifest.is_file():
        return {}
    return {
        item["name"]: item["subtype"]
        for item in json.loads(manifest.read_text(encoding="utf-8")).get(
            "joint_states", []
        )
        if item.get("name") and item.get("subtype") in {"hinge", "spin"}
    }


def _range_material(name, color):
    mat = material(name, (*color, 1.0))
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.82
        bsdf.inputs["Metallic"].default_value = 0.0
        specular = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get(
            "Specular"
        )
        if specular:
            specular.default_value = 0.18
    return mat


def joint_range_materials():
    return (
        _range_material("joint_axis_blue", (0.02, 0.26, 0.90)),
        _range_material("joint_motion_purple", (0.43, 0.10, 0.72)),
        _range_material("joint_fixed_gray", (0.45, 0.5, 0.55)),
    )


def _parent_local(obj, parent):
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)


def _range_curve(name, points, parent, mat, width):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = width
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for target, point in zip(spline.points, points):
        target.co = (*point, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    _parent_local(obj, parent)
    obj.data.materials.append(mat)
    return obj


def _range_cone(name, parent, base_position, direction, radius, length, mat):
    bpy.ops.mesh.primitive_cone_add(
        vertices=24, radius1=radius, radius2=0.0, depth=length,
    )
    obj = bpy.context.object
    obj.name = name
    _parent_local(obj, parent)
    direction = Vector(direction).normalized()
    obj.location = base_position + direction * length * 0.5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    return obj


def _range_axis_visual(name, origin_object, axis, length, mat, width):
    axis = Vector(axis).normalized()
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=width, depth=length,
    )
    obj = bpy.context.object
    obj.name = f"joint_axis::{name}"
    _parent_local(obj, origin_object)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    return obj


def _revolute_trajectory_points(axis, point, lower, upper):
    length = math.sqrt(sum(component * component for component in axis))
    if length <= 1e-12:
        raise ValueError("revolute axis must be nonzero")
    axis = tuple(component / length for component in axis)
    point = tuple(float(component) for component in point)
    dot = sum(a * p for a, p in zip(axis, point))
    cross = (
        axis[1] * point[2] - axis[2] * point[1],
        axis[2] * point[0] - axis[0] * point[2],
        axis[0] * point[1] - axis[1] * point[0],
    )
    span = min(upper - lower, 2.0 * math.pi)
    segments = max(24, min(192, int(64 * span / (2.0 * math.pi))))
    angles = [lower + span * index / segments for index in range(segments + 1)]
    return [
        tuple(
            point[index] * math.cos(angle)
            + cross[index] * math.sin(angle)
            + axis[index] * dot * (1.0 - math.cos(angle))
            for index in range(3)
        )
        for angle in angles
    ]


def _range_revolute_arc(
    name, origin_object, axis, point, lower, upper, mat, width,
):
    axis = Vector(axis).normalized()
    point = Vector(point)
    points = [
        Vector(value)
        for value in _revolute_trajectory_points(axis, point, lower, upper)
    ]
    center = axis * point.dot(axis)
    radius = (point - center).length
    if radius <= 1e-8:
        raise ValueError(f"joint {name} child mesh has no point away from its axis")
    arc = _range_curve(
        f"joint_range_arc::{name}", points, origin_object, mat, width,
    )
    _range_cone(
        f"joint_range_arrow_lower::{name}", origin_object, points[0],
        -axis.cross(points[0]), radius * 0.03, radius * 0.07, mat,
    )
    _range_cone(
        f"joint_range_arrow_upper::{name}", origin_object, points[-1],
        axis.cross(points[-1]), radius * 0.03, radius * 0.07, mat,
    )
    zero = Vector(_revolute_trajectory_points(axis, point, 0.0, 0.0)[0])
    radial = zero - center
    _range_curve(
        f"joint_zero_tick::{name}",
        (center + radial * 0.85, center + radial * 1.15),
        origin_object, mat, width,
    )
    return arc


def _child_trajectory_point(
    motion_object, mesh_objects, axis, prefer_axis_middle=False,
):
    axis = Vector(axis).normalized()
    to_motion = motion_object.matrix_world.inverted()

    def points():
        for obj in mesh_objects:
            transform = to_motion @ obj.matrix_world
            for vertex in obj.data.vertices:
                yield transform @ vertex.co

    best_point, best_distance = None, -1.0
    axis_min, axis_max = math.inf, -math.inf
    for point in points():
        axial = point.dot(axis)
        axis_min, axis_max = min(axis_min, axial), max(axis_max, axial)
        distance = (point - axis * axial).length_squared
        if distance > best_distance:
            best_point, best_distance = point.copy(), distance
    if best_point is None:
        raise ValueError("revolute child link contains no mesh vertices")
    if prefer_axis_middle:
        axis_middle = (axis_min + axis_max) * 0.5
        best_score = (math.inf, math.inf)
        for point in points():
            axial = point.dot(axis)
            distance = (point - axis * axial).length_squared
            if distance < 0.95 ** 2 * best_distance:
                continue
            score = (abs(axial - axis_middle), -distance)
            if score < best_score:
                best_point, best_score = point.copy(), score
    return best_point


def _range_prismatic(
    name, origin_object, axis, lower, upper, mat, width, marker_radius,
):
    axis = Vector(axis).normalized()
    start, end = axis * lower, axis * upper
    result = _range_curve(
        f"joint_range_line::{name}", (start, end), origin_object, mat, width,
    )
    _range_cone(
        f"joint_range_arrow_lower::{name}", origin_object, start, -axis,
        marker_radius, marker_radius * 2.5, mat,
    )
    _range_cone(
        f"joint_range_arrow_upper::{name}", origin_object, end, axis,
        marker_radius, marker_radius * 2.5, mat,
    )
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=20, ring_count=10, radius=marker_radius * 0.7,
    )
    zero = bpy.context.object
    zero.name = f"joint_zero::{name}"
    _parent_local(zero, origin_object)
    zero.data.materials.append(mat)
    return result


def create_joint_range_visual(
    name, joint_type, origin_object, axis, lower, upper,
    diagonal, axis_material, motion_material, width, fixed_material,
    trajectory_point=None,
):
    if joint_type in {"revolute", "continuous"}:
        _range_axis_visual(
            name, origin_object, axis, diagonal * 0.80, axis_material, width,
        )
        limits = (-math.pi, math.pi) if joint_type == "continuous" else (
            lower, upper,
        )
        return _range_revolute_arc(
            name, origin_object, axis, trajectory_point,
            *limits, motion_material, width,
        )
    if joint_type == "prismatic":
        return _range_prismatic(
            name, origin_object, axis, lower, upper,
            axis_material, width, diagonal * 0.012,
        )
    if joint_type == "fixed":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8, radius=diagonal * 0.012,
        )
        marker = bpy.context.object
        marker.name = f"joint_fixed::{name}"
        _parent_local(marker, origin_object)
        marker.data.materials.append(fixed_material)
        return marker
    raise ValueError(f"unsupported joint type: {joint_type}")


def set_local_matrix(obj, matrix):
    obj.matrix_parent_inverse.identity()
    obj.matrix_local = matrix


def y_up_to_z_up_matrix():
    return Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def transform_robot_basis(robot, basis):
    inverse = basis.inverted()
    rot = basis.to_3x3()
    for link in robot["links"].values():
        for visual in link["visuals"]:
            visual["origin"] = basis @ visual["origin"] @ inverse
    for joint in robot["joints"]:
        joint["origin"] = basis @ joint["origin"] @ inverse
        joint["axis"] = (rot @ joint["axis"]).normalized()
    return robot


def override_joint_range_deg(robot, range_deg):
    if range_deg is None:
        return robot
    lower, upper = (math.radians(float(value)) for value in range_deg)
    for joint in robot["joints"]:
        joint["lower"] = lower
        joint["upper"] = upper
    return robot


def mesh_objects_for_link(link_name):
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name.startswith(f"{link_name}_")]


def bbox_center(objects):
    mn, mx = bbox(objects)
    return (mn + mx) * 0.5


def signed_angle_about_axis(from_vec, to_vec, axis):
    axis = axis.normalized()
    from_vec = from_vec - axis * from_vec.dot(axis)
    to_vec = to_vec - axis * to_vec.dot(axis)
    if from_vec.length == 0.0 or to_vec.length == 0.0:
        return 0.0
    from_vec.normalize()
    to_vec.normalize()
    return math.atan2(axis.dot(from_vec.cross(to_vec)), from_vec.dot(to_vec))


def override_joint_range_for_included_angle(robot, motion_empty, included_angle_deg):
    if included_angle_deg is None:
        return robot
    for joint in robot["joints"]:
        parent_objects = mesh_objects_for_link(joint["parent"])
        child_objects = mesh_objects_for_link(joint["child"])
        if not parent_objects or not child_objects:
            continue
        motion = motion_empty[joint["name"]]
        origin = motion.matrix_world.translation
        axis = (motion.matrix_world.to_3x3() @ joint["axis"]).normalized()
        current = signed_angle_about_axis(
            bbox_center(parent_objects) - origin,
            bbox_center(child_objects) - origin,
            axis,
        )
        lower, upper = (math.radians(float(value)) - current for value in included_angle_deg)
        joint["lower"] = lower
        joint["upper"] = upper
    return robot


def build_kinematic_scene(robot, mesh_root):
    link_empty = {}
    motion_empty = {}
    joints_by_parent = {}
    for joint in robot["joints"]:
        joints_by_parent.setdefault(joint["parent"], []).append(joint)

    for name in robot["links"]:
        empty = bpy.data.objects.new(f"link_{name}", None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.06
        bpy.context.collection.objects.link(empty)
        link_empty[name] = empty

    for root in robot["roots"]:
        link_empty[root].matrix_world = Matrix.Identity(4)

    def attach_subtree(parent_link):
        for joint in joints_by_parent.get(parent_link, []):
            motion = bpy.data.objects.new(f"joint_{joint['name']}_motion", None)
            motion.empty_display_type = "SINGLE_ARROW"
            motion.empty_display_size = 0.08
            bpy.context.collection.objects.link(motion)
            motion.parent = link_empty[joint["parent"]]
            set_local_matrix(motion, joint["origin"])
            motion_empty[joint["name"]] = motion

            child = link_empty[joint["child"]]
            child.parent = motion
            set_local_matrix(child, Matrix.Identity(4))
            attach_subtree(joint["child"])

    for root in robot["roots"]:
        attach_subtree(root)

    for link_name, link in robot["links"].items():
        for i, visual in enumerate(link["visuals"]):
            mesh_path = resolve_mesh_path(visual["filename"], mesh_root)
            imported = import_mesh(mesh_path)
            holder = bpy.data.objects.new(f"visual_{link_name}_{i}", None)
            bpy.context.collection.objects.link(holder)
            holder.parent = link_empty[link_name]
            set_local_matrix(holder, visual["origin"])
            holder.scale = visual["scale"]
            for obj in imported:
                if obj.parent is None:
                    obj.parent = holder
                    obj.matrix_parent_inverse.identity()
                obj.name = f"{link_name}_{obj.name}"
                obj.show_transparent = True
    return link_empty, motion_empty


def build_local_laptop_scene(robot, mesh_root):
    if len(robot["joints"]) != 1:
        return build_kinematic_scene(robot, mesh_root)
    joint = robot["joints"][0]
    parent_link = robot["links"][joint["parent"]]
    child_link = robot["links"][joint["child"]]
    parent_imported = []
    child_imported = []
    for visual in parent_link["visuals"]:
        parent_imported.extend(import_mesh(resolve_mesh_path(visual["filename"], mesh_root)))
    for visual in child_link["visuals"]:
        child_imported.extend(import_mesh(resolve_mesh_path(visual["filename"], mesh_root)))

    rig = bpy.data.objects.new("laptop_instruct_urdf_root", None)
    bpy.context.collection.objects.link(rig)
    base_link = bpy.data.objects.new("laptop_0_base_link", None)
    bpy.context.collection.objects.link(base_link)
    base_link.parent = rig
    joint_frame = bpy.data.objects.new(f"laptop_0_joint_{joint['name']}_frame", None)
    bpy.context.collection.objects.link(joint_frame)
    joint_frame.parent = base_link
    joint_frame.matrix_local = joint_frame_matrix(joint["origin"], joint["axis"])
    joint_motion = bpy.data.objects.new(f"laptop_0_joint_{joint['name']}", None)
    bpy.context.collection.objects.link(joint_motion)
    joint_motion.parent = joint_frame
    for obj in root_objects(parent_imported):
        parent_keep_world(obj, base_link)
    for obj in root_objects(child_imported):
        parent_keep_world(obj, joint_motion)
    local_joint = dict(joint)
    local_joint["axis"] = Vector((1.0, 0.0, 0.0))
    local_joint["lower"] = 0.0
    local_joint["upper"] = joint["upper"] - joint["lower"]
    return {"laptop_0_base_link": base_link}, {joint["name"]: joint_motion}, {"joints": [local_joint]}


def child_subtree_links(robot, joint_name):
    joint = next(j for j in robot["joints"] if j["name"] == joint_name)
    children = {j["parent"]: j["child"] for j in robot["joints"]}
    result = {joint["child"]}
    changed = True
    while changed:
        changed = False
        for parent, child in children.items():
            if parent in result and child not in result:
                result.add(child)
                changed = True
    return result


def animate_joint(joint, motion, frame_start, frame_end, revolute_angle_deg, prismatic_distance, relative_joint_motion=False):
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    motion.rotation_mode = "QUATERNION"
    axis = joint["axis"].normalized()
    rest = motion.matrix_local.copy()
    if joint["type"] == "revolute":
        if relative_joint_motion:
            a0 = 0.0
            a1 = joint["upper"] - joint["lower"]
        else:
            a0 = joint["lower"]
            a1 = joint["upper"]
        for frame, angle in [(frame_start, a0), ((frame_start + frame_end) // 2, a1), (frame_end, a0)]:
            bpy.context.scene.frame_set(frame)
            motion.matrix_local = rest @ Matrix.Rotation(angle, 4, axis)
            motion.keyframe_insert(data_path="location", frame=frame)
            motion.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    elif joint["type"] == "prismatic":
        for frame, dist in [(frame_start, 0.0), ((frame_start + frame_end) // 2, prismatic_distance), (frame_end, 0.0)]:
            bpy.context.scene.frame_set(frame)
            motion.matrix_local = rest @ Matrix.Translation(axis * dist)
            motion.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            motion.keyframe_insert(data_path="location", frame=frame)


def apply_joint_pose(joint, motion, joint_pose):
    if joint_pose == "animated":
        return
    value = {"zero": 0.0, "lower": joint["lower"], "upper": joint["upper"]}[joint_pose]
    rest = motion.matrix_local.copy()
    if joint["type"] in {"revolute", "continuous"}:
        motion.rotation_mode = "QUATERNION"
        motion.matrix_local = rest @ Matrix.Rotation(
            value, 4, joint["axis"].normalized(),
        )
    elif joint["type"] == "prismatic":
        motion.matrix_local = rest @ Matrix.Translation(
            joint["axis"].normalized() * value,
        )


def visualize_joints(robot, motion_empty, target_joint_name, revolute_angle_deg, prismatic_distance, frame_start, frame_end, relative_joint_motion=False, axis_only=False, joint_pose="animated", hide_joint_visual=False):
    cyan = material("joint_cyan", (0.0, 0.9, 1.0, 1.0))
    for joint in robot["joints"]:
        if target_joint_name and joint["name"] != target_joint_name:
            continue
        motion = motion_empty[joint["name"]]
        origin = motion.matrix_world.copy()
        if not hide_joint_visual:
            create_axis_visual(joint["name"], origin, joint["axis"], mat=cyan)
            if axis_only:
                pass
            elif joint["type"] == "revolute":
                create_revolute_arc(joint["name"], origin, joint["axis"], 0.18, revolute_angle_deg, cyan)
            elif joint["type"] == "prismatic":
                create_prismatic_arrow(joint["name"], origin, joint["axis"], prismatic_distance, cyan)
        if joint_pose == "animated":
            animate_joint(joint, motion, frame_start, frame_end, revolute_angle_deg, prismatic_distance, relative_joint_motion)
        else:
            apply_joint_pose(joint, motion, joint_pose)


def bbox(objects):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        points.extend(obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box)
    mn = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    mx = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return mn, mx


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_view(camera_y_sign=-1.0, camera_view="front"):
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    mn, mx = bbox(meshes)
    center = (mn + mx) * 0.5
    extent = mx - mn
    light_data = bpy.data.lights.new("Key_Area", "AREA")
    light = bpy.data.objects.new("Key_Area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((0.0, -1.5, 1.2))
    light.data.energy = 350
    light.data.size = 5

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    distance = max(extent) * 2.2
    height = max(extent) * 1.0
    offsets = {
        "front": Vector((0.0, -distance, height)),
        "back": Vector((0.0, distance, height)),
        "side": Vector((distance, 0.0, height)),
        "top": Vector((0.0, -0.01 * distance, distance)),
        "iso": Vector((distance * 0.85, -distance * 0.75, height * 0.75)),
    }
    camera.location = center + offsets.get(camera_view, Vector((0.0, float(camera_y_sign) * distance, height)))
    look_at(camera, center)
    camera.data.angle = math.radians(45)
    bpy.context.scene.camera = camera

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (1, 1, 1)
    bpy.context.scene.render.resolution_x = 1400
    bpy.context.scene.render.resolution_y = 900
    engines = [e.identifier for e in bpy.context.scene.render.bl_rna.properties["engine"].enum_items]
    bpy.context.scene.render.engine = next(
        name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH") if name in engines
    )


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_meshes_transparent(alpha=0.35):
    mat = material("transparent_mesh_debug", (0.8, 0.8, 0.8, alpha))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def render_video(output_video):
    scene = bpy.context.scene
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found in PATH")
    frame_dir = Path(output_video).with_suffix("")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    scene.render.filepath = str(frame_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(scene.render.fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ],
        check=True,
    )


def run(
    urdf_path,
    mesh_root,
    target_joint_name,
    output_blend=OUTPUT_BLEND,
    output_preview=OUTPUT_PREVIEW,
    output_video=None,
    frame_start=FRAME_START,
    frame_end=FRAME_END,
    import_mode="generic",
    up_axis="z",
    relative_joint_motion=False,
    camera_y_sign=-1.0,
    camera_view="front",
    joint_pose="animated",
    joint_range_deg=None,
    included_angle_deg=None,
    axis_only=False,
    transparent_mesh=False,
    hide_joint_visual=False,
    save_blend=True,
):
    clear_scene()
    robot = parse_urdf(urdf_path)
    if up_axis == "y":
        robot = transform_robot_basis(robot, y_up_to_z_up_matrix())
    robot = override_joint_range_deg(robot, joint_range_deg)
    if import_mode == "generic":
        _, motion_empty = build_kinematic_scene(robot, mesh_root)
        animation_robot = robot
    else:
        _, motion_empty, animation_robot = build_local_laptop_scene(robot, mesh_root)
    animation_robot = override_joint_range_for_included_angle(animation_robot, motion_empty, included_angle_deg)
    if transparent_mesh:
        make_meshes_transparent()
    visualize_joints(animation_robot, motion_empty, target_joint_name, REVOLUTE_ANGLE_DEG, PRISMATIC_DISTANCE, frame_start, frame_end, relative_joint_motion, axis_only, joint_pose, hide_joint_visual)
    setup_view(camera_y_sign, camera_view)
    configure_specular_reflections(bpy)
    output_blend = Path(output_blend)
    output_preview = Path(output_preview)
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    if save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    bpy.context.scene.render.filepath = str(output_preview)
    bpy.ops.render.render(write_still=True)
    if output_video:
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        render_video(output_video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=str(URDF_PATH))
    parser.add_argument("--mesh-root", default=str(MESH_ROOT))
    parser.add_argument("--joint", default=TARGET_JOINT_NAME)
    parser.add_argument("--output-blend", default=str(OUTPUT_BLEND))
    parser.add_argument("--output-preview", default=str(OUTPUT_PREVIEW))
    parser.add_argument("--image-output-dir")
    parser.add_argument("--output-video")
    parser.add_argument("--frame-start", type=int, default=FRAME_START)
    parser.add_argument("--frame-end", type=int, default=FRAME_END)
    parser.add_argument("--import-mode", choices=("local", "generic"), default="generic")
    parser.add_argument("--up-axis", choices=("z", "y"), default="z")
    parser.add_argument("--relative-joint-motion", action="store_true")
    parser.add_argument("--camera-y-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--camera-view", choices=("front", "back", "side", "top", "iso"), default="front")
    parser.add_argument("--joint-pose", choices=("animated", "zero", "lower", "upper"), default="animated")
    parser.add_argument("--joint-range-deg", nargs=2, type=float, metavar=("LOWER", "UPPER"))
    parser.add_argument("--included-angle-deg", nargs=2, type=float, metavar=("LOWER", "UPPER"))
    parser.add_argument("--axis-only", action="store_true")
    parser.add_argument("--transparent-mesh", action="store_true")
    parser.add_argument("--hide-joint-visual", action="store_true")
    parser.add_argument("--preserve-materials", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args = parser.parse_args(argv)
    common = {
        "urdf_path": Path(args.urdf),
        "mesh_root": Path(args.mesh_root),
        "target_joint_name": args.joint,
        "output_blend": Path(args.output_blend),
        "output_video": Path(args.output_video) if args.output_video else None,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "import_mode": args.import_mode,
        "up_axis": args.up_axis,
        "relative_joint_motion": args.relative_joint_motion,
        "camera_y_sign": args.camera_y_sign,
        "camera_view": args.camera_view,
        "joint_range_deg": args.joint_range_deg,
        "included_angle_deg": args.included_angle_deg,
        "axis_only": args.axis_only,
        "transparent_mesh": args.transparent_mesh,
        "hide_joint_visual": args.hide_joint_visual,
    }
    if args.image_output_dir:
        output_dir = Path(args.image_output_dir)
        for filename, pose in (
            ("lower", "lower"), ("rest", "zero"), ("upper", "upper"),
        ):
            run(
                **common,
                output_preview=output_dir / f"{filename}.png",
                joint_pose=pose,
                save_blend=False,
            )
    else:
        run(
            **common,
            output_preview=Path(args.output_preview),
            joint_pose=args.joint_pose,
        )


if __name__ == "__main__":
    main()
