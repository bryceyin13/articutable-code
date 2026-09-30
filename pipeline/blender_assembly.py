#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gram.repair_math import bbox_scale_factors, rigid_axis_mapping_and_scale
from pipeline.urdf_coordinates import (
    blender_visual_matrix, uses_explicit_gltf_asset_basis)
from pipeline.common import mesh_path
from pipeline.render_settings import configure_specular_reflections


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"no mesh imported from {path}")
    roots = [obj for obj in imported if obj.parent not in imported]
    return imported, mesh_objects, roots


def import_mesh(path):
    path = Path(path)
    if path.suffix.lower() in {".glb", ".gltf"}:
        return import_glb(path)
    before = set(bpy.context.scene.objects)
    if path.suffix.lower() == ".obj":
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(
                filepath=str(path), forward_axis="Y", up_axis="Z",
            )
        else:
            bpy.ops.import_scene.obj(
                filepath=str(path), axis_forward="Y", axis_up="Z",
            )
    else:
        raise ValueError(f"unsupported articulated visual mesh format: {path.suffix}")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"no mesh imported from {path}")
    roots = [obj for obj in imported if obj.parent not in imported]
    return imported, mesh_objects, roots


def combined_bbox(mesh_objects):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in mesh_objects:
        obj_eval = obj.evaluated_get(depsgraph)
        points.extend(obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box)
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v


def glb_path_for(manifest_path, object_id):
    return mesh_path(manifest_path, object_id)


def set_most_open_animation_frame(meshes):
    ranges = [obj.animation_data.action.frame_range for obj in meshes if obj.animation_data and obj.animation_data.action]
    if not ranges:
        return None
    start = math.floor(min(frame_range[0] for frame_range in ranges))
    end = math.ceil(max(frame_range[1] for frame_range in ranges))
    candidates = range(start, end + 1, max(1, (end - start) // 120))
    best = None
    for frame in candidates:
        bpy.context.scene.frame_set(frame)
        min_v, max_v = combined_bbox(meshes)
        candidate = (float(max_v.z - min_v.z), frame)
        if best is None or candidate > best:
            best = candidate
    bpy.context.scene.frame_set(best[1])
    return best[1]


def repair_animated_part_sizes(meshes, object_blueprint):
    parts = {part["part_id"]: part for part in object_blueprint["articulation"]["parts"]}
    joint = object_blueprint["articulation"]["joint"]
    ranges = [obj.animation_data.action.frame_range for obj in meshes if obj.animation_data and obj.animation_data.action]
    start = math.floor(min(frame_range[0] for frame_range in ranges))
    end = math.ceil(max(frame_range[1] for frame_range in ranges))
    bpy.context.scene.frame_set(start)
    start_matrices = {obj: obj.matrix_world.copy() for obj in meshes}
    bpy.context.scene.frame_set(end)
    motion = {
        obj: (obj.matrix_world.translation - start_matrices[obj].translation).length
        + obj.matrix_world.to_quaternion().rotation_difference(start_matrices[obj].to_quaternion()).angle
        for obj in meshes
    }
    child = max(meshes, key=motion.get)
    parent = min(meshes, key=motion.get)
    def attachment_candidates(obj):
        mins = [min(corner[i] for corner in obj.bound_box) for i in range(3)]
        maxs = [max(corner[i] for corner in obj.bound_box) for i in range(3)]
        extents = [maxs[i] - mins[i] for i in range(3)]
        ranked_axes = sorted(range(3), key=lambda index: extents[index], reverse=True)
        edge_axis = ranked_axes[1]
        thickness_axis = ranked_axes[2]
        center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
        points = []
        for side in (mins[edge_axis], maxs[edge_axis]):
            for surface in (mins[thickness_axis], maxs[thickness_axis]):
                point = center[:]
                point[edge_axis] = side
                point[thickness_axis] = surface
                local = Vector(point)
                points.append((local, obj.matrix_world @ local))
        return points

    parent_candidates = attachment_candidates(parent)
    child_candidates = attachment_candidates(child)
    _, parent_anchor, child_anchor = min(
        ((parent_world - child_world).length, parent_local, child_local)
        for parent_local, parent_world in parent_candidates
        for child_local, child_world in child_candidates
    )
    anchors = {parent: parent_anchor, child: child_anchor}
    for obj, part_id in ((parent, joint["parent_part"]), (child, joint["child_part"])):
        extents = [max(corner[i] for corner in obj.bound_box) - min(corner[i] for corner in obj.bound_box) for i in range(3)]
        target = sorted((float(value) for value in parts[part_id]["bbox_cm"]), reverse=True)
        desired = [0.0, 0.0, 0.0]
        for rank, axis in enumerate(sorted(range(3), key=lambda index: extents[index], reverse=True)):
            desired[axis] = target[rank] / target[0]
        factors = [desired[i] / extents[i] for i in range(3)]
        anchor = anchors[obj]
        for vertex in obj.data.vertices:
            for axis in range(3):
                vertex.co[axis] = anchor[axis] + factors[axis] * (vertex.co[axis] - anchor[axis])
        obj.data.update()
        obj["blueprint_part_id"] = part_id
        obj["relative_size_scale_xyz"] = [float(value) for value in factors]


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
            (x_axis.x, y_axis.x, z_axis.x, origin.x),
            (x_axis.y, y_axis.y, z_axis.y, origin.y),
            (x_axis.z, y_axis.z, z_axis.z, origin.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def configure_joint_control(motion, joint):
    lower, upper = joint["lower"], joint["upper"]
    value = min(max(0.0, lower), upper)
    motion["joint_value"] = value
    motion.id_properties_ui("joint_value").update(
        min=lower, max=upper, soft_min=lower, soft_max=upper, precision=4,
        description=f"{joint['type']} position from URDF limits",
    )
    motion.empty_display_type = "CIRCLE"
    motion.empty_display_size = 0.08
    motion.show_name = True

    def drive(curve, expression):
        variable = curve.driver.variables.new()
        variable.name = "value"
        variable.type = "SINGLE_PROP"
        variable.targets[0].id = motion
        variable.targets[0].data_path = '["joint_value"]'
        curve.driver.expression = expression

    if joint["type"] in {"revolute", "continuous"}:
        motion.rotation_mode = "AXIS_ANGLE"
        motion.rotation_axis_angle = (value, 1.0, 0.0, 0.0)
        drive(motion.driver_add("rotation_axis_angle", 0), "value")
    elif joint["type"] == "prismatic":
        drive(motion.driver_add("location", 0), "value")


def parse_urdf_package(package):
    xml = ET.parse(package / "model.urdf").getroot()
    links = {}
    for link in xml.findall("link"):
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh") if visual is not None else None
        if mesh is None:
            raise RuntimeError(f"link {link.attrib['name']} has no visual mesh")
        origin = visual.find("origin")
        links[link.attrib["name"]] = {
            "path": package / mesh.attrib["filename"],
            "xyz": tuple(float(value) for value in (
                origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            ).split()),
            "rpy": tuple(float(value) for value in (
                origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            ).split()),
            "scale": tuple(float(value) for value in mesh.attrib.get("scale", "1 1 1").split()),
        }
    joints = []
    for element in xml.findall("joint"):
        origin, axis, limit = element.find("origin"), element.find("axis"), element.find("limit")
        joints.append({
            "name": element.attrib["name"],
            "type": element.attrib.get("type", "fixed"),
            "parent": element.find("parent").attrib["link"],
            "child": element.find("child").attrib["link"],
            "origin": Vector(float(value) for value in (
                origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            ).split()),
            "axis": Vector(float(value) for value in (
                axis.attrib.get("xyz", "1 0 0") if axis is not None else "1 0 0"
            ).split()).normalized(),
            "lower": float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
            "upper": float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
        })
    children = {joint["child"] for joint in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise RuntimeError(f"URDF must have one root link, found {len(roots)}")
    return links, joints, roots[0]


def import_articulated_asset(package, object_id):
    links, joints, root_link = parse_urdf_package(package)
    compensate_gltf_basis = uses_explicit_gltf_asset_basis(package)
    rig = bpy.data.objects.new(f"{object_id}_urdf_root", None)
    bpy.context.collection.objects.link(rig)
    frames, imported, meshes, root_meshes = {}, [], [], []

    root_frame = bpy.data.objects.new(f"{object_id}_{root_link}", None)
    bpy.context.collection.objects.link(root_frame)
    root_frame.parent = rig
    frames[root_link] = root_frame
    pending = list(joints)
    while pending:
        joint = next((item for item in pending if item["parent"] in frames), None)
        if joint is None:
            raise RuntimeError("URDF joint graph is disconnected")
        axis_frame = joint_frame_matrix(joint["origin"], joint["axis"])
        frame = bpy.data.objects.new(f"{object_id}_joint_{joint['name']}_frame", None)
        motion = bpy.data.objects.new(f"{object_id}_joint_{joint['name']}", None)
        child = bpy.data.objects.new(f"{object_id}_{joint['child']}", None)
        for obj in (frame, motion, child):
            bpy.context.collection.objects.link(obj)
        frame.parent, frame.matrix_local = frames[joint["parent"]], axis_frame
        motion.parent = frame
        child.parent = motion
        child.matrix_local = axis_frame.to_3x3().to_4x4().inverted()
        motion["joint_name"] = joint["name"]
        motion["joint_type"], motion["joint_axis"] = joint["type"], [float(v) for v in joint["axis"]]
        motion["joint_lower"], motion["joint_upper"] = joint["lower"], joint["upper"]
        configure_joint_control(motion, joint)
        frames[joint["child"]] = child
        pending.remove(joint)

    for link_name, visual in links.items():
        mesh_path = visual["path"]
        link_imported, link_meshes, link_roots = import_mesh(mesh_path)
        imported.extend(link_imported)
        meshes.extend(link_meshes)
        if link_name == root_link:
            root_meshes.extend(link_meshes)
        holder = bpy.data.objects.new(f"{object_id}_{link_name}_visual", None)
        bpy.context.collection.objects.link(holder)
        holder.parent = frames[link_name]
        holder.matrix_local = blender_visual_matrix(
            visual["xyz"], visual["rpy"], visual["path"],
            compensate_gltf_basis)
        holder.scale = visual["scale"]
        for obj in link_roots:
            obj.parent = holder
            obj.matrix_parent_inverse.identity()
    root_min, root_max = combined_bbox(root_meshes)
    rig["placement_anchor_before_scale"] = [float((root_min.x + root_max.x) * 0.5), float((root_min.y + root_max.y) * 0.5), float(root_min.z)]
    return str(package), str(package / "model.urdf"), imported, meshes, [rig]


def import_asset(root, manifest, object_id, blueprint_path, articulation_packages):
    if object_id in articulation_packages:
        return import_articulated_asset(articulation_packages[object_id], object_id)

    glb_path = glb_path_for(manifest, object_id)
    imported, meshes, roots = import_glb(glb_path)
    source_asset = str(glb_path.relative_to(root))
    articulation_asset = None
    return source_asset, articulation_asset, imported, meshes, roots


def target_xy_yaw(pose, object_id):
    item = pose[object_id]
    size_m = [v / 100.0 for v in item["scale_cm"]]
    x_m, y_m, _ = [v / 100.0 for v in item["translation_cm"]]
    return size_m, x_m, y_m, float(item.get("yaw_deg", 0.0))


def add_object(root, manifest, pose, object_id, support_top_m, blueprint_path, articulation_packages):
    source_asset, articulation_asset, imported, meshes, roots = import_asset(root, manifest, object_id, blueprint_path, articulation_packages)
    min_v, max_v = combined_bbox(meshes)
    center = Vector(roots[0].get("placement_anchor_before_scale", (min_v + max_v) * 0.5))
    extents = max_v - min_v

    group = bpy.data.objects.new(object_id, None)
    bpy.context.collection.objects.link(group)
    group["bbox_center_before_scale"] = [float(v) for v in center]
    group["bbox_extent_before_scale"] = [float(v) for v in extents]
    hinge_origin = roots[0].get("hinge_origin_before_scale")
    if hinge_origin:
        group["hinge_origin_before_scale"] = [float(v) for v in hinge_origin]

    for obj in roots:
        obj.matrix_world.translation -= center
        obj.parent = group
        obj.matrix_parent_inverse = group.matrix_world.inverted()

    size_m, x_m, y_m, yaw_deg = target_xy_yaw(pose, object_id)
    axis_mapping_turns = 0
    if object_id in articulation_packages:
        scales = bbox_scale_factors(extents, size_m, preserve_proportions=True)
        mapped_extents = extents
    else:
        axis_mapping_turns, uniform_scale = rigid_axis_mapping_and_scale(extents, size_m)
        scales = (uniform_scale, uniform_scale, uniform_scale)
        mapped_extents = (
            (extents.y, extents.x, extents.z)
            if axis_mapping_turns
            else extents
        )
    actual_size_m = [float(mapped_extents[i] * scales[i]) for i in range(3)]
    if object_id == "table_0":
        z_m = actual_size_m[2] / 2.0
    elif object_id in articulation_packages:
        z_m = support_top_m
    else:
        z_m = support_top_m + actual_size_m[2] / 2.0
    loc_m = (x_m, y_m, z_m)

    group.scale = scales
    yaw_offset_deg = 90.0 * axis_mapping_turns
    group.rotation_euler[2] = math.radians(yaw_deg + yaw_offset_deg)
    group.location = loc_m

    for obj in imported:
        obj.name = f"{object_id}_{obj.name}"

    return {
        "object_id": object_id,
        "source_asset": source_asset,
        "articulation_asset": articulation_asset,
        "target_size_m": size_m,
        "actual_size_m": actual_size_m,
        "location_m": list(loc_m),
        "yaw_deg": yaw_deg,
        "yaw_offset_deg": yaw_offset_deg,
        "axis_mapping_quarter_turns": axis_mapping_turns,
        "dimension_error_m": [actual_size_m[i] - size_m[i] for i in range(3)],
        "bbox_extent_before_scale": [float(v) for v in extents],
        "scale_factor": float(scales[0]) if scales[0] == scales[1] == scales[2] else None,
        "scale_xyz": list(scales),
    }


def gram_joint_visualizer(root):
    name = "tabletop_gram_joint_visualizer"
    path = (
        Path(root) / "gram/visualize_urdf_joint.py"
    ).resolve(strict=True)
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve() != path:
            raise RuntimeError(f"conflicting GRAM visualizer: {module.__file__}")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GRAM visualizer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def add_joint_visualizations(root, object_id, articulation_package):
    if articulation_package is None:
        return []
    urdfviz = gram_joint_visualizer(root)

    group = bpy.data.objects.get(object_id)
    if group is None:
        return [{"object_id": object_id, "enabled": True, "created": False, "reason": "missing object group"}]
    bpy.context.view_layer.update()
    axis_material, motion_material, fixed_material = urdfviz.joint_range_materials()
    meshes = [obj for obj in group.children_recursive if obj.type == "MESH"]
    mn, mx = combined_bbox(meshes)
    diagonal = max((mx - mn).length, 1e-4)
    _, joints, _ = parse_urdf_package(Path(articulation_package))
    joint_subtypes = urdfviz.package_joint_subtypes(articulation_package)
    for obj in list(bpy.data.objects):
        if obj.get("joint_visualization_object_id") == object_id:
            bpy.data.objects.remove(obj, do_unlink=True)
    results = []
    for joint in joints:
        motion = bpy.data.objects.get(f"{object_id}_joint_{joint['name']}")
        if motion is None:
            continue
        before = set(bpy.data.objects)
        location, rotation, _ = motion.parent.matrix_world.decompose()
        origin = bpy.data.objects.new(
            f"joint_range_origin::{object_id}_{joint['name']}", None,
        )
        bpy.context.scene.collection.objects.link(origin)
        origin.matrix_world = (
            Matrix.Translation(location) @ rotation.to_matrix().to_4x4()
        )
        trajectory_point = None
        if joint["type"] in {"revolute", "continuous"}:
            child_frame = bpy.data.objects.get(f"{object_id}_{joint['child']}")
            child_visual = bpy.data.objects.get(
                f"{object_id}_{joint['child']}_visual"
            )
            if child_frame is None or child_visual is None:
                raise RuntimeError(
                    f"cannot find child visual for joint {object_id}:{joint['name']}"
                )
            child_meshes = [
                obj for obj in child_visual.children_recursive
                if obj.type == "MESH"
            ]
            point_at_zero = urdfviz._child_trajectory_point(
                motion, child_meshes, Vector((1.0, 0.0, 0.0)),
                joint_subtypes.get(joint["name"]) == "hinge",
            )
            point_world_at_zero = motion.parent.matrix_world @ point_at_zero
            trajectory_point = origin.matrix_world.inverted() @ point_world_at_zero
        urdfviz.create_joint_range_visual(
            f"{object_id}_{joint['name']}", joint["type"], origin,
            Vector((1.0, 0.0, 0.0)), joint["lower"], joint["upper"],
            diagonal, axis_material=axis_material,
            motion_material=motion_material, width=diagonal * 0.006,
            fixed_material=fixed_material, trajectory_point=trajectory_point,
        )
        for obj in set(bpy.data.objects) - before:
            obj["joint_visualization"] = True
            obj["joint_visualization_object_id"] = object_id
            obj.hide_render = True
        results.append({
            "object_id": object_id, "enabled": True, "created": True,
            "joint_name": joint["name"], "joint_type": joint["type"],
            "joint_axis": [float(v) for v in joint["axis"]],
            "joint_limit": [joint["lower"], joint["upper"]],
            "source": "articulation_manifest",
        })
    return results


def remove_joint_visualizations():
    for obj in list(bpy.data.objects):
        if obj.get("joint_visualization"):
            bpy.data.objects.remove(obj, do_unlink=True)


def iter_descendants(obj):
    for child in obj.children:
        yield child
        yield from iter_descendants(child)


def add_joint_limit_ghost(object_id, joint_name, motion_empty, angle_rad, mat):
    screen_root = bpy.data.objects.get(f"{object_id}_world.001")
    joint_frame = motion_empty.parent
    if screen_root is None or joint_frame is None:
        return []
    transform = joint_frame.matrix_world @ Matrix.Rotation(angle_rad, 4, "X") @ joint_frame.matrix_world.inverted()
    ghosts = []
    for obj in iter_descendants(screen_root):
        if obj.type != "MESH":
            continue
        ghost = obj.copy()
        ghost.data = obj.data.copy()
        ghost.name = f"{object_id}_{joint_name}_limit_ghost"
        ghost.animation_data_clear()
        ghost.data.materials.clear()
        ghost.data.materials.append(mat)
        ghost.matrix_world = transform @ obj.matrix_world
        bpy.context.collection.objects.link(ghost)
        ghosts.append(ghost)
    return ghosts


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def set_white_world(scene):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("WhiteWorld")
    scene.world.color = (1.0, 1.0, 1.0)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = 1.0


def bbox_corners(min_v, max_v):
    return [
        Vector((x, y, z))
        for x in (min_v.x, max_v.x)
        for y in (min_v.y, max_v.y)
        for z in (min_v.z, max_v.z)
    ]


def camera_from_blueprint(camera_cfg, scene_min, scene_max, table_top_m):
    azimuth = math.radians(float(camera_cfg.get("azimuth_deg", 0.0)))
    elevation = math.radians(float(camera_cfg.get("elevation_deg", 55.0)))

    target = Vector((0.0, 0.0, table_top_m * 0.5))
    extent = scene_max - scene_min
    scene_radius = max(extent.x, extent.y, extent.z) * 0.5
    multipliers = {"near": 2.0, "medium": 2.7, "far": 3.2}
    distance = scene_radius * multipliers.get(camera_cfg.get("distance_level", "medium"), 2.7)
    horizontal = distance * math.cos(elevation)
    return (
        target
        + Vector(
            (
                math.sin(azimuth) * horizontal,
                -math.cos(azimuth) * horizontal,
                distance * math.sin(elevation),
            )
        ),
        target,
    )


def fit_camera_to_scene(camera, target, scene_min, scene_max, margin=0.08):
    corners = bbox_corners(scene_min, scene_max)
    direction = (camera.location - target).normalized()
    distance = (camera.location - target).length
    for _ in range(80):
        bpy.context.view_layer.update()
        coords = [world_to_camera_view(bpy.context.scene, camera, corner) for corner in corners]
        min_x = min(coord.x for coord in coords)
        max_x = max(coord.x for coord in coords)
        min_y = min(coord.y for coord in coords)
        max_y = max(coord.y for coord in coords)
        if min_x >= margin and max_x <= 1.0 - margin and min_y >= margin and max_y <= 1.0 - margin:
            break
        distance *= 1.06
        camera.location = target + direction * distance
        look_at(camera, target)
    return distance


def setup_camera_and_light(blueprint, scene_min, scene_max, table_top_m):
    light_data = bpy.data.lights.new("Key_Area", type="AREA")
    light = bpy.data.objects.new("Key_Area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, -1.8, 2.8)
    light.data.energy = 75
    light.data.size = 6.0

    fill_data = bpy.data.lights.new("Fill_Area", type="AREA")
    fill = bpy.data.objects.new("Fill_Area", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (1.8, 1.2, 2.0)
    fill.data.energy = 18
    fill.data.size = 6.0

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location, target = camera_from_blueprint(blueprint.get("camera", {}), scene_min, scene_max, table_top_m)
    look_at(camera, target)
    camera.data.type = "PERSP"
    camera.data.angle = math.radians(48.0)
    distance = fit_camera_to_scene(camera, target, scene_min, scene_max)
    bpy.context.scene.camera = camera
    return {
        "location_m": [float(v) for v in camera.location],
        "target_m": [float(v) for v in target],
        "fov_deg": 48.0,
        "distance_m": float(distance),
        "azimuth_deg": blueprint.get("camera", {}).get("azimuth_deg", 0),
        "elevation_deg": blueprint.get("camera", {}).get("elevation_deg", 55),
    }


def render_preview(path):
    scene = bpy.context.scene
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 1152
    engines = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_WORKBENCH"
    set_white_world(scene)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--image-to-3d-manifest", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--pose-scale", required=True)
    parser.add_argument("--articulation-manifest", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--output-preview", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-joint-viz", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    root = Path(args.root)
    object_ids = [item for item in args.objects.split(",") if item]
    manifest = Path(args.image_to_3d_manifest)
    pose = read_json(args.pose_scale)
    blueprint = read_json(args.blueprint)
    articulation_manifest = Path(args.articulation_manifest)
    articulation_root = articulation_manifest.parents[1]
    articulation_packages = {
        item["object_id"]: articulation_root / item["package"]
        for item in read_json(articulation_manifest).get("items", [])
    }

    clear_scene()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    ordered_ids = []
    if "table_0" in object_ids:
        ordered_ids.append("table_0")
    ordered_ids.extend(object_id for object_id in object_ids if object_id != "table_0")

    items = []
    table_top_m = pose["table_0"]["scale_cm"][2] / 100.0
    for object_id in ordered_ids:
        item = add_object(root, manifest, pose, object_id, table_top_m, args.blueprint, articulation_packages)
        items.append(item)
        if object_id == "table_0":
            table_top_m = item["actual_size_m"][2]
    joint_visualization = {"enabled": not args.no_joint_viz, "items": []}
    scene_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    scene_min, scene_max = combined_bbox(scene_meshes)
    camera_info = setup_camera_and_light(blueprint, scene_min, scene_max, table_top_m)
    set_white_world(bpy.context.scene)
    configure_specular_reflections(bpy)

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.output_glb, export_format="GLB")
    if not args.no_render:
        render_preview(args.output_preview)
    if not args.no_joint_viz:
        joint_visualization["items"] = [
            item
            for object_id in ordered_ids
            for item in add_joint_visualizations(
                root, object_id, articulation_packages.get(object_id)
            )
        ]
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)

    write_json(
        args.output_manifest,
        {
            "output_glb": str(Path(args.output_glb).relative_to(root)),
            "output_blend": str(Path(args.output_blend).relative_to(root)),
            "output_preview": str(Path(args.output_preview).relative_to(root)),
            "camera": camera_info,
            "items": items,
            "joint_visualization": joint_visualization,
        },
    )


if __name__ == "__main__":
    main()
