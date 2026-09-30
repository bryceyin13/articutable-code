#!/usr/bin/env python3
import json
import hashlib
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.blender_assembly import import_articulated_asset as import_assembled_asset
from pipeline.render_settings import configure_specular_reflections
from psgsr.support_pivot import support_pivot
from gram.urdf_assets import validate_rotation_matrix


ROOT = Path(
    os.environ.get("TABLETOP_ALIGNMENT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SCENE_PATH = Path(os.environ.get(
    "TABLETOP_SCENE_PATH", ROOT / "outputs/unified_scene_alignment/scene_alignment.json"
))
OUTPUT_DIR = Path(os.environ.get(
    "TABLETOP_RENDER_OUTPUT_DIR", ROOT / "outputs/unified_scene_alignment/render"
))
_ORIGINAL_MESH_VERTICES = {}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    _ORIGINAL_MESH_VERTICES.clear()


def import_asset(path, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = list(set(bpy.data.objects) - before)
    controller = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(controller)
    for obj in objects:
        if obj.parent is None:
            obj.parent = controller
    return controller, objects


def import_articulated_urdf_asset(path, name, orientation_matrix=None):
    before = set(bpy.data.objects)
    import_assembled_asset(Path(path), name)
    objects = list(set(bpy.data.objects) - before)
    controller = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(controller)
    roots = [obj for obj in objects if obj.parent not in objects]
    for obj in roots:
        obj.parent = controller
    if orientation_matrix is not None:
        orientation = Matrix(
            validate_rotation_matrix(orientation_matrix),
        ).to_4x4()
        for obj in roots:
            obj.matrix_basis = orientation @ obj.matrix_basis
        bpy.context.view_layer.update()
    return controller, objects


def apply_articulated_joint_state(objects, item):
    articulation = item.get("articulation")
    states = item.get("joint_states") or [item.get("joint_state", {})]
    states = {state.get("name"): state for state in states}
    if not articulation:
        targets = {
            obj.get("joint_name"): obj
            for obj in objects if obj.get("joint_name")
        }
        for name, state in states.items():
            target = targets.get(name)
            if target is None or "joint_value" not in target:
                continue
            target["joint_value"] = float(state.get(
                "scene_current_q",
                state.get(
                    "relative_angle_rad",
                    state.get("relative_displacement", 0.0),
                ),
            ))
            target.update_tag()
        bpy.context.view_layer.update()
        return
    joints = articulation.get("joints") or [articulation["joint"]]
    if all(joint.get("blender_object") for joint in joints):
        for joint in joints:
            state = states.get(joint["name"], {})
            target = next(
                (obj for obj in objects if obj.name == joint["blender_object"]), None)
            if target is None:
                raise ValueError(
                    f"missing assembled Blender joint {joint['blender_object']}")
            joint_type = joint.get("type", "revolute")
            if joint_type == "revolute":
                value = float(state.get("relative_angle_rad", 0.0))
            elif joint_type == "prismatic":
                value = float(state.get(
                    "scene_current_q", state.get("relative_displacement", 0.0)))
            else:
                raise ValueError(f"unsupported assembled joint type: {joint_type}")
            if "joint_value" in target:
                target["joint_value"] = value
                target.update_tag()
            elif joint_type == "revolute":
                target.rotation_mode = "AXIS_ANGLE"
                target.rotation_axis_angle = (value, 1.0, 0.0, 0.0)
            else:
                target.location.x = value
        bpy.context.view_layer.update()
        return
    joint = joints[0]
    state = next(iter(states.values()), {})
    if joint.get("blender_object"):
        target = next((obj for obj in objects if obj.name == joint["blender_object"]), None)
        if target is None:
            raise ValueError(f"missing assembled Blender joint {joint['blender_object']}")
        joint_type = joint.get("type", "revolute")
        if joint_type == "revolute":
            value = float(state.get("relative_angle_rad", 0.0))
        elif joint_type == "prismatic":
            value = float(state.get(
                "scene_current_q", state.get("relative_displacement", 0.0)))
        else:
            raise ValueError(f"unsupported assembled joint type: {joint_type}")
        if "joint_value" in target:
            target["joint_value"] = value
            target.update_tag()
        elif joint_type == "revolute":
            target.rotation_mode = "AXIS_ANGLE"
            target.rotation_axis_angle = (value, 1.0, 0.0, 0.0)
        else:
            target.location.x = value
        bpy.context.view_layer.update()
        return
    child_name = articulation["child_node"]
    target = next((obj for obj in objects if obj.name == child_name), None)
    if target is None:
        raise ValueError(f"missing native animated Blender node {child_name}")
    relative_angle = float(state.get("relative_angle_rad", 0.0))
    frame = float(joint["initial_frame"]) + relative_angle * float(joint["frame_per_rad"])
    start, end = (float(value) for value in joint["animation_frame_range"])
    frame = min(max(frame, start), end)
    integer_frame = math.floor(frame)
    scene = bpy.context.scene
    scene.frame_set(integer_frame, subframe=frame - integer_frame)
    bpy.context.view_layer.update()


def bounds(objects):
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(corner) for obj in objects if hasattr(obj, "bound_box") for corner in obj.bound_box]
    return Vector(map(min, zip(*points))), Vector(map(max, zip(*points)))


def rebase_controller_to_support(controller, objects):
    if controller.get("support_pivot_rebased"):
        return
    bpy.context.view_layer.update()
    vertices = [
        tuple(obj.matrix_world @ vertex.co)
        for obj in objects if obj.type == "MESH"
        for vertex in obj.data.vertices
    ]
    if not vertices:
        return
    pivot, _ = support_pivot(np.asarray(vertices, dtype=float))
    roots = [obj for obj in objects if obj.parent == controller]
    world_matrices = {obj: obj.matrix_world.copy() for obj in roots}
    controller.location = Vector(pivot)
    bpy.context.view_layer.update()
    for obj, matrix in world_matrices.items():
        obj.matrix_world = matrix
    controller["support_pivot_rebased"] = True
    bpy.context.view_layer.update()


def place_asset(
    controller, objects, scale, location, yaw_deg=0, z_anchor="bottom",
    roll_x_deg=0.0, roll_y_deg=0.0,
):
    if z_anchor == "bottom":
        rebase_controller_to_support(controller, objects)
        controller.scale = scale
        controller.rotation_mode = "XYZ"
        controller.rotation_euler[0] = math.radians(roll_x_deg)
        controller.rotation_euler[1] = math.radians(roll_y_deg)
        controller.rotation_euler[2] = math.radians(yaw_deg)
        controller.location = Vector(location)
        bpy.context.view_layer.update()
        return
    controller.scale = scale
    controller.rotation_mode = "XYZ"
    controller.rotation_euler[0] = math.radians(roll_x_deg)
    controller.rotation_euler[1] = math.radians(roll_y_deg)
    controller.rotation_euler[2] = math.radians(yaw_deg)
    bpy.context.view_layer.update()
    low, high = bounds(objects)
    center = (low + high) * 0.5
    anchor_z = high.z if z_anchor == "top" else low.z
    controller.location += Vector((location[0] - center.x, location[1] - center.y, location[2] - anchor_z))
    bpy.context.view_layer.update()


def scale_asset_below_top(controller, objects, anchor_from_top, scale_ratio):
    """Stretch only geometry below a fixed horizontal plane."""
    meshes = [obj for obj in objects if obj.type == "MESH"]
    for obj in meshes:
        key = obj.as_pointer()
        if key not in _ORIGINAL_MESH_VERTICES:
            obj.data = obj.data.copy()
            _ORIGINAL_MESH_VERTICES[key] = [
                vertex.co.copy() for vertex in obj.data.vertices
            ]
        for vertex, coordinate in zip(
            obj.data.vertices, _ORIGINAL_MESH_VERTICES[key],
        ):
            vertex.co = coordinate
    bpy.context.view_layer.update()
    controller_inverse = controller.matrix_world.inverted()
    transforms = {
        obj: controller_inverse @ obj.matrix_world
        for obj in meshes
    }
    top = max(
        (transforms[obj] @ coordinate).z
        for obj in meshes
        for coordinate in _ORIGINAL_MESH_VERTICES[obj.as_pointer()]
    )
    anchor = top + float(anchor_from_top)
    for obj in meshes:
        transform = transforms[obj]
        inverse = transform.inverted()
        for vertex, coordinate in zip(
            obj.data.vertices, _ORIGINAL_MESH_VERTICES[obj.as_pointer()],
        ):
            point = transform @ coordinate
            if point.z < anchor:
                point.z = anchor + (point.z - anchor) * float(scale_ratio)
            vertex.co = inverse @ point
        obj.data.update()
    bpy.context.view_layer.update()


def reanchor_asset(controller, objects, location):
    if controller.get("support_pivot_rebased"):
        controller.location = Vector(location)
        bpy.context.view_layer.update()
        return
    low, high = bounds(objects)
    center = (low + high) * 0.5
    controller.location += Vector((location[0] - center.x, location[1] - center.y, location[2] - low.z))
    bpy.context.view_layer.update()


def set_articulated_initial_state(camera, controller, objects, target_bbox, image_size, location):
    actions = {
        obj.animation_data.action
        for obj in objects
        if obj.animation_data and obj.animation_data.action
    }
    if not actions:
        return None
    start = math.floor(min(action.frame_range[0] for action in actions))
    end = math.ceil(max(action.frame_range[1] for action in actions))
    width, height = image_size
    target = (
        target_bbox[0] / width,
        1.0 - target_bbox[3] / height,
        target_bbox[2] / width,
        1.0 - target_bbox[1] / height,
    )
    scene = bpy.context.scene
    best = None
    base_yaw = controller.rotation_euler.z
    scene.frame_set(start)
    for yaw_offset in (0, 180):
        controller.rotation_euler.z = base_yaw + math.radians(yaw_offset)
        reanchor_asset(controller, objects, location)
        bpy.context.view_layer.update()
        projected = [
            world_to_camera_view(scene, camera, obj.matrix_world @ Vector(corner))
            for obj in objects
            if obj.type == "MESH"
            for corner in obj.bound_box
        ]
        if not projected:
            continue
        candidate = (
            min(point.x for point in projected),
            min(point.y for point in projected),
            max(point.x for point in projected),
            max(point.y for point in projected),
        )
        loss = sum((candidate[index] - target[index]) ** 2 for index in range(4))
        if best is None or loss < best[0]:
            best = (loss, yaw_offset)
    controller.rotation_euler.z = base_yaw + math.radians(best[1])
    reanchor_asset(controller, objects, location)
    return {"initial_frame": start, "yaw_offset_deg": best[1], "bbox_loss": best[0], "action_range": [start, end]}


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def create_camera(name):
    data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    return camera


def configure_vggt_camera(camera, frame, view, image_size, intrinsic=None):
    world_from_vggt = Matrix(frame["rotation_world_from_vggt"])
    world_scale = float(frame["scale_world_per_vggt"])
    world_offset = Vector(frame["translation_world"])
    extrinsic = view["extrinsic_world_to_camera"]
    camera_from_vggt = Matrix([row[:3] for row in extrinsic[:3]])
    camera_translation = Vector([row[3] for row in extrinsic[:3]])
    camera_center_vggt = -(camera_from_vggt.transposed() @ camera_translation)
    camera.location = world_offset + world_scale * (world_from_vggt @ camera_center_vggt)

    # OpenCV camera axes are +X right, +Y down, +Z forward; Blender uses
    # +X right, +Y up, -Z forward.
    blender_to_opencv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (
        world_from_vggt @ camera_from_vggt.transposed() @ blender_to_opencv
    ).to_quaternion()

    width, height = image_size
    intrinsic = intrinsic or view["intrinsic"]
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = intrinsic[0][0] * camera.data.sensor_width / width
    camera.data.shift_x = 0.5 - intrinsic[0][2] / width
    camera.data.shift_y = intrinsic[1][2] / height - 0.5
    scene = bpy.context.scene
    fx_pixels = intrinsic[0][0] * scene.render.resolution_x / width
    fy_pixels = intrinsic[1][1] * scene.render.resolution_y / height
    camera["pixel_aspect_y"] = fx_pixels / fy_pixels


def configure_front_orthographic(camera, objects, aspect, margin=1.1):
    bpy.context.view_layer.update()
    local_points = [
        camera.matrix_world.inverted() @ (obj.matrix_world @ Vector(corner))
        for obj in objects
        if hasattr(obj, "bound_box")
        for corner in obj.bound_box
    ]
    low_x, high_x = min(point.x for point in local_points), max(point.x for point in local_points)
    low_y, high_y = min(point.y for point in local_points), max(point.y for point in local_points)
    local_center = Vector(((low_x + high_x) * 0.5, (low_y + high_y) * 0.5, 0.0))
    camera.location += camera.rotation_quaternion @ local_center
    camera.data.type = "ORTHO"
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    camera.data.ortho_scale = max(high_y - low_y, (high_x - low_x) / aspect) * margin
    bpy.context.view_layer.update()
    projected = [
        world_to_camera_view(bpy.context.scene, camera, obj.matrix_world @ Vector(corner))
        for obj in objects
        if hasattr(obj, "bound_box")
        for corner in obj.bound_box
    ]
    normalized_span = max(
        max(point.x for point in projected) - min(point.x for point in projected),
        max(point.y for point in projected) - min(point.y for point in projected),
    )
    camera.data.ortho_scale *= normalized_span * margin


def add_shadow_ground(z):
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, z))
    ground = bpy.context.object
    ground.name = "Shadow_Ground"
    material = bpy.data.materials.new("Shadow_Ground_White")
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    ground.data.materials.append(material)
    return ground


def setup_ambient_occlusion_compositor(strength=0.35):
    scene = bpy.context.scene
    bpy.context.view_layer.use_pass_ambient_occlusion = True
    tree = bpy.data.node_groups.new("Contact_Shadow_Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    multiply = tree.nodes.new("ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = strength
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    composite = tree.nodes.new("NodeGroupOutput")
    tree.links.new(render_layers.outputs["Image"], multiply.inputs[1])
    tree.links.new(render_layers.outputs["Ambient Occlusion"], multiply.inputs[2])
    tree.links.new(multiply.outputs["Color"], composite.inputs["Image"])


def setup_world():
    world = bpy.data.worlds.new("WhiteWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.7
    for name, location, energy in (
        ("Key_Area", (0.0, -1.8, 2.8), 75),
        ("Fill_Area", (1.8, 1.2, 2.0), 18),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.size = 6.0
        light_data.use_shadow = True
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        bpy.context.scene.collection.objects.link(light)

    softbox_data = bpy.data.lights.new("Studio_Diffuse_Softbox", "AREA")
    softbox_data.energy = 60
    softbox_data.size = 6.0
    softbox_data.use_shadow = True
    softbox = bpy.data.objects.new("Studio_Diffuse_Softbox", softbox_data)
    softbox.location = (0.0, -0.8, 3.5)
    softbox.hide_render = True
    bpy.context.scene.collection.objects.link(softbox)


def render(camera, output):
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(camera.get("pixel_aspect_y", 1.0))
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main():
    scene_data = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()
    setup_world()
    table = scene_data["table"]
    table_controller, table_objects = import_asset(table["mesh"], "table_0")
    table_pivot = table.get("scale_pivot") or {}
    if "leg_scale_z" in table_pivot:
        scale_asset_below_top(
            table_controller, table_objects,
            table_pivot["leg_anchor_local_z"],
            float(table_pivot["leg_scale_z"]) / float(table["scale_xyz"][2]),
        )
    place_asset(
        table_controller, table_objects,
        table.get("blender_scale_xyz", table["scale_xyz"]),
        (0, 0, 0), table.get("yaw_deg", 0.0), z_anchor="top",
    )
    table_top = 0.0
    all_objects = list(table_objects)
    placed = {}
    for object_id, item in scene_data["objects"].items():
        if item["asset_type"] == "articulated_urdf":
            controller, imported = import_articulated_urdf_asset(
                item["asset"], object_id, item.get("asset_orientation_matrix"),
            )
        else:
            controller, imported = import_asset(item["asset"], object_id)
        if item["asset_type"] == "articulated_urdf":
            apply_articulated_joint_state(imported, item)
        yaw = item["yaw_deg"] + 90 * item.get("axis_mapping_quarter_turns", 0)
        scale = float(item["uniform_scale"])
        translation = item["translation_world_m"]
        place_asset(
            controller,
            imported,
            (scale, scale, scale),
            (translation[0], translation[1], table_top + translation[2]),
            yaw,
            roll_x_deg=item.get("roll_x_deg", 0.0),
            roll_y_deg=item.get("roll_y_deg", 0.0),
        )
        all_objects.extend(imported)
        placed[object_id] = (controller, imported)

    scene_low, scene_high = bounds(all_objects)
    add_shadow_ground(scene_low.z - 0.002)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 576
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    setup_ambient_occlusion_compositor()
    configure_specular_reflections(bpy, scene)

    front = create_camera("front_camera")
    frame = scene_data["cameras"]["front"]["world_frame"]
    front_cameras = json.loads((ROOT / "outputs/vggt_front_only/cameras.json").read_text(encoding="utf-8"))
    front_camera = next(iter(front_cameras["views"].values()))
    raw_intrinsic = scene_data["cameras"]["front"]["raw_intrinsic"]
    raw_size = scene_data["cameras"]["front"]["raw_to_network"]["raw_size"]
    configure_vggt_camera(front, frame, front_camera, raw_size, raw_intrinsic)
    configure_front_orthographic(front, all_objects, scene.render.resolution_x / scene.render.resolution_y)
    initial_states = {}
    for object_id, item in scene_data["objects"].items():
        if item["asset_type"] != "articulated_urdf":
            continue
        controller, imported = placed[object_id]
        translation = item["translation_world_m"]
        state = set_articulated_initial_state(
            front,
            controller,
            imported,
            item["front_instance_geometry"]["bbox_xyxy"],
            scene_data["segmentation"]["front_image_size"],
            (translation[0], translation[1], table_top + translation[2]),
        )
        if state:
            initial_states[object_id] = state
    if os.environ.get("TABLETOP_RENDER_FRONT", "1") != "0":
        render(front, OUTPUT_DIR / "recovered_front.png")

    top = create_camera("top_camera")
    top.data.type = "ORTHO"
    scene_width = scene_high.x - scene_low.x
    scene_depth = scene_high.y - scene_low.y
    top.data.ortho_scale = max(scene_width, scene_depth * (16 / 9)) * 1.15
    top.location = (0, 0, 2.0)
    top.rotation_euler = (0, 0, 0)
    look_at(top, (0, 0, 0))
    render(top, OUTPUT_DIR / "recovered_top.png")
    assets = {
        "table_0": table["mesh"],
        **{object_id: item["asset"] for object_id, item in scene_data["objects"].items()},
    }
    (OUTPUT_DIR / "render_manifest.json").write_text(
        json.dumps(
            {
                "renderer": "Blender 5.1 EEVEE",
                "lighting": {
                    "key_area": {"location_m": [0.0, -1.8, 2.8], "energy_w": 75, "size_m": 6.0},
                    "fill_area": {"location_m": [1.8, 1.2, 2.0], "energy_w": 18, "size_m": 6.0},
                    "diffuse_softbox": {"location_m": [0.0, -0.8, 3.5], "energy_w": 60, "size_m": 6.0, "hidden_during_optimization": True},
                    "world": {"color": [1.0, 1.0, 1.0], "strength": 0.7},
                    "shadow_ground_z_m": scene_low.z - 0.002,
                },
                "front_camera": {"projection": "orthographic", "orientation_source": "VGGT full extrinsic"},
                "assets": {
                    object_id: {
                        "path": path,
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    }
                    for object_id, path in assets.items()
                },
                "articulated_initial_states": initial_states,
                "outputs": ["recovered_front.png", "recovered_top.png", "unified_scene.blend"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "unified_scene.blend"))


if __name__ == "__main__":
    main()
