#!/usr/bin/env python3
"""Render final GLBs with one fixed Stage 10 preview style."""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.render_settings import configure_specular_reflections


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item", action="append", nargs=2, required=True,
        metavar=("MODEL_GLB", "PREVIEW_PNG"),
    )
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def bounds(objects):
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("GLB contains no mesh geometry")
    lower = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    upper = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return lower, upper


def point_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render(model, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(model))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    lower, upper = bounds(imported)
    center = (lower + upper) * 0.5
    diagonal = (upper - lower).length
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError(f"GLB has invalid bounds: {model}")

    scene = bpy.context.scene
    engines = {
        item.identifier
        for item in scene.render.bl_rna.properties["engine"].enum_items
    }
    scene.render.engine = next(
        name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH")
        if name in engines
    )
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(output)
    scene.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    background.inputs["Strength"].default_value = 0.5
    scene.world = world

    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera_data.type = "PERSP"
    camera_data.angle = math.radians(42)
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    direction = Vector((1.35, -2.0, 1.15)).normalized()
    radius = diagonal * 0.5
    distance = radius / math.sin(camera_data.angle * 0.5) * 1.18
    camera.location = center + direction * distance
    point_at(camera, center)
    scene.camera = camera

    sunlight = bpy.data.lights.new("NaturalLight", "SUN")
    sunlight.energy = 1.0
    sunlight.angle = math.radians(20)
    sun = bpy.data.objects.new("NaturalLight", sunlight)
    scene.collection.objects.link(sun)
    sun.rotation_euler = tuple(map(math.radians, (30, 0, -35)))

    configure_specular_reflections(bpy, scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


def main():
    for model, output in arguments().item:
        render(Path(model).resolve(), Path(output).resolve())


if __name__ == "__main__":
    main()
