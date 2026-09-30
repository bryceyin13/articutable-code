from __future__ import annotations

import argparse
import contextlib
import io
import json
import socket
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gram.render import primitive_color, remap_face_labels
from pipeline.render_settings import configure_specular_reflections


def parse_args():
    args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--face-ids", type=Path)
    parser.add_argument("--face-triangles", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-mesh", type=Path)
    parser.add_argument("--server-socket", type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--no-save-blend", action="store_true")
    parsed = parser.parse_args(args)
    job_args = (parsed.mesh, parsed.face_ids, parsed.face_triangles, parsed.output_dir)
    if parsed.server_socket is None and not all(job_args):
        parser.error("--mesh, --face-ids, --face-triangles and --output-dir are required")
    if parsed.server_socket is not None and any(job_args):
        parser.error("--server-socket cannot be combined with a render job")
    return parsed


def emission_material(name, rgb):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*[x / 255 for x in rgb], 1)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def split_primitives(source, labels):
    mesh = source.data
    if labels.shape != (len(mesh.polygons),):
        raise RuntimeError(f"face labels {labels.shape} do not match {len(mesh.polygons)} polygons")
    objects, centroids = {}, {}
    for primitive in sorted(map(int, np.unique(labels))):
        selected = [mesh.polygons[i] for i in np.flatnonzero(labels == primitive)]
        vertices, faces = [], []
        for polygon in selected:
            face = []
            for vertex_index in polygon.vertices:
                face.append(len(vertices))
                vertices.append(source.matrix_world @ mesh.vertices[vertex_index].co)
            faces.append(face)
        part_mesh = bpy.data.meshes.new(f"primitive_{primitive}")
        part_mesh.from_pydata(vertices, [], faces)
        part = bpy.data.objects.new(f"primitive_{primitive}", part_mesh)
        bpy.context.collection.objects.link(part)
        part.data.materials.append(emission_material(f"primitive_{primitive}", primitive_color(primitive)))
        objects[primitive] = part
        centroids[primitive] = sum((Vector(v) for v in vertices), Vector()) / len(vertices)
    bpy.data.objects.remove(source, do_unlink=True)
    return objects, centroids


def face_triangles(source):
    mesh = source.data
    if any(len(polygon.vertices) != 3 for polygon in mesh.polygons):
        raise RuntimeError("working GLB must contain only triangular faces")
    return np.asarray([
        [tuple(source.matrix_world @ mesh.vertices[index].co) for index in polygon.vertices]
        for polygon in mesh.polygons
    ])


def point_camera(camera, point=Vector((0, 0, 0))):
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def import_normalized_source(path):
    before = set(bpy.context.scene.objects)
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"unsupported source mesh format: {path.suffix}")
    objects = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not objects:
        raise RuntimeError(f"source mesh contains no mesh objects: {path}")
    points = [o.matrix_world @ vertex.co for o in objects for vertex in o.data.vertices]
    lower = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    upper = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center, diagonal = (lower + upper) / 2, (upper - lower).length
    if diagonal <= 0:
        raise RuntimeError("source mesh has invalid bounds")
    for obj in objects:
        obj.data = obj.data.copy()
        matrix = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            vertex.co = (matrix @ vertex.co - center) / diagonal
        obj.matrix_world.identity()
        obj.data.update()
    return objects


def add_tags(centroids, camera, scene):
    tags, projected = [], {}
    toward_camera = (camera.location - Vector((0, 0, 0))).normalized()
    for primitive, centroid in centroids.items():
        tag = bpy.data.curves.new(f"tag_{primitive}", "FONT")
        tag.body = f"ID {primitive}"
        tag.align_x = "CENTER"
        tag.align_y = "CENTER"
        tag.size = 0.045
        tag.extrude = 0.001
        outline = bpy.data.objects.new(f"tag_outline_{primitive}", tag)
        bpy.context.collection.objects.link(outline)
        outline.location = centroid + toward_camera * 0.014
        outline.rotation_euler = (camera.location - outline.location).to_track_quat("Z", "Y").to_euler()
        outline.scale = (1.12, 1.12, 1.12)
        outline.data.materials.append(emission_material(f"tag_outline_mat_{primitive}", (0, 0, 0)))
        foreground = outline.copy()
        foreground.data = outline.data.copy()
        foreground.name = f"tag_{primitive}"
        foreground.location = centroid + toward_camera * 0.015
        foreground.scale = (1, 1, 1)
        foreground.data.materials.clear()
        foreground.data.materials.append(emission_material(f"tag_mat_{primitive}", (255, 255, 255)))
        bpy.context.collection.objects.link(foreground)
        tags.extend((outline, foreground))
        p = world_to_camera_view(scene, camera, foreground.location)
        projected[str(primitive)] = [round(float(p.x) * scene.render.resolution_x, 3),
                                     round((1 - float(p.y)) * scene.render.resolution_y, 3)]
    return tags, projected


def render_job(args) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    sources = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(sources) != 1:
        raise RuntimeError(f"working GLB must import as one mesh object, got {len(sources)}")
    labels = np.load(args.face_ids)
    if labels.shape != (len(sources[0].data.polygons),):
        reference = np.load(args.face_triangles)
        reference = reference[:, :, (0, 2, 1)]
        reference[:, :, 1] *= -1
        labels = remap_face_labels(reference, labels, face_triangles(sources[0]))
        print(
            f"[gram-render] Blender dropped {len(reference) - len(labels)} faces; "
            "remapped remaining segmentation labels",
            flush=True,
        )
    primitives, centroids = split_primitives(sources[0], labels)
    bpy.ops.export_scene.gltf(filepath=str(args.output_dir / "primitives.glb"), export_format="GLB")
    if not args.no_save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir / "primitives.blend"))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles = bpy.context.preferences.addons["cycles"].preferences
    cycles.compute_device_type = "CUDA"
    cycles.get_devices()
    cuda_devices = [device for device in cycles.devices if device.type == "CUDA"]
    if not cuda_devices:
        raise RuntimeError("Cycles CUDA device is unavailable")
    for device in cycles.devices:
        device.use = device.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.render.film_transparent = False
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
    background.inputs["Strength"].default_value = 2.5
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.35
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    positions = {"front": (0, -2, 0), "right": (2, 0, 0), "back": (0, 2, 0),
                 "left": (-2, 0, 0), "top": (0, 0, 2)}
    source_objects = import_normalized_source(args.source_mesh) if args.source_mesh else []
    configure_specular_reflections(bpy, scene)
    for obj in source_objects:
        obj.hide_render = True
    views = {}
    for name, position in positions.items():
        camera.location = position
        point_camera(camera)
        bpy.context.view_layer.update()
        if source_objects:
            for obj in primitives.values():
                obj.hide_render = True
            for obj in source_objects:
                obj.hide_render = False
            scene.render.filepath = str(args.output_dir / f"view_{name}_rgb.png")
            bpy.ops.render.render(write_still=True)
            for obj in source_objects:
                obj.hide_render = True
            for obj in primitives.values():
                obj.hide_render = False
        tags, projected = add_tags(centroids, camera, scene)
        scene.render.filepath = str(args.output_dir / f"view_{name}.png")
        bpy.ops.render.render(write_still=True)
        views[name] = {
            "camera_matrix": [[float(x) for x in row] for row in camera.matrix_world],
            "projection": {
                "type": "orthographic",
                "ortho_scale": float(camera.data.ortho_scale),
                "image_size": [scene.render.resolution_x, scene.render.resolution_y],
            },
            "tag_positions": projected, "visible_primitive_ids": sorted(centroids),
        }
        for tag in tags:
            bpy.data.objects.remove(tag, do_unlink=True)
    payload = {
        "palette": {str(i): primitive_color(i) for i in sorted(centroids)},
        "source_mesh": str(args.source_mesh) if args.source_mesh else None,
        "views": views,
    }
    (args.output_dir / "views.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serve(socket_path: Path, resolution: int) -> None:
    if socket_path.exists():
        raise FileExistsError(f"Blender server socket already exists: {socket_path}")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        listener.listen()
        print(f"[gram-blender-server] ready: {socket_path}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                with connection.makefile("r", encoding="utf-8") as request_file:
                    request = json.loads(request_file.readline())
                if request.get("shutdown") is True:
                    connection.sendall(b'{"ok": true}\n')
                    return
                stdout, stderr = io.StringIO(), io.StringIO()
                try:
                    args = SimpleNamespace(
                        mesh=Path(request["mesh"]),
                        face_ids=Path(request["face_ids"]),
                        face_triangles=Path(request["face_triangles"]),
                        output_dir=Path(request["output_dir"]),
                        source_mesh=(
                            Path(request["source_mesh"])
                            if request.get("source_mesh") else None
                        ),
                        resolution=int(request.get("resolution", resolution)),
                        no_save_blend=not bool(request.get("save_blend", True)),
                    )
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        render_job(args)
                    response = {
                        "ok": True, "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(),
                    }
                except BaseException:
                    response = {
                        "ok": False, "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(), "error": traceback.format_exc(),
                    }
                connection.sendall((json.dumps(response) + "\n").encode("utf-8"))
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.server_socket is not None:
        serve(args.server_socket, args.resolution)
    else:
        render_job(args)


if __name__ == "__main__":
    main()
