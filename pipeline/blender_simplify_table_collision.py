#!/usr/bin/env python3
"""Build a world-space, low-poly collision proxy for table_0."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="table_0")
    parser.add_argument("--target-faces", type=int, default=0)
    parser.add_argument("--max-error-m", type=float, default=0.002)
    return parser.parse_args(argv)


def belongs_to(obj, object_id):
    while obj is not None:
        if obj.name == object_id or obj.name.startswith(f"{object_id}."):
            return True
        obj = obj.parent
    return False


def world_triangles(objects):
    vertices = []
    faces = []
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend(tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices)
        faces.extend(
            tuple(offset + index for index in triangle.vertices)
            for triangle in mesh.loop_triangles
        )
    return vertices, faces


def mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def simplify(vertices, faces, target_faces):
    obj = mesh_object("table_collision_proxy", vertices, faces)
    welded = bmesh.new()
    welded.from_mesh(obj.data)
    bmesh.ops.remove_doubles(welded, verts=welded.verts, dist=1.0e-6)
    welded.to_mesh(obj.data)
    welded.free()
    obj.data.update()
    dissolve = obj.modifiers.new("Planar collision reduction", "DECIMATE")
    dissolve.decimate_type = "DISSOLVE"
    dissolve.angle_limit = math.radians(0.1)
    dissolve.use_dissolve_boundaries = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=dissolve.name)
    obj.data.calc_loop_triangles()
    current_faces = len(obj.data.loop_triangles)

    modifier = obj.modifiers.new("QEM collision reduction", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = min(1.0, target_faces / current_faces)
    modifier.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.calc_loop_triangles()
    reduced_vertices = [tuple(vertex.co) for vertex in obj.data.vertices]
    reduced_faces = [tuple(face.vertices) for face in obj.data.loop_triangles]
    bpy.data.objects.remove(obj, do_unlink=True)
    return reduced_vertices, reduced_faces


def samples(vertices, faces):
    for vertex in vertices:
        yield vertex
    for a, b, c in faces:
        yield tuple(
            (vertices[a][axis] + vertices[b][axis] + vertices[c][axis]) / 3.0
            for axis in range(3)
        )


def max_surface_error(original, reduced):
    original_vertices, original_faces = original
    reduced_vertices, reduced_faces = reduced
    original_tree = BVHTree.FromPolygons(
        original_vertices, original_faces, all_triangles=True,
    )
    reduced_tree = BVHTree.FromPolygons(
        reduced_vertices, reduced_faces, all_triangles=True,
    )
    if original_tree is None or reduced_tree is None:
        raise RuntimeError("could not build BVH for table collision proxy")

    maximum = 0.0
    for point in samples(original_vertices, original_faces):
        nearest = reduced_tree.find_nearest(point)
        if nearest is None:
            raise RuntimeError("table collision proxy has no nearest surface")
        maximum = max(maximum, float(nearest[3]))
    for point in samples(reduced_vertices, reduced_faces):
        nearest = original_tree.find_nearest(point)
        if nearest is None:
            raise RuntimeError("original table has no nearest surface")
        maximum = max(maximum, float(nearest[3]))
    return maximum


def topology(vertices, faces, tolerance=1.0e-6):
    ids = {}
    welded = []
    for point in vertices:
        key = tuple(round(float(value) / tolerance) for value in point)
        welded.append(ids.setdefault(key, len(ids)))

    parent = list(range(len(ids)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    edges = Counter()
    used = set()
    for face in faces:
        face = [welded[index] for index in face]
        used.update(face)
        for first, second in zip(face, face[1:] + face[:1]):
            union(first, second)
            edges[tuple(sorted((first, second)))] += 1
    return {
        "components": len({find(index) for index in used}),
        "boundary_edges": sum(count == 1 for count in edges.values()),
        "nonmanifold_edges": sum(count > 2 for count in edges.values()),
    }


def main():
    args = arguments()
    if (args.target_faces != 0 and args.target_faces < 4) or args.max_error_m <= 0:
        raise ValueError(
            "target faces must be 0 or >= 4 and max error must be positive"
        )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(args.input.resolve()))
    table_meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and belongs_to(obj, args.object_id)
    ]
    if not table_meshes:
        raise RuntimeError(f"no mesh descendants found for {args.object_id}")

    original = world_triangles(table_meshes)
    original_faces = len(original[1])
    source_topology = topology(*original)
    target = (
        original_faces if args.target_faces == 0
        else min(args.target_faces, original_faces)
    )
    reduced = original
    error = 0.0
    rejected = None
    while target < original_faces:
        candidate = simplify(*original, target)
        candidate_error = max_surface_error(original, candidate)
        if candidate_error <= args.max_error_m:
            reduced, error = candidate, candidate_error
            if rejected is not None:
                rejected_target, rejected_error = rejected
                fraction = (
                    (rejected_error - args.max_error_m)
                    / (rejected_error - candidate_error)
                )
                refined_target = math.ceil(
                    rejected_target + fraction * (target - rejected_target)
                )
                if rejected_target + 250 < refined_target < target - 250:
                    refined = simplify(*original, refined_target)
                    refined_error = max_surface_error(original, refined)
                    if refined_error <= args.max_error_m:
                        reduced, error = refined, refined_error
            break
        rejected = target, candidate_error
        target = min(original_faces, target * 2)

    collision_topology = topology(*reduced)
    topology_fallback = not (
        collision_topology["components"] == source_topology["components"]
        and collision_topology["boundary_edges"]
        <= max(32, 2 * source_topology["boundary_edges"])
        and collision_topology["nonmanifold_edges"]
        <= source_topology["nonmanifold_edges"]
    )
    if topology_fallback:
        reduced = original
        error = 0.0
        collision_topology = source_topology

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(reduced[0], dtype=np.float32)
    faces = np.asarray(reduced[1], dtype=np.int32)
    np.savez_compressed(
        args.output,
        points=vertices,
        triangles=faces,
        source_faces=np.int64(original_faces),
        collision_faces=np.int64(len(faces)),
        max_sampled_error_m=np.float64(error),
        source_components=np.int64(source_topology["components"]),
        collision_components=np.int64(collision_topology["components"]),
        source_boundary_edges=np.int64(source_topology["boundary_edges"]),
        collision_boundary_edges=np.int64(
            collision_topology["boundary_edges"]
        ),
        topology_fallback=np.bool_(topology_fallback),
    )
    print(json.dumps({
        "object_id": args.object_id,
        "source_faces": original_faces,
        "collision_faces": len(faces),
        "max_sampled_error_m": error,
        "source_topology": source_topology,
        "collision_topology": collision_topology,
        "topology_fallback": topology_fallback,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
