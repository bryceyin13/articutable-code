"""Coordinate adapters for loading URDF visual assets in Blender."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def urdf_origin_matrix(xyz, rpy):
    from mathutils import Matrix

    roll, pitch, yaw = rpy
    return Matrix.Translation(xyz) @ (
        Matrix.Rotation(yaw, 4, "Z")
        @ Matrix.Rotation(pitch, 4, "Y")
        @ Matrix.Rotation(roll, 4, "X")
    )


def uses_explicit_gltf_asset_basis(package):
    package = Path(package)
    manifest = Path(package) / "manifest.json"
    if manifest.is_file() and int(json.loads(manifest.read_text(
            encoding="utf-8")).get("coordinate_contract_version", 1)) >= 2:
        return True
    urdf = package / "model.urdf"
    if not urdf.is_file():
        return False
    for visual in ET.parse(urdf).getroot().findall(".//visual"):
        mesh = visual.find("geometry/mesh")
        origin = visual.find("origin")
        if mesh is None or origin is None:
            continue
        if Path(mesh.attrib.get("filename", "")).suffix.lower() not in {
                ".glb", ".gltf"}:
            continue
        rpy = tuple(float(value) for value in origin.attrib.get(
            "rpy", "0 0 0").split())
        if len(rpy) == 3 and abs(rpy[0] - math.pi / 2) < 1e-6:
            return True
    return False


def blender_visual_matrix(xyz, rpy, mesh_path, compensate_gltf_basis=False):
    from mathutils import Matrix

    matrix = urdf_origin_matrix(xyz, rpy)
    if (compensate_gltf_basis
            and Path(mesh_path).suffix.lower() in {".glb", ".gltf"}):
        matrix @= Matrix.Rotation(-math.pi / 2, 4, "X")
    return matrix
