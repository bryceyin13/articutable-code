from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj

from .artifacts import KinematicSpec, write_json
from .joints import JointEstimate
from .parts import LinkGeometry


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _origin(parent, xyz, rpy=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _numbers(xyz), "rpy": _numbers(rpy)})


def _bake_zero_pose(spec: KinematicSpec, links: Mapping[int, LinkGeometry],
                    joints: Mapping[int, JointEstimate], zero_positions: Mapping[int, float]
                    ) -> tuple[dict[int, LinkGeometry], dict[int, JointEstimate]]:
    expected = {joint.child for joint in spec.joints}
    if not set(zero_positions) <= expected or not np.isfinite(list(zero_positions.values())).all():
        raise ValueError("semantic zero positions must be finite and reference known joints")
    transforms = {0: np.eye(4)}
    baked_joints = {}
    children = {link.id: [] for link in spec.links}
    for joint in spec.joints:
        children[joint.parent].append(joint)

    def visit(parent: int) -> None:
        parent_transform = transforms[parent]
        for joint_spec in children[parent]:
            child, estimate = joint_spec.child, joints[joint_spec.child]
            axis = parent_transform[:3, :3] @ estimate.axis
            pivot = trimesh.transform_points([estimate.pivot], parent_transform)[0]
            amount = zero_positions.get(child, 0.0)
            motion = (trimesh.transformations.rotation_matrix(amount, axis, pivot)
                      if estimate.type == "revolute" else
                      trimesh.transformations.translation_matrix(axis * amount))
            transforms[child] = motion @ parent_transform
            final_axis = transforms[child][:3, :3] @ estimate.axis
            final_pivot = trimesh.transform_points([estimate.pivot], transforms[child])[0]
            baked_joints[child] = JointEstimate(
                estimate.type, estimate.subtype, final_axis, final_pivot,
                estimate.lower, estimate.upper)
            visit(child)

    visit(0)
    baked_links = {}
    for index, link in links.items():
        visual, collision = link.visual_mesh.copy(), link.collision_mesh.copy()
        visual.apply_transform(transforms[index]); collision.apply_transform(transforms[index])
        baked_links[index] = LinkGeometry(
            link.id, link.primitive_ids, visual, collision, link.visual_path, link.collision_path)
    return baked_links, baked_joints


def export_urdf(spec: KinematicSpec, links: Mapping[int, LinkGeometry],
                joints: Mapping[int, JointEstimate], output_dir: Path, density: float,
                zero_positions: Mapping[int, float] | None = None,
                state_metadata: Mapping[int, Mapping] | None = None) -> Path:
    if density <= 0 or not math.isfinite(density):
        raise ValueError("density must be positive and finite")
    if set(links) != {link.id for link in spec.links}:
        raise ValueError("export refused unvalidated kinematic tree: link geometry mismatch")
    expected_joints = {joint.child for joint in spec.joints}
    if set(joints) != expected_joints:
        raise ValueError("export refused unvalidated kinematic tree: joint estimate mismatch")
    if not set(state_metadata or {}) <= expected_joints:
        raise ValueError("joint state metadata references unknown joints")
    zero_positions = dict(zero_positions or {})
    states = {}
    for child in expected_joints:
        state = (state_metadata or {}).get(child)
        if state is None:
            zero = float(zero_positions.get(child, 0.0))
            if not math.isfinite(zero):
                raise ValueError(f"joint {child} has a non-finite zero position")
            states[child] = {
                "zero_q": zero, "mesh_current_q": -zero, "scene_current_q": -zero,
            }
            continue
        required = {"zero_q", "mesh_current_q", "scene_current_q"}
        if not required <= set(state):
            raise ValueError(f"joint {child} state metadata is missing canonical fields")
        zero, mesh, scene = (
            float(state["zero_q"]), float(state["mesh_current_q"]),
            float(state["scene_current_q"]),
        )
        if (not np.isfinite([zero, mesh, scene]).all()
                or not math.isclose(zero, -mesh, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(f"joint {child} state metadata violates zero_q=-mesh_current_q")
        states[child] = {
            "zero_q": zero, "mesh_current_q": mesh, "scene_current_q": scene,
        }
    zero_positions = zero_positions or {
        child: state["zero_q"] for child, state in states.items()
    }
    for child, state in states.items():
        if not math.isclose(
                float(zero_positions.get(child, 0.0)), state["zero_q"],
                rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"joint {child} zero position disagrees with state metadata")
    links, joints = _bake_zero_pose(spec, links, joints, zero_positions)
    scene_links, _ = _bake_zero_pose(
        spec, links, joints,
        {child: state["scene_current_q"] for child, state in states.items()},
    )
    y_up_to_z_up = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    joints = {
        child: JointEstimate(
            estimate.type,
            estimate.subtype,
            y_up_to_z_up[:3, :3] @ estimate.axis,
            trimesh.transform_points([estimate.pivot], y_up_to_z_up)[0],
            estimate.lower,
            estimate.upper,
        )
        for child, estimate in joints.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    robot = ET.Element("robot", {"name": "gram"})
    link_origins = {0: np.zeros(3)}
    for joint in spec.joints:
        link_origins[joint.child] = joints[joint.child].pivot
    manifest = {
        "coordinate_contract_version": 2,
        "density_kg_m3": density,
        "working_frame": {"up_axis": "Y", "unit": "meter", "handedness": "right"},
        "urdf_frame": {"up_axis": "Z", "unit": "meter", "handedness": "right"},
        "working_to_urdf": y_up_to_z_up.tolist(),
        "scene_rigid_mesh": "scene_rigid.glb",
        "scene_rigid_state": "scene_current_q",
        "links": [],
        "joint_states": [],
    }
    scene_rigid = trimesh.util.concatenate(
        [scene_links[index].visual_mesh for index in sorted(scene_links)])
    material = getattr(scene_rigid.visual, "material", None)
    if material is not None:
        if hasattr(material, "to_pbr"):
            material = material.to_pbr()
        material.name = f"{output_dir.parent.name}_scene_rigid"
        scene_rigid.visual.material = material
    scene_rigid.export(output_dir / manifest["scene_rigid_mesh"])
    for link_spec in spec.links:
        link = links[link_spec.id]
        origin = link_origins[link_spec.id]
        visual_name, collision_name = f"link_{link.id}_visual.glb", f"link_{link.id}_collision.obj"
        visual_mesh = link.visual_mesh.copy()
        material = getattr(visual_mesh.visual, "material", None)
        if material is not None:
            if hasattr(material, "to_pbr"):
                material = material.to_pbr()
            material.name = f"{output_dir.parent.name}_link_{link.id}"
            visual_mesh.visual.material = material
        visual_mesh.export(mesh_dir / visual_name)
        collision_mesh = link.collision_mesh.copy()
        collision_mesh.apply_transform(y_up_to_z_up)
        (mesh_dir / collision_name).write_text(export_obj(collision_mesh, include_texture=False, include_color=False), encoding="utf-8")
        node = ET.SubElement(robot, "link", {"name": f"link_{link.id}"})
        visual = ET.SubElement(node, "visual")
        _origin(visual, -origin, (math.pi / 2, 0, 0))
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", {"filename": f"meshes/{visual_name}"})
        collision = ET.SubElement(node, "collision")
        _origin(collision, -origin)
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", {"filename": f"meshes/{collision_name}"})
        hull = collision_mesh.convex_hull
        mass = max(float(hull.volume) * density, 1e-4)
        inertia = np.asarray(hull.moment_inertia, dtype=float) * (mass / float(hull.mass))
        center = np.asarray(hull.center_mass, dtype=float) - origin
        if not np.isfinite(inertia).all() or not np.isfinite(center).all():
            raise ValueError(f"link {link.id} has non-finite inertial values")
        inertial = ET.SubElement(node, "inertial")
        _origin(inertial, center)
        ET.SubElement(inertial, "mass", {"value": f"{mass:.12g}"})
        ET.SubElement(inertial, "inertia", {
            "ixx": f"{inertia[0, 0]:.12g}", "ixy": f"{inertia[0, 1]:.12g}",
            "ixz": f"{inertia[0, 2]:.12g}", "iyy": f"{inertia[1, 1]:.12g}",
            "iyz": f"{inertia[1, 2]:.12g}", "izz": f"{inertia[2, 2]:.12g}",
        })
        manifest["links"].append({"id": link.id, "mass": mass, "visual": visual_name, "collision": collision_name})
    for joint_spec in spec.joints:
        estimate = joints[joint_spec.child]
        if estimate.lower is None or estimate.upper is None or estimate.lower > estimate.upper:
            raise ValueError(f"joint {joint_spec.child} has invalid limits")
        node = ET.SubElement(robot, "joint", {"name": f"joint_{joint_spec.child}", "type": estimate.type})
        ET.SubElement(node, "parent", {"link": f"link_{joint_spec.parent}"})
        ET.SubElement(node, "child", {"link": f"link_{joint_spec.child}"})
        _origin(node, link_origins[joint_spec.child] - link_origins[joint_spec.parent])
        ET.SubElement(node, "axis", {"xyz": _numbers(estimate.axis)})
        ET.SubElement(node, "limit", {"lower": f"{estimate.lower:.12g}", "upper": f"{estimate.upper:.12g}",
                                          "effort": "100", "velocity": "1"})
        state = states[joint_spec.child]
        joint_state = {
            "name": f"joint_{joint_spec.child}", "child": joint_spec.child,
            "type": estimate.type,
            "zero_q": state["zero_q"],
            "mesh_current_q": state["mesh_current_q"],
            "scene_current_q": state["scene_current_q"],
        }
        if estimate.subtype is not None:
            joint_state["subtype"] = estimate.subtype
        manifest["joint_states"].append(joint_state)
    ET.indent(robot, space="  ")
    path = output_dir / "model.urdf"
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
    ET.parse(path)
    for mesh in robot.findall(".//mesh"):
        if not (output_dir / mesh.attrib["filename"]).exists():
            raise ValueError(f"missing exported mesh: {mesh.attrib['filename']}")
    write_json(output_dir / "manifest.json", manifest)
    return path
