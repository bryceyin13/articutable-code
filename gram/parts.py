from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .artifacts import KinematicSpec, LinkSpec, write_json


@dataclass
class LinkGeometry:
    id: int
    primitive_ids: tuple[int, ...]
    visual_mesh: trimesh.Trimesh
    collision_mesh: trimesh.Trimesh
    visual_path: Path
    collision_path: Path

    @property
    def face_count(self) -> int:
        return len(self.visual_mesh.faces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "primitive_ids": list(self.primitive_ids), "face_count": self.face_count,
            "area": float(self.visual_mesh.area), "bounds": self.visual_mesh.bounds.tolist(),
            "watertight": bool(self.visual_mesh.is_watertight),
            "visual_path": str(self.visual_path), "collision_path": str(self.collision_path),
        }


def merge_tiny_leaf_links(
        face_primitive_ids: np.ndarray,
        spec: KinematicSpec,
        min_link_faces: int,
) -> tuple[KinematicSpec, list[dict[str, Any]]]:
    labels = np.asarray(face_primitive_ids)
    links = {link.id: link for link in spec.links}
    joints = {joint.child: joint for joint in spec.joints}
    counts = {
        link.id: int(np.isin(labels, link.primitive_ids).sum())
        for link in spec.links
    }
    merges = []
    while tiny := sorted(
        link_id for link_id, count in counts.items()
        if link_id != 0 and count < min_link_faces
    ):
        parents = {joint.parent for joint in joints.values()}
        leaves = [link_id for link_id in tiny if link_id and link_id not in parents]
        if not leaves:
            link_id = tiny[0]
            kind = "fixed base" if link_id == 0 else "non-leaf"
            raise ValueError(
                f"{kind} link {link_id} has {counts[link_id]} faces, "
                f"fewer than {min_link_faces}"
            )
        for child in leaves:
            joint = joints.pop(child)
            leaf, parent = links.pop(child), links[joint.parent]
            links[parent.id] = LinkSpec(
                parent.id,
                parent.name,
                tuple(sorted((*parent.primitive_ids, *leaf.primitive_ids))),
            )
            merges.append({
                "link": child,
                "parent": parent.id,
                "face_count": counts.pop(child),
                "primitive_ids": list(leaf.primitive_ids),
            })
            counts[parent.id] += merges[-1]["face_count"]

    if not merges:
        return spec, merges

    old_ids = sorted(links)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    remaining = set(old_ids)
    payload = {
        "links": [{
            "id": remap[old_id],
            "name": links[old_id].name,
            "primitive_ids": list(links[old_id].primitive_ids),
        } for old_id in old_ids],
        "joints": [{
            "parent": remap[joint.parent],
            "child": remap[joint.child],
            "type": joint.type,
            **({"subtype": joint.subtype} if joint.subtype else {}),
        } for joint in spec.joints if joint.child in remaining],
        "drawer_groups": [{
            "parent": remap[group.parent],
            "children": [remap[child] for child in group.children if child in remaining],
        } for group in spec.drawer_groups
        if group.parent in remaining
        and any(child in remaining for child in group.children)],
        "persistent_spin_children": [
            remap[child] for child in spec.persistent_spin_children
            if child in remaining
        ],
    }
    primitive_ids = {
        primitive for link in spec.links for primitive in link.primitive_ids
    }
    return KinematicSpec.from_dict(payload, primitive_ids), merges


def merge_link_into_parent(
        spec: KinematicSpec, child: int) -> tuple[KinematicSpec, dict[str, Any]]:
    joint = next((item for item in spec.joints if item.child == child), None)
    if joint is None:
        raise ValueError(f"link {child} has no parent joint")
    links = {link.id: link for link in spec.links}
    parent = links[joint.parent]
    removed = links.pop(child)
    links[parent.id] = LinkSpec(
        parent.id, parent.name,
        tuple(sorted((*parent.primitive_ids, *removed.primitive_ids))),
    )
    remaining_joints = []
    for item in spec.joints:
        if item.child == child:
            continue
        remaining_joints.append({
            "parent": joint.parent if item.parent == child else item.parent,
            "child": item.child,
            "type": item.type,
            **({"subtype": item.subtype} if item.subtype else {}),
        })
    grouped_drawers: dict[int, set[int]] = {}
    for group in spec.drawer_groups:
        group_parent = joint.parent if group.parent == child else group.parent
        children = {item for item in group.children if item != child}
        if children:
            grouped_drawers.setdefault(group_parent, set()).update(children)
    old_ids = sorted(links)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    payload = {
        "links": [{
            "id": remap[old_id], "name": links[old_id].name,
            "primitive_ids": list(links[old_id].primitive_ids),
        } for old_id in old_ids],
        "joints": [{
            **item,
            "parent": remap[item["parent"]],
            "child": remap[item["child"]],
        } for item in remaining_joints],
        "drawer_groups": [{
            "parent": remap[parent_id],
            "children": sorted(remap[item] for item in children),
        } for parent_id, children in sorted(grouped_drawers.items())],
        "persistent_spin_children": [
            remap[item] for item in spec.persistent_spin_children if item != child
        ],
    }
    primitive_ids = {
        primitive for link in spec.links for primitive in link.primitive_ids
    }
    return KinematicSpec.from_dict(payload, primitive_ids), {
        "parent": joint.parent,
        "child": child,
        "type": joint.type,
        **({"subtype": joint.subtype} if joint.subtype else {}),
        "child_name": removed.name,
        "merged_primitive_ids": list(removed.primitive_ids),
    }


def save_link_meshes(links: dict[int, LinkGeometry], output_dir: Path) -> None:
    visual_dir, collision_dir = output_dir / "visual", output_dir / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for index in sorted(links):
        link = links[index]
        visual_path = visual_dir / f"link_{index}.glb"
        collision_path = collision_dir / f"link_{index}.obj"
        link = LinkGeometry(
            link.id, link.primitive_ids, link.visual_mesh, link.collision_mesh,
            visual_path, collision_path,
        )
        links[index] = link
        link.visual_mesh.export(visual_path)
        link.collision_mesh.export(collision_path)
        item = link.to_dict()
        item["visual_path"] = str(visual_path.relative_to(output_dir))
        item["collision_path"] = str(collision_path.relative_to(output_dir))
        payload.append(item)
    write_json(output_dir / "parts.json", {"links": payload})


def build_link_meshes(mesh: trimesh.Trimesh, face_primitive_ids: np.ndarray, spec: KinematicSpec,
                      output_dir: Path, min_link_faces: int = 1) -> dict[int, LinkGeometry]:
    labels = np.asarray(face_primitive_ids)
    if labels.shape != (len(mesh.faces),):
        raise ValueError(f"face labels {labels.shape} do not match {len(mesh.faces)} mesh faces")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh contains NaN or infinite vertices")
    visual_dir, collision_dir = output_dir / "visual", output_dir / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)
    links: dict[int, LinkGeometry] = {}
    selected_total = 0
    for link in spec.links:
        indices = np.flatnonzero(np.isin(labels, link.primitive_ids))
        if link.id != 0 and len(indices) < min_link_faces:
            raise ValueError(f"link {link.id} has {len(indices)} faces, fewer than {min_link_faces}")
        selected_total += len(indices)
        part = mesh.submesh([indices], append=True, repair=False)
        if not isinstance(part, trimesh.Trimesh) or not np.isfinite(part.vertices).all() or part.area <= 0:
            raise ValueError(f"link {link.id} has invalid or zero-area geometry")
        centered = np.asarray(part.vertices) - np.asarray(part.vertices).mean(axis=0)
        if np.linalg.matrix_rank(centered) < 3:
            raise ValueError(f"link {link.id} collision source has zero 3D extent")
        visual_path, collision_path = visual_dir / f"link_{link.id}.glb", collision_dir / f"link_{link.id}.obj"
        links[link.id] = LinkGeometry(
            link.id, link.primitive_ids, part, part.copy(),
            visual_path, collision_path,
        )
    if selected_total != len(mesh.faces):
        raise ValueError(f"link partition selected {selected_total} of {len(mesh.faces)} faces")
    for joint in spec.joints:
        if joint.parent not in links or joint.child not in links:
            raise ValueError(f"joint {joint.child} references a missing materialized link")
    save_link_meshes(links, output_dir)
    return links
