from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


@dataclass(frozen=True)
class RunPaths:
    root: Path
    primitives: Path
    views: Path
    kinematics: Path
    parts: Path
    joints: Path
    physics: Path
    urdf: Path

    @classmethod
    def create(cls, root: Path) -> "RunPaths":
        paths = cls(root, *(root / name for name in (
            "01_primitives", "02_views", "03_kinematics", "04_parts",
            "05_joints", "06_physics", "07_urdf",
        )))
        root.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class MeshFrame:
    center: tuple[float, float, float]
    scale: float

    def __post_init__(self) -> None:
        if len(self.center) != 3 or not np.isfinite(self.center).all():
            raise ValueError("center must be three finite values")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive and finite")

    def to_working(self, points):
        return (np.asarray(points, dtype=float) - self.center) * self.scale

    def to_source(self, points):
        return np.asarray(points, dtype=float) / self.scale + self.center


@dataclass(frozen=True)
class PrimitiveManifest:
    mesh_path: str
    face_ids_path: str
    primitive_ids: tuple[int, ...]
    seed: int
    old_to_new: tuple[tuple[int, int], ...] = ()
    frame: MeshFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveManifest":
        frame = payload.get("frame")
        return cls(
            mesh_path=str(payload["mesh_path"]),
            face_ids_path=str(payload["face_ids_path"]),
            primitive_ids=tuple(map(int, payload["primitive_ids"])),
            seed=int(payload["seed"]),
            old_to_new=tuple((int(a), int(b)) for a, b in payload.get("old_to_new", ())),
            frame=MeshFrame(tuple(frame["center"]), float(frame["scale"])) if frame else None,
        )


@dataclass(frozen=True)
class LinkSpec:
    id: int
    name: str
    primitive_ids: tuple[int, ...]


@dataclass(frozen=True)
class JointSpec:
    parent: int
    child: int
    type: Literal["revolute", "prismatic"]
    subtype: Literal["hinge", "spin"] | None = None


@dataclass(frozen=True)
class DrawerGroup:
    parent: int
    children: tuple[int, ...]


@dataclass(frozen=True)
class KinematicSpec:
    links: tuple[LinkSpec, ...]
    joints: tuple[JointSpec, ...]
    drawer_groups: tuple[DrawerGroup, ...] = ()
    persistent_spin_children: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.drawer_groups:
            payload.pop("drawer_groups")
        if not self.persistent_spin_children:
            payload.pop("persistent_spin_children")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], primitive_ids: set[int]) -> "KinematicSpec":
        try:
            links = tuple(LinkSpec(int(x["id"]), str(x["name"]).strip(), tuple(map(int, x["primitive_ids"]))) for x in payload["links"])
            joints = tuple(JointSpec(
                int(x["parent"]), int(x["child"]), x["type"], x.get("subtype"),
            ) for x in payload["joints"])
            drawer_groups = tuple(DrawerGroup(
                int(x["parent"]), tuple(map(int, x["children"])),
            ) for x in payload.get("drawer_groups", ()))
            persistent_spin_children = tuple(map(
                int, payload.get("persistent_spin_children", ())))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid kinematic schema: {exc}") from exc
        validate_dense_ids([x.id for x in links], "link")
        validate_primitive_partition(links, primitive_ids)
        validate_tree(links, joints)
        validate_drawer_groups(drawer_groups, joints)
        validate_persistent_spin_children(persistent_spin_children, joints)
        return cls(links, joints, drawer_groups, persistent_spin_children)


def validate_dense_ids(ids: list[int], label: str) -> None:
    if ids != list(range(len(ids))):
        raise ValueError(f"{label} IDs must be dense, ordered, and start at 0")


def validate_primitive_partition(links: tuple[LinkSpec, ...], expected: set[int]) -> None:
    seen: set[int] = set()
    for link in links:
        if not link.name:
            raise ValueError(f"link {link.id} has an empty name")
        for primitive in link.primitive_ids:
            if primitive in seen:
                raise ValueError(f"primitive {primitive} assigned more than once")
            seen.add(primitive)
    if seen != expected:
        missing, invented = sorted(expected - seen), sorted(seen - expected)
        raise ValueError(f"primitive partition mismatch: missing={missing}, invented={invented}")


def validate_tree(links: tuple[LinkSpec, ...], joints: tuple[JointSpec, ...]) -> None:
    ids = {link.id for link in links}
    if not links or 0 not in ids:
        raise ValueError("link 0 must be the fixed base")
    if len(joints) != len(links) - 1:
        raise ValueError("kinematics must contain one rooted tree")
    children: dict[int, list[int]] = {i: [] for i in ids}
    parents: set[int] = set()
    for joint in joints:
        if joint.type not in ("revolute", "prismatic"):
            raise ValueError(f"unsupported joint type: {joint.type}")
        if joint.type == "revolute" and joint.subtype not in ("hinge", "spin"):
            raise ValueError("revolute joint subtype must be hinge or spin")
        if joint.type == "prismatic" and joint.subtype is not None:
            raise ValueError("prismatic joint must not have a subtype")
        if joint.parent not in ids or joint.child not in ids or joint.child == 0:
            raise ValueError("joint references an invalid parent or child")
        if joint.child in parents:
            raise ValueError(f"link {joint.child} has multiple parents")
        parents.add(joint.child)
        children[joint.parent].append(joint.child)
    visited: set[int] = set()

    def visit(node: int, active: set[int]) -> None:
        if node in active:
            raise ValueError("kinematic tree contains a loop")
        if node in visited:
            return
        active.add(node)
        for child in children[node]:
            visit(child, active)
        active.remove(node)
        visited.add(node)

    visit(0, set())
    if visited != ids:
        raise ValueError("kinematic tree is disconnected from base link 0")


def validate_drawer_groups(
        groups: tuple[DrawerGroup, ...], joints: tuple[JointSpec, ...]) -> None:
    prismatic = {(joint.parent, joint.child) for joint in joints
                 if joint.type == "prismatic"}
    seen: set[int] = set()
    parents: set[int] = set()
    for group in groups:
        if group.parent in parents:
            raise ValueError(f"drawer parent {group.parent} belongs to multiple groups")
        parents.add(group.parent)
        if not group.children or len(set(group.children)) != len(group.children):
            raise ValueError("drawer group children must be non-empty and unique")
        for child in group.children:
            if (group.parent, child) not in prismatic:
                raise ValueError(
                    f"drawer child {child} must be a direct prismatic child of {group.parent}")
            if child in seen:
                raise ValueError(f"drawer child {child} belongs to multiple groups")
            seen.add(child)


def validate_persistent_spin_children(
        children: tuple[int, ...], joints: tuple[JointSpec, ...]) -> None:
    if len(set(children)) != len(children):
        raise ValueError("persistent spin children must be unique")
    spin_children = {
        joint.child for joint in joints
        if joint.type == "revolute" and joint.subtype == "spin"
    }
    invalid = sorted(set(children) - spin_children)
    if invalid:
        raise ValueError(
            f"persistent spin children must reference direct spin children: {invalid}")
