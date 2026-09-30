import re
import os
import math
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath


def validate_rotation_matrix(value):
    try:
        matrix = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        raise ValueError("asset orientation must be a finite 3x3 rotation matrix") from None
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or not all(
        math.isfinite(item) for row in matrix for item in row
    ):
        raise ValueError("asset orientation must be a finite 3x3 rotation matrix")
    products = [
        sum(matrix[row][axis] * matrix[column][axis] for axis in range(3))
        for row in range(3) for column in range(3)
    ]
    expected = [
        1.0 if row == column else 0.0
        for row in range(3) for column in range(3)
    ]
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if (
        any(
            abs(actual - target) > 1e-6
            for actual, target in zip(products, expected)
        )
        or abs(determinant - 1.0) > 1e-6
    ):
        raise ValueError("asset orientation must be a finite 3x3 rotation matrix")
    return matrix


def read_glb_pbr_material(path):
    """Read the single glTF metallic-roughness material used by a visual GLB."""
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"invalid GLB: {path}")
    _magic, version, total_length = struct.unpack_from("<4sII", data)
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if version != 2 or total_length != len(data) or chunk_type != b"JSON":
        raise ValueError(f"unsupported GLB: {path}")
    document = json.loads(data[20:20 + json_length].rstrip(b" \0"))
    materials = document.get("materials", [])
    if len(materials) != 1:
        raise ValueError(
            f"visual GLB must contain exactly one material: {path}"
        )
    material = materials[0]
    pbr = material.get("pbrMetallicRoughness", {})
    textures = document.get("textures", [])

    def texture_info(name):
        reference = pbr.get(name)
        if reference is None:
            return None
        texture = textures[int(reference["index"])]
        source = texture.get("source")
        if source is None:
            source = texture.get("extensions", {}).get(
                "EXT_texture_webp", {}
            ).get("source")
        if source is None:
            raise ValueError(f"GLB texture has no image source: {path}")
        sampler_index = texture.get("sampler")
        sampler = (
            document.get("samplers", [])[int(sampler_index)]
            if sampler_index is not None else {}
        )
        return {
            "image_index": int(source),
            "tex_coord_index": int(reference.get("texCoord", 0)),
            "wrap_s": int(sampler.get("wrapS", 10497)),
            "wrap_t": int(sampler.get("wrapT", 10497)),
        }

    base = [float(value) for value in pbr.get(
        "baseColorFactor", [1.0, 1.0, 1.0, 1.0]
    )]
    if len(base) != 4 or not all(math.isfinite(value) for value in base):
        raise ValueError(f"invalid GLB base color factor: {path}")
    return {
        "base_color_factor": base,
        "metallic_factor": float(pbr.get("metallicFactor", 1.0)),
        "roughness_factor": float(pbr.get("roughnessFactor", 1.0)),
        "base_color_texture": texture_info("baseColorTexture"),
        "metallic_roughness_texture": texture_info(
            "metallicRoughnessTexture"
        ),
        "alpha_mode": str(material.get("alphaMode", "OPAQUE")),
        "alpha_cutoff": float(material.get("alphaCutoff", 0.5)),
        "double_sided": bool(material.get("doubleSided", False)),
    }


def orient_urdf_root(package_root, rotation_matrix, prefix):
    """Apply a baked asset rotation inside the URDF's outer scene scale."""
    if rotation_matrix is None:
        return False
    matrix = validate_rotation_matrix(rotation_matrix)
    if all(
        abs(matrix[row][column] - (row == column)) <= 1e-12
        for row in range(3) for column in range(3)
    ):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError(f"invalid URDF namespace prefix: {prefix!r}")

    urdf_path = Path(package_root) / "model.urdf"
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    links = [node for node in robot if node.tag.rsplit("}", 1)[-1] == "link"]
    joints = [node for node in robot if node.tag.rsplit("}", 1)[-1] == "joint"]
    children = {
        endpoint.attrib["link"]
        for joint in joints
        for endpoint in joint
        if endpoint.tag.rsplit("}", 1)[-1] == "child"
    }
    roots = [link.attrib["name"] for link in links if link.attrib["name"] not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link: {urdf_path}")

    names = {node.attrib["name"] for node in (*links, *joints)}
    wrapper = f"{prefix}_asset_frame"
    while wrapper in names:
        wrapper += "_"
    joint_name = f"{wrapper}_joint"
    while joint_name in names:
        joint_name += "_"

    pitch = math.atan2(-matrix[2][0], math.hypot(matrix[0][0], matrix[1][0]))
    if math.hypot(matrix[0][0], matrix[1][0]) > 1e-9:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0

    ET.SubElement(robot, "link", {"name": wrapper})
    fixed = ET.SubElement(robot, "joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(fixed, "parent", {"link": wrapper})
    ET.SubElement(fixed, "child", {"link": roots[0]})
    ET.SubElement(fixed, "origin", {
        "xyz": "0 0 0",
        "rpy": " ".join(f"{value:.12g}" for value in (roll, pitch, yaw)),
    })
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return True


def _lexical_absolute(path):
    return Path(os.path.abspath(str(path)))


def _reject_symlink_chain(trusted_root, target, label):
    trusted_root = _lexical_absolute(trusted_root)
    target = _lexical_absolute(target)
    try:
        relative = target.relative_to(trusted_root)
    except ValueError:
        raise ValueError(f"{label} must remain beneath trusted root: {target}") from None
    component = trusted_root
    for part in relative.parts:
        component /= part
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks: {component}")
    trusted_resolved = trusted_root.resolve()
    target_resolved = target.resolve()
    if target_resolved != trusted_resolved and trusted_resolved not in target_resolved.parents:
        raise ValueError(f"{label} resolves outside trusted root: {target}")
    return target


def validate_urdf_assets(package_root, trusted_root=None):
    package_root = _reject_symlink_chain(
        trusted_root if trusted_root is not None else Path(package_root).parent,
        package_root,
        "URDF package",
    )
    urdf_path = package_root / "model.urdf"
    if not urdf_path.is_file() or urdf_path.is_symlink() or urdf_path.stat().st_size == 0:
        raise ValueError(f"URDF must be a non-empty regular file: {urdf_path}")
    try:
        xml = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML: {urdf_path}") from exc
    filenames = [
        node.attrib.get("filename")
        for kind in xml.iter()
        if kind.tag.rsplit("}", 1)[-1] in {"visual", "collision"}
        for node in kind.iter()
        if node.tag.rsplit("}", 1)[-1] == "mesh"
    ]
    if not filenames:
        raise ValueError(f"URDF has no visual/collision mesh references: {urdf_path}")
    assets = []
    for filename in filenames:
        if (
            not isinstance(filename, str)
            or not filename
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", filename)
            or Path(filename).is_absolute()
            or PureWindowsPath(filename).is_absolute()
            or ".." in Path(filename).parts
            or ".." in PureWindowsPath(filename).parts
        ):
            raise ValueError(f"invalid URDF mesh filename: {filename!r}")
        referenced = package_root / filename
        _reject_symlink_chain(package_root, referenced, "URDF mesh")
        asset = referenced.resolve()
        package_resolved = package_root.resolve()
        if package_resolved not in asset.parents:
            raise ValueError(f"URDF mesh resolves outside package: {filename!r}")
        if not asset.is_file() or asset.stat().st_size == 0:
            raise ValueError(f"URDF mesh must be an existing non-empty regular file: {filename!r}")
        if asset not in assets:
            assets.append(asset)
    return assets


def namespace_urdf_assets(package_root, prefix):
    """Give a copied URDF package scene-global robot/link/joint names."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError(f"invalid URDF mesh prefix: {prefix!r}")
    package_root = Path(package_root)
    urdf_path = package_root / "model.urdf"
    validate_urdf_assets(package_root)
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    robot.set("name", prefix)
    link_names = {}
    for link in (node for node in robot if node.tag.rsplit("}", 1)[-1] == "link"):
        old_name = link.attrib["name"]
        link_names[old_name] = f"{prefix}_{old_name}"
        link.set("name", link_names[old_name])
    for joint in (node for node in robot if node.tag.rsplit("}", 1)[-1] == "joint"):
        joint.set("name", f"{prefix}_{joint.attrib['name']}")
        for endpoint in joint:
            if endpoint.tag.rsplit("}", 1)[-1] in {"parent", "child"}:
                endpoint.set("link", link_names[endpoint.attrib["link"]])
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def use_visual_meshes_for_collisions(package_root):
    """Preserve each link's original mesh geometry for downstream SDF cooking."""
    urdf_path = Path(package_root) / "model.urdf"
    tree = ET.parse(urdf_path)
    changed = 0
    for link in tree.getroot().findall("link"):
        visual = link.find("visual")
        collision = link.find("collision")
        if visual is None and collision is None:
            continue
        visual_mesh = visual.find("geometry/mesh") if visual is not None else None
        collision_mesh = (
            collision.find("geometry/mesh") if collision is not None else None
        )
        if visual_mesh is None or collision_mesh is None:
            raise ValueError(
                f"link {link.get('name')!r} requires visual and collision meshes"
            )
        collision_mesh.attrib.clear()
        collision_mesh.attrib.update(visual_mesh.attrib)
        visual_origin = visual.find("origin")
        collision_origin = collision.find("origin")
        if collision_origin is None:
            collision_origin = ET.SubElement(collision, "origin")
        collision_origin.attrib.clear()
        if visual_origin is not None:
            collision_origin.attrib.update(visual_origin.attrib)
        changed += 1
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    validate_urdf_assets(package_root)
    return changed


def _vector(text, default):
    values = [float(value) for value in (text or "").split()]
    if not values:
        values = list(default)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected a finite three-component vector, got {text!r}")
    return values


def _format_vector(values):
    return " ".join(f"{value:.12g}" for value in values)


def _rotation_from_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _scale_in_rotated_frame(scale, rpy):
    rotation = _rotation_from_rpy(rpy)
    matrix = [
        [
            sum(rotation[axis][row] * scale[axis] * rotation[axis][column]
                for axis in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    tolerance = max(scale) * 1e-6
    if any(
        abs(matrix[row][column]) > tolerance
        for row in range(3) for column in range(3) if row != column
    ):
        raise ValueError(
            "nonuniform scene scale combined with a non-axis-aligned URDF "
            "geometry rotation would require shear"
        )
    return [matrix[index][index] for index in range(3)]


def _axis_scale(axis, scale, rpy=(0.0, 0.0, 0.0)):
    source_length = math.sqrt(sum(value * value for value in axis))
    if source_length <= 1e-12:
        raise ValueError("URDF joint axis must be nonzero")
    axis = [value / source_length for value in axis]
    rotation = _rotation_from_rpy(rpy)
    parent_axis = [
        sum(rotation[row][column] * axis[column] for column in range(3))
        for row in range(3)
    ]
    scaled_parent_axis = [
        scale[index] * parent_axis[index] for index in range(3)
    ]
    scaled_axis = [
        sum(rotation[row][column] * scaled_parent_axis[row] for row in range(3))
        for column in range(3)
    ]
    length = math.sqrt(sum(value * value for value in scaled_axis))
    return [value / length for value in scaled_axis], length


def urdf_prismatic_scales(urdf_path, scale):
    """Return physical distance multipliers for the URDF's prismatic joints."""
    scale = _vector(_format_vector(scale), ())
    if any(value <= 0 for value in scale):
        raise ValueError("scene scale must contain three positive values")
    robot = ET.parse(urdf_path).getroot()
    factors = {}
    for joint in robot.findall("joint"):
        if joint.get("type") != "prismatic":
            continue
        origin = joint.find("origin")
        rpy = _vector(origin.get("rpy") if origin is not None else None, (0, 0, 0))
        axis_node = joint.find("axis")
        axis = _vector(axis_node.get("xyz") if axis_node is not None else None, (1, 0, 0))
        _scaled_axis, factor = _axis_scale(axis, scale, rpy)
        factors[joint.get("name")] = factor
    return factors


def bake_urdf_scale(package_root, scale):
    """Bake an outer scene scale into a copied URDF package in place."""
    package_root = Path(package_root)
    urdf_path = package_root / "model.urdf"
    scale = _vector(_format_vector(scale), ())
    if any(value <= 0 for value in scale):
        raise ValueError("scene scale must contain three positive values")
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    for link in robot.findall("link"):
        for item in list(link.findall("visual")) + list(link.findall("collision")):
            origin = item.find("origin")
            rpy = _vector(origin.get("rpy") if origin is not None else None, (0, 0, 0))
            if origin is not None:
                xyz = _vector(origin.get("xyz"), (0, 0, 0))
                origin.set("xyz", _format_vector(
                    [xyz[index] * scale[index] for index in range(3)]
                ))
            geometry = item.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                raise ValueError("scaled Isaac URDF requires mesh geometry")
            local_scale = _scale_in_rotated_frame(scale, rpy)
            existing = _vector(mesh.get("scale"), (1, 1, 1))
            mesh.set("scale", _format_vector(
                [existing[index] * local_scale[index] for index in range(3)]
            ))

        inertial = link.find("inertial")
        origin = inertial.find("origin") if inertial is not None else None
        if origin is not None:
            xyz = _vector(origin.get("xyz"), (0, 0, 0))
            origin.set("xyz", _format_vector(
                [xyz[index] * scale[index] for index in range(3)]
            ))

    factors = {}
    for joint in robot.findall("joint"):
        origin = joint.find("origin")
        rpy = _vector(origin.get("rpy") if origin is not None else None, (0, 0, 0))
        if origin is not None:
            xyz = _vector(origin.get("xyz"), (0, 0, 0))
            origin.set("xyz", _format_vector(
                [xyz[index] * scale[index] for index in range(3)]
            ))
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        axis_node = joint.find("axis")
        if axis_node is None:
            axis_node = ET.SubElement(joint, "axis")
        axis = _vector(axis_node.get("xyz"), (1, 0, 0))
        scaled_axis, factor = _axis_scale(axis, scale, rpy)
        axis_node.set("xyz", _format_vector(scaled_axis))
        if joint.get("type") != "prismatic":
            continue
        factors[joint.get("name")] = factor
        limit = joint.find("limit")
        if limit is not None:
            for attribute in ("lower", "upper", "velocity"):
                if attribute in limit.attrib:
                    limit.set(attribute, f"{float(limit.get(attribute)) * factor:.12g}")

    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return factors


def urdf_root_pose(scene_item, root_offset_local):
    """Convert Blender's placed support anchor into the URDF root frame."""
    scale = scene_item["scale_xyz"]
    scaled = [float(root_offset_local[i]) * float(scale[i]) for i in range(3)]
    roll_x_deg = float(scene_item.get("roll_x_deg", 0.0))
    roll_x = math.radians(roll_x_deg)
    rolled_x = [
        scaled[0],
        math.cos(roll_x) * scaled[1] - math.sin(roll_x) * scaled[2],
        math.sin(roll_x) * scaled[1] + math.cos(roll_x) * scaled[2],
    ]
    roll_y_deg = float(scene_item.get("roll_y_deg", 0.0))
    roll_y = math.radians(roll_y_deg)
    rolled = [
        math.cos(roll_y) * rolled_x[0] + math.sin(roll_y) * rolled_x[2],
        rolled_x[1],
        -math.sin(roll_y) * rolled_x[0] + math.cos(roll_y) * rolled_x[2],
    ]
    yaw = math.radians(float(scene_item["yaw_deg"]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated = [
        cosine * rolled[0] - sine * rolled[1],
        sine * rolled[0] + cosine * rolled[1],
        rolled[2],
    ]
    return {
        "translation_m": [
            float(scene_item["location_m"][i]) + rotated[i] for i in range(3)
        ],
        "yaw_deg": float(scene_item["yaw_deg"]),
        "roll_x_deg": roll_x_deg,
        "roll_y_deg": roll_y_deg,
        "scale": [float(value) for value in scale],
    }
