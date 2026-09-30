#!/usr/bin/env python3
import math
import statistics
from pathlib import Path


PROFILE_FIELDS = {
    "box": (("width", "length", "height"), ("width", "length", "height")),
    "panel": (("width", "length", "thickness"), ("width", "length", "thickness")),
    "axial": (("radial_1", "radial_2", "height"), ("outer_diameter", "outer_diameter", "height")),
    "sphere": (("x", "y", "z"), ("diameter", "diameter", "diameter")),
    "sweep": (("span", "reach", "section"), ("span", "reach", "section_diameter")),
    "freeform": (("x", "y", "z"), ("width", "length", "height")),
}


def canonical_zup_to_package_yup(vector):
    x, y, z = (float(value) for value in vector)
    return (x, -z, y)


def _blender_zup_to_package_yup(vector):
    x, y, z = (float(value) for value in vector)
    return (x, z, -y)


def blender_zup_affine_to_package_yup(transform):
    matrix = transform["matrix"]
    columns = []
    for package_axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        blender_axis = canonical_zup_to_package_yup(package_axis)
        moved = tuple(
            sum(float(matrix[row][column]) * blender_axis[column] for column in range(3))
            for row in range(3)
        )
        columns.append(_blender_zup_to_package_yup(moved))
    package_matrix = tuple(
        tuple(columns[column][row] for column in range(3))
        for row in range(3)
    )
    return {
        "matrix": package_matrix,
        "offset": _blender_zup_to_package_yup(transform["offset"]),
    }


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("axis must contain three numbers")
    vector = tuple(float(item) for item in value)
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-9:
        raise ValueError("axis must be nonzero")
    return tuple(item / length for item in vector)


def frame_from_axes(axes):
    vectors = tuple(_normalize(axis) for axis in axes)
    if len(vectors) != 3:
        raise ValueError("frame must contain three axes")
    if any(abs(_dot(vectors[i], vectors[j])) > 1e-5 for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("frame axes must be orthogonal")
    if _dot(_cross(vectors[0], vectors[1]), vectors[2]) < 1 - 1e-5:
        raise ValueError("frame axes must be right-handed")
    return tuple(tuple(vectors[column][row] for column in range(3)) for row in range(3))


def radial_frame(joint_axis, anchor, centroid):
    axis = _normalize(joint_axis)
    direction = tuple(float(centroid[i]) - float(anchor[i]) for i in range(3))
    radial = tuple(direction[i] - _dot(direction, axis) * axis[i] for i in range(3))
    radial = _normalize(radial)
    normal = _normalize(_cross(axis, radial))
    return frame_from_axes((axis, radial, normal))


def profile_axes_and_dimensions(part):
    geometry_type = part.get("geometry_type")
    if geometry_type == "none":
        return (), ()
    if geometry_type not in PROFILE_FIELDS:
        raise ValueError(f"unsupported geometry_type: {geometry_type}")
    axis_names, dimension_names = PROFILE_FIELDS[geometry_type]
    local_axes = part.get("local_axes") or {}
    axes = tuple(_normalize(local_axes.get(name)) for name in axis_names)
    frame_from_axes(axes)
    bbox = part.get("bbox_cm")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 3:
            raise ValueError("bbox_cm must contain three values")
        values = tuple(float(value) for value in bbox)
    else:
        dimensions = part.get("dimensions_cm") or {}
        values = tuple(float(dimensions.get(name, 0)) for name in dimension_names)
    if any(value <= 0 for value in values):
        raise ValueError(f"{geometry_type} dimensions must be positive")
    return axes, values


def bbox_scale_factors(measured, target, preserve_proportions=False):
    if len(measured) != 3 or len(target) != 3:
        raise ValueError("measured and target extents must contain three values")
    if any(float(value) <= 0 for value in measured) or any(float(value) <= 0 for value in target):
        raise ValueError("bbox extents must be positive")
    if preserve_proportions:
        factor = max(float(value) for value in target) / max(float(value) for value in measured)
        return (factor, factor, factor)
    return tuple(float(target[i]) / float(measured[i]) for i in range(3))


def rigid_axis_mapping_and_scale(measured, target):
    """Match X/Y orientation, keep Z as height, and return one uniform scale."""
    if len(measured) != 3 or len(target) != 3:
        raise ValueError("measured and target extents must contain three values")
    measured = tuple(float(value) for value in measured)
    target = tuple(float(value) for value in target)
    if any(value <= 0 for value in measured + target):
        raise ValueError("bbox extents must be positive")

    candidates = []
    for turns, mapped in ((0, measured), (1, (measured[1], measured[0], measured[2]))):
        scale = float(statistics.median(target[i] / mapped[i] for i in range(3)))
        error = sum(math.log(mapped[i] * scale / target[i]) ** 2 for i in range(3))
        candidates.append((error, turns, scale))
    _, turns, scale = min(candidates)
    return turns, scale


def estimate_units_per_cm(measurements):
    ratios = []
    for measured, blueprint in measurements:
        measured = float(measured)
        blueprint = float(blueprint)
        if measured <= 0 or blueprint <= 0:
            raise ValueError("scale measurements must be positive")
        ratios.append(measured / blueprint)
    if not ratios:
        raise ValueError("at least one scale measurement is required")
    return float(statistics.median(ratios))


def scale_factors(measured, target, geometry_type):
    if len(measured) != 3 or len(target) != 3:
        raise ValueError("measured and target extents must contain three values")
    ratios = tuple(float(target[i]) / float(measured[i]) for i in range(3))
    if geometry_type == "freeform":
        factor = float(statistics.median(ratios))
        ratios = (factor, factor, factor)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError(f"invalid scale factor: {ratios}")
    return ratios


def affine_about(anchor, frame, scales):
    anchor = tuple(float(value) for value in anchor)
    scales = tuple(float(value) for value in scales)
    matrix = tuple(
        tuple(sum(frame[row][axis] * scales[axis] * frame[col][axis] for axis in range(3)) for col in range(3))
        for row in range(3)
    )
    transformed_anchor = tuple(sum(matrix[row][col] * anchor[col] for col in range(3)) for row in range(3))
    offset = tuple(anchor[i] - transformed_anchor[i] for i in range(3))
    return {"matrix": matrix, "offset": offset}


def affine_between(source_anchor, target_anchor, frame, scales):
    transform = affine_about(source_anchor, frame, scales)
    moved = transform_point(transform, source_anchor)
    transform["offset"] = tuple(
        transform["offset"][i] + float(target_anchor[i]) - moved[i]
        for i in range(3)
    )
    return transform


def transform_point(transform, point):
    point = tuple(float(value) for value in point)
    matrix = transform["matrix"]
    offset = transform["offset"]
    result = tuple(sum(matrix[row][col] * point[col] for col in range(3)) + offset[row] for row in range(3))
    return tuple(0.0 if abs(value) < 1e-12 else value for value in result)


def transform_obj(source, destination, transform):
    source = Path(source)
    destination = Path(destination)
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("v "):
            fields = line.rstrip("\r\n").split()
            point = transform_point(transform, fields[1:4])
            values = " ".join(format(value, ".12g") for value in point)
            suffix = " " + " ".join(fields[4:]) if len(fields) > 4 else ""
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            line = f"v {values}{suffix}{ending}"
        lines.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="utf-8")
