#!/usr/bin/env python3
"""Geometry-only support pivot shared by CPU alignment and Blender placement."""

import numpy as np


def _cross(origin, first, second):
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _convex_hull(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _polygon_centroid(polygon):
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return polygon.mean(axis=0), 0.0
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * float(cross.sum())
    if abs(signed_area) < 1e-12:
        return polygon.mean(axis=0), 0.0
    centroid = np.array([
        np.sum((polygon[:, 0] + following[:, 0]) * cross),
        np.sum((polygon[:, 1] + following[:, 1]) * cross),
    ]) / (6.0 * signed_area)
    return centroid, abs(signed_area)


def support_pivot(vertices):
    """Return the XY support-hull area centroid at the mesh's minimum Z."""
    vertices = np.asarray(vertices, dtype=float)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if not len(vertices):
        raise ValueError("cannot compute a support pivot from an empty mesh")
    z_min = float(vertices[:, 2].min())
    height = max(float(np.ptp(vertices[:, 2])), 1e-6)
    fallback = vertices[np.argmin(vertices[:, 2]), :2]
    for fraction in (0.002, 0.005, 0.01, 0.02):
        epsilon = max(fraction * height, 1e-7)
        bottom = vertices[vertices[:, 2] <= z_min + epsilon, :2]
        if not len(bottom):
            continue
        fallback = bottom.mean(axis=0)
        hull = _convex_hull(bottom)
        centroid, area = _polygon_centroid(hull)
        if area > 1e-12 * max(float(np.ptp(vertices[:, :2], axis=0).prod()), 1.0):
            return np.array([centroid[0], centroid[1], z_min]), epsilon
    return np.array([fallback[0], fallback[1], z_min]), 0.02 * height


def center_vertices_on_support(vertices):
    vertices = np.asarray(vertices, dtype=float).copy()
    pivot, _ = support_pivot(vertices)
    vertices -= pivot
    return vertices
