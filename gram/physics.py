from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .artifacts import write_json
from .joints import JointEstimate


def transform_revolute(points: np.ndarray, axis: np.ndarray, pivot: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    relative = np.asarray(points, dtype=float) - pivot
    cosine, sine = np.cos(angle), np.sin(angle)
    return pivot + relative * cosine + np.cross(axis, relative) * sine + np.outer(relative @ axis, axis) * (1 - cosine)


def transform_prismatic(points: np.ndarray, axis: np.ndarray, distance: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    return np.asarray(points, dtype=float) + axis / np.linalg.norm(axis) * distance


def _moved_mesh(mesh: trimesh.Trimesh, estimate: JointEstimate, amount: float) -> trimesh.Trimesh:
    moved = mesh.copy()
    if estimate.type == "revolute":
        moved.vertices = transform_revolute(moved.vertices, estimate.axis, estimate.pivot, amount)
    else:
        moved.vertices = transform_prismatic(moved.vertices, estimate.axis, amount)
    return moved


class CollisionEvaluator:
    def __init__(self, obstacles: Sequence[trimesh.Trimesh] = ()):
        try:
            self.manager = trimesh.collision.CollisionManager()
        except ValueError as exc:
            raise RuntimeError("python-fcl is required for collision evaluation; use Ubuntu/WSL2") from exc
        for index, obstacle in enumerate(obstacles):
            self.manager.add_object(f"obstacle_{index}", obstacle)
        self.collision_floor = None

    def calibrate(self, moving: trimesh.Trimesh, clearance: float) -> None:
        if clearance < 0:
            raise ValueError("collision clearance must be non-negative")
        distance = float(self.manager.min_distance_single(moving))
        self.collision_floor = min(0.0, distance) - clearance

    def distance_and_collision(self, moving: trimesh.Trimesh) -> tuple[float, bool]:
        distance = float(self.manager.min_distance_single(moving))
        collision = bool(self.manager.in_collision_single(moving))
        if self.collision_floor is not None:
            collision = collision and distance < self.collision_floor
        return distance, collision


def _surface_distances(points: np.ndarray, obstacles: Sequence[trimesh.Trimesh],
                       fallback: np.ndarray) -> np.ndarray:
    if not obstacles:
        return cKDTree(fallback).query(points)[0]
    distances = []
    for obstacle in obstacles:
        try:
            _, distance, _ = trimesh.proximity.closest_point(obstacle, points)
        except ModuleNotFoundError as exc:
            raise RuntimeError("accelerated closest-point queries require the rtree package") from exc
        distances.append(distance)
    return np.min(distances, axis=0)


def _spatial_sample(points: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("sample size must be positive")
    points = np.asarray(points)
    if len(points) <= maximum:
        return points
    selected = np.empty(maximum, dtype=int)
    selected[0] = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    nearest = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for index in range(1, maximum):
        selected[index] = int(np.argmax(nearest))
        nearest = np.minimum(nearest, np.sum((points - points[selected[index]]) ** 2, axis=1))
    return points[selected]


@dataclass
class PhysicsConfig:
    diagonal: float
    contact_weight: float = 1.0
    collision_weight: float = 10.0
    axis_reg_weight: float = 0.1
    pivot_reg_weight: float = 0.1
    trajectory_span: float = 1.0
    coarse_contacts: int = 500
    fine_contacts: int = 1000
    coarse_maxiter: int = 80
    fine_maxiter: int = 120
    trace_path: Path | None = None
    objective: Callable[[JointEstimate, np.ndarray, np.ndarray], tuple[float, int]] | None = None
    collision_evaluator: CollisionEvaluator | None = None
    axis_limit_degrees: float | None = None


def trajectory_objective(estimate: JointEstimate, initial: JointEstimate, contact_points: np.ndarray,
                         moving_subtree, obstacles, config: PhysicsConfig,
                         samples: np.ndarray | None = None) -> tuple[float, int]:
    if samples is None:
        samples = (np.deg2rad([-30, -15, 15, 30]) * config.trajectory_span
                   if estimate.type == "revolute" else
                   np.array([-.08, -.04, .04, .08]) * config.diagonal * config.trajectory_span)
    if config.objective:
        return config.objective(estimate, contact_points, samples)
    evaluator = config.collision_evaluator or CollisionEvaluator(obstacles)
    contact_loss, collisions = 0.0, 0
    for amount in samples:
        moved_contacts = (transform_revolute(contact_points, estimate.axis, estimate.pivot, amount)
                          if estimate.type == "revolute" else transform_prismatic(contact_points, estimate.axis, amount))
        contact_loss += float(np.mean(_surface_distances(moved_contacts, obstacles, contact_points) ** 2))
        if moving_subtree is not None:
            _, collided = evaluator.distance_and_collision(_moved_mesh(moving_subtree, estimate, amount))
            collisions += int(collided)
    contact_loss /= len(samples)
    axis_reg = 1 - abs(float(np.dot(estimate.axis, initial.axis)))
    pivot_reg = float(np.sum((estimate.pivot - initial.pivot) ** 2)) / config.diagonal ** 2
    return (config.contact_weight * contact_loss + config.collision_weight * collisions
            + config.axis_reg_weight * axis_reg + config.pivot_reg_weight * pivot_reg), collisions


def refine_joint(estimate: JointEstimate, contact_points: np.ndarray, moving_subtree,
                 obstacles: Sequence[trimesh.Trimesh], config: PhysicsConfig) -> JointEstimate:
    if config.diagonal <= 0:
        raise ValueError("working mesh diagonal must be positive")
    trace = []

    def limit_axis(axis):
        axis = np.asarray(axis, dtype=float)
        if np.linalg.norm(axis) < 1e-10:
            return estimate.axis
        axis /= np.linalg.norm(axis)
        if config.axis_limit_degrees is None:
            return axis
        if not 0 < config.axis_limit_degrees <= 180:
            raise ValueError("axis_limit_degrees must be in (0, 180]")
        if np.dot(axis, estimate.axis) < 0:
            axis = -axis
        cosine = float(np.clip(np.dot(axis, estimate.axis), -1, 1))
        limit = np.deg2rad(config.axis_limit_degrees)
        if np.arccos(cosine) <= limit:
            return axis
        tangent = axis - cosine * estimate.axis
        tangent /= np.linalg.norm(tangent)
        return np.cos(limit) * estimate.axis + np.sin(limit) * tangent

    def unpack(vector):
        axis = limit_axis(vector[:3])
        return JointEstimate(estimate.type, estimate.subtype, axis, vector[3:], estimate.lower, estimate.upper)

    displacement = 0.05 * config.diagonal
    bounds = [(None, None)] * 3 + [(float(x - displacement), float(x + displacement)) for x in estimate.pivot]

    def optimize(start, points, samples, maxiter, stage):
        def objective(vector):
            candidate = unpack(vector)
            loss, collided = trajectory_objective(
                candidate, estimate, points, moving_subtree, obstacles, config, samples)
            trace.append({"stage": stage, "loss": float(loss), "collided_states": int(collided),
                          "estimate": candidate.to_dict()})
            return loss

        result = minimize(objective, np.r_[start.axis, start.pivot], method="Nelder-Mead", bounds=bounds,
                          options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-9})
        return unpack(result.x)

    coarse_samples = (np.deg2rad([-15, 15]) * config.trajectory_span
                      if estimate.type == "revolute" else
                      np.array([-.04, .04]) * config.diagonal * config.trajectory_span)
    fine_samples = (np.deg2rad([-30, -15, 15, 30]) * config.trajectory_span
                    if estimate.type == "revolute" else
                    np.array([-.08, -.04, .04, .08]) * config.diagonal * config.trajectory_span)
    coarse = optimize(estimate, _spatial_sample(contact_points, config.coarse_contacts),
                      coarse_samples, config.coarse_maxiter, "coarse")
    candidate = optimize(coarse, _spatial_sample(contact_points, config.fine_contacts),
                         fine_samples, config.fine_maxiter, "fine")
    before_loss, before_collisions = trajectory_objective(
        estimate, estimate, contact_points, moving_subtree, obstacles, config, fine_samples)
    trace.append({"stage": "validation", "loss": float(before_loss),
                  "collided_states": int(before_collisions), "estimate": estimate.to_dict()})
    after_loss, after_collisions = trajectory_objective(
        candidate, estimate, contact_points, moving_subtree, obstacles, config, fine_samples)
    trace.append({"stage": "validation", "loss": float(after_loss),
                  "collided_states": int(after_collisions), "estimate": candidate.to_dict()})
    accepted = after_loss < before_loss and after_collisions <= before_collisions
    if config.trace_path:
        write_json(config.trace_path, {"accepted": accepted, "before_loss": before_loss,
                                      "after_loss": after_loss, "evaluations": trace})
    return candidate if accepted else estimate


def _limit_evaluator(estimate, moving, obstacles, amount, contact_points=None, contact_threshold=None,
                     collision_evaluator=None):
    moved = _moved_mesh(moving, estimate, amount)
    collision_evaluator = collision_evaluator or CollisionEvaluator(obstacles)
    _, collision = collision_evaluator.distance_and_collision(moved)
    fraction = 1.0
    if contact_points is not None and obstacles and contact_threshold:
        moved_points = (transform_revolute(contact_points, estimate.axis, estimate.pivot, amount)
                        if estimate.type == "revolute" else transform_prismatic(contact_points, estimate.axis, amount))
        fraction = float(np.mean(_surface_distances(moved_points, obstacles[:1], contact_points) <= contact_threshold))
    return collision, fraction


def _scan_limits(maximum: float, step: float, evaluator, require_contact: bool) -> tuple[float, float]:
    collision, _ = evaluator(0.0)
    if collision:
        raise ValueError("joint rest state is in collision")
    limits = []
    for sign in (-1, 1):
        last = 0.0
        value = step
        while value <= maximum + 1e-12:
            collision, fraction = evaluator(sign * value)
            if collision or (require_contact and fraction < 0.05):
                break
            last = value
            value += step
        limits.append(sign * last)
    return limits[0], limits[1]


def find_revolute_limits(estimate: JointEstimate, moving, obstacles, *, evaluator=None,
                          collision_evaluator=None, step_degrees: float = 2.0,
                          max_degrees: float = 180.0) -> tuple[float, float]:
    if evaluator is None:
        collision_evaluator = collision_evaluator or CollisionEvaluator(obstacles)
        evaluator = lambda amount: _limit_evaluator(
            estimate, moving, obstacles, amount, collision_evaluator=collision_evaluator)
    return _scan_limits(np.deg2rad(max_degrees), np.deg2rad(step_degrees), evaluator, False)


def find_prismatic_limits(estimate: JointEstimate, moving, obstacles, *, diagonal: float,
                           evaluator=None, contact_points=None, contact_ratio: float = 0.015,
                           collision_evaluator=None, step_ratio: float = 0.01,
                           max_ratio: float = 1.0) -> tuple[float, float]:
    if evaluator is None:
        collision_evaluator = collision_evaluator or CollisionEvaluator(obstacles)
        evaluator = lambda amount: _limit_evaluator(
            estimate, moving, obstacles, amount, contact_points, contact_ratio * diagonal,
            collision_evaluator)
    return _scan_limits(max_ratio * diagonal, step_ratio * diagonal, evaluator, True)
