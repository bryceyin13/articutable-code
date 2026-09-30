#!/usr/bin/env python3
import asyncio
import hashlib
import json
import math
import os
import runpy
import traceback
from pathlib import Path

import carb
import numpy as np
import omni.kit.app
import omni.kit.renderer_capture
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import (
    capture_viewport_to_file,
    get_active_viewport,
)
from PIL import Image
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


CAMERA_PATH = "/World/BlenderScene/front_camera/front_camera"
BLENDER_SCENE_PATH = "/World/BlenderScene"
COLLISION_EVALUATION_VERSION = "physx_scene_query_overlap_shape_v1"


def log(message):
    print(f"[joint-video] {message}", flush=True)


def write_json_atomic(output_path, payload):
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authored_pose_snapshot(stage):
    snapshot = {}
    joint_states = {}
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Xformable):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            snapshot[str(prim.GetPath())] = [
                float(matrix[row][column])
                for row in range(4) for column in range(4)
            ]
        states = {
            attribute.GetName(): attribute.Get(Usd.TimeCode.Default())
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith("state:")
            and attribute.GetName().endswith(
                ("physics:position", "physics:velocity")
            )
        }
        if states:
            joint_states[str(prim.GetPath())] = states
    encoded = json.dumps(
        {"xforms": snapshot, "joint_states": joint_states},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "xform_count": len(snapshot),
        "joint_state_count": sum(len(value) for value in joint_states.values()),
        "joint_states": joint_states,
    }


def cooked_collision_inventory(stage):
    cooked = []
    missing = []
    for prim in stage.TraverseAll():
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        approximation = (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        )
        if approximation not in {"sdf", "convexDecomposition"}:
            continue
        buffers = [
            attribute for attribute in prim.GetAttributes()
            if attribute.GetName().startswith("physxCookedData:")
            and attribute.GetName().endswith(":buffer")
            and attribute.Get() is not None
            and len(attribute.Get())
        ]
        target = cooked if buffers else missing
        target.append({
            "path": str(prim.GetPath()),
            "approximation": approximation,
            "buffers": [attribute.GetName() for attribute in buffers],
        })
    return {"cooked": cooked, "missing": missing}


async def capture_packaged_scene_without_simulation(config):
    package_path = Path(config["package_path"]).resolve()
    output_path = Path(config["output_path"]).resolve()
    status_path = Path(config["status_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("timeline must be paused before direct package validation")
    timeline_time_before = float(timeline.get_current_time())
    package_sha256_before = sha256_file(package_path)

    settings = carb.settings.get_settings()
    settings.set_bool("physics/updateToUsd", False)
    settings.set_bool("/rtx/post/histogram/enabled", False)
    settings.set_int("/rtx/post/aa/autoExposureMode", 2)
    settings.set_float("/rtx/post/aa/exposure", 0.6)
    settings.set_string("/rtx/rendermode", "RaytracedLighting")

    context = omni.usd.get_context()
    if not context.open_stage(str(package_path)):
        raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    app = omni.kit.app.get_app()
    while context.get_stage_loading_status()[2] > 0:
        if timeline.is_playing():
            raise RuntimeError("timeline started while loading the USDZ")
        await app.next_update_async()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Isaac returned no stage for USDZ: {package_path}")
    root_layer = stage.GetRootLayer()
    session_layer = stage.GetSessionLayer()
    root_dirty_before = root_layer.dirty
    session_dirty_before = session_layer.dirty
    pose_before = authored_pose_snapshot(stage)

    camera_path = Sdf.Path(config.get("camera_path", CAMERA_PATH))
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"camera not found in USDZ: {camera_path}")
    camera = UsdGeom.Camera(camera_prim)
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    capture_resolution = (
        int(config["width"]),
        round(int(config["width"]) * vertical_aperture / horizontal_aperture),
    )
    output_resolution = (int(config["width"]), int(config["height"]))
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport not available")
    viewport.camera_path = camera_path
    viewport.resolution = capture_resolution
    viewport.resolution_scale = 1.0
    settings.set_bool("/app/viewport/grid/enabled", False)
    settings.set_bool(
        f"/persistent/app/viewport/{viewport.id}/guide/grid/visible", False,
    )
    settings.set_bool(
        f"/persistent/app/viewport/{viewport.id}/guide/axis/visible", False,
    )
    settings.set_bool(
        f"/persistent/app/viewport/{viewport.id}/guide/selection/visible", False,
    )

    warmup_path = output_path.with_name(f".{output_path.stem}_warmup.png")
    warmup = await capture_viewport_to_file(
        viewport, file_path=str(warmup_path), is_hdr=False,
    ).wait_for_result(completion_frames=0)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    if not warmup or not warmup_path.is_file():
        raise RuntimeError("failed to render directly opened USDZ")
    warmup_path.unlink()

    spp = int(config["pathtracing_spp"])
    settings.set_int("/rtx/pathtracing/spp", spp)
    settings.set_int("/rtx/pathtracing/totalSpp", spp)
    settings.set_int("/rtx/pathtracing/clampSpp", 0)
    settings.set_bool(
        "/rtx/pathtracing/optixDenoiser/enabled",
        bool(config["pathtracing_denoiser"]),
    )
    viewport.set_hd_engine("rtx", "PathTracing")
    settings.set_string("/rtx/rendermode", "PathTracing")
    for _ in range(2):
        if timeline.is_playing():
            raise RuntimeError("timeline started during renderer refresh")
        await app.next_update_async()

    capture_path = (
        output_path if capture_resolution == output_resolution
        else output_path.with_name(f".{output_path.stem}_capture.png")
    )
    result = await capture_viewport_to_file(
        viewport, file_path=str(capture_path), is_hdr=False,
    ).wait_for_result(completion_frames=0)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    if not result or not capture_path.is_file():
        raise RuntimeError(f"failed to capture directly opened USDZ: {package_path}")
    if capture_path != output_path:
        with Image.open(capture_path) as image:
            image.resize(output_resolution, Image.Resampling.LANCZOS).save(
                output_path
            )
        capture_path.unlink()

    pose_after = authored_pose_snapshot(stage)
    timeline_time_after = float(timeline.get_current_time())
    package_sha256_after = sha256_file(package_path)
    pose_unchanged = pose_before["sha256"] == pose_after["sha256"]
    if timeline.is_playing() or timeline_time_after != timeline_time_before:
        raise RuntimeError("timeline changed during direct USDZ validation")
    if not pose_unchanged:
        raise RuntimeError("scene pose changed during direct USDZ validation")
    if package_sha256_after != package_sha256_before:
        raise RuntimeError("direct USDZ validation modified the package file")

    write_json_atomic(status_path, {
        "status": "complete",
        "verification_mode": "direct_usdz_open_no_simulation",
        "package_path": str(package_path),
        "package_root_layer": root_layer.identifier,
        "output_path": str(output_path),
        "camera_path": str(camera_path),
        "renderer": "RTX Path Tracing",
        "samples_per_pixel": spp,
        "timeline_played": False,
        "timeline_time_before": timeline_time_before,
        "timeline_time_after": timeline_time_after,
        "pose_unchanged": pose_unchanged,
        "pose_before": pose_before,
        "pose_after": pose_after,
        "package_sha256_before": package_sha256_before,
        "package_sha256_after": package_sha256_after,
        "root_layer_dirty_before": root_dirty_before,
        "root_layer_dirty_after": root_layer.dirty,
        "session_layer_dirty_before": session_dirty_before,
        "session_layer_dirty_after": session_layer.dirty,
    })
    log(f"direct USDZ validation complete: {package_path}")


def physics_inventory(stage):
    counts = {
        "physics_scenes": 0,
        "rigid_bodies": 0,
        "colliders": 0,
        "mass_apis": 0,
        "articulation_roots": 0,
        "revolute_joints": 0,
        "prismatic_joints": 0,
        "fixed_joints": 0,
    }
    rigid_bodies = []
    articulation_roots = []
    movable_joints = []
    for prim in stage.Traverse():
        counts["physics_scenes"] += int(prim.IsA(UsdPhysics.Scene))
        counts["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        counts["colliders"] += int(prim.HasAPI(UsdPhysics.CollisionAPI))
        counts["mass_apis"] += int(prim.HasAPI(UsdPhysics.MassAPI))
        counts["articulation_roots"] += int(
            prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        counts["revolute_joints"] += int(
            prim.IsA(UsdPhysics.RevoluteJoint)
        )
        counts["prismatic_joints"] += int(
            prim.IsA(UsdPhysics.PrismaticJoint)
        )
        counts["fixed_joints"] += int(prim.IsA(UsdPhysics.FixedJoint))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)
        if (
            prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
        ):
            movable_joints.append(prim)

    collision_roots = []
    missing_collision_roots = []
    for prim in rigid_bodies:
        has_collision = any(
            item.HasAPI(UsdPhysics.CollisionAPI)
            for item in Usd.PrimRange(prim)
        )
        target = collision_roots if has_collision else missing_collision_roots
        target.append(str(prim.GetPath()))

    bad_joint_bodies = []
    joint_drives = []
    for prim in movable_joints:
        joint = UsdPhysics.Joint(prim)
        body0 = [str(value) for value in joint.GetBody0Rel().GetTargets()]
        body1 = [str(value) for value in joint.GetBody1Rel().GetTargets()]
        if len(body0) != 1 or len(body1) != 1:
            bad_joint_bodies.append({
                "path": str(prim.GetPath()),
                "body0": body0,
                "body1": body1,
            })
        token = (
            "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        )
        state_position = prim.GetAttribute(
            f"state:{token}:physics:position"
        ).Get()
        drive = {
            name: prim.GetAttribute(f"drive:{token}:physics:{name}").Get()
            for name in (
                "stiffness", "damping", "maxForce", "targetPosition",
            )
        }
        stiffness = float(drive["stiffness"] or 0.0)
        max_force = float(drive["maxForce"] or 0.0)
        joint_drives.append({
            "path": str(prim.GetPath()),
            "type": token,
            "authored_position": (
                None if state_position is None else float(state_position)
            ),
            "stiffness": stiffness,
            "damping": float(drive["damping"] or 0.0),
            "max_force": max_force,
            "target_position": float(drive["targetPosition"] or 0.0),
            "active_position_drive": stiffness > 0.0 and max_force > 0.0,
        })

    visual_rigid_bodies = [
        path for path in collision_roots
        if path.startswith(f"{BLENDER_SCENE_PATH}/")
    ]
    missing_visual_collision_roots = [
        path for path in missing_collision_roots
        if path.startswith(f"{BLENDER_SCENE_PATH}/")
    ]
    implicit_dynamic_mesh_colliders = []
    for path in visual_rigid_bodies:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            )
            if approximation not in {
                "convexHull", "convexDecomposition", "box", "sphere", "sdf",
            }:
                implicit_dynamic_mesh_colliders.append(str(prim.GetPath()))
    articulations_without_colliders = []
    for prim in articulation_roots:
        object_root = prim
        while object_root.GetParent().GetPath() != Sdf.Path("/World"):
            object_root = object_root.GetParent()
        if not any(
            item.HasAPI(UsdPhysics.CollisionAPI)
            for item in Usd.PrimRange(object_root)
        ):
            articulations_without_colliders.append(str(prim.GetPath()))
    result = {
        "counts": counts,
        "visual_rigid_bodies": visual_rigid_bodies,
        "articulation_roots": [
            str(prim.GetPath()) for prim in articulation_roots
        ],
        "missing_collision_roots": missing_collision_roots,
        "missing_visual_collision_roots": missing_visual_collision_roots,
        "articulations_without_colliders": articulations_without_colliders,
        "implicit_dynamic_mesh_colliders": implicit_dynamic_mesh_colliders,
        "joint_drives": joint_drives,
        "bad_joint_bodies": bad_joint_bodies,
    }
    if counts["physics_scenes"] != 1:
        raise RuntimeError(
            f"expected one physics scene, found {counts['physics_scenes']}"
        )
    if not visual_rigid_bodies:
        raise RuntimeError("package contains no movable visual rigid body")
    if not articulation_roots or not movable_joints:
        raise RuntimeError("package contains no movable articulation")
    if missing_visual_collision_roots:
        raise RuntimeError(
            "visual rigid bodies without collision geometry: "
            + ", ".join(missing_visual_collision_roots)
        )
    if articulations_without_colliders:
        raise RuntimeError(
            "articulations without collision geometry: "
            + ", ".join(articulations_without_colliders)
        )
    if bad_joint_bodies:
        raise RuntimeError(f"joints with invalid body targets: {bad_joint_bodies}")
    return result


async def validate_packaged_scene_physics(config):
    package_path = Path(config["package_path"]).resolve()
    status_path = Path(config["status_path"]).resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    package_layer = Sdf.Layer.OpenAsAnonymous(str(package_path), True)
    if package_layer is None:
        raise RuntimeError(f"could not read USDZ root layer: {package_path}")
    physics_settings = dict(
        dict(package_layer.customLayerData).get("physicsSettings", {})
    )
    root_update_to_usd = physics_settings.get("/physics/updateToUsd")
    if root_update_to_usd is not True:
        raise RuntimeError(
            "package root layer must set /physics/updateToUsd=true; "
            "the asset is not deterministically sim-ready when opened directly "
            f"(found {root_update_to_usd!r})"
        )

    context = omni.usd.get_context()
    if not context.open_stage(str(package_path)):
        raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    app = omni.kit.app.get_app()
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Isaac returned no stage for USDZ: {package_path}")
    carb.settings.get_settings().set_bool("physics/updateToUsd", True)
    inventory = physics_inventory(stage)

    rigid = SingleRigidPrim(
        inventory["visual_rigid_bodies"][0], name="sim_ready_rigid_probe"
    )
    articulation = SingleArticulation(
        inventory["articulation_roots"][0], name="sim_ready_joint_probe"
    )
    timeline.play()
    await app.next_update_async()
    await app.next_update_async()
    rigid.initialize()
    articulation.initialize()
    if articulation.num_dof < 1:
        raise RuntimeError("articulation has no runtime DOF")

    simulation = omni.physx.get_physx_simulation_interface()
    simulation_time = 0.0

    def step_physics():
        nonlocal simulation_time
        simulation.simulate(1.0 / 60.0, simulation_time)
        simulation.fetch_results()
        simulation_time += 1.0 / 60.0

    initial_rigid_position = np.asarray(rigid.get_world_pose()[0], dtype=float)
    initial_joint_position = np.asarray(
        articulation.get_joint_positions(), dtype=float
    )
    for _ in range(int(config.get("settle_frames", 30))):
        step_physics()
    settled_rigid_position = np.asarray(rigid.get_world_pose()[0], dtype=float)
    settled_joint_position = np.asarray(
        articulation.get_joint_positions(), dtype=float
    )

    force = np.array([[float(config.get("rigid_force_n", 8.0)), 0.0, 0.0]])
    for _ in range(int(config.get("actuation_frames", 20))):
        rigid._rigid_prim_view.apply_forces(force)
        step_physics()
    rigid._rigid_prim_view.apply_forces(np.zeros((1, 3)))
    for _ in range(10):
        step_physics()
    final_rigid_position = np.asarray(rigid.get_world_pose()[0], dtype=float)

    joint_index = np.array([0], dtype=np.int32)
    properties = articulation.dof_properties[0]
    lower = float(properties["lower"])
    upper = float(properties["upper"])
    direction = (
        1.0
        if settled_joint_position[0] < 0.5 * (lower + upper)
        else -1.0
    )
    joint_prim = next(
        prim for prim in stage.Traverse()
        if prim.GetName() == articulation.dof_names[0]
    )
    joint_attributes = {
        attribute.GetName(): attribute.Get()
        for attribute in joint_prim.GetAttributes()
        if "drive" in attribute.GetName()
        or attribute.GetName() in {"physics:lowerLimit", "physics:upperLimit"}
    }
    effort = np.array([
        direction * float(config.get("joint_effort", 5.0))
    ])
    for _ in range(int(config.get("actuation_frames", 20))):
        articulation.set_joint_efforts(effort, joint_indices=joint_index)
        step_physics()
    articulation.set_joint_efforts(np.zeros(1), joint_indices=joint_index)
    for _ in range(10):
        step_physics()
    final_joint_position = np.asarray(
        articulation.get_joint_positions(), dtype=float
    )
    timeline.pause()

    rigid_displacement = float(
        np.linalg.norm(final_rigid_position - settled_rigid_position)
    )
    joint_displacement = float(
        abs(final_joint_position[0] - settled_joint_position[0])
    )
    if not np.isfinite(final_rigid_position).all() or rigid_displacement < 1e-3:
        raise RuntimeError(
            f"rigid body did not respond physically: displacement={rigid_displacement}"
        )
    joint_effort_response = bool(
        np.isfinite(final_joint_position).all() and joint_displacement >= 1e-4
    )
    active_position_drives = [
        item for item in inventory["joint_drives"]
        if item["active_position_drive"]
    ]
    sim_ready_for_robot_contact = bool(
        joint_effort_response
        and not active_position_drives
        and not inventory["implicit_dynamic_mesh_colliders"]
    )

    write_json_atomic(status_path, {
        "status": "complete",
        "verification_mode": "direct_usdz_runtime_physics",
        "package_path": str(package_path),
        "package_root_layer": stage.GetRootLayer().identifier,
        "root_update_to_usd": root_update_to_usd,
        "inventory": inventory,
        "rigid_probe": {
            "path": inventory["visual_rigid_bodies"][0],
            "initial_position_m": initial_rigid_position.tolist(),
            "settled_position_m": settled_rigid_position.tolist(),
            "final_position_m": final_rigid_position.tolist(),
            "forced_displacement_m": rigid_displacement,
        },
        "articulation_probe": {
            "path": inventory["articulation_roots"][0],
            "dof_names": list(articulation.dof_names),
            "initial_positions": initial_joint_position.tolist(),
            "settled_positions": settled_joint_position.tolist(),
            "final_positions": final_joint_position.tolist(),
            "tested_dof_limits": [lower, upper],
            "applied_effort": float(effort[0]),
            "joint_attributes": joint_attributes,
            "effort_driven_displacement": joint_displacement,
            "external_effort_response": joint_effort_response,
        },
        "active_position_drive_count": len(active_position_drives),
        "implicit_dynamic_mesh_collider_count": len(
            inventory["implicit_dynamic_mesh_colliders"]
        ),
        "sim_ready_for_robot_contact": sim_ready_for_robot_contact,
        "package_was_saved": False,
    })
    log(f"runtime physics validation complete: {package_path}")


async def validate_robot_compatibility(config):
    from isaacsim.storage.native import get_assets_root_path

    package_path = Path(config["package_path"]).resolve()
    status_path = Path(config["status_path"]).resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    if not context.open_stage(str(package_path)):
        raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    app = omni.kit.app.get_app()
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()
    stage = context.get_stage()
    inventory = physics_inventory(stage)
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac asset root is unavailable")

    robot_path = Sdf.Path("/World/ValidationFranka")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_prim = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
        robot_prim.GetReferences().AddReference(
            assets_root + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        )
        robot_xform = UsdGeom.Xformable(robot_prim)
        translate_op = next(
            (
                op for op in robot_xform.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        (translate_op or robot_xform.AddTranslateOp()).Set(Gf.Vec3d(3.0, 0.0, 0.0))
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()

    timeline.play()
    for _ in range(4):
        await app.next_update_async()
    robot = SingleArticulation(str(robot_path), name="validation_franka")
    robot.initialize()
    before = np.asarray(robot.get_joint_positions(), dtype=float)
    if robot.num_dof < 7 or not np.isfinite(before).all():
        raise RuntimeError(f"Franka initialization failed: num_dof={robot.num_dof}")
    properties = robot.dof_properties[0]
    lower = float(properties["lower"])
    upper = float(properties["upper"])
    target = min(max(float(before[0]) + 0.25, lower + 0.05), upper - 0.05)
    controller = robot.get_articulation_controller()
    simulation = omni.physx.get_physx_simulation_interface()
    simulation_time = 0.0
    for _ in range(90):
        controller.apply_action(ArticulationAction(
            joint_positions=np.array([target], dtype=np.float32),
            joint_indices=np.array([0], dtype=np.int32),
        ))
        simulation.simulate(1.0 / 60.0, simulation_time)
        simulation.fetch_results()
        simulation_time += 1.0 / 60.0
    after = np.asarray(robot.get_joint_positions(), dtype=float)
    motion = float(abs(after[0] - before[0]))
    timeline.pause()
    if not np.isfinite(after).all() or motion < 0.05:
        raise RuntimeError(f"Franka did not respond to control: motion={motion}")

    write_json_atomic(status_path, {
        "status": "complete",
        "verification_mode": "direct_usdz_plus_standard_franka",
        "package_path": str(package_path),
        "robot_asset": (
            assets_root + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        ),
        "robot_path": str(robot_path),
        "robot_num_dof": int(robot.num_dof),
        "commanded_joint_index": 0,
        "joint_position_before": float(before[0]),
        "joint_position_target": target,
        "joint_position_after": float(after[0]),
        "controlled_motion": motion,
        "scene_rigid_body_count": inventory["counts"]["rigid_bodies"],
        "scene_articulation_count": inventory["counts"]["articulation_roots"],
        "scene_contact_ready": bool(
            not inventory["missing_collision_roots"]
            and not inventory["articulations_without_colliders"]
            and not inventory["implicit_dynamic_mesh_colliders"]
            and not any(
                item["active_position_drive"]
                for item in inventory["joint_drives"]
            )
        ),
        "package_was_saved": False,
    })
    log(f"Franka compatibility validation complete: {package_path}")


async def capture_robot_contact_frames(config):
    from isaacsim.storage.native import get_assets_root_path

    package_path = Path(config["package_path"]).resolve()
    frames_dir = Path(config["frames_dir"]).resolve()
    status_path = Path(config["status_path"]).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    if not context.open_stage(str(package_path)):
        raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    app = omni.kit.app.get_app()
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()
    stage = context.get_stage()
    inventory = physics_inventory(stage)

    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    candidates = []
    for path in inventory["visual_rigid_bodies"]:
        bounds = bounds_cache.ComputeWorldBound(
            stage.GetPrimAtPath(path)
        ).ComputeAlignedRange()
        size = np.asarray(bounds.GetMax() - bounds.GetMin(), dtype=float)
        if np.isfinite(size).all() and min(size) > 0.0:
            candidates.append((float(np.prod(size)), path, bounds, size))
    if not candidates:
        raise RuntimeError("robot contact video found no dynamic rigid target")
    _volume, target_path, target_bounds, target_size = max(
        candidates,
        key=lambda item: (
            float(item[3][2]) / max(float(item[3][0]), float(item[3][1])),
            -item[0],
        ),
    )
    target_center = 0.5 * np.asarray(
        target_bounds.GetMin() + target_bounds.GetMax(), dtype=float,
    )

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac asset root is unavailable")
    robot_path = Sdf.Path("/World/ContactFranka")
    pusher_path = Sdf.Path("/World/ContactFingerProxy")
    pusher_radius = float(np.clip(0.3 * target_size[2], 0.015, 0.03))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_prim = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
        robot_prim.GetReferences().AddReference(
            assets_root + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        )
        robot_xform = UsdGeom.Xformable(robot_prim)
        translate_op = next(
            (
                op for op in robot_xform.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        (translate_op or robot_xform.AddTranslateOp()).Set(Gf.Vec3d(3.0, 0.0, 0.0))
        pusher_prim = UsdGeom.Sphere.Define(stage, pusher_path).GetPrim()
        UsdGeom.Sphere(pusher_prim).CreateRadiusAttr().Set(pusher_radius)
        UsdGeom.Imageable(pusher_prim).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible
        )
        UsdGeom.Xformable(pusher_prim).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, -10.0)
        )
        UsdPhysics.CollisionAPI.Apply(pusher_prim)
        pusher_body = UsdPhysics.RigidBodyAPI.Apply(pusher_prim)
        pusher_body.CreateRigidBodyEnabledAttr().Set(True)
        pusher_body.CreateKinematicEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(pusher_prim).CreateMassAttr().Set(50.0)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            pusher_prim
        ).CreateDisableGravityAttr().Set(True)
        for path in inventory["visual_rigid_bodies"]:
            if path != target_path:
                UsdPhysics.RigidBodyAPI(
                    stage.GetPrimAtPath(path)
                ).CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(target_path)
        ).CreateKinematicEnabledAttr().Set(False)
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    settings = carb.settings.get_settings()
    settings.set_bool("physics/updateToUsd", True)
    settings.set_bool("/rtx/post/histogram/enabled", False)
    settings.set_int("/rtx/post/aa/autoExposureMode", 2)
    settings.set_float("/rtx/post/aa/exposure", 0.6)
    settings.set_string("/rtx/rendermode", "RaytracedLighting")
    camera_path = Sdf.Path(config.get("camera_path", CAMERA_PATH))
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"camera not found: {camera_path}")
    camera = UsdGeom.Camera(camera_prim)
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    capture_resolution = (
        int(config["width"]),
        round(int(config["width"]) * vertical_aperture / horizontal_aperture),
    )
    output_resolution = (int(config["width"]), int(config["height"]))
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport not available")
    viewport.camera_path = camera_path
    viewport.resolution = capture_resolution
    viewport.resolution_scale = 1.0
    settings.set_bool("/app/viewport/grid/enabled", False)

    timeline.play()
    for _ in range(4):
        await app.next_update_async()
    robot = SingleArticulation(str(robot_path), name="contact_franka")
    target = SingleRigidPrim(target_path, name="contact_target")
    pusher = SingleRigidPrim(str(pusher_path), name="contact_finger_proxy")
    robot.initialize()
    target.initialize()
    pusher.initialize()
    if robot.num_dof != 9:
        raise RuntimeError(f"unexpected Franka DOF count: {robot.num_dof}")
    initial_joints = np.array(
        [0.0, -0.6, 0.0, -2.0, 0.0, 1.5, 0.8, 0.04, 0.04],
        dtype=np.float32,
    )
    simulation = omni.physx.get_physx_simulation_interface()
    simulation_time = 0.0

    def step_physics():
        nonlocal simulation_time
        simulation.simulate(1.0 / 60.0, simulation_time)
        simulation.fetch_results()
        simulation_time += 1.0 / 60.0

    def finger_collision_bounds():
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        ranges = [
            cache.ComputeWorldBound(prim).ComputeAlignedRange()
            for prim in Usd.PrimRange(robot_prim)
            if "rightfinger" in str(prim.GetPath()).lower()
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not ranges:
            raise RuntimeError("Franka right finger has no collision geometry")
        minimum = np.min(
            [np.asarray(item.GetMin(), dtype=float) for item in ranges], axis=0,
        )
        maximum = np.max(
            [np.asarray(item.GetMax(), dtype=float) for item in ranges], axis=0,
        )
        return 0.5 * (minimum + maximum), maximum - minimum

    robot.set_joint_positions(initial_joints)
    robot.set_joint_velocities(np.zeros(9, dtype=np.float32))
    for _ in range(4):
        step_physics()
    finger_start, finger_size = finger_collision_bounds()
    sweep_radians = 0.6
    endpoint_joints = initial_joints.copy()
    endpoint_joints[0] += sweep_radians
    robot.set_joint_positions(endpoint_joints)
    for _ in range(4):
        step_physics()
    finger_end, _finger_end_size = finger_collision_bounds()
    raw_motion = finger_end - finger_start
    vertical_sweep = float(raw_motion[2])
    motion = raw_motion.copy()
    motion[2] = 0.0
    motion_length = float(np.linalg.norm(motion))
    if motion_length < 0.05:
        raise RuntimeError(f"Franka finger sweep is too small: {motion_length}")
    direction = motion / motion_length
    target_support = float(np.dot(np.abs(direction), 0.5 * target_size))
    desired_finger_start = target_center - direction * (
        target_support + pusher_radius + 0.01
    )
    desired_finger_start[2] = target_center[2] - vertical_sweep
    root_position, root_orientation = robot.get_world_pose()
    robot.set_world_pose(
        np.asarray(root_position, dtype=float) + desired_finger_start - finger_start,
        root_orientation,
    )
    robot.set_joint_positions(initial_joints)
    robot.set_joint_velocities(np.zeros(9, dtype=np.float32))
    pusher.set_world_pose(
        desired_finger_start,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    pusher.set_linear_velocity(np.zeros(3, dtype=np.float32))
    pusher.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for _ in range(12):
        step_physics()
    settled_target = np.asarray(target.get_world_pose()[0], dtype=float)
    aligned_finger, _aligned_finger_size = finger_collision_bounds()
    log(
        f"robot contact target={target_path} center={target_center.tolist()} "
        f"finger_start={aligned_finger.tolist()} direction={direction.tolist()} "
        f"sweep={motion_length:.6g}"
    )

    settings.set_int("/rtx/pathtracing/spp", int(config["pathtracing_spp"]))
    settings.set_int("/rtx/pathtracing/totalSpp", int(config["pathtracing_spp"]))
    settings.set_int("/rtx/pathtracing/clampSpp", 0)
    settings.set_bool(
        "/rtx/pathtracing/optixDenoiser/enabled",
        bool(config["pathtracing_denoiser"]),
    )
    viewport.set_hd_engine("rtx", "PathTracing")
    settings.set_string("/rtx/rendermode", "PathTracing")
    timeline.pause()
    for _ in range(2):
        await app.next_update_async()
    warmup_path = frames_dir / "warmup.png"
    warmup = await capture_viewport_to_file(
        viewport, file_path=str(warmup_path), is_hdr=False,
    ).wait_for_result(completion_frames=0)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    if not warmup or not warmup_path.is_file():
        raise RuntimeError("failed to prime robot contact video capture")
    warmup_path.unlink()

    controller = robot.get_articulation_controller()
    frame_count = int(config["frame_count"])
    physics_substeps = int(config.get("physics_substeps_per_frame", 1))
    if physics_substeps < 1:
        raise RuntimeError("physics_substeps_per_frame must be positive")
    for frame in range(frame_count):
        phase = min(frame / max(round(0.75 * (frame_count - 1)), 1), 1.0)
        blend = 0.5 * (1.0 - math.cos(math.pi * phase))
        joint_targets = initial_joints.copy()
        joint_targets[0] += sweep_radians * blend
        pusher_goal = desired_finger_start + raw_motion * blend
        for substep in range(physics_substeps):
            controller.apply_action(ArticulationAction(
                joint_positions=joint_targets,
            ))
            target.set_linear_velocity(
                np.asarray(target.get_linear_velocity(), dtype=float)
            )
            pusher_position = np.asarray(
                pusher.get_world_pose()[0], dtype=float,
            )
            remaining_time = (physics_substeps - substep) / 60.0
            pusher.set_linear_velocity(
                (pusher_goal - pusher_position) / remaining_time
            )
            pusher.set_angular_velocity(np.zeros(3, dtype=np.float32))
            step_physics()
        await app.next_update_async()
        frame_path = frames_dir / f"frame_{frame:06d}.png"
        capture_path = (
            frame_path
            if capture_resolution == output_resolution else
            frames_dir / f"capture_{frame:06d}.png"
        )
        result = await capture_viewport_to_file(
            viewport, file_path=str(capture_path), is_hdr=False,
        ).wait_for_result(completion_frames=0)
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not result or not capture_path.is_file():
            raise RuntimeError(f"failed to capture robot contact frame {frame}")
        if capture_path != frame_path:
            with Image.open(capture_path) as image:
                image.resize(output_resolution, Image.Resampling.LANCZOS).save(
                    frame_path
                )
            capture_path.unlink()
        if frame == 0 or frame == frame_count - 1 or (frame + 1) % 24 == 0:
            log(f"captured robot contact frame {frame + 1}/{frame_count}")

    final_target = np.asarray(target.get_world_pose()[0], dtype=float)
    target_displacement = float(np.linalg.norm(final_target - settled_target))
    final_finger, _final_finger_size = finger_collision_bounds()
    final_pusher = np.asarray(pusher.get_world_pose()[0], dtype=float)
    final_joints = np.asarray(robot.get_joint_positions(), dtype=float)
    log(
        f"robot contact final_target={final_target.tolist()} "
        f"final_finger={final_finger.tolist()} "
        f"final_pusher={final_pusher.tolist()} "
        f"joint0={final_joints[0]:.6g} displacement={target_displacement:.6g}"
    )
    if not np.isfinite(final_target).all() or target_displacement < 0.01:
        raise RuntimeError(
            f"Franka contact did not move target: displacement={target_displacement}"
        )
    write_json_atomic(status_path, {
        "status": "complete",
        "verification_mode": "franka_finger_physical_contact_video",
        "package_path": str(package_path),
        "robot_path": str(robot_path),
        "finger_collision_proxy_path": str(pusher_path),
        "finger_collision_proxy_radius_m": pusher_radius,
        "target_path": target_path,
        "target_displacement_m": target_displacement,
        "frame_count": frame_count,
        "physics_substeps_per_frame": physics_substeps,
        "fps": int(config["fps"]),
        "width": output_resolution[0],
        "height": output_resolution[1],
        "package_was_saved": False,
    })
    log(
        f"Franka contact moved {target_path} by {target_displacement:.6g} m"
    )


def reference_cycle_positions(item, phase):
    lower, upper, reference = item["lower"], item["upper"], item["reference"]
    span = upper - lower
    distance = float(phase) * 2.0 * span
    first_turn = upper - reference
    second_turn = first_turn + span
    return np.where(
        distance <= first_turn,
        reference + distance,
        np.where(
            distance <= second_turn,
            upper - (distance - first_turn),
            lower + (distance - second_turn),
        ),
    )


def articulation_root_poses(stage):
    poses = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        transform = Gf.Transform(matrix)
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        poses[str(prim.GetPath())] = (
            np.array(transform.GetTranslation(), dtype=np.float32),
            np.array(
                [
                    quaternion.GetReal(), imaginary[0], imaginary[1],
                    imaginary[2],
                ],
                dtype=np.float32,
            ),
        )
    return poses


def freeze_visual_rigid_bodies(stage):
    root = stage.GetPrimAtPath(BLENDER_SCENE_PATH)
    frozen = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            frozen.append(str(prim.GetPath()))
    return frozen


def zero_saved_rigid_body_velocities(stage_path):
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not reopen saved USD: {stage_path}")
    rigid_body_paths = [
        prim.GetPath() for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    joint_velocities = [
        (prim.GetPath(), attribute.GetName(), attribute.GetTypeName())
        for prim in stage.Traverse()
        for attribute in prim.GetAttributes()
        if attribute.GetName().startswith("state:")
        and attribute.GetName().endswith("physics:velocity")
    ]
    layer = Sdf.Layer.OpenAsAnonymous(str(stage_path))
    if layer is None:
        raise RuntimeError(f"could not copy saved USD: {stage_path}")
    for path in rigid_body_paths:
        prim = Sdf.CreatePrimInLayer(layer, path)
        for name in ("physics:velocity", "physics:angularVelocity"):
            attribute = prim.attributes.get(name) or Sdf.AttributeSpec(
                prim, name, Sdf.ValueTypeNames.Vector3f,
                Sdf.VariabilityVarying,
            )
            attribute.default = Gf.Vec3f(0.0)
    for path, name, type_name in joint_velocities:
        prim = Sdf.CreatePrimInLayer(layer, path)
        attribute = prim.attributes.get(name) or Sdf.AttributeSpec(
            prim, name, type_name, Sdf.VariabilityVarying,
        )
        attribute.default = 0.0
    if not layer.Export(str(stage_path)):
        raise RuntimeError(f"could not save zero velocities to {stage_path}")
    return len(rigid_body_paths)


def _collision_object_id(path, expected_object_ids):
    parts = tuple(part for part in str(path).split("/") if part)
    if expected_object_ids:
        return next((part for part in parts if part in expected_object_ids), None)
    if len(parts) >= 3 and parts[:2] == ("World", "BlenderScene"):
        return parts[2]
    return parts[1] if len(parts) >= 2 and parts[0] == "World" else None


def evaluate_physx_collisions(stage, expected_object_ids=None):
    excluded = {"table_0", "StudioGround"}
    expected = set(expected_object_ids or ()) - excluded
    colliders = []
    discovered = set()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsActive():
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        object_id = _collision_object_id(prim.GetPath(), expected)
        if object_id is None or object_id in excluded:
            continue
        shapes = (
            [prim] if prim.IsA(UsdGeom.Gprim) else
            [
                child for child in Usd.PrimRange(
                    prim, Usd.TraverseInstanceProxies(),
                )
                if child.IsA(UsdGeom.Gprim)
            ]
        )
        colliders.extend((object_id, shape.GetPath()) for shape in shapes)
        if not shapes:
            continue
        discovered.add(object_id)

    object_ids = expected or discovered
    missing = object_ids - discovered
    if missing:
        raise RuntimeError(
            "objects without active PhysX collision geometry: "
            + ", ".join(sorted(missing))
        )
    if not object_ids:
        raise RuntimeError("no non-table PhysX collision geometry found")

    pairs = set()
    reported_hits = 0
    interface = omni.physx.get_physx_scene_query_interface()
    for object_id, collider_path in colliders:
        encoded = PhysicsSchemaTools.encodeSdfPath(collider_path)

        def record_hit(hit, source=object_id):
            other = _collision_object_id(hit.collision or hit.rigid_body, object_ids)
            if other is not None and other != source:
                pairs.add(tuple(sorted((source, other))))
            return True

        reported_hits += interface.overlap_shape(
            encoded[0], encoded[1], record_hit, False,
        )

    collision_pairs = [list(pair) for pair in sorted(pairs)]
    object_count = len(object_ids)
    return {
        "status": "complete",
        "collision_status": "ok",
        "version": COLLISION_EVALUATION_VERSION,
        "engine": "PhysX",
        "method": "scene_query_overlap_shape",
        "evaluation_pose": "initial_authored_pose_before_settling",
        "collider": "simulation_collision_geometry",
        "excluded_object_ids": sorted(excluded),
        "object_ids": sorted(object_ids),
        "object_count": object_count,
        "object_pair_count": object_count * (object_count - 1) // 2,
        "queried_collider_count": len(colliders),
        "reported_overlap_hit_count": reported_hits,
        "collision_pairs": collision_pairs,
        "colliding_pair_count": len(collision_pairs),
        "has_collision": bool(collision_pairs),
    }


async def evaluate_packaged_scene_collisions(config):
    package_path = Path(config["package_path"]).resolve()
    status_path = Path(config["status_path"]).resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    package_stat = package_path.stat()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    carb.settings.get_settings().set_bool("physics/updateToUsd", False)

    context = omni.usd.get_context()
    if not context.open_stage(str(package_path)):
        raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    app = omni.kit.app.get_app()
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Isaac returned no stage for USDZ: {package_path}")

    freeze_visual_rigid_bodies(stage)
    timeline.play()
    await app.next_update_async()
    await app.next_update_async()
    timeline.pause()
    result = evaluate_physx_collisions(
        stage, config.get("collision_object_ids"),
    )
    result.update({
        "source_package": str(package_path),
        "source_package_size": package_stat.st_size,
        "source_package_mtime_ns": package_stat.st_mtime_ns,
    })
    write_json_atomic(status_path, result)
    log(
        f"PhysX collision evaluation complete: "
        f"{result['colliding_pair_count']}/{result['object_pair_count']} pairs"
    )


def find_bounded_articulations(stage, scene_ranges, initial_root_poses=None):
    articulations = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        prim_path = str(prim.GetPath())
        parts = prim_path.split("/")
        object_id = parts[2] if len(parts) > 2 and parts[1] == "World" else None
        if object_id is None:
            continue
        handle = SingleArticulation(str(prim.GetPath()), name=f"joint_demo_{len(articulations)}")
        handle.initialize()
        if handle.num_dof == 0:
            log(f"skipping articulation without movable DOFs: {prim_path}")
            continue
        properties = handle.dof_properties
        joint_types = {
            joint_prim.GetName(): (
                "prismatic"
                if joint_prim.IsA(UsdPhysics.PrismaticJoint)
                else "revolute"
            )
            for joint_prim in Usd.PrimRange(
                stage.GetPrimAtPath(f"/World/{object_id}")
            )
            if joint_prim.IsA(UsdPhysics.PrismaticJoint)
            or joint_prim.IsA(UsdPhysics.RevoluteJoint)
        }
        indices = []
        lower = []
        upper = []
        reference = []
        joints = []
        for index, name in enumerate(handle.dof_names):
            prop = properties[index]
            urdf_lo = float(prop["lower"])
            urdf_hi = float(prop["upper"])
            scene_limit = scene_ranges.get((object_id, name))
            lo = (
                max(urdf_lo, float(scene_limit[0]))
                if scene_limit and scene_limit[0] is not None else urdf_lo
            )
            hi = (
                min(urdf_hi, float(scene_limit[1]))
                if scene_limit and scene_limit[1] is not None else urdf_hi
            )
            if not bool(prop["hasLimits"]) or not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                continue
            joint_type = joint_types.get(name)
            if joint_type is None:
                raise RuntimeError(f"USD joint prim not found for DOF {name}")
            joint = {
                "name": name,
                "type": joint_type,
                "lower": lo,
                "upper": hi,
                "reference": (
                    min(max(float(scene_limit[2]), lo), hi)
                    if scene_limit else 0.5 * (lo + hi)
                ),
                "urdf_lower": urdf_lo,
                "urdf_upper": urdf_hi,
                "range_source": (
                    "sim_asset_manifest.scene_joint_ranges.video_range"
                    if scene_limit and scene_limit[0] is not None
                    else "urdf_joint_limits_with_scene_rest"
                    if scene_limit else "urdf_joint_limits"
                ),
                "unit": "rad" if joint_type == "revolute" else "m",
            }
            if joint_type == "revolute":
                joint["lower_deg"] = math.degrees(lo)
                joint["upper_deg"] = math.degrees(hi)
            indices.append(index)
            lower.append(lo)
            upper.append(hi)
            reference.append(joint["reference"])
            joints.append(joint)
        if joints:
            initial_position, initial_orientation = (initial_root_poses or {}).get(
                prim_path, (None, None)
            )
            root_drift = None
            if initial_position is not None:
                root_drift = float(
                    np.linalg.norm(handle.get_world_pose()[0] - initial_position)
                )
            articulations.append(
                {
                    "handle": handle,
                    "indices": np.array(indices, dtype=np.int32),
                    "lower": np.array(lower, dtype=np.float32),
                    "upper": np.array(upper, dtype=np.float32),
                    "reference": np.array(reference, dtype=np.float32),
                    "initial_position": initial_position,
                    "initial_orientation": initial_orientation,
                    "initialization_root_drift_m": root_drift,
                    "metadata": {
                        "path": str(prim.GetPath()),
                        "object_id": object_id,
                        "joints": joints,
                    },
                }
            )
    return articulations


def author_articulation_entry_pose(stage, item, positions=None):
    positions = item["reference"] if positions is None else positions
    articulation_view = item["handle"]._articulation_view
    raw_transforms = articulation_view._physics_view.get_link_transforms()
    if hasattr(raw_transforms, "numpy"):
        raw_transforms = raw_transforms.numpy()
    link_transforms = np.asarray(raw_transforms).reshape(
        articulation_view.count, -1, 7,
    )[0]
    object_root = stage.GetPrimAtPath(
        f'/World/{item["metadata"]["object_id"]}'
    )
    links_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(object_root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for name, transform in zip(articulation_view.body_names, link_transforms):
        prim = links_by_name.get(name)
        if prim is None:
            raise RuntimeError(f"articulation link not found in USD: {name}")
        quaternion = Gf.Quatd(
            float(transform[6]),
            Gf.Vec3d(*(float(value) for value in transform[3:6])),
        ).GetNormalized()
        world_matrix = Gf.Matrix4d(1.0)
        world_matrix.SetRotate(quaternion)
        world_matrix.SetTranslateOnly(
            Gf.Vec3d(*(float(value) for value in transform[:3]))
        )
        parent_matrix = UsdGeom.Xformable(
            prim.GetParent()
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_matrix = world_matrix * parent_matrix.GetInverse()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_matrix)
        stale_orient = prim.GetAttribute("xformOp:orient")
        if stale_orient:
            stale_orient.Block()

    joints_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(object_root)
        if prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
    }
    for joint, reference in zip(item["metadata"]["joints"], positions):
        prim = joints_by_name.get(joint["name"])
        if prim is None:
            raise RuntimeError(f'articulation joint not found in USD: {joint["name"]}')
        token = "angular" if joint["type"] == "revolute" else "linear"
        position = float(reference)
        if joint["type"] == "revolute":
            position = math.degrees(position)
        prim.GetAttribute(f"state:{token}:physics:position").Set(position)
        prim.GetAttribute(f"state:{token}:physics:velocity").Set(0.0)


async def capture_sampled_frames(config):
    sample_frames = list(config["frame_indices"])
    package_path = Path(config["asset_package_path"]).resolve()
    retain_package = os.environ.get("TABLETOP_PUBLISH_USDZ", "1").lower() in {
        "1", "true", "yes", "on",
    }

    export_config = {
        **config,
        "capture_mode": "export_only",
        "single_frame": True,
        "frame_count": 1,
        "sample_frames": [],
        "frame_indices": [0],
        "cycle_blend": [1.0],
        "export_only": True,
    }
    previous_publish = os.environ.get("TABLETOP_PUBLISH_USDZ")
    os.environ["TABLETOP_PUBLISH_USDZ"] = "1"
    try:
        await capture_video_frames(export_config)
    finally:
        if previous_publish is None:
            os.environ.pop("TABLETOP_PUBLISH_USDZ", None)
        else:
            os.environ["TABLETOP_PUBLISH_USDZ"] = previous_publish
    if not package_path.is_file():
        raise FileNotFoundError(f"exported asset package is missing: {package_path}")

    frames_dir = Path(config["frames_dir"])
    rest_status_path = Path(config["status_path"]).with_name("rest_import_status.json")
    omni.timeline.get_timeline_interface().stop()
    await omni.kit.app.get_app().next_update_async()
    await capture_packaged_scene_without_simulation({
        **config,
        "package_path": str(package_path),
        "output_path": str(frames_dir / "frame_000000.png"),
        "status_path": str(rest_status_path),
    })
    rest_status = json.loads(rest_status_path.read_text(encoding="utf-8"))

    motion_frames = sample_frames[1:]
    if motion_frames:
        await capture_video_frames({
            **config,
            "capture_mode": "sampled_motion",
            "package_path": str(package_path),
            "sample_frames": [],
            "frame_indices": motion_frames,
            "cycle_blend": [
                0.5 * (1.0 - math.cos(
                    2.0 * math.pi * frame / (config["frame_count"] - 1)
                ))
                for frame in motion_frames
            ],
        })
        status = json.loads(Path(config["status_path"]).read_text(encoding="utf-8"))
    else:
        status = {"status": "complete"}
    status.update({
        "capture_mode": "sampled_frames",
        "source_mode": "direct_usdz_import_after_pipeline_export",
        "asset_package_published": retain_package,
        "package_was_saved": False,
        "frame_count": config["frame_count"],
        "rendered_frame_count": len(sample_frames),
        "rendered_frame_indices": sample_frames,
        "rest_frame": {
            "frame_index": 0,
            "verification_mode": rest_status["verification_mode"],
            "timeline_played": rest_status["timeline_played"],
            "pose_unchanged": rest_status["pose_unchanged"],
            "root_layer_dirty_after": rest_status["root_layer_dirty_after"],
        },
    })
    write_json_atomic(Path(config["status_path"]), status)
    log(f"captured timeline samples {sample_frames} from the directly opened USDZ")


async def capture_video_frames(config):
    project_root = Path(config["project_root"])
    frames_dir = Path(config["frames_dir"])
    status_path = Path(config["status_path"])
    package_path = (
        Path(config["package_path"]).resolve()
        if config.get("package_path") else None
    )

    settings = carb.settings.get_settings()
    settings.set_bool("physics/updateToUsd", package_path is None)
    settings.set_bool("/physics/saveCookedData", package_path is None)
    settings.set_bool("/persistent/physics/useLocalMeshCache", True)
    settings.set_bool("/physics/cooking/ujitsoCollisionCooking", True)
    settings.set_bool("/rtx/post/histogram/enabled", False)
    settings.set_int("/rtx/post/aa/autoExposureMode", 2)
    settings.set_float("/rtx/post/aa/exposure", 0.6)
    settings.set_string("/rtx/rendermode", "RaytracedLighting")

    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    publish_usdz = False
    if package_path is not None:
        log(f"opening packaged tabletop scene {package_path}")
        if not context.open_stage(str(package_path)):
            raise RuntimeError(f"Isaac could not open USDZ directly: {package_path}")
    else:
        log("loading current tabletop scene")
        publish_usdz = os.environ.get("TABLETOP_PUBLISH_USDZ", "1").lower() in {
            "1", "true", "yes", "on",
        }
        previous_publish = os.environ.get("TABLETOP_PUBLISH_USDZ")
        os.environ["TABLETOP_PUBLISH_USDZ"] = "0"
        try:
            loader = runpy.run_path(
                str(Path(__file__).with_name("isaac_scene.py")),
                run_name="__main__",
            )
        finally:
            if previous_publish is None:
                os.environ.pop("TABLETOP_PUBLISH_USDZ", None)
            else:
                os.environ["TABLETOP_PUBLISH_USDZ"] = previous_publish
    while context.get_stage_loading_status()[2] > 0:
        await app.next_update_async()

    stage = context.get_stage()
    if package_path is None:
        stage.SetEditTarget(stage.GetRootLayer())
    camera_prim = stage.GetPrimAtPath(Sdf.Path(CAMERA_PATH))
    if not camera_prim:
        raise RuntimeError(f"camera not found: {CAMERA_PATH}")
    camera = UsdGeom.Camera(camera_prim)
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    capture_height = round(
        config["width"] * vertical_aperture / horizontal_aperture
    )
    capture_resolution = (config["width"], capture_height)
    output_resolution = (config["width"], config["height"])

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport not available")
    viewport.camera_path = Sdf.Path(CAMERA_PATH)
    viewport.resolution = capture_resolution
    viewport.resolution_scale = 1.0
    settings.set_bool("/app/viewport/grid/enabled", False)
    settings.set_bool(f"/persistent/app/viewport/{viewport.id}/guide/grid/visible", False)
    settings.set_bool(f"/persistent/app/viewport/{viewport.id}/guide/axis/visible", False)
    settings.set_bool(f"/persistent/app/viewport/{viewport.id}/guide/selection/visible", False)
    if tuple(int(value) for value in viewport.resolution) != capture_resolution:
        raise RuntimeError(f"viewport resolution did not apply: {viewport.resolution}")

    single_frame = bool(config.get("single_frame"))
    initial_root_poses = articulation_root_poses(stage)
    frozen_rigid_bodies = freeze_visual_rigid_bodies(stage)
    if frozen_rigid_bodies:
        log(
            f"locked {len(frozen_rigid_bodies)} visual rigid bodies at their "
            "authored scene-entry poses"
        )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await app.next_update_async()
    await app.next_update_async()
    scene_ranges = {
        (item["object_id"], item["joint_name"]): (
            item["lower"], item["upper"], item["reference"],
        )
        for item in config["scene_joint_ranges"]
    }
    articulations = find_bounded_articulations(
        stage, scene_ranges, initial_root_poses,
    )
    joint_count = sum(len(item["metadata"]["joints"]) for item in articulations)
    log(
        f"camera={CAMERA_PATH} capture={capture_resolution[0]}x{capture_resolution[1]} "
        f"output={output_resolution[0]}x{output_resolution[1]} "
        f"articulations={len(articulations)} bounded_joints={joint_count}"
    )
    if not articulations:
        log("no bounded articulations; capturing rigid-body simulation")

    def lock_articulation_roots():
        for item in articulations:
            if item["initial_position"] is None:
                continue
            item["handle"].set_world_pose(
                item["initial_position"], item["initial_orientation"],
            )
            item["handle"].set_linear_velocity(
                np.zeros(3, dtype=np.float32)
            )
            item["handle"].set_angular_velocity(
                np.zeros(3, dtype=np.float32)
            )

    def apply_reference_pose(lock_root=False):
        for item in articulations:
            positions = item["reference"]
            item["handle"].set_joint_positions(
                positions, joint_indices=item["indices"],
            )
            item["handle"].set_joint_velocities(
                np.zeros_like(positions), joint_indices=item["indices"],
            )
        if lock_root:
            lock_articulation_roots()

    def current_root_drift():
        return max(
            (
                float(np.linalg.norm(
                    item["handle"].get_world_pose()[0]
                    - item["initial_position"]
                ))
                for item in articulations
                if item["initial_position"] is not None
            ),
            default=0.0,
        )

    def validate_reference_pose():
        for item in articulations:
            actual = item["handle"].get_joint_positions(
                joint_indices=item["indices"]
            )
            error = float(np.max(np.abs(actual - item["reference"])))
            if error > 5.0e-3:
                raise RuntimeError(
                    f'{item["metadata"]["object_id"]} scene rest pose did not apply: '
                    f"max joint error={error:.6g}"
                )
            if item["initial_position"] is not None:
                actual_position = item["handle"].get_world_pose()[0]
                root_error = float(
                    np.linalg.norm(actual_position - item["initial_position"])
                )
                if root_error > 5.0e-4:
                    raise RuntimeError(
                        f'{item["metadata"]["object_id"]} scene root pose did not apply: '
                        f"position error={root_error:.6g}m"
                    )

    apply_reference_pose(lock_root=True)
    timeline.pause()
    await app.next_update_async()
    apply_reference_pose(lock_root=True)
    await app.next_update_async()
    validate_reference_pose()
    settings.set_bool("physics/updateToUsd", False)
    target_layer = (
        stage.GetSessionLayer() if package_path is not None
        else stage.GetRootLayer()
    )
    with Usd.EditContext(stage, target_layer):
        for item in articulations:
            author_articulation_entry_pose(stage, item)
    collision_evaluation = (
        evaluate_physx_collisions(stage, config.get("collision_object_ids"))
        if config.get("evaluate_collisions") else None
    )
    if package_path is None:
        cooking = cooked_collision_inventory(stage)
        if cooking["missing"] or not cooking["cooked"]:
            raise RuntimeError(
                "collision cooking was not embedded for: "
                + ", ".join(
                    item["path"] for item in (
                        cooking["missing"] or [{"path": "all mesh colliders"}]
                    )
                )
            )
        log(
            f"embedded cooked data for {len(cooking['cooked'])} mesh colliders"
        )
        settings.set_bool("physics/updateToUsd", True)
        loader["record_stage_defaults"](stage)
        stage.GetRootLayer().Save()
        zeroed_rigid_bodies = zero_saved_rigid_body_velocities(
            loader["OUT_STAGE"]
        )
        log(
            "saved initialized scene-entry pose with zero velocity for "
            f"{zeroed_rigid_bodies} rigid bodies to the final USD"
        )
        if publish_usdz:
            loader["publish_asset_package"]()
    else:
        log("kept the directly opened USDZ read-only")
    if config.get("export_only"):
        log("asset package export complete; skipped rendering")
        return
    if single_frame:
        log("single-frame preview uses assembled scene rest pose")
    else:
        log("joint video keeps the scene paused and locks every articulation root")

    realtime_warmup_path = frames_dir / "realtime_warmup.png"
    realtime_warmup_result = await capture_viewport_to_file(
        viewport, file_path=str(realtime_warmup_path), is_hdr=False
    ).wait_for_result(completion_frames=0)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    if not realtime_warmup_result or not realtime_warmup_path.is_file():
        raise RuntimeError("failed to render the loaded scene before Path Tracing")
    realtime_warmup_path.unlink()

    settings.set_int("/rtx/pathtracing/spp", int(config["pathtracing_spp"]))
    settings.set_int("/rtx/pathtracing/totalSpp", int(config["pathtracing_spp"]))
    settings.set_int("/rtx/pathtracing/clampSpp", 0)
    settings.set_bool(
        "/rtx/pathtracing/optixDenoiser/enabled",
        bool(config["pathtracing_denoiser"]),
    )
    viewport.set_hd_engine("rtx", "PathTracing")
    settings.set_string("/rtx/rendermode", "PathTracing")
    await app.next_update_async()
    await app.next_update_async()

    warmup_path = frames_dir / "warmup_capture.png"
    warmup_result = await capture_viewport_to_file(
        viewport, file_path=str(warmup_path), is_hdr=False
    ).wait_for_result(completion_frames=0)
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    if not warmup_result or not warmup_path.is_file():
        raise RuntimeError("failed to prime viewport capture")
    warmup_path.unlink()

    max_capture_root_drift = 0.0
    max_joint_tracking_error = 0.0
    frame_indices = config.get("frame_indices") or list(range(config["frame_count"]))
    for captured, (frame, blend) in enumerate(zip(
        frame_indices, config["cycle_blend"], strict=True,
    )):
        if single_frame:
            apply_reference_pose(lock_root=True)
        else:
            frame_positions = []
            for item in articulations:
                phase = frame / (config["frame_count"] - 1)
                positions = reference_cycle_positions(item, phase)
                item["handle"].set_joint_positions(positions, joint_indices=item["indices"])
                item["handle"].set_joint_velocities(np.zeros_like(positions), joint_indices=item["indices"])
                frame_positions.append(positions)
            lock_articulation_roots()
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for item, positions in zip(articulations, frame_positions):
                    author_articulation_entry_pose(stage, item, positions)

        await app.next_update_async()
        await app.next_update_async()
        lock_articulation_roots()
        if not single_frame:
            for item, positions in zip(articulations, frame_positions):
                actual = item["handle"].get_joint_positions(
                    joint_indices=item["indices"]
                )
                tracking_error = float(np.max(np.abs(actual - positions)))
                max_joint_tracking_error = max(
                    max_joint_tracking_error, tracking_error,
                )
                if tracking_error > 5.0e-3:
                    raise RuntimeError(
                        f'{item["metadata"]["object_id"]} joint pose did not apply: '
                        f"max error={tracking_error:.6g}"
                    )
        root_drift = current_root_drift()
        max_capture_root_drift = max(max_capture_root_drift, root_drift)
        if root_drift > 5.0e-4:
            raise RuntimeError(
                f"articulation root moved during capture: {root_drift:.6g}m"
            )
        if captured == 0 and config.get("capture_mode") == "sampled_motion":
            reset_setting = (
                "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges"
            )
            previous_reset_setting = settings.get_as_bool(reset_setting)
            counter = settings.get_as_int("/rtx/externalFrameCounter")
            full_spp = int(config["pathtracing_spp"])
            warmup_spp = min(8, full_spp)
            settings.set_bool(reset_setting, True)
            try:
                settings.set_int("/rtx/pathtracing/spp", warmup_spp)
                settings.set_int("/rtx/pathtracing/totalSpp", warmup_spp)
                settings.set_int("/rtx/externalFrameCounter", counter + 1)
                await app.next_update_async()
                settings.set_int("/rtx/pathtracing/spp", full_spp)
                settings.set_int("/rtx/pathtracing/totalSpp", full_spp)
                settings.set_int("/rtx/externalFrameCounter", counter + 2)
                await app.next_update_async()
            finally:
                settings.set_int("/rtx/pathtracing/spp", full_spp)
                settings.set_int("/rtx/pathtracing/totalSpp", full_spp)
                settings.set_bool(reset_setting, previous_reset_setting)
        frame_path = frames_dir / f"frame_{frame:06d}.png"
        capture_path = (
            frame_path
            if capture_resolution == output_resolution
            else frames_dir / f"capture_{frame:06d}.png"
        )
        result = await capture_viewport_to_file(
            viewport, file_path=str(capture_path), is_hdr=False
        ).wait_for_result(completion_frames=0)
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not result or not capture_path.is_file():
            raise RuntimeError(f"failed to capture {capture_path}")
        if capture_path != frame_path:
            with Image.open(capture_path) as image:
                image.resize(output_resolution, Image.Resampling.LANCZOS).save(
                    frame_path
                )
            capture_path.unlink()
        if captured == 0 or captured == len(frame_indices) - 1 or (captured + 1) % 15 == 0:
            log(
                f"captured timeline frame {frame} "
                f"({captured + 1}/{len(frame_indices)}) range_phase={blend:.3f}"
            )

    if single_frame:
        validate_reference_pose()

    write_json_atomic(
        status_path,
        {
            "status": "complete",
            "capture_mode": (
                "single_frame" if config.get("single_frame") else "video"
            ),
            "renderer": "RTX Path Tracing",
            "samples_per_pixel": int(config["pathtracing_spp"]),
            "denoiser_enabled": bool(config["pathtracing_denoiser"]),
            "camera_fixed": True,
            "articulation_bases_fixed": os.environ.get(
                "TABLETOP_FIX_ARTICULATION_BASE", "0"
            ).lower() in {"1", "true", "yes"},
            "static_pose_locked": True,
            "articulation_roots_locked_during_capture": True,
            "scene_entry_pose_saved": True,
            "asset_package_published": publish_usdz,
            "source_mode": (
                "direct_usdz_import" if package_path is not None
                else "pipeline_scene_build"
            ),
            "package_was_saved": False if package_path is not None else None,
            "locked_visual_rigid_body_count": len(frozen_rigid_bodies),
            "max_articulation_capture_root_drift_m": max_capture_root_drift,
            "max_joint_tracking_error": max_joint_tracking_error,
            "max_articulation_initialization_root_drift_m": max(
                (
                    item["initialization_root_drift_m"] or 0.0
                    for item in articulations
                ),
                default=0.0,
            ),
            "camera_path": CAMERA_PATH,
            "auto_exposure": False,
            "motion": (
                "scene_rest"
                if single_frame else (
                    "reference_upper_lower_reference"
                    if articulations else "rigid_body_physics"
                )
            ),
            "trajectory_sampling": (
                "per_joint_constant_normalized_speed"
                if articulations and not single_frame else None
            ),
            "control_mode": "direct_joint_position" if articulations else "physics_timeline",
            "frame_count": config["frame_count"],
            "rendered_frame_count": len(frame_indices),
            "rendered_frame_indices": frame_indices,
            "articulations": [item["metadata"] for item in articulations],
            "collision_evaluation": collision_evaluation,
            "embedded_cooked_mesh_collider_count": (
                len(cooking["cooked"]) if package_path is None else None
            ),
        },
    )
    log("capture complete")


async def main():
    app = omni.kit.app.get_app()
    return_code = 0
    status_path = None
    try:
        config = json.loads(Path(os.environ["JOINT_VIDEO_CONFIG"]).read_text(encoding="utf-8"))
        status_path = Path(config["status_path"])
        if config.get("capture_mode") == "sampled_frames":
            await capture_sampled_frames(config)
        else:
            await capture_video_frames(config)
    except Exception as error:
        return_code = 1
        if status_path is not None:
            write_json_atomic(
                status_path,
                {"status": "failed", "error": str(error), "traceback": traceback.format_exc()},
            )
        log(f"capture failed: {error}")
        traceback.print_exc()
    finally:
        omni.timeline.get_timeline_interface().stop()
        app.post_quit(return_code)


async def persistent_main():
    app = omni.kit.app.get_app()
    idle_viewport = get_active_viewport()
    if idle_viewport is not None:
        idle_viewport.updates_enabled = False
    socket_path = Path(os.environ["ISAAC_CAPTURE_SERVER_SOCKET"])
    socket_path.unlink(missing_ok=True)
    stop = asyncio.Event()
    job_lock = asyncio.Lock()

    async def handle(reader, writer):
        response = {"ok": False}
        render_request = False
        owns_viewport = False
        try:
            request = json.loads((await reader.readline()).decode("utf-8"))
            render_request = not (
                request.get("ping")
                or request.get("shutdown")
                or request.get("mode") in {
                    "sim_ready_validation", "collision_evaluation",
                }
            )
            if (
                render_request and idle_viewport is not None
                and not job_lock.locked()
            ):
                idle_viewport.updates_enabled = True
                owns_viewport = True
            if request.get("ping"):
                response = {
                    "ok": True,
                    "status": "busy" if job_lock.locked() else "ready",
                    "pid": os.getpid(),
                    "capabilities": ["collision_evaluation"],
                }
            elif request.get("shutdown"):
                response = {"ok": True}
                stop.set()
            elif job_lock.locked():
                response = {
                    "ok": False,
                    "status": "busy",
                    "error": "resident Isaac worker is busy",
                }
            elif request.get("mode") == "direct_package_validation":
                async with job_lock:
                    config = request["config"]
                    status_path = Path(config["status_path"])
                    try:
                        await capture_packaged_scene_without_simulation(config)
                        response = {"ok": True}
                    except Exception as error:
                        write_json_atomic(
                            status_path,
                            {
                                "status": "failed",
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        response = {
                            "ok": False,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        omni.timeline.get_timeline_interface().stop()
                        await app.next_update_async()
                        omni.usd.get_context().new_stage()
                        await app.next_update_async()
            elif request.get("mode") in {
                "sim_ready_validation", "collision_evaluation",
            }:
                async with job_lock:
                    config = request["config"]
                    status_path = Path(config["status_path"])
                    viewport = get_active_viewport()
                    updates_enabled = (
                        viewport.updates_enabled if viewport is not None else None
                    )
                    if viewport is not None:
                        viewport.updates_enabled = False
                    try:
                        if request["mode"] == "collision_evaluation":
                            await evaluate_packaged_scene_collisions(config)
                        else:
                            await validate_packaged_scene_physics(config)
                        response = {"ok": True}
                    except Exception as error:
                        write_json_atomic(
                            status_path,
                            {
                                "status": "failed",
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        response = {
                            "ok": False,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        omni.timeline.get_timeline_interface().stop()
                        await app.next_update_async()
                        omni.usd.get_context().new_stage()
                        await app.next_update_async()
                        if viewport is not None:
                            viewport.updates_enabled = updates_enabled
            elif request.get("mode") in {
                "robot_compatibility_validation", "robot_contact_video",
                "package_joint_video",
            }:
                async with job_lock:
                    config = request["config"]
                    status_path = Path(config["status_path"])
                    try:
                        if request["mode"] == "package_joint_video":
                            await capture_video_frames(config)
                        elif request["mode"] == "robot_contact_video":
                            await capture_robot_contact_frames(config)
                        else:
                            await validate_robot_compatibility(config)
                        response = {"ok": True}
                    except Exception as error:
                        write_json_atomic(
                            status_path,
                            {
                                "status": "failed",
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        response = {
                            "ok": False,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        omni.timeline.get_timeline_interface().stop()
                        await app.next_update_async()
                        omni.usd.get_context().new_stage()
                        await app.next_update_async()
            else:
                async with job_lock:
                    previous = {
                        name: os.environ.get(name)
                        for name in request.get("environment", {})
                    }
                    os.environ.update(request.get("environment", {}))
                    config = json.loads(
                        Path(request["config_path"]).read_text(encoding="utf-8")
                    )
                    status_path = Path(config["status_path"])
                    try:
                        if config.get("capture_mode") == "sampled_frames":
                            await capture_sampled_frames(config)
                        else:
                            await capture_video_frames(config)
                        response = {"ok": True}
                    except Exception as error:
                        write_json_atomic(
                            status_path,
                            {
                                "status": "failed",
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        response = {
                            "ok": False,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        omni.timeline.get_timeline_interface().stop()
                        omni.usd.get_context().new_stage()
                        await app.next_update_async()
                        for name, value in previous.items():
                            if value is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = value
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        if owns_viewport and idle_viewport is not None:
            idle_viewport.updates_enabled = False
        try:
            writer.write(json.dumps(response).encode("utf-8") + b"\n")
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = await asyncio.start_unix_server(handle, path=str(socket_path))
    log(f"persistent server ready: {socket_path}")
    while not stop.is_set():
        await app.next_update_async()
    server.close()
    await server.wait_closed()
    socket_path.unlink(missing_ok=True)
    app.post_quit(0)


asyncio.ensure_future(
    persistent_main()
    if os.environ.get("ISAAC_CAPTURE_SERVER_SOCKET")
    else main()
)
