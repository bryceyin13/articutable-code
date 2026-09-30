#!/usr/bin/env python3
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gram.urdf_assets import (
    bake_urdf_scale,
    read_glb_pbr_material,
    urdf_prismatic_scales,
    use_visual_meshes_for_collisions,
)

try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None


PROJECT_ROOT = Path(os.environ.get("TABLETOP_ROOT", PROJECT_ROOT)).resolve()
EXPORT_ROOT = Path(os.environ.get("TABLETOP_SIM_EXPORT_ROOT", PROJECT_ROOT / "outputs/sim_export")).resolve()
MANIFEST_PATH = Path(os.environ.get("TABLETOP_SIM_MANIFEST_PATH", EXPORT_ROOT / "scene_manifest.json")).resolve()
OUTPUT_ROOT = Path(os.environ.get("TABLETOP_SIM_OUTPUT_ROOT", EXPORT_ROOT)).resolve()
OUT_STAGE = OUTPUT_ROOT / "isaac_loaded_scene.usd"
OUT_PACKAGE = OUTPUT_ROOT / "isaac_asset_package.usdz"
BLENDER_SCENE_PATH = "/World/BlenderScene"
CAMERA_PATH = f"{BLENDER_SCENE_PATH}/front_camera/front_camera"
WHITE_BACKPLATE = (1.0, 1.0, 1.0)
WHITE_GROUND = (0.92, 0.92, 0.92)
LIGHT_TYPES = {
    "CylinderLight", "DiskLight", "DistantLight", "DomeLight", "RectLight",
    "SphereLight",
}
JOINT_DEMO_ENABLED = os.environ.get("TABLETOP_JOINT_DEMO", "0").lower() in {
    "1",
    "true",
    "yes",
}
JOINT_DEMO_SPEED = float(os.environ.get("TABLETOP_JOINT_DEMO_SPEED", "0.8"))
JOINT_DEMO_RANGE_FRACTION = float(
    os.environ.get("TABLETOP_JOINT_DEMO_RANGE_FRACTION", "0.85")
)
JOINT_DEMO_DISABLE_GRAVITY = os.environ.get(
    "TABLETOP_JOINT_DEMO_DISABLE_GRAVITY",
    "1",
).lower() in {"1", "true", "yes"}
SDF_RESOLUTION = int(os.environ.get("TABLETOP_SDF_RESOLUTION", "128"))
SDF_SUBGRID_RESOLUTION = int(
    os.environ.get("TABLETOP_SDF_SUBGRID_RESOLUTION", "6")
)
SDF_TRIANGLE_REDUCTION_FACTOR = float(
    os.environ.get("TABLETOP_SDF_TRIANGLE_REDUCTION_FACTOR", "0.25")
)
DYNAMIC_COLLISION_MODE = os.environ.get(
    "TABLETOP_DYNAMIC_COLLISION_MODE", "convexDecomposition"
)
if DYNAMIC_COLLISION_MODE not in {"sdf", "convexDecomposition"}:
    raise ValueError(
        "TABLETOP_DYNAMIC_COLLISION_MODE must be sdf or convexDecomposition"
    )
TABLE_COLLISION_TARGET_FACES = int(
    os.environ.get("TABLETOP_TABLE_COLLISION_TARGET_FACES", "0")
)
TABLE_COLLISION_MAX_ERROR_M = float(
    os.environ.get("TABLETOP_TABLE_COLLISION_MAX_ERROR_M", "0.002")
)
CONVEX_DECOMPOSITION_MAX_HULLS = 128
CONVEX_DECOMPOSITION_ERROR_PERCENTAGE = 1.0
CONVEX_DECOMPOSITION_VOXEL_RESOLUTION = 1_000_000
CONVEX_DECOMPOSITION_HULL_VERTEX_LIMIT = 64
DYNAMIC_CONTACT_OFFSET_M = 0.002

# Match the publication material profile used by the Stage 7 URDF renders.
STAGE7_ARTICULATION_PBR_TUNING = {
    "adjustable_desk_lamp_0": (0.30, 0.25),
    "scissors_0": (0.45, 0.28),
    "storage_drawer_box_0": (0.38, 0.30),
    "water_bottle_0": (0.82, 0.20),
}


def log(msg):
    print(f"[tabletop-isaac] {msg}", flush=True)


def apply_xform(
    prim, translate=None, yaw_deg=None, scale=None,
    roll_x_deg=None, roll_y_deg=None,
):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if yaw_deg is not None:
        xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(yaw_deg))
    if roll_y_deg is not None:
        xform.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble).Set(
            float(roll_y_deg)
        )
    if roll_x_deg is not None:
        xform.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(
            float(roll_x_deg)
        )
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = (scale, scale, scale)
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))


def configure_rtx_background():
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", True)
    settings.set_string("/rtx/post/backgroundZeroAlpha/backplateTexture", "")
    settings.set_float_array(
        "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor",
        list(WHITE_BACKPLATE),
    )
    settings.set_float(
        "/rtx/post/backgroundZeroAlpha/backplateLuminanceScaleV2",
        1.0,
    )
    log("configured white studio background")


def add_white_studio_ground(stage, table_prim):
    table_bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeWorldBound(table_prim).ComputeAlignedRange()
    table_min_z = float(table_bounds.GetMin()[2])
    if not math.isfinite(table_min_z):
        raise RuntimeError("table_0 has no finite world-space lower bound")

    thickness = 0.02
    ground_size = float(os.environ.get("TABLETOP_STUDIO_GROUND_SIZE", "1000"))
    ground_top_z = table_min_z - 0.002
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/StudioGround"))
    ground.CreateSizeAttr(1.0)
    apply_xform(
        ground.GetPrim(),
        translate=(0.0, 0.0, ground_top_z - thickness / 2),
        scale=(ground_size, ground_size, thickness),
    )

    material = UsdShade.Material.Define(
        stage,
        Sdf.Path("/World/Looks/WhiteGround"),
    )
    shader = UsdShade.Shader.Define(
        stage,
        Sdf.Path("/World/Looks/WhiteGround/PreviewSurface"),
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(*WHITE_GROUND)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)
    log(
        f"placed {ground_size:g}m StudioGround top at z={ground_top_z:.6f}, "
        "below table_0"
    )


def configure_fixed_camera(stage):
    camera_path = Sdf.Path(CAMERA_PATH)
    camera = stage.GetPrimAtPath(camera_path)
    if not camera or not camera.IsA(UsdGeom.Camera):
        raise RuntimeError(f"camera not found: {camera_path}")

    # USD stores these values in tenths of a stage unit. Isaac Sim 4.5's RTX
    # camera reads the numbers using its centimetre-stage convention instead,
    # so convert the Blender camera's physical optics before rendering.
    unit_scale = UsdGeom.GetStageMetersPerUnit(stage) / 0.01
    usd_camera = UsdGeom.Camera(camera)
    optical_attributes = (
        usd_camera.GetFocalLengthAttr(),
        usd_camera.GetHorizontalApertureAttr(),
        usd_camera.GetVerticalApertureAttr(),
        usd_camera.GetHorizontalApertureOffsetAttr(),
        usd_camera.GetVerticalApertureOffsetAttr(),
    )
    original = [float(attribute.Get()) for attribute in optical_attributes]
    for attribute, value in zip(optical_attributes, original):
        attribute.Set(value * unit_scale)
    log(
        "converted camera optics for Isaac RTX: "
        f"focal={original[0]:.6g}->{original[0] * unit_scale:.6g}, "
        f"aperture={original[1]:.6g}x{original[2]:.6g}->"
        f"{original[1] * unit_scale:.6g}x{original[2] * unit_scale:.6g}"
    )

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport not available")
    viewport.camera_path = camera_path
    width, height = viewport.resolution
    log(f"viewport resolution: {width}x{height}")
    log(f"active camera: {CAMERA_PATH}")


def record_stage_defaults(stage):
    data = dict(stage.GetRootLayer().customLayerData)

    camera_settings = dict(data.get("cameraSettings", {}))
    camera_settings["boundCamera"] = CAMERA_PATH
    data["cameraSettings"] = camera_settings

    render_settings = dict(data.get("renderSettings", {}))
    render_settings.update(
        {
            "rtx:post:backgroundZeroAlpha:enabled": True,
            "rtx:post:backgroundZeroAlpha:backgroundComposite": True,
            "rtx:post:backgroundZeroAlpha:backgroundDefaultColor": Gf.Vec3f(
                *WHITE_BACKPLATE
            ),
        }
    )
    data["renderSettings"] = render_settings

    physics_settings = dict(data.get("physicsSettings", {}))
    physics_settings["/physics/updateToUsd"] = True
    data["physicsSettings"] = physics_settings
    stage.GetRootLayer().customLayerData = data


def add_collision(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)


def blender_executable():
    configured = (
        os.environ.get("TABLETOP_BLENDER_BIN")
        or os.environ.get("GRAM_BLENDER")
        or os.environ.get("BLENDER_BIN")
    )
    if configured:
        return configured
    found = shutil.which("blender")
    if found:
        return found
    raise FileNotFoundError(
        "Blender not found; set TABLETOP_BLENDER_BIN, GRAM_BLENDER, or BLENDER_BIN"
    )


def add_table_collision_proxy(stage, table_prim, visual_usd):
    if (
        TABLE_COLLISION_TARGET_FACES not in {0}
        and TABLE_COLLISION_TARGET_FACES < 4
    ) or TABLE_COLLISION_MAX_ERROR_M <= 0:
        raise ValueError(
            "table collision target faces must be 0 or >= 4 and max error positive"
        )
    output = OUTPUT_ROOT / "collision_proxies/table_0.npz"
    command = [
        blender_executable(), "--background", "--factory-startup",
        "--python-exit-code", "1", "--python",
        str(Path(__file__).with_name("blender_simplify_table_collision.py")), "--",
        "--input", str(visual_usd), "--output", str(output),
        "--object-id", "table_0",
        "--target-faces", str(TABLE_COLLISION_TARGET_FACES),
        "--max-error-m", str(TABLE_COLLISION_MAX_ERROR_M),
    ]
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    result = subprocess.run(
        command, text=True, capture_output=True, env=environment,
    )
    if result.returncode:
        raise RuntimeError(
            "table collision simplification failed:\n"
            + (result.stderr or result.stdout)[-4000:]
        )

    with np.load(output) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
        source_faces = int(data["source_faces"])
        collision_faces = int(data["collision_faces"])
        max_error = float(data["max_sampled_error_m"])
        source_components = int(data["source_components"])
        collision_components = int(data["collision_components"])
        source_boundary_edges = int(data["source_boundary_edges"])
        collision_boundary_edges = int(data["collision_boundary_edges"])
        topology_fallback = bool(data["topology_fallback"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"invalid table collision points: {points.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(f"invalid table collision triangles: {triangles.shape}")

    proxy = UsdGeom.Mesh.Define(
        stage, Sdf.Path("/World/CollisionProxies/table_0")
    )
    proxy.CreatePointsAttr().Set([Gf.Vec3f(*map(float, point)) for point in points])
    proxy.CreateFaceVertexCountsAttr().Set([3] * len(triangles))
    proxy.CreateFaceVertexIndicesAttr().Set(triangles.reshape(-1).tolist())
    proxy.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    proxy.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    collider = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    collider.CreateAttribute(
        "tabletop:sourceFaceCount", Sdf.ValueTypeNames.Int,
    ).Set(source_faces)
    collider.CreateAttribute(
        "tabletop:collisionFaceCount", Sdf.ValueTypeNames.Int,
    ).Set(collision_faces)
    collider.CreateAttribute(
        "tabletop:maxSampledErrorM", Sdf.ValueTypeNames.Double,
    ).Set(max_error)
    collider.CreateAttribute(
        "tabletop:topologyFallback", Sdf.ValueTypeNames.Bool,
    ).Set(topology_fallback)

    visual_colliders = [
        prim for prim in Usd.PrimRange(table_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if visual_colliders:
        raise RuntimeError(
            f"table visual meshes unexpectedly have collision: {visual_colliders}"
        )
    log(
        "created exact static Triangle Mesh proxy for table_0: "
        f"{source_faces}->{collision_faces} faces, "
        f"sampled error={max_error * 1000:.3f}mm, "
        f"components={source_components}->{collision_components}, "
        f"boundary edges={source_boundary_edges}->{collision_boundary_edges}, "
        f"topology fallback={topology_fallback}"
    )


def apply_sdf_collision(collider):
    if PhysxSchema is None:
        raise RuntimeError("PhysxSchema SDF collision API is unavailable")
    if (
        SDF_RESOLUTION <= 0
        or SDF_SUBGRID_RESOLUTION < 0
        or not 0.0 < SDF_TRIANGLE_REDUCTION_FACTOR <= 1.0
    ):
        raise ValueError(
            "SDF resolution must be positive, subgrid nonnegative, and "
            "triangle reduction factor in (0, 1]"
        )
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MeshCollisionAPI.Apply(
        collider
    ).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider)
    sdf.CreateSdfResolutionAttr().Set(SDF_RESOLUTION)
    sdf.CreateSdfSubgridResolutionAttr().Set(SDF_SUBGRID_RESOLUTION)
    sdf.CreateSdfBitsPerSubgridPixelAttr().Set(
        PhysxSchema.Tokens.bitsPerPixel16
    )
    sdf.CreateSdfEnableRemeshingAttr().Set(False)
    sdf.CreateSdfTriangleCountReductionFactorAttr().Set(
        SDF_TRIANGLE_REDUCTION_FACTOR
    )
    contact = PhysxSchema.PhysxCollisionAPI.Apply(collider)
    contact.CreateContactOffsetAttr().Set(DYNAMIC_CONTACT_OFFSET_M)
    contact.CreateRestOffsetAttr().Set(0.0)


def apply_convex_decomposition_collision(collider):
    if PhysxSchema is None:
        raise RuntimeError("PhysxSchema convex decomposition API is unavailable")
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MeshCollisionAPI.Apply(
        collider
    ).CreateApproximationAttr().Set("convexDecomposition")
    decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
        collider
    )
    decomposition.CreateMaxConvexHullsAttr().Set(
        CONVEX_DECOMPOSITION_MAX_HULLS
    )
    decomposition.CreateErrorPercentageAttr().Set(
        CONVEX_DECOMPOSITION_ERROR_PERCENTAGE
    )
    decomposition.CreateVoxelResolutionAttr().Set(
        CONVEX_DECOMPOSITION_VOXEL_RESOLUTION
    )
    decomposition.CreateHullVertexLimitAttr().Set(
        CONVEX_DECOMPOSITION_HULL_VERTEX_LIMIT
    )
    decomposition.CreateShrinkWrapAttr().Set(True)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(collider)
    contact.CreateContactOffsetAttr().Set(DYNAMIC_CONTACT_OFFSET_M)
    contact.CreateRestOffsetAttr().Set(0.0)


def add_dynamic_collision(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"cannot build dynamic collision for {root_path}: no meshes")
    for mesh in meshes:
        if DYNAMIC_COLLISION_MODE == "sdf":
            apply_sdf_collision(mesh)
        else:
            apply_convex_decomposition_collision(mesh)


def add_rigid_asset(stage, obj):
    path = f"/World/{obj['object_id']}"
    asset = EXPORT_ROOT / obj["asset_usd"]
    prim = UsdGeom.Xform.Define(stage, Sdf.Path(path)).GetPrim()
    prim.GetReferences().AddReference(str(asset))
    if obj["type"] == "dynamic_rigid":
        add_dynamic_collision(stage, path)
    else:
        add_collision(stage, path)

    if obj["type"] == "dynamic_rigid":
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = obj.get("physics", {}).get("mass_kg")
        if mass:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    log(f"loaded {obj['object_id']} from {asset}")


def localize_blender_visual_scene(manifest):
    source = EXPORT_ROOT / manifest["scene_visual"]["usd"]
    destination_root = OUTPUT_ROOT / "visual_scene"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)
    destination = destination_root / source.name
    shutil.copy2(source, destination)
    source_textures = source.parent / "textures"
    if source_textures.is_dir():
        shutil.copytree(source_textures, destination_root / "textures")
    return destination


def add_blender_visual_scene(stage, visual_usd):
    prim = UsdGeom.Xform.Define(stage, Sdf.Path(BLENDER_SCENE_PATH)).GetPrim()
    prim.GetReferences().AddReference(str(visual_usd))
    log(f"loaded Blender visual scene from {visual_usd}")
    return prim


def find_named_prim(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name or prim.GetName().startswith(f"{name}_"):
            return prim
    return None


def setup_visual_rigid(stage, root_path, obj, visual_usd):
    prim = find_named_prim(stage, root_path, obj["object_id"])
    if prim is None:
        log(f"warning: {obj['object_id']} not found in Blender visual scene")
        return
    if obj["object_id"] == "table_0":
        add_table_collision_proxy(stage, prim, visual_usd)
    elif obj["type"] == "dynamic_rigid":
        add_dynamic_collision(stage, str(prim.GetPath()))
    else:
        add_collision(stage, str(prim.GetPath()))
    if obj["type"] == "dynamic_rigid":
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = obj.get("physics", {}).get("mass_kg")
        if mass:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    log(f"enabled physics for {obj['object_id']} in Blender visual scene at {prim.GetPath()}")


def enable_urdf_importer():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)


def set_if_exists(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def define_gltf_pbr_material(
    stage, material_path, pbr, base_color_texture, metallic_roughness_texture,
):
    """Author a portable USD Preview Surface matching glTF metallic-roughness."""
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(
        stage, Sdf.Path(material_path).AppendChild("PBR")
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    if pbr["alpha_mode"] not in {"OPAQUE", "MASK", "BLEND"}:
        raise ValueError(f"unsupported glTF alpha mode: {pbr['alpha_mode']}")
    base = pbr["base_color_factor"]
    uv_readers = {}

    def uv_reader(index):
        if index not in uv_readers:
            reader = UsdShade.Shader.Define(
                stage, Sdf.Path(material_path).AppendChild(f"UV{index}")
            )
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(
                "st" if index == 0 else f"st{index}"
            )
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            uv_readers[index] = reader
        return uv_readers[index]

    def add_texture(name, path, settings, color_space, scale):
        if path is None or settings is None:
            return None
        texture = UsdShade.Shader.Define(
            stage, Sdf.Path(material_path).AppendChild(name)
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(path))
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            color_space
        )
        wrap_modes = {10497: "repeat", 33648: "mirror", 33071: "clamp"}
        for name, setting in (
            ("wrapS", settings["wrap_s"]),
            ("wrapT", settings["wrap_t"]),
        ):
            if setting not in wrap_modes:
                raise ValueError(f"unsupported glTF texture wrap mode: {setting}")
            texture.CreateInput(name, Sdf.ValueTypeNames.Token).Set(
                wrap_modes[setting]
            )
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*scale)
        )
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(0.0)
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader(settings["tex_coord_index"]).ConnectableAPI(), "result"
        )
        for output_name, value_type in (
            ("rgb", Sdf.ValueTypeNames.Float3),
            ("r", Sdf.ValueTypeNames.Float),
            ("g", Sdf.ValueTypeNames.Float),
            ("b", Sdf.ValueTypeNames.Float),
            ("a", Sdf.ValueTypeNames.Float),
        ):
            texture.CreateOutput(output_name, value_type)
        return texture

    base_texture = add_texture(
        "baseColorTex", base_color_texture, pbr["base_color_texture"],
        "sRGB", base,
    )
    if base_texture:
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).ConnectToSource(base_texture.ConnectableAPI(), "rgb")
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*base[:3])
        )

    mr_texture = add_texture(
        "metallicRoughnessTex", metallic_roughness_texture,
        pbr["metallic_roughness_texture"],
        "raw", (1.0, pbr["roughness_factor"], pbr["metallic_factor"], 1.0),
    )
    if mr_texture:
        shader.CreateInput(
            "metallic", Sdf.ValueTypeNames.Float
        ).ConnectToSource(mr_texture.ConnectableAPI(), "b")
        shader.CreateInput(
            "roughness", Sdf.ValueTypeNames.Float
        ).ConnectToSource(mr_texture.ConnectableAPI(), "g")
    else:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            pbr["metallic_factor"]
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            pbr["roughness_factor"]
        )

    if pbr["alpha_mode"] != "OPAQUE":
        opacity = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        if base_texture:
            opacity.ConnectToSource(base_texture.ConnectableAPI(), "a")
        else:
            opacity.Set(base[3])
        if pbr["alpha_mode"] == "MASK":
            shader.CreateInput(
                "opacityThreshold", Sdf.ValueTypeNames.Float
            ).Set(pbr["alpha_cutoff"])
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def connected_texture(shader, input_name):
    input_value = shader.GetInput(input_name)
    if not input_value:
        return None
    sources, _invalid = input_value.GetConnectedSources()
    if not sources:
        return None
    texture = UsdShade.Shader(sources[0].source.GetPrim())
    path = texture.GetInput("file").Get()
    return Path(path.path).name if isinstance(path, Sdf.AssetPath) else None


def restore_rigid_pbr_materials(stage, manifest):
    restored = 0
    for obj in manifest["objects"]:
        if obj["type"] not in {"static_rigid", "dynamic_rigid"}:
            continue
        root = find_named_prim(stage, BLENDER_SCENE_PATH, obj["object_id"])
        meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"no visual mesh found for {obj['object_id']}")
        binding = meshes[0].GetRelationship("material:binding")
        targets = binding.GetTargets() if binding else []
        if len(targets) != 1:
            raise RuntimeError(
                f"expected one material for {obj['object_id']}, found {len(targets)}"
            )
        source_material = UsdShade.Material(stage.GetPrimAtPath(targets[0]))
        source_shader, _name, _type = source_material.ComputeSurfaceSource()
        base_name = connected_texture(source_shader, "diffuseColor")
        mr_name = connected_texture(source_shader, "roughness")
        pbr = read_glb_pbr_material(obj["source_asset"])
        material = define_gltf_pbr_material(
            stage,
            f"/World/Looks/Rigid/{obj['object_id']}",
            pbr,
            f"visual_scene/textures/{base_name}" if base_name else None,
            f"visual_scene/textures/{mr_name}" if mr_name else None,
        )
        for mesh in meshes:
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
            UsdGeom.Mesh(mesh).CreateDoubleSidedAttr().Set(pbr["double_sided"])
        restored += 1
    log(f"restored original glTF PBR materials for {restored} rigid objects")
    return restored


def restore_articulation_pbr_materials(obj, urdf_path, asset_path):
    base_path = asset_path.parent / "configuration/model_base.usd"
    stage = Usd.Stage.Open(str(base_path))
    if stage is None:
        raise RuntimeError(f"could not open imported articulation USD: {base_path}")
    stage.RemovePrim(f"/{obj['object_id']}/Looks")
    xml = ET.parse(urdf_path).getroot()
    restored = 0
    for link in xml.findall("link"):
        visual = link.find("./visual/geometry/mesh")
        if visual is None:
            continue
        mesh_path = Path(visual.get("filename"))
        glb_path = urdf_path.parent / mesh_path
        pbr = read_glb_pbr_material(glb_path)
        tuning = STAGE7_ARTICULATION_PBR_TUNING.get(obj["object_id"])
        if tuning is not None:
            pbr = {
                **pbr,
                "metallic_factor": tuning[0],
                "roughness_factor": tuning[1],
            }
        texture_root = base_path.parent / "materials/textures"

        def imported_texture(settings):
            if settings is None:
                return None
            matches = sorted(texture_root.glob(
                f"{mesh_path.stem}_texture{settings['image_index']}.*"
            ))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one imported texture for {glb_path}, found {matches}"
                )
            return Path("materials/textures") / matches[0].name

        material = define_gltf_pbr_material(
            stage,
            f"/{obj['object_id']}/Looks/{link.get('name')}_PBR",
            pbr,
            imported_texture(pbr["base_color_texture"]),
            imported_texture(pbr["metallic_roughness_texture"]),
        )
        prefixes = (
            f"/visuals/{link.get('name')}/",
            f"/meshes/{mesh_path.stem}/",
        )
        meshes = [
            prim for prim in stage.TraverseAll()
            if prim.IsA(UsdGeom.Mesh)
            and str(prim.GetPath()).startswith(prefixes)
        ]
        if not meshes:
            raise RuntimeError(f"imported visual mesh missing for {link.get('name')}")
        for mesh in meshes:
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
            UsdGeom.Mesh(mesh).CreateDoubleSidedAttr().Set(pbr["double_sided"])
        restored += 1
    stage.GetRootLayer().Save()
    log(
        f"restored original glTF PBR materials for {restored} links in "
        f"{obj['object_id']}"
    )
    return restored


def configure_articulation_sdf(obj, asset_path):
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"could not open imported articulation USD: {asset_path}")
    merged_colliders = [
        prim for prim in stage.TraverseAll()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    ]
    if not merged_colliders:
        raise RuntimeError(f"imported articulation has no mesh colliders: {asset_path}")
    colliders = []
    for merged in merged_colliders:
        if merged.IsA(UsdGeom.Mesh):
            colliders.append(merged)
            continue

        # URDF import represents a link collider as an instanceable MeshMerge
        # Xform. Cook SDF data on the actual collision meshes instead.
        merged.SetInstanceable(False)
        UsdGeom.Imageable(merged).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        meshes = [
            prim for prim in Usd.PrimRange(merged)
            if prim.IsA(UsdGeom.Mesh)
        ]
        if not meshes:
            raise RuntimeError(
                f"imported merged collider has no meshes: {merged.GetPath()}"
            )
        merged.RemoveAPI(UsdPhysics.CollisionAPI)
        merged.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        merged.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
        merged.RemoveAPI(PhysxSchema.PhysxMeshMergeCollisionAPI)
        colliders.extend(meshes)

    for collider in colliders:
        apply_sdf_collision(collider)
    stage.GetRootLayer().Save()
    log(
        f"configured {len(colliders)} sparse SDF link colliders for "
        f"{obj['object_id']} at resolution {SDF_RESOLUTION}"
    )


def import_urdf_asset(obj):
    scale = obj.get("root_pose", {}).get("scale", (1.0, 1.0, 1.0))
    cache_root = os.environ.get("TABLETOP_IMPORTED_ARTICULATION_CACHE")
    if cache_root and all(abs(float(value) - 1.0) <= 1e-12 for value in scale):
        cached = Path(cache_root) / obj["object_id"]
        cached_asset = cached / "model.usd"
        if cached_asset.is_file():
            destination = OUTPUT_ROOT / "imported_articulations" / obj["object_id"]
            shutil.copytree(cached, destination, dirs_exist_ok=True)
            asset_path = destination / "model.usd"
            log(f"reused imported {obj['object_id']} asset from {cached_asset}")
            return asset_path

    enable_urdf_importer()
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")

    set_if_exists(import_config, "merge_fixed_joints", False)
    set_if_exists(import_config, "convex_decomp", True)
    set_if_exists(import_config, "import_inertia_tensor", False)
    fix_base = os.environ.get("TABLETOP_FIX_ARTICULATION_BASE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    set_if_exists(import_config, "fix_base", fix_base)
    set_if_exists(import_config, "collision_from_visuals", False)
    set_if_exists(import_config, "make_default_prim", True)
    set_if_exists(import_config, "create_physics_scene", False)

    source_package = EXPORT_ROOT / Path(obj["asset"]).parent
    scaled_package = OUTPUT_ROOT / "scaled_articulations" / obj["object_id"]
    if scaled_package.exists():
        shutil.rmtree(scaled_package)
    shutil.copytree(source_package, scaled_package)
    collision_links = use_visual_meshes_for_collisions(scaled_package)
    factors = bake_urdf_scale(scaled_package, scale)
    urdf_path = scaled_package / "model.urdf"
    asset_path = (
        OUTPUT_ROOT / "imported_articulations" / obj["object_id"] / "model.usd"
    )
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(asset_path),
        get_articulation_root=False,
    )
    if not status:
        raise RuntimeError(f"URDFParseAndImportFile failed: {urdf_path}")
    restore_articulation_pbr_materials(obj, urdf_path, asset_path)
    configure_articulation_sdf(obj, asset_path)
    log(
        f"converted {obj['object_id']} scaled URDF from {urdf_path} to "
        f"{asset_path}; collision_links={collision_links}; "
        f"prismatic_scales={factors}"
    )
    return asset_path


def reference_urdf(stage, obj, asset_path):
    target_path = f"/World/{obj['object_id']}"
    if stage.GetPrimAtPath(target_path):
        stage.RemovePrim(target_path)
    prim = UsdGeom.Xform.Define(stage, Sdf.Path(target_path)).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))
    pose = obj.get("root_pose", {})
    apply_xform(
        prim, pose.get("translation_m"), pose.get("yaw_deg"),
        None, pose.get("roll_x_deg"), pose.get("roll_y_deg"),
    )
    log(f"referenced {obj['object_id']} articulation at {target_path}")
    return target_path, asset_path


def author_initial_joint_states(stage, manifest):
    if PhysxSchema is None:
        raise RuntimeError("PhysxSchema.JointStateAPI is unavailable")
    for obj in manifest["objects"]:
        if obj["type"] != "articulated":
            continue
        package_manifest = (
            EXPORT_ROOT / Path(obj["asset"]).parent / "manifest.json"
        )
        states = json.loads(package_manifest.read_text(encoding="utf-8")).get(
            "joint_states", []
        )
        prismatic_scales = urdf_prismatic_scales(
            EXPORT_ROOT / obj["asset"],
            obj.get("root_pose", {}).get("scale", (1.0, 1.0, 1.0)),
        )
        joints = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(
                stage.GetPrimAtPath(f"/World/{obj['object_id']}")
            )
            if prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
        }
        for state in states:
            name = str(state["name"])
            prefix = f'{obj["object_id"]}_'
            if not name.startswith(prefix):
                name = prefix + name
            joint = joints.get(name)
            if joint is None:
                raise RuntimeError(
                    f"joint {name} from {package_manifest} is missing in imported USD"
                )
            revolute = joint.IsA(UsdPhysics.RevoluteJoint)
            token = UsdPhysics.Tokens.angular if revolute else UsdPhysics.Tokens.linear
            position = float(state["scene_current_q"])
            if revolute:
                position = math.degrees(position)
            else:
                position *= prismatic_scales.get(name, 1.0)
            joint_state = PhysxSchema.JointStateAPI.Apply(joint, token)
            joint_state.CreatePositionAttr().Set(position)
            joint_state.CreateVelocityAttr().Set(0.0)
            drive_name = "angular" if revolute else "linear"
            for attribute_name in ("stiffness", "damping"):
                attribute = joint.GetAttribute(
                    f"drive:{drive_name}:physics:{attribute_name}"
                )
                if attribute:
                    attribute.Set(0.0)
            log(f"authored initial joint state {name}={position:g}")


def export_portable_stage(stage, asset_references):
    stage.GetRootLayer().Export(str(OUT_STAGE))
    saved = Usd.Stage.Open(str(OUT_STAGE))
    for prim_path, asset_path in asset_references:
        prim = saved.GetPrimAtPath(prim_path)
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(
            os.path.relpath(asset_path, OUT_STAGE.parent)
        )
    saved.GetRootLayer().Save()


def publish_asset_package():
    OUT_PACKAGE.unlink(missing_ok=True)
    if not UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(OUT_STAGE)), str(OUT_PACKAGE),
    ):
        raise RuntimeError(f"failed to package complete Isaac asset: {OUT_PACKAGE}")
    log(f"published complete Isaac asset package: {OUT_PACKAGE}")


def setup_stage():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    return stage


def lighting_mode():
    mode = os.environ.get("TABLETOP_LIGHTING_MODE", "isaac").strip().lower()
    if mode not in {"blender", "isaac"}:
        raise ValueError("TABLETOP_LIGHTING_MODE must be blender or isaac")
    return mode


def add_isaac_studio_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    dome.CreateIntensityAttr(float(os.environ.get("TABLETOP_DOME_INTENSITY", "1000")))
    dome.GetPrim().CreateAttribute(
        "visibleInPrimaryRay",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)
    softbox = UsdLux.RectLight.Define(stage, Sdf.Path("/World/DiffuseSoftbox"))
    softbox.CreateIntensityAttr(
        float(os.environ.get("TABLETOP_SOFTBOX_INTENSITY", "4500"))
    )
    softbox.CreateWidthAttr(6.0)
    softbox.CreateHeightAttr(6.0)
    softbox.CreateNormalizeAttr(True)
    apply_xform(softbox.GetPrim(), translate=(0.0, -0.8, 3.5))


def configure_embedded_blender_lights(stage, root):
    scale = float(os.environ.get("TABLETOP_BLENDER_LIGHT_INTENSITY_SCALE", "75"))
    configured = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() not in LIGHT_TYPES:
            continue
        if prim.GetTypeName() == "DomeLight":
            prim.SetActive(False)
            continue
        prim.SetActive(True)
        intensity = prim.GetAttribute("inputs:intensity")
        source_intensity = float(intensity.Get() or 0.0)
        converted_intensity = source_intensity * scale
        intensity.Set(converted_intensity)
        configured += 1
        log(
            f"converted Blender light {prim.GetPath()}: "
            f"intensity={source_intensity:g}->{converted_intensity:g}"
        )
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/BlenderWorldLight"))
    dome.CreateIntensityAttr(
        float(os.environ.get("TABLETOP_BLENDER_WORLD_INTENSITY", "1000"))
    )
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    dome.GetPrim().CreateAttribute(
        "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool,
    ).Set(False)
    log(f"using {configured} embedded Blender light(s) and one root world light")


def disable_embedded_lights(root):
    disabled = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() in LIGHT_TYPES:
            prim.SetActive(False)
            disabled += 1
    log(f"disabled {disabled} embedded Blender light(s)")


def find_articulation_roots(stage, candidate_paths=None):
    roots = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
    for path in candidate_paths or []:
        if stage.GetPrimAtPath(path):
            roots.append(str(path))
    unique_roots = []
    seen = set()
    for path in roots:
        if path in seen:
            continue
        seen.add(path)
        unique_roots.append(path)
    return unique_roots


def finite_joint_limits(dof_properties):
    joints = []
    for index, properties in enumerate(dof_properties):
        lower = float(properties["lower"])
        upper = float(properties["upper"])
        if not (math.isfinite(lower) and math.isfinite(upper)):
            continue
        if upper <= lower:
            continue
        if abs(upper - lower) < 1.0e-5:
            continue
        joints.append((index, lower, upper))
    return joints


def demo_targets(joints, elapsed, phase_offset):
    targets = []
    indices = []
    fraction = min(max(JOINT_DEMO_RANGE_FRACTION, 0.05), 1.0)
    for ordinal, (index, lower, upper) in enumerate(joints):
        center = 0.5 * (lower + upper)
        amplitude = 0.5 * (upper - lower) * fraction
        phase = elapsed * JOINT_DEMO_SPEED + phase_offset + ordinal * 0.73
        targets.append(center + amplitude * math.sin(phase))
        indices.append(index)
    return (
        np.array(targets, dtype=np.float32),
        np.array(indices, dtype=np.int32),
    )


class JointRangeDemo:
    def __init__(self, articulations):
        self._articulations = articulations
        self._elapsed = 0.0
        self._subscription = None

    def start(self):
        self._subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="tabletop_joint_range_demo",
            )
        )

    def _on_update(self, event):
        dt = event.payload.get("dt", 1.0 / 60.0)
        self._elapsed += float(dt)
        for item in self._articulations:
            targets, indices = demo_targets(
                item["joints"],
                self._elapsed,
                item["phase_offset"],
            )
            item["controller"].apply_action(
                ArticulationAction(
                    joint_positions=targets,
                    joint_indices=indices,
                )
            )


async def start_joint_range_demo(stage, candidate_paths=None):
    if not JOINT_DEMO_ENABLED:
        log("joint range demo disabled")
        return

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(8):
        await app.next_update_async()

    demo_articulations = []
    for ordinal, root_path in enumerate(find_articulation_roots(stage, candidate_paths)):
        articulation = SingleArticulation(
            root_path,
            name=f"tabletop_joint_demo_{ordinal}",
        )
        try:
            articulation.initialize()
        except Exception as exc:
            log(f"warning: could not initialize articulation {root_path}: {exc}")
            continue

        joints = finite_joint_limits(articulation.dof_properties)
        if not joints:
            log(f"joint range demo skipped {root_path}: no finite movable DOFs")
            continue

        if JOINT_DEMO_DISABLE_GRAVITY:
            articulation.disable_gravity()

        demo_articulations.append(
            {
                "root_path": root_path,
                "controller": articulation.get_articulation_controller(),
                "joints": joints,
                "phase_offset": ordinal * 1.17,
            }
        )
        log(f"joint range demo added {root_path}: {len(joints)} DOF(s)")

    if not demo_articulations:
        log("joint range demo found no movable articulations")
        return

    demo = JointRangeDemo(demo_articulations)
    demo.start()
    globals()["tabletop_joint_range_demo"] = demo
    log("joint range demo running")


def main():
    log(f"reading {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    omni.usd.get_context().new_stage()
    articulated_assets = {}
    for obj in manifest["objects"]:
        if obj["type"] != "articulated":
            continue
        try:
            articulated_assets[obj["object_id"]] = import_urdf_asset(obj)
        except Exception as exc:
            log(f"URDF import failed for {obj['object_id']}: {exc}")

    visual_usd = localize_blender_visual_scene(manifest)
    stage = setup_stage()
    visual_root = add_blender_visual_scene(stage, visual_usd)
    restore_rigid_pbr_materials(stage, manifest)
    demo_root_candidates = []
    asset_references = [(
        BLENDER_SCENE_PATH,
        visual_usd,
    )]

    for obj in manifest["objects"]:
        if obj["type"] in {"static_rigid", "dynamic_rigid"}:
            setup_visual_rigid(
                stage, str(visual_root.GetPath()), obj, visual_usd,
            )
        elif obj["object_id"] in articulated_assets:
            asset_references.append(
                reference_urdf(stage, obj, articulated_assets[obj["object_id"]])
            )
            root_path = f"/World/{obj['object_id']}"
            demo_root_candidates.extend([f"{root_path}/base", root_path])

    mode = lighting_mode()
    if mode == "isaac":
        disable_embedded_lights(visual_root)
        add_isaac_studio_lights(stage)
        log("using Isaac studio lighting")
    else:
        configure_embedded_blender_lights(stage, visual_root)

    configure_rtx_background()
    table_prim = find_named_prim(stage, str(visual_root.GetPath()), "table_0")
    if table_prim is None:
        raise RuntimeError("table_0 not found in Blender visual scene")
    add_white_studio_ground(stage, table_prim)
    configure_fixed_camera(stage)
    record_stage_defaults(stage)

    leaked = [
        path for path in ("/visuals", "/meshes", "/colliders")
        if stage.GetPrimAtPath(path)
    ]
    if leaked:
        raise RuntimeError(f"URDF import leaked shared root prims: {leaked}")
    author_initial_joint_states(stage, manifest)
    export_portable_stage(stage, asset_references)
    log(f"saved {OUT_STAGE}")
    if os.environ.get("TABLETOP_PUBLISH_USDZ", "1").lower() in {
        "1", "true", "yes", "on",
    }:
        publish_asset_package()
    context = omni.usd.get_context()
    if not context.open_stage(str(OUT_STAGE)):
        raise RuntimeError(f"could not reopen final scene: {OUT_STAGE}")
    stage = context.get_stage()
    log(f"reopened final scene from {OUT_STAGE}")
    asyncio.ensure_future(start_joint_range_demo(stage, demo_root_candidates))


main()
