#!/usr/bin/env python3
"""Shared Blender render hyperparameters used by every pipeline stage."""

import os
import re


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def specular_reflections_enabled(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get("BLENDER_SPECULAR_REFLECTIONS", "1").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        "BLENDER_SPECULAR_REFLECTIONS must be 0/1, false/true, no/yes, or off/on"
    )


def disabled_specular_materials(environ=None):
    environ = os.environ if environ is None else environ
    return tuple(
        name.strip()
        for name in environ.get(
            "BLENDER_DISABLE_SPECULAR_MATERIALS", "Material_0"
        ).split(",")
        if name.strip()
    )


def _matches_material(material_name, configured_names):
    source_name = re.sub(r"\.\d{3}$", "", material_name)
    return material_name in configured_names or source_name in configured_names


def _set_input(node_tree, node, names, value):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            for link in list(getattr(socket, "links", ())):
                node_tree.links.remove(link)
            socket.default_value = value


def _disable_material_specular(material):
    if hasattr(material, "metallic"):
        material.metallic = 0.0
    if hasattr(material, "specular_intensity"):
        material.specular_intensity = 0.0
    if not material.use_nodes or material.node_tree is None:
        return
    tree = material.node_tree
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            _set_input(tree, node, ("Metallic",), 0.0)
            _set_input(tree, node, ("IOR",), 1.0)
            _set_input(tree, node, ("Specular IOR Level", "Specular"), 0.0)
            _set_input(
                tree, node,
                ("Coat Weight", "Clearcoat", "Anisotropic IOR Level"),
                0.0,
            )
        elif node.type == "BSDF_GLASS":
            _set_input(tree, node, ("IOR",), 1.0)
        elif node.type == "BSDF_GLOSSY":
            _set_input(tree, node, ("Color",), (0.0, 0.0, 0.0, 1.0))
            _set_input(tree, node, ("Roughness",), 1.0)


def configure_specular_reflections(bpy_module, scene=None, environ=None):
    """Apply global and material-level reflection settings after asset import."""
    enabled = specular_reflections_enabled(environ)
    configured_names = disabled_specular_materials(environ)
    scene = scene or bpy_module.context.scene
    scene["specular_reflections_enabled"] = enabled
    scene["specular_disabled_materials"] = ",".join(configured_names)
    if enabled and not configured_names:
        return True

    if not enabled:
        cycles = getattr(scene, "cycles", None)
        if cycles is not None and hasattr(cycles, "glossy_bounces"):
            cycles.glossy_bounces = 0
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = False

    for material in bpy_module.data.materials:
        if enabled and not _matches_material(material.name, configured_names):
            continue
        _disable_material_specular(material)
    return enabled
