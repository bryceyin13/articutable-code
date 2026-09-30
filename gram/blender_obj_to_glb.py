#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.render_settings import configure_specular_reflections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args = parser.parse_args(argv)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=str(Path(args.input)))
    configure_specular_reflections(bpy)
    bpy.ops.export_scene.gltf(filepath=str(Path(args.output)), export_format="GLB")


if __name__ == "__main__":
    main()
