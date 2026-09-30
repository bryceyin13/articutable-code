#!/usr/bin/env python3
import argparse
import gc
import shutil
import traceback
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT / "multiviews"
OUTPUT_ROOT = PROJECT / "trellis2_multiviews"
TRELLIS_ROOT = PROJECT.parent / "third_party" / "trellis2"
MODEL_ROOT = PROJECT.parent / "third_party" / "trellis2-model"


def discover_objects(root=INPUT_ROOT, view="front"):
    return sorted(path.parent.name for path in Path(root).glob(f"*/{view}.png"))


def is_complete(output, name):
    output = Path(output)
    return all(
        (output / filename).is_file() and (output / filename).stat().st_size > 0
        for filename in (f"{name}.glb", "preview.png", "preview.mp4")
    )


def load_pipeline():
    import OpenEXR
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap

    hdri = OpenEXR.File(str(TRELLIS_ROOT / "assets/hdri/forest.exr")).channels()["RGB"].pixels
    envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_ROOT))
    pipeline.cuda()
    return pipeline, envmap


def generate(name, pipeline, envmap, view="front", target_faces=10000, output_root=OUTPUT_ROOT, variant=None, remesh=True, pipeline_type="1024_cascade", shape_guidance_strength=7.5):
    import imageio
    import o_voxel
    from PIL import Image
    from trellis2.utils import render_utils

    source = INPUT_ROOT / name / f"{view}.png"
    output = Path(output_root) / name
    if variant:
        output /= variant
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "input.png")

    mesh = pipeline.run(
        Image.open(source),
        pipeline_type=pipeline_type,
        shape_slat_sampler_params={"guidance_strength": shape_guidance_strength},
    )[0]
    mesh.simplify(16777216)
    video = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=envmap, resolution=512, num_frames=120)
    )
    imageio.mimsave(output / "preview.mp4", video, fps=15)
    imageio.imwrite(output / "preview.png", video[0])
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target_faces,
        texture_size=4096,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(output / f"{name}.glb", extension_webp=True)


def release_cuda_memory():
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("objects", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--view", default="front")
    parser.add_argument("--target-faces", type=int, default=10000)
    parser.add_argument("--pipeline-type", choices=("512", "1024", "1024_cascade", "1536_cascade"), default="1024_cascade")
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--pipeline-512-object", action="append", default=[])
    parser.add_argument("--oom-record", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--variant")
    parser.add_argument("--no-remesh", dest="remesh", action="store_false")
    parser.set_defaults(remesh=True)
    return parser


def main():
    args = build_parser().parse_args()
    names = args.objects or discover_objects(view=args.view)
    if args.list:
        print("\n".join(names))
        return

    outputs = {
        name: args.output_root / name / args.variant if args.variant else args.output_root / name
        for name in names
    }
    pending = [name for name in names if not is_complete(outputs[name], name)]
    if not pending:
        print("All requested objects are complete.", flush=True)
        return

    pipeline, envmap = load_pipeline()
    failures = []
    pipeline_512_objects = set(args.pipeline_512_object)
    for name in pending:
        try:
            print(f"[{name}] starting", flush=True)
            pipeline_type = "512" if name in pipeline_512_objects else args.pipeline_type
            generate(name, pipeline, envmap, args.view, args.target_faces, args.output_root, args.variant, args.remesh, pipeline_type, args.shape_guidance_strength)
            print(f"[{name}] complete", flush=True)
        except Exception as error:
            failures.append(name)
            traceback.print_exc()
            if "out of memory" in str(error).lower() and args.oom_record:
                args.oom_record.parent.mkdir(parents=True, exist_ok=True)
                args.oom_record.write_text(name + "\n", encoding="utf-8")
                return 75
        finally:
            release_cuda_memory()
    if failures:
        raise SystemExit(f"Failed objects: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
