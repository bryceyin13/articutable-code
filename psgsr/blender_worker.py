#!/usr/bin/env python3
"""Persistent Blender worker for the accelerated two-camera alignment."""

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.argv.append("--blender-worker")
from psgsr import core as pipeline
sys.argv.remove("--blender-worker")


def receive_line(connection):
    chunks = []
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def execute(request):
    command = request["command"]
    if command == "render_jobs":
        for scene_json, output_dir, save_blend in request["jobs"]:
            pipeline.blender_render_one(
                Path(scene_json), Path(output_dir), bool(save_blend),
                scene_mode=request.get("scene_mode", "persistent"),
                views=tuple(request.get("views", ("front", "top"))),
                samples=request.get("samples"),
            )
    elif command == "top_templates":
        pipeline.blender_render_top_templates(
            Path(request["scene_json"]), Path(request["output_dir"]),
            int(request.get("render_scale", 1)),
            scene_mode=request.get("scene_mode", "persistent"),
        )
    elif command == "bake_mesh_orientations":
        for source, destination, matrix in request["jobs"]:
            pipeline.blender_bake_mesh_orientation(source, destination, matrix)
    elif command != "quit":
        raise ValueError(f"unknown Blender worker command: {command}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        server.settimeout(1.0)
        while True:
            try:
                connection, _ = server.accept()
            except socket.timeout:
                if os.getppid() != args.parent_pid:
                    break
                continue
            request = {}
            with connection:
                try:
                    request = json.loads(receive_line(connection))
                    execute(request)
                    response = {"ok": True}
                except Exception as error:
                    response = {
                        "ok": False,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
            if request.get("command") == "quit":
                break
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
