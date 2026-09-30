import json
import math
import os
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def env_positive_int(name, default):
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def path(*parts):
    return ROOT.joinpath(*parts)


def _contained_path(root, parts):
    root = Path(root).resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path is outside {root}: {candidate}")
    return candidate


def project_path(context, *parts):
    """Resolve source/configuration data from an explicit run context."""
    return _contained_path(context.project_root, parts)


def attempt_path(attempt, *parts):
    """Resolve an output inside the active attempt's temporary directory."""
    return _contained_path(attempt.temp_root, parts)


def ensure_dirs():
    for name in [
        "data",
        "masks",
        "crops",
        "completed_instances",
        "multiviews",
        "rigid_meshes",
        "articulated_assets/laptop_0",
        "outputs",
    ]:
        path(name).mkdir(parents=True, exist_ok=True)


def log_step(stage, message):
    print(f"[{stage}] {message}", flush=True)


def write_json(out_path, data):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(in_path):
    return json.loads(in_path.read_text(encoding="utf-8"))


def mesh_path(manifest_path, object_id):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = read_json(manifest_path)
    try:
        item = next(item for item in manifest["items"] if item["object_id"] == object_id)
    except StopIteration:
        raise ValueError(f"image-to-3d manifest has no object: {object_id}") from None
    relative = Path(item["mesh"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid image-to-3d mesh path: {relative}")
    attempt_root = manifest_path.parents[1]
    mesh = (attempt_root / relative).resolve()
    if attempt_root != mesh and attempt_root not in mesh.parents:
        raise ValueError(f"mesh is outside image-to-3d attempt: {mesh}")
    if not mesh.is_file():
        raise FileNotFoundError(f"missing normalized image-to-3d mesh: {mesh}")
    return mesh


def png_chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(out_path, width, height, bg, shapes):
    pixels = [[list(bg) for _ in range(width)] for _ in range(height)]

    def put(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = list(color)

    def rect(x0, y0, x1, y1, color):
        x0, x1 = sorted((max(0, int(x0)), min(width, int(x1))))
        y0, y1 = sorted((max(0, int(y0)), min(height, int(y1))))
        for y in range(y0, y1):
            row = pixels[y]
            for x in range(x0, x1):
                row[x] = list(color)

    def ellipse(cx, cy, rx, ry, color):
        for y in range(int(cy - ry), int(cy + ry) + 1):
            for x in range(int(cx - rx), int(cx + rx) + 1):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                    put(x, y, color)

    def point_in_poly(x, y, pts):
        inside = False
        j = len(pts) - 1
        for i, pt in enumerate(pts):
            xi, yi = pt
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def polygon(pts, color):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for y in range(max(0, int(min(ys))), min(height, int(max(ys)) + 1)):
            for x in range(max(0, int(min(xs))), min(width, int(max(xs)) + 1)):
                if point_in_poly(x + 0.5, y + 0.5, pts):
                    put(x, y, color)

    for shape in shapes:
        kind = shape[0]
        if kind == "rect":
            _, x0, y0, x1, y1, color = shape
            rect(x0, y0, x1, y1, color)
        elif kind == "ellipse":
            _, cx, cy, rx, ry, color = shape
            ellipse(cx, cy, rx, ry, color)
        elif kind == "polygon":
            _, pts, color = shape
            polygon(pts, color)

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for rgb in row:
            raw.extend(rgb)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + png_chunk(b"IEND", b"")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def draw_object_image(out_path, object_id, mask=False):
    if mask:
        write_png(out_path, 256, 256, (0, 0, 0), [("rect", 48, 48, 208, 208, (255, 255, 255))])
        return
    colors = {
        "table_0": (218, 197, 164),
        "laptop_0": (70, 84, 102),
        "mouse_0": (80, 82, 86),
        "mug_0": (231, 231, 224),
        "notebook_0": (48, 93, 150),
        "plant_0": (66, 132, 79),
    }
    color = colors.get(object_id, (160, 160, 160))
    shapes = [("rect", 0, 0, 256, 256, (255, 255, 255))]
    if object_id == "laptop_0":
        shapes += [
            ("rect", 48, 145, 208, 178, (85, 90, 100)),
            ("rect", 66, 70, 190, 145, color),
        ]
    elif object_id in ("mouse_0", "mug_0", "plant_0"):
        shapes.append(("ellipse", 128, 128, 60, 72, color))
    else:
        shapes.append(("rect", 48, 80, 208, 176, color))
    write_png(out_path, 256, 256, (255, 255, 255), shapes)


def write_obj(out_path, verts, faces):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# generated by ArticuTable\n"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts]
    lines += ["f " + " ".join(str(i) for i in face) + "\n" for face in faces]
    out_path.write_text("".join(lines), encoding="utf-8")


def write_box_obj(out_path, size_m, center=(0.0, 0.0, 0.0)):
    sx, sy, sz = [v / 2.0 for v in size_m]
    cx, cy, cz = center
    verts = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    faces = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    write_obj(out_path, verts, faces)


def write_screen_obj(out_path):
    width, thick, height = 0.32, 0.012, 0.22
    x0, x1 = -width / 2, width / 2
    y0, y1 = 0.0, thick
    z0, z1 = 0.0, height
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    write_obj(out_path, verts, faces)


def write_cylinder_obj(out_path, radius_m, height_m, segments=24):
    verts = [(0.0, 0.0, -height_m / 2), (0.0, 0.0, height_m / 2)]
    for z in (-height_m / 2, height_m / 2):
        for i in range(segments):
            a = 2 * math.pi * i / segments
            verts.append((radius_m * math.cos(a), radius_m * math.sin(a), z))
    faces = []
    bottom_start = 3
    top_start = 3 + segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((1, bottom_start + j, bottom_start + i))
        faces.append((2, top_start + i, top_start + j))
        faces.append((bottom_start + i, bottom_start + j, top_start + j, top_start + i))
    write_obj(out_path, verts, faces)


def usd_header():
    return """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
"""


def usd_cube(name, pose, kind):
    tx, ty, tz = [v / 100 for v in pose["translation_cm"]]
    sx, sy, sz = [v / 100 for v in pose["scale_cm"]]
    yaw = pose.get("yaw_deg", 0)
    return f"""    def Cube "{name}" (
        customData = {{
            string operability_type = "{kind}"
        }}
    )
    {{
        double size = 1
        float3 xformOp:translate = ({tx:.4f}, {ty:.4f}, {tz:.4f})
        float xformOp:rotateZ = {yaw:.4f}
        float3 xformOp:scale = ({sx:.4f}, {sy:.4f}, {sz:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
    }}
"""
