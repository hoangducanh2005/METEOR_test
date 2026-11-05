"""Render a PNK ego-frame LiDAR scan and its 3D labels in bird's-eye view."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    "Car": (255, 194, 46),
    "Rider": (41, 211, 219),
    "Pedestrian": (255, 91, 150),
}
UNKNOWN_COLOR = (235, 235, 235)


@dataclass(frozen=True)
class Box:
    number: int
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    label: str
    occlusion: str
    lidar_points: str
    track_id: str

    def footprint(self) -> np.ndarray:
        """Return the four BEV corners; box yaw is around ego +Z."""
        local = np.array([
            [-self.length / 2, -self.width / 2],
            [ self.length / 2, -self.width / 2],
            [ self.length / 2,  self.width / 2],
            [-self.length / 2,  self.width / 2],
        ])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s], [s, c]])
        return local @ rotation.T + np.array([self.x, self.y])


def _parse_boxes(label_path: Path) -> list[Box]:
    boxes: list[Box] = []
    with label_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 11:
                raise ValueError(f"Invalid 3D label at {label_path}:{line_number}: expected 11 fields")
            boxes.append(Box(
                number=len(boxes) + 1,
                x=float(fields[0]), y=float(fields[1]), z=float(fields[2]),
                length=float(fields[3]), width=float(fields[4]), height=float(fields[5]),
                yaw=float(fields[6]), label=fields[7], occlusion=fields[8],
                lidar_points=fields[9], track_id=fields[10],
            ))
    return boxes


def _find_lidar(raw_root: Path, stem: str) -> Path:
    matches = list(raw_root.glob(f"*/LIDAR/LIDAR_MERGED_EGO/{stem}.laz"))
    if not matches:
        raise FileNotFoundError(f"No merged ego LiDAR frame named {stem}.laz under {raw_root}")
    if len(matches) > 1:
        raise ValueError(f"More than one LiDAR frame matches {stem}: {matches}")
    return matches[0]


def _find_pair(raw_root: Path, label_root: Path, timestamp: str | None,
               allow_missing_labels: bool = False) -> tuple[Path, Path | None]:
    if timestamp:
        stem = Path(timestamp).stem
        if stem.endswith(".txt"):
            stem = Path(stem).stem
        label_matches = list(label_root.rglob(f"{stem}.txt"))
        if not label_matches and "-" not in stem:
            label_matches = list(label_root.rglob(f"{stem}-*.txt"))
        if not label_matches and allow_missing_labels:
            return _find_lidar(raw_root, stem), None
        if len(label_matches) != 1:
            raise FileNotFoundError(f"Expected one label matching '{timestamp}', found {len(label_matches)}")
        label_path = label_matches[0]
        return _find_lidar(raw_root, label_path.stem), label_path

    for label_path in sorted(label_root.rglob("*.txt")):
        try:
            lidar_path = _find_lidar(raw_root, label_path.stem)
            return lidar_path, label_path
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No timestamp-matched label/LiDAR pair found under {label_root} and {raw_root}")


def _read_points(lidar_path: Path, x_range: tuple[float, float],
                 y_range: tuple[float, float], max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    cloud = laspy.read(lidar_path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32, copy=False)
    intensity = np.asarray(cloud.intensity, dtype=np.float32)
    keep = (
        (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
        (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1])
    )
    xyz, intensity = xyz[keep], intensity[keep]
    count_in_view = len(xyz)
    if len(xyz) > max_points:
        stride = int(np.ceil(len(xyz) / max_points))
        xyz, intensity = xyz[::stride], intensity[::stride]
    return xyz, intensity, count_in_view


def _render(lidar_path: Path, label_path: Path | None, output_path: Path,
            x_range: tuple[float, float], y_range: tuple[float, float],
            pixels_per_meter: int, max_points: int) -> tuple[int, int]:
    boxes = _parse_boxes(label_path) if label_path else []
    points, intensity, count_in_view = _read_points(lidar_path, x_range, y_range, max_points)

    margin, panel_top, side_width = 76, 76, 460
    map_width = int(round((y_range[1] - y_range[0]) * pixels_per_meter))
    map_height = int(round((x_range[1] - x_range[0]) * pixels_per_meter))
    image = Image.new("RGB", (margin * 2 + map_width + side_width,
                               panel_top + map_height + 84), (13, 20, 28))
    draw = ImageDraw.Draw(image)
    left, top = margin, panel_top
    right, bottom = left + map_width, top + map_height

    # Grid and axes use ego coordinates: +X forward, +Y left.
    for x in np.arange(np.ceil(x_range[0] / 10) * 10, x_range[1] + 0.1, 10):
        py = int(round(top + (x_range[1] - x) * pixels_per_meter))
        draw.line((left, py, right, py), fill=(41, 54, 65), width=1)
        draw.text((left + 5, py + 3), f"X={x:.0f}m", fill=(145, 160, 171))
    for y in np.arange(np.ceil(y_range[0] / 10) * 10, y_range[1] + 0.1, 10):
        px = int(round(left + (y_range[1] - y) * pixels_per_meter))
        draw.line((px, top, px, bottom), fill=(41, 54, 65), width=1)
        draw.text((px + 3, bottom + 5), f"{y:.0f}", fill=(145, 160, 171))
    draw.rectangle((left, top, right, bottom), outline=(115, 132, 145), width=2)

    if len(points):
        px = np.rint(left + (y_range[1] - points[:, 1]) * pixels_per_meter).astype(np.int32)
        py = np.rint(top + (x_range[1] - points[:, 0]) * pixels_per_meter).astype(np.int32)
        valid = (px >= left) & (px < right) & (py >= top) & (py < bottom)
        px, py, intensity = px[valid], py[valid], intensity[valid]
        lo, hi = np.percentile(intensity, [5, 99]) if len(intensity) else (0, 1)
        shade = np.clip((intensity - lo) / max(float(hi - lo), 1.0), 0.22, 1.0)
        colors = np.column_stack((shade * 92, shade * 202, shade * 218)).astype(np.uint8)
        canvas = np.asarray(image).copy()
        canvas[py, px] = colors
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for box in boxes:
        corners = box.footprint()
        if (corners[:, 0].max() < x_range[0] or corners[:, 0].min() > x_range[1] or
                corners[:, 1].max() < y_range[0] or corners[:, 1].min() > y_range[1]):
            continue
        polygon = [
            (int(round(left + (y_range[1] - y) * pixels_per_meter)),
             int(round(top + (x_range[1] - x) * pixels_per_meter)))
            for x, y in corners
        ]
        color = CLASS_COLORS.get(box.label, UNKNOWN_COLOR)
        box_draw.polygon(polygon, fill=(*color, 45))
        box_draw.line(polygon + [polygon[0]], fill=(*color, 255), width=3, joint="curve")
        if x_range[0] <= box.x <= x_range[1] and y_range[0] <= box.y <= y_range[1]:
            center = (int(round(left + (y_range[1] - box.y) * pixels_per_meter)),
                      int(round(top + (x_range[1] - box.x) * pixels_per_meter)))
            text = str(box.number)
            bounds = box_draw.textbbox(center, text, font=font, anchor="mm")
            box_draw.rectangle(bounds, fill=(0, 0, 0, 220))
            box_draw.text(center, text, fill=(*color, 255), font=font, anchor="mm")
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 25), "PNK LiDAR BEV - 3D annotation footprints", fill=(245, 248, 250))
    subtitle = "Ego axes: +X forward, +Y left | box outlines are 3D box footprints"
    if label_path is None:
        subtitle = "NO LABEL FILE - unannotated frame; zero boxes does not mean an empty scene"
    draw.text((margin, 45), subtitle,
              fill=(190, 203, 212))
    side_x = right + 24
    draw.text((side_x, top), "BOX INDEX -> CLASS / TRACK", fill=(245, 248, 250))
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1
    summary = "  ".join(f"{name}: {n}" for name, n in sorted(counts.items())) or "no boxes"
    draw.text((side_x, top + 22), summary, fill=(170, 190, 202))
    draw.text((side_x, top + 43), f"Box count: {len(boxes)} | LiDAR points in view: {count_in_view:,}",
              fill=(170, 190, 202))
    line_y = top + 76
    line_step = 17
    max_rows = max(1, (map_height - 84) // line_step)
    for box in boxes[:max_rows]:
        color = CLASS_COLORS.get(box.label, UNKNOWN_COLOR)
        row = f"{box.number:>3}  {box.label:<11} track {box.track_id:<5} ({box.x:5.1f},{box.y:5.1f})m"
        draw.text((side_x, line_y), row, fill=color)
        line_y += line_step
    remaining = len(boxes) - min(len(boxes), max_rows)
    if remaining > 0:
        draw.text((side_x, line_y), f"... {remaining} more; see CSV", fill=(190, 203, 212))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "class", "track_id", "x_m", "y_m", "z_m",
                         "length_m", "width_m", "height_m", "yaw_rad", "occlusion", "lidar_points"])
        for b in boxes:
            writer.writerow([b.number, b.label, b.track_id, b.x, b.y, b.z,
                             b.length, b.width, b.height, b.yaw, b.occlusion, b.lidar_points])
    return len(boxes), count_in_view


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one timestamp-matched PNK merged-ego LiDAR frame and 3D labels as a numbered BEV PNG."
    )
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\mentor_data\PNKData"),
                        help="Root containing session folders with LIDAR/LIDAR_MERGED_EGO")
    parser.add_argument("--label-root", type=Path,
                        default=Path(r"D:\mentor_data\PNKData_meta_fresh\label\2026_08_14_vf6_01_02"),
                        help="Root containing segment folders and timestamped .txt labels")
    parser.add_argument("--timestamp", help="Exact label/LiDAR stem, e.g. 1772718324-099732876")
    parser.add_argument("--allow-missing-labels", action="store_true",
                        help="Render LiDAR without boxes when this timestamp has no annotation file")
    parser.add_argument("--label-file", type=Path, help="Use this exact annotation .txt instead of searching label-root")
    parser.add_argument("--lidar-file", type=Path, help="Use this exact .laz file instead of timestamp lookup")
    parser.add_argument("--output", type=Path, default=Path("out/pnk_lidar_bev/frame_bev.png"))
    parser.add_argument("--x-range", type=float, nargs=2, default=(-20.0, 80.0), metavar=("MIN", "MAX"),
                        help="Ego X display range in meters (forward is +X)")
    parser.add_argument("--y-range", type=float, nargs=2, default=(-50.0, 50.0), metavar=("MIN", "MAX"),
                        help="Ego Y display range in meters (+Y is left)")
    parser.add_argument("--pixels-per-meter", type=int, default=9)
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Maximum plotted points after ROI filtering; deterministic thinning")
    args = parser.parse_args()

    if args.label_file:
        label_path = args.label_file
        lidar_path = args.lidar_file or _find_lidar(args.raw_root, label_path.stem)
    elif args.lidar_file:
        lidar_path = args.lidar_file
        matches = list(args.label_root.rglob(f"{lidar_path.stem}.txt"))
        if len(matches) > 1 or (not matches and not args.allow_missing_labels):
            parser.error(f"Expected exactly one label for {lidar_path.stem}, found {len(matches)}")
        label_path = matches[0] if matches else None
    else:
        lidar_path, label_path = _find_pair(
            args.raw_root, args.label_root, args.timestamp, args.allow_missing_labels
        )

    if not lidar_path.is_file():
        parser.error(f"LiDAR file does not exist: {lidar_path}")
    if label_path is not None and not label_path.is_file():
        parser.error(f"Label file does not exist: {label_path}")
    box_count, point_count = _render(
        lidar_path, label_path, args.output,
        tuple(args.x_range), tuple(args.y_range), args.pixels_per_meter, args.max_points,
    )
    print(f"LiDAR   : {lidar_path}")
    print(f"Labels  : {label_path if label_path else 'NONE (timestamp has no annotation file)'}")
    print(f"BEV PNG : {args.output.resolve()}")
    print(f"Box CSV : {args.output.with_suffix('.csv').resolve()}")
    print(f"Boxes   : {box_count}; LiDAR points in displayed ROI: {point_count}")


if __name__ == "__main__":
    main()
