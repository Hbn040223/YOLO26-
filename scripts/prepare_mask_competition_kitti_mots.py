"""Freeze a KITTI Tracking + MOTS near-Car cohort before running inference."""
import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import SEEK_CUR, SEEK_END, SEEK_SET
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"


class RangeArchive:
    """Seekable, bounded HTTP reads for the public ZIP central directory and members."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        with urlopen(Request(url, method="HEAD"), timeout=25) as reply:
            self.size = int(reply.headers["Content-Length"])
        self.start = -1
        self.buffer = b""

    def tell(self):
        return self.position

    def seek(self, offset, whence=SEEK_SET):
        base = {SEEK_SET: 0, SEEK_CUR: self.position, SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, count=-1):
        start = self.position
        end = self.size if count < 0 else min(self.size, start + count)
        if end <= start:
            return b""
        if start < self.start or end > self.start + len(self.buffer):
            last = min(self.size, max(end, start + 65536)) - 1
            request = Request(self.url, headers={"Range": "bytes={}-{}".format(start, last)})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=35) as reply:
                        expected = "bytes {}-{}/{}".format(start, last, self.size)
                        if reply.status != 206 or reply.headers["Content-Range"] != expected:
                            raise ValueError("archive server did not honor the requested byte range")
                        data = reply.read()
                    if len(data) != last - start + 1:
                        raise OSError("incomplete ZIP range")
                    break
                except OSError:
                    if attempt == 2:
                        raise
            self.start, self.buffer = start, data
        self.position = end
        return self.buffer[start - self.start:end - self.start]

    def seekable(self):
        return True


def tracking(archive, seq):
    frames = defaultdict(list)
    for line in archive.read("training/label_02/{}.txt".format(seq)).decode().splitlines():
        frame, track, cls, _, occlusion, _, *v = line.split()
        frames[int(frame)].append((int(track) + 1, cls, int(occlusion), tuple(map(float, v[:4])), float(v[9])))
    return frames


def mots(archive, seq):
    frames = defaultdict(list)
    for line in archive.read("instances_txt/{}.txt".format(seq)).decode().splitlines():
        frame, track, cls, height, width, counts = line.split()
        if cls == "1":
            frames[int(frame)].append((int(track) - 1000 + 1,
                                       {"size": [int(height), int(width)], "counts": counts.encode()}))
    return frames


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def select(tracks, masks, per_sequence):
    used = set()
    selected = []
    candidates = []
    for frame, items in tracks.items():
        cars = [item for item in items if item[1] == "Car" and item[2] < 3 and item[4] > 0]
        ids = {track for track, _ in masks.get(frame, ())}
        near = [x for x in cars if x[4] <= 10 and x[0] in ids]
        if near:
            intersect = sum(any(x[0] != y[0] and y[0] in ids and overlap(x[3], y[3]) for y in cars) for x in near)
            candidates.append((frame, {x[0] for x in near}, intersect))
    while candidates and len(selected) < per_sequence:
        frame, ids, _ = max(candidates, key=lambda x: (len(x[1] - used), x[2], len(x[1]), -x[0]))
        if not ids - used:
            break
        selected.append(frame)
        used |= ids
        candidates = [x for x in candidates if abs(frame - x[0]) >= 25]
    return sorted(selected)


def pixel_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def fetch_side(side, data, keys):
    reader = RangeArchive("{}/data_tracking_image_{}.zip".format(S3, side))
    with ZipFile(reader) as archive:
        for seq, frame in keys:
            name = "training/image_0{}/{}/{:06}.png".format(side, seq, frame)
            output = data / "{}_{:06}_{}.png".format(seq, frame, side)
            if not output.exists():
                output.write_bytes(archive.read(name))
            print("image", side, seq, frame, flush=True)


def make_frame(data, seq, frame, tracks, masks, calib, known):
    left, right = (data / "{}_{:06}_{}.png".format(seq, frame, n) for n in (2, 3))
    hashes = [pixel_hash(left), pixel_hash(right)]
    if any(value in known for value in hashes):
        return None
    with Image.open(left) as image:
        width, height = image.size
    labels = np.zeros((height, width), dtype=np.uint16)
    candidates = dict(masks)
    targets, ignore = [], []
    for track, cls, occlusion, box, depth in tracks:
        x0, y0, x1, y1 = box
        box = [max(0., x0), max(0., y0), min(width, x1), min(height, y1)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if cls != "Car" or occlusion >= 3 or depth <= 0:
            if cls in ("Car", "Van", "DontCare"):
                ignore.append(box)
            continue
        has_mask = track in candidates
        if has_mask:
            mask = coco_mask.decode(candidates[track]).astype(bool)
            y, x = np.where(mask)
            mask_box = [x.min(), y.min(), x.max() + 1, y.max() + 1]
            iw = max(0, min(box[2], mask_box[2]) - max(box[0], mask_box[0]))
            ih = max(0, min(box[3], mask_box[3]) - max(box[1], mask_box[1]))
            coverage = iw * ih / ((mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1]))
            if coverage < .50:
                raise ValueError("MOTS and Tracking Car IDs do not align in {}:{}:{}".format(seq, frame, track))
            labels[mask] = track
        targets.append({"id": track, "left_box": box, "depth_m": depth, "mask_available": has_mask})
    if not any(t["depth_m"] <= 10 and t["mask_available"] for t in targets):
        raise ValueError("selected frame lacks a labeled near Car")
    label_path = data / "{}_{:06}_car.png".format(seq, frame)
    Image.fromarray(labels).save(label_path)
    return {"frame_id": "{}_{:06}".format(seq, frame), "sequence": seq, "left": str(left), "right": str(right),
            "labels": str(label_path), "calibration": calib, "targets": targets, "ignore_boxes": ignore,
            "decoded_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("kitti2015", type=Path)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--per-sequence", type=int, default=3)
    args = parser.parse_args()
    source = args.directory
    data = source / "cohort"
    data.mkdir(exist_ok=True)
    definitions = {}
    with ZipFile(source / "data_tracking_label_2.zip") as label_archive, ZipFile(source / "instances_txt.zip") as mask_archive, ZipFile(source / "data_tracking_calib.zip") as camera_archive:
        for seq in args.sequences:
            tracks, masks = tracking(label_archive, seq), mots(mask_archive, seq)
            camera = {line.split()[0].rstrip(":"): list(map(float, line.split()[1:]))
                      for line in camera_archive.read("training/calib/{}.txt".format(seq)).decode().splitlines() if line.strip()}
            p2, p3 = camera["P2"], camera["P3"]
            calibration = [p2[0] * (p2[3] / p2[0] - p3[3] / p3[0]), p2[2] - p3[2]]
            for frame in select(tracks, masks, args.per_sequence):
                definitions[seq, frame] = tracks[frame], masks[frame], calibration
    keys = sorted(definitions)
    frozen = data / "selected_frames.json"
    frozen.write_text(json.dumps(keys, indent=2))
    print("selected", len(keys), keys, flush=True)
    if not keys:
        raise ValueError("no close Car frames found")
    pathsets = [sorted((args.kitti2015 / "training" / name).glob("*_10.png")) for name in ("image_2", "image_3")]
    assert [len(paths) for paths in pathsets] == [200, 200]
    with ThreadPoolExecutor(3) as executor:
        download = [executor.submit(fetch_side, side, data, keys) for side in (2, 3)]
        known = set(executor.map(pixel_hash, pathsets[0] + pathsets[1]))
        for item in download:
            item.result()
    manifest = []
    for key in keys:
        row = make_frame(data, *key, *definitions[key], known)
        if row is None:
            print("excluded exact overlap", key, flush=True)
        else:
            manifest.append(row)
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print("manifest", len(manifest), flush=True)


if __name__ == "__main__":
    main()
