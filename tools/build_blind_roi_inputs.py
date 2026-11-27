"""Materialize a local blind read and two matched, one-image Qwen manifests."""
import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("data/vision-roi-48")
REVIEW = ROOT / "review"
READ = ROOT / "visual-read-local.txt"
OUT = ROOT / "reviewed-v1"
ADAPTER = Path("data/vlm-11code-fold0-v2/adapter")
PROMPT = Path("kwater_lab/system_prompt_11code.txt")
ROLES = {"o": "object", "c": "counterpart", "b": "barrier"}
REL = {"s": "supported", "u": "unclear", "c": "contradicted"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rect(spec, w, h):
    a, b, c, d = map(int, spec[1:].split(","))
    assert 0 <= a < c <= 100 and 0 <= b < d <= 100
    return ROLES[spec[0]], [round(a*w/100), round(b*h/100), round(c*w/100), round(d*h/100)]


def box_at(cx, cy, w, h):
    cw, ch = round(.60*w), round(.60*h)
    x = max(0, min(w-cw, round(cx-cw/2)))
    y = max(0, min(h-ch, round(cy-ch/2)))
    return (x, y, x+cw, y+ch)


def panel(image):
    return ImageOps.pad(image, (512, 384), Image.Resampling.LANCZOS, color=(96,96,96))


def complete_roi_arm(packet):
    """Add a full-coverage ROI arm. Keep the fixed-scale comparison intact."""
    geometry = json.loads((packet/"geometry-local.json").read_bytes())
    manifests = []
    expanded = 0
    dest = packet/"roi_full"
    dest.mkdir()
    for row in geometry:
        q = row["id"]
        ann = json.loads((packet/(q+".json")).read_bytes())
        x, y, xx, yy = row["roi"]
        for shape in ann["shapes"]:
            (a,b),(c,d) = shape["points"]
            x,y,xx,yy = min(x,a),min(y,b),max(xx,c),max(yy,d)
        crop = (x,y,xx,yy)
        expanded += int(list(crop) != row["roi"])
        with Image.open(REVIEW/(q+".png")) as im:
            canvas = Image.new("RGB",(1024,384))
            canvas.paste(panel(im),(0,0))
            canvas.paste(panel(im.crop(crop)),(512,0))
            target = dest/(q+".png")
            canvas.save(target,compress_level=2)
        manifests.append(dict(sample_id=q,image="roi_full/"+target.name))
        row["roi_full"] = crop
        row["roi_full_sha256"] = digest(target)
    write(packet/"roi_full.json", manifests)
    write(packet/"full-geometry-local.json", geometry)
    audit = json.loads((packet/"aggregate.json").read_bytes())
    audit.update(input_images=audit["input_images"] + len(manifests),
                 expanded_roi_photos=expanded,
                 full_geometry_sha256=digest(packet/"full-geometry-local.json"))
    write(packet/"aggregate.json", audit)
    return expanded


def build(out):
    with (REVIEW / "observations.csv").open(encoding="utf-8-sig") as f:
        queue = list(csv.DictReader(f))
    notes = READ.read_text(encoding="utf-8").splitlines()
    assert len(queue) == len(notes) == 48
    arms = {"roi": [], "fixed": []}
    for arm in arms:
        (out / arm).mkdir()
    counts, roles, digests, geometry = Counter(), Counter(), {}, []
    empty = clipped = identical = 0
    for i, (row, line) in enumerate(zip(queue, notes)):
        q = row["id"]
        assert q == f"R{i+1:02}"
        relation, a, b, obs, reason = line.split("|")
        src = REVIEW / (q+".png")
        with Image.open(src) as im:
            w, h = im.size
            boxes = dict(rect(s, w, h) for s in (a, b) if s)
            row.update(object_visible="visible" if "object" in boxes else "unclear",
                       counterpart_visible="visible" if "counterpart" in boxes else "unclear",
                       relation=REL[relation], observation=obs, uncertainty_reason=reason,
                       reviewer="Codex single blind visual review")
            counts[row["relation"]] += 1
            roles.update(boxes.keys())
            annotation = json.loads((REVIEW/(q+".json")).read_bytes())
            annotation.update(imagePath="../review/"+src.name, shapes=[
                dict(label=label, points=[coords[:2], coords[2:]], group_id=1,
                     shape_type="rectangle", flags={}) for label, coords in boxes.items()])
            write(out/(q+".json"), annotation)
            # Four predetermined corner crops, with no reference to labels or annotations.
            fx, fy = (.3, .7)[i%2]*w, (.3, .7)[(i//2)%2]*h
            fixed = box_at(fx, fy, w, h)
            roi = fixed
            if boxes:
                x0, y0, x1, y1 = zip(*boxes.values())
                roi = box_at((min(x0)+max(x1))/2, (min(y0)+max(y1))/2, w, h)
                clipped += int(min(x0)<roi[0] or min(y0)<roi[1] or max(x1)>roi[2] or max(y1)>roi[3])
            else:
                empty += 1
            context = panel(im)
            for arm, crop in (("fixed",fixed), ("roi",roi)):
                canvas = Image.new("RGB", (1024,384))
                canvas.paste(context,(0,0))
                canvas.paste(panel(im.crop(crop)),(512,0))
                name = f"{arm}/{q}.png"
                canvas.save(out/name, compress_level=2)
                digests[name] = digest(out/name)
                arms[arm].append(dict(sample_id=q, image=name))
            identical += int(digests[f"roi/{q}.png"]==digests[f"fixed/{q}.png"])
            geometry.append(dict(id=q, fixed=fixed, roi=roi, source_sha256=digest(src)))
    with (out/"observations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=queue[0].keys())
        writer.writeheader()
        writer.writerows(queue)
    for arm,v in arms.items():
        write(out/(arm+".json"), v)
    write(out/"geometry-local.json", geometry)
    write(out/"input-sha256-local.json", digests)
    write(out/"aggregate.json", dict(photos=len(queue), relations=counts, boxes=roles,
          fallback_photos=empty, clipped_union_photos=clipped, byte_identical_pairs=identical,
          input_images=len(digests), canvas=[1024,384], model="Qwen3-VL-4B + fold0 v2 118",
          adapter_path=ADAPTER.as_posix(), adapter_sha256=digest(ADAPTER/"adapter_model.safetensors"),
          prompt_sha256=digest(PROMPT), blind_read_sha256=digest(READ),
          input_index_sha256=digest(out/"input-sha256-local.json"), qwen_generations=0))


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"Existing packet preserved: {OUT}")
    with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
        packet = Path(scratch)/"packet"
        packet.mkdir()
        build(packet)
        complete_roi_arm(packet)
        packet.rename(OUT)
