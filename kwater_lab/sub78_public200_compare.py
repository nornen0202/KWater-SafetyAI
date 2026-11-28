"""Check and compare same-panel official-kit outputs with an optional local photo review."""
import argparse
import hashlib
from html import escape
from itertools import combinations
import json
import os
from pathlib import Path
import sys

from .fold0_production_compare import image_id
from .hybrid_public200_timing import DEFAULT_MANIFEST, check_panel_files, panel, sha256_file
from .metrics import read_index
from .offline_archive_audit import run_output_checker


SEED = "sub78-public200-review-20261128"


def codes(row):
    return frozenset(row["hazards"])


def severity4_codes(hazards):
    return {code for code, hazard in hazards.items() if hazard["risk_level"]["severity"]["score"] == 4}


def category(before, after):
    if before == after:
        return "same_code_set"
    if not before:
        return "empty_to_hazard"
    if not after:
        return "hazard_to_empty"
    return "hazard_code_changed"


def photo_grid(destination, cases, font_name):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from textwrap import wrap

    if not cases:
        raise ValueError("No changed photos for the grid")
    text_blocks = []
    for index, (_, official, predictions) in enumerate(cases):
        rows = [("공식", official), *((name + "차", codes) for name, codes in predictions.items())]
        lines = [f"{index + 1:02d}"]
        for name, found in rows:
            lines.extend(wrap(f"{name}   {', '.join(sorted(found)) or 'no_visible_hazard'}", 54))
        text_blocks.append("\n".join(lines))
    cell_width, pad = 800, 24
    cell_height = 432 + 38 * max(text.count("\n") + 1 for text in text_blocks)
    width = 2 * cell_width + 3 * pad
    height = 100 + ((len(cases) + 1) // 2) * (cell_height + pad)
    canvas = Image.new("RGB", (width, height), "#eef2f6")
    draw = ImageDraw.Draw(canvas)
    header = ImageFont.truetype(str(font_name), 34)
    body = ImageFont.truetype(str(font_name), 23)
    draw.text((pad, pad), "같은 사진의 6·7·8차 판정과 공식 주석", font=header, fill="#192d40")
    for index, ((photo, _, _), text) in enumerate(zip(cases, text_blocks)):
        x = pad + (index % 2) * (cell_width + pad)
        y = 86 + (index // 2) * (cell_height + pad)
        draw.rectangle((x, y, x + cell_width, y + cell_height), fill="white")
        with Image.open(photo) as original:
            thumb = ImageOps.contain(ImageOps.exif_transpose(original).convert("RGB"), (cell_width - 24, 378))
        canvas.paste(thumb, (x + (cell_width - thumb.width) // 2, y + (400 - thumb.height) // 2))
        draw.multiline_text((x + 16, y + 404), text, font=body, fill="#192d40", spacing=4)
    canvas.save(destination)

def local_review(path, groups, outputs, labels, input_dir, font_name):
    if not path.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Private photo reviews must be written below data/")
    parts = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>200장 판정 대조</title>",
             "<style>body{font:18px system-ui;margin:2rem auto;max-width:1100px}img{max-width:100%;max-height:460px}"
             "article{border-top:1px solid #999;padding:1rem}table{border-collapse:collapse;width:100%}"
             "td,th{border:1px solid #bbb;padding:.6rem;vertical-align:top;text-align:left}</style>",
             "<h1>같은 사진에서 달라진 판정</h1><p>개발 패널 일부입니다. 정확도나 리더보드 점수로 해석할 수 없습니다.</p>"]
    headings = {"empty_to_hazard": "정상 → 위험", "hazard_to_empty": "위험 → 정상",
                "hazard_code_changed": "위험 코드 변경"}
    cases = []
    n = 0
    for kind, rows in groups.items():
        for row in rows:
            n += 1
            sid = row["sample_id"]
            iid = image_id(sid)
            img = input_dir / "images" / f"{iid}{Path(row['image']).suffix}"
            cases.append((img, labels[sid], {name: codes(pred[iid]) for name, pred in outputs.items()}))
            url = Path(os.path.relpath(img, path.parent)).as_posix()
            parts.append(f"<article><h2>{n:02d} · {headings[kind]}</h2><img src='{escape(url, quote=True)}' alt='현장 사진'>")
            parts.append("<p>공식 위험 코드 " + escape(", ".join(sorted(labels[sid])) or "no_visible_hazard") + "</p>")
            parts.append("<table><tr><th>후보</th><th>위험 · 근거 · 예방조치</th></tr>")
            for name, pred in outputs.items():
                hazards = pred[iid]["hazards"]
                lines = [" / ".join([code, str(h.get("evidence", "")), str(h.get("prevention", ""))]) for code, h in hazards.items()]
                detail = "<br>".join(escape(line) for line in lines) if lines else "no_visible_hazard"
                parts.append(f"<tr><td>{escape(name)}</td><td>{detail}</td></tr>")
            parts.append("</table></article>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    photo_grid(path.with_name("review.png"), cases, font_name)


def compare(args):
    witness = json.loads(args.panel_witness.read_bytes())
    check_panel_files(args.input_dir, args.manifest, witness)
    rows = panel(args.manifest)
    ids = [image_id(row["sample_id"]) for row in rows]
    report = {"scope": "Development latency panel; full1045 training overlaps with this panel.",
              "input_manifest_sha256": witness["input_manifest_sha256"], "photos": len(ids),
              "outputs": {}, "comparisons": {}}
    outputs = {}
    for name, path in zip(("6", "7", "8"), (args.sixth, args.seventh, args.eighth)):
        out = read_index(path / "submission.jsonl", lambda row: (row["image_id"], row), limit_size=True)
        checked = run_output_checker(args.kit, args.input_dir, path, Path(sys.executable))
        if list(out) != ids or not checked["all_rows_valid"] or checked["valid_rows"] != len(ids):
            raise ValueError(f"Candidate {name} differs from the exact ordered panel or fails the official checker")
        outputs[name] = out
        report["outputs"][name] = {"official_check": checked, "sha256": sha256_file(path / "submission.jsonl"),
                                   "hazard_photos": sum(bool(codes(row)) for row in out.values()),
                                   "hazard_rows": sum(len(row["hazards"]) for row in out.values())}
    for left, right in combinations(outputs, 2):
        counts = dict.fromkeys(("same_code_set", "empty_to_hazard", "hazard_to_empty", "hazard_code_changed"), 0)
        for iid in ids:
            counts[category(codes(outputs[left][iid]), codes(outputs[right][iid]))] += 1
        report["comparisons"][f"{left}_to_{right}"] = counts
    labels = read_index(args.labels, lambda row: (row["sample_id"], row["target"]["hazards"]))
    gold_by_iid = {image_id(row["sample_id"]): labels[row["sample_id"]] for row in rows}
    severe_by_iid = {iid: severity4_codes(gold) for iid, gold in gold_by_iid.items() if severity4_codes(gold)}
    safe_ids = [iid for iid, gold in gold_by_iid.items() if not gold]
    matched = {arm: {iid for iid, severe in severe_by_iid.items() if codes(pred[iid]) & severe}
               for arm, pred in outputs.items()}
    report["full_panel_label_diagnostics"] = {
        "scope": "Training-exposed development data; descriptive comparison only.",
        "official_no_visible_hazard": len(safe_ids),
        "official_severity4_photos": len(severe_by_iid),
        "arms": {arm: {"safe_false_alarms": sum(bool(codes(pred[iid])) for iid in safe_ids),
                       "severity4_complete_misses": len(severe_by_iid) - len(matched[arm])}
                 for arm, pred in outputs.items()},
        "severity4_code_match_7_to_8": {
            "gains": len(matched["8"] - matched["7"]),
            "losses": len(matched["7"] - matched["8"]),
            "both_match": len(matched["7"] & matched["8"]),
            "neither_matches": len(severe_by_iid) - len(matched["7"] | matched["8"]),
        },
    }
    report["disappeared_hazard_audit"] = {}
    for prior in ("6", "7"):
        counts = dict.fromkeys(("changed_photos", "official_no_visible_hazard", "official_hazard",
                                "official_severity4", "at_least_one_correct_prior_code",
                                "all_prior_codes_wrong_on_official_hazard", "at_least_one_correct_prior_severity4_code"), 0)
        for row in rows:
            iid = image_id(row["sample_id"])
            before = codes(outputs[prior][iid])
            if not before or codes(outputs["8"][iid]):
                continue
            gold = labels[row["sample_id"]]
            severe = severity4_codes(gold)
            counts["changed_photos"] += 1
            counts["official_no_visible_hazard"] += not gold
            counts["official_hazard"] += bool(gold)
            counts["official_severity4"] += bool(severe)
            counts["at_least_one_correct_prior_code"] += bool(before & gold.keys())
            counts["all_prior_codes_wrong_on_official_hazard"] += bool(gold) and not before & gold.keys()
            counts["at_least_one_correct_prior_severity4_code"] += bool(before & severe)
        report["disappeared_hazard_audit"][f"{prior}_to_8"] = counts
    if args.review:
        groups = {key: [] for key in ("empty_to_hazard", "hazard_to_empty", "hazard_code_changed")}
        order = sorted(rows, key=lambda row: hashlib.sha256(f"{SEED}/{row['sample_id']}".encode()).digest())
        seen_groups = set()
        for row in order:
            iid = image_id(row["sample_id"])
            key = category(codes(outputs["7"][iid]), codes(outputs["8"][iid]))
            if key in groups and len(groups[key]) < 2 and row["group_id"] not in seen_groups:
                groups[key].append(row)
                seen_groups.add(row["group_id"])
        primary_counts = {key: len(values) for key, values in groups.items()}
        for row in order:
            iid = image_id(row["sample_id"])
            key = category(codes(outputs["6"][iid]), codes(outputs["8"][iid]))
            if key in groups and len(groups[key]) < 2 and row["group_id"] not in seen_groups:
                groups[key].append(row)
                seen_groups.add(row["group_id"])
        local_review(args.review, groups, outputs, labels, args.input_dir, args.font)
        report["review_selection"] = {"seed": SEED, "comparison": "7_to_8", "fill_short_categories_using": "6_to_8",
                                      "primary_counts": primary_counts, "at_most_one_photo_per_group": True,
                                      "per_category": {key: len(values) for key, values in groups.items()}}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sixth", "seventh", "eighth", "input-dir", "panel-witness", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--kit", type=Path, default=Path("data/local-kit/kit"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--labels", type=Path, default=Path("data/train/train.jsonl"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args), indent=2) + "\n", encoding="utf-8")
