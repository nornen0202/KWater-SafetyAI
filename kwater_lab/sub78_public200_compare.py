"""Check and compare same-panel official-kit outputs with an optional local photo review."""
import argparse
import hashlib
from html import escape
from itertools import combinations
import json
import os
from pathlib import Path
import re
import sys

from .fold0_production_compare import image_id
from .hybrid_public200_timing import DEFAULT_MANIFEST, check_panel_files, panel, sha256_file
from .metrics import read_index
from .offline_archive_audit import run_output_checker


SEED = "sub78-public200-review-20261128"
SEED_8B = "sub78-public200-8b-review-20261128"
DISPLAY_NAMES = {"6": "6차 4B 단독", "7": "7차 4B 하이브리드",
                 "8": "전량 4B 단독 후보", "8B": "8B 단독 후보"}


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


def match_changes(before, after, total):
    return {"gains": len(after - before), "losses": len(before - after),
            "both_match": len(before & after), "neither_matches": total - len(before | after)}


def research_8b_receipt(args, output_sha):
    proof = json.loads(args.research_8b_archive_proof.read_bytes())
    observed_shas = re.findall(r"\b[a-fA-F0-9]{64}\b", args.research_8b_remote_sha.read_text(encoding="utf-8-sig"))
    if len(observed_shas) != 1 or proof["sha256"] != observed_shas[0].lower():
        raise ValueError("Seoul reconstructed archive differs from the locally verified research ZIP")
    remote_sha = observed_shas[0].lower()
    log = args.research_8b_kit_log.read_text(encoding="utf-8")
    clocks = re.findall(r"종료 코드 (\d+), 실행 (\d+)초 \+ 설치 (\d+)초 = (\d+)초 / 상한 (\d+)초", log)
    checked = re.findall(r"^유효 결과\s+(\d+)/(\d+)\s*$", log, re.MULTILINE)
    result = re.findall(r"^RESULT (\w+)", log, re.MULTILINE)
    telemetry = re.findall(r"^MAIN_PY_TELEMETRY (\{.*\})$", log, re.MULTILINE)
    if len(clocks) != 1 or checked != [("200", "200")] or result != ["OK"] or len(telemetry) != 1:
        raise ValueError("Expected the official 200-photo Seoul research-8B kit success receipt")
    returned, execution, install, total, limit = map(int, clocks[0])
    generation = json.loads(telemetry[0])
    if (returned or total != execution + install or total > limit or limit != 1800
        or "네트워크 차단" not in log or "이 컴퓨터에서는 보안 제한 없이" in log
        or generation["input_photos"] != 200 or generation["output_submission_sha256"] != output_sha
        or generation["enable_hybrid_booster"] is not False):
        raise ValueError("Official kit clock, sandbox, or output fingerprint did not reconcile")
    return {"archive_sha256": remote_sha, "kit_log_sha256": sha256_file(args.research_8b_kit_log),
            "execution_seconds": execution, "install_seconds": install, "total_seconds": total,
            "limit_seconds": limit, "raw_vlm_parsed_count": generation["raw_vlm_parsed_count"]}


def research_8b_overlap(args, rows, output, seventh, labels):
    manifest = json.loads(args.manifest.read_bytes())
    train = [r for r in manifest if r["split"] == "dev" and r["dev_fold"] != 4]
    primary = [r for r in manifest if r["split"] == "dev" and r["dev_fold"] == 4]
    mask = json.loads(args.research_8b_fold4_mask.read_bytes())
    if mask["canonical_manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Fold 4 mask belongs to a different manifest")
    excluded = {row["fold4_group_id"] for row in mask["confirmed_cross_group_links"]}
    sensitivity = [r for r in primary if r["group_id"] not in excluded]
    train_ids = {r["sample_id"] for r in train}
    train_groups = {r["group_id"] for r in train}
    known = read_index(args.research_8b_fold4_original, lambda row: (row["image_id"], row), limit_size=True)
    if len(train_ids) != 791 or len(primary) != 175 or len(sensitivity) != 107 or set(known) != {r["sample_id"] for r in primary}:
        raise ValueError("Expected exact prior Fold 4 prediction coverage and 791/175/107 split geometry")

    def fold_overlap(cohort):
        members = {r["sample_id"] for r in cohort}
        overlap = [r for r in rows if r["sample_id"] in members]
        changed = [r for r in overlap if codes(known[r["sample_id"]]) != codes(output[image_id(r["sample_id"])])]
        overlap_ids = [r["sample_id"] for r in overlap]
        status_changed = sum(bool(codes(known[sid])) != bool(codes(output[image_id(sid)])) for sid in overlap_ids)
        safe = [sid for sid in overlap_ids if not labels[sid]]
        severe = {sid: severity4_codes(labels[sid]) for sid in overlap_ids if severity4_codes(labels[sid])}
        safe_matches = [{sid for sid in safe if not codes(pred[image_id(sid)])} for pred in (seventh, output)]
        severe_matches = [{sid for sid, gold in severe.items() if codes(pred[image_id(sid)]) & gold}
                          for pred in (seventh, output)]
        seven_changes = sum(codes(seventh[image_id(sid)]) != codes(output[image_id(sid)]) for sid in overlap_ids)
        return {
            "photos": len(overlap), "groups": len({r["group_id"] for r in overlap}),
            "eval1_to_packaged_code_sets_changed": len(changed), "eval1_to_packaged_status_changed": status_changed,
            "changed_groups": len({r["group_id"] for r in changed}),
            "seventh_full1045_exposed_to_these_photos": True,
            "seventh_to_packaged_8B_code_sets_changed": seven_changes,
            "official_no_visible_hazard": len(safe), "official_severity4_photos": len(severe),
            "seventh_to_8B_correct_no_alarm": match_changes(*safe_matches, len(safe)),
            "seventh_to_8B_severity4_match": match_changes(*severe_matches, len(severe)),
        }

    return {
        "scope": "Exact training and Fold 4 overlap on the development timing panel.",
        "trained_photos": sum(r["sample_id"] in train_ids for r in rows),
        "trained_groups": len({r["group_id"] for r in rows if r["group_id"] in train_groups}),
        "fold4_175_overlap": fold_overlap(primary), "fold4_107_overlap": fold_overlap(sensitivity),
        "original_fold4_prediction_sha256": sha256_file(args.research_8b_fold4_original),
    }


def photo_grid(destination, cases, font_name):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from textwrap import wrap

    if not cases:
        raise ValueError("No changed photos for the grid")
    text_blocks = []
    for index, (_, official, predictions) in enumerate(cases):
        rows = [("공식", official), *((DISPLAY_NAMES[name], codes) for name, codes in predictions.items())]
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
    draw.text((pad, pad), "같은 개발 사진의 모델 판정과 공식 주석", font=header, fill="#192d40")
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
                "hazard_code_changed": "위험 코드 변경",
                "severity4_gained": "중대 4 정답 회복", "severity4_lost": "중대 4 정답 유실",
                "safe_alarm_removed": "정상 오경보 해소", "safe_alarm_added": "정상 오경보 발생"}
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
                parts.append(f"<tr><td>{escape(DISPLAY_NAMES[name])}</td><td>{detail}</td></tr>")
            parts.append("</table></article>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    photo_grid(path.with_name("review.png"), cases, font_name)


def select_research_review(rows, safe_ids, severe_matches, outputs):
    safe_ids = set(safe_ids)
    rules = {
        "severity4_gained": (severe_matches["8B"] - severe_matches["7"], 2),
        "severity4_lost": (severe_matches["7"] - severe_matches["8B"], 2),
        "safe_alarm_removed": ({iid for iid in safe_ids if codes(outputs["7"][iid]) and not codes(outputs["8B"][iid])}, 1),
        "safe_alarm_added": ({iid for iid in safe_ids if codes(outputs["8B"][iid]) and not codes(outputs["7"][iid])}, 1),
    }
    chosen = {name: [] for name in rules}
    seen_groups = set()
    order = sorted(rows, key=lambda row: hashlib.sha256(f"{SEED_8B}/{row['sample_id']}".encode()).digest())
    for name, (eligible, limit) in rules.items():
        for row in order:
            if len(chosen[name]) == limit:
                break
            if image_id(row["sample_id"]) in eligible and row["group_id"] not in seen_groups:
                chosen[name].append(row)
                seen_groups.add(row["group_id"])
    return chosen


def compare(args):
    witness = json.loads(args.panel_witness.read_bytes())
    check_panel_files(args.input_dir, args.manifest, witness)
    rows = panel(args.manifest)
    ids = [image_id(row["sample_id"]) for row in rows]
    report = {"scope": "Development latency panel; full1045 training overlaps with this panel.",
              "input_manifest_sha256": witness["input_manifest_sha256"], "photos": len(ids),
              "groups": len({r["group_id"] for r in rows}),
              "outputs": {}, "comparisons": {}}
    outputs = {}
    paths = {"6": args.sixth, "7": args.seventh, "8": args.eighth}
    if args.research_8b:
        paths["8B"] = args.research_8b
    for name, path in paths.items():
        out = read_index(path / "submission.jsonl", lambda row: (row["image_id"], row), limit_size=True)
        checked = run_output_checker(args.kit, args.input_dir, path, Path(sys.executable))
        if list(out) != ids or not checked["all_rows_valid"] or checked["valid_rows"] != len(ids):
            raise ValueError(f"Candidate {name} differs from the exact ordered panel or fails the official checker")
        outputs[name] = out
        report["outputs"][name] = {"official_check": checked, "sha256": sha256_file(path / "submission.jsonl"),
                                   "hazard_photos": sum(bool(codes(row)) for row in out.values()),
                                   "hazard_rows": sum(len(row["hazards"]) for row in out.values())}
    changed_group_counts = {}
    for left, right in combinations(outputs, 2):
        counts = dict.fromkeys(("same_code_set", "empty_to_hazard", "hazard_to_empty", "hazard_code_changed"), 0)
        changed_groups = set()
        for iid, row in zip(ids, rows):
            change = category(codes(outputs[left][iid]), codes(outputs[right][iid]))
            counts[change] += 1
            if change != "same_code_set":
                changed_groups.add(row["group_id"])
        report["comparisons"][f"{left}_to_{right}"] = counts
        changed_group_counts[f"{left}_to_{right}"] = len(changed_groups)
    report["changed_groups"] = changed_group_counts
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
        "severity4_code_match_7_to_8": match_changes(matched["7"], matched["8"], len(severe_by_iid)),
    }
    if args.research_8b:
        readiness = json.loads(Path("results/sub78-package-readiness-20261128.json").read_bytes())
        kit_history = json.loads(Path("results/seoul-l4-roadmap-experiments-20261128.json").read_bytes())
        sixth_timing = kit_history["accepted_sixth_public200"]
        seventh_timing = kit_history["full1045_seventh_public200"]
        fourth_timing = readiness["fallback_8_full1045_epoch1_standalone"]["official_200_timing"]
        for arm, timing in (("6", sixth_timing), ("7", seventh_timing), ("8", fourth_timing)):
            if timing["output_sha256"] != report["outputs"][arm]["sha256"]:
                raise ValueError(f"The {arm} output differs from its timing receipt")
        report["research_8b"] = {
            "official_kit": research_8b_receipt(args, report["outputs"]["8B"]["sha256"]),
            "overlap": research_8b_overlap(args, rows, outputs["8B"], outputs["7"], labels),
            "severity4_code_match_7_to_8B": match_changes(matched["7"], matched["8B"], len(severe_by_iid)),
            "correct_no_alarm_7_to_8B": match_changes(
                *({iid for iid in safe_ids if not codes(outputs[arm][iid])} for arm in ("7", "8B")), len(safe_ids)
            ),
        }
        report["official_kit_total_seconds"] = {
            "6": sixth_timing["kit_total_seconds"], "7": seventh_timing["kit_clock"]["total_seconds"],
            "8": fourth_timing["official_kit_clock_seconds"]["total"],
            "8B": report["research_8b"]["official_kit"]["total_seconds"],
        }
        report["archive_sha256"] = {
            "6": sixth_timing["archive_sha256"], "7": seventh_timing["archive_sha256"],
            "8": readiness["fallback_8_full1045_epoch1_standalone"]["archive_sha256"],
            "8B": report["research_8b"]["official_kit"]["archive_sha256"],
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
    if args.review and args.research_8b:
        groups = select_research_review(rows, safe_ids, matched, outputs)
        local_review(args.review, groups, outputs, labels, args.input_dir, args.font)
        report["review_selection"] = {
            "seed": SEED_8B, "comparison": "7_to_8B", "at_most_one_photo_per_group": True,
            "per_category": {name: len(selected) for name, selected in groups.items()},
        }
    elif args.review:
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
    parser.add_argument("--research-8b", type=Path, help="Recovered official output directory for the research 8B archive")
    parser.add_argument("--research-8b-kit-log", type=Path)
    parser.add_argument("--research-8b-remote-sha", type=Path)
    parser.add_argument("--research-8b-archive-proof", type=Path,
                        default=Path("results/fold4-8b-epoch1-research-package-20261128.json"))
    parser.add_argument("--research-8b-fold4-original", type=Path,
                        default=Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1/predictions.jsonl"))
    parser.add_argument("--research-8b-fold4-mask", type=Path,
                        default=Path("data/fold4-crosssplit-visual-review/fold4-group-exclusions-local.json"))
    args = parser.parse_args()
    if args.research_8b and not (args.research_8b_kit_log and args.research_8b_remote_sha):
        parser.error("Research 8B requires the recovered kit log and independently measured remote archive SHA")
    args.out.write_text(json.dumps(compare(args), indent=2) + "\n", encoding="utf-8")
