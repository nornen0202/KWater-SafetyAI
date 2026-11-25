"""Local K-water v1.0 hazard-code evaluation (standard library only).

    python -m kwater_lab.metrics --gold dev.jsonl --pred submission.jsonl
    python -m kwater_lab.metrics --gold dev.jsonl --id-map ids.jsonl --pred a.jsonl \
        --compare b.jsonl --groups groups.jsonl --seed 42 --resamples 2000

Gold can be train.jsonl rows (nested target, original sample IDs) or flat targets
with evaluation IDs. The optional ID map has sample_id and image_id (32 lowercase
hex). Groups have image_id (in prediction ID space) and group_id. Each file must
cover exactly the evaluated photos. Input errors abort local comparisons. The
organizer's per-row invalidation, LLM grade and private score are not simulated.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


CODES = (
    "fall_height", "slip_trip", "caught_in_machine", "struck_by_lifting",
    "electrical_energy", "drowning_inundation", "fire_explosion", "confined_space",
    "chemical_exposure", "thermal_weather", "other",
)
HEX_ID = re.compile(r"[0-9a-f]{32}")
LIKELIHOOD = ("최하", "하", "중", "상", "최상")
SEVERITY = ("소", "중", "대", "최대")
RISK_BANDS = ((3, "매우낮음"), (6, "낮음"), (8, "보통"), (12, "약간높음"),
              (15, "높음"), (20, "매우높음"))


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_index(path, decode, *, limit_size=False):
    index = {}
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                if limit_size and len(line.removesuffix(b"\n").removesuffix(b"\r")) > 2**20:
                    raise ValueError("row exceeds 1 MiB")
                row = json.loads(line.decode("utf-8"), object_pairs_hook=unique_keys,
                                 parse_constant=reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                key, value = decode(row)
                if key in index:
                    raise ValueError(f"duplicate ID: {key}")
                index[key] = value
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not index:
        raise ValueError(f"empty file: {path}")
    return index


def check_keys(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{name} must have exactly {sorted(expected)}")


def check_text(value, name, limit=None):
    if not isinstance(value, str) or not value.strip() or (limit and len(value) > limit):
        raise ValueError(f"invalid {name}")
    if limit and (not value.replace("\ufeff", "").strip()
                  or re.search(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]", value)):
        raise ValueError(f"forbidden control character or blank {name}")
    return value


def check_level(obj):
    check_keys(obj, ("likelihood", "severity", "risk"), "risk_level")
    for name in obj:
        check_keys(obj[name], ("grade", "score"), name)
        if type(obj[name]["score"]) is not int:
            raise ValueError(f"{name}.score must be an integer")
    for name, grades in (("likelihood", LIKELIHOOD), ("severity", SEVERITY)):
        score = obj[name]["score"]
        if not 1 <= score <= len(grades) or obj[name]["grade"] != grades[score - 1]:
            raise ValueError(f"inconsistent {name}")
    product = obj["likelihood"]["score"] * obj["severity"]["score"]
    grade = next(grade for maximum, grade in RISK_BANDS if product <= maximum)
    if obj["risk"] != {"grade": grade, "score": product}:
        raise ValueError("risk grade or product mismatch")


def decode_target(row, *, local_ids=False):
    required = {"schema_version", "image_id", "status", "hazards"}
    if row.get("status") == "no_visible_hazard":
        required.add("scene_evidence")
    check_keys(row, required, "target")
    image_id = check_text(row["image_id"], "image_id")
    if not local_ids and not HEX_ID.fullmatch(image_id):
        raise ValueError(f"expected 32 lowercase hexadecimal digits: {image_id}")
    hazards = row["hazards"]
    if row["schema_version"] != "1.0" or not isinstance(hazards, dict):
        raise ValueError("expected schema_version 1.0 and hazards object")
    if row["status"] == "no_visible_hazard":
        check_text(row["scene_evidence"], "scene_evidence", 240)
        if hazards:
            raise ValueError("no_visible_hazard requires empty hazards")
    elif row["status"] != "hazard" or not 1 <= len(hazards) <= 3:
        raise ValueError("expected hazard with 1..3 codes or no_visible_hazard")
    for code, item in hazards.items():
        if code not in CODES:
            raise ValueError(f"unknown hazard: {code}")
        fields = {"evidence", "prevention", "risk_level"}
        if isinstance(item, dict) and "references" in item:
            fields.add("references")
        check_keys(item, fields, code)
        check_text(item["evidence"], "evidence", 240)
        check_text(item["prevention"], "prevention", 480)
        check_level(item["risk_level"])
        refs = item.get("references", [])
        if not isinstance(refs, list) or len(refs) > 3:
            raise ValueError("expected 0..3 references")
        for ref in refs:
            check_keys(ref, ("kind", "id"), "reference")
            if ref["kind"] not in ("case", "catalog", "law"):
                raise ValueError("unknown reference kind")
            ref_id = check_text(ref["id"], "reference id")
            patterns = {"law": r"OSH-[0-9]+(?:-[0-9]+)?", "case": r"B[0-9]{13}/[A-Za-z0-9]{20}",
                        "catalog": r"[CP][0-9a-f]{14}"}
            if not re.fullmatch(patterns[ref["kind"]], ref_id):
                raise ValueError(f"invalid {ref['kind']} ID: {ref_id}")
    return image_id, frozenset(hazards)


def decode_gold(row):
    if "target" not in row:
        return decode_target(row)
    if not isinstance(row["target"], dict):
        raise ValueError("target must be an object")
    image_id, codes = decode_target(row["target"], local_ids=True)
    if row.get("sample_id", image_id) != image_id:
        raise ValueError("sample_id differs from target.image_id")
    return image_id, codes


def same_ids(expected, actual, name):
    missing = expected.keys() - actual.keys()
    extra = actual.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"{name} ID mismatch: missing={len(missing)} {sorted(missing)[:3]}, "
                         f"extra={len(extra)} {sorted(extra)[:3]}")


def load_gold(path, id_map=None):
    gold = read_index(path, decode_gold)
    if id_map is None:
        return gold
    mapping = read_index(id_map, lambda row: (check_text(row["sample_id"], "sample_id"),
                                              check_text(row["image_id"], "image_id")))
    same_ids(gold, mapping, "ID map")
    if len(mapping) != len(set(mapping.values())) or not all(HEX_ID.fullmatch(v) for v in mapping.values()):
        raise ValueError("ID map destinations must be unique 32-digit lowercase hexadecimal IDs")
    return {mapping[key]: codes for key, codes in gold.items()}


def load_predictions(path, gold):
    # Official training rows may retain sample IDs such as S16C159. Exact global
    # ID matching still applies when both sides use these local IDs.
    local_ids = any(not HEX_ID.fullmatch(key) for key in gold)
    predictions = read_index(path, lambda row: decode_target(row, local_ids=local_ids),
                             limit_size=True)
    same_ids(gold, predictions, "predictions")
    return predictions


def ratio(num, den):
    return num / den if den else None


def counts(ids, gold, pred):
    actual = Counter()
    predicted = Counter()
    hits = Counter()
    for key in ids:
        actual.update(gold[key])
        predicted.update(pred[key])
        hits.update(gold[key] & pred[key])
    return actual, predicted, hits


def macro_f1(actual, predicted, hits):
    active = [code for code in CODES if actual[code] or predicted[code]]
    return mean(2 * hits[c] / (actual[c] + predicted[c]) for c in active) if active else None


def evaluate(gold, predictions, groups=None):
    """Score ID -> frozenset(code) mappings from validated rows. Undefined rates are None."""
    if not gold:
        raise ValueError("gold is empty")
    same_ids(gold, predictions, "predictions")
    if groups is not None:
        same_ids(gold, groups, "groups")
    actual, predicted, hits = counts(gold, gold, predictions)
    f1 = macro_f1(actual, predicted, hits)
    per_code = {}
    for code in CODES:
        record = {"support": actual[code], "predicted": predicted[code], "tp": hits[code],
                  "fp": predicted[code] - hits[code], "fn": actual[code] - hits[code],
                  "f1": ratio(2 * hits[code], actual[code] + predicted[code])}
        if groups is not None:
            record["support_groups"] = len({groups[k] for k in gold if code in gold[k]})
        per_code[code] = record
    negatives = sum(not v for v in gold.values())
    positives = len(gold) - negatives
    false_alarms = sum(not gold[k] and bool(predictions[k]) for k in gold)
    full_misses = sum(bool(gold[k]) and not predictions[k] for k in gold)
    return {
        "photos": len(gold), "active_codes": sum(row["f1"] is not None for row in per_code.values()),
        "macro_f1": f1, "quantitative_points_out_of_70": 70*f1 if f1 is not None else None,
        "exact_set_matches": sum(gold[k] == predictions[k] for k in gold),
        "no_hazard_photos": negatives, "no_hazard_false_alarms": false_alarms,
        "no_hazard_false_alarm_rate": ratio(false_alarms, negatives),
        "hazard_photos": positives, "hazard_completely_missed": full_misses,
        "hazard_predicted_empty_rate": ratio(full_misses, positives), "per_code": per_code,
    }


def percentile(sorted_values, fraction):
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def paired_bootstrap(gold, first, second, groups, *, resamples=2000, seed=42):
    """Bootstrap whole cases with replacement. Repeat all photos for every case draw."""
    for name, mapping in (("A", first), ("B", second), ("groups", groups)):
        same_ids(gold, mapping, name)
    members = defaultdict(list)
    for key in sorted(gold):
        members[groups[key]].append(key)
    cases = sorted(members)
    if resamples < 1 or len(cases) < 2:
        raise ValueError("paired bootstrap requires positive resamples and at least two groups")
    generator = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [key for group in generator.choices(cases, k=len(cases)) for key in members[group]]
        a = macro_f1(*counts(sample, gold, first))
        b = macro_f1(*counts(sample, gold, second))
        if a is not None and b is not None:
            deltas.append(b - a)
    deltas.sort()
    a = macro_f1(*counts(gold, gold, first))
    b = macro_f1(*counts(gold, gold, second))
    return {"groups": len(cases), "seed": seed, "resamples": resamples,
            "defined_resamples": len(deltas), "undefined_resamples": resamples - len(deltas),
            "macro_f1_delta_b_minus_a": b - a if a is not None and b is not None else None,
            "percentile_95_ci": [percentile(deltas, q) for q in (0.025, 0.975)] if deltas else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--compare", type=Path, help="candidate B for paired comparison")
    parser.add_argument("--groups", type=Path, help="JSONL with image_id and group_id")
    parser.add_argument("--id-map", type=Path, help="JSONL with sample_id and evaluation image_id")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.compare and not args.groups:
        parser.error("--compare requires --groups")
    try:
        gold = load_gold(args.gold, args.id_map)
        first = load_predictions(args.pred, gold)
        groups = None
        if args.groups:
            groups = read_index(args.groups, lambda row: (check_text(row["image_id"], "image_id"),
                                                          check_text(row["group_id"], "group_id")))
        result = {"a": evaluate(gold, first, groups)}
        if args.compare:
            second = load_predictions(args.compare, gold)
            result["b"] = evaluate(gold, second, groups)
            result["paired"] = paired_bootstrap(gold, first, second, groups,
                                                resamples=args.resamples, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
