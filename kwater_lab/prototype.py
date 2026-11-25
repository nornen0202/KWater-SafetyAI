"""Classify a photo locally and construct a validated K-water v1.0 row.

    python -m kwater_lab.prototype
    python -m kwater_lab.prototype --input-dir /input --output-dir /output --cache MODEL_CACHE

Default mode runs the organizer's public illustration and saves experiment evidence.
Directory mode reads evaluation IDs and produces submission.jsonl using the same inference.
The local runner requires the Qwen GGUFs and llama-server prepared by visual_probe.
"""

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .metrics import HEX_ID, LIKELIHOOD, RISK_BANDS, SEVERITY, decode_target, unique_keys
from .visual_probe import (
    CONDITIONS, DEFAULT_CACHE, FILES, MODEL_REPO, MODEL_REV, RUNTIME, SCHEMA,
    digest, encoded_image, generate, local_server, run_trial,
)


PROMPT = CONDITIONS["evidence_gate"] + (
    "\nscene_observation에 사진에서 확인한 영역과 상태를 짧게 기록하세요. 위험이 없다고 판단했다면 그 근거를 "
    "이 필드에 적으세요. 각 위험에는 prevention으로 구체적인 예방조치 한 문장을 쓰세요. "
    "likelihood에 노출과 방호 상태를 고려한 가능성(1 최하, 2 하, 3 중, 4 상, 5 최상), "
    "severity에 예상 결과의 중대성(1 소, 2 중, 3 대, 4 최대)을 숫자로 넣으세요."
)
ITEM = {
    "code": SCHEMA["properties"]["hazards"]["items"]["properties"]["code"],
    "evidence": {"type": "string"}, "prevention": {"type": "string"},
    "likelihood": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "severity": {"type": "integer", "enum": [1, 2, 3, 4]},
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_observation": {"type": "string"},
        "hazards": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "properties": ITEM, "required": list(ITEM), "additionalProperties": False,
        }},
    },
    "required": ["scene_observation", "hazards"], "additionalProperties": False,
}

ELECTRICAL_PREVENTION = "해당 구역의 작업과 접근을 제한하고, 자격을 갖춘 전기 담당자에게 현장 확인을 요청한다."

# Written after inspecting the first two runs. This is a diagnostic retest only.
BOUNDARY_PROMPT = CONDITIONS["evidence_gate"] + """
분류 전에 아래 코드별 관측 기준을 적용하세요.
caught_in_machine: 회전·왕복 운동부나 협착점에 사람 또는 물체가 닿을 노출 경로가 보이는지 확인하세요.
chemical_exposure: 화학물질의 표시, 유출 또는 비산과 작업자에게 노출될 경로가 보이는지 확인하세요.
confined_space: 출입이 제한된 설비 내부나 밀폐 장소에서 사람이 진입 또는 작업하는 정황이 보이는지 확인하세요.
drowning_inundation: 수면 또는 유입되는 물과 작업 위치의 관계에서 빠지거나 잠길 가능성이 보이는지 확인하세요.
electrical_energy: 노출 충전부, 손상된 절연, 전기설비와 인체의 접촉 경로 등 감전을 뒷받침하는 상태가 보이는지 확인하세요.
fall_height: 높은 작업면에서 낮은 면으로 떨어질 수 있는 가장자리·개구부 등 높이차와 방호 상태를 확인하세요.
fire_explosion: 불꽃·고온 점화원과 가연 조건, 혹은 압력 용기의 손상 등 발화·파열 조건이 보이는지 확인하세요.
other: 목록에 속하지 않는 별개의 구체적 위해 요인과 노출 경로가 보이는지 확인하세요.
slip_trip: 같은 이동면에서 사람을 걸려 넘어지거나 미끄러지게 할 장애물, 표면 상태가 보이는지 확인하세요.
struck_by_lifting: 인양·적재·비래 물체의 예상 낙하 또는 이동 경로와 그 아래나 주변 노출이 보이는지 확인하세요.
thermal_weather: 폭염·한랭·기상 노출을 설명할 작업 조건과 보호 상태가 보이는지 확인하세요."""


def assemble(image_id, content):
    hazards = {}
    for item in content["hazards"]:
        code = item["code"]
        if code in hazards:
            raise ValueError(f"Duplicate model hazard: {code}")
        likelihood, severity = item["likelihood"], item["severity"]
        if (type(likelihood) is not int or not 1 <= likelihood <= len(LIKELIHOOD)
                or type(severity) is not int or not 1 <= severity <= len(SEVERITY)):
            raise ValueError("Invalid model risk score")
        product = likelihood * severity
        grade = next(label for maximum, label in RISK_BANDS if product <= maximum)
        hazards[code] = {
            "evidence": item["evidence"],
            "prevention": ELECTRICAL_PREVENTION if code == "electrical_energy" else item["prevention"],
            "references": [],
            "risk_level": {
                "likelihood": {"score": likelihood, "grade": LIKELIHOOD[likelihood - 1]},
                "severity": {"score": severity, "grade": SEVERITY[severity - 1]},
                "risk": {"score": product, "grade": grade},
            },
        }
    result = {"schema_version": "1.0", "image_id": image_id,
              "status": "hazard" if hazards else "no_visible_hazard", "hazards": hazards}
    if not hazards:
        result["scene_evidence"] = content["scene_observation"]
    decode_target(result)
    return result


def infer(port, key, image_id, path):
    result = generate(port, key, encoded_image(path), PROMPT, RESPONSE_SCHEMA, 640)
    if result["finish_reason"] != "stop":
        raise ValueError(f"Model output did not finish for {image_id}: {result['finish_reason']}")
    raw = json.loads(result["raw_output"], object_pairs_hook=unique_keys)
    return assemble(image_id, raw), result


def manifest_images(input_dir):
    seen = set()
    with (input_dir / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            image_id = json.loads(line, object_pairs_hook=unique_keys)["image_id"]
            if not isinstance(image_id, str) or not HEX_ID.fullmatch(image_id) or image_id in seen:
                raise ValueError(f"Invalid or duplicate manifest image_id: {image_id}")
            seen.add(image_id)
            yield image_id, input_dir / "images" / f"{image_id}.jpg"
    if not seen:
        raise ValueError("Empty input manifest")


def write_predictions(input_dir, output_dir, port, key):
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=output_dir, mode="w", encoding="utf-8",
                                            newline="\n", delete=False)
    try:
        with temporary as stream:
            for image_id, path in manifest_images(input_dir):
                row, _ = infer(port, key, image_id, path)
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        Path(temporary.name).replace(output_dir / "submission.jsonl")
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("results/prototype_controlled.json"))
    parser.add_argument("--taxonomy-followup", type=Path, metavar="REPORT",
                        help="also run the post-hoc code-definition retest on the public example")
    args = parser.parse_args()
    if bool(args.input_dir) != bool(args.output_dir):
        parser.error("--input-dir and --output-dir are required together")
    if args.input_dir and args.taxonomy_followup:
        parser.error("--taxonomy-followup uses the public illustration only")
    cache = args.cache.resolve()
    with local_server(cache, cache / "prototype-server.log") as (port, key, startup):
        if args.input_dir:
            write_predictions(args.input_dir, args.output_dir, port, key)
            return
        path = cache / "public-example.jpg"
        image_hash = digest(path)
        if image_hash != FILES[path.name]:
            raise ValueError("Public sample checksum changed")
        # A local evaluation-style ID; the model receives only the JPEG and prompt.
        row, trace = infer(port, key, image_hash[:32], path)
        if args.taxonomy_followup:
            followup = run_trial(port, key, encoded_image(path), BOUNDARY_PROMPT)
    label = json.loads((cache / "public-example-label.json").read_text(encoding="utf-8"))
    expected = set(label["target"]["hazards"])
    predicted = set(row["hazards"])
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Local prototype on the organizer's public training illustration",
        "image_sha256": image_hash, "sample_id": label["sample_id"], "model": MODEL_REPO,
        "revision": MODEL_REV, "runtime": RUNTIME, "official_competition_lora": False,
        "official_law_corpus_used": False, "references_policy": "Empty pending access to official corpus.",
        "prevention_policy": "Use a fixed restricted-access and qualified-review action for electrical_energy.",
        "validation": "Passed local kwater_lab.metrics.decode_target v1.0 schema checks",
        "expected_codes": sorted(expected), "matched": sorted(expected & predicted),
        "missed": sorted(expected - predicted), "extra": sorted(predicted - expected),
        "generated_row": row, "model_response": trace, "max_tokens": 640, "seed": 42, **startup,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Schema validated.", row["status"], sorted(predicted), trace["elapsed_s"], "s")
    if args.taxonomy_followup:
        followup_report = {
            "experiment": "taxonomy_boundaries", "post_hoc": True,
            "design": "Prompt drafted after comparing both first-run predictions to the public gold label.",
            "scope": "Diagnostic retest on the same public illustration. No generalization estimate.",
            "image_sha256": image_hash, "sample_id": label["sample_id"],
            "model": MODEL_REPO, "revision": MODEL_REV, "runtime": RUNTIME, "official_competition_lora": False,
            "seed": 42, "max_tokens": 384, "temperature": 0, "expected_codes": sorted(expected),
            "condition": followup,
        }
        if followup["valid"]:
            guesses = set(followup["predicted_codes"])
            followup_report.update(matched=sorted(expected & guesses), missed=sorted(expected - guesses),
                                   extra=sorted(guesses - expected))
        args.taxonomy_followup.parent.mkdir(parents=True, exist_ok=True)
        args.taxonomy_followup.write_text(json.dumps(followup_report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        print("Post-hoc taxonomy retest.", followup["predicted_codes"], followup["elapsed_s"], "s")


if __name__ == "__main__":
    main()
