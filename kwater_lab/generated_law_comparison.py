"""Compare four law selectors on completed, locally stored model generations."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .adapter_scale_pilot import load_entry, sample
from .attach_law import PINNED, choose_examples
from .law_examples import law_ids, select_articles, training_examples
from .law_retrieval import ArticleIndex, article_texts, official_ranker_type, official_ranking
from .metrics import CODES, decode_target, unique_keys
from .observation_pilot import write_aggregate
from .prompt_pilot import ARMS, digest, parse_generation
from .train_head import file_hash
from .visual_probe import DEFAULT_CACHE, FILES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def raw_scale_cases(rows, lookup, entry):
    if [row["sample_id"] for row in rows] != list(lookup):
        raise ValueError("Raw adapter arm differs from the registered photo order")
    cases = []
    for row in rows:
        pred = row["prediction"]
        fields = {}
        # The official inference writer records a parse failure as an empty object.
        if pred != {}:
            if entry.validate_prediction(pred):
                raise ValueError("Unexpected invalid, nonempty adapter generation")
            for item in pred["hazards"]:
                code = entry.CODE_MAP.get(item["code"])
                if code and code not in fields and len(fields) < 3:
                    fields[code] = {"evidence": entry.clip(item["evidence"], 240),
                                    "prevention": entry.clip(item["prevention"], 480)}
        cases.append((lookup[row["sample_id"]], fields))
    return cases


def load_generations(root):
    example = read(root / "results/prototype.json")
    image = DEFAULT_CACHE / "public-example.jpg"
    target_path = DEFAULT_CACHE / "public-example-label.json"
    sid = example["sample_id"]
    ref_path = root / "results/visual_probe.json"
    raw_path = root / "data/vlm-candidate/generations.json"
    ref, receipt = read(ref_path), read(root / "results/vlm_candidate_public.json")
    if (file_hash(image) != FILES[image.name] or example["image_sha256"] != FILES[image.name]
            or ref["sample_id"] != sid or file_hash(target_path) != ref["label_sha256"]
            or file_hash(ref_path) != receipt["reference_sha256"] or file_hash(raw_path) != receipt["local_raw_sha256"]):
        raise ValueError("Public generation, image or label provenance changed")
    groups = {"public_structured": [(sid, example["generated_row"]["hazards"])]}
    hashes = {"public_structured": file_hash(root / "results/prototype.json"), "public_qwen3": file_hash(ref_path),
              "public_qwen35": file_hash(raw_path)}
    for name, trials in (("public_qwen3", ref["conditions"]), ("public_qwen35", read(raw_path))):
        for arm, record in trials.items():
            if not record["valid"]:
                raise ValueError("Public trial was not a valid code generation")
            items = json.loads(record["raw_output"])["hazards"]
            hazards = {v["code"]: v for v in items}
            if sorted(hazards) != sorted(record["predicted_codes"]) or len(hazards) != len(items):
                raise ValueError("Mismatch between the generated text and the recorded code set")
            groups[name + "_" + arm] = [(sid, hazards)]
    prompt_check = root / "results/prompt_format_replay.json"
    if prompt_check.exists():
        report = read(prompt_check)
        design = read(root / "data/prompt-pilot/design.json")
        signature = digest(design)
        anchors = design["anchors"]
        if signature != report["signature"] or len(anchors) != 6 or tuple(design["arms"]) != ARMS:
            raise ValueError("Unexpected six-photo design")
        folder = root / "data/prompt-pilot" / signature[:12]
        for arm in ARMS:
            check = report["conditions"][arm]
            submission = folder / "format-replay" / arm / "out/submission.jsonl"
            if file_hash(submission) != check["submission_sha256"]:
                raise ValueError("Six-photo submission changed")
            accepted = [json.loads(s, object_pairs_hook=unique_keys) for s in submission.read_text(encoding="utf-8").splitlines()]
            rows, cases, hashes_for_arm = [], [], []
            for i in range(len(anchors)):
                actual = anchors[(i + int(arm == "swapped_11")) % len(anchors)]
                path = folder / f"{i}-{arm}.json"
                record = read(path)
                if record["signature"] != signature or record["image_sha256"] != actual["sha256"]:
                    raise ValueError("Prompt receipt disagrees with the image shown")
                hashes_for_arm.append(file_hash(path))
                _, row = parse_generation(arm, actual["sid"], record["generation"])
                if row is not None:
                    rows.append(row)
                cases.append((actual["sid"], row["hazards"] if row else {}))
            if rows != accepted or len(rows) != check["kit_valid"] or digest(hashes_for_arm) != check["receipts_digest"]:
                raise ValueError("Raw prompt output differs from the kit-checked records")
            name = f"prompt_{arm}"
            groups[name] = cases
            hashes[name + "_raw"] = check["receipts_digest"]
            hashes[name + "_kit"] = check["submission_sha256"]
    scale = root / "results/adapter_scale_pilot.json"
    report = read(scale) if scale.exists() else {}
    local = root / "data/adapter-scale-pilot"
    lookup = {hashlib.sha256(("kwater/scale/" + s).encode()).hexdigest()[:32]: s
              for s, _ in sample(read(root / "data/prefix-pilot/exposure.json"))}
    for factor in ("1.0", "0.75"):
        raw = local / f"generated-scale-{factor}.json"
        measured = report.get("conditions", {}).get(factor)
        if not raw.exists() and measured is None:
            continue
        ref = read(root / "results/official_baseline_10.json")["main_sha256"]
        if file_hash(root / "data/baseline/main.py") != ref or report.get("main_sha256", ref) != ref:
            raise ValueError("Official adapter entry point changed")
        name = f"adapter_scale_{factor}"
        hashes[name + "_raw"] = file_hash(raw)
        if measured and hashes[name + "_raw"] != measured["raw_sha256"]:
            raise ValueError("Recorded raw generation hash does not match")
        groups[name] = raw_scale_cases(read(raw), lookup, load_entry(root / "data/baseline"))
        check = report.get("format_replay", {}).get("conditions", {}).get(factor)
        if check:
            path = local / f"replay-{factor}/out/submission.jsonl"
            if file_hash(path) != check["submission_sha256"] or check["kit_valid"] != len(lookup):
                raise ValueError("Unexpected adapter replay hash or rejected output")
            rows = [json.loads(s, object_pairs_hook=unique_keys) for s in path.read_text(encoding="utf-8").splitlines()]
            actual = [(lookup[decode_target(r)[0]], {c: {k: h[k] for k in ("evidence", "prevention")}
                       for c, h in r["hazards"].items()}) for r in rows]
            if actual != groups[name]:
                raise ValueError("Raw and kit-accepted code sets or wording differ")
            hashes[name + "_kit"] = file_hash(path)
    return groups, hashes, {sid: read(target_path)["target"]["hazards"]}


def rerank(prior, body, examples, actions):
    pool = set(prior[:3] + body[:3] + examples[:3])
    scores = Counter()
    for ranking in (examples, actions):
        for i, article in enumerate(ranking, 1):
            if article in pool:
                scores[article] += 1 / (60 + i)
    return sorted(pool, key=lambda article: (-scores[article], article))


def run(root):
    pilot = root / "data/prefix-pilot"
    paths = {"audit": root / "results/train_audit_full.json", "exposure": pilot / "exposure.json",
             "labels": root / "data/train/train.jsonl", "rag": pilot / "rag.py", "laws": pilot / "laws.jsonl"}
    hashes = {k: file_hash(path) for k, path in paths.items()}
    if hashes != PINNED:
        raise ValueError("Law library or training provenance changed")
    arms, receipts, public = load_generations(root)
    known = read(paths["exposure"])
    inventory = read(paths["audit"])["inventory"]
    needed = set(known["exposed_ids"]) | {s for cases in arms.values() for s, _ in cases}
    with paths["labels"].open(encoding="utf-8") as stream:
        labels = {v["sample_id"]: v["target"]["hazards"] for v in map(json.loads, stream) if v["sample_id"] in needed}
    for sid, hazards in public.items():
        if {c:set(law_ids(h)) for c,h in labels[sid].items()} != {c:set(law_ids(h)) for c,h in hazards.items()}:
            raise ValueError("Public and training reference sets differ")
    laws = [json.loads(line) for line in paths["laws"].read_text(encoding="utf-8").splitlines()]
    active = select_articles(laws)
    index = ArticleIndex(article_texts(paths["laws"]), "bigram")
    ranker_type = official_ranker_type(paths["rag"])
    reviewed = root / "results/train_audit_reviewed.json"
    reviewed_data = read(reviewed) if reviewed.exists() else {}
    groups = reviewed_data.get("assignments", {}) if reviewed_data.get("status") == "ready" else {}
    if groups and (reviewed_data.get("labels_sha256") != hashes["labels"] or set(groups) != set(inventory)):
        raise ValueError("Incomplete or stale reviewed incident assignment")
    libraries, results = {}, {}
    for arm, cases in arms.items():
        counts = Counter()
        methods = {m: Counter(cited=0, joint_top1=0, joint_top3=0, cited_wrong_code=0)
                   for m in ("code_frequency", "article_body", "fold_examples", "prevention_rerank")}
        for sid, predicted in cases:
            ids, _ = choose_examples(known, inventory, [sid], None)
            retained = tuple(s for s in ids if not groups or groups[s]["group_id"] != groups[sid]["group_id"])
            if retained not in libraries:
                examples, freqs = training_examples(retained, labels, active)
                libraries[retained] = ranker_type(laws, train=examples), freqs
            retriever, freqs = libraries[retained]
            counts.update(photos=1, predicted_codes=len(predicted), gold_with_law=sum(bool(law_ids(h)) for h in labels[sid].values()))
            for code, hazard in predicted.items():
                if code not in CODES:
                    raise ValueError("Unknown predicted code")
                text, action = hazard.get("evidence", ""), hazard.get("prevention", "")
                counts.update(missing_prevention=int(not bool(action.strip())), correct_codes=int(code in labels[sid]))
                expected = set(law_ids(labels[sid][code])) if code in labels[sid] else set()
                frequency = freqs[code]
                prior = sorted(frequency, key=lambda article: (-frequency[article], article))
                body = index.rank(text + " " + action)
                learned = official_ranking(retriever, text, action, code)
                revised = rerank(prior, body, learned, index.rank(action) if action else [])
                for name, order in zip(methods, (prior, body, learned, revised), strict=True):
                    if any(a not in active for a in order[:3]):
                        raise ValueError("Inactive article in the top three")
                    methods[name].update(cited=int(bool(order)), joint_top1=int(bool(order and order[0] in expected)),
                        joint_top3=int(bool(set(order[:3]) & expected)), cited_wrong_code=int(bool(order) and code not in labels[sid]))
        results[arm] = {**counts, "methods": methods}
    return {"scope": "Real local generation replay. Repeated public arms share one photo. Top1/top3 compare article IDs only.",
            "ranking_inputs": "predicted code and generated evidence/prevention when present; remove explicit article mentions",
            "source_exclusion": "self, same pilot prefix, pixel match or dHash<=4, and reviewed incident if available",
            "reviewed_incidents": bool(groups), "retrieval_hashes": hashes, "generation_hashes": receipts,
            "evidence_for_legal_support_reviewed": False, "arms": results}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write_aggregate(root / "results/generated_law_comparison.json", run(root))
