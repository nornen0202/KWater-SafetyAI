"""Build and score local blinded minimal pairs from saved model generations; CPU only."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import hmac
from html import escape
import io
import json
from pathlib import Path
import secrets
import shutil
import unicodedata

from .adapter_scale_pilot import sample
from .metrics import CODES, unique_keys
from .train_head import file_hash, save_atomic


COHORTS = ("public_clean", "prompt_six", "adapter_eighteen")
FIELDS = ("evidence", "prevention")
CHOICES = ("A", "B", "tie", "both_unsuitable", "insufficient_information")
PUBLIC_IMAGE = Path("G:/CodexData/kwater-rd-20261125/visual-probe/public-example.jpg")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def write(path, obj):
    save_atomic(path, lambda f: f.write((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def code_map(root):
    tree = ast.parse((root/"data/baseline/main.py").read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CODE_MAP" for t in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError("Missing organizer code map")


def adapter_generations(local, expected_ids):
    for factor in ("1.0", "0.75"):
        final = local / f"generated-scale-{factor}.json"
        batch = read(final) if final.is_file() else []
        if final.is_file() and [row["sample_id"] for row in batch] != expected_ids:
            raise ValueError("Scale generation batch differs from its registered photo order")
        partial = []
        for i, checkpoint in enumerate(sorted((local / f"progress-{factor}").glob("*.json"))):
            row = read(checkpoint)
            if i >= len(expected_ids) or checkpoint.name != f"{i+1:02d}.json" or row["sample_id"] != expected_ids[i]:
                raise ValueError("Scale progress row does not match its registered photo")
            if batch and row != batch[i]:
                raise ValueError("Scale progress row conflicts with the final generation")
            partial.append(row)
        for row in batch or partial:
            yield factor, row


def sources(root, public_img):
    rows = {}
    def add(trial, cohort, sid, picture, image_sha, receipt_sha, hazards, mapping=None):
        fields = {}
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict):
                    continue
                code = item.get("code")
                code = mapping.get(code) if mapping else code
                if code in CODES and code not in fields:
                    fields[code] = {field: item[field] for field in FIELDS
                                    if isinstance(item.get(field), str) and item[field].strip()}
        if trial in rows:
            raise ValueError("Duplicate generation identity")
        rows[trial] = dict(cohort=cohort, sid=sid, image=picture, image_sha=image_sha,
                           receipt_sha=receipt_sha, fields=fields, parsed=isinstance(hazards, list))

    public = root/"results/prototype.json"
    raw = read(public)
    if raw["official_competition_lora"] or raw["model_response"]["finish_reason"] != "stop":
        raise ValueError("Expected the archived clean-model public generation")
    gen = json.loads(raw["model_response"]["raw_output"], object_pairs_hook=unique_keys)
    add("public", COHORTS[0], raw["sample_id"], public_img, raw["image_sha256"],
        file_hash(public), gen["hazards"])

    audit = read(root/"results/train_audit_full.json")["inventory"]
    design_path = root/"data/prompt-pilot/design.json"
    if design_path.exists():
        design = read(design_path)
        sig = sha(canonical(design))
        directory = design_path.parent / sig[:12]
        for i, anchor in enumerate(design["anchors"]):
            for arm in design["arms"]:
                path = directory / f"{i}-{arm}.json"
                if not path.exists():
                    continue
                r = read(path)
                case = design["anchors"][(i+1) % len(design["anchors"])] if arm == "swapped_11" else anchor
                if r["signature"] != sig or r["image_sha256"] != case["sha256"] or audit[case["sid"]]["sha256"] != case["sha256"]:
                    raise ValueError("Prompt receipt provenance mismatch")
                if not r["generation"].get("raw_output", "").strip():
                    continue
                hs = None
                if r["generation"]["finish_reason"] == "stop":
                    try:
                        hs = json.loads(r["generation"]["raw_output"], object_pairs_hook=unique_keys).get("hazards")
                    except (ValueError, AttributeError):
                        pass
                add(f"prompt/{i}/{arm}", COHORTS[1], case["sid"], root/"data/train"/case["image"],
                    case["sha256"], file_hash(path), hs)

    scale_dir = root/"data/adapter-scale-pilot"
    if scale_dir.is_dir():
        lookup = {sha("kwater/scale/" + s)[:32]: s for s,_ in sample(read(root/"data/prefix-pilot/exposure.json"))}
        if len(lookup) != 18:
            raise ValueError("Expected 18 distinct registered scale photos")
        mapping = code_map(root)
        for factor, rec in adapter_generations(scale_dir, list(lookup)):
            sid = lookup[rec["sample_id"]]
            item = audit[sid]
            pred = rec["prediction"]
            hs = pred.get("hazards") if isinstance(pred, dict) else None
            add(f"scale/{factor}/{sid}", COHORTS[2], sid, root/"data/train"/item["image"],
                item["sha256"], sha(canonical(rec)), hs, mapping)
    return rows


def edit(original, old, new, tokenizer):
    if (not old or not new or old == new or old.strip() != old or new.strip() != new
            or original.count(old) != 1 or not all(unicodedata.is_normalized("NFC", s) for s in (original, old, new))):
        raise ValueError("Expected one exact, nonempty, unpadded NFC phrase replacement")
    offset = original.index(old)
    end = offset + len(old)
    changed = original[:offset] + new + original[end:]
    def sizes(text):
        return (len(text), len(text.encode("utf-8")), len(text.split()),
                len(tokenizer.encode(text, add_special_tokens=False).ids))
    before, after = sizes(original), sizes(changed)
    if before != after or len(old) > 28 or len(old.split()) > 5:
        raise ValueError(f"Single-phrase or equal-length constraint failed: {before}, {after}")
    if not(original[:offset] == changed[:offset] and original[end:] == changed[offset+len(new):]):
        raise ValueError("Unchanged context mismatch")
    return changed, before


def blinded(secret, *key):
    return hmac.new(secret, canonical(key).encode(), hashlib.sha256).hexdigest()


def template(form, target):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["pair_id","choice","comment"])
    writer.writerows((r["pair_id"], "", "") for r in form)
    save_atomic(target, lambda f: f.write(stream.getvalue().encode("utf-8-sig")))


def review_page(forms, instructions, target):
    pieces = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
              '<title>사진별 문구 비교</title><style>body{font:17px/1.65 system-ui,sans-serif;color:#182525;'
              'max-width:64rem;margin:2rem auto;padding:0 1rem}article{margin:3rem 0;border-top:1px solid #a8b7b1}'
              'img{max-width:100%;max-height:68vh;display:block}p span{font-weight:bold}code{font-size:.8em}</style>',
              '<h1>사진별 문구 비교</h1><p>사진을 눌러 원본 크기로 볼 수 있습니다. 목록 순서는 빈 CSV와 같습니다. '
              '평가자별 CSV 사본의 choice에 A, B, tie, both_unsuitable, insufficient_information 중 하나를 입력하세요. '
              '판단 근거는 comment에 적을 수 있습니다.</p>']
    for r in forms:
        image = escape(r["image"], quote=True)
        pieces.extend(('<article>', f'<h2>{escape(r["code"])}</h2><code>{escape(r["pair_id"])}</code>',
                       f'<p>{escape(instructions[r["field"]])}</p>',
                       f'<a href="{image}" target="_blank" rel="noopener"><img src="{image}" alt="비교 대상 현장 사진"></a>',
                       f'<p><span>A</span>　{escape(r["A"])}</p><p><span>B</span>　{escape(r["B"])}</p></article>'))
    pieces.append('</html>')
    save_atomic(target, lambda f:f.write("\n".join(pieces).encode()))


def score(private, cards, cohorts):
    ratings = {c: Counter() for c in COHORTS}
    covered = {c: set() for c in COHORTS}
    reviewers = {c: set() for c in COHORTS}
    field_votes = {c: {field: Counter() for field in FIELDS} for c in COHORTS}
    for path in sorted((private/"ratings").glob("*.csv")):
        seen = set()
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["pair_id","choice","comment"]:
                raise ValueError("Unexpected review columns")
            for row in reader:
                key, vote = row["pair_id"], row["choice"]
                if None in row or key not in cards or key in seen or vote not in ("",*CHOICES):
                    raise ValueError(f"Invalid or duplicate ballot in {path.name}")
                seen.add(key)
                if not vote:
                    continue
                card = cards[key]
                c = card["cohort"]
                if vote in ("A","B"):
                    vote = "source" if vote == card["original_side"] else "edited"
                ratings[c][vote] += 1
                field_votes[c][card["field"]][vote] += 1
                covered[c].add(key)
                reviewers[c].add(path.stem)
    for c, summary in cohorts.items():
        decisive = ratings[c]["source"] + ratings[c]["edited"]
        summary.update(human_rated_pairs=len(covered[c]), human_unrated_pairs=summary["built_pairs"]-len(covered[c]),
                       human_votes=sum(ratings[c].values()), human_reviewers=len(reviewers[c]),
                       choices=dict(ratings[c]), votes_by_field={f:dict(v) for f,v in field_votes[c].items()},
                       source_preference_among_directional_votes=ratings[c]["source"]/decisive if decisive else None)


def build(root, public_img):
    from tokenizers import Tokenizer
    private = root/"data/preference-pilot"
    (private/"ratings").mkdir(parents=True, exist_ok=True)
    seed_path = private/"side-order.key"
    if not seed_path.exists():
        try:
            with seed_path.open("xb") as out:
                out.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    secret = seed_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("Invalid blinding key")
    rows = sources(root, public_img)
    # The recipe lives locally: exact phrases, source identities, and preparation
    # screening are deliberately unavailable in the aggregate repository.
    recipes = read(private/"recipes.json")
    tokenpath = root/"data/baseline/base_model/tokenizer.json"
    tok = Tokenizer.from_file(str(tokenpath))
    forms, cards, units, pooled = [], {}, set(), {}
    photos = private/"blind/photos"
    photos.mkdir(parents=True,exist_ok=True)
    for recipe in recipes:
        trial, code, field = recipe["trial"], recipe["code"], recipe["field"]
        gen = rows[trial]
        original = gen["fields"][code][field]
        if (field not in FIELDS or recipe["preparer_screened_safe"] is not True or sha(original) != recipe["text_sha256"]
                or gen["image_sha"] != recipe["image_sha256"] or gen["receipt_sha"] != recipe["receipt_sha256"]):
            raise ValueError("Private recipe does not match its frozen source")
        changed, sizes = edit(original, recipe["from"], recipe["to"], tok)
        unit = gen["cohort"],gen["sid"],field
        if unit in units:
            raise ValueError("At most one pair per cohort, photo, and field")
        units.add(unit)
        identity = blinded(secret, trial,code,field,sha(original),sha(changed))
        pair = identity[:24]
        if pair in cards:
            raise ValueError("Pair identity collision")
        # Assign stable sides in pairs so the difference in source-left counts is <=1
        # per cohort and field. Append recipes to retain blind identities on refresh.
        stratum = gen["cohort"],field
        n = pooled.get(stratum,0)
        pooled[stratum] = n+1
        original_first = (int(blinded(secret,"side",stratum,n//2)[:2],16)%2) == n%2
        values = (original,changed) if original_first else (changed,original)
        photo = photos / (blinded(secret,"photo",gen["image_sha"])[:24] + gen["image"].suffix.lower())
        if not photo.exists():
            if file_hash(gen["image"]) != gen["image_sha"]:
                raise ValueError("Source photo checksum mismatch")
            shutil.copyfile(gen["image"], photo)
        if file_hash(photo) != gen["image_sha"]:
            raise ValueError("Blind photo checksum mismatch")
        forms.append(dict(pair_id=pair, image=photo.relative_to(private/"blind").as_posix(), code=code,
                          field=field, A=values[0], B=values[1]))
        cards[pair] = dict(cohort=gen["cohort"], field=field, code=code, sid=gen["sid"], trial=trial,
                           original_side="A" if original_first else "B", source_field_sha256=sha(original),
                           edited_field_sha256=sha(changed), source_receipt_sha256=gen["receipt_sha"],
                           image_sha256=gen["image_sha"], lengths=list(sizes))
    # A partially completed human ballot must never be reinterpreted after a recipe change.
    keypath = private/"blind-key.json"
    if keypath.exists():
        old = read(keypath)
        if any(cards.get(pair) != meta for pair, meta in old.items()):
            raise ValueError("An existing blind pair changed; preserve the previous recipes and order")
    formpath = private/"blind/review.json"
    previous_forms = read(formpath)["pairs"] if formpath.exists() else []
    candidates = {r["pair_id"]: r for r in forms}
    for prior in previous_forms:
        if candidates.pop(prior["pair_id"], None) != prior:
            raise ValueError("Existing blind form changed; only append newly prepared questions")
    forms = previous_forms + sorted(candidates.values(), key=lambda r:blinded(secret,"form",r["pair_id"]))
    instructions = {"evidence":"같은 사진과 코드에서 실제로 확인되는 대상·상태·위치를 어느 문장이 더 충실히 기술합니까?",
                    "prevention":"같은 사진과 코드에서 어느 조치가 더 안전하고 구체적이며 현장에서 실행 가능합니까?",
                    "tie":"두 문구의 적절성이 비슷함", "both_unsuitable":"양쪽 모두 부적절함",
                    "insufficient_information":"사진만으로 판단하기 어려움",
                    "how_to_rate":"코드는 모델의 예측입니다. 사진에서 뒷받침되지 않으면 양쪽 부적절이나 판단 곤란을 선택할 수 있습니다. 사진을 원본 크기로 열고 평가자별로 독립적으로 작성합니다."}
    cohorts = {}
    planned = dict(zip(COHORTS,(1,6,18),strict=True))
    for c in COHORTS:
        batch = [r for r in rows.values() if r["cohort"] == c]
        selected = [r for r in cards.values() if r["cohort"] == c]
        cohorts[c] = {"planned_source_photos":planned[c],"source_generations_observed":len(batch),
                      "parsed_source_generations":sum(r["parsed"] for r in batch),
                      "source_photos_observed":len({r["image_sha"] for r in batch}),
                      "generations_used_in_pairs":len({r["trial"] for r in selected}),
                      "photos_with_pairs":len({r["image_sha256"] for r in selected}),
                      "built_pairs":len(selected),"pairs_by_field":dict(Counter(r["field"] for r in selected))}
    score(private,cards,cohorts)
    write(keypath,cards)
    write(formpath, {"instructions":instructions, "choices":CHOICES, "pairs":forms})
    review_page(forms, instructions, private/"blind/review.html")
    template(forms,private/"blind/blank.csv")
    summary = {"scope":"Paired wording preparation and human preference inventory. No new inference. No official LLM score.",
               "cohorts":cohorts, "checks":{"same_photo_and_code":True,"one_exact_span":True,
                    "equal_NFC_codepoints_UTF8_bytes_words_Qwen_tokens":True,"reviewer_sees_source_side":False,
                    "tokenizer_sha256":file_hash(tokenpath),"recipe_sha256":file_hash(private/"recipes.json")},
               "eligibility":"Preparer screened both alternatives for safe wording; eligibility is not a human preference vote."}
    write(root/"results/preference_pilot.json",summary)
    return summary


def tally(root):
    private = root/"data/preference-pilot"
    cards = read(private/"blind-key.json")
    previous = read(root/"results/preference_pilot.json")
    form = read(private/"blind/review.json")["pairs"]
    if {row["pair_id"] for row in form} != cards.keys() or len(cards) != len(form):
        raise ValueError("Blind form and private key differ")
    for row in form:
        item = cards[row["pair_id"]]
        photo = (private/"blind"/row["image"]).resolve()
        if not photo.is_relative_to((private/"blind/photos").resolve()) or file_hash(photo) != item["image_sha256"]:
            raise ValueError("Blind photograph differs from the locked pair")
        a = item["original_side"]
        b = "B" if a == "A" else "A"
        if (sha(row[a]) != item["source_field_sha256"] or sha(row[b]) != item["edited_field_sha256"]
                or row["code"] != item["code"] or row["field"] != item["field"]):
            raise ValueError("Blind alternative differs from the locked pair")
    for c, summary in previous["cohorts"].items():
        if summary["built_pairs"] != sum(item["cohort"] == c for item in cards.values()):
            raise ValueError("Aggregate count differs from the locked design")
    score(private,cards,previous["cohorts"])
    write(root/"results/preference_pilot.json",previous)
    return previous


def inventory(root, public_img):
    generations = sources(root, public_img)
    record = {}
    for trial, item in generations.items():
        record[trial] = {"cohort":item["cohort"], "sid":item["sid"], "image_sha256":item["image_sha"],
                         "receipt_sha256":item["receipt_sha"], "parsed":item["parsed"],
                         "source_field_hashes":{code:{k:sha(text) for k,text in fields.items()}
                                                for code,fields in item["fields"].items()}}
    write(root/"data/preference-pilot/source-inventory.json",record)
    return len(record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--public-image", type=Path, default=PUBLIC_IMAGE)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--score-only", action="store_true", help="Tally actual human ballots for the frozen forms")
    mode.add_argument("--inventory-only", action="store_true", help="Record local source identities and hashes for preparation")
    args = p.parse_args()
    if args.inventory_only:
        print("Saved generation inventory:", inventory(args.root.resolve(), args.public_image))
    else:
        report = tally(args.root.resolve()) if args.score_only else build(args.root.resolve(), args.public_image)
        print(json.dumps(report["cohorts"], ensure_ascii=False, indent=2))
