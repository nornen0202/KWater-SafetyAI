"""Independent CPU-only offline visual readings; public output contains counts only."""

import argparse
import base64
from collections import Counter
from hashlib import sha256
import hmac
import io
import json
from pathlib import Path
import secrets

from .citation_review_panel import DATA, ROOT, freeze, read
from .law_reranker_11code import article_to_osh_id, extract_terms
from .visual_probe import DEFAULT_CACHE, generate, local_server


MODELS = {
    "lfm": {"LFM2-VL-450M-Q8_0.gguf": "e97704a0cf0a1d00ca604b4c672c82f4234318dba9d43a7f7a4c0d2df6747844",
            "mmproj-LFM2-VL-450M-Q8_0.gguf": "a6a813fae4ba3f90852a990fdd4786a017dba0e80e599a366853ba539ee2b5ef"},
    "qwen": {"Qwen3.5-0.8B-Q4_K_M.gguf": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
             "mmproj-F16.gguf": "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"},
}
CACHE = DEFAULT_CACHE.parent / "citation-offline-judges"
RUBRIC = ("Review only this photograph, the supplied observation, the recommended prevention and legal excerpts. "
          "Independently grade each excerpt A and B. support = does it support the particular prevention action? "
          "fit = does its regulated object/work apply to the described visible hazard? "
          "Use direct, partial (including conditional), unrelated, or unknown if evidence is insufficient. "
          "Grade the identical action sequence once: safe, unsafe, unclear, or NA when no sequencing is needed. "
          "When there is no recommendation or citation, use NA. Equal alternatives may have equal ratings. "
          "Do not assume hidden electrical state or work conditions. Return JSON and a brief reason.")
RATING = ["direct", "partial", "unrelated", "unknown", "NA"]
FIELDS = {k: {"type": "string", "enum": RATING} for k in ("support_A", "support_B", "fit_A", "fit_B")}
FIELDS.update(order={"type": "string", "enum": ["safe", "unsafe", "unclear", "NA"]}, reason={"type": "string"})
SCHEMA = {"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False}


def law_excerpt(article, observation, prevention):
    """Keep the title and complete best matching lines from long articles."""
    if len(article) <= 720:
        return article
    lines = article.splitlines()
    query = extract_terms(observation + " " + prevention)
    def overlap(line):
        return sum((query & extract_terms(line)).values())
    # A single long statutory paragraph remains explicitly truncated.
    ranked = sorted(range(1, len(lines)), key=lambda i: (-overlap(lines[i]), i))
    excerpt = [lines[0][:355]]
    room = 680 - len(excerpt[0])
    chosen = []
    for i in ranked:
        if len(lines[i]) + 1 <= room:
            chosen.append(i)
            room -= len(lines[i]) + 1
    excerpt.extend(lines[i] for i in sorted(chosen))
    return "\n".join(excerpt) + "\n[발췌]"


def build_cards():
    panel = read(ROOT / "panel.json")
    rows = {r["image_id"]: r for r in map(json.loads, (DATA / "v3_epoch1_11code_standalone.jsonl").read_bytes().splitlines())}
    cards = []
    for p in panel:
        sid = p["sample_id"]
        rec = rows[sid]
        base = dict(photo=p["image"], image_sha256=p["image_sha256"], group=p["group_id"], kind="panel")
        if not rec["hazards"]:
            cards.append(dict(**base, sid=sid, code="none", evidence=rec.get("scene_evidence", ""),
                              prevention="", current="", candidate=""))
        for code, h in sorted(rec["hazards"].items()):
            law, = [r["id"] for r in h["references"] if r["kind"] == "law"]
            cards.append(dict(**base, sid=sid, code=code, evidence=h["evidence"], prevention=h["prevention"],
                              current=law, candidate=law))
    # An independent diagnostic tests whether each reader can distinguish incompatible provisions.
    # Use already reviewed photos only. Keep diagnostics outside the refrozen panel counts.
    from .fold4_citation_alignment import scope_veto
    from .prevention_law_guard import build_active_records
    active = build_active_records(list(map(json.loads,Path("data/baseline/assets/laws.jsonl").read_bytes().splitlines())))
    for code, relevant, foil in (("fire_explosion", "OSH-241", "OSH-45"),
                                 ("caught_in_machine", "OSH-87", "OSH-45")):
        from .fold4_code_diagnosis import DEV
        candidates = []
        for sid, record in rows.items():
            h = record["hazards"].get(code)
            if h and h.get("references") and h["references"][0]["id"] == "OSH-319":
                if scope_veto(code,(h["evidence"],h["prevention"],""),relevant,"OSH-319",active) == relevant:
                    candidates.append(sid)
        sid, = candidates
        p = next(r for r in read(DEV) if r["sample_id"] == sid)
        h = rows[sid]["hazards"][code]
        for reverse in (False, True):
            cards.append(dict(sid=sid, code=code, photo=p["image"], image_sha256=p["sha256"], group=p["group_id"],
                              evidence=h["evidence"], prevention=h["prevention"], kind="diagnostic",
                              current=foil if reverse else relevant, candidate=relevant if reverse else foil,
                              relevant=relevant))
    return cards


def prepare(wave):
    if wave == "panel":
        witness = read(Path("results/fold4-citation-review-panel-v2.json"))
        if witness["exact_epoch1_changes_on_panel"] != 0:
            raise ValueError("This identical-citation panel template must be regenerated for a changed panel")
        cards = build_cards()
    else:
        cards = read(ROOT / wave / "matched-private.json")
    dest = ROOT / wave
    dest.mkdir(exist_ok=True)
    keypath = dest / "random-key.bin"
    if not keypath.exists():
        with keypath.open("xb") as stream:
            stream.write(secrets.token_bytes(32))
    seed = keypath.read_bytes()
    def token(label):
        return hmac.new(seed, label.encode(), "sha256").hexdigest()
    laws = {article_to_osh_id(r["article"]): r["text"] for r in map(json.loads, Path("data/baseline/assets/laws.jsonl").read_bytes().splitlines())
            if r["record_type"] == "article" and r["status"] == "active"}
    key = {}
    for reader in MODELS:
        deck = []
        for i, card in enumerate(cards):
            identity = f"{i}/{card['sid']}/{card['code']}"
            tag = token(reader + identity)[:20]
            flip = (int(token("order/" + identity), 16) % 2 == 0) != (reader == "qwen")
            A, B = (card["candidate"], card["current"]) if flip else (card["current"], card["candidate"])
            def excerpt(law):
                return law_excerpt(laws[law], card["evidence"], card["prevention"]) if law else "(none)"
            prompt = (RUBRIC + "\nObservation: " + card["evidence"] + "\nPrevention: " + (card["prevention"] or "(none)")
                      + "\nA: " + excerpt(A) + "\nB: " + excerpt(B))
            # Photo paths are consumed by the host image loader and never appear in the model prompt.
            root = ROOT / "prospective-input" if card.get("source") == "71407" else Path("data/train")
            deck.append(dict(tag=tag, path=(root / card["photo"]).as_posix(), image_sha256=card["image_sha256"], prompt=prompt))
            key[tag] = dict(reader=reader, slot=i, flip=flip, kind=card["kind"], code=card["code"],
                            relevant_side=("A" if card.get("relevant")==A else "B") if card.get("relevant") else None)
        freeze(dest / f"{reader}-deck.json", sorted(deck, key=lambda x: x["tag"]))
    freeze(dest / "sealed-key.json", key)


def run(reader, wave):
    from PIL import Image, ImageOps
    dest = ROOT / wave
    deck = read(dest / f"{reader}-deck.json")
    out = dest / reader
    out.mkdir(exist_ok=True)
    flags = ["--device", "none", "-ngl", "0", "--no-mmproj-offload", "-t", "2", "-tb", "2",
             "--prio", "-1", "--parallel", "1", "--no-warmup"]
    if reader == "qwen":
        flags += ["--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}',
                  "-b", "256", "-ub", "64", "--load-mode", "none"]
    if all((out / (c["tag"] + ".json")).exists() for c in deck):
        return
    with local_server(CACHE / reader, dest / f"{reader}.log", weights=MODELS[reader],
                      runtime_root=DEFAULT_CACHE, extra_flags=flags, context_size=4096,
                      startup_timeout=600) as (port, secret, _):
        for card in deck:
            path = out / (card["tag"] + ".json")
            if path.exists():
                continue
            pixels = Path(card["path"]).read_bytes()
            if sha256(pixels).hexdigest() != card["image_sha256"]:
                raise ValueError("Frozen image differs")
            with Image.open(io.BytesIO(pixels)) as source:
                img = ImageOps.exif_transpose(source).convert("RGB")
                img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
            imgbytes = buf.getvalue()
            result = generate(port, secret, "data:image/jpeg;base64,"+base64.b64encode(imgbytes).decode(),
                              card["prompt"], SCHEMA, 155)
            freeze(path, dict(raw=result["raw_output"], finish=result["finish_reason"], seconds=result["elapsed_s"],
                              tag=card["tag"], prompt_sha256=sha256(card["prompt"].encode()).hexdigest(),
                              presented_image_sha256=sha256(imgbytes).hexdigest(), weights=MODELS[reader]))
            print(reader, len(list(out.glob("*.json"))), "/", len(deck), flush=True)


def summarize(wave):
    """Open the mapping only after both decks have a full immutable set of raw returns."""
    dest = ROOT / wave
    envelopes = {}
    for reader in MODELS:
        deck = read(dest / f"{reader}-deck.json")
        paths = sorted((dest / reader).glob("*.json"))
        if {p.stem for p in paths} != {r["tag"] for r in deck}:
            raise ValueError(f"{reader}: waiting for complete deck")
        for c in deck:
            path = dest / reader / (c["tag"]+".json")
            raw = read(path)
            if (raw["weights"] != MODELS[reader] or raw["tag"] != c["tag"]
                    or raw["prompt_sha256"] != sha256(c["prompt"].encode()).hexdigest()):
                raise ValueError("Rating is not bound to the frozen prompt/model")
            envelopes[c["tag"]] = raw
    lock = freeze(dest / "both-readers-locked.json", {tag: sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
                                                       for tag, raw in envelopes.items()})
    key = read(dest / "sealed-key.json")
    tally = {r: {} for r in MODELS}
    by_slot = {}
    levels = {"direct": 2, "partial": 1, "unrelated": 0}
    for tag, raw in envelopes.items():
        meta = key[tag]
        r = tally[meta["reader"]].setdefault(meta["kind"], Counter())
        r["presentations"] += 1
        try:
            grade = json.loads(raw["raw"])
            valid = (raw["finish"] == "stop" and isinstance(grade, dict) and grade.keys() == FIELDS.keys()
                     and all(grade[k] in spec["enum"] for k, spec in FIELDS.items() if "enum" in spec)
                     and isinstance(grade["reason"],str) and bool(grade["reason"].strip()))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            r["invalid"] += 1
            continue
        r["valid"] += 1
        r["order_" + grade["order"]] += 1
        pair = ["B", "A"] if meta["flip"] else ["A", "B"]
        if meta["code"] == "none":
            r["normal"] += 1
            r["normal_all_na"] += all(grade[k]=="NA" for k in FIELDS if k!="reason")
        slot = by_slot.setdefault(meta["slot"], {})
        slot[meta["reader"]] = {"order":grade["order"]}
        for axis in ("support", "fit"):
            current, new = (grade[f"{axis}_{side}"] for side in pair)
            for name, level in (("current", current), ("candidate", new)):
                r[f"{axis}_{name}_{level}"] += 1
            if meta["code"] != "none":
                r[f"{axis}_exact_tie"] += current == new
            pref = "unassessable"
            if current in levels and new in levels:
                diff = levels[new] - levels[current]
                pref = "better" if diff>0 else "worse" if diff<0 else "tie"
            r[f"{axis}_candidate_{pref}"] += 1
            slot[meta["reader"]][axis] = pref
            if meta["kind"] == "diagnostic":
                winner = grade[axis + "_" + meta["relevant_side"]]
                foil = grade[axis + "_" + ("B" if meta["relevant_side"] == "A" else "A")]
                r[f"{axis}_selected_relevant"] += winner in levels and foil in levels and levels[winner] > levels[foil]
        slot["kind"] = meta["kind"]
    agreement = {}
    for entry in by_slot.values():
        if not MODELS.keys() <= entry.keys():
            continue
        c = agreement.setdefault(entry["kind"],Counter())
        c["jointly_valid"] += 1
        for axis in ("support","fit","order"):
            left, right = (entry[reader][axis] for reader in MODELS)
            c[axis+"_agreement"] += left == right
            if axis != "order" and "unassessable" not in (left,right):
                c[axis+"_both_assessable"] += 1
                c[axis+"_assessable_agreement"] += left == right
    summary = dict(raw_lock_sha256=lock, ratings=tally, agreement=agreement, models=MODELS,
                   key_sha256=sha256((dest/"sealed-key.json").read_bytes()).hexdigest())
    freeze(Path(f"results/citation-review-{wave}-readings.json"),summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "summarize", *MODELS])
    p.add_argument("--wave", choices=["panel", "challenge"], default="panel")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.wave)
    elif args.action == "summarize":
        summarize(args.wave)
    else:
        run(args.action, args.wave)
