"""Native 11-code OSH-* statutory retriever & action-support reranker.

Fixes two systematic citation errors in previous submissions:
  1. Indexes training cases directly under the 11 official competition hazard codes
     (no lossy 8-code remapping, restoring OSH-4-2 for chemical_exposure/dust and
     separating OSH-52 structural shoring, OSH-200 vehicle contact, OSH-20 restricted zones
     within `other`).
  2. Combines active `laws.jsonl` article/clause BM25-like lexical scoring, training-split
     11-code nearest-case support, and statutory priors for ultra-rare K-water codes
     (`drowning_inundation`, `confined_space`, `thermal_weather`), returning 1-2 well-supported
     `{"kind": "law", "id": "OSH-..."}` references.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from math import log, sqrt
from pathlib import Path
import re
import unicodedata

from .metrics import CODES, unique_keys
from .train_head import save_atomic


_ART_RE = re.compile(r"(?:제)?0*(\d+)(?:(?:조(?:의|-)?|의|-)0*(\d+))?(?:조)?")
_WORD_RE = re.compile(r"[가-힣]{2,}|[a-z0-9]{2,}")
_AMEND_RE = re.compile(r"<[^>]*>|\[(?:본조|전문|제목|제\d+조)[^\]]*\]")
_STOP = frozenset(
    "사업주는 사업주 근로자가 근로자 해당 경우 위험 작업에는 작업을 작업하는 "
    "하여야 한다 있다 없도록 경우에는 아니하다 등에 위한 의하여 따라 다음 "
    "위하여 준수하여야 조치를 조치하도록 안전하게 작업자가 보인다 "
    "확보한다 설치한다 실시한다 방지한다 필요한 적절한 사진에서 확인된다".split()
)

# Statutory domain priors from 산업안전보건기준에 관한 규칙 for codes with sparse training support
STATUTORY_CODE_PRIORS: dict[str, dict[str, float]] = {
    "drowning_inundation": {"OSH-43": 0.22, "OSH-54": 0.20, "OSH-348": 0.16, "OSH-38": 0.14},
    "confined_space": {"OSH-619": 0.34, "OSH-620": 0.24, "OSH-622": 0.18},
    "thermal_weather": {"OSH-559": 0.28, "OSH-560": 0.22, "OSH-566": 0.20, "OSH-313": 0.16},
    "chemical_exposure": {"OSH-4-2": 0.20, "OSH-420": 0.16, "OSH-450": 0.14, "OSH-607": 0.14},
    "fire_explosion": {"OSH-241": 0.24, "OSH-239": 0.20, "OSH-232": 0.15, "OSH-230": 0.14},
    "electrical_energy": {"OSH-301": 0.24, "OSH-304": 0.20, "OSH-311": 0.15, "OSH-318": 0.14},
    "caught_in_machine": {"OSH-87": 0.26, "OSH-88": 0.16, "OSH-92": 0.14},
    "struck_by_lifting": {"OSH-132": 0.18, "OSH-146": 0.16, "OSH-163": 0.15, "OSH-14": 0.15, "OSH-3": 0.12},
    "fall_height": {"OSH-43": 0.22, "OSH-42": 0.20, "OSH-56": 0.18, "OSH-24": 0.12},
    "slip_trip": {"OSH-3": 0.24, "OSH-22": 0.20, "OSH-4": 0.12},
    "other": {"OSH-52": 0.16, "OSH-20": 0.16, "OSH-200": 0.16, "OSH-172": 0.12, "OSH-3": 0.12},
}


def article_to_osh_id(article_str: str) -> str | None:
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(article_str or "")))
    if s.startswith("OSH-"):
        return s if re.fullmatch(r"OSH-[0-9]+(?:-[0-9]+)?", s) else None
    m = _ART_RE.fullmatch(s)
    if not m:
        return None
    base = int(m.group(1))
    branch = int(m.group(2)) if m.group(2) else None
    return f"OSH-{base}-{branch}" if branch is not None else f"OSH-{base}"


@lru_cache(maxsize=16384)
def extract_terms(text: str) -> Counter:
    norm = unicodedata.normalize("NFKC", str(text or "")).lower()
    norm = _AMEND_RE.sub(" ", norm)
    feats: Counter = Counter()
    for w in _WORD_RE.findall(norm):
        if w in _STOP or w.isdigit():
            continue
        feats["w:" + w] += 1.6
        if "가" <= w[0] <= "힣":
            for sz in (2, 3):
                for i in range(len(w) - sz + 1):
                    feats[w[i : i + sz]] += 1.0
    return feats


def hazard_query_terms(evidence: str, prevention: str, scene_note: str = "") -> Counter:
    out: Counter = Counter()
    for txt, wt in ((evidence, 1.0), (prevention, 1.0), (scene_note, 0.42)):
        for k, v in extract_terms(str(txt or "")).items():
            out[k] += wt * v
    return out


class _TFIDFIndex:
    def __init__(self, docs: list[Counter]):
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        df = Counter(term for d in docs for term in d)
        n_docs = len(docs)
        self.idf = {
            t: min(5.2, log((n_docs + 1.0) / (f + 1.0)) + 1.0)
            for t, f in df.items()
        }
        for idx, doc in enumerate(docs):
            for t, w in self._vec(doc).items():
                self.postings[t].append((idx, w))

    def _vec(self, feats: Counter) -> dict[str, float]:
        weights = {t: log(1.0 + c) * self.idf[t] for t, c in feats.items() if t in self.idf}
        norm = sqrt(sum(w * w for w in weights.values()))
        return {t: w / norm for t, w in weights.items()} if norm > 0 else {}

    def query(self, feats: Counter) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t, qw in self._vec(feats).items():
            for idx, dw in self.postings[t]:
                scores[idx] += qw * dw
        return scores


class Native11LawRetriever:
    """Clause-granular + native 11-code OSH-* ranker using active statutory clauses + training cases."""

    def __init__(self, law_rows: list[dict], train_cases: list[dict] | None = None):
        # Index every active clause (article, paragraph, item, subitem) individually so specific
        # provisions are never diluted by long multi-paragraph articles, then map to canonical OSH-<n>.
        records_by_id = {r.get("id"): r for r in law_rows if r.get("id") and r.get("status") == "active"}
        clause_osh: list[str] = []
        clause_docs: list[Counter] = []
        valid_osh_set: set[str] = set()

        for row in law_rows:
            if row.get("status") != "active" or not row.get("text"):
                continue
            if row.get("record_type") not in ("article", "paragraph", "item", "subitem"):
                continue
            osh_id = article_to_osh_id(row.get("article", ""))
            if not osh_id:
                continue
            valid_osh_set.add(osh_id)
            clause_osh.append(osh_id)

            body = str(row.get("text") or "")
            title = str(row.get("title") or "")
            parts: list[tuple[str, float]] = [(body, 1.0), (title, 1.95)]
            parent = records_by_id.get(row.get("parent_id"))
            if parent and parent.get("text"):
                lead = str(parent.get("heading_text") or parent["text"]).splitlines()[0][:250]
                parts.append((lead, 0.22))
            doc_feats: Counter = Counter()
            for part, wt in parts:
                for t, c in extract_terms(part).items():
                    doc_feats[t] += wt * c
            clause_docs.append(doc_feats)

        self.clause_osh = clause_osh
        self.osh_ids = sorted(valid_osh_set, key=lambda x: [int(p) for p in x.removeprefix("OSH-").split("-")])
        self.osh_to_idx = {oid: i for i, oid in enumerate(self.osh_ids)}
        self.clause_index = _TFIDFIndex(clause_docs)

        # Build per-code case banks from native 11-code training cases
        self.case_indices: dict[str, _TFIDFIndex] = {}
        self.case_refs: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        self.code_priors: dict[str, Counter] = defaultdict(Counter)
        self.pair_cooccur: dict[str, Counter] = defaultdict(Counter)
        self.single_vs_multi: dict[str, Counter] = defaultdict(Counter)

        code_docs: dict[str, list[Counter]] = defaultdict(list)
        for row in train_cases or ():
            if "target" in row and isinstance(row["target"].get("hazards"), dict):
                s_note = str((row.get("notes") or {}).get("scene_note") or "")
                for code, h in row["target"]["hazards"].items():
                    refs = tuple(
                        dict.fromkeys(
                            r["id"]
                            for r in h.get("references", ())
                            if r.get("kind") == "law" and r.get("id") in self.osh_to_idx
                        )
                    )
                    q = hazard_query_terms(h.get("evidence", ""), h.get("prevention", ""), s_note)
                    if code in CODES and refs and q:
                        code_docs[code].append(q)
                        self.case_refs[code].append(refs)
                        self.code_priors[code].update(refs)
                        self.single_vs_multi[code]["multi" if len(refs) >= 2 else "single"] += 1
                        if len(refs) >= 2:
                            self.pair_cooccur[code][tuple(sorted(refs[:2]))] += 1
            elif "code" in row and "osh_ids" in row:
                code = row["code"]
                refs = tuple(dict.fromkeys(oid for oid in row["osh_ids"] if oid in self.osh_to_idx))
                q = hazard_query_terms(row.get("evidence", ""), row.get("prevention", ""), row.get("scene_note", ""))
                if code in CODES and refs and q:
                    code_docs[code].append(q)
                    self.case_refs[code].append(refs)
                    self.code_priors[code].update(refs)
                    self.single_vs_multi[code]["multi" if len(refs) >= 2 else "single"] += 1
                    if len(refs) >= 2:
                        self.pair_cooccur[code][tuple(sorted(refs[:2]))] += 1

        for code, docs in code_docs.items():
            self.case_indices[code] = _TFIDFIndex(docs)

    def select_osh_references(
        self,
        code: str,
        evidence: str,
        prevention: str,
        scene_note: str = "",
        max_refs: int = 2,
        *,
        ranked_candidates: bool = False,
    ) -> list[dict[str, str]]:
        q_full = hazard_query_terms(evidence, prevention, scene_note)
        if not q_full:
            fallback = next(iter(STATUTORY_CODE_PRIORS.get(code, {})), None)
            return [{"kind": "law", "id": fallback}] if fallback and fallback in self.osh_to_idx else []

        # 1. Clause-granular statutory score (max-pooled per canonical OSH-<n> article)
        best_clause_for_osh: dict[str, float] = {}
        for c_i, sc in self.clause_index.query(q_full).items():
            if sc >= 0.055:
                oid = self.clause_osh[c_i]
                if sc > best_clause_for_osh.get(oid, 0.0):
                    best_clause_for_osh[oid] = sc

        scores: dict[str, float] = defaultdict(float)
        for oid, sc in best_clause_for_osh.items():
            scores[oid] += 0.68 * sc

        # 2. Nearest 11-code training case retrieval
        c_idx = self.case_indices.get(code)
        n_cases = len(self.case_refs.get(code, ()))
        neighbor_multi_votes = 0.0
        neighbor_total_votes = 0.0
        if c_idx is not None and n_cases > 0:
            sims = c_idx.query(q_full)
            nbrs = sorted((i for i, s in sims.items() if s >= 0.065), key=lambda i: (-sims[i], i))[:25]
            sup_map: dict[str, list[float]] = defaultdict(list)
            for n_i in nbrs:
                refs_i = self.case_refs[code][n_i]
                neighbor_total_votes += sims[n_i]
                if len(refs_i) >= 2:
                    neighbor_multi_votes += sims[n_i]
                for pos, oid in enumerate(refs_i):
                    sup_map[oid].append(sims[n_i] * (1.0 if pos == 0 else 0.92))
            for oid, matches in sup_map.items():
                matches.sort(reverse=True)
                strongest = matches[0]
                repeated = sum(matches[1:5]) / 4.0
                emp_prior = sqrt(self.code_priors[code][oid] / max(n_cases, 1))
                scores[oid] += 0.98 * strongest + 0.36 * repeated + 0.13 * emp_prior

        # 3. Domain statutory prior ONLY when training case support is sparse (< 5 cases)
        if n_cases < 5:
            prior_weight = 0.95 if n_cases < 2 else 0.35
            for oid, p_val in STATUTORY_CODE_PRIORS.get(code, {}).items():
                if oid in self.osh_to_idx:
                    scores[oid] += prior_weight * p_val

        if not scores:
            return []
        ranked = sorted(scores.keys(), key=lambda oid: (-scores[oid], self.osh_to_idx[oid]))
        top1 = ranked[0]
        if scores[top1] < 0.075:
            return []
        if ranked_candidates:
            # Expose ranked candidates before the production co-citation gate.
            return [{"kind": "law", "id": oid} for oid in ranked[:max_refs]]

        selected = [top1]
        if max_refs >= 2 and len(ranked) >= 2 and neighbor_total_votes > 0:
            top2 = ranked[1]
            pair_key = tuple(sorted((top1, top2)))
            co_cnt = self.pair_cooccur.get(code, {}).get(pair_key, 0)
            multi_frac = neighbor_multi_votes / neighbor_total_votes
            ratio = scores[top2] / max(scores[top1], 1e-6)
            # Add 2nd law ONLY when nearest training cases predominantly co-cite 2 laws (e.g. OSH-22 + OSH-3)
            if co_cnt >= 8 and multi_frac >= 0.58 and ratio >= 0.84:
                selected.append(top2)

        return [{"kind": "law", "id": oid} for oid in selected[:max_refs]]


def export_compact_case_bank(train_rows: list[dict], out_path: Path) -> None:
    """Export compact non-image 11-code statutory case bank for self-contained submission packaging."""
    cases = []
    for row in train_rows:
        t = row["target"]
        if t.get("status") != "hazard":
            continue
        s_note = str((row.get("notes") or {}).get("scene_note") or "")
        for code, h in sorted(t["hazards"].items()):
            osh_ids = [r["id"] for r in h.get("references", ()) if r.get("kind") == "law"]
            if osh_ids:
                cases.append({
                    "code": code,
                    "osh_ids": osh_ids,
                    "evidence": h["evidence"],
                    "prevention": h["prevention"],
                    "scene_note": s_note,
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(
        out_path,
        lambda f: f.write(("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)).encode("utf-8")),
    )


def read_selected_cases(path: Path, ids: set[str]):
    """Parse targets only for permitted case or evaluation IDs."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            match = re.match(r'^\s*\{\s*"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")', line)
            if match is None:
                raise ValueError(f"Expected sample_id as the leading key in {path}")
            if json.loads(match[1]) in ids:
                yield json.loads(line, object_pairs_hook=unique_keys)


def evaluate_5fold_law_retrieval(root: Path, manifest_path: Path, out_path: Path) -> dict:
    laws = [
        json.loads(line)
        for line in (root / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev_rows = [r for r in manifest if r["split"] == "dev"]
    train_by_id = {r["sample_id"]: r for r in read_selected_cases(root / "data/train/train.jsonl", {r["sample_id"] for r in dev_rows})}

    # Also compare against the legacy 8-code LawRetriever[:1] mapping
    import sys
    if str(root / "data/baseline") not in sys.path:
        sys.path.insert(0, str(root / "data/baseline"))
    from scripts.rag import LawRetriever as LegacyLawRetriever  # type: ignore

    inv_8map = {
        "fall_height": "fall_height",
        "slip_trip": "slip_trip",
        "struck_by_lifting": "struck_crushed",
        "caught_in_machine": "caught_in_machine",
        "other": "collision_contact",
        "electrical_energy": "electrical_energy",
        "fire_explosion": "fire_thermal",
        "chemical_exposure": "hazardous_exposure",
    }

    total_haz = 0
    native_hit1 = 0
    native_exact = 0
    native_tp = 0
    native_pred = 0
    native_gold = 0

    legacy_hit1 = 0
    legacy_exact = 0
    legacy_tp = 0
    legacy_pred = 0
    per_code_hits: dict[str, Counter] = defaultdict(Counter)

    for fold in range(5):
        tr_ids = [r["sample_id"] for r in dev_rows if r["dev_fold"] != fold]
        va_ids = [r["sample_id"] for r in dev_rows if r["dev_fold"] == fold]

        tr_full = [train_by_id[sid] for sid in tr_ids]
        native_ret = Native11LawRetriever(laws, tr_full)

        # Convert tr_full to legacy 8-code format for apples-to-apples leak-free comparison
        legacy_cases = []
        for r in tr_full:
            t = r["target"]
            if t["status"] != "hazard":
                continue
            s_note = str((r.get("notes") or {}).get("scene_note") or "")
            hz_list = []
            for code, h in t["hazards"].items():
                c8 = inv_8map.get(code)
                if not c8:
                    continue
                rel = [
                    {"law_name": "산업안전보건기준에 관한 규칙", "article": ref["id"].removeprefix("OSH-")}
                    for ref in h.get("references", ())
                    if ref.get("kind") == "law"
                ]
                hz_list.append({"code": c8, "evidence": h["evidence"], "prevention": h["prevention"], "related_laws": rel})
            if hz_list:
                legacy_cases.append({"target": {"scene_note": s_note, "hazards": hz_list}})
        legacy_ret = LegacyLawRetriever(laws, legacy_cases)

        for sid in va_ids:
            r = train_by_id[sid]
            t = r["target"]
            if t["status"] != "hazard":
                continue
            s_note = str((r.get("notes") or {}).get("scene_note") or "")
            for code, h in t["hazards"].items():
                gold_osh = [ref["id"] for ref in h.get("references", ()) if ref.get("kind") == "law"]
                if not gold_osh:
                    continue
                gold_set = set(gold_osh)
                total_haz += 1
                native_gold += len(gold_set)

                pred_native = [x["id"] for x in native_ret.select_osh_references(code, h["evidence"], h["prevention"], s_note, max_refs=2)]
                pred_native_set = set(pred_native)
                if pred_native and pred_native[0] in gold_set:
                    native_hit1 += 1
                    per_code_hits[code]["native_hit1"] += 1
                if pred_native_set == gold_set:
                    native_exact += 1
                native_tp += len(pred_native_set & gold_set)
                native_pred += len(pred_native_set)
                per_code_hits[code]["support"] += 1

                c8 = inv_8map.get(code, "")
                leg_raw = legacy_ret.select({"code": c8, "evidence": h["evidence"], "prevention": h["prevention"]}, s_note)[:1]
                pred_leg = [f"OSH-{x['article']}" for x in leg_raw if x.get("article")]
                pred_leg_set = set(pred_leg)
                if pred_leg and pred_leg[0] in gold_set:
                    legacy_hit1 += 1
                    per_code_hits[code]["legacy_hit1"] += 1
                if pred_leg_set == gold_set:
                    legacy_exact += 1
                legacy_tp += len(pred_leg_set & gold_set)
                legacy_pred += len(pred_leg_set)

    native_f1 = 2.0 * native_tp / max(native_gold + native_pred, 1)
    legacy_f1 = 2.0 * legacy_tp / max(native_gold + legacy_pred, 1)
    summary = {
        "evaluated_hazards_with_law_refs": total_haz,
        "legacy_8code_top1": {
            "hit_at_1": round(legacy_hit1 / max(total_haz, 1), 4),
            "exact_set_match": round(legacy_exact / max(total_haz, 1), 4),
            "micro_f1": round(legacy_f1, 4),
        },
        "native_11code_reranker_top1_2": {
            "hit_at_1": round(native_hit1 / max(total_haz, 1), 4),
            "exact_set_match": round(native_exact / max(total_haz, 1), 4),
            "micro_f1": round(native_f1, 4),
        },
        "per_code_hit_at_1": {
            c: {
                "support": per_code_hits[c]["support"],
                "legacy_hit1": per_code_hits[c]["legacy_hit1"],
                "native_hit1": per_code_hits[c]["native_hit1"],
            }
            for c in CODES
            if per_code_hits[c]["support"] > 0
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(out_path, lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    parser.add_argument("--out", type=Path, default=Path("results/law_reranker_11code_oof.json"))
    args = parser.parse_args()
    res = evaluate_5fold_law_retrieval(args.root.resolve(), args.manifest.resolve(), args.out.resolve())
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
