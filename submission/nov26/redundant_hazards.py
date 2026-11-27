"""Suppress a redundant item only when both generated descriptions point to an existing code."""

import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.rag import _features, _SparseIndex


class RedundancyCheck:
    def __init__(self, config):
        self.cases = config["cases"]
        self.eligible = set(config["eligible_codes"])
        self.evidence_threshold = config["evidence_threshold"]
        self.prevention_threshold = config["prevention_threshold"]
        self.keys = ("evidence", "prevention")
        self.indexes = [_SparseIndex([_features(row[key]) for row in self.cases]) for key in self.keys]

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def similarity(self, field, a, b):
        index = self.indexes[field]
        va = index._vector(_features(a))
        vb = index._vector(_features(b))
        return sum(v * vb.get(k, 0) for k, v in va.items())

    def ranked_codes(self, field, text):
        by_code = defaultdict(dict)
        for idx, sim in self.indexes[field].scores(_features(text)).items():
            row = self.cases[idx]
            groups = by_code[row["code"]]
            groups[row["group"]] = max(groups.get(row["group"], 0), sim)
        ranked = []
        for code, groups in by_code.items():
            top = sorted(groups.values(), reverse=True)[:2]
            if len(top) == 2:
                ranked.append((code, sum(top) / 2))
        return sorted(ranked, key=lambda r: (-r[1], r[0]))

    def supported_code(self, h):
        proposal = None
        for i, key in enumerate(self.keys):
            rank = self.ranked_codes(i, h[key])
            if len(rank) < 2 or rank[0][1] < 0.18 or rank[0][1] - rank[1][1] < 0.035:
                return None
            if proposal is not None and proposal != rank[0][0]:
                return None
            proposal = rank[0][0]
        return proposal

    def independent_observation(self, hazard, survivor):
        for clause in re.split(r"[,.;!?\n]", hazard["evidence"]):
            if len(re.findall("[가-힣]", clause)) < 5:
                continue
            if self.similarity(0, clause, survivor["evidence"]) >= self.evidence_threshold:
                continue
            ranked = self.ranked_codes(0, clause)
            if ranked and ranked[0][0] == hazard["code"]:
                return True
        return False

    def reconcile(self, prediction):
        hazards = prediction["hazards"]
        if len(hazards) < 2:
            return
        supports = [self.supported_code(h) for h in hazards]
        anchored = {h["code"]:h for h, code in zip(hazards, supports)
                    if h["code"] == code and code in self.eligible}
        filtered = []
        for hazard, proposed in zip(hazards, supports):
            original = hazard["code"]
            survivor = anchored.get(proposed)
            if (survivor is None or proposed == original or original not in self.eligible
                    or self.similarity(0, hazard["evidence"], survivor["evidence"]) < self.evidence_threshold
                    or self.similarity(1, hazard["prevention"], survivor["prevention"]) < self.prevention_threshold
                    or self.independent_observation(hazard, survivor)):
                filtered.append(hazard)
        prediction["hazards"] = filtered
