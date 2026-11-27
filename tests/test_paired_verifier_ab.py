"""CPU-only tests for the approved paired-verifier data boundary."""

from copy import deepcopy
import unittest

from kwater_lab.paired_verifier_ab import control_matches, eligible_rows, validate_slots, verified_source_pairs


def photo(sid, fold=1, group=None, codes=(), family="S_C"):
    return {"sample_id": sid, "split": "dev", "dev_fold": fold, "group_id": group or sid,
            "image": sid + ".png", "sha256": sid, "codes": list(codes), "family": family}


def test_pool():
    pairs = []
    slots = []
    rows = []
    for i in range(4):
        positive, negative, control = (
            photo(f"pos-{i}", codes=("fall_height",)),
            photo(f"neg-{i}"),
            photo(f"orig-{i}"),
        )
        records = {"code": "fall_height", "control": "no_visible"}
        pairs.append({**records, "members": [positive, negative]})
        slots += [
            {**records, "decision": "accept", "A": positive["sample_id"], "B": positive["sample_id"], "treatment_review": "approved"},
            {**records, "decision": "reject", "A": control["sample_id"], "B": negative["sample_id"], "treatment_review": "approved"},
        ]
        rows.extend((positive, negative, control))
    return rows, slots, pairs


class PairedVerifierDataTests(unittest.TestCase):
    def test_protected_folds_and_confirmed_train_group_excluded(self):
        rows = [photo("allowed"), photo("fold0", 0), photo("fold4", 4), photo("linked", 3)]
        self.assertEqual([r["sample_id"] for r in eligible_rows(rows, {"linked"})], ["allowed"])
        rows[-1]["sha256"] = rows[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "fingerprint crossing"):
            eligible_rows(rows, {"linked"})

    def test_balanced_directions_and_no_duplicate_or_unapproved_slot(self):
        rows, slots, pairs = test_pool()
        canonical = {r["sample_id"]: r for r in rows}
        index = verified_source_pairs(pairs)
        report = validate_slots(slots, canonical, index)
        self.assertEqual(report["photos_per_arm"], {"A": 8, "B": 8})
        self.assertFalse(report["pilot_only"])
        self.assertTrue(validate_slots(slots[:2], canonical, index)["pilot_only"])
        for mutate in (
            lambda x: x[1].update(A=x[3]["A"]),
            lambda x: x[0].update(decision="reject"),
            lambda x: x[1].update(B="protected"),
            lambda x: x[0].update(treatment_review="uncertain"),
        ):
            copy = deepcopy(slots)
            mutate(copy)
            with self.assertRaises(ValueError):
                validate_slots(copy, canonical, index)

    def test_unique_ordinary_negative_controls_preserve_source_and_scene(self):
        rows, _, source = test_pool()
        reserved = {r["sample_id"] for p in source for r in p["members"]}
        reviewed = [(p, p["members"][1]) for p in source]
        first = control_matches(rows, reviewed, reserved)
        self.assertEqual(set(first), {f"orig-{i}" for i in range(4)})
        self.assertEqual(first, control_matches(list(reversed(rows)), reviewed, reserved))
        with self.assertRaisesRegex(ValueError, "No unique"):
            control_matches(rows[:-1], reviewed, reserved)


if __name__ == "__main__":
    unittest.main()
