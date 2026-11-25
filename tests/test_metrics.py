import copy
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kwater_lab import metrics


def codes(*labels):
    return frozenset(labels)


def row(key, *labels):
    prediction = {"schema_version": "1.0", "image_id": key, "status": "hazard", "hazards": {}}
    for label in labels:
        prediction["hazards"][label] = {
            "evidence": "통로 바닥에 케이블이 보인다.", "prevention": "보행 경로에서 케이블을 정리한다.",
            "references": [{"kind": "law", "id": "OSH-3"}],
            "risk_level": {"likelihood": {"grade": "하", "score": 2},
                           "severity": {"grade": "소", "score": 1},
                           "risk": {"grade": "매우낮음", "score": 2}},
        }
    if not labels:
        prediction.update(status="no_visible_hazard", scene_evidence="보이는 범위에 통로를 막는 물체가 없다.")
    return prediction


class MetricsTests(unittest.TestCase):
    def test_documented_formula_cases(self):
        fall = codes("fall_height")
        slip = codes("slip_trip")
        electricity = codes("electrical_energy")
        empty = codes()
        for actual, predicted, expected in (
            ([fall, empty], [fall, empty], 1),
            ([fall, empty], [fall, codes("other")], 1/2),
            ([fall, fall, empty], [fall, empty, empty], 2/3),
            ([fall | slip, electricity, empty], [fall, electricity | slip, empty], 2/3),
        ):
            with self.subTest(expected=expected, actual=actual):
                result = metrics.evaluate(dict(enumerate(actual)), dict(enumerate(predicted)))
                self.assertAlmostEqual(result["macro_f1"], expected)
                self.assertAlmostEqual(result["quantitative_points_out_of_70"], expected*70)

    def test_empty_and_disjoint_predictions(self):
        result = metrics.evaluate({"a": codes()}, {"a": codes()})
        self.assertIsNone(result["macro_f1"])
        self.assertIsNone(result["quantitative_points_out_of_70"])
        self.assertIsNone(result["hazard_predicted_empty_rate"])
        self.assertEqual(result["no_hazard_false_alarm_rate"], 0)
        result = metrics.evaluate({"a": codes("fall_height"), "b": codes("other"), "c": codes()},
                                  {"a": codes(), "b": codes("slip_trip"), "c": codes("other")})
        self.assertEqual(result["hazard_predicted_empty_rate"], 0.5)
        self.assertEqual(result["no_hazard_false_alarm_rate"], 1)
        self.assertEqual(result["per_code"]["other"],
                         {"support": 1, "predicted": 1, "tp": 0, "fp": 1, "fn": 1, "f1": 0})
        self.assertIsNone(result["per_code"]["thermal_weather"]["f1"])

    def test_all_risk_combinations(self):
        likelihood = ("최하", "하", "중", "상", "최상")
        severity = ("소", "중", "대", "최대")
        risk = ["매우낮음"]*3 + ["낮음"]*3 + ["보통"]*2 + ["약간높음"]*4 + ["높음"]*3 + ["매우높음"]*5
        for (p, pl), (s, sl) in itertools.product(enumerate(likelihood, 1), enumerate(severity, 1)):
            metrics.check_level({"likelihood": {"score": p, "grade": pl},
                                 "severity": {"score": s, "grade": sl},
                                 "risk": {"score": p*s, "grade": risk[p*s-1]}})

    def test_reject_schema_errors(self):
        key = "1"*32
        positive = row(key, "slip_trip")
        bad = []
        for field, value in (("references", [{"kind": "law", "id": "제3조"}]),
                             ("prevention", "가"*481), ("evidence", "")):
            value_row = copy.deepcopy(positive)
            value_row["hazards"]["slip_trip"][field] = value
            bad.append(value_row)
        wrong_level = copy.deepcopy(positive)
        wrong_level["hazards"]["slip_trip"]["risk_level"]["risk"]["score"] = 3
        bool_score = copy.deepcopy(positive)
        bool_score["hazards"]["slip_trip"]["risk_level"]["severity"]["score"] = True
        bad.extend([wrong_level, bool_score, dict(positive, scene_evidence="없음"),
                    dict(positive, status="no_visible_hazard", scene_evidence="통로"),
                    row(key, "bad"), row(key, *metrics.CODES[:4]),
                    dict(row(key), schema_version=1.0), row("s1")])
        for entry in bad:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                metrics.decode_target(entry)

    def test_jsonl_strictness(self):
        key = "0"*32
        gold = {key: codes()}
        encoded = json.dumps(row(key)).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.jsonl"
            for malformed in (b"", b"\xef\xbb\xbf" + encoded, b"\n", b"{", b"\xff", b"[]",
                              b'{"status":"hazard","status":"no_visible_hazard"}',
                              b'{"value":NaN}', b'{"value":Infinity}', encoded + b"\n" + encoded,
                              json.dumps(row("f"*32)).encode(), encoded + b" "*2**20):
                with self.subTest(prefix=malformed[:60]), self.assertRaises(ValueError):
                    path.write_bytes(malformed)
                    metrics.load_predictions(path, gold)
            path.write_bytes(encoded)
            with self.assertRaisesRegex(ValueError, "missing=1"):
                metrics.load_predictions(path, {**gold, "1"*32: codes()})

    def test_official_kit_reference_and_text_contract(self):
        entry = row("3"*32, "other")
        item = entry["hazards"]["other"]
        del item["references"]
        self.assertEqual(metrics.decode_target(entry)[1], codes("other"))
        item["references"] = [{"kind": "case", "id": "B" + "9"*13 + "/" + "X"*20},
                              {"kind": "catalog", "id": "P" + "f"*14}]
        self.assertEqual(metrics.decode_target(entry)[1], codes("other"))
        for ref in ({"kind": "catalog", "id": "P" + "F"*14}, {"kind": "case", "id": "B123/x"}):
            item["references"] = [ref]
            with self.assertRaises(ValueError):
                metrics.decode_target(entry)
        item["references"] = []
        for invalid in ("근거\n줄바꿈", "\ufeff", "설명\u2028문장"):
            item["evidence"] = invalid
            with self.assertRaises(ValueError):
                metrics.decode_target(entry)

    def test_paired_resamples_use_whole_groups(self):
        gold = {"a": codes("fall_height"), "b": codes("fall_height"), "c": codes()}
        first = {"a": codes(), "b": codes(), "c": codes()}
        groups = {"a": "incident1", "b": "incident1", "c": "incident2"}
        result = metrics.paired_bootstrap(gold, first, gold, groups, resamples=50, seed=2)
        self.assertEqual(result["macro_f1_delta_b_minus_a"], 1)
        self.assertEqual(result["percentile_95_ci"], [1, 1])
        self.assertGreater(result["undefined_resamples"], 0)
        self.assertEqual(result["undefined_resamples"] + result["defined_resamples"], 50)
        self.assertEqual(metrics.paired_bootstrap(gold, first, gold, groups,
                                                  resamples=50, seed=2), result)
        self.assertEqual(metrics.evaluate(gold, first, groups)["per_code"]["fall_height"]["support_groups"], 1)
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            metrics.paired_bootstrap(gold, first, gold, dict.fromkeys(gold, "g"))

    def test_train_ids_to_held_out_manifest_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            gold_path = folder / "gold.jsonl"
            pred_path = folder / "pred.jsonl"
            ids_path = folder / "ids.jsonl"
            groups_path = folder / "groups.jsonl"

            def write(path, records):
                path.write_text("\n".join(json.dumps(v) for v in records), encoding="utf8")

            first = row("S16C159", "slip_trip")
            second = row("S16C147")
            write(gold_path, [{"sample_id": r["image_id"], "target": r} for r in (first, second)])
            write(pred_path, [second, first])
            gold = metrics.load_gold(gold_path)
            self.assertEqual(gold, metrics.load_predictions(pred_path, gold))
            first_id, second_id = "0"*32, "1"*32

            mapping = [{"sample_id": "S16C159", "image_id": first_id},
                       {"sample_id": "S16C147", "image_id": second_id}]
            write(ids_path, mapping)
            write(pred_path, [row(second_id), row(first_id, "slip_trip")])
            write(groups_path, [{"image_id": k, "group_id": f"case{i}"}
                                for i, k in enumerate((first_id, second_id))])
            command = [sys.executable, "-m", "kwater_lab.metrics", "--gold", str(gold_path),
                       "--pred", str(pred_path), "--compare", str(pred_path), "--id-map", str(ids_path),
                       "--groups", str(groups_path), "--resamples", "10"]
            output = json.loads(subprocess.check_output(command))
            self.assertEqual(output["a"]["macro_f1"], 1)
            self.assertEqual(output["paired"]["macro_f1_delta_b_minus_a"], 0)
            mapping[1]["image_id"] = first_id
            write(ids_path, mapping)
            with self.assertRaisesRegex(ValueError, "unique"):
                metrics.load_gold(gold_path, ids_path)
            with self.assertRaisesRegex(ValueError, "differs"):
                metrics.decode_gold({"sample_id": "wrong", "target": first})


if __name__ == "__main__":
    unittest.main()
