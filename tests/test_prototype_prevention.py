import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kwater_lab.metrics import decode_target
from kwater_lab.prototype import ELECTRICAL_PREVENTION, assemble, write_predictions


class PreventionTests(unittest.TestCase):
    def test_directory_output_selects_prevention_by_code(self):
        image_id = "a" * 32
        content = {
            "scene_observation": "작업 구역이 보인다.",
            "hazards": [
                {"code": "electrical_energy", "evidence": "전기 작업 구역이 보인다.",
                 "prevention": "모델의 임시 문장", "likelihood": 3, "severity": 4},
                {"code": "slip_trip", "evidence": "통로에 물건이 보인다.",
                 "prevention": "통로의 물건을 정리한다.", "likelihood": 2, "severity": 2},
            ],
        }
        response = {"finish_reason": "stop", "raw_output": json.dumps(content, ensure_ascii=False)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "images").mkdir()
            (path / "images" / f"{image_id}.jpg").write_bytes(b"local input fixture")
            (path / "manifest.jsonl").write_text(json.dumps({"image_id": image_id}) + "\n", encoding="utf-8")
            with patch("kwater_lab.prototype.generate", return_value=response):
                write_predictions(path, path / "output", 0, "fixture")
            outputs = list((path / "output").iterdir())
            self.assertEqual([file.name for file in outputs], ["submission.jsonl"])
            row = json.loads(outputs[0].read_text(encoding="utf-8"))
        self.assertEqual(decode_target(row), (image_id, frozenset({"electrical_energy", "slip_trip"})))
        electric = row["hazards"]["electrical_energy"]
        self.assertEqual(electric["prevention"], ELECTRICAL_PREVENTION)
        self.assertNotEqual(electric["prevention"], content["hazards"][0]["prevention"])
        self.assertEqual(electric["evidence"], content["hazards"][0]["evidence"])
        self.assertEqual(electric["references"], [])
        self.assertEqual(electric["risk_level"]["risk"], {"score": 12, "grade": "약간높음"})
        self.assertEqual(row["hazards"]["slip_trip"]["prevention"], content["hazards"][1]["prevention"])

    def test_no_hazard_retains_scene_evidence(self):
        row = assemble("b" * 32, {"scene_observation": "통로가 비어 있다.", "hazards": []})
        self.assertEqual(row["hazards"], {})
        self.assertEqual(row["status"], "no_visible_hazard")
        self.assertEqual(row["scene_evidence"], "통로가 비어 있다.")


if __name__ == "__main__":
    unittest.main()
