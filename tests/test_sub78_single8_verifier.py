"""Coverage and precision boundaries for the frozen 8B verifier response."""

import unittest

from kwater_lab.sub78_single8_verifier import WEIGHT_HASH, check_response


class Single8VerifierTest(unittest.TestCase):
    def test_exact_order_precision_and_payload(self):
        cand = [{"sample_id": "x", "code": "a"}, {"sample_id": "y", "code": "b"}]
        response = {
            "manifest_sha256": "pinned", "precision": "nf4", "arm": "epoch-1",
            "adapter_sha256": WEIGHT_HASH["epoch-1"],
            "rows": [{"key": "x|a", "margin": 0.0, "raw": "a"},
                     {"key": "y|b", "margin": -1.0, "raw": "b"}],
        }
        args = (cand, "pinned", "nf4", "epoch-1")
        check_response(response, *args)
        for alteration in (
            {"rows": response["rows"][::-1]},
            {"rows": response["rows"][:1]},
            {"rows": [response["rows"][0]] * 2},
            {"rows": [dict(response["rows"][0], margin=float("nan")), response["rows"][1]]},
            {"rows": [dict(response["rows"][0], raw=None), response["rows"][1]]},
            {"precision": "bf16"},
            {"adapter_sha256": WEIGHT_HASH["epoch-2"]},
            {"manifest_sha256": "other"},
        ):
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                check_response({**response, **alteration}, *args)
