"""Exercise the regular spatial-head training path on a real CUDA device."""

import numpy as np
import pytest
import torch

from kwater_lab.spatial_probe import _fit_scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("regional,top_k", [(False, 1), (True, 1), (True, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_head_training_preserves_input_dtype_and_learns_separable_codes(
    regional: bool, top_k: int, dtype: type[np.float32] | type[np.float64]
) -> None:
    features = np.zeros((4, 16, 2), dtype=dtype)
    features[:, :, 0] = np.asarray([-2, -1, 1, 2], dtype=dtype)[:, None]
    labels = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=dtype)

    scores = _fit_scores(
        features,
        labels,
        features,
        regional=regional,
        top_k=top_k,
        seed=17,
        steps=30,
        device_name="cuda",
    )

    assert scores.dtype == dtype
    np.testing.assert_array_equal(scores > 0, labels.astype(bool))
