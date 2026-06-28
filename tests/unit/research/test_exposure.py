from __future__ import annotations

import numpy as np

from quant_platform.research.exposure import exposure_profile, simulate_exposure_paths


def test_exposure_profile_statistics() -> None:
    paths = np.array([[1.0, -2.0], [3.0, 4.0]])

    result = exposure_profile(paths, confidence=0.95)

    assert result["epe"] == 2.0
    assert result["ene"] == 0.5
    assert result["step_count"] == 2


def test_simulated_exposure_paths_are_reproducible() -> None:
    first = simulate_exposure_paths(2, 3, seed=5)
    second = simulate_exposure_paths(2, 3, seed=5)

    np.testing.assert_allclose(first, second)
