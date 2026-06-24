import numpy as np
import pandas as pd
import pytest

from hydrating.models import PowerLaw


@pytest.fixture(scope="module")
def powerlaw_data():
    stage = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    discharge = PowerLaw().func(stage.to_numpy(), a=0.1, h0=0.5, b=2.0)
    return pd.DataFrame({"stage": stage, "discharge": discharge})


@pytest.fixture(scope="module")
def powerlaw_reference_data():
    true_params = {"a": 0.1, "h0": 0.65, "b": 2.5}
    stage_exact = np.linspace(1.0, 10.0, 20)
    discharge_exact = PowerLaw().func(stage_exact, **true_params)

    rng = np.random.default_rng(12)
    stage_noisy = stage_exact + rng.normal(0, 0.01, size=stage_exact.size)
    discharge_noisy = discharge_exact + discharge_exact * rng.normal(
        0, 0.1, size=discharge_exact.size
    )

    return {
        "true_params": true_params,
        "stage_exact": stage_exact,
        "discharge_exact": discharge_exact,
        "stage_noisy": stage_noisy,
        "discharge_noisy": discharge_noisy,
    }
