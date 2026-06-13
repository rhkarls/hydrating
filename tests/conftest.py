import pandas as pd
import pytest

from hydrating import models


@pytest.fixture(scope="module")
def powerlaw_data():
    stage = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    discharge = models.PowerLaw().func(stage.to_numpy(), a=0.1, h_zero=0.5, b=2.0)
    return pd.DataFrame({"stage": stage, "discharge": discharge})
