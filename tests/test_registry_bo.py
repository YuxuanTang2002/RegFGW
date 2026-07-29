import numpy as np
import pytest
from regfgw.registry_bo import BOParams, RegistryPriorBO

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_init", 0),
        ("n_iter", 0),
        ("acq_candidates", 0),
        ("xi", -0.1),
        ("penalty", -0.1),
    ],
)
def test_bo_params(name, value):
    with pytest.raises(ValueError):
        BOParams(**{name: value})

def test_periodic_embedding():
    with pytest.raises(ValueError):
        RegistryPriorBO.periodic_embedding(
            np.zeros((3, 3)),
        )
    coords = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.75],
            [1.0, 1.0]
        ],
    )
    embedded = RegistryPriorBO.periodic_embedding(coords)
    assert embedded.shape == (3, 4)
    np.testing.assert_allclose(embedded[0], embedded[2], atol=1e-12)
    np.testing.assert_allclose(embedded[1], [0.0, 1.0, 0.0, -1.0], atol=1e-12)
