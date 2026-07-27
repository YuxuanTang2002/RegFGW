import numpy as np
import pytest
from regfgw.interface_construction import InterfaceBuilder, InterfaceParams, ZSLParams
from pymatgen.core import Lattice, Structure

@pytest.mark.parametrize(
    ("name", "value"), 
    [
        ("max_area", 0.0),
        ("max_area_ratio_tol", -0.1),
        ("max_length_tol", -0.1),
        ("max_angle_tol", -0.1),
    ], 
)
def test_zsl_params(name, value):
    with pytest.raises(ValueError):
        ZSLParams(**{name: value})

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("film_layers", 0),
        ("substrate_layers", 0),
        ("gap", -0.1),
        ("vacuum", 0.0), 
    ]
)
def test_interface_params(name, value):
    with pytest.raises(ValueError):
        InterfaceParams(**{name: value})

def test_neg_miller():
    result = InterfaceBuilder.neg_miller((-1, 0, 1))
    assert result == (1, 0, -1)

def test_mirror_along_normal():
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.5, 0.5, 0.4],
            [0.2, 0.2, 0.8],
        ],
        dtype=float,
    )
    mirrored = InterfaceBuilder.mirror_along_normal(frac_coords, [0, 1, 2])
    np.testing.assert_allclose(mirrored[:, 2], [0.8, 0.6, 0.2])

def test_interface_area():
    structure = Structure(
        lattice=Lattice.orthorhombic(2.0, 3.0, 5.0),
        species=["H"],
        coords=[[0.0, 0.0, 0.0]],
    )
    area = InterfaceBuilder.interface_area(structure)
    assert area == pytest.approx(6.0)
