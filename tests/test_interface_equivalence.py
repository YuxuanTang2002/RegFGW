import pytest
from regfgw.interface_equivalence import InterfaceMatchParams, InterfaceMatcher

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ltol", -0.1),
        ("stol", -0.1),
        ("angle_tol", -0.1),
    ],
)
def test_match_params(name, value):
    with pytest.raises(ValueError):
        InterfaceMatchParams(**{name: value})

def test_group_interfaces(monkeypatch):
    matcher = InterfaceMatcher(InterfaceMatchParams())
    monkeypatch.setattr(
        matcher,
        "check_equivalence",
        lambda structure_a, structure_b: structure_a == structure_b,
    )
    groups = matcher.group_interfaces(
        ["A", "B", "A", "C", "B"]
    )
    member_in_groups = [group.member_indices for group in groups]
    assert member_in_groups == [[0, 2], [1, 4], [3]]
