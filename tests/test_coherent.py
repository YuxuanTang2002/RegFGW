import argparse
import pytest
from regfgw.cli.coherent import build_parser, select_interface_candidates, validate_args

def test_optimize_mode():
    parser = build_parser()
    args = argparse.Namespace(mode="optimize", embedding=None)
    with pytest.raises(SystemExit):
        validate_args(args, parser)

def test_select_interface_candidates(monkeypatch):
    interfaces = [
        {"cand_id": 10},
        {"cand_id": 20},
        {"cand_id": 30},
    ]
    monkeypatch.setattr("builtins.input", lambda _: "3, 1, 3")
    selected = select_interface_candidates(interfaces)
    assert selected == [{"cand_id": 10}, {"cand_id": 30}]
