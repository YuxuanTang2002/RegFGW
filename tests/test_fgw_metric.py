import networkx as nx
import numpy as np
import pytest
from regfgw.fgw_metric import FGWBuilder, FGWScoreParams, FGWScorer

def make_graph(features, distances):
    graph = nx.Graph()

    for index, feature in enumerate(features):
        graph.add_node(index, feature=feature)

    for (i, j), distance in distances.items():
        graph.add_edge(i, j, distance=distance)

    return graph

def test_build_input_matrices():
    graph1 = make_graph(
        features=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        distances={
            (0, 1): 2.0,
        },
    )
    graph2 = make_graph(
        features=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        distance={
            {0, 1}: 3.0,
        },
    )
    inputs = FGWBuilder.build_input_matrices(graph1, graph2)
    np.testing.assert_allclose(
        inputs.M,
        [
            [0.0, np.sqrt(2.0)],
            [np.sqrt(2.0), 0.0],
        ],
    )
    np.testing.assert_allclose(
        inputs.C1,
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ],
    )
    np.testing.assert_allclose(
        inputs.C2,
        [
            [0.0, 3.0],
            [3.0, 0.0],
        ],
    )
    np.testing.assert_allclose(inputs.p, [0.5, 0.5])
    np.testing.assert_allclose(inputs.q, [0.5, 0.5])

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("alpha", -0.1),
        ("alpha", 1.1),
        ("max_iter", 0),
        ("tol_rel", -0.1),
        ("tol_abs", -0.1),
        ("n_starts", 0),
        ("init_ipfp_tol", -0.1),
        ("init_ipfp_max_iter", 0),
    ],
)
def test_score_params(name, value):
    with pytest.raises(ValueError):
        FGWScoreParams(**{name: value})

def test_ipfp_projection():
    scorer = FGWScorer(builder=FGWBuilder(), score_params=FGWScoreParams())
    base = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )
    p = np.array([0.4, 0.6])
    q = np.array([0.3, 0.7])
    coupling = scorer.ipfp_projection(base, p, q)
    np.testing.assert_allclose(coupling.sum(axis=1), p, atol=1e-9)
    np.testing.assert_allclose(coupling.sum(axis=0), q, atol=1e-9)
