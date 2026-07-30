import numpy as np
from ase import Atoms
from regfgw.structure_to_graph import GraphEncoder

def test_load_element_embedding_csv(tmp_path):
    embedding_path = tmp_path / "embedding.csv"
    embedding_path.write_text(
        "feature_1,element,feature_2\n"
        "1.0,H,0.0\n"
        "0.0,He,1.0\n",
        encoding="utf-8",
    )
    lookup, feature_dim = GraphEncoder.load_element_embedding(embedding_path)
    assert feature_dim == 2
    np.testing.assert_allclose(lookup["H"], [1.0, 0.0])
    np.testing.assert_allclose(lookup["He"], [0.0, 1.0])

def test_load_element_embedding_json(tmp_path):
    embedding_path = tmp_path / "embedding.json"
    embedding_path.write_text(
        '{"H": [1.0, 0.0], "He": [0.0, 1.0]}',
        encoding="utf-8",
    )
    lookup, feature_dim = GraphEncoder.load_element_embedding(embedding_path)
    assert feature_dim == 2
    np.testing.assert_allclose(lookup["H"], [1.0, 0.0])
    np.testing.assert_allclose(lookup["He"], [0.0, 1.0])

def test_cluster_layers_by_z():
    atoms = Atoms(
        symbols=["H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0001],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0001],
        ],
    )
    layers = GraphEncoder.cluster_layers_by_z(atoms)
    assert layers == [[0, 1], [2, 3]]

def test_atoms_to_graph(tmp_path):
    embedding_path = tmp_path / "embedding.json"
    embedding_path.write_text(
        '{"H": [1.0, 0.0]}',
        encoding="utf-8",
    )
    encoder = GraphEncoder(embedding_path)
    atoms = Atoms(
        symbols=["H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, False],
    )
    graph = encoder.atoms_to_graph(atoms, selected_indices=[0, 1, 2], cutoff=2.0)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
