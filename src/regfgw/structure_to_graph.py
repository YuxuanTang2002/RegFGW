import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from ase import Atoms
from ase.neighborlist import neighbor_list
from collections import defaultdict
from pathlib import Path
from pymatgen.core.interface import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Dict, Any, List, DefaultDict, Tuple

# -----------------------------------------------------------------------------
# Main encoder
# -----------------------------------------------------------------------------

class GraphEncoder:
    """
    Convert pymatgen Interface objects into graph pairs for FGW-style usage

    Notes
    -----
    * Node features: radial basis expansion(RBF) of neighbor distances +  element embedding aggregation
    * Edge features: minimum image convention(MIC) distance between selected atoms (a fully-connected graph)
    * PBC convention: (True, True, False) for slab-like systems
    """
    def __init__(self, embedding_path: str | Path | None = None):
        self.feature_lookup = None
        self.feature_dim = None

        if embedding_path is None:
            raise ValueError("embedding_path must be provided (.csv/.json).")

        self.feature_lookup, self.feature_dim = self.load_element_embedding(embedding_path)
        self.cutoff = None

    # -----------------------------------------------------------------------------
    # Cutoff determination
    # -----------------------------------------------------------------------------

    @staticmethod
    def compute_third_shell_distance(atoms: Atoms, selected_indices: List[int]):
        """Estimate a characteristic cutoff distance based on the third coordination shell."""
        d_matrix = atoms.get_all_distances(mic=True)
        r3_max = None

        for i in selected_indices:
            ds = np.sort(d_matrix[i][d_matrix[i] > 1e-9])
            if ds.size == 0:
                continue
            shells = [[float(ds[0])]]
            ref = float(ds[0])
            for d in ds[1:]:
                if abs(float(d)-ref) <= 0.1:
                    shells[-1].append(d)
                else:
                    shells.append([d])
                    ref = d
                if len(shells) > 3:
                    break
            if len(shells) < 3:
                raise ValueError("Less than 3 shell layers. Increase slab size.")
            r3_i = max(shells[2])
            if r3_max is None or r3_i > r3_max:
                r3_max = r3_i

        if r3_max is None:
            raise ValueError("No valid centers with neighbours.")

        return r3_max

    # -------------------------------------------------------------------------
    # Element embedding
    # -------------------------------------------------------------------------

    @staticmethod
    def load_element_embedding(embedding_path: str | Path):
        """
        Load element embedding vectors from a CSV or JSON file.

        CSV format: contain column 'element' and other columns are embedding dimensions.
        JSON format: {"H": [...], "C": [...], ...}
        """
        embedding_path = Path(embedding_path)

        if embedding_path.suffix == ".csv":
            feature_df = pd.read_csv(embedding_path)
            if "element" not in feature_df.columns:
                raise ValueError("Embedding CSV must contain a column named 'element'.")
            feature_lookup = {
                row["element"]: np.array(row.iloc[1:], dtype=float)
                for _, row in feature_df.iterrows()
            }
        elif embedding_path.suffix == ".json":
            with open(embedding_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            feature_lookup = {elem: np.array(feats, dtype=float) for elem, feats in data.items()}
        else:
            raise ValueError("Supported embedding file format: .csv or .json")

        if not feature_lookup:
            raise ValueError("Empty embedding file.")

        dims = {v.shape for v in feature_lookup.values()}

        if len(dims) != 1:
            raise ValueError(f"Inconsistent embedding dimensions: {dims}")

        feature_dim = next(iter(dims))[0]

        return feature_lookup, feature_dim

    # -------------------------------------------------------------------------
    # Layer selection
    # -------------------------------------------------------------------------

    @staticmethod
    def cluster_layers_by_z(atoms: Atoms):
        """
        Cluster atoms into layers by identical (within tolerance) z coordinate.
        Return a list of layers, each layer is a list of atom indices (in the original atoms indexing).
        """
        n = len(atoms)

        if n == 0:
            return []

        z = atoms.get_positions()[:, 2].astype(float)
        idx_sorted = np.argsort(z).astype(int)
        z_sorted = z[idx_sorted] #z[idx_sorted] = [z[idx_sorted[0]], z[idx_sorted[1]], ..., z[idx_sorted[n-1]]]
        layers: List[List[int]] = []
        current = [int(idx_sorted[0])]
        z_ref = float(z_sorted[0])

        for i in range(1, n):
            zi = float(z_sorted[i])
            if abs(zi - z_ref) <= 1e-3:
                current.append(int(idx_sorted[i]))
            else:
                layers.append(current)
                current = [int(idx_sorted[i])]
                z_ref = zi

        layers.append(current)

        return layers

    @staticmethod
    def group_layers_into_periods(layers: List[List[int]], period_layers: int):
        """Group consecutive layers into blocks(periods)."""
        if period_layers <= 0:
            raise ValueError("period_layers must be > 0")

        nl = len(layers)
        periods: List[List[int]] = []
        npd = nl // period_layers

        for p in range(npd):
            s = p * period_layers
            e = s + period_layers
            block_layers = layers[s:e]
            atom_idx = sorted([int(a) for layer in block_layers for a in layer])
            periods.append(atom_idx)

        return periods

    def select_bulk_core_indices(self, bulk_atoms: Atoms, period_layers: int):
        """Pick the middle period as the bulk core."""
        layers = self.cluster_layers_by_z(bulk_atoms)
        periods = self.group_layers_into_periods(layers, period_layers)

        if not periods:
            return []

        bulk_core = periods[len(periods) // 2]

        return bulk_core

    def select_sided_interface_indices(
            self,
            sub_atoms: Atoms, film_atoms: Atoms,
            sub_period_layers: int, film_period_layers: int,
    ):
        """
        Pick last period (near interface side) of substrate and first period (near interface side) of film.
        Both are returned in the local indexing of sub_atoms/film_atoms.
        """
        sub_layers = self.cluster_layers_by_z(sub_atoms)
        sub_periods = self.group_layers_into_periods(sub_layers, sub_period_layers)
        sub_pick = sub_periods[-1] if sub_periods else []
        film_layers = self.cluster_layers_by_z(film_atoms)
        film_periods = self.group_layers_into_periods(film_layers, film_period_layers)
        film_pick = film_periods[0] if film_periods else []
        return sub_pick, film_pick

    # -------------------------------------------------------------------------
    # Feature construction
    # -------------------------------------------------------------------------

    def compute_rbf_el_features(
            self,
            atoms: Atoms,
            selected_indices: List[int],
            cutoff: float,
            n_rbf: int,
    ):
        """
        Compute node features for selected atoms.

        Feature for each selected atom i: sum_{j in neigh(i)}  f_c(d_ij) * exp(-gamma*(d_ij-c_r)^2) * emb(elem_j)
        * c_r: evenly spaced centers in [0, cutoff]
        * f_c: cosine cutoff (0 outside cutoff)

        Returns
        -------
        rbf_el: (n_sel, n_rbf * emb_dim)
        """
        centers = np.linspace(0.0, cutoff, n_rbf, dtype=float)
        width = cutoff / max(n_rbf - 1, 1)
        gamma = 1.0 / (width * width + 1e-12)
        symbols: List[str] = atoms.get_chemical_symbols()

        if self.feature_lookup is None:
            raise ValueError("GraphEncoder.feature_lookup is None. Pass embedding_path to GraphEncoder.")

        emb_dim = int(self.feature_dim)
        rc = cutoff + 1.0
        i_list, j_list, d_list = neighbor_list("ijd", atoms, rc)
        bucket: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)

        for i_center, j_neigh, dist in zip(i_list, j_list, d_list):
            bucket[int(i_center)].append((int(j_neigh), float(dist)))

        n_sel = len(selected_indices)
        rbf_el = np.zeros((n_sel, n_rbf * emb_dim), dtype=float)

        for ii, i_idx in enumerate(selected_indices):
            neigh = bucket.get(int(i_idx), [])
            if not neigh:
                continue

            js = [j for j, _ in neigh]
            ds = np.array([d for _, d in neigh], dtype=float)
            w = np.zeros_like(ds)
            mask = ds < cutoff

            if np.any(mask):
                w[mask] = 0.5 * (np.cos(np.pi * ds[mask] / cutoff) + 1.0)

            diff2 = (ds[None, :] - centers[:, None]) ** 2
            rbf = np.exp(-gamma * diff2)
            w_rbf = w[None, :] * rbf
            feat = np.zeros((n_rbf, emb_dim), dtype=float)

            for jj, j_idx in enumerate(js):
                elem = symbols[int(j_idx)]
                if elem not in self.feature_lookup:
                    raise ValueError(f"Element '{elem}' not found in embedding.")
                emb = self.feature_lookup[elem]
                feat += w_rbf[:, jj][:, None] * emb[None, :]

            rbf_el[ii] = feat.reshape(-1)

        return rbf_el

    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------

    def atoms_to_graph(
            self,
            atoms: Atoms,
            selected_indices: List[int],
            cutoff: float,
    ):
        """Build a fully-connected graph on selected atoms."""
        sel = [int(i) for i in selected_indices]
        atoms = atoms.copy()
        atoms.pbc = (True, True, False)
        atoms.wrap()
        n_atoms = len(atoms)

        if len(sel) == 0:
            raise ValueError("selected_indices is empty.")

        if len(set(sel)) != len(sel):
            raise ValueError("selected_indices contains duplicates.")

        if min(sel) < 0 or max(sel) >= int(n_atoms):
            raise IndexError(f"selected_indices out of range [0, {n_atoms - 1}].")

        n_sel = len(sel)
        n_rbf = int(np.ceil(3.0 * cutoff))
        rbf_el = self.compute_rbf_el_features(atoms, sel, cutoff=cutoff, n_rbf=n_rbf)
        g = nx.Graph()

        for node_i in range(n_sel):
            g.add_node(int(node_i), feature=[float(x) for x in rbf_el[node_i]])

        atoms = atoms.copy()
        atoms.pbc = (True, True, False)
        atoms.wrap()
        d = atoms.get_all_distances(mic=True)

        for i in range(n_sel):
            gi = sel[i]
            for j in range(i + 1, n_sel):
                gj = sel[j]
                dij = float(d[gi, gj])
                if not np.isfinite(dij) or dij < 0.0:
                    raise ValueError(f"Invalid MIC distance for pair ({gi},{gj}): {dij}")
                g.add_edge(int(i), int(j), distance=dij)

        return g

    # -------------------------------------------------------------------------
    # Structure dump for debugging
    # -------------------------------------------------------------------------

    @staticmethod
    def build_base_path(interface: Dict[str, Any]):
        """Build a filename stem from interface metadata."""
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        s = interface["substrate_miller"]
        f = interface["film_miller"]
        t = interface["termination"]
        cand_id = interface["cand_id"]
        sub_tag = f"{s[0]}{s[1]}{s[2]}"
        film_tag = f"{f[0]}{f[1]}{f[2]}"
        f_t, s_t = t
        f_t = f_t.replace("/", "-")
        s_t = s_t.replace("/", "-")
        term_tag = f"{s_t}_{f_t}"
        fname = f"sub{sub_tag}_film{film_tag}_term{term_tag}_cand{cand_id}"
        path = os.path.join(out_dir, fname)
        return path

    @staticmethod
    def write_subset_cif(atoms: Atoms, indices, path: str, suffix: str):
        """Write a subset of atoms to CIF (for structure sanity check)."""
        sub_atoms = atoms[list(indices)]
        out_path = f"{path}_{suffix}.cif"
        sub_atoms.write(out_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def prepare_bulk_cache(
            self,
            interface: Dict[str, Any],
            sub_period_layers: int,
            film_period_layers: int,
            structure_check: bool = False,
    ):
        sub_bulk_struct = interface["substrate_bulk"]
        film_bulk_struct = interface["film_bulk"]

        if sub_bulk_struct is None or film_bulk_struct is None:
            raise ValueError("substrate bulk / film bulk is None. Build bulk references first.")

        # Convert bulks to ASE Atoms
        sub_bulk_atoms: Atoms = AseAtomsAdaptor.get_atoms(sub_bulk_struct)
        film_bulk_atoms: Atoms = AseAtomsAdaptor.get_atoms(film_bulk_struct)
        # Select indices in the middle of bulks (bulk cores)
        sub_bulk_core = self.select_bulk_core_indices(sub_bulk_atoms, sub_period_layers)
        film_bulk_core = self.select_bulk_core_indices(film_bulk_atoms, film_period_layers)

        # Optional debugging dumps
        if structure_check:
            base_path = self.build_base_path(interface)
            self.write_subset_cif(sub_bulk_atoms, sub_bulk_core, base_path, "sub_bulk_core")
            self.write_subset_cif(film_bulk_atoms, film_bulk_core, base_path, "film_bulk_core")

        # Compute cutoff from third shell distances
        r3_sub = self.compute_third_shell_distance(sub_bulk_atoms, sub_bulk_core)
        r3_film = self.compute_third_shell_distance(film_bulk_atoms, film_bulk_core)
        cutoff = max(r3_sub, r3_film)
        self.cutoff = cutoff
        # Build bulk graphs
        g_sub_bulk = self.atoms_to_graph(sub_bulk_atoms, sub_bulk_core, cutoff=cutoff)
        g_film_bulk = self.atoms_to_graph(film_bulk_atoms, film_bulk_core, cutoff=cutoff)

        return g_sub_bulk, g_film_bulk

    def build_sided_interface_graphs(
            self,
            interface: Dict[str, Any],
            sub_period_layers: int,
            film_period_layers: int,
            structure_check: bool = False,
    ):
        interface_struct: Interface = interface["interface"]
        itf_atoms: Atoms = AseAtomsAdaptor.get_atoms(interface_struct)
        sub_idx = np.array(interface_struct.substrate_indices, dtype=int)
        film_idx = np.array(interface_struct.film_indices, dtype=int)
        # Local atoms objects for layer clustering.
        sub_atoms = itf_atoms[sub_idx]
        film_atoms = itf_atoms[film_idx]
        sub_pick_local, film_pick_local = self.select_sided_interface_indices(
            sub_atoms, film_atoms,
            sub_period_layers=sub_period_layers, film_period_layers=film_period_layers,
        )
        # Map local indices in sub_atoms/film_atoms to global indices in itf_atoms,
        sub_pick_global = [int(sub_idx[k]) for k in sub_pick_local]
        film_pick_global = [int(film_idx[k]) for k in film_pick_local]

        # Optional debugging dumps
        if structure_check:
            base_path = self.build_base_path(interface)
            self.write_subset_cif(itf_atoms, sub_pick_global, base_path, "sub_pick_global")
            self.write_subset_cif(itf_atoms, film_pick_global, base_path, "film_pick_global")

        cutoff = self.cutoff

        if cutoff is None:
            raise RuntimeError("cutoff not cached. Call prepare_bulk_cache() first.")

        g_itf_sub = self.atoms_to_graph(itf_atoms, sub_pick_global, cutoff=cutoff)
        g_itf_film = self.atoms_to_graph(itf_atoms, film_pick_global, cutoff=cutoff)

        return g_itf_sub, g_itf_film

