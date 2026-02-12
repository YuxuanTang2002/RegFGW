import numpy as np
import networkx as nx
import ot
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FGWInputs:
    """
    Inputs required by POT's fused_gromov_wasserstein2:
    * M: (n1, n2) feature cost matrix
    * C1: (n1, n1) structure cost matrix for graph1
    * C2: (n2, n2) structure cost matrix for graph2
    * p: (n1,) probability distribution for graph1 nodes
    * q: (n2,) probability distribution for graph2 nodes
    """
    M: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    p: np.ndarray
    q: np.ndarray

@dataclass(frozen=True)
class FGWBuildParams:
    """
    Parameters controlling how a NetworkX graph is transferred into FGW input matrices
    * node_attr: node attribute storing node feature vectors
    * edge_attr: edge attribute storing pairwise distances
    * feature_metric: metric passed to scipy.spatial.distance.cdist for building M (feature-cost matrix)
    """
    node_attr: str = "feature"
    edge_attr: str = "distance"
    feature_metric: str = "euclidean"

# -----------------------------------------------------------------------------
# FGW matrix builder
# -----------------------------------------------------------------------------

class FGWBuilder:
    def __init__(self, params: FGWBuildParams = FGWBuildParams()):
        self.params = params

    @staticmethod
    def check_dense_labels(g: nx.Graph):
        """Enforce nodes are labeled as 0, 1, .., n-1, for mapping matrices back to graph nodes."""
        n = int(g.number_of_nodes())

        if n == 0:
            raise ValueError("Graph has 0 node.")

        if set(g.nodes()) != set(range(n)):
            raise RuntimeError(f"Graph nodes must have dense labels 0, ..., {n-1}.")

        return n

    def extract_feature_matrix(self, g: nx.Graph):
        """Build X (n, d) feature matrix from node attribute."""
        n = self.check_dense_labels(g)
        attr = self.params.node_attr
        rows = []
        dim = None

        for i in range(n):
            if attr not in g.nodes[i]:
                raise ValueError(f"Node {i} missing '{attr}' attribute")
            v = np.asarray(g.nodes[i][attr], dtype=float).ravel()
            if dim is None:
                dim = v.shape[0]
                if dim == 0:
                    raise ValueError(f"Node attribute '{attr}' cannot be empty")
            elif v.shape[0] != dim:
                raise RuntimeError(f"Inconsistent '{attr}' length: node {i} has {v.shape[0]}, expect {dim}")
            rows.append(v)

        return np.stack(rows, axis=0)

    def extract_structure_matrix(self, g: nx.Graph):
        """
        Build C (n, n) structure matrix from edge attribute.

        Notes
        -----
        A complete graph (edge exists for every i<j) is required.
        """
        n = self.check_dense_labels(g)
        attr = self.params.edge_attr
        c = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i+1, n):
                if not g.has_edge(i, j):
                    raise ValueError(f"Missing edge ({i},{j}). A complete graph required.")
                d = g[i][j].get(attr, None)
                if d is None:
                    raise ValueError(f"Edge ({i},{j}) missing '{attr}' attribute")
                d = float(g.edges[i,j][attr])
                if not np.isfinite(d) or d < 0.0:
                    raise ValueError(f"Invalid '{attr}' on edge ({i},{j}): {d}")
                c[i, j] = d
                c[j, i] = d

        return c

    def extract_mass_matrix(self, g: nx.Graph):
        """
        Uniform weight distribution over nodes.
        Treat each node as equally important when matching.
        """
        n = self.check_dense_labels(g)
        return np.ones(n, dtype=float) / n

    def build_input_matrices(self, g_itf: nx.Graph, g_bulk: nx.Graph):
        """
        Construct FGWInputs (M, C1, C2, p, q) from two graphs.
        * C1/C2: edge distance matrices (structure term)
        * M: cdist(features_interface, features_bulk) (feature term)
        * p/q: Uniform node masses
        """
        c1 = self.extract_structure_matrix(g_itf)
        c2 = self.extract_structure_matrix(g_bulk)
        x1 = self.extract_feature_matrix(g_itf)
        x2 = self.extract_feature_matrix(g_bulk)
        m = cdist(x1, x2, metric=self.params.feature_metric)
        p = self.extract_mass_matrix(g_itf)
        q = self.extract_mass_matrix(g_bulk)
        return FGWInputs(M=m, C1=c1, C2=c2, p=p, q=q)

# -----------------------------------------------------------------------------
# FGW scorer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FGWScoreParams:
    """
    Parameters passed to POT's fused_gromov_wasserstein2

    Parameters
    ----------
    alpha: interpolation between feature term (M) and structure term (C1/C2)
    n_starts: number of initial couplings tried, the minimum objective is selected
    init_seed/init_ipfp_tol/init_ipfp_max_iter: settings for projecting random matrices to satisfy marginals (p, q)
    """
    alpha: float = 0.5
    loss_fun: str = "square_loss"
    symmetric: bool = True
    max_iter: int = 10000
    tol_rel: float = 1e-9
    tol_abs: float = 1e-9
    armijo: bool = True
    log: bool = False
    n_starts: int = 1
    init_seed: int = 0
    init_ipfp_tol: float = 1e-10
    init_ipfp_max_iter: int = 1000

class FGWScorer:
    def __init__(self, builder: FGWBuilder, score_params=FGWScoreParams()):
        self.builder = builder
        self.params = score_params
        # RNG is used for generating initial couplings in multi-start
        self.rng = np.random.default_rng(self.params.init_seed)
        # Cache random initial bases: key = (n1, n2, n_starts)
        self.base_cache: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        # Cache normalization scales per bulk graph: key = id(g_bulk)
        self.norm_cache: Dict[int, Tuple[float, float]] = {}
        # After normalization, enforce feature/structure scales are comparable to avoid FGW is dominated by one term.
        self.scale_ratio_max = 10.0

    # -----------------------------------------------------------------------------
    # Initialization  helpers
    # -----------------------------------------------------------------------------

    def get_fixed_base_list(self, n1: int, n2: int, n_starts: int):
        """
        Return a fixed list of random base matrices base_k (k=1, ..., n_start-1),
        k=0 is reserved for outer(p, q)
        """
        if n_starts < 2:
            return []

        key = (n1, n2, n_starts)
        if key in self.base_cache:
            return self.base_cache[key]

        base_list = []
        for _ in range(n_starts - 1):
            base_list.append(self.rng.random((n1, n2)))

        self.base_cache[key] = base_list
        return base_list

    def ipfp_projection(self, base: np.ndarray, p: np.ndarray, q: np.ndarray):
        """
        IPFP projection with convergence check,
        enforce g.sum(axis=1) == p and g.sum(axis=0) == q,
        raise RuntimeError if not converged.
        """
        g0 = np.asarray(base, dtype=float).copy()
        g0 = np.maximum(g0, 1e-12)
        r_err = np.inf
        c_err = np.inf

        for _ in range(self.params.init_ipfp_max_iter):
            row_sum = g0.sum(axis=1)
            g0 = g0 * (p / np.maximum(row_sum, 1e-300))[:, None]
            col_sum = g0.sum(axis=0)
            g0 = g0 * (q / np.maximum(col_sum, 1e-300))[None, :]
            r_err = np.sum(np.abs(g0.sum(axis=1) - p))
            c_err = np.sum(np.abs(g0.sum(axis=0) - q))

            if max(r_err, c_err) <= self.params.init_ipfp_tol:
                return g0

        raise RuntimeError(
            f"IPFP did not converge after {self.params.init_ipfp_max_iter} iterations "
            f"residual={max(r_err, c_err):.3e}"
        )

    # -----------------------------------------------------------------------------
    # Normalization  helpers
    # -----------------------------------------------------------------------------

    def normalize_inputs(self, g_bulk, M, C1, C2):
        """Normalize (M, C1, C2) to improve numerical stability and comparability."""
        def average_finite(m):
            if not np.all(np.isfinite(m)):
                raise ValueError("Non-finite values found in feature-cost matrix")

            mu = float(m.mean())

            if mu < 1e-12:
                raise ValueError("Scale too small, invalid feature input")

            return mu

        def average_upper_finite(c):
            iu = np.triu_indices(c.shape[0], k=1)
            vals = c[iu]

            if not np.all(np.isfinite(vals)):
                raise ValueError("Non-finite upper-triangle values found in structure matrix")

            mu = float(vals.mean())

            if mu < 1e-12:
                raise ValueError("Scale too small, invalid structure input")

            return mu

        key = id(g_bulk)
        scales = self.norm_cache.get(key)

        if scales is None:
            m_scale = average_finite(M)
            c_scale = 0.5 * (average_upper_finite(C1) + average_upper_finite(C2))
            scales = (m_scale, c_scale)
            self.norm_cache[key] = scales

        m_scale, c_scale = scales
        m_norm = M / m_scale
        c1_norm = C1 / c_scale
        c2_norm = C2 / c_scale
        # Comparability check (after normalization)
        m_mu = average_finite(m_norm)
        c1_mu = average_upper_finite(c1_norm)
        c2_mu = average_upper_finite(c2_norm)
        mus = np.array([m_mu, c1_mu, c2_mu], dtype=float)
        ratio = mus.max() / mus.min()

        if ratio > self.scale_ratio_max:
            raise ValueError(f"Scale mismatch: mean(M)={m_mu:.3e}, mean(C1)={c1_mu:.3e}, mean(C2)={c2_mu:.3e}")

        return m_norm, c1_norm, c2_norm

    # -----------------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------------

    def one_side_fgw(self, g_itf, g_bulk):
        """
        FGW distance between one-sided interface graph and its corresponding bulk graph.

        Multi-start: run FGW from multiple initial couplings g0 and take the min value.
        """
        inputs = self.builder.build_input_matrices(g_itf, g_bulk)
        m, c1, c2 = self.normalize_inputs(g_bulk, inputs.M, inputs.C1, inputs.C2)
        p, q = inputs.p, inputs.q
        g0_base = np.outer(inputs.p, inputs.q)
        n_starts = int(self.params.n_starts)

        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")

        # n_starts == 1, always start with a neutral starting point (g0_base)
        out = ot.gromov.fused_gromov_wasserstein2(
            m, c1, c2, p, q,
            loss_fun=self.params.loss_fun,
            alpha=self.params.alpha,
            symmetric=self.params.symmetric,
            armijo=self.params.armijo,
            log=self.params.log,
            G0=g0_base,
            max_iter=self.params.max_iter,
            tol_rel=self.params.tol_rel,
            tol_abs=self.params.tol_abs,
        )
        best_score = float(out[0] if isinstance(out, tuple) else out)
        # n_start > 1, start with cached random base list
        n1, n2 = inputs.p.shape[0], inputs.q.shape[0]
        base_list = self.get_fixed_base_list(n1, n2, n_starts=n_starts)

        for base in base_list:
            g0 = self.ipfp_projection(base, inputs.p, inputs.q)
            out = ot.gromov.fused_gromov_wasserstein2(
                m, c1, c2, p, q,
                loss_fun=self.params.loss_fun,
                alpha=self.params.alpha,
                symmetric=self.params.symmetric,
                armijo=self.params.armijo,
                log=self.params.log,
                G0=g0,
                max_iter=self.params.max_iter,
                tol_rel=self.params.tol_rel,
                tol_abs=self.params.tol_abs,
            )
            score = float(out[0] if isinstance(out, tuple) else out)
            if score < best_score:
                best_score = score

        return best_score

    def score_with_fgw(self, sub_pair, film_pair):
        """
        Interface score = FGW(substrate_main interface vs substrate bulk)
        + FGW(film_main interface vs film bulk)
        """
        g_itf_sub_main, g_sub_bulk = sub_pair
        g_itf_film_main, g_film_bulk = film_pair
        sub_fgw = self.one_side_fgw(g_itf_sub_main, g_sub_bulk)
        film_fgw = self.one_side_fgw(g_itf_film_main, g_film_bulk)
        return float(sub_fgw + film_fgw)
