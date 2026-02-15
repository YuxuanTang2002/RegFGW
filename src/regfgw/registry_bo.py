import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pymatgen.core.interface import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from ase import  Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.data import covalent_radii, vdw_radii
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from .structure_to_graph import GraphEncoder
from .fgw_metric import FGWScorer

# -----------------------------------------------------------------------------
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass
class BORecord:
    """
    Record for a single BO evaluation step

    Attributes
    ----------
    step: optimization step
    score: registry fgw score (lower is more bulk-like)
    registry: shifted interface structure evaluated at this step
    """
    step: int
    score: float
    registry: Interface

@dataclass(frozen=True)
class BOParams:
    """
    Hyperparameters controlling Bayesian optimization(BO)

    Attributes
    ----------
    n_init: number of initial random registry samples
    n_iter: number of BO refinement iterations
    acq_candidates: number of acquisition candidates per BO iteration
    seed: random seed for reproducibility
    xi: exploration parameter in expected improvement(ei)
    penalty: Large penalty assigned to geometrically invalid registries
    """
    n_init: int = 25
    n_iter: int = 75
    acq_candidates: int = 4000
    seed: int = 0
    xi: float = 0.01
    penalty: float = 1e6

# -----------------------------------------------------------------------------
# Registry optimization engine
# -----------------------------------------------------------------------------

class RegistryPriorBO:
    """
    Bayesian optimization engine for interface registry search

    Notes
    -----
    1) The search space consists of fractional in-plane transitions (shift_a, shift_b) applied to the film slab.
    2) The objective function is defined as:
    FGW(graph(interface_substrate_side), graph(substrate_bulk_reference)) + FGW(graph(interface_film_side), graph(film_bulk_reference))
    3) physical continuity constraints are enforced prior to scoring to exclude unexpected atomic overlaps or broken interfacial bonding.
    """
    def __init__(
            self,
            encoder: GraphEncoder,
            scorer: FGWScorer,
            bo_params: BOParams,
            structure_check=False,
            ):
        self.g_sub_bulk = None
        self.g_film_bulk = None
        self.enc = encoder
        self.structure_check = structure_check
        self.scorer = scorer
        self.params = bo_params

    # -----------------------------------------------------------------------------
    # Registry manipulation utilities
    # -----------------------------------------------------------------------------

    @staticmethod
    def shift_film(interface: Interface, shift_a: float = 0.0, shift_b: float = 0.0, shift_c: float = 0.0):
        """
        Apply registry shift to the film slab.

        Parameters
        ----------
        interface: coherent interface object
        shift_a, shift_b: fractional translations along in-plane lattice vectors
        shift_c: cartesian translation along surface normal (Å)
        """
        itf = copy.deepcopy(interface)
        film_idx = list(itf.film_indices)
        itf.translate_sites(
            film_idx,
            [float(shift_a), float(shift_b), 0.0],
            frac_coords=True,
            to_unit_cell=True,
        )

        if abs(float(shift_c)) > 0.0:
            itf.translate_sites(
                film_idx,
                [0.0, 0.0, float(shift_c)],
                frac_coords=False,
                to_unit_cell=False,
            )

        return itf

    # -----------------------------------------------------------------------------
    # Physical continuity constraints
    # -----------------------------------------------------------------------------

    @staticmethod
    def check_registry_continuity(itf: Interface):
        """
        Enforce physical continuity constraints at the interface

        check:
        1) No interatomic distance smaller than covalent radius sum (avoid atomic overlap)
        2) At least one interfacial atom pair within van der Waals range

        Returns
        -------
        True if registry satisfies physical continuity constraints.
        """
        c_all = np.array([s.coords[2] for s in itf.sites], dtype=float)
        sub_c_max = float(np.max(c_all[itf.substrate_indices]))
        film_c_min = float(np.min(c_all[itf.film_indices]))
        sub_indices = [i for i in itf.substrate_indices if (sub_c_max - c_all[i]) <= 6.0]
        film_indices = [j for j in itf.film_indices if (c_all[j] - film_c_min) <= 6.0]

        if not sub_indices or not film_indices:
            raise ValueError("No atoms in interface window")

        atom_nums = np.array([s.specie.Z for s in itf.sites], dtype=int)
        sub_nums = atom_nums[sub_indices]
        film_nums = atom_nums[film_indices]
        itf_nums = np.concatenate([sub_nums, film_nums]).astype(int)
        missing_rc = sorted({n for n in itf_nums if covalent_radii[n] <= 0.0})
        missing_rv = sorted({n for n in itf_nums if vdw_radii[n] <= 0.0})

        if missing_rc or missing_rv:
            raise ValueError(
                "Radii table incomplete: " + "; ".join(
                    s for s in (
                        f"covalent_radii invalid for atom_nums={missing_rc}" if missing_rc else "",
                        f"vdw_radii invalid for atom_nums={missing_rv}" if missing_rv else "",
                    ) if s
                )
            )

        rc_sub = np.array([float(covalent_radii[n]) for n in sub_nums], dtype=float)
        rc_film = np.array([float(covalent_radii[n]) for n in film_nums], dtype=float)
        rv_sub = np.array([float(vdw_radii[n]) for n in sub_nums], dtype=float)
        rv_film = np.array([float(vdw_radii[n]) for n in film_nums], dtype=float)
        cov_sum = rc_sub[:, None] + rc_film[None, :]
        vdw_sum = rv_sub[:, None] + rv_film[None, :]
        d = np.empty((len(sub_indices), len(film_indices)), dtype=float)

        for a, i in enumerate(sub_indices):
            for b, j in enumerate(film_indices):
                d[a, b] = float(itf.get_distance(i, j))

        if np.any(d < cov_sum):
            return False

        if np.all(d > vdw_sum):
            return False

        return True

    # -----------------------------------------------------------------------------
    # Registry scoring
    # -----------------------------------------------------------------------------

    def score_registry(
            self,
            interface: Dict[str, Any],
            shift_a: float = 0.0,
            shift_b: float = 0.0,
            shift_c: float = 0.0,
            structure_check: bool = False,
    ):
        """
        Evaluate registry score after applying shift.

        Procedure
        ---------
        1) Apply registry shift.
        2) Check physical continuity constraints.
        3) Construct graph pairs (cache bulk graphs of the initial interface for all registries).
        4) Compute FGW-based bulk similarity score.

        Returns
        -------
        score: float, returns inf if physical constraints fails
        shifted_itf: Interface, shifted interface structure
        """
        if self.g_sub_bulk is None or self.g_film_bulk is None:
            self.g_sub_bulk, self.g_film_bulk = self.enc.prepare_bulk_cache(
                interface,
                sub_period_layers=interface["sub_period_layers"],
                film_period_layers=interface["film_period_layers"],
                structure_check=structure_check,
            )

        base_itf = interface["interface"]
        shifted_itf = self.shift_film(base_itf, shift_a, shift_b, shift_c)

        if not self.check_registry_continuity(shifted_itf):
            return float("inf"), shifted_itf

        new_interface = dict(interface)
        new_interface["interface"] = shifted_itf
        g_itf_sub, g_itf_film = self.enc.build_sided_interface_graphs(
            new_interface,
            sub_period_layers=interface["sub_period_layers"],
            film_period_layers=interface["film_period_layers"],
            structure_check=structure_check,
        )
        score = self.scorer.score_with_fgw(
            (g_itf_sub, self.g_sub_bulk),
            (g_itf_film, self.g_film_bulk),
        )

        return float(score), shifted_itf

    # -----------------------------------------------------------------------------
    # Normal registry feasibility scanning
    # -----------------------------------------------------------------------------

    def suggest_shift_c_interval(self, interface: Dict[str, Any]):
        """
        Suggest physically feasible normal shift interval.

        Notes
        -----
        1) Scan shift_c values to determine:
        * Lower bound: onset of atomic overlap
        * Upper bound: loss of interfacial interaction
        2) A three-point moving average is used to identify the most bulk-continuous normal window.
        3) This routine is diagnostic and does not modify the optimization space directly.
        """
        base_itf = interface["interface"]
        first_true = None
        last_true = None

        for i in range(51):
            shift_c = -i * 0.1
            itf_try = self.shift_film(base_itf, shift_c=float(shift_c))
            if self.check_registry_continuity(itf_try):
                if first_true is None:
                    first_true = float(shift_c)
                last_true = float(shift_c)
            else:
                if first_true is not None:
                    break

        if first_true is None:
            raise RuntimeError(
                "No feasible shift_c found" 
                "Interface geometrically incompatible with current physical constraints"
            )

        c_range = np.arange(last_true, first_true+1e-6, 0.1, dtype=float)
        scores = np.empty_like(c_range, dtype=float)

        for i, shift_c in enumerate(map(float, c_range)):
            score, _ = self.score_registry(interface, shift_c=shift_c)
            scores[i] = score

        if len(c_range) < 3:
            best_idx = int(np.nanargmin(scores))
            sug_c_min = sug_c_max = c_range[best_idx]
        else:
            avg_scores = np.array([np.mean(scores[i:i+3]) for i in range(len(scores)-2)], dtype=float)
            best_idx = int(np.nanargmin(avg_scores))
            sug_c_min, sug_c_max = c_range[best_idx], c_range[best_idx+2]

        shift_c = 0.5 * (sug_c_min + sug_c_max)

        return float(shift_c)

    # -----------------------------------------------------------------------------
    # Main optimizer
    # -----------------------------------------------------------------------------

    def bayes_optimize_registry(self, interface: Dict[str, Any], out_best: bool = False, out_traj: bool = False):
        """
        Perform Bayesian optimization over in-plane registry space.

        Search space
        ------------
        shift_a: [0, 1]
        shift_b: [0, 1]

        Algorithm
        ---------
        1) Random initialization (n_init samples)
        2) Fit Gaussian Process surrogate model.
        3) Maximum ei acquisition.
        4) Evaluate selected registry.
        5) Repeat for n_iter iterations.

        Returns
        -------
        best_record: BORecord
        records: List[BORecord], full optimization trajectory
        """
        self.g_sub_bulk = None
        self.g_film_bulk = None
        rng = np.random.default_rng(self.params.seed)
        records: List[BORecord] = []
        best_record: Optional[BORecord] = None
        x: List[List[float]] = []
        y: List[float] = []

        def evaluate_expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float):
            sigma = np.maximum(sigma, 1e-12)
            z = (best-mu-xi) / sigma
            cdf = norm.cdf(z)
            pdf = norm.pdf(z)
            return (best-mu-xi) * cdf + sigma * pdf

        shift_c = self.suggest_shift_c_interval(interface)
        _, _ = self.score_registry(interface, shift_c=shift_c, structure_check=self.structure_check)

        # Random initialization
        for i in range(self.params.n_init):
            shift_a = float(rng.uniform(0.0, 1.0))
            shift_b = float(rng.uniform(0.0, 1.0))
            score, reg = self.score_registry(interface, shift_a=shift_a, shift_b=shift_b, shift_c=shift_c)
            step = len(records)
            record = BORecord(step=step, score=score, registry=reg)
            records.append(record)
            x.append([shift_a, shift_b])
            y.append(score if np.isfinite(score) else self.params.penalty)
            if np.isfinite(score) and (best_record is None or score < best_record.score):
                best_record = record

        if best_record is None:
            raise RuntimeError(
                "All initial samples failed (score=inf)." 
                "No registry passed check_registry_continuity, BO cannot proceed."
            )

        # Gaussian Process surrogate
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.2, length_scale_bounds=(1e-3, 1e2), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.params.seed, alpha=1e-6)

        # BO refinement loop
        for i in range(self.params.n_iter):
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            mask = np.isfinite(y_array) & (y_array < self.params.penalty * 0.5)
            if np.count_nonzero(mask) < 3:
                gp.fit(x_array, y_array)
            else:
                gp.fit(x_array[mask], y_array[mask])
            cand_a = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
            cand_b = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
            x_cand = np.stack([cand_a,cand_b], axis=1)
            y_mu, y_sigma = gp.predict(x_cand, return_std=True)
            y_best = float(np.min(y_array))
            ei = evaluate_expected_improvement(y_mu, y_sigma, y_best, float(self.params.xi))
            best_idx = int(np.argmax(ei))
            shift_a = float(x_cand[best_idx, 0])
            shift_b = float(x_cand[best_idx, 1])
            score, reg = self.score_registry(interface, shift_a=shift_a, shift_b=shift_b, shift_c=shift_c)
            step = len(records)
            record = BORecord(step=step, score=score, registry=reg)
            records.append(record)
            x.append([shift_a, shift_b])
            y.append(score if np.isfinite(score) else self.params.penalty)
            if np.isfinite(score) and (best_record is None or score < best_record.score):
                best_record = record

        # Optional structure output
        if out_best:
            base_path = self.enc.build_base_path(interface)
            cif_path = f"{base_path}_initial.cif"
            best_reg = AseAtomsAdaptor.get_atoms(best_record.registry)
            write(cif_path, best_reg)

        if out_traj:
            base_path = self.enc.build_base_path(interface)
            traj_path = f"{base_path}.traj"
            traj = Trajectory(traj_path, mode="w")
            for record in records:
                atoms: Atoms = AseAtomsAdaptor.get_atoms(record.registry)
                atoms.info["fgw_score"] = float(record.score)
                traj.write(atoms)
            traj.close()

        return best_record, records

























