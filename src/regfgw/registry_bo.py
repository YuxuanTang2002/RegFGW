import copy
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pymatgen.core.interface import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from tqdm import tqdm
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.data import covalent_radii, vdw_radii
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from .structure_to_graph import GraphEncoder
from .fgw_metric import FGWScorer
from .interface_equivalence import InterfaceMatcher, InterfaceMatchParams

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
    score: registry FGW score (lower is more bulk-like)
    registry: shifted interface structure evaluated at this step
    """
    step: int
    score: float
    registry: Interface

@dataclass(frozen=True)
class BOParams:
    """
    Hyperparameters controlling Bayesian optimization (BO)

    Attributes
    ----------
    n_init: number of initial registry samples
    n_iter: number of BO refinement iterations
    acq_candidates: number of acquisition candidates per BO iteration
    seed: random seed for reproducibility
    xi: exploration parameter in expected improvement (EI)
    penalty: large penalty assigned to geometrically invalid registries
    """
    n_init: int = 8
    n_iter: int = 50
    acq_candidates: int = 4096
    seed: int = 0
    xi: float = 1e-4
    penalty: float = 1e6

    def __post_init__(self):
        if self.n_init < 1 or self.n_init & (self.n_init-1):
            raise ValueError("n_init must be a positive power of 2 for Sobol initialization.")

        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")

        if self.acq_candidates < 1:
            raise ValueError("acq_candidates must be at least 1.")

        if self.xi <= 0.0:
            raise ValueError("xi must be positive.")

        if self.penalty <= 0.0:
            raise ValueError("penalty must be positive.")

# -----------------------------------------------------------------------------
# Registry optimization engine
# -----------------------------------------------------------------------------

class RegistryPriorBO:
    """
    Bayesian optimization engine for interface registry search

    Notes
    -----
    1) The search space consists of fractional in-plane translations (shift_a, shift_b) applied to the film slab.
    2) The objective function is defined as:
    FGW(graph(interface_substrate_side), graph(substrate_bulk_reference)) + FGW(graph(interface_film_side), graph(film_bulk_reference))
    3) Physical continuity constraints are enforced prior to scoring to exclude unexpected atomic overlaps or broken interfacial bonding.
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
    def periodic_embedding(coord: np.ndarray):
        """
        Map fractional registry coordinates onto a periodic representation.

        Parameters
        ----------
        coord: (n_samples, 2), shift_a and shift_b in [0, 1)

        Returns
        -------
        (n_samples, 4), [cos(2πa), sin(2πa), cos(2πb), sin(2πb)]
        """
        coord = np.asarray(coord, dtype=float)

        if coord.ndim == 1:
            coord = coord.reshape(1, -1)

        if coord.shape[1] != 2:
            raise ValueError(f"Unexpected registry coordinates with shape {coord.shape}.")

        shift_a = coord[:, 0]
        shift_b = coord[:, 1]

        return np.column_stack([
            np.cos(2.0 * np.pi * shift_a),
            np.sin(2.0 * np.pi * shift_a),
            np.cos(2.0 * np.pi * shift_b),
            np.sin(2.0 * np.pi * shift_b),
        ])

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

    @staticmethod
    def count_interface_pairs(itf: Interface):
        """
        Count interfacial atom pairs within a finite interface window.

        Parameters
        ----------
        itf: Interface object

        Returns
        -------
        dict{
        "sub_indices": List[int]
        "film_indices": List[int]
        "sub_nums": np.ndarray
        "film_nums": np.ndarray
        "cov_sum": np.ndarray, shape = (n_sub, n_film)
        "vdw_sum": np.ndarray, shape = (n_sub, n_film)
        "d": np.ndarray, shape = (n_sub, n_film)
        }
        """
        c_all = np.array([s.coords[2] for s in itf.sites], dtype=float)
        sub_c_max = float(np.max(c_all[itf.substrate_indices]))
        film_c_min = float(np.min(c_all[itf.film_indices]))
        sub_indices = [i for i in itf.substrate_indices if (sub_c_max - c_all[i]) <= 3.0]
        film_indices = [j for j in itf.film_indices if (c_all[j] - film_c_min) <= 3.0]

        if not sub_indices or not film_indices:
            raise ValueError("No atoms were found in interface window.")

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

        return {
            "sub_indices": sub_indices,
            "film_indices": film_indices,
            "sub_nums": sub_nums,
            "film_nums": film_nums,
            "cov_sum": cov_sum,
            "vdw_sum": vdw_sum,
            "d": d,
        }

    # -----------------------------------------------------------------------------
    # Physical continuity constraints
    # -----------------------------------------------------------------------------

    def check_registry_continuity(self, itf: Interface):
        """
        Enforce physical continuity constraints at the interface

        Check:
        1) No interatomic distance smaller than covalent radius sum (avoid atomic overlap)
        2) Sufficient interfacial contact within van der Waals range

        Returns
        -------
        is_valid: bool, whether the registry passes the continuity check.
        reason: str, "valid", "too_close" or "too_far"
        """
        pairs = self.count_interface_pairs(itf)
        sub_indices = pairs["sub_indices"]
        film_indices = pairs["film_indices"]
        cov_sum = pairs["cov_sum"]
        vdw_sum = pairs["vdw_sum"]
        d = pairs["d"]

        # Overlap check (too close)
        overlap = (d < (cov_sum - 0.2))
        n_sub_overlap = np.count_nonzero(np.any(overlap, axis=1))
        n_film_overlap = np.count_nonzero(np.any(overlap, axis=0))
        max_overlap_atoms = max(1, int(0.05 * min(len(sub_indices), len(film_indices))))

        if max(n_sub_overlap, n_film_overlap) > max_overlap_atoms:
            return False, "too_close"

        # Contact check (too far)
        contact = (d <= (vdw_sum + 0.2))
        n_sub_contact = np.count_nonzero(np.any(contact, axis=1))
        n_film_contact = np.count_nonzero(np.any(contact, axis=0))
        min_contact_atoms = max(3, int(0.05 * min(len(sub_indices), len(film_indices))))

        if min(n_sub_contact, n_film_contact) < min_contact_atoms:
            return False, "too_far"

        return True, "valid"

    # -----------------------------------------------------------------------------
    # Normal registry feasibility scanning
    # -----------------------------------------------------------------------------

    def suggest_shift_c(self, interface: Dict[str, Any]):
        """
        Suggest a uniform near-contact normal shift for the reference interface.

        Strategy
        --------
        1) Use a small set of representative reference registries.
        2) For each registry, scan shift_c values along the surface normal and return the first shift
        at which the reference interface enters a physically meaningful near-contact regime.
        3) Take the median shift_c over the reference registries.
        """
        base_itf = interface["interface"]
        scan_grid = [-i * 0.05 for i in range(100)]
        grid = [0.0, 0.25, 0.5, 0.75]
        ref_registries = [(a, b) for a in grid for b in grid]
        ref_shift_cs: List[float] = []
        total_steps = len(ref_registries) * len(scan_grid)

        with tqdm(total=total_steps, desc="Near-contact scanning") as pbar:
            for shift_a, shift_b in ref_registries:
                ref_itf = self.shift_film(base_itf, shift_a=float(shift_a), shift_b=float(shift_b))
                found_shift_c = None
                for shift_c in scan_grid:
                    itf_try = self.shift_film(ref_itf, shift_c=float(shift_c))
                    pairs = self.count_interface_pairs(itf_try)
                    sub_indices = pairs["sub_indices"]
                    film_indices = pairs["film_indices"]
                    cov_sum = pairs["cov_sum"]
                    d = pairs["d"]
                    # Near-contact shell based on covalent radii + small buffer
                    contact = (d <= (cov_sum + 0.25))
                    n_sub_contact = np.count_nonzero(np.any(contact, axis=1))
                    n_film_contact = np.count_nonzero(np.any(contact, axis=0))
                    min_contact_atoms = max(3, int(0.05 * min(len(sub_indices), len(film_indices))))
                    if min(n_sub_contact, n_film_contact) >= min_contact_atoms:
                        found_shift_c = float(shift_c)
                        pbar.update(1)
                        break
                    pbar.update(1)
                if found_shift_c is not None:
                    remaining = len(scan_grid) - (scan_grid.index(found_shift_c) + 1)
                    if remaining > 0:
                        pbar.update(remaining)
                else:
                    raise RuntimeError(
                        f"No near-contact shift_c found for reference registry (shift_a={shift_a}, shift_b={shift_b}). "
                        "Check the interface geometry and gap settings."
                    )
                ref_shift_cs.append(found_shift_c)

        suggested_shift_c = float(np.median(ref_shift_cs))
        tqdm.write(f"Suggested shift_c: {suggested_shift_c:.6f} Å")

        return suggested_shift_c

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
            continuity_check: bool = True,
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
        score: float, returns inf if physical constraints fail
        shifted_itf: Interface, shifted interface structure
        continuity_status: str, "valid", "too_close" or "too_far"
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
        continuity_status = "not_checked"

        if continuity_check:
            is_valid, continuity_status = self.check_registry_continuity(shifted_itf)
            if not is_valid:
                return float("inf"), shifted_itf, continuity_status

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

        return float(score), shifted_itf, continuity_status

    # -----------------------------------------------------------------------------
    # Main optimizer
    # -----------------------------------------------------------------------------

    def bayes_optimize_registry(
            self,
            interface: Dict[str, Any],
            budget: int = 3,
            unique: bool = False,
            out_traj: bool = False,
            dft_gap_offset: float = 0.0,
            shift_c: Optional[float] = None,
            continuity_check: bool = True,
    ):
        """
        Perform Bayesian optimization over in-plane registry space.

        Search space
        ------------
        shift_a: [0, 1]
        shift_b: [0, 1]

        Algorithm
        ---------
        1) Scrambled Sobol initialization (n_init samples)
        2) Fit Gaussian Process surrogate model.
        3) Maximum EI acquisition.
        4) Evaluate selected registry.
        5) Repeat for n_iter iterations.

        Returns
        -------
        records: List[BORecord]
        """
        if budget < 1:
            raise ValueError("budget must be at least 1.")
        
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

        if shift_c is None:
            shift_c = self.suggest_shift_c(interface)

        if self.structure_check:
            self.g_sub_bulk = None
            self.g_film_bulk = None
            self.score_registry(interface, shift_c=shift_c, structure_check=True, continuity_check=continuity_check)

        total_steps = self.params.n_init + self.params.n_iter
        m = int(np.log2(self.params.n_init))
        sobol = qmc.Sobol(
            d=2,
            scramble=True,
            rng=rng,
        )
        initial_points = sobol.random_base2(m)
        initial_failure_counts = {"too_close": 0, "too_far": 0}

        with tqdm(total=total_steps, desc="BO initialization", leave=False) as pbar:
            # Sobol initialization
            for shift_a, shift_b in initial_points:
                shift_a = float(shift_a)
                shift_b = float(shift_b)
                score, reg, continuity_status = self.score_registry(
                    interface, 
                    shift_a=shift_a, 
                    shift_b=shift_b, 
                    shift_c=shift_c,
                    continuity_check=continuity_check,
                )
                if continuity_status in initial_failure_counts:
                    initial_failure_counts[continuity_status] += 1
                step = len(records)
                record = BORecord(step=step, score=score, registry=reg)
                records.append(record)
                x.append([shift_a, shift_b])
                y.append(score if np.isfinite(score) else self.params.penalty)
                if np.isfinite(score) and (best_record is None or score < best_record.score):
                    best_record = record
                if best_record is not None:
                    pbar.set_postfix(best=f"{best_record.score:.6g}")
                pbar.update(1)
            if best_record is None:
                raise RuntimeError(
                    "All Sobol initialization samples failed the interface continuity check.\n"
                    f"too_close={initial_failure_counts['too_close']}, "
                    f"too_far={initial_failure_counts['too_far']}, "
                    f"shift_c={shift_c:.6f} Å.\n"
                    "Bayesian optimization cannot proceed. "
                    "Provide a manual shift_c value or inspect the interface structure. "
                )
            pbar.set_description("BO refinement")
            # Gaussian Process surrogate
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(4),
                length_scale_bounds=(1e-2, 10.0),
                nu=1.5,
            )
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.params.seed, alpha=1e-6)
            # BO refinement loop
            refine_finite = False
            refinement_failure_counts = {"too_close": 0, "too_far": 0}

            for _ in range(self.params.n_iter):
                x_array = np.asarray(x, dtype=float)
                y_array = np.asarray(y, dtype=float)
                mask = np.isfinite(y_array) & (y_array < self.params.penalty * 0.5)
                n_valid = int(np.count_nonzero(mask))
                if n_valid < 3:
                    raise RuntimeError(
                        "Too few valid registries to fit GP model.\n"
                        f"valid={n_valid}, "
                        f"too_close={initial_failure_counts['too_close']}, "
                        f"too_far={initial_failure_counts['too_far']}, "
                        f"shift_c={shift_c:.6f} Å.\n"
                        "Registry space likely dominated by invalid configurations. "
                        "Provide a manual shift_c value or inspect the interface structure. "
                    )
                else:
                    x_train = self.periodic_embedding(x_array[mask])
                    gp.fit(x_train, y_array[mask])
                cand_a = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
                cand_b = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
                x_cand = np.stack([cand_a, cand_b], axis=1)
                x_cand_train = self.periodic_embedding(x_cand)
                y_mu, y_sigma = gp.predict(x_cand_train, return_std=True)
                y_best = float(np.min(y_array[mask]))
                ei = evaluate_expected_improvement(y_mu, y_sigma, y_best, float(self.params.xi))
                best_idx = int(np.argmax(ei))
                shift_a = float(x_cand[best_idx, 0])
                shift_b = float(x_cand[best_idx, 1])
                score, reg, continuity_status = self.score_registry(
                    interface, 
                    shift_a=shift_a, 
                    shift_b=shift_b, 
                    shift_c=shift_c,
                    continuity_check=continuity_check,
                )
                if continuity_status in refinement_failure_counts:
                    refinement_failure_counts[continuity_status] += 1
                step = len(records)
                record = BORecord(step=step, score=score, registry=reg)
                records.append(record)
                x.append([shift_a, shift_b])
                y.append(score if np.isfinite(score) else self.params.penalty)
                if np.isfinite(score):
                    refine_finite = True
                    if best_record is None or score < best_record.score:
                        best_record = record
                if best_record is not None:
                    pbar.set_postfix(best=f"{best_record.score:.6g}")
                pbar.update(1)
            if not refine_finite:
                raise RuntimeError(
                    "All refinement samples failed the interface continuity check.\n"
                    f"too_close={refinement_failure_counts['too_close']}, "
                    f"too_far={refinement_failure_counts['too_far']}, "
                    f"shift_c={shift_c:.6f} Å.\n"
                    "Bayesian optimization cannot proceed. "
                    "Provide a manual shift_c value or inspect the interface structure. "
                )

        out_records = [record for record in records if np.isfinite(record.score)]

        if unique:
            matcher = InterfaceMatcher(
                InterfaceMatchParams(
                    ltol=1e-5,
                    stol=1e-3,
                    angle_tol=1e-3,
                )
            )
            groups = matcher.group_interfaces([record.registry for record in out_records])
            out_records = [out_records[group.rep_index] for group in groups]
            print(f"[Unique] Reduced {len(records)} registries to {len(out_records)} unique registries.")

        out_records.sort(key=lambda record: record.score)

        if len(out_records) < budget:
            print(
                f"[Warn] Requested {budget} structures, but only "
                f"{len(out_records)} valid"
                f"{' unique' if unique else ''} registries are available."
            )

        out_records = out_records[:budget]
        base_path = self.enc.build_base_path(interface)

        for out_idx, record in enumerate(out_records):
            registry = self.shift_film(record.registry, shift_c=dft_gap_offset)
            atoms: Atoms = AseAtomsAdaptor.get_atoms(registry)
            cif_path = f"{base_path}_initial_{out_idx}.cif"
            write(cif_path, atoms)

        if out_traj:
            traj_path = f"{base_path}.traj"
            traj = Trajectory(traj_path, mode="w")
            for record in records:
                atoms: Atoms = AseAtomsAdaptor.get_atoms(record.registry)
                atoms.info["fgw_score"] = float(record.score)
                traj.write(atoms)
            traj.close()

        return best_record, out_records
