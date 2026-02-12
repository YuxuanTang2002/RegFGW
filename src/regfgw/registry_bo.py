import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pymatgen.core.interface import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.data import covalent_radii, vdw_radii
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from .structure_to_graph import GraphEncoder
from .fgw_metric import FGWScorer

@dataclass(frozen=True)
class BOParams:
    n_init: int = 12
    n_iter: int = 60
    acq_candidates: int = 4000
    seed: int = 0
    xi: float = 0.01
    penalty: float = 1e6

@dataclass
class BORecord:
    step: int
    score: float
    registry: Interface

class RegistryPriorBO:
    def __init__(
            self,
            encoder: GraphEncoder,
            scorer: FGWScorer,
            bo_params: BOParams,
            sub_period_layers: int = 4,
            film_period_layers: int = 4,
            structure_check=False,
            ):
        self.g_sub_bulk = None
        self.g_film_bulk = None
        self.enc = encoder
        self.sub_period_layers = sub_period_layers
        self.film_period_layers = film_period_layers
        self.structure_check = structure_check
        self.scorer = scorer
        self.params = bo_params

    @staticmethod
    def shift_film(interface: Interface, shift_a: float = 0.0, shift_b: float = 0.0, shift_c: float = 0.0):
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
    def check_registry_continuity(itf: Interface):
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

    def score_registry(
            self,
            interface: Dict[str, Any],
            shift_a: float = 0.0,
            shift_b: float = 0.0,
            shift_c: float = 0.0,
            structure_check: bool = False,
    ):
        if self.g_sub_bulk is None or self.g_film_bulk is None:
            self.g_sub_bulk, self.g_film_bulk = self.enc.prepare_bulk_cache(
                interface,
                sub_period_layers=self.sub_period_layers,
                film_period_layers=self.film_period_layers,
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
            sub_period_layers=self.sub_period_layers,
            film_period_layers=self.film_period_layers,
            structure_check=structure_check,
        )
        score = self.scorer.score_with_fgw(
            (g_itf_sub, self.g_sub_bulk),
            (g_itf_film, self.g_film_bulk),
        )

        return float(score), shifted_itf

    def suggest_shift_c_interval(self, interface: Dict[str, Any]):
        base_itf = interface["interface"]
        c_low = 0.0

        while True:
            c_try = float(c_low - 0.1)
            itf_try = self.shift_film(base_itf, shift_c=c_try)
            if not self.check_registry_continuity(itf_try):
                break
            c_low = c_try

        c_high = 0.0

        while True:
            c_try = float(c_high + 0.1)
            itf_try = self.shift_film(base_itf, shift_c=c_try)
            if not self.check_registry_continuity(itf_try):
                break
            c_high = c_try

        if c_low == 0.0 and c_high == 0.0 and not self.check_registry_continuity(base_itf):
            raise RuntimeError(
                "No feasible shift_c found" 
                "Interface geometrically incompatible with current physical constraints"
            )

        c_range = np.arange(c_low, c_high+0.05, 0.1, dtype=float)
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

        print(f"[c-scan] feasible shift_c range (Å): [{c_low:.2f}, {c_high:.2f}]")
        print(f"[c-scan] suggested bulk-continuous interval (Å): [{sug_c_min:.2f}, {sug_c_max:.2f}]")

    def bayes_optimize_registry(self, interface: Dict[str, Any], out_structure: bool = False):
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

        _, _ = self.score_registry(interface, structure_check=self.structure_check)
        self.suggest_shift_c_interval(interface)

        for i in range(self.params.n_init):
            shift_a = float(rng.uniform(0.0, 1.0))
            shift_b = float(rng.uniform(0.0, 1.0))
            score, reg = self.score_registry(interface, shift_a=shift_a, shift_b=shift_b)
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

        kernel = ConstantKernel(1.0, (1e-3,1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.params.seed)

        for i in range(self.params.n_iter):
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            gp.fit(x_array, y_array)
            cand_a = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
            cand_b = rng.uniform(0.0, 1.0, size=self.params.acq_candidates)
            x_cand = np.stack([cand_a,cand_b], axis=1)
            y_mu, y_sigma = gp.predict(x_cand, return_std=True)
            y_best = float(np.min(y_array))
            ei = evaluate_expected_improvement(y_mu, y_sigma, y_best, float(self.params.xi))
            best_idx = int(np.argmax(ei))
            shift_a = float(x_cand[best_idx, 0])
            shift_b = float(x_cand[best_idx, 1])
            score, reg = self.score_registry(interface, shift_a=shift_a, shift_b=shift_b)
            step = len(records)
            record = BORecord(step=step, score=score, registry=reg)
            records.append(record)
            x.append([shift_a, shift_b])
            y.append(score if np.isfinite(score) else self.params.penalty)
            if np.isfinite(score) and (best_record is None or score < best_record.score):
                best_record = record

        if out_structure:
            base_path = self.enc.build_base_path(interface)
            cif_path = f"{base_path}_initial.cif"
            best_reg = AseAtomsAdaptor.get_atoms(best_record.registry)
            write(cif_path, best_reg)

        return best_record, records

























