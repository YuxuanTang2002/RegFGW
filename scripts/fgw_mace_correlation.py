import argparse
import json
import csv
import numpy as np
from pathlib import Path
from numpy.linalg import LinAlgError
from tqdm import tqdm
from dataclasses import dataclass
from monty.json import MontyDecoder
from ase import Atoms
from ase.optimize import FIRE, BFGS
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import BOParams, RegistryPriorBO
from regfgw.structure_to_graph import GraphEncoder
from mace.calculators import mace_mp

@dataclass(frozen=True)
class GridSampleConfig:
    grid_a: int = 19
    grid_b: int = 19

def set_atomic_constraints(
    atoms: Atoms,
    substrate_indices,
    film_indices,
    free_sub_top_frac: float | None = None,
    free_film_bottom_frac: float | None = None,
):
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    sub_idx = list(substrate_indices)
    film_idx = list(film_indices)
    fixed_indices = []

    if free_sub_top_frac is not None:
        if 0.0 <= free_sub_top_frac <= 1.0:
            sub_z = z[sub_idx]
            sub_z_min, sub_z_max = float(sub_z.min()), float(sub_z.max())
            sub_free_threshold = sub_z_min + (sub_z_max - sub_z_min) * (1 - free_sub_top_frac)
            for i in sub_idx:
                if z[i] < sub_free_threshold:
                    fixed_indices.append(i)
        else:
            raise ValueError("free_sub_top_frac must be between 0 and 1.")

    if free_film_bottom_frac is not None:
        if 0.0 <= free_film_bottom_frac <= 1.0:
            film_z = z[film_idx]
            film_z_min, film_z_max = float(film_z.min()), float(film_z.max())
            film_free_threshold = film_z_min + (film_z_max - film_z_min) * free_film_bottom_frac
            for i in film_idx:
                if z[i] > film_free_threshold:
                    fixed_indices.append(i)
        else:
            raise ValueError("free_film_bottom_frac must be between 0 and 1.")

    atoms.set_constraint(FixAtoms(indices=sorted(set(fixed_indices))))

def main():
    p = argparse.ArgumentParser(description="Sample uniform registry grid to check the correlation of fgw distance and mace relaxed energy.")
    p.add_argument("--record-json", required=True, help="Interface record JSON")
    p.add_argument("--embedding", required=True, help="Element embedding JSON/CSV")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--mace-mode", choices=["opt", "scf"], default="opt", help="opt mode for relaxation, scf mode for single-point energy calculation.")
    p.add_argument("--mace-model", type=str, default="medium")
    p.add_argument("--mace-device", type=str, default="cuda")
    p.add_argument("--mace-dtype", type=str, default="float32")
    p.add_argument("--mace-gap-offset", type=float, default=0.0, help="Additional gap offset applied for MACE relaxation.")
    p.add_argument("--optimizer", default="fire", choices=["fire", "bfgs"])
    p.add_argument("--fmax", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--free-sub-top-frac", type=float, default=1.0, help="Top fraction of substrate slab allowed to relax")
    p.add_argument("--free-film-bottom-frac", type=float, default=1.0, help="Bottom fraction of film slab allowed to relax")
    p.add_argument("--output", choices=["fgw", "energy"], default="energy", help="Quantity to output")
    args = p.parse_args()

    if args.mace_gap_offset != 0.0 and (args.output != "energy" or args.mace_mode != "opt"):
        p.error("--mace-gap-offset requires MACE relaxation mode.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load interface record
    with open(args.record_json, "r", encoding="utf-8") as f:
        record = json.load(f, cls=MontyDecoder)

    # Build FGW scorer
    encoder = GraphEncoder(embedding_path=args.embedding)
    fgw_builder = FGWBuilder(FGWBuildParams(feature_metric="euclidean"))
    scorer = FGWScorer(
        builder=fgw_builder,
        score_params=FGWScoreParams(
            alpha=0.75,
            n_starts=80,
            init_seed=0,
        ),
    )
    bo = RegistryPriorBO(
        encoder=encoder,
        scorer=scorer,
        bo_params=BOParams(
            n_init=8,
            n_iter=50,
            acq_candidates=4096,
            seed=0,
            xi=1e-4,
            penalty=1e6,
        ),
        structure_check=False,
    )
    # Suggest near-contact gap
    shift_c = float(bo.suggest_shift_c(record))
    # Build uniform registry grid
    grid_a = (np.arange(GridSampleConfig.grid_a, dtype=float) + 0.5) / float(GridSampleConfig.grid_a)
    grid_b = (np.arange(GridSampleConfig.grid_b, dtype=float) + 0.5) / float(GridSampleConfig.grid_b)
    # Build MACE calculator
    calc = None

    if args.output == "energy":
        if args.mace_mode == "opt" and args.mace_gap_offset != 0.0:
            print(f"[Info] Applied an additional gap offset {args.mace_gap_offset:.3f} Å for MACE relaxation.")
        calc = mace_mp(model=args.mace_model, device=args.mace_device, default_dtype=args.mace_dtype)
        if args.mace_mode == "opt":
            summary_path = out_dir / "mace_relaxed_energy.csv"
            energy_col = "relaxed_energy"
        else:
            summary_path = out_dir / "mace_single_point_energy.csv"
            energy_col = "single_point_energy"
        header = ["shift_a", "shift_b", "shift_c", energy_col]
    else:
        summary_path = out_dir / "fgw_distance.csv"
        header = ["shift_a", "shift_b", "shift_c", "fgw_distance"]

    total = len(grid_a) * len(grid_b)
    opt_class = FIRE if args.optimizer == "fire" else BFGS
    fgw_ds = []
    energies = []

    with summary_path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Loop over registry grid
        with tqdm(total=total, desc=f"Calculating {args.output} over a uniform grid {GridSampleConfig.grid_a}x{GridSampleConfig.grid_b}") as pbar:
            for a in grid_a:
                for b in grid_b:
                    if args.output == "fgw":
                        score, _, _ = bo.score_registry(
                            record,
                            shift_a=float(a),
                            shift_b=float(b),
                            shift_c=shift_c,
                        )
                        if np.isfinite(score):
                            fgw_ds.append(float(score))
                            score = f"{float(score):.12f}"
                        else:
                            score = ""
                        writer.writerow(
                            [
                                f"{float(a):.3f}",
                                f"{float(b):.3f}",
                                f"{float(shift_c):.3f}",
                                score,
                            ]
                        )
                    else:
                        reg_mace = RegistryPriorBO.shift_film(
                            record["interface"],
                            shift_a=float(a),
                            shift_b=float(b),
                            shift_c=shift_c+args.mace_gap_offset,
                        )
                        substrate_indices = reg_mace.substrate_indices
                        film_indices = reg_mace.film_indices
                        atoms: Atoms = AseAtomsAdaptor.get_atoms(reg_mace)
                        atoms.pbc = (True, True, False)
                        atoms.wrap()
                        atoms.calc = calc
                        if args.mace_mode == "opt":
                            set_atomic_constraints(
                                atoms=atoms,
                                substrate_indices=substrate_indices,
                                film_indices=film_indices,
                                free_sub_top_frac=args.free_sub_top_frac,
                                free_film_bottom_frac=args.free_film_bottom_frac,
                            )
                            optimizer = opt_class(atoms, logfile=None)
                            try:
                                with tqdm(total=int(args.max_steps), desc=f"Relax a={a:.3f}, b={b:.3f}",
                                          leave=False) as pbar_relax:
                                    optimizer.attach(lambda: pbar_relax.update(1), interval=1)
                                    converged = optimizer.run(fmax=float(args.fmax), steps=int(args.max_steps))
                                    energy = float(atoms.get_potential_energy()) if converged else None
                            except (RuntimeError, ValueError, LinAlgError):
                                energy = None
                        else:
                            try:
                                energy = float(atoms.get_potential_energy())
                            except (RuntimeError, ValueError, LinAlgError):
                                energy = None
                        if energy is not None:
                            energies.append(energy)
                            energy = f"{float(energy):.12f}"
                        else:
                            energy = ""
                        writer.writerow(
                            [
                                f"{float(a):.3f}",
                                f"{float(b):.3f}",
                                f"{float(shift_c + args.mace_gap_offset):.3f}",
                                energy,
                            ]
                        )
                    pbar.update(1)

    if args.output == "fgw":
        print(
            f"[Done] Write FGW distances in {out_dir}. "
            f"Valid FGW points: {len(fgw_ds)}/{total}."
        )
    elif args.mace_mode == "opt":
        print(
            f"[Done] Write MACE relaxed energies in {out_dir}. " 
            f"Valid energy points: {len(energies)}/{total}."
        )
    else:
        print(
            f"[Done] Write MACE single-point energies in {out_dir}. " 
            f"Valid energy points: {len(energies)}/{total}."
        )

if __name__ == "__main__":
    main()
