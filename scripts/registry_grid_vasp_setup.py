import argparse
import numpy as np
import math
import json
import csv
from tqdm import tqdm
from ase import Atoms
from monty.json import MontyDecoder
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from pymatgen.core.interface import Interface
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar, Incar, Kpoints
from regfgw.structure_to_graph import GraphEncoder
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import RegistryPriorBO, BOParams

@dataclass(frozen=True)
class GridPoint:
    shift_a: float
    shift_b: float
    shift_c: float
    fgw_score: float
    registry: Interface

@dataclass(frozen=True)
class GridSampleConfig:
    grid_a: int = 8
    grid_b: int = 8
    n_low: int = 8
    n_mid: int = 8
    n_high: int = 8

    @property
    def n_cases(self):
        return self.n_low + self.n_mid + self.n_high

@dataclass(frozen=True)
class RelaxConfig:
    encut: float = 520.0
    prec: str = "Accurate"
    ediff: float = 1e-5
    ediffg: float = -0.03
    nelm: int = 120
    algo: str = "Normal"
    addgrid: bool = True
    amix: float = 0.15
    bmix: float = 0.001
    lreal: Union[str, bool] = "Auto"
    lasph: bool = True
    ibrion: int = 2
    nsw: int = 100
    potim: float = 0.05
    isif: int = 2
    ismear: int = 0
    sigma: float = 0.1
    isym: int = 0
    # Formal relaxation setting after pre-relaxation without dipole correction
    istart: int = 1
    icharg: int = 0
    ldipol: bool = True
    idipol: int = 3
    dipol: Optional[List[float]] = None
    # Output control
    lwave: bool = True
    lcharg: bool = True

@dataclass(frozen=True)
class StaticConfig:
    encut: float = 520.0
    prec: str = "Accurate"
    ediff: float = 1e-6
    nelm: int = 120
    algo: str = "Normal"
    addgrid: bool = True
    amix: float = 0.15
    bmix: float = 0.001
    lreal: Union[str, bool] = "Auto"
    lasph: bool = True
    ibrion: int = -1
    nsw: int = 0
    ismear: int = 0
    sigma: float = 0.05
    isym: int = 0
    ldipol: bool = True
    idipol: int = 3
    dipol: Optional[List[float]] = None
    lwave: bool = False
    lcharg: bool = False

@dataclass(frozen=True)
class SgeJobConfig:
    job_name: str = "OPT"
    project: str = "UCL_chemM_Butler"
    queue_type: str = "Gold"
    cores: int = 32
    walltime: str = "48:00:00"
    mem_per_core: str = "4G"

@dataclass(frozen=True)
class SlurmJobConfig:
    job_name: str = "OPT"
    cores: int = 32
    walltime: str = "48:00:00"
    mem_per_core: str = "4G"

def scan_registry_grid(base_itf: Dict[str, Any], bo: RegistryPriorBO, cfg: GridSampleConfig):
    shift_c = float(bo.suggest_shift_c(base_itf))
    grid_a = np.arange(cfg.grid_a, dtype=float) / cfg.grid_a
    grid_b = np.arange(cfg.grid_b, dtype=float) / cfg.grid_b
    points: List[GridPoint] = []
    total = len(grid_a) * len(grid_b)

    with tqdm(total=total, desc="Scanning registry grid") as pbar:
        for a in grid_a:
            for b in grid_b:
                score, reg = bo.score_registry(
                    base_itf,
                    shift_a=float(a),
                    shift_b=float(b),
                    shift_c=shift_c,
                )
                if np.isfinite(score):
                    points.append(
                        GridPoint(
                            shift_a=float(a),
                            shift_b=float(b),
                            shift_c=shift_c,
                            fgw_score=float(score),
                            registry=reg,
                        )
                    )
                pbar.update(1)

    if len(points) < cfg.n_cases:
        raise RuntimeError(f"Too few finite grid points to build a pool: {len(points)} < 24.")

    points = sorted(points, key=lambda p: p.fgw_score)

    return points

def select_cases(points: List[GridPoint], cfg: GridSampleConfig):
    n = len(points)
    low = list(points[:cfg.n_low])
    high = list(points[-cfg.n_high:])
    mid_region = list(points[cfg.n_low:n-cfg.n_high])
    mid_indices = np.linspace(0, len(mid_region)-1, cfg.n_mid, dtype=int)
    mid = [mid_region[i] for i in mid_indices]
    pool = low + mid + high
    pool = sorted(pool, key=lambda p: p.fgw_score)
    return pool

def build_dipol(itf: Interface, idipol: int):
    if idipol not in (1, 2, 3):
        raise ValueError(f"Invalid IDIPOL: {idipol}. Must be 1, 2, or 3.")

    atoms: Atoms = AseAtomsAdaptor.get_atoms(itf)
    mass_center_frac = atoms.get_center_of_mass(scaled=True)
    dipol = [0.5, 0.5, 0.5]
    dipol[idipol - 1] = float(mass_center_frac[idipol - 1])

    return dipol

def prepare_incar(case_dir: Path, cfg: RelaxConfig | StaticConfig):
    d = asdict(cfg)
    incar_dict = {k.upper(): v for k, v in d.items() if v is not None}
    incar = Incar(incar_dict)
    incar.write_file(case_dir / "INCAR")

def set_selective_dynamics(
        itf: Interface,
        free_sub_top_frac: Optional[float] = None,
        free_film_bottom_frac: Optional[float] = None,
):
    z = np.array([site.coords[2] for site in itf.sites], dtype=float)
    dynamics: List[List[bool]] = [[True,True,True] for _ in itf.sites]
    sub_idx = list(itf.substrate_indices)
    film_idx = list(itf.film_indices)

    if free_sub_top_frac is not None:
        if 0.0 <= free_sub_top_frac <= 1.0:
            sub_z = z[sub_idx]
            sub_z_min, sub_z_max = float(sub_z.min()), float(sub_z.max())
            sub_free_threshold = sub_z_min + (sub_z_max - sub_z_min) * (1.0 - free_sub_top_frac)
            for i in sub_idx:
                if z[i] < sub_free_threshold:
                    dynamics[i] = [False, False, False]
        else:
            raise ValueError("free_sub_top_frac must be between 0 and 1.")

    if free_film_bottom_frac is not None:
        if 0.0 <= free_film_bottom_frac <= 1.0:
            film_z = z[film_idx]
            film_z_min, film_z_max = float(film_z.min()), float(film_z.max())
            film_free_threshold = film_z_min + (film_z_max - film_z_min) * free_film_bottom_frac
            for i in film_idx:
                if z[i] > film_free_threshold:
                    dynamics[i] = [False, False, False]
        else:
            raise ValueError("free_film_bottom_frac must be between 0 and 1.")

    return dynamics

def prepare_poscar(
        case_dir: Path,
        itf: Interface,
        free_sub_top_frac: Optional[float] = None,
        free_film_bottom_frac: Optional[float] = None,
):
    dynamics = set_selective_dynamics(itf, free_sub_top_frac, free_film_bottom_frac)
    poscar = Poscar(itf, selective_dynamics=dynamics)
    poscar.write_file(case_dir / "POSCAR")

def prepare_kpoints(case_dir: Path, itf: Interface, kspacing: float):
    rec_lengths = itf.lattice.reciprocal_lattice.lengths
    kx, ky, kz = [int(math.ceil(g/kspacing)) for g in rec_lengths]
    kz = 1
    kpoints = Kpoints.gamma_automatic(tuple((kx, ky, kz)))
    kpoints.write_file(case_dir / "KPOINTS")

def prepare_potcar(case_dir: Path, potcar_root: Path, sym_potcar_map: dict | None = None):
    poscar_path = case_dir / "POSCAR"
    poscar = Poscar.from_file(poscar_path)
    symbols = list(poscar.site_symbols)
    out_path = case_dir / "POTCAR"

    with out_path.open("wb") as fout:
        for el in symbols:
            pot_name = sym_potcar_map.get(el, el) if sym_potcar_map else el
            pot_path = potcar_root / pot_name / "POTCAR"
            if not pot_path.exists():
                raise FileNotFoundError(f"POTCAR not found for {pot_name}")
            fout.write(pot_path.read_bytes())

def prepare_job_array(out_dir: Path, job_cfg: SgeJobConfig | SlurmJobConfig, sample_cfg: GridSampleConfig, scheduler: str):
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    if scheduler == "sge":
        scripts = f"""#!/bin/bash -l
#$ -N {job_cfg.job_name}
#$ -P {job_cfg.queue_type}
#$ -A {job_cfg.project}
#$ -t 1-{sample_cfg.n_cases}
#$ -pe mpi {job_cfg.cores}
#$ -l h_rt={job_cfg.walltime}
#$ -l mem={job_cfg.mem_per_core}
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err

module unload -f compilers mpi
module load compilers/intel/2019/update5
module load mpi/intel/2019/update5/intel

export PATH=$HOME/vasp.6.4.2/bin:$PATH

BASE_DIR=$(pwd)
CASE_DIR=$(printf "case%02d" $SGE_TASK_ID)

echo "Using VASP: $(which vasp_std)"
echo "Running case: $CASE_DIR"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd "$BASE_DIR/$CASE_DIR" || exit 1
gerun vasp_std > vasp.out

echo "Finish: $(date)"
"""
    else:
        scripts = f"""#!/bin/bash -l
#SBATCH --job-name={job_cfg.job_name}
#SBATCH --array=1-{sample_cfg.n_cases}
#SBATCH --nodes=1
#SBATCH --ntasks={job_cfg.cores}
#SBATCH --cpus-per-task=1
#SBATCH --time={job_cfg.walltime}
#SBATCH --output=logs/%x.%A.%a.out
#SBATCH --error=logs/%x.%A.%a.err

module purge
module load ucl-stack/2025-12
module load compilers/intel-oneapi/2024.2.1/gcc-12.3.0-avx2
module load mpi/intel-oneapi-mpi/2021.14.0/intel-oneapi-2024.2.1-avx2
module load intel-oneapi-mkl/2023.2.0-intel-oneapi-mpi/intel-oneapi-2024.2.1-avx2

export PATH=$HOME/vasp.6.4.2.ng/bin:$PATH
export OMP_NUM_THREADS=1

BASE_DIR=$(pwd)
CASE_DIR=$(printf "case%02d" $SLURM_ARRAY_TASK_ID)

echo "Using VASP: $(which vasp_std)"
echo "Running case: $CASE_DIR"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd "$BASE_DIR/$CASE_DIR" || exit 1
srun vasp_std > vasp.out

echo "Finish: $(date)"
"""

    path = out_dir / "submit_job_array.sh"
    path.write_text(scripts, encoding="utf-8", newline="\n")
    path.chmod(0o755)

def main():
    p = argparse.ArgumentParser(
        description=(
            "Generate VASP inputs for interface registry sampling. "
            "In-plane registries are evaluated on a grid using FGW structural compatibility scores, "
            "and representative registries are selected for subsequent DFT validation."
        )
    )
    p.add_argument("--record-json", required=True, help="Interface record JSON")
    p.add_argument("--embedding", required=True, help="Element embedding JSON/CSV")
    p.add_argument("--potcar-root", required=True, help="Root directory of POTCAR library")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--dft-gap-offset", type=float, default=0.0, help="Additional normal gap offset applied before structure output")
    p.add_argument("--mode", choices=["opt", "scf"], default="opt")
    p.add_argument("--scheduler", choices=["sge", "slurm"], default="sge", help="Output job submission script for SGE or SLURM.")
    p.add_argument("--kspacing", type=float, default=0.25, help="Reciprocal-spcae k-point spacing")
    p.add_argument("--free-sub-top-frac", type=float, default=0.5, help="Top fraction of substrate slab allowed to relax")
    p.add_argument("--free-film-bottom-frac", type=float, default=0.5, help="Bottom fraction of film slab allowed to relax")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Mode] Registry-grid {args.mode} VASP setup")

    with open(args.record_json, "r", encoding="utf-8") as f:
        record = json.load(f, cls=MontyDecoder)

    encoder = GraphEncoder(embedding_path=args.embedding)
    fgw_builder = FGWBuilder(FGWBuildParams(feature_metric="euclidean"))
    scorer = FGWScorer(
        builder=fgw_builder,
        score_params=FGWScoreParams(
            alpha=0.5,
            n_starts=80,
            init_seed=0,
        ),
    )
    bo = RegistryPriorBO(
        encoder=encoder,
        scorer=scorer,
        bo_params=BOParams(
            n_init=20,
            n_iter=60,
            acq_candidates=3000,
            seed=0,
            xi=0.01,
            penalty=1e6,
        ),
        structure_check=False,
    )

    # -----------------------------------------------------------------------------
    # Grid scan and case selection
    # -----------------------------------------------------------------------------

    sample_cfg = GridSampleConfig()
    points = scan_registry_grid(record, bo, sample_cfg)
    pool = select_cases(points, sample_cfg)

    if args.dft_gap_offset != 0.0:
        print(f"[Note] DFT gap offset enabled: {args.dft_gap_offset:.3f}Å will be applied.")

    # Write summary CSV
    summary_path = out_dir / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "shift_a", "shift_b", "shift_c", "fgw_score"])
        for i, point in enumerate(pool, start=1):
            writer.writerow([
                i,
                f"{point.shift_a:.3f}",
                f"{point.shift_b:.3f}",
                f"{point.shift_c + args.dft_gap_offset:.3f}",
                f"{point.fgw_score:.12f}",
            ])

    # -----------------------------------------------------------------------------
    # Prepare VASP inputs
    # -----------------------------------------------------------------------------

    if args.mode == "opt":
        incar_cfg = RelaxConfig()
        job_name = "OPT"
        print("[Note] Relaxation INCAR will be generated.")
    else:
        incar_cfg = StaticConfig()
        job_name = "SCF"
        print("[Note] Static calculation INCAR will be generated.")

    if getattr(incar_cfg, "ldipol", False):
        if not hasattr(incar_cfg, "idipol"):
            raise ValueError("IDIPOL must be set in INCAR config.")
        ref_reg = pool[0].registry
        ref_reg = RegistryPriorBO.shift_film(ref_reg, shift_c=args.dft_gap_offset)
        dipol = build_dipol(ref_reg, idipol=incar_cfg.idipol)
        incar_cfg = replace(incar_cfg, dipol=dipol)
        print(f"[Note] DIPOL set from reference structure: {dipol}")


    if args.scheduler == "sge":
        job_cfg = SgeJobConfig(job_name=job_name)
    else:
        job_cfg = SlurmJobConfig(job_name=job_name)

    potcar_root = Path(args.potcar_root)
    sym_potcar_map = {"Ga": "Ga_d", "Na": "Na_pv", "K": "K_pv"}

    for i, point in enumerate(pool, start=1):
        case_dir = out_dir / f"case{i:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        reg = point.registry
        reg = RegistryPriorBO.shift_film(reg, shift_c=args.dft_gap_offset)
        prepare_incar(case_dir=case_dir, cfg=incar_cfg)
        prepare_kpoints(case_dir=case_dir, itf=reg, kspacing=args.kspacing)
        prepare_poscar(
            case_dir=case_dir,
            itf=reg,
            free_sub_top_frac=args.free_sub_top_frac,
            free_film_bottom_frac=args.free_film_bottom_frac,
        )
        prepare_potcar(case_dir=case_dir, potcar_root=potcar_root, sym_potcar_map=sym_potcar_map)

    prepare_job_array(out_dir=out_dir, job_cfg=job_cfg, sample_cfg=sample_cfg, scheduler=args.scheduler)
    print(f"[Done] Prepared {sample_cfg.n_cases} cases in: {out_dir}")

if __name__ == "__main__":
    main()





















