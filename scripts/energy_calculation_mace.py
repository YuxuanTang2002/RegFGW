import argparse
import os
import tempfile
from tqdm import tqdm
from mace.calculators import mace_mp
from ase.io.trajectory import Trajectory

def compute_and_write_mace_energy(
        traj_path,
        model="medium",
        device="cpu",
        default_dtype="float64",
):
    """
    Compute MACE total energies for all frames in an ASE trajectory from Bayesian optimization
    and write them into atoms.info["mace_energy"].

    """
    calc = mace_mp(model=model, device=device, default_dtype=default_dtype)
    traj = Trajectory(traj_path, "r")
    frames = []

    for atoms in tqdm(traj, desc="Computing MACE energies"):
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        atoms.info["mace_energy"] = energy
        frames.append(atoms)

    traj.close()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".traj") as tmp:
        tmp_path = tmp.name

    new_traj = Trajectory(tmp_path, "w")

    for f in frames:
        new_traj.write(f)

    new_traj.close()
    os.replace(tmp_path, traj_path)
    print(f"[done] mace energies written to info['mace_energy'] in {traj_path}")

def main():
    p = argparse.ArgumentParser(description="Compute MACE energies and write them into a traj file.")
    p.add_argument("traj_file", help="Input ASE trajectory file (.traj).")
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cpu")
    p.add_argument("--default_dtype", default="float64")
    args = p.parse_args()
    compute_and_write_mace_energy(
        traj_path=args.traj_file,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
    )

if __name__ == "__main__":
    main()




