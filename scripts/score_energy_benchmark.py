import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from mace.calculators import mace_mp
from regfgw.interface_construction import InterfaceBuilder, InterfaceParams
from regfgw.structure_to_graph import GraphEncoder
from regfgw.fgw_metric import FGWBuildParams, FGWBuilder, FGWScoreParams, FGWScorer

def main():
    substrate = Structure.from_file("cells/GaAs.cif")
    film = Structure.from_file("cells/GaP.cif")
    gen = InterfaceBuilder(substrate=substrate, film=film, max_miller_idx=1, interface_params=InterfaceParams(gap=2.3))
    initial_itfs = gen.sum_interface_records(build_bulk_refs=True, structure_check=True)
    print(f"Generated {len(initial_itfs)} minimum-area interfaces.")

    base_itf = initial_itfs[-1]  # 这是 interface dict：{"interface","substrate_bulk","film_bulk",...}

    builder = FGWBuilder(FGWBuildParams())

    scorer = FGWScorer(builder, FGWScoreParams(
        alpha=0.3,
        n_starts=30,
        init_seed=0,
    ))

    # NEW: 实例化 convertor 并传入 GridScanner
    convertor = GraphEncoder(embedding_path="embeddings/cgnf.json")  # 如果你的 GConvertor 有 cutoff/n_rbf 参数，在这里传

    scanner = GridScanner(
        scorer=scorer,
        convertor=convertor,
        na=50, nb=1,
    )
    records = scanner.scan_grids(base_itf)
    # records = scanner.scan_grids_3d(
    #     base_interface=base_itf,  # 你原来传给 scan_grids 的那个 dict
    #     z_min=-3.0,  # 让 film 往下靠近 2 Å（负号表示靠近）
    #     z_max=0.0,  # 0 表示原始间距
    #     nz=30  # z 方向步数（线性）
    # )

    # records = scanner.scan_grids(base_itf)
    print(f"Got {len(records)} records.")

    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

    scores = np.array([r.score for r in records], dtype=float)
    energies = np.empty(len(records), dtype=float)

    for i, r in enumerate(tqdm(records, desc="Computing single point energies")):
        r: ScanRecord
        itf_atoms: Atoms = AseAtomsAdaptor.get_atoms(r.interface)
        itf_atoms.pbc = (True, True, False)
        itf_atoms.calc = calc
        energies[i] = itf_atoms.get_potential_energy()

    # shift_a = np.array([r.shift_a for r in records], dtype=float)
    #
    # # 保险起见按 shift_a 排序（画出来更像“曲线”）
    # order = np.argsort(shift_a)
    # shift_a = shift_a[order]
    # scores_sorted = scores[order]
    # energies_sorted = energies[order]
    #
    # fig, (ax1, ax2) = plt.subplots(
    #     2, 1, figsize=(7, 6), sharex=True,
    #     gridspec_kw={"hspace": 0.08}
    # )
    #
    # # score vs shift_a
    # ax1.scatter(shift_a, scores_sorted, s=30, alpha=0.75, color="tab:blue",
    #             edgecolors="black", linewidths=0.3)
    # ax1.set_ylabel("FGW score", fontsize=12)
    # ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)
    #
    # # energy vs shift_a
    # ax2.scatter(shift_a, energies_sorted, s=30, alpha=0.75, color="tab:red",
    #             edgecolors="black", linewidths=0.3)
    # ax2.set_xlabel("In-plane shift along a (fractional)", fontsize=12)
    # ax2.set_ylabel("Energy (eV)", fontsize=12)
    # ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    # ax2.spines["top"].set_visible(False)
    # ax2.spines["right"].set_visible(False)
    # plt.show()

    fig2, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        scores,
        energies,
        s=35,
        alpha=0.75,
        color="tab:purple",
        edgecolors="black",
        linewidths=0.35,
    )

    ax.set_xlabel("FGW score", fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title("Energy vs FGW score", fontsize=13)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
