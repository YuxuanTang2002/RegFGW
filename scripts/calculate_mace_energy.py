import sys
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp
from tqdm import tqdm


# =============================
# Publication-style formatting
# =============================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while (j + 1) < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window < 3:
        return y.copy()

    w = int(window)
    if w % 2 == 0:
        w += 1

    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = float(np.nanmean(y_pad[i:i + w]))
    return out


def rolling_quantile_band(y: np.ndarray, window: int, q_low=0.25, q_high=0.75):
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w < 3:
        return y.copy(), y.copy()
    if w % 2 == 0:
        w += 1

    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")

    y_low = np.empty_like(y, dtype=float)
    y_high = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        seg = y_pad[i:i + w]
        y_low[i] = float(np.nanquantile(seg, q_low))
        y_high[i] = float(np.nanquantile(seg, q_high))
    return y_low, y_high


def main():
    if len(sys.argv) != 2:
        print("Usage: python score_energy_benchmark.py file.traj")
        return

    traj_path = sys.argv[1]
    title = os.path.splitext(os.path.basename(traj_path))[0]

    calc = mace_mp(
        model="medium",
        device="cpu",
        default_dtype="float64",
    )

    traj = Trajectory(traj_path, "r")

    scores, energies, frames = [], [], []

    for atoms in tqdm(traj, desc="Computing MACE energies"):
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        atoms.info["mace_energy_ev"] = e

        scores.append(float(atoms.info["fgw_score"]))
        energies.append(e)
        frames.append(atoms)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".traj") as tmp:
        tmp_path = tmp.name

    out = Trajectory(tmp_path, "w")
    for at in frames:
        out.write(at)
    out.close()
    traj.close()
    os.replace(tmp_path, traj_path)

    x = np.asarray(scores)
    y = np.asarray(energies)

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    w = max(5, int(round(0.10 * len(xs))))
    y_trend = moving_average(ys, w)
    y_q25, y_q75 = rolling_quantile_band(ys, w)

    rho = spearman_r(x, y)

    scatter_color = "#4C72B0"
    trend_color = "#1F3A5F"
    band_color = "#4C72B0"

    fig = plt.figure(figsize=(5.4, 4.1), dpi=140)
    ax = plt.gca()

    ax.scatter(x, y, s=20, alpha=0.40, color=scatter_color, linewidths=0)
    ax.fill_between(xs, y_q25, y_q75, color=band_color, alpha=0.18, linewidth=0)
    ax.plot(xs, y_trend, color=trend_color, linewidth=2.4)

    ax.set_xlabel("FGW score (bulk continuity)")
    ax.set_ylabel("MACE total energy (eV)")
    ax.set_title(title)

    ax.text(
        0.02, 0.98,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    print(f"[done] energies written into {traj_path}")
    print(f"[spearman] rho = {rho:.6f}")


if __name__ == "__main__":
    main()
