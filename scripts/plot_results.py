from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def method_style(method: str) -> tuple[str, str]:
    styles = {
        "rusanov": ("#D55E00", "-"),
        "hll": ("#0072B2", "--"),
        "roe": ("#009E73", "-."),
    }
    return styles.get(method, ("black", "-"))


def plot_profiles() -> None:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(RESULTS.glob("profile_*.csv")):
        parts = path.stem.split("_")
        method = parts[-1]
        case = "_".join(parts[1:-1])
        grouped[case].append(path)

    for case, paths in grouped.items():
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for path in paths:
            method = path.stem.split("_")[-1]
            rows = read_csv(path)
            x = [float(row["x"]) for row in rows]
            h = [float(row["h"]) for row in rows]
            u = [float(row["u"]) for row in rows]
            color, linestyle = method_style(method)
            axes[0].plot(x, h, label=method, color=color, linestyle=linestyle)
            axes[1].plot(x, u, label=method, color=color, linestyle=linestyle)

        axes[0].set_ylabel("h")
        axes[1].set_ylabel("u")
        axes[1].set_xlabel("x")
        axes[0].grid(True, alpha=0.25)
        axes[1].grid(True, alpha=0.25)
        axes[0].legend()
        fig.suptitle(case.replace("_", " "))
        fig.tight_layout()
        fig.savefig(PLOTS / f"profile_{case}.png", dpi=160)
        plt.close(fig)


def plot_convergence() -> None:
    path = RESULTS / "convergence_smooth.csv"
    if not path.exists():
        return

    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: int(row["N"]))
        n = [int(row["N"]) for row in method_rows]
        e = [float(row["l1_h"]) for row in method_rows]
        color, linestyle = method_style(method)
        axis.loglog(n, e, marker="o", label=method, color=color, linestyle=linestyle)

    axis.set_xlabel("N")
    axis.set_ylabel("L1 error for h")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "convergence_smooth.png", dpi=160)
    plt.close(fig)


def plot_mass() -> None:
    for path in sorted(RESULTS.glob("mass_*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        t = [float(row["t"]) for row in rows]
        mass0 = float(rows[0]["mass"])
        dm = [float(row["mass"]) - mass0 for row in rows]
        name = path.stem.replace("mass_", "")
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(t, dm)
        axis.set_xlabel("t")
        axis.set_ylabel("M(t) - M(0)")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS / f"mass_{name}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit("results directory not found; run the C++ executable first")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_profiles()
    plot_convergence()
    plot_mass()
    print(f"Plots written to {PLOTS}")


if __name__ == "__main__":
    main()
