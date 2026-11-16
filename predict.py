"""Evaluate the released checkpoint on the released case.

    python predict.py

Prints the field error of the trained model on data/P1_L08_C1.npz and writes a
PNG that compares the prediction with the reference field.

The model is the one described in the paper: a multilayer perceptron evaluated at
coordinates measured from the centre of each turbine, shared by all turbines, with
the individual deficits composed by a fixed root-sum-square rule.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
D = 126.0                                   # rotor diameter, m
N_TURB = 9

# Layout L08 of the paper, in metres.  Written out here so that the script runs
# on the released case without any other file.
LAYOUT_L08 = np.array([
    [1246.18, -212.48], [1156.96, -532.49], [670.54, 291.76],
    [1284.88, 593.29], [976.17, 526.58], [333.97, -326.58],
    [363.14, 537.76], [748.08, -188.64], [1509.10, -507.60],
], dtype=np.float32)

# Inflow condition C1 of the paper: 11.4 m/s, 6 % turbulence, shear 0.14.
U_REF, TI, ALPHA = 11.4, 6.0, 0.14


class CondField(nn.Module):
    """The network: (relative x, relative y, three inflow scalars) -> deficit."""

    def __init__(self, hidden: int = 64, n_layers: int = 2, n_cond: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2 + n_cond, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor, turb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        rel = (xy[:, None, :] - turb[None, :, :]) / D          # (M, N, 2)
        c = cond[None, None, :].expand(rel.shape[0], rel.shape[1], -1)
        delta = torch.clamp(F.softplus(self.mlp(torch.cat([rel, c], -1)).squeeze(-1)), max=0.75)
        s = (delta ** 2).sum(-1) + 1e-6                        # fixed composition, no fit
        return torch.clamp(1.0 - torch.sqrt(s), min=0.05)      # u / U_ref


def load_model(path: Path) -> CondField:
    ckpt = torch.load(path, map_location="cpu")
    net = CondField(hidden=ckpt.get("hidden", 64), n_cond=ckpt.get("n_cond", 3))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "p1_rss.pt"))
    ap.add_argument("--case", default=str(ROOT / "data" / "P1_L08_C1.npz"))
    ap.add_argument("--png", default="", help="optional path for a comparison figure")
    args = ap.parse_args()

    ref = np.load(args.case)
    u_ref_field, x, y = ref["U_mean"], ref["x"], ref["y"]
    print(f"case {Path(args.case).name}: grid {len(y)} x {len(x)}, "
          f"U_ref = {U_REF} m/s, TI = {TI} %, alpha = {ALPHA}")

    xx, yy = np.meshgrid(x, y)
    xy = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
    turb = torch.tensor(LAYOUT_L08, dtype=torch.float32)
    cond = torch.tensor([U_REF / 11.4, TI / 10.0, ALPHA / 0.2], dtype=torch.float32)

    net = load_model(Path(args.model))
    with torch.no_grad():
        pred = (net(xy, turb, cond) * U_REF).numpy().reshape(len(y), len(x))

    rmse = float(np.sqrt(((pred - u_ref_field) ** 2).mean()))
    print(f"parameters     {sum(p.numel() for p in net.parameters())}")
    print(f"RMSE           {rmse:.4f} m/s  ({rmse / U_REF * 100:.2f} % of U_ref)")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        for ax, f, t in zip(axes, (pred, u_ref_field), ("model", "FAST.Farm")):
            im = ax.imshow(f, origin="lower", cmap="viridis", vmin=4, vmax=9,
                           extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
            ax.set_title(t)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im, ax=axes, label="U (m/s)")
        fig.savefig(args.png, dpi=300)
        print(f"figure         {args.png}")


if __name__ == "__main__":
    main()
