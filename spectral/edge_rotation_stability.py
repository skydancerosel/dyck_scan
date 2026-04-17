#!/usr/bin/env python3
"""
#4: Rotation / stability of edge vs bulk directions over time.

Track subspace angles between Gram SVD directions at consecutive steps:
  - angle(v_1(t), v_1(t+Δ))  per direction
  - principal angles for span(v1,v2) and span(v1,v2,v3)

For both Dyck and SCAN.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

from spectral.gram_edge_functional_modes import get_attn_param_vector

DYCK_CKPT_DIR = Path(__file__).resolve().parent / "fourier_dyck_checkpoints"
SCAN_CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_dyck_plots"
SCAN_FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"

GRAM_WINDOW = 5


def compute_gram_svd(snapshots, center_idx, window):
    start = max(1, center_idx - window + 1)
    end = min(len(snapshots) - 1, center_idx)
    if end - start + 1 < 2:
        return None
    deltas = []
    for i in range(start, end + 1):
        theta_prev = get_attn_param_vector(snapshots[i-1]["state_dict"]).numpy()
        theta_curr = get_attn_param_vector(snapshots[i]["state_dict"]).numpy()
        deltas.append(theta_curr - theta_prev)
    X = np.stack(deltas)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return {"singular_values": S, "Vh": Vh}


def direction_angle(v1, v2):
    """Angle in degrees between two unit vectors."""
    cos = np.clip(np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), 0, 1)
    return np.degrees(np.arccos(cos))


def subspace_principal_angles(A, B):
    """Principal angles (degrees) between column spans of A and B."""
    angles_rad = subspace_angles(A, B)
    return np.degrees(angles_rad)


def analyze_dataset(ckpt_dir, dataset_name, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for tag, ckpt_name in [("grok", f"{dataset_name}_grok_fourier.pt"),
                            ("memo", f"{dataset_name}_memo_fourier.pt")]:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        snapshots = ckpt["snapshots"]
        n_snaps = len(snapshots)

        print(f"\n{'='*50}")
        print(f"{dataset_name} {tag}: {n_snaps} snapshots")
        print(f"{'='*50}")

        # Compute Gram SVD at every valid center index
        svd_history = []
        for center_idx in range(GRAM_WINDOW, n_snaps - 1):
            gram = compute_gram_svd(snapshots, center_idx, GRAM_WINDOW)
            if gram is None:
                continue
            svd_history.append({
                "step": snapshots[center_idx]["step"],
                "center_idx": center_idx,
                "Vh": gram["Vh"],
                "S": gram["singular_values"],
            })

        if len(svd_history) < 2:
            print("  Not enough Gram windows")
            continue

        # Track rotation of individual directions
        steps = [h["step"] for h in svd_history]
        v1_angles = []
        v2_angles = []
        v3_angles = []
        span12_angles = []  # max principal angle for span(v1,v2)
        span123_angles = []

        for i in range(1, len(svd_history)):
            Vh_prev = svd_history[i-1]["Vh"]
            Vh_curr = svd_history[i]["Vh"]

            # Per-direction angles
            n_dirs = min(Vh_prev.shape[0], Vh_curr.shape[0])
            if n_dirs >= 1:
                v1_angles.append(direction_angle(Vh_prev[0], Vh_curr[0]))
            else:
                v1_angles.append(90)
            if n_dirs >= 2:
                v2_angles.append(direction_angle(Vh_prev[1], Vh_curr[1]))
            else:
                v2_angles.append(90)
            if n_dirs >= 3:
                v3_angles.append(direction_angle(Vh_prev[2], Vh_curr[2]))
            else:
                v3_angles.append(90)

            # Subspace angles for span(v1,v2)
            if n_dirs >= 2:
                A = Vh_prev[:2].T  # [p, 2]
                B = Vh_curr[:2].T
                pa = subspace_principal_angles(A, B)
                span12_angles.append(pa.max())
            else:
                span12_angles.append(90)

            if n_dirs >= 3:
                A = Vh_prev[:3].T
                B = Vh_curr[:3].T
                pa = subspace_principal_angles(A, B)
                span123_angles.append(pa.max())
            else:
                span123_angles.append(90)

        results[tag] = {
            "steps": steps[1:],
            "v1_angles": v1_angles,
            "v2_angles": v2_angles,
            "v3_angles": v3_angles,
            "span12_angles": span12_angles,
            "span123_angles": span123_angles,
            "singular_values": [h["S"].tolist() for h in svd_history],
            "sv_steps": steps,
        }

        # Summary
        print(f"  v1 rotation: mean={np.mean(v1_angles):.1f}°, "
              f"min={np.min(v1_angles):.1f}°, max={np.max(v1_angles):.1f}°")
        print(f"  v2 rotation: mean={np.mean(v2_angles):.1f}°")
        print(f"  v3 rotation: mean={np.mean(v3_angles):.1f}°")
        print(f"  span(v1,v2) max angle: mean={np.mean(span12_angles):.1f}°")

    if not results:
        return results

    # ── Plot 1: Per-direction rotation angles ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset_name}: Direction Rotation Between Consecutive Windows", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        ax = axes[i]
        r = results[tag]
        ax.plot(r["steps"], r["v1_angles"], 'o-', color="steelblue", label="v1 (edge)",
                markersize=3, linewidth=1)
        ax.plot(r["steps"], r["v2_angles"], 's-', color="royalblue", label="v2 (edge)",
                markersize=3, linewidth=1)
        ax.plot(r["steps"], r["v3_angles"], '^-', color="coral", label="v3 (bulk)",
                markersize=3, linewidth=1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Rotation angle (degrees)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 95)
        ax.axhline(y=90, ls='--', color='gray', alpha=0.3, label='orthogonal')

    plt.tight_layout()
    fig.savefig(fig_dir / f"rotation_per_direction_{dataset_name}.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Subspace stability ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset_name}: Subspace Stability (Max Principal Angle)", fontsize=14)

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        ax = axes[i]
        r = results[tag]
        ax.plot(r["steps"], r["span12_angles"], 'o-', color="steelblue",
                label="span(v1,v2)", markersize=3)
        ax.plot(r["steps"], r["span123_angles"], 's-', color="coral",
                label="span(v1,v2,v3)", markersize=3)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Max principal angle (degrees)")
        ax.set_title(f"{tag}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 95)

    plt.tight_layout()
    fig.savefig(fig_dir / f"subspace_stability_{dataset_name}.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Combined singular values + rotation ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{dataset_name}: Singular Values + Direction Rotation", fontsize=14)

    for col, tag in enumerate(["grok", "memo"]):
        if tag not in results:
            continue
        r = results[tag]

        # Top: singular values
        ax = axes[0, col]
        sv_arr = np.array(r["singular_values"])
        for k in range(min(4, sv_arr.shape[1])):
            ax.plot(r["sv_steps"], sv_arr[:, k], '-', label=f"σ_{k+1}", linewidth=1.5)
        ax.set_ylabel("σ")
        ax.set_title(f"{tag}: Singular values")
        ax.legend(fontsize=7)

        # Bottom: rotation
        ax = axes[1, col]
        ax.plot(r["steps"], r["v1_angles"], 'o-', color="steelblue",
                label="v1", markersize=2)
        ax.plot(r["steps"], r["v2_angles"], 's-', color="royalblue",
                label="v2", markersize=2)
        ax.plot(r["steps"], r["v3_angles"], '^-', color="coral",
                label="v3", markersize=2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Rotation (°)")
        ax.set_title(f"{tag}: Direction stability")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 95)

    plt.tight_layout()
    fig.savefig(fig_dir / f"sv_and_rotation_{dataset_name}.png", dpi=150)
    plt.close(fig)

    # Save
    torch.save(results, fig_dir / f"rotation_stability_{dataset_name}.pt")
    return results


def main():
    dyck_results = analyze_dataset(DYCK_CKPT_DIR, "dyck", FIG_DIR)
    scan_results = analyze_dataset(SCAN_CKPT_DIR, "scan", SCAN_FIG_DIR)

    print("\n" + "="*70)
    print("SUMMARY: Edge/Bulk Direction Rotation Stability")
    print("="*70)

    for name, results in [("DYCK", dyck_results), ("SCAN", scan_results)]:
        for tag in ["grok", "memo"]:
            if tag not in results:
                continue
            r = results[tag]
            # Split into pre-grok and post-grok halves
            mid = len(r["v1_angles"]) // 2
            pre = slice(0, mid)
            post = slice(mid, None)
            print(f"\n  {name} {tag}:")
            print(f"    v1 rotation: pre-grok={np.mean(r['v1_angles'][pre]):.1f}°, "
                  f"post-grok={np.mean(r['v1_angles'][post]):.1f}°")
            print(f"    v2 rotation: pre-grok={np.mean(r['v2_angles'][pre]):.1f}°, "
                  f"post-grok={np.mean(r['v2_angles'][post]):.1f}°")
            print(f"    v3 rotation: pre-grok={np.mean(r['v3_angles'][pre]):.1f}°, "
                  f"post-grok={np.mean(r['v3_angles'][post]):.1f}°")
            print(f"    span(v1,v2): pre={np.mean(r['span12_angles'][pre]):.1f}°, "
                  f"post={np.mean(r['span12_angles'][post]):.1f}°")


if __name__ == "__main__":
    main()
