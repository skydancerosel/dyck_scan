#!/usr/bin/env python3
"""
Compositional structure test for SCAN.

SCAN commands are compositional: "jump left twice" decomposes into
action_type(jump) × direction(left) × repetition(twice).

Tests whether the model's representations factorize into:
  (a) Action type features (jump/walk/run/look/turn)
  (b) Modifier features (left/right/opposite/around)
  (c) Repetition features (twice/thrice)
  (d) Cross-terms of the above
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from spectral.fourier_functional_scan import load_scan_model_at_step, extract_scan_hidden_reps

CKPT_DIR = Path(__file__).resolve().parent / "fourier_scan_checkpoints"
FIG_DIR = Path(__file__).resolve().parent / "fourier_scan_plots"

# SCAN command vocabulary categories
ACTION_VERBS = {"jump", "walk", "run", "look", "turn"}
DIRECTIONS = {"left", "right"}
MODIFIERS = {"opposite", "around"}
REPETITIONS = {"twice", "thrice"}
CONNECTORS = {"and", "after"}


def extract_command_features(cmd_vocab, src_tokens):
    """Extract compositional features from command tokens.

    For each sequence, identify:
    - Primary action verb (first verb in command)
    - Direction modifier
    - Repetition count
    """
    N, S = src_tokens.shape

    # Feature arrays
    action_feat = np.zeros((N, S, len(ACTION_VERBS)), dtype=np.float32)
    dir_feat = np.zeros((N, S, 2), dtype=np.float32)  # left/right
    rep_feat = np.zeros((N, S, 2), dtype=np.float32)  # twice/thrice
    mod_feat = np.zeros((N, S, 2), dtype=np.float32)  # opposite/around

    action_list = sorted(ACTION_VERBS)
    action_idx = {v: i for i, v in enumerate(action_list)}

    for i in range(N):
        # Decode command tokens
        for j in range(S):
            tok_id = src_tokens[i, j].item()
            if tok_id == 0:  # PAD
                continue
            tok = cmd_vocab.idx2token.get(tok_id, "")

            if tok in action_idx:
                action_feat[i, j, action_idx[tok]] = 1.0
            elif tok == "left":
                dir_feat[i, j, 0] = 1.0
            elif tok == "right":
                dir_feat[i, j, 1] = 1.0
            elif tok == "twice":
                rep_feat[i, j, 0] = 1.0
            elif tok == "thrice":
                rep_feat[i, j, 1] = 1.0
            elif tok == "opposite":
                mod_feat[i, j, 0] = 1.0
            elif tok == "around":
                mod_feat[i, j, 1] = 1.0

    return {
        "action": action_feat,
        "direction": dir_feat,
        "repetition": rep_feat,
        "modifier": mod_feat,
    }


def composition_probe_scan(enc_reps, dec_reps, cmd_feats, tgt_tokens,
                           src_mask, tgt_mask):
    """Test compositional structure in encoder/decoder representations."""
    results = {}

    # ── Encoder probing: can representations predict command components? ──
    N_enc, S, D_enc = enc_reps.shape
    enc_flat = enc_reps.reshape(-1, D_enc)
    src_mask_flat = src_mask.reshape(-1).astype(bool)
    enc_valid = enc_flat[src_mask_flat]

    n = len(enc_valid)
    perm = np.random.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]

    probe = Ridge(alpha=1.0)

    # Encoder → action verb (multi-class via argmax of one-hot)
    action_flat = cmd_feats["action"].reshape(-1, cmd_feats["action"].shape[-1])[src_mask_flat]
    action_label = action_flat.argmax(axis=-1)
    has_action = action_flat.sum(axis=-1) > 0
    if has_action.sum() > 100:
        enc_act = enc_valid[has_action]
        lab_act = action_label[has_action]
        n_a = len(enc_act)
        perm_a = np.random.permutation(n_a)
        s_a = int(0.7 * n_a)
        probe.fit(enc_act[perm_a[:s_a]], lab_act[perm_a[:s_a]])
        pred = probe.predict(enc_act[perm_a[s_a:]])
        results["enc→action_verb"] = r2_score(lab_act[perm_a[s_a:]], pred)

    # Encoder → direction
    dir_flat = cmd_feats["direction"].reshape(-1, 2)[src_mask_flat]
    dir_label = dir_flat[:, 1] - dir_flat[:, 0]  # -1 left, 0 none, 1 right
    probe.fit(enc_valid[tr], dir_label[tr])
    pred = probe.predict(enc_valid[te])
    results["enc→direction"] = r2_score(dir_label[te], pred)

    # Encoder → repetition
    rep_flat = cmd_feats["repetition"].reshape(-1, 2)[src_mask_flat]
    rep_label = rep_flat[:, 0] + 2 * rep_flat[:, 1]  # 0=none, 1=twice, 2=thrice
    probe.fit(enc_valid[tr], rep_label[tr])
    pred = probe.predict(enc_valid[te])
    results["enc→repetition"] = r2_score(rep_label[te], pred)

    # Cross features: action × direction
    cross_ad = np.concatenate([
        cmd_feats["action"].reshape(-1, cmd_feats["action"].shape[-1])[src_mask_flat],
        cmd_feats["direction"].reshape(-1, 2)[src_mask_flat],
    ], axis=-1)
    cross_full = np.concatenate([cross_ad, cmd_feats["repetition"].reshape(-1, 2)[src_mask_flat],
                                  cmd_feats["modifier"].reshape(-1, 2)[src_mask_flat]], axis=-1)

    # ── Decoder probing: can decoder reps predict action tokens? ──
    N_dec, T, D_dec = dec_reps.shape
    dec_flat = dec_reps.reshape(-1, D_dec)
    tgt_mask_flat = tgt_mask.reshape(-1).astype(bool)
    tgt_flat = tgt_tokens.reshape(-1)[tgt_mask_flat]
    dec_valid = dec_flat[tgt_mask_flat]

    n_d = len(dec_valid)
    perm_d = np.random.permutation(n_d)
    s_d = int(0.7 * n_d)

    probe.fit(dec_valid[perm_d[:s_d]], tgt_flat[perm_d[:s_d]])
    pred = probe.predict(dec_valid[perm_d[s_d:]])
    results["dec→action_token"] = r2_score(tgt_flat[perm_d[s_d:]], pred)

    # Decoder → position in sequence
    pos_array = np.tile(np.arange(T), N_dec)[tgt_mask_flat]
    probe.fit(dec_valid[perm_d[:s_d]], pos_array[perm_d[:s_d]])
    pred = probe.predict(dec_valid[perm_d[s_d:]])
    results["dec→position"] = r2_score(pos_array[perm_d[s_d:]], pred)

    # Command features → action token (from cmd features broadcast to decoder)
    # How well can compositional input features predict output tokens?
    # Build per-sequence average command features from the unflatten originals
    n_cmd_feats = (cmd_feats["action"].shape[-1] + cmd_feats["direction"].shape[-1] +
                   cmd_feats["repetition"].shape[-1] + cmd_feats["modifier"].shape[-1])
    cmd_all = np.concatenate([
        cmd_feats["action"], cmd_feats["direction"],
        cmd_feats["repetition"], cmd_feats["modifier"],
    ], axis=-1)  # [N, S, n_cmd_feats]

    cmd_avg = np.zeros((N_dec, n_cmd_feats))
    for i in range(N_dec):
        valid_pos = src_mask[i].astype(bool)
        if valid_pos.sum() > 0:
            cmd_avg[i] = cmd_all[i, valid_pos].mean(axis=0)

    # Broadcast to decoder positions
    cmd_broadcast = np.repeat(cmd_avg, T, axis=0)[tgt_mask_flat]
    if len(cmd_broadcast) > 0:
        probe.fit(cmd_broadcast[perm_d[:s_d]], tgt_flat[perm_d[:s_d]])
        pred = probe.predict(cmd_broadcast[perm_d[s_d:]])
        results["cmd_feats→action_token"] = r2_score(tgt_flat[perm_d[s_d:]], pred)

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    analysis_steps = [0, 1000, 2000, 5000, 10000]

    all_results = {}

    for tag, ckpt_name in [("grok", "scan_grok_fourier.pt"), ("memo", "scan_memo_fourier.pt")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Skipping {tag}")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        test_src = ckpt["test_src"][:200]
        test_tgt_in = ckpt["test_tgt_in"][:200]
        test_tgt_out = ckpt["test_tgt_out"][:200]
        cmd_vocab = ckpt["cmd_vocab"]

        src_mask = (test_src != 0).numpy()
        tgt_mask = (test_tgt_out != -100).numpy()
        tgt_tokens = test_tgt_out.numpy()

        cmd_feats = extract_command_features(cmd_vocab, test_src)

        print(f"\n{'='*50}")
        print(f"Composition test: {tag}")
        print(f"{'='*50}")

        all_results[tag] = {}

        for target_step in analysis_steps:
            model, _, actual_step = load_scan_model_at_step(ckpt_path, target_step)
            reps = extract_scan_hidden_reps(model, test_src, test_tgt_in)

            # Use deepest encoder/decoder layer
            enc_reps = reps.get("enc_2", reps.get("enc_1", reps.get("enc_0")))
            dec_reps = reps.get("dec_2", reps.get("dec_1", reps.get("dec_0")))

            comp_result = composition_probe_scan(
                enc_reps, dec_reps, cmd_feats, tgt_tokens, src_mask, tgt_mask)
            all_results[tag][actual_step] = comp_result

            if actual_step % 2000 == 0 or actual_step == 0:
                print(f"\n  Step {actual_step}:")
                for k, v in sorted(comp_result.items()):
                    print(f"    {k}: R²={v:.3f}")

    if not all_results:
        print("No checkpoints found.")
        return

    # ── Plot: Composition R² comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, tag in enumerate(["grok", "memo"]):
        if tag not in all_results:
            continue
        ax = axes[i]
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        r = all_results[tag][late]
        keys = sorted(r.keys())
        vals = [r[k] for k in keys]
        colors = plt.cm.Set2(np.linspace(0, 1, len(keys)))
        ax.barh(range(len(keys)), vals, color=colors)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.replace("→", "\n→") for k in keys], fontsize=8)
        ax.set_xlabel("R²")
        ax.set_title(f"{tag} step={late}")
        ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "composition_r2_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot: R² over training ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, key in enumerate(["enc→action_verb", "dec→action_token"]):
        ax = axes[col]
        for tag, color, ls in [("grok", "steelblue", "-"), ("memo", "coral", "--")]:
            if tag not in all_results:
                continue
            steps_sorted = sorted(all_results[tag].keys())
            vals = [all_results[tag][s].get(key, np.nan) for s in steps_sorted]
            ax.plot(steps_sorted, vals, f'o{ls}', color=color, label=tag, markersize=4)
        ax.set_title(key)
        ax.set_xlabel("Training step")
        ax.set_ylabel("R²")
        ax.legend()
        ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "composition_r2_training.png", dpi=150)
    plt.close(fig)

    torch.save(all_results, FIG_DIR / "composition_results.pt")
    print(f"\nSaved figures to {FIG_DIR}")

    print("\n" + "="*60)
    print("SUMMARY: SCAN Compositional Structure Test")
    print("="*60)
    for tag in all_results:
        steps = sorted(all_results[tag].keys())
        late = steps[-1]
        print(f"\n{tag} (step {late}):")
        for k, v in sorted(all_results[tag][late].items()):
            print(f"  {k}: R²={v:.3f}")


if __name__ == "__main__":
    main()
