"""Recording-level occlusion fidelity (Fig. 6)."""
import numpy as np
import torch

from config import MODEL, OUTPUT_ROOT, PRIMARY_SEED, PRIVATE_I
from data.dataset import normalize_bpm
from data.fixed_splits import load_private_test
from experiments.common import device, load_full_source
from training.io import save_csv
from training.trainer import predict_scores_array

BOOTSTRAPS = 1000
WINDOWS = (8, 32, 64)
RANDOM_MASKS = 5
RNG_SEED = 20261113


@torch.inference_mode()
def match_centers(model, x):
    device = next(model.parameters()).device
    centers = []
    for start in range(0, len(x), 32):
        batch = torch.as_tensor(x[start:start + 32], dtype=torch.float32, device=device)
        info = model.shapelet_match_info(batch)
        winner = info["best_scores"].argmax(dim=1, keepdim=True)
        index = info["best_indices"].gather(1, winner).squeeze(1).cpu().numpy()
        centers.extend(index + model.specific.kernel_size // 2)
    return np.asarray(centers)


def interpolate(x, starts, length):
    modified = x.copy()
    for signal, start in zip(modified[:, 0], starts):
        stop = start + length
        if start == 0 or stop == len(signal):
            signal[start:stop] = signal[stop if start == 0 else start - 1]
        else:
            signal[start:stop] = np.linspace(signal[start - 1], signal[stop], length + 2)[1:-1]
    return modified


def random_starts(rng, targets, length, seq_len):
    before = np.maximum(0, targets - length + 1)
    after_start = targets + length
    after = np.maximum(0, seq_len - length - after_start + 1)
    draw = rng.integers(before + after)
    return np.where(draw < before, draw, after_start + draw - before)


def main():
    test = load_private_test(PRIVATE_I, MODEL.seq_len)
    labels = test.y
    dev = device()
    model, state = load_full_source(PRIVATE_I, PRIMARY_SEED, dev)
    norm = state["normalization"]
    x = normalize_bpm(test.x, norm["mean"], norm["std"])

    def scores(signals):
        return predict_scores_array(model, signals, dev, batch_size=64)

    clean = scores(x)
    centers = match_centers(model, x)
    rng = np.random.default_rng(RNG_SEED)
    classes = [np.flatnonzero(labels == cls) for cls in (0, 1)]
    if not all(group.size for group in classes):
        raise ValueError("Recording-level stratified bootstrap requires both test classes")
    summary = []
    for length in WINDOWS:
        target = np.clip(centers - length // 2, 0, x.shape[-1] - length)
        matched = np.abs(scores(interpolate(x, target, length)) - clean)
        random = []
        for _ in range(RANDOM_MASKS):
            starts = random_starts(rng, target, length, x.shape[-1])
            random.append(np.abs(scores(interpolate(x, starts, length)) - clean))
        random = np.mean(random, axis=0)
        effects = np.stack((matched, random, matched - random), axis=1)
        bootstrap_rng = np.random.default_rng(RNG_SEED)
        boot = np.empty((BOOTSTRAPS, 3))
        for i in range(BOOTSTRAPS):
            indices = np.concatenate([bootstrap_rng.choice(group, size=len(group), replace=True) for group in classes])
            boot[i] = effects[indices].mean(axis=0)
        for i, name in enumerate(("matched", "random", "matched_minus_random")):
            low, high = np.quantile(boot[:, i], (0.025, 0.975))
            summary.append({"window_samples": length, "condition": name,
                            "mean_abs_probability_change": float(effects[:, i].mean()),
                            "ci95_low": float(low), "ci95_high": float(high)})
    out = OUTPUT_ROOT / "fidelity"
    save_csv(summary, out / "fidelity_summary.csv")
    fidelity_plot(summary, out)


def fidelity_plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    ticks = np.arange(len(WINDOWS))
    for condition, offset, color, label in (
        ("matched", -0.18, "#006A70", "Matched region"),
        ("random", 0.18, "#A7B6C2", "Random region"),
    ):
        rows = [row for row in summary if row["condition"] == condition]
        means = [row["mean_abs_probability_change"] * 100 for row in rows]
        ax.bar(ticks + offset, means, width=0.34, color=color, label=label)
        for xpos, row in zip(ticks + offset, rows):
            low, high = 100 * row["ci95_low"], 100 * row["ci95_high"]
            ax.vlines(xpos, low, high, color="#263445", linewidth=1)
            ax.hlines([low, high], xpos - 0.045, xpos + 0.045, color="#263445", linewidth=1)
    ax.set_xticks(ticks, [str(length) for length in WINDOWS])
    ax.set_xlabel("Occlusion window (samples)")
    ax.set_ylabel("Mean absolute probability change (pp)")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"fidelity_occlusion.{suffix}", dpi=300)
    plt.close(fig)
