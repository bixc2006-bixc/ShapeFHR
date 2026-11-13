"""Run the experiments in Sections IV.E鈥揑 with local data and output paths."""
import argparse
from dataclasses import asdict
from itertools import groupby

import numpy as np

from baselines.runner import BASELINES, Baseline
from config import (DATASETS, FINAL_TOPK_PER_CLASS, MODEL, MODEL_VERSION, OUTPUT_ROOT, PAPER_SEEDS,
                    PRIMARY_SEED, PRIVATE_I, REFM_TEST_PATH, ROBUSTNESS_BASE_SEED,
                    ROBUSTNESS_COVERAGE, ROBUSTNESS_REPEATS, TOPK_STUDY, TRAIN_CONFIGS)
from data.dataset import normalize_bpm
from data.fixed_splits import load_private_test, paper_files, read_signal
from data.refm import load_refm_test
from experiments.common import (VARIANTS, device, load_full_source, main_run_dir, prepare_dataset, run_training,
                                save_dataset_protocol)
from robustness.artifacts import SAMPLE_RATE_HZ, inject_artifact_noise
from training.io import load_json, save_csv, save_json
from training.metrics import PAPER_METRICS, metrics_from_scores
from training.seed import set_seed
from training.trainer import predict_scores_array

MODELS = ("shapefhr", *BASELINES)
ABLATIONS = tuple(name for name in VARIANTS if name != "full")


def metric_row(scores, labels, threshold, **keys):
    metrics = metrics_from_scores(labels, scores, threshold)
    return {**keys, **{name: metrics[name] for name in PAPER_METRICS}}


def report(rows, root, group_keys, *, write_summary=True):
    save_csv(rows, root / "per_seed.csv")
    if not write_summary:
        return
    def key(row):
        return tuple(row[name] for name in group_keys)

    columns = [*PAPER_METRICS, *(f"{name}_delta" for name in PAPER_METRICS if f"{name}_delta" in rows[0])]
    summary = []
    for values, group in groupby(sorted(rows, key=key), key=key):
        items = list(group)
        entry = {**dict(zip(group_keys, values)), "n_seeds": len(items)}
        for name in columns:
            scores = [row[name] for row in items]
            entry[f"{name}_mean"] = float(np.mean(scores))
            entry[f"{name}_sd"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
        summary.append(entry)
    save_csv(summary, root / "mean_sd.csv")


def training_run(source, name, seed, split):
    path = main_run_dir(source.key, seed) if name == "shapefhr" else baseline_run(source, name, seed)
    if name == "shapefhr":
        result = run_training(dataset_cfg=source, train_cfg=TRAIN_CONFIGS[source.key], split=split,
                              output_dir=path, seed=seed, topk_per_class=FINAL_TOPK_PER_CLASS)
        return result["test_metrics"]
    set_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    baseline = Baseline(name, source, device())
    threshold, scores = baseline.fit(split, seed, path)
    metrics = metrics_from_scores(split.y_test, scores, threshold)
    save_json({
        "protocol_version": MODEL_VERSION,
        "source": source.key, "model": name, "seed": seed, "threshold": threshold,
        "signal_key": source.signal_key,
        "normalization": {"mean": split.mean, "std": split.std},
        "feature_norm": baseline.feature_norm, "test_metrics": metrics,
        "settings": baseline.settings, "train_config": asdict(baseline.train_config),
    }, path / "result.json")
    return metrics


def baseline_run(source, name, seed):
    return OUTPUT_ROOT / "baselines" / source.key / name / f"seed_{seed}"


def load_source_baseline(source, name, seed, dev):
    path = baseline_run(source, name, seed)
    state = load_json(path / "result.json")
    # Earlier Private I runs already used raw_fhr but did not record the field.
    legacy_signal_key = PRIVATE_I.signal_key if source.key == PRIVATE_I.key else None
    if ((state["source"], state["model"], state["seed"]) != (source.key, name, seed)
            or state.get("protocol_version") != MODEL_VERSION
            or state.get("signal_key", legacy_signal_key) != source.signal_key):
        raise ValueError("Train this baseline with the current source configuration first")
    model = Baseline(name, source, dev)
    if (state["train_config"] != asdict(model.train_config)
            or state["settings"] != model.settings):
        raise ValueError("Train this baseline with the current source configuration first")
    model.restore(path, state["feature_norm"])
    return model, state


def frozen_source(source, name, seed):
    """Return a predictor accepting bpm and the saved source validation threshold."""
    dev = device()
    if name == "shapefhr":
        model, state = load_full_source(source, seed, dev)

        def predict(normalized):
            return predict_scores_array(model, normalized, dev, TRAIN_CONFIGS[source.key].batch_size)
    else:
        model, state = load_source_baseline(source, name, seed, dev)
        predict = model.predict

    norm = state["normalization"]

    def predict_bpm(bpm):
        return predict(normalize_bpm(bpm, norm["mean"], norm["std"]))

    return predict_bpm, float(state["threshold"])


def train(sources, names):
    rows = []
    for source in sources:
        split = prepare_dataset(source)
        save_dataset_protocol(dataset_cfg=source, split=split, output_dir=OUTPUT_ROOT / "main" / source.key)
        for name in names:
            for seed in PAPER_SEEDS:
                metrics = training_run(source, name, seed, split)
                rows.append({"source": source.key, "model": name, "seed": seed,
                             **{key: metrics[key] for key in PAPER_METRICS}})
        del split
    report(rows, OUTPUT_ROOT / "within_dataset", ("source", "model"))


def cross_dataset(sources, names):
    target = load_refm_test(REFM_TEST_PATH)
    rows = []
    for source in sources:
        for name in names:
            for seed in PAPER_SEEDS:
                predict, threshold = frozen_source(source, name, seed)
                rows.append({"threshold": threshold,
                             **metric_row(predict(target.x), target.y, threshold,
                                          source=source.key, model=name, seed=seed, target="REFM TEST")})
    report(rows, OUTPUT_ROOT / "cross_dataset", ("source", "model", "target"))


def sensitivity(task):
    source = PRIVATE_I
    root = OUTPUT_ROOT / task
    split = prepare_dataset(source)
    rows = []
    if task == "ablation":
        cases = [(variant, FINAL_TOPK_PER_CLASS, seed) for variant in ("full", *ABLATIONS)
                 for seed in PAPER_SEEDS]
    else:
        cases = [("full", k, PRIMARY_SEED) for k in TOPK_STUDY]
    validation_gmeans = {}
    for variant, k, seed in cases:
        if variant == "full" and k == FINAL_TOPK_PER_CLASS:
            result = load_json(main_run_dir(source.key, seed) / "result.json")
            if (result.get("protocol_version") != MODEL_VERSION
                    or result["train_config"] != asdict(TRAIN_CONFIGS[source.key])):
                raise ValueError("Train the full model with the current source configuration first")
        else:
            result = run_training(dataset_cfg=source, train_cfg=TRAIN_CONFIGS[source.key], split=split,
                                  output_dir=root / variant / f"k_{k}" / f"seed_{seed}",
                                  seed=seed, topk_per_class=k, variant=variant)
        rows.append({"source": source.key, "variant": variant, "k_per_class": k, "seed": seed,
                     **{name: result["test_metrics"][name] for name in PAPER_METRICS}})
        if task == "topk":
            validation_gmeans[k] = result["val_metrics"]["gmean"]
    if task == "topk":
        selected = max(validation_gmeans, key=validation_gmeans.get)
        print(f"[topk] Validation G-mean selects K={selected}; manuscript configuration uses K={FINAL_TOPK_PER_CLASS}.")
    report(rows, root, ("source", "variant", "k_per_class"), write_summary=task == "ablation")


def corrupt(raw, domain, repeat):
    return np.stack([
        inject_artifact_noise(trace, coverage=ROBUSTNESS_COVERAGE,
                              rng=np.random.default_rng([ROBUSTNESS_BASE_SEED, domain, repeat, i]))
        for i, trace in enumerate(raw)
    ])


def artifact_preview(raw, noisy, labels):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 5), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    time = np.arange(raw.shape[1]) / SAMPLE_RATE_HZ
    for row, cls in enumerate((0, 1)):
        index = np.flatnonzero(labels == cls)[0]
        for col, (signals, title) in enumerate(((raw, "Original FHR"), (noisy, "Artifact-corrupted FHR"))):
            ax = axes[row, col]
            ax.plot(time, signals[index], lw=0.8)
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel("FHR (bpm)")
            if row == 1:
                ax.set_xlabel("Time (s)")
    out = OUTPUT_ROOT / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / "artifact_examples.pdf")


def robustness(names):
    target = load_refm_test(REFM_TEST_PATH)
    private = load_private_test(PRIVATE_I, MODEL.seq_len)
    rows = []
    for domain, (label, raw, y) in enumerate((("private1", private.x, private.y), ("refm", target.x, target.y))):
        perturbed = [corrupt(raw, domain, repeat) for repeat in range(ROBUSTNESS_REPEATS)]
        if domain == 0:
            artifact_preview(raw, perturbed[0], y)
        for name in names:
            for seed in PAPER_SEEDS:
                predict, threshold = frozen_source(PRIVATE_I, name, seed)
                clean = metric_row(predict(raw), y, threshold)
                repeats = [metric_row(predict(item), y, threshold) for item in perturbed]
                row = {"source": PRIVATE_I.key, "target": label, "model": name, "seed": seed}
                for key in PAPER_METRICS:
                    row[key] = float(np.mean([result[key] for result in repeats]))
                    row[f"{key}_delta"] = row[key] - clean[key]
                rows.append(row)
        del perturbed
    report(rows, OUTPUT_ROOT / "robustness", ("source", "target", "model"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train", "cross", "ablation", "topk", "robustness",
                                        "visualize", "fidelity", "complexity", "check"))
    parser.add_argument("--sources", nargs="+", choices=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    if args.sources and args.task not in {"train", "cross", "check"}:
        parser.error("--sources applies to train, cross and check")
    if args.models and args.task not in {"train", "cross", "robustness", "complexity"}:
        parser.error("--models applies to train, cross, robustness and complexity")
    sources = [DATASETS[key] for key in dict.fromkeys(args.sources or DATASETS)]
    names = tuple(dict.fromkeys(args.models or MODELS))
    if args.task == "train":
        train(sources, names)
    elif args.task == "cross":
        cross_dataset(sources, names)
    elif args.task in {"ablation", "topk"}:
        sensitivity(args.task)
    elif args.task == "robustness":
        robustness(names)
    elif args.task == "visualize":
        from experiments.run_visualization import main as visualize
        visualize()
    elif args.task == "fidelity":
        from experiments.fidelity import main as fidelity
        fidelity()
    elif args.task == "complexity":
        from experiments.complexity import main as complexity
        complexity(names)
    else:
        for source in sources:
            parts = paper_files(source)
            for paths, labels, _ in parts.values():
                read_signal(paths[0], source.signal_key, source.label_key, labels[0], MODEL.seq_len)
            print(f"{source.key}: fixed split and first record in each partition verified")
        print(f"REFM TEST: {len(load_refm_test(REFM_TEST_PATH).y)} records verified")


if __name__ == "__main__":
    main()
