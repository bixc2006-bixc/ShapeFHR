"""Per-record FP32 inference counts and latency with explicit operator conventions."""
from collections import Counter
import math
import statistics
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode

from baselines.runner import TREES
from config import DATASETS, MODEL, OUTPUT_ROOT, PRIMARY_SEED
from data.dataset import normalize_bpm
from data.fixed_splits import paper_files, read_signal
from experiments.common import load_full_source
from models.optimized_inference import OptimizedShapeFHR, verify_optimization
from training.io import save_csv, save_json


class ScalarOps(TorchDispatchMode):
    def __init__(self):
        self.ops = Counter()
        self.seen = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        op = func._schema.name.split("::")[-1]
        self.seen[op] += 1
        out = result[0] if isinstance(result, tuple) and result else result
        if not isinstance(out, torch.Tensor) or not (out.is_floating_point() or out.is_complex()):
            return result

        n = out.numel()
        first = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if op in {"add", "sub", "rsub", "mul", "div", "sqrt", "rsqrt", "reciprocal",
                  "pow", "abs", "exp", "log", "tanh", "neg", "square"}:
            factor = 1
            if out.is_complex():
                factor = 2
                if op == "mul" and all(isinstance(t, torch.Tensor) and t.is_complex()
                                      for t in args[:2]):
                    factor = 6
            if op == "abs" and first is not None and first.is_complex():
                factor = 4
            self.ops[op] += factor * n
        elif op in {"sum", "mean", "avg_pool1d", "avg_pool2d", "_adaptive_avg_pool2d"}:
            # Adaptive pooling regions can overlap at non-divisible lengths.
            if op == "_adaptive_avg_pool2d":
                axes = zip(first.shape[-2:], out.shape[-2:])
                area = math.prod(sum(math.ceil((i + 1) * a / b) - (i * a // b)
                                     for i in range(b)) for a, b in axes)
                self.ops[op] += math.prod(out.shape[:-2]) * area
            elif op.startswith("avg_pool"):
                kernel = args[1]
                self.ops[op] += n * (math.prod(kernel) if isinstance(kernel, (tuple, list)) else kernel)
            elif first.is_floating_point():
                self.ops[op] += first.numel() - (n if op == "sum" else 0)
        elif op in {"std", "var", "var_mean", "std_mean"}:
            self.ops[op] += 4 * first.numel() + n
        elif op in {"_softmax", "softmax", "_safe_softmax"}:
            dim = args[1]
            self.ops[op] += 4 * n - n // out.shape[dim]
        elif op == "native_layer_norm":
            self.ops[op] += 7 * n + result[1].numel() * 2
        elif op in {"native_batch_norm", "_native_batch_norm_legit_no_training",
                    "_native_batch_norm_legit"}:
            self.ops[op] += 4 * n + 2 * first.shape[1]
        elif op in {"sigmoid", "silu", "gelu"}:
            self.ops[op] += {"sigmoid": 4, "silu": 5, "gelu": 5}[op] * n
        elif op in {"_fft_r2c", "_fft_c2r"}:
            x = first if op == "_fft_r2c" else out
            length = math.prod(x.shape[axis] for axis in args[1])
            self.ops[op] += round(2.5 * x.numel() * math.log2(length))
        elif op == "_weight_norm_interface":
            self.ops[op] += 3 * n + 2 * result[1].numel()
        return result


def measure_forward(model, inputs):
    # Lower fused operations to comparable CPU ATen operations for counting.
    # Latency below uses the normal backend.
    trainable = [param for param in model.parameters() if param.requires_grad]
    for param in trainable:
        param.requires_grad_(False)
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    scalar = ScalarOps()
    try:
        with (torch.no_grad(), torch.backends.mkldnn.flags(enabled=False),
              sdpa_kernel(SDPBackend.MATH), scalar, FlopCounterMode(display=False) as mac):
            output = model(*inputs).softmax(dim=1)
            if output.shape != (1, 2) or not torch.isfinite(output).all():
                raise RuntimeError(f"Invalid inference output {output}")
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
        for param in trainable:
            param.requires_grad_(True)
    matrix = mac.get_total_flops()
    if matrix <= 0:
        raise RuntimeError("No matrix or convolution operations were counted")
    return matrix, sum(scalar.ops.values()), dict(scalar.ops), dict(scalar.seen)


def timing(call, device, warmups=8, repeats=30):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmups):
        call()
    sync()
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter_ns()
        call()
        sync()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
            "timed_repeats": repeats,
            "median_ms": statistics.median(elapsed),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                             if device.type == "cuda" else None),
            "peak_gpu_increment_mib": ((torch.cuda.max_memory_allocated(device) - baseline) / 2**20
                                       if device.type == "cuda" else None)}


@torch.no_grad()
def profile_nn(model, arrays, input_pipeline=None):
    model.cpu().eval().requires_grad_(False)
    inputs = tuple(torch.from_numpy(x).float() for x in arrays)
    dense, scalar, breakdown, observed = measure_forward(model, inputs)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    device_inputs = tuple(x.to(dev) for x in inputs)
    row = {
        "parameters": sum(x.numel() for x in model.parameters()),
        "inference_parameters": sum(x.numel() for n, x in model.named_parameters() if n != "prototypes"),
        "model_flops": dense + scalar,
        "conv_matmul_flops": dense, "scalar_flops": scalar,
        "scalar_breakdown": breakdown, "aten_ops": observed,
        **timing(lambda: model(*device_inputs).softmax(1), dev),
    }
    if input_pipeline is not None:
        def pipeline():
            return model(*(torch.as_tensor(a, device=dev) for a in input_pipeline())).softmax(1)
        row["pipeline_median_ms"] = timing(pipeline, dev, warmups=3, repeats=10)["median_ms"]
    else:
        row["pipeline_median_ms"] = row["median_ms"]
    return row


def stat_flops(t):
    # Means, two-pass population std, first differences, abs, range, ratios.
    return t + (4*t + 1) + 2*max(1, t//20) + 1 + (t-1)*3 + 4*(t-1)+1 + 4


def feature_flops(name, templates):
    total = 0
    if name in {"mlp_stat", "xgboost_stat", "shapelet_xgboost"}:
        total += stat_flops(MODEL.seq_len)
    if name in {"shapelet_cnn_attention", "shapelet_xgboost"}:
        if templates is None:
            return None
        from scipy.fft import next_fast_len
        t = MODEL.seq_len
        total += 3*t - 2  # cumulative sums of x and x²
        lengths = [min(t, np.asarray(e.values).size) for e in templates]
        total += sum(11*(t-m+1) for m in set(lengths))  # cached rolling moments
        for m in lengths:
            n = next_fast_len(t + m - 1, real=True)
            # Three real FFTs per template, complex product, window distances.
            total += round(7.5*n*math.log2(n)) + 6*(n//2+1) + 8*m + 7*(t-m+1) + 1
        if name == "shapelet_cnn_attention":
            total += 2*len(templates)
    return total


def profile_tree(booster, inputs, pipeline):
    import json

    dumps = [json.loads(item) for item in booster.get_dump(dump_format="json")]
    def tree_counts(node):
        if "leaf" in node:
            return 0, 1, 0
        children = [tree_counts(child) for child in node["children"]]
        return (1 + sum(c[0] for c in children), sum(c[1] for c in children),
                1 + max(c[2] for c in children))
    counts = [tree_counts(tree) for tree in dumps]
    pred = lambda: booster.inplace_predict(inputs[0])
    row = {"parameters": None, "inference_parameters": None,
           "tree_nodes": sum(a + b for a, b, _ in counts),
           "tree_split_thresholds": sum(a for a, _, _ in counts),
           "tree_leaf_weights": sum(b for _, b, _ in counts),
           "tree_count": len(dumps), "tree_max_path_comparisons": sum(c for _, _, c in counts),
           "model_flops": len(dumps) + 4,
           **timing(pred, torch.device("cpu"))}
    row["pipeline_median_ms"] = (timing(lambda: booster.inplace_predict(pipeline()[0]),
                                      torch.device("cpu"), warmups=3, repeats=10)["median_ms"]
                                 if pipeline is not None else None)
    return row


def plot_complexity(rows, out):
    from matplotlib.figure import Figure

    for source in DATASETS:
        items = [row for row in rows if row["source"] == source][::-1]
        if not items:
            continue
        figure = Figure(figsize=(10, max(3, 0.32 * len(items) + 1.6)), layout="constrained")
        axes = figure.subplots(1, 2, sharey=True)
        labels = [row["model"] for row in items]
        axes[0].set_yticks(range(len(items)), labels)
        for ax, field, title in zip(axes, ("parameters", "figure_flops"),
                                    ("Learned parameters", "Estimated FLOPs per record")):
            ax.set_xscale("log")
            ax.set_title(title)
            ax.set_xlabel("Log scale")
            ax.grid(axis="x", alpha=0.25)
            values = [row[field] for row in items if row[field] is not None]
            lower = min(values, default=1) / 2
            upper = max(values, default=10) * 3
            ax.set_xlim(lower, upper)
            for i, row in enumerate(items):
                value = row[field]
                if value is None:
                    ax.text(lower * 1.15, i, f"{row['tree_nodes']:,} tree nodes", va="center")
                else:
                    color = "#006A70" if row["model"] == "shapefhr" else "#55738F"
                    ax.hlines(i, lower, value, color=color, alpha=0.3)
                    ax.scatter(value, i, color=color)
                    ax.annotate(f"{value:,}", (value, i), xytext=(5, 0),
                                textcoords="offset points", va="center", fontsize=8)
        figure.suptitle(f"{DATASETS[source].name} | 4,800 samples | batch size 1")
        figure.savefig(out / f"complexity_{source}.svg")


def main(names):
    from run import load_source_baseline

    rows = []
    cpu = torch.device("cpu")
    out = OUTPUT_ROOT / "complexity"
    for source in DATASETS.values():
        paths, labels, _ = paper_files(source)["test"]
        raw = read_signal(paths[0], source.signal_key, source.label_key, labels[0], MODEL.seq_len)[None]
        for name in names:
            print(f"[complexity] {source.key} {name}", flush=True)
            if name == "shapefhr":
                model, state = load_full_source(source, PRIMARY_SEED, cpu)
                norm = state["normalization"]
                x = normalize_bpm(raw, norm["mean"], norm["std"])
                optimized = OptimizedShapeFHR(model)
                error, checked, fast = verify_optimization(model, optimized, MODEL.seq_len)
                data = profile_nn(model, (x,))
                optimized.cpu()
                _, fallback = optimized.selected_starts(torch.from_numpy(x))
                if fallback:
                    # Fig. A1 counts the fixed-weight matching path. Find a repeatable
                    # input that exercises that path when the real record hits the guard.
                    generator = torch.Generator().manual_seed(8701)
                    for _ in range(32):
                        candidate = torch.randn(x.shape, generator=generator)
                        if not optimized.selected_starts(candidate)[1]:
                            break
                    else:
                        raise RuntimeError("Unable to profile the fixed-weight matching path")
                    profile_x = candidate.numpy()
                else:
                    profile_x = x
                fast_profile = profile_nn(optimized, (profile_x,))
                data.update({"optimized_flops": fast_profile["model_flops"],
                             "optimized_median_ms": fast_profile["median_ms"],
                             "optimized_profile_input": "synthetic" if fallback else "test_record",
                             "figure_flops": fast_profile["model_flops"],
                             "optimization_max_abs_error": error,
                             "optimization_checked": checked, "optimization_fast_cases": fast})
            else:
                baseline, state = load_source_baseline(source, name, PRIMARY_SEED, cpu)
                norm = state["normalization"]
                x = normalize_bpm(raw, norm["mean"], norm["std"])

                def pipeline():
                    return (baseline.features(x),)

                if name in TREES:
                    baseline.model.set_param({"nthread": 1})
                    data = profile_tree(baseline.model, pipeline(), pipeline)
                else:
                    data = profile_nn(baseline.model, pipeline(), pipeline)
                data["feature_flops"] = feature_flops(name, baseline.templates)
                data["pipeline_flops"] = data["model_flops"] + data["feature_flops"]
                data["figure_flops"] = data["pipeline_flops"]

            result = {"source": source.key, "model": name, **data}
            save_json(result, out / source.key / f"{name}.json")
            rows.append({key: value for key, value in result.items() if not isinstance(value, dict)})
    save_csv(rows, out / "summary.csv")
    plot_complexity(rows, out)
