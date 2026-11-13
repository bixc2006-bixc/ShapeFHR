"""Memory-bounded z-normalized sliding L2 features for the tabular baselines."""
import numpy as np
from scipy.signal import fftconvolve


def distances(x, entries, chunk_size=64):
    x = np.asarray(x)
    n, length = x.shape
    output = np.empty((n, len(entries)), dtype=np.float32)
    templates = []
    for entry in entries:
        s = np.asarray(entry.values, dtype=np.float32).reshape(-1)
        if s.size > length:
            s = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, s.size), s).astype(np.float32)
        s = s.astype(np.float64)
        std = s.std()
        s = (s - s.mean()) / (std if std >= 1e-6 else 1.0)
        templates.append(s)
    # Process identical lengths together to reuse moments without retaining
    # a full rolling-moment matrix for every distinct length.
    order = sorted(range(len(templates)), key=lambda i: templates[i].size)
    for start in range(0, n, chunk_size):
        batch = x[start:start + chunk_size].astype(np.float64, copy=False)
        prefix = np.pad(batch.cumsum(axis=1), ((0, 0), (1, 0)))
        prefix_sq = np.pad((batch * batch).cumsum(axis=1), ((0, 0), (1, 0)))
        previous_length = 0
        for i in order:
            s = templates[i]
            m = s.size
            if m != previous_length:
                mean = (prefix[:, m:] - prefix[:, :-m]) / m
                var = np.maximum((prefix_sq[:, m:] - prefix_sq[:, :-m]) / m - mean**2, 0)
                std = np.sqrt(var)
                safe = np.where(std < 1e-6, 1.0, std)
                window_ss = m * var / safe**2
                previous_length = m
            dot = (fftconvolve(batch, s[None, ::-1], mode="valid", axes=1)
                   - mean * s.sum()) / safe
            squared = window_ss + np.dot(s, s) - 2 * dot
            output[start:start+len(batch), i] = np.sqrt(np.maximum(squared.min(axis=1), 0))
    return output
