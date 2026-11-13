from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(ch_in, ch_out, kernel):
    return nn.Sequential(nn.Conv1d(ch_in, ch_out, kernel, padding=kernel // 2),
                         nn.ReLU(), nn.MaxPool1d(2))


class Fetalet(nn.Module):
    """Reimplementation of Qin et al., doi:10.1007/s40747-025-02095-1."""

    def __init__(self, seq_len=4800, shapelet_length=30, n_prototypes=5,
                 alpha=0.5, margin=1.0):
        super().__init__()
        self.alpha, self.margin = alpha, margin
        self.shared = nn.Sequential(conv_block(1, 16, 7), conv_block(16, 32, 5))
        self.global_block = conv_block(32, 64, 3)
        self.global_projection = nn.Linear(64 * (seq_len // 8), 64)
        self.shapelet_pool = nn.AdaptiveAvgPool1d(shapelet_length)
        self.prototypes = nn.Parameter(torch.zeros(n_prototypes, 32, shapelet_length))
        self.head = nn.Linear(64 + 32 * shapelet_length, 2)

    def features(self, x):
        shared = self.shared(x)
        shapelets = self.shapelet_pool(shared)
        global_features = self.global_projection(self.global_block(shared).flatten(1))
        return self.head(torch.cat((global_features, shapelets.flatten(1)), dim=1)), shapelets

    def forward(self, x):
        return self.features(x)[0]

    def training_objective(self, x, y, classification_loss):
        logits, candidates = self.features(x)
        delta = candidates.float()[:, None] - self.prototypes.float()[None, :]
        distance = torch.linalg.vector_norm(delta.flatten(2), dim=-1).min(dim=1).values
        metric = torch.where(y == 1, distance, F.relu(self.margin - distance)).mean()
        loss = self.alpha * classification_loss(logits.float(), y) + (1 - self.alpha) * metric
        return logits, loss

    @torch.no_grad()
    def initialize_prototypes(self, positive_train, seed, device):
        import numpy as np
        from sklearn.cluster import KMeans

        k = len(self.prototypes)
        if len(positive_train) < k:
            raise ValueError(f"Fetalet K-means requires at least {k} positive training records")
        self.to(device).eval()
        embeddings = []
        for batch in np.array_split(positive_train, (len(positive_train) + 63) // 64):
            x = torch.as_tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
            embeddings.append(self.shapelet_pool(self.shared(x)).flatten(1).cpu().numpy())
        centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(np.concatenate(embeddings))
        self.prototypes.copy_(torch.as_tensor(centers.cluster_centers_, device=device).reshape_as(self.prototypes))
