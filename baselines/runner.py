"""Baseline implementations from the project under the shared paper protocol."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from config import MODEL, PROJECT_ROOT, FINAL_TOPK_PER_CLASS, TRAIN_CONFIGS
from data.dataset import make_loaders
from data.shapelets import load_shapelets
from baselines.cnn1d import CNN1D
from baselines.edgemtsc import EdgeMTSC
from baselines.fetalet import Fetalet
from baselines.freconvnet import FreConvNet
from baselines.inception_time import InceptionTime
from baselines.mlp_stat import MLPStatClassifier
from baselines.original_shapeformer import OriginalShapeFormer
from baselines.shapelet_cnn_attention import ShapeletCNNAttentionClassifier
from baselines.shapelet_features import distances
from baselines.tcn import TCNClassifier
from baselines.transformer_only import TransformerOnly
from models.branches import statistical_features
from training.io import load_json
from training.threshold import best_gmean_operating_point
from training.trainer import train_model, predict_scores_array

TREES = ("xgboost_stat", "shapelet_xgboost")
BASELINES = (
    *TREES, "cnn1d", "transformer_only", "inception_time", "original_shapeformer",
    "tcn", "mlp_stat", "shapelet_cnn_attention", "freconvnet", "edgemtsc", "fetalet",
)
SHAPELET_MODELS = {"shapelet_xgboost", "original_shapeformer", "shapelet_cnn_attention"}


class Baseline:
    def __init__(self, name, source, device):
        self.name, self.device = name, device
        self.settings = load_json(PROJECT_ROOT / "configs" / "baselines.json")[name]
        self.templates = (load_shapelets(source.shapelets_path, FINAL_TOPK_PER_CLASS)
                          if name in SHAPELET_MODELS else [])
        self.train_config = replace(
            TRAIN_CONFIGS[source.key],
            fp_penalty_lambda=0.0, shapelet_deviation_lambda=0.0, freeze_shapelets_epochs=0,
        )
        self.feature_norm = None
        self.model = None if name in TREES else self._network()

    def _network(self):
        name, opts = self.name, self.settings
        dropout = self.train_config.dropout
        if name == "original_shapeformer":
            cfg = SimpleNamespace(seq_len=MODEL.seq_len, in_channels=1,
                                  num_classes=MODEL.num_classes, dropout=dropout, **opts)
            return OriginalShapeFormer(cfg, self.templates)
        if name == "transformer_only":
            return TransformerOnly(seq_len=MODEL.seq_len, d_model=opts["d_model"], nhead=opts["nhead"],
                                   num_layers=opts["num_layers"], dim_feedforward=opts["dim_feedforward"],
                                   dropout=dropout)
        if name in {"freconvnet", "edgemtsc"}:
            return {"freconvnet": FreConvNet, "edgemtsc": EdgeMTSC}[name](
                seq_len=MODEL.seq_len, in_channels=1, num_classes=MODEL.num_classes,
                dropout=dropout, **opts["architecture"],
            )
        if name == "fetalet":
            return Fetalet(seq_len=MODEL.seq_len)
        if name == "shapelet_cnn_attention":
            return ShapeletCNNAttentionClassifier(num_shapelets=len(self.templates), dropout=dropout)
        if name == "inception_time":
            return InceptionTime(in_ch=1, out_ch=32, depth=6, dropout=dropout)
        return {"cnn1d": CNN1D, "tcn": TCNClassifier, "mlp_stat": MLPStatClassifier}[name](dropout=dropout)

    def features(self, x, *, fit=False):
        name = self.name
        if name in {"xgboost_stat", "mlp_stat", "shapelet_xgboost"}:
            stat = np.concatenate([statistical_features(torch.from_numpy(batch)).numpy()
                                   for batch in np.array_split(x, (len(x) + 255) // 256)])
            if name != "shapelet_xgboost":
                return stat
            return np.concatenate((distances(x[:, 0], self.templates), stat), axis=1)
        if name == "shapelet_cnn_attention":
            shape = distances(x[:, 0], self.templates)
            if fit:
                self.feature_norm = (shape.mean(0), np.maximum(shape.std(0), 1e-6))
            mean, std = self.feature_norm
            return (shape - mean) / std
        return x

    def fit(self, split, seed, output):
        features = replace(split, x_train=self.features(split.x_train, fit=True),
                           x_val=self.features(split.x_val), x_test=self.features(split.x_test))
        if self.name in TREES:
            import xgboost as xgb
            y = split.y_train
            params = dict(self.settings["training"], seed=seed, objective="binary:logistic",
                          scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
            rounds = params.pop("rounds")
            train = xgb.DMatrix(features.x_train, label=y)
            valid = xgb.DMatrix(features.x_val, label=split.y_val)
            validation = []

            def validation_gmean(scores, matrix):
                threshold, gmean = best_gmean_operating_point(matrix.get_label(), scores)
                validation.append((threshold, gmean))
                return "gmean", gmean

            self.model = xgb.train(
                {**params, "disable_default_eval_metric": 1}, train, num_boost_round=rounds,
                evals=[(valid, "validation")], custom_metric=validation_gmean,
                verbose_eval=False,
            )
            best_round = max(range(len(validation)), key=lambda i: validation[i][1])
            self.model = self.model[:best_round + 1]
            threshold = validation[best_round][0]
            self.model.save_model(output / "model.ubj")
        else:
            if self.name == "fetalet":
                self.model.initialize_prototypes(features.x_train[split.y_train == 1, 0], seed, self.device)
            loaders = make_loaders(features, self.train_config, seed)
            result = train_model(self.model, loaders, self.train_config, self.device)
            threshold = result["best_threshold"]
            torch.save(self.model.state_dict(), output / "model.pt")
        return float(threshold), self._predict_features(features.x_test)

    def restore(self, output, feature_norm):
        if self.name in TREES:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(output / "model.ubj")
        else:
            self.model.load_state_dict(torch.load(output / "model.pt", map_location="cpu", weights_only=True))
            self.model.to(self.device).eval()
        if feature_norm is not None:
            self.feature_norm = tuple(np.asarray(value, dtype=np.float32) for value in feature_norm)

    def _predict_features(self, features):
        if self.name in TREES:
            return self.model.inplace_predict(features)
        return predict_scores_array(self.model, features, self.device, self.train_config.batch_size)

    def predict(self, normalized):
        return self._predict_features(self.features(normalized))
