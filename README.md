# ShapeFHR

Research implementation of **ShapeFHR: Global Latent-Space Shapelet Matching for Interpretable Fetal Heart Rate Classification**.

ShapeFHR uses a shared convolution to encode a fetal heart rate (FHR) recording and its reference shapelets. Each pooled shapelet embedding is matched by inner product against all encoded time positions. The signed difference at the best match forms a shapelet token. A shapelet-specific branch, a generic temporal branch and a statistical branch feed a linear classifier through direct feature concatenation.

## Installation

Use Python 3.10–3.12. For GPU training, install the PyTorch build that matches your CUDA environment, then install the remaining requirements.

```bash
python -m pip install -r requirements.txt
python run.py --help
```

## Local data

Place these files under `local_data/` or set `SHAPEFHR_DATA_ROOT` to your data directory. Each private recording is a trusted NPZ with a binary `label` (0 normal, 1 abnormal).

| Local input | Format |
| --- | --- |
| `processed_batch1/` | Private I NPZ files with FHR under `raw_fhr` |
| `processed_batch3_20min/` | Private II NPZ files with FHR under `raw_fhr` |
| `splits/private1.npz`, `splits/private2.npz` | Original fixed split manifests, with `{train,val,test}_file_names`, `{train,val,test}_indices` and `y_{train,val,test}` |
| `shapelets_9_1.pkl` | Private I training-only shapelet bank in model input units |
| `shapelets_batch3_rawfhr_trainonly.pkl` | Private II shapelet bank generated from the training partition's `raw_fhr` in model input units |
| `BabyBeatAnalyzer_TEST.ts` | Official normalized REFM TEST partition from [NeuroFetalNet](https://github.com/BlackThompson/NeuroFetalNet) |

The original manifests and shapelet banks are local inputs. Use trusted files. The manifests specify 15,246 / 5,082 / 5,083 records for Private I and 34,223 / 11,408 / 11,408 for Private II in train / validation / test order. All methods use the same manifest for a given source. The loader computes the normalization from the training partition.

The official REFM TEST file is converted to bpm using `bpm = z * 11.838785 + 142.63593`. For a local export that is already in bpm, set `SHAPEFHR_REFM_TEST` to the folder containing one NPZ per record with `clean_fhr` and `label`. Both inputs must contain 519 normal and 521 abnormal recordings. The evaluation uses only the FHR channel.

```bash
python run.py check
```

If the original shapelet bank is unavailable, this command generates a training-only bank of 15 templates per class using PIP candidates and CID-based information gain. For every experiment on a given private source, use the same bank generated from its fixed training partition.

```bash
python -m experiments.discover_shapelets private1 --out local_data/shapelets_9_1.pkl
python -m experiments.discover_shapelets private2 --out local_data/shapelets_batch3_rawfhr_trainonly.pkl
```

Generate the Private II bank from `raw_fhr` and train Private II models on that field before running evaluations with this release.

## Run the experiments

Run from the repository root. The standard multi-seed runs use **42, 2024, 3407, 777 and 1234** on **both** private datasets, with fixed partition membership. The default training command runs ShapeFHR and all 12 baselines. Train the applicable models before running subsequent evaluations. Use a fresh output directory for this release so the comparisons all use the current training protocol.

| Manuscript | Command |
| --- | --- |
| Table I · Within-dataset evaluation | `python run.py train` |
| Table II · Direct cross-dataset evaluation | `python run.py cross` |
| Table III · Branch and matching ablations | `python run.py ablation` |
| Table IV · Top-K sensitivity, Private I seed 42 | `python run.py topk` |
| Tables V–VI and Fig. 4 · Composite-artifact robustness, Private I source | `python run.py robustness` |
| Fig. 5 · Representative shapelet matches | `python run.py visualize` |
| Fig. 6 · Occlusion fidelity, Private I seed 42 | `python run.py fidelity` |
| Appendix Fig. A1 · Inference complexity | `python run.py complexity` |

For example, `python run.py train --models shapefhr` trains ShapeFHR for ten source–seed pairs, followed by `python run.py cross --models shapefhr` for ten direct REFM evaluations. `--sources private1` or `--sources private2` can select a source for training or cross-dataset evaluation. The cross-dataset command loads the saved source model and uses its source-training normalization and source-validation threshold as-is. REFM TEST is used for inference and scoring only.

The visualization command plots the learned checkpoint shapelet and an equal-length real FHR context around the selected global match. It generates representative examples from the supplied private test set. The complexity command uses fixed-weight matching for the ShapeFHR FLOPs count and includes statistical feature extraction.

The default output directory is `outputs/`. Set `SHAPEFHR_OUTPUT_ROOT` to use a different directory. Checkpoints and metrics are generated locally. Use a separate output directory for each run when retaining previous summaries.

## Paper settings

| Setting | Value |
| --- | --- |
| Input and final bank | Single-channel FHR, 4,800 samples, five shapelets per class |
| Branch dimensions | 128 / 64 / 32 |
| Shared encoder | Conv1D, kernel 8, stride 1, valid padding |
| Generic temporal branch | Kernels 7 / 15 / 31, average-pooling stride 8 |
| Transformers | One layer, eight heads, feed-forward width 256 |
| Deep-model training | AdamW, maximum 300 epochs, batch size 128, weight decay 0.001, focal loss with γ=1.5 and α=0.60 |
| Initial learning rate | Private I 0.0001, Private II 0.00008 |
| ShapeFHR shapelet freeze | First 10 epochs on Private I, first 5 on Private II |
| Selection and threshold | Maximize G-mean on the source validation partition (thresholds 0.01–0.99 in steps of 0.005) |
| ShapeFHR loss | Focal loss plus soft false-positive penalty (0.05) and shapelet deviation penalty (0.5) from the first epoch |
| Matching controls | Global signal-space CID over all valid window starts and local latent-space matching within the template interval extended by 200 samples on each side |

The defaults below define the public run protocol. All neural runs use the shared training configuration. The private datasets differ in the learning rate and ShapeFHR freeze period specified in the manuscript. Full ShapeFHR, ablations and Top-K use the same loss coefficients and training settings within a source. The shared configuration uses dropout 0.30, early-stopping patience 100, the existing weighted sampler and learning-rate scheduler, and disables training-time augmentation for all methods. Each baseline retains its architecture and method-specific objective terms where applicable. Fetalet retains its prototype-distance term with the focal classification objective. XGBoost uses the source validation G-mean to select the boosting round and threshold.

For robustness, each test recording and each of three repetitions gets a seeded set of non-overlapping artifacts covering 5%–10% of its samples. Coverage counts positions in artifact intervals, including the boundary ramps. At each step the paper's event weights are renormalized over the types that fit the remaining budget and spike limit. Perturbations are applied in bpm before source normalization. Every model and training seed receives the same perturbed recordings within a test domain. The clean and perturbed evaluations use the same frozen source checkpoint and threshold. The protocol averages the three repetitions within a training seed before summarizing across seeds.

This repository contains source code, configuration and documentation. Figures and metrics generated by these commands are outputs of new runs. The paper provides the reported figures and numbers. The private data, fixed split manifests, template banks, trained weights and experimental outputs are held locally.

## Citation

If you use this implementation, cite *ShapeFHR: Global Latent-Space Shapelet Matching for Interpretable Fetal Heart Rate Classification*.
