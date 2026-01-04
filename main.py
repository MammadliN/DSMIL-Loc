import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.collate import collate_localization_bags, collate_whale_bags
from models.mil_net import ImprovedLocalizationMILNet
from preprocessing.bag_creation import DynamicWhaleBagCreator
from scripts.train_cv import _macro_f1, evaluate, print_metrics, train_model
from training.losses import StableMILLoss
from utils.config_loader import get_default_config


# Path to the root directory of AnuraSet
ANURASET_ROOT = Path(r"C:\\Users\\noma01\\PycharmProjects\\WSSED\\PAM datasets\\AnuraSet\\audio")
# Fraction of training set to carve out as validation (0.0 disables validation)
VALIDATION_FRACTION = 0.0


@dataclass
class BagInstance:
    spectrogram: np.ndarray
    feature_vector: List[float]
    labels: np.ndarray
    timestamp: float


class AnuraSetBagDataset(Dataset):
    """Dataset that groups 3s audio windows into 60s bags using metadata.csv."""

    def __init__(
        self,
        root_dir: Path,
        metadata: pd.DataFrame,
        subset: str,
        class_columns: List[str],
        bag_duration: int = 60,
        instance_duration: int = 3,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.metadata = metadata[metadata["subset"] == subset].copy()
        self.class_columns = class_columns
        self.bag_duration = bag_duration
        self.instance_duration = instance_duration

        # feature extraction helper (reuses existing preprocessing logic)
        self.creator = DynamicWhaleBagCreator(
            root_dir=str(root_dir),
            output_dir=str(root_dir / "_anura_cache"),
            bag_duration=bag_duration,
            instance_duration=instance_duration,
            instance_overlap=0,
        )

        self.temporal_feature_keys = [
            "rms_energy",
            "peak_amplitude",
            "zero_crossing_rate",
            "envelope_mean",
            "envelope_std",
            "skewness",
            "kurtosis",
            "temporal_centroid",
        ]
        self.physics_feature_keys = [
            "snr",
            "source_intensity",
            "spectral_slope",
            "Bm-Ant-A_energy_ratio",
            "Bm-Ant-B_energy_ratio",
            "Bm-Ant-C_energy_ratio",
            "Bp-20Hz_energy_ratio",
            "Bp-High_energy_ratio",
        ]

        self.bags = self._build_bags()

    def _build_bags(self) -> List[Dict]:
        bags: List[Dict] = []

        # Assign bag index inside each base file using 60s windows
        self.metadata["bag_index"] = (self.metadata["min_t"] // self.bag_duration).astype(int)

        grouped = self.metadata.groupby(["site", "fname", "bag_index"])
        for (site, fname, bag_idx), group in tqdm(grouped, desc="Preparing bags"):
            group = group.sort_values("min_t")
            instances: List[BagInstance] = []

            for _, row in group.iterrows():
                audio_path = self._resolve_audio_path(site, row)
                audio, _ = librosa.load(audio_path, sr=self.creator.target_sr)

                spec_features = self.creator.extract_spectrogram_features(audio)
                spec = self._pad_spectrogram(spec_features["spectrogram"])
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

                freqs = librosa.fft_frequencies(sr=self.creator.target_sr, n_fft=self.creator.nfft)
                temporal_features = self.creator.extract_temporal_features(audio)
                physics_features = self.creator.extract_physics_features(
                    audio, spec_features["spec_mag"], freqs
                )

                feature_vector = [
                    *(temporal_features.get(k, 0.0) for k in self.temporal_feature_keys),
                    *(physics_features.get(k, 0.0) for k in self.physics_feature_keys),
                ]
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                label_vector = row[self.class_columns].astype(float).to_numpy(dtype=np.float32)

                timestamp = (
                    datetime.fromisoformat(str(row["date"]))
                    + timedelta(seconds=float(row["min_t"]))
                ).timestamp()

                instances.append(
                    BagInstance(
                        spectrogram=spec,
                        feature_vector=feature_vector,
                        labels=label_vector,
                        timestamp=timestamp,
                    )
                )

            if not instances:
                continue

            bag_label = np.maximum.reduce([inst.labels for inst in instances])
            bag_id = f"{fname}_bag{bag_idx:03d}"

            bags.append(
                {
                    "bag_id": bag_id,
                    "site": site,
                    "bag_label": bag_label,
                    "instances": instances,
                }
            )

        return bags

    def _resolve_audio_path(self, site: str, row: pd.Series) -> Path:
        start = int(row["min_t"])
        end = int(row["max_t"])
        fname = f"{row['fname']}_{start}_{end}.wav"
        return self.root_dir / site / fname

    @staticmethod
    def _pad_spectrogram(spec: np.ndarray, target_shape: Tuple[int, int] = (129, 235)) -> np.ndarray:
        freq_bins, time_bins = spec.shape
        freq_pad = max(0, target_shape[0] - freq_bins)
        time_pad = max(0, target_shape[1] - time_bins)

        spec_padded = np.pad(
            spec,
            ((0, freq_pad), (0, time_pad)),
            mode="constant",
            constant_values=spec.min() if spec.size else 0.0,
        )
        return spec_padded[: target_shape[0], : target_shape[1]]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Dict:
        bag = self.bags[idx]
        instances = bag["instances"]

        spectrograms = np.stack([inst.spectrogram for inst in instances])
        features = np.stack([inst.feature_vector for inst in instances])
        instance_labels = np.stack([inst.labels for inst in instances])
        timestamps = [inst.timestamp for inst in instances]

        return {
            "bag_id": bag["bag_id"],
            "spectrograms": torch.FloatTensor(spectrograms),
            "features": torch.FloatTensor(features),
            "bag_label": torch.FloatTensor(bag["bag_label"]),
            "num_instances": torch.tensor(len(instances), dtype=torch.long),
            "site": bag["site"],
            "instance_labels": torch.FloatTensor(instance_labels),
            "instance_timestamps": timestamps,
        }


def _optimize_task_a_thresholds(probs: np.ndarray, labels: np.ndarray, step: float = 0.01, max_iter: int = 50) -> np.ndarray:
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)

    def score(ths: np.ndarray) -> float:
        preds = (probs > ths).astype(float)
        return _macro_f1(torch.tensor(preds), torch.tensor(labels)).item()

    base_f1 = score(thresholds)
    for _ in range(max_iter):
        improved = False
        for c in range(thresholds.shape[0]):
            best_thr = thresholds[c]
            best_f1 = base_f1
            for delta in (-step, step):
                candidate = np.clip(thresholds[c] + delta, 0.01, 0.99)
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[c] = candidate
                candidate_f1 = score(candidate_thresholds)
                if candidate_f1 > best_f1:
                    best_f1 = candidate_f1
                    best_thr = candidate
            if best_thr != thresholds[c]:
                thresholds[c] = best_thr
                base_f1 = best_f1
                improved = True
        if not improved:
            break
    return thresholds


def _macro_f1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = (preds * labels).sum(axis=0)
    fp = (preds * (1 - labels)).sum(axis=0)
    fn = ((1 - preds) * labels).sum(axis=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.mean(f1))


def _optimize_task_b_single(instance_probs: np.ndarray, instance_labels: np.ndarray, step: float = 0.01, max_iter: int = 50) -> np.ndarray:
    thresholds = np.full(instance_labels.shape[-1], 0.5, dtype=np.float32)

    def score(ths: np.ndarray) -> float:
        preds = (instance_probs > ths).astype(float)
        return _macro_f1_np(preds.reshape(-1, preds.shape[-1]), instance_labels.reshape(-1, instance_labels.shape[-1]))

    base_f1 = score(thresholds)
    for _ in range(max_iter):
        improved = False
        for c in range(thresholds.shape[0]):
            best_thr = thresholds[c]
            best_f1 = base_f1
            for delta in (-step, step):
                candidate = np.clip(thresholds[c] + delta, 0.01, 0.99)
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[c] = candidate
                candidate_f1 = score(candidate_thresholds)
                if candidate_f1 > best_f1:
                    best_f1 = candidate_f1
                    best_thr = candidate
            if best_thr != thresholds[c]:
                thresholds[c] = best_thr
                base_f1 = best_f1
                improved = True
        if not improved:
            break
    return thresholds


def _double_threshold_predictions(probs: np.ndarray, on_thr: float, off_thr: float) -> np.ndarray:
    state = False
    preds = []
    for p in probs:
        if not state and p >= on_thr:
            state = True
        elif state and p < off_thr:
            state = False
        preds.append(1.0 if state else 0.0)
    return np.array(preds, dtype=np.float32)


def _optimize_task_b_double(
    instance_probs: np.ndarray, instance_labels: np.ndarray, instance_counts: List[int]
) -> Dict[str, Tuple[float, float]]:
    num_classes = instance_labels.shape[-1]
    best_thresholds: Dict[str, Tuple[float, float]] = {}

    flat_labels = instance_labels.reshape(-1, num_classes)

    # build per-bag slices from flat probs
    sequences = []
    start = 0
    for count in instance_counts:
        end = start + count
        sequences.append(instance_probs[start:end])
        start = end

    for c in range(num_classes):
        best_f1 = -1.0
        best_on, best_off = 0.5, 0.2
        for on_thr in np.linspace(0.2, 0.8, 13):
            for off_thr in np.linspace(0.05, on_thr - 0.05, max(1, int((on_thr - 0.05) / 0.05))):
                preds_sequence = []
                for probs_seq in sequences:
                    preds_sequence.append(_double_threshold_predictions(probs_seq[:, c], on_thr, off_thr))
                preds_sequence = np.concatenate(preds_sequence)
                f1 = _macro_f1_np(preds_sequence.reshape(-1, 1), flat_labels[:, c : c + 1])
                if f1 > best_f1:
                    best_f1 = f1
                    best_on, best_off = on_thr, off_thr
        best_thresholds[str(c)] = (round(float(best_on), 3), round(float(best_off), 3))

    return best_thresholds


def _collect_outputs(model: ImprovedLocalizationMILNet, loader: DataLoader, device: torch.device):
    bag_probs, bag_labels = [], []
    instance_probs, instance_labels = [], []
    instance_counts: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting outputs"):
            if batch is None:
                continue

            spectrograms = batch["spectrograms"].to(device)
            features = batch["features"].to(device)
            labels = batch["bag_labels"].to(device)
            num_instances = batch["num_instances"].to(device)

            outputs = model(spectrograms, features, num_instances)

            bag_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            bag_labels.append(labels.cpu().numpy())

            raw_scores = outputs["raw_scores"]
            valid_mask = torch.arange(raw_scores.size(1), device=device).unsqueeze(0) < num_instances.unsqueeze(1)
            safe_scores = torch.where(valid_mask.unsqueeze(-1), raw_scores, torch.zeros_like(raw_scores))
            batch_probs = torch.sigmoid(safe_scores).cpu().numpy()
            batch_labels = batch["instance_labels"].cpu().numpy()

            # collect per-bag sequences to allow variable instance counts across batches
            for i in range(batch_probs.shape[0]):
                inst_count = int(num_instances[i].item())
                instance_counts.append(inst_count)
                instance_probs.append(batch_probs[i, :inst_count])
                instance_labels.append(batch_labels[i, :inst_count])

    return (
        np.concatenate(bag_probs),
        np.concatenate(bag_labels),
        np.concatenate(instance_probs),
        np.concatenate(instance_labels),
        instance_counts,
    )


def _create_validation_split(
    metadata: pd.DataFrame,
    class_columns: List[str],
    val_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Split training recordings into train/validation by 60s source clip.

    The dataset only provides train/test in the CSV. We build a validation set by
    sampling 60-second base clips (identified by the fname within each site) so
    that all 3-second windows from the same clip stay together. We also try to
    ensure every class has at least one positive recording in validation.
    """

    if val_fraction <= 0:
        return metadata.copy()

    rng = np.random.default_rng(seed)

    train_df = metadata[metadata["subset"] == "train"].copy()
    other_df = metadata[metadata["subset"] != "train"].copy()

    if train_df.empty:
        return metadata.copy()

    train_groups = train_df[["site", "fname"]].drop_duplicates()
    if train_groups.empty:
        return metadata.copy()

    n_val = max(1, int(round(len(train_groups) * val_fraction)))
    sampled_idx = train_groups.sample(n=n_val, random_state=seed).reset_index(drop=True)
    sampled_keys = set(map(tuple, sampled_idx.values))

    train_mask = train_df.apply(lambda r: (r["site"], r["fname"]) in sampled_keys, axis=1)
    val_df = train_df[train_mask].copy()
    remaining_train = train_df[~train_mask].copy()

    # Ensure validation has at least one positive clip per class when possible
    for cls in class_columns:
        if val_df[cls].sum() > 0:
            continue

        # find a remaining recording with a positive label for this class
        positive_groups = (
            remaining_train.groupby(["site", "fname"]).filter(lambda g: g[cls].sum() > 0)
        )
        if positive_groups.empty:
            continue

        candidate_keys = positive_groups[["site", "fname"]].drop_duplicates()
        chosen = candidate_keys.sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        move_mask = (remaining_train["site"] == chosen["site"]) & (
            remaining_train["fname"] == chosen["fname"]
        )

        val_df = pd.concat([val_df, remaining_train[move_mask]], ignore_index=True)
        remaining_train = remaining_train[~move_mask]

    val_df.loc[:, "subset"] = "validation"
    remaining_train.loc[:, "subset"] = "train"

    return pd.concat([remaining_train, val_df, other_df], ignore_index=True)


def _sample_subset(meta: pd.DataFrame, subset: str, fraction: float, seed: int) -> pd.DataFrame:
    if not (0 < fraction <= 1):
        raise ValueError(f"Fraction for {subset} must be in (0, 1], got {fraction}")
    subset_df = meta[meta["subset"] == subset].copy()
    if fraction < 1:
        subset_df = subset_df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return subset_df


def prepare_datasets(
    root_dir: Path,
    fraction: float = 1.0,
    seed: int = 42,
    val_fraction: float = VALIDATION_FRACTION,
) -> Tuple[AnuraSetBagDataset, AnuraSetBagDataset, AnuraSetBagDataset, List[str]]:
    metadata_path = root_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)

    # class columns are all columns after subset
    subset_idx = list(metadata.columns).index("subset")
    class_columns = metadata.columns[subset_idx + 1 :].tolist()

    metadata = _create_validation_split(metadata, class_columns, val_fraction=val_fraction, seed=seed)

    train_meta = _sample_subset(metadata, "train", fraction, seed)
    val_meta = _sample_subset(metadata, "validation", fraction, seed)
    test_meta = _sample_subset(metadata, "test", fraction, seed)

    train_ds = AnuraSetBagDataset(root_dir, train_meta, subset="train", class_columns=class_columns)
    val_ds = AnuraSetBagDataset(root_dir, val_meta, subset="validation", class_columns=class_columns)
    test_ds = AnuraSetBagDataset(root_dir, test_meta, subset="test", class_columns=class_columns)

    return train_ds, val_ds, test_ds, class_columns


def run_pipeline(
    root_dir: Path = ANURASET_ROOT,
    fraction: float = 1.0,
    seed: int = 42,
    val_fraction: float = VALIDATION_FRACTION,
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
):
    config = get_default_config()
    config["paths"]["results_dir"] = str(Path("results") / "anuraset")
    config["training"]["num_epochs"] = 1000
    config["training"]["batch_size"] = 2
    config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["training"]["patience"] = early_stopping_patience

    train_ds, val_ds, test_ds, class_columns = prepare_datasets(
        root_dir, fraction=fraction, seed=seed, val_fraction=val_fraction
    )
    num_classes = len(class_columns)

    config["model"]["num_classes"] = num_classes
    config["evaluation"]["task_a_thresholds"] = [0.5] * num_classes

    device = torch.device(config["training"]["device"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_whale_bags,
    )
    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_whale_bags,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_whale_bags,
    )

    model = ImprovedLocalizationMILNet(
        feature_dim=config["model"]["feature_dim"],
        num_heads=config["model"]["num_heads"],
        num_classes=num_classes,
    ).to(device)

    # loss with optional class weights
    pos_weight = config["training"].get("pos_weight")
    pos_weight_tensor = (
        torch.tensor(pos_weight, dtype=torch.float32).to(device) if pos_weight is not None else None
    )
    criterion = StableMILLoss(
        pos_weight=pos_weight_tensor,
        smooth_factor=config["training"]["label_smoothing"],
        focal_gamma=config["training"]["focal_gamma"],
    )

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model, history, checkpoint_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        output_dir=results_dir,
        early_stopping_enabled=early_stopping,
    )

    # Rebuild criterion for evaluation to avoid using potentially moved tensor
    eval_criterion = StableMILLoss(
        pos_weight=pos_weight_tensor,
        smooth_factor=config["training"]["label_smoothing"],
        focal_gamma=config["training"]["focal_gamma"],
    )

    # Threshold tuning on test set (acts as validation for weakly supervised setup)
    tuning_eval_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_localization_bags,
    )

    (
        tune_bag_probs,
        tune_bag_labels,
        tune_inst_probs,
        tune_inst_labels,
        tune_inst_counts,
    ) = _collect_outputs(model, tuning_eval_loader, device)

    task_a_thresholds = _optimize_task_a_thresholds(tune_bag_probs, tune_bag_labels)
    task_b_single_thresholds = _optimize_task_b_single(tune_inst_probs, tune_inst_labels)
    task_b_double_thresholds = _optimize_task_b_double(tune_inst_probs, tune_inst_labels, tune_inst_counts)

    with open(results_dir / "thresholds.json", "w") as f:
        json.dump(
            {
                "task_a": task_a_thresholds.tolist(),
                "task_b_single": task_b_single_thresholds.tolist(),
                "task_b_double": task_b_double_thresholds,
            },
            f,
            indent=2,
        )

    # Test evaluation using tuned Task A thresholds
    test_metrics = evaluate(
        model,
        test_loader,
        eval_criterion,
        device,
        threshold=torch.tensor(task_a_thresholds).to(device),
    )
    print_metrics(test_metrics)

    # Task B evaluation on test set
    test_eval_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_localization_bags,
    )
    (
        test_bag_probs,
        test_bag_labels,
        test_inst_probs,
        test_inst_labels,
        test_inst_counts,
    ) = _collect_outputs(model, test_eval_loader, device)

    task_b_single_preds = (test_inst_probs > task_b_single_thresholds).astype(float)
    task_b_single_f1 = _macro_f1_np(
        task_b_single_preds.reshape(-1, task_b_single_preds.shape[-1]),
        test_inst_labels.reshape(-1, test_inst_labels.shape[-1]),
    )

    task_b_double_preds = []
    # rebuild per-bag sequences for hysteresis decoding while keeping flat ordering
    for c in range(test_inst_probs.shape[-1]):
        on_thr, off_thr = task_b_double_thresholds[str(c)]
        class_preds = []
        start = 0
        for count in test_inst_counts:
            end = start + count
            seq = test_inst_probs[start:end, c]
            class_preds.append(_double_threshold_predictions(seq, on_thr, off_thr))
            start = end
        task_b_double_preds.append(np.concatenate(class_preds, axis=0))
    task_b_double_preds = np.stack(task_b_double_preds, axis=-1)
    task_b_double_f1 = _macro_f1_np(
        task_b_double_preds.reshape(-1, task_b_double_preds.shape[-1]),
        test_inst_labels.reshape(-1, test_inst_labels.shape[-1]),
    )

    with open(results_dir / "task_b_metrics.json", "w") as f:
        json.dump(
            {
                "single_threshold_macro_f1": task_b_single_f1,
                "double_threshold_macro_f1": task_b_double_f1,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnuraSet MIL pipeline")
    parser.add_argument("--root", type=Path, default=ANURASET_ROOT, help="Path to AnuraSet root directory")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Single fraction of each split to use (0-1], e.g., 0.1 for 10% quick run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fractional sampling")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VALIDATION_FRACTION,
        help="Fraction of training recordings to hold out as validation (0 disables validation)",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable trend-based early stopping (disabled by default)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Patience for early stopping when enabled",
    )

    args = parser.parse_args()

    run_pipeline(
        root_dir=args.root,
        fraction=args.fraction,
        seed=args.seed,
        val_fraction=args.val_fraction,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )
