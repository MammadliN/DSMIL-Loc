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
ANURASET_ROOT = Path("/path/to/AnuraSet")


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

                freqs = librosa.fft_frequencies(sr=self.creator.target_sr, n_fft=self.creator.nfft)
                temporal_features = self.creator.extract_temporal_features(audio)
                physics_features = self.creator.extract_physics_features(
                    audio, spec_features["spec_mag"], freqs
                )

                feature_vector = [
                    *(temporal_features.get(k, 0.0) for k in self.temporal_feature_keys),
                    *(physics_features.get(k, 0.0) for k in self.physics_feature_keys),
                ]

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


def _optimize_task_b_double(instance_probs: np.ndarray, instance_labels: np.ndarray) -> Dict[str, Tuple[float, float]]:
    num_classes = instance_labels.shape[-1]
    best_thresholds: Dict[str, Tuple[float, float]] = {}

    flat_labels = instance_labels.reshape(-1, num_classes)
    flat_probs = instance_probs.reshape(-1, instance_probs.shape[-1])

    for c in range(num_classes):
        best_f1 = -1.0
        best_on, best_off = 0.5, 0.2
        for on_thr in np.linspace(0.2, 0.8, 13):
            for off_thr in np.linspace(0.05, on_thr - 0.05, max(1, int((on_thr - 0.05) / 0.05))):
                preds = (flat_probs[:, c] > on_thr).astype(float)
                # apply hysteresis per bag to respect sequences
                preds_sequence = []
                start = 0
                for probs_seq in instance_probs[:, :, c]:
                    preds_sequence.append(_double_threshold_predictions(probs_seq, on_thr, off_thr))
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
            instance_probs.append(torch.sigmoid(safe_scores).cpu().numpy())
            instance_labels.append(batch["instance_labels"].cpu().numpy())

    return (
        np.concatenate(bag_probs),
        np.concatenate(bag_labels),
        np.concatenate(instance_probs),
        np.concatenate(instance_labels),
    )


def prepare_datasets(root_dir: Path) -> Tuple[AnuraSetBagDataset, AnuraSetBagDataset, AnuraSetBagDataset, List[str]]:
    metadata_path = root_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)

    # class columns are all columns after subset
    subset_idx = list(metadata.columns).index("subset")
    class_columns = metadata.columns[subset_idx + 1 :].tolist()

    train_ds = AnuraSetBagDataset(root_dir, metadata, subset="train", class_columns=class_columns)
    val_ds = AnuraSetBagDataset(root_dir, metadata, subset="validation", class_columns=class_columns)
    test_ds = AnuraSetBagDataset(root_dir, metadata, subset="test", class_columns=class_columns)

    return train_ds, val_ds, test_ds, class_columns


def run_pipeline():
    config = get_default_config()
    config["paths"]["results_dir"] = str(Path("results") / "anuraset")
    config["training"]["num_epochs"] = 5
    config["training"]["batch_size"] = 2
    config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, test_ds, class_columns = prepare_datasets(ANURASET_ROOT)
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
    )

    # Rebuild criterion for evaluation to avoid using potentially moved tensor
    eval_criterion = StableMILLoss(
        pos_weight=pos_weight_tensor,
        smooth_factor=config["training"]["label_smoothing"],
        focal_gamma=config["training"]["focal_gamma"],
    )

    # Validation evaluation for threshold tuning
    val_eval_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_localization_bags,
    )

    val_bag_probs, val_bag_labels, val_inst_probs, val_inst_labels = _collect_outputs(
        model, val_eval_loader, device
    )

    task_a_thresholds = _optimize_task_a_thresholds(val_bag_probs, val_bag_labels)
    task_b_single_thresholds = _optimize_task_b_single(val_inst_probs, val_inst_labels)
    task_b_double_thresholds = _optimize_task_b_double(val_inst_probs, val_inst_labels)

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

    # Evaluate on validation with tuned thresholds
    val_metrics = evaluate(
        model,
        val_loader,
        eval_criterion,
        device,
        threshold=torch.tensor(task_a_thresholds).to(device),
    )
    print_metrics(val_metrics)

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
    test_bag_probs, test_bag_labels, test_inst_probs, test_inst_labels = _collect_outputs(
        model, test_eval_loader, device
    )

    task_b_single_preds = (test_inst_probs > task_b_single_thresholds).astype(float)
    task_b_single_f1 = _macro_f1_np(
        task_b_single_preds.reshape(-1, task_b_single_preds.shape[-1]),
        test_inst_labels.reshape(-1, test_inst_labels.shape[-1]),
    )

    task_b_double_preds = []
    for c in range(test_inst_probs.shape[-1]):
        on_thr, off_thr = task_b_double_thresholds[str(c)]
        class_preds = []
        for seq in test_inst_probs[:, :, c]:
            class_preds.append(_double_threshold_predictions(seq, on_thr, off_thr))
        task_b_double_preds.append(np.stack(class_preds))
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
    run_pipeline()
