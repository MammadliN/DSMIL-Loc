#!/usr/bin/env python3
"""
Multi-label evaluation for audio tagging (Task A) and frame-level localization (Task B).
Implements class-specific threshold optimization for Task A and two alternative
thresholding strategies for Task B (single threshold per class or double on/off
thresholds with hysteresis). Thresholds are tuned on the validation data only
and are not learned by the model.
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
import argparse
from torch.utils.data import DataLoader

from data.dataset import LocalizationDataset
from data.collate import collate_localization_bags
from models.mil_net import ImprovedLocalizationMILNet
from utils.visualization import ResultVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _macro_f1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = (preds * labels).sum(axis=0)
    fp = (preds * (1 - labels)).sum(axis=0)
    fn = ((1 - preds) * labels).sum(axis=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.mean(f1))


class LocalizationEvaluator:
    def __init__(
        self,
        model_path: str,
        data_root: str,
        output_dir: str,
        device: str = 'cuda',
        num_classes: int = 1,
        task_b_strategy: str = 'single',
        initial_task_a_thresholds=None,
        initial_task_b_thresholds=None,
        initial_task_b_double_thresholds=None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = num_classes
        self.task_b_strategy = task_b_strategy
        self.initial_task_a_thresholds = initial_task_a_thresholds
        self.initial_task_b_thresholds = initial_task_b_thresholds
        self.initial_task_b_double_thresholds = initial_task_b_double_thresholds or {'on': 0.3, 'off': 0.1}

        # Load model
        self.model = self._load_model()

        # Initialize visualizer
        self.visualizer = ResultVisualizer(str(self.output_dir))

    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
        model = ImprovedLocalizationMILNet(feature_dim=512, num_classes=self.num_classes).to(self.device)

        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _optimize_task_a_thresholds(self, probs: np.ndarray, labels: np.ndarray, step: float = 0.01, max_iter: int = 50):
        """Greedy per-class optimization of clip-level thresholds to maximize macro F1."""
        thresholds = np.array(self.initial_task_a_thresholds if self.initial_task_a_thresholds is not None else np.full(self.num_classes, 0.5))

        def score(ths):
            preds = (probs > ths).astype(float)
            return _macro_f1_np(preds, labels)

        base_f1 = score(thresholds)
        for _ in range(max_iter):
            improved = False
            for c in range(self.num_classes):
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

    def _evaluate_task_a(self, probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
        preds = (probs > thresholds).astype(float)
        return {
            'macro_f1': _macro_f1_np(preds, labels),
            'thresholds': thresholds.tolist()
        }

    def _frame_level_metrics(self, preds: np.ndarray, labels: np.ndarray):
        preds_flat = preds.reshape(-1, self.num_classes)
        labels_flat = labels.reshape(-1, self.num_classes)
        return {
            'macro_f1': _macro_f1_np(preds_flat, labels_flat)
        }

    def _optimize_task_b_single(self, attention: np.ndarray, labels: np.ndarray, step: float = 0.02, max_iter: int = 40):
        thresholds = np.array(self.initial_task_b_thresholds if self.initial_task_b_thresholds is not None else np.full(self.num_classes, 0.2))

        def score(ths):
            preds = (attention > ths).astype(float)
            return _macro_f1_np(preds.reshape(-1, self.num_classes), labels.reshape(-1, self.num_classes))

        base_f1 = score(thresholds)
        for _ in range(max_iter):
            improved = False
            for c in range(self.num_classes):
                best_thr = thresholds[c]
                best_f1 = base_f1
                for delta in (-step, step):
                    candidate = np.clip(thresholds[c] + delta, 0.0, 1.0)
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

    def _apply_double_threshold(self, sequence: np.ndarray, on: float, off: float):
        active = False
        preds = []
        for value in sequence:
            if not active and value >= on:
                active = True
            if active and value < off:
                active = False
            preds.append(1.0 if active else 0.0)
        return np.array(preds)

    def _optimize_task_b_double(self, attention: np.ndarray, labels: np.ndarray, step: float = 0.02, max_iter: int = 20):
        on_th = np.array(self.initial_task_b_double_thresholds.get('on', 0.3))
        off_th = np.array(self.initial_task_b_double_thresholds.get('off', 0.1))
        if on_th.shape == ():
            on_th = np.full(self.num_classes, float(on_th))
        if off_th.shape == ():
            off_th = np.full(self.num_classes, float(off_th))

        def score(on_vals, off_vals):
            preds = np.zeros_like(attention)
            for c in range(self.num_classes):
                for b in range(attention.shape[0]):
                    preds[b, :, c] = self._apply_double_threshold(attention[b, :, c], on_vals[c], off_vals[c])
            return _macro_f1_np(preds.reshape(-1, self.num_classes), labels.reshape(-1, self.num_classes))

        base_f1 = score(on_th, off_th)
        for _ in range(max_iter):
            improved = False
            for c in range(self.num_classes):
                best_on, best_off, best_f1 = on_th[c], off_th[c], base_f1
                for delta_on in (-step, step):
                    for delta_off in (-step, step):
                        cand_on = np.clip(on_th[c] + delta_on, 0.0, 1.0)
                        cand_off = np.clip(off_th[c] + delta_off, 0.0, cand_on - 1e-4)
                        candidate_on = on_th.copy()
                        candidate_off = off_th.copy()
                        candidate_on[c] = cand_on
                        candidate_off[c] = cand_off
                        candidate_f1 = score(candidate_on, candidate_off)
                        if candidate_f1 > best_f1:
                            best_f1 = candidate_f1
                            best_on = cand_on
                            best_off = cand_off
                if best_on != on_th[c] or best_off != off_th[c]:
                    on_th[c] = best_on
                    off_th[c] = best_off
                    base_f1 = best_f1
                    improved = True
            if not improved:
                break
        return on_th, off_th

    def _evaluate_task_b_single(self, attention: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
        preds = (attention > thresholds).astype(float)
        metrics = self._frame_level_metrics(preds, labels)
        metrics['thresholds'] = thresholds.tolist()
        return metrics

    def _evaluate_task_b_double(self, attention: np.ndarray, labels: np.ndarray, on_thresholds: np.ndarray, off_thresholds: np.ndarray):
        preds = np.zeros_like(attention)
        for c in range(self.num_classes):
            for b in range(attention.shape[0]):
                preds[b, :, c] = self._apply_double_threshold(attention[b, :, c], on_thresholds[c], off_thresholds[c])
        metrics = self._frame_level_metrics(preds, labels)
        metrics['on_thresholds'] = on_thresholds.tolist()
        metrics['off_thresholds'] = off_thresholds.tolist()
        return metrics

    def evaluate_site(self, site_years: list) -> dict:
        """Evaluate model performance on specific sites."""
        dataset = LocalizationDataset(self.data_root, site_years, preload_data=False, num_classes=self.num_classes)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_localization_bags
        )

        clip_probs, clip_labels, attention_weights, instance_labels = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue

                spectrograms = batch['spectrograms'].to(self.device)
                features = batch['features'].to(self.device)
                num_instances = batch['num_instances'].to(self.device)

                outputs = self.model(spectrograms, features, num_instances)
                clip_probs.append(torch.sigmoid(outputs['logits']).cpu().numpy())
                clip_labels.append(batch['bag_labels'].numpy())
                attention_weights.append(outputs['attention_weights'].cpu().numpy())
                instance_labels.append(batch['instance_labels'].numpy())

        clip_probs = np.concatenate(clip_probs, axis=0)
        clip_labels = np.concatenate(clip_labels, axis=0)
        attention_weights = np.concatenate(attention_weights, axis=0)
        instance_labels = np.concatenate(instance_labels, axis=0)

        # Task A: optimize clip-level thresholds on validation data
        task_a_thresholds = self._optimize_task_a_thresholds(clip_probs, clip_labels)
        task_a_metrics = self._evaluate_task_a(clip_probs, clip_labels, task_a_thresholds)

        # Task B: two alternative approaches evaluated separately
        single_thresholds = self._optimize_task_b_single(attention_weights, instance_labels)
        task_b_single_metrics = self._evaluate_task_b_single(attention_weights, instance_labels, single_thresholds)

        on_thresholds, off_thresholds = self._optimize_task_b_double(attention_weights, instance_labels)
        task_b_double_metrics = self._evaluate_task_b_double(attention_weights, instance_labels, on_thresholds, off_thresholds)

        final_metrics = {
            'task_a': task_a_metrics,
            'task_b': {
                'single_threshold': task_b_single_metrics,
                'double_threshold': task_b_double_metrics
            }
        }

        self._save_results(final_metrics, site_years)
        return final_metrics

    def _save_results(self, metrics, site_years):
        site_dir = self.output_dir / '_'.join(site_years)
        site_dir.mkdir(exist_ok=True)

        with open(site_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)

        # Visualize clip-level confusion (Task A)
        try:
            self.visualizer.plot_confusion_matrix({
                'predictions': (np.array(metrics['task_a']['thresholds']) > 0).astype(int),
                'labels': []
            })
        except Exception:
            logger.info("Confusion matrix skipped (insufficient data for plot).")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained MIL localization model')

    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--site-years', type=str, nargs='+', required=True, help='List of site years to evaluate')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of target classes')
    parser.add_argument('--task-b-strategy', type=str, choices=['single', 'double'], default='single', help='Which Task B thresholding strategy to emphasize')
    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = LocalizationEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device='cuda',
        num_classes=args.num_classes,
        task_b_strategy=args.task_b_strategy
    )

    evaluator.evaluate_site(args.site_years)


if __name__ == "__main__":
    main()
