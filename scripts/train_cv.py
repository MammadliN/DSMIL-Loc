#!/usr/bin/env python3
"""
Script for training a whale call detection model using cross-validation
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import LocalizationDataset
from data.collate import collate_whale_bags
from models.mil_net import ImprovedLocalizationMILNet
from training.losses import StableMILLoss
from training.earlystopping import TrendBasedEarlyStopping
from utils.visualization import ResultVisualizer
from utils.config_loader import ConfigLoader, get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _macro_f1(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute macro F1 for multi-label predictions."""
    tp = (predictions * labels).sum(dim=0)
    fp = (predictions * (1 - labels)).sum(dim=0)
    fn = ((1 - predictions) * labels).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_per_class.mean()


def _serialize_history(history: dict) -> dict:
    """Convert training history into JSON-serializable primitives.

    Ensures numpy scalars and tensors are cast to floats and recreates lists
    to avoid accidental object sharing that can trigger circular reference
    detection during ``json.dump``.
    """

    def _to_float(value):
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return value

    serialized = {}
    for key in ["train_f1", "val_f1", "train_loss", "val_loss", "learning_rates"]:
        serialized[key] = [
            _to_float(v) for v in history.get(key, [])
        ]

    serialized["trends"] = []
    for trend in history.get("trends", []):
        if isinstance(trend, dict):
            serialized["trends"].append({k: _to_float(v) for k, v in trend.items()})
        else:
            serialized["trends"].append(_to_float(trend))

    return serialized


def _prepare_threshold(threshold, num_classes: int, device: torch.device) -> torch.Tensor:
    """Return a tensor threshold of shape (num_classes,) on the target device.

    Handles scalars, sequences, numpy arrays, and tensors. If a double-threshold
    array is provided (shape ``[2, num_classes]``), the "on" row (index 0) is
    used for clip-level comparisons inside training/validation.
    """
    # Convert to tensor on the right device/dtype
    if isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.to(device=device, dtype=torch.float32)
    else:
        threshold_tensor = torch.as_tensor(threshold, dtype=torch.float32, device=device)

    # If double thresholds are passed, use the "on" thresholds for Task A logic
    if threshold_tensor.ndim > 1 and threshold_tensor.shape[0] == 2:
        threshold_tensor = threshold_tensor[0]

    # Flatten and broadcast to the number of classes
    threshold_tensor = threshold_tensor.flatten()
    if threshold_tensor.numel() == 1:
        threshold_tensor = threshold_tensor.expand(num_classes)
    elif threshold_tensor.numel() != num_classes:
        # Safeguard: truncate or pad (with the last value) to expected size
        if threshold_tensor.numel() > num_classes:
            threshold_tensor = threshold_tensor[:num_classes]
        else:
            pad_value = threshold_tensor[-1].item()
            pad_len = num_classes - threshold_tensor.numel()
            threshold_tensor = torch.cat([
                threshold_tensor,
                torch.full((pad_len,), pad_value, device=device, dtype=threshold_tensor.dtype)
            ])

    return threshold_tensor


def train_epoch(model, loader, criterion, optimizer, device, config):
    """Run single training epoch with stability improvements"""
    epoch_metrics = {'loss': 0.0, 'f1': 0.0}
    num_batches = len(loader)
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        if batch is None:
            continue
            
        # Move data to device
        spectrograms = batch['spectrograms'].to(device)
        features = batch['features'].to(device)
        labels = batch['bag_labels'].to(device)
        num_instances = batch['num_instances'].to(device)
        
        # Forward pass
        outputs = model(spectrograms, features, num_instances)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['max_grad_norm']
        )
        
        optimizer.step()
        
        # Update metrics
        epoch_metrics['loss'] += loss.item() / num_batches
        
        # Compute F1 score
        evaluation_cfg = config.get('evaluation', {})
        raw_threshold = evaluation_cfg.get('task_a_thresholds', 0.5)
        threshold_tensor = _prepare_threshold(raw_threshold, outputs['logits'].shape[-1], device)
        predictions = (torch.sigmoid(outputs['logits']) > threshold_tensor).float()

        batch_f1 = _macro_f1(predictions, labels)
        epoch_metrics['f1'] += batch_f1.item() / num_batches
        
        # Log progress
        if (batch_idx + 1) % config['logging']['log_every'] == 0:
            logger.info(f"Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}, F1: {batch_f1.item():.4f}")
    
    return epoch_metrics

def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model on validation or test set"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    threshold_tensor = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
                
            # Move data to device
            spectrograms = batch['spectrograms'].to(device)
            features = batch['features'].to(device)
            batch_labels = batch['bag_labels'].to(device)
            num_instances = batch['num_instances'].to(device)
            
            # Forward pass
            outputs = model(spectrograms, features, num_instances)
            loss = criterion(outputs, batch_labels)
            
            # Store loss
            total_loss += loss.item()
            
            # Lazily prepare threshold to match current batch/device
            if threshold_tensor is None:
                threshold_tensor = _prepare_threshold(
                    threshold, outputs['logits'].shape[-1], device
                )

            # Get predictions
            batch_preds = (torch.sigmoid(outputs['logits']) > threshold_tensor).float()
            
            # Store predictions and labels
            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Compute metrics
    tp = (predictions * labels).sum(axis=0)
    fp = (predictions * (1 - labels)).sum(axis=0)
    fn = ((1 - predictions) * labels).sum(axis=0)
    tn = ((1 - predictions) * (1 - labels)).sum(axis=0)

    precision = np.divide(tp, tp + fp + 1e-8)
    recall = np.divide(tp, tp + fn + 1e-8)
    f1 = np.divide(2 * precision * recall, precision + recall + 1e-8)

    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': float(np.mean((tp + tn) / (tp + tn + fp + fn + 1e-8))),
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'true_positives': tp.tolist(),
        'false_positives': fp.tolist(),
        'false_negatives': fn.tolist(),
        'true_negatives': tn.tolist(),
        'predictions': predictions,
        'labels': labels
    }
    
    return metrics

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    config,
    output_dir,
    early_stopping_enabled: bool = True,
):
    """Train model with optional early stopping and save checkpoints"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = None
    if early_stopping_enabled and val_loader is not None and len(val_loader) > 0:
        early_stopping = TrendBasedEarlyStopping(
            patience=config['training']['patience'],
            window_size=10,
            min_epochs=config['training']['warmup_epochs'],
            min_improvement=0.01
        )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize loss criterion
    pos_weight_cfg = config['training'].get('pos_weight')
    pos_weight = torch.tensor(pos_weight_cfg, dtype=torch.float32).to(device) if pos_weight_cfg is not None else None

    criterion = StableMILLoss(
        pos_weight=pos_weight,
        smooth_factor=config['training']['label_smoothing'],
        focal_gamma=config['training']['focal_gamma']
    )
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training history
    history = {
        'train_f1': [],
        'val_f1': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'trends': []
    }
    
    # Define checkpoint path
    checkpoint_path = output_dir / 'best_model.pt'
    best_model_state = None
    best_metric = -float('inf')
    
    logger.info("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validation phase
        evaluation_cfg = config.get('evaluation', {})
        val_threshold = evaluation_cfg.get('task_a_thresholds', 0.5)
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate(model, val_loader, criterion, device, threshold=val_threshold)
        else:
            val_metrics = None
        
        # Update learning rate
        metric_for_scheduler = val_metrics['f1'] if val_metrics is not None else train_metrics['f1']
        scheduler.step(metric_for_scheduler)
        
        # Store history
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'] if val_metrics is not None else train_metrics['f1'])
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'] if val_metrics is not None else train_metrics['loss'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        logger.info(f"\nEpoch {epoch+1}:")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        if val_metrics is not None:
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        else:
            logger.info("Validation skipped (no validation loader)")
        
        # Check early stopping with trend analysis
        trend_info = {'status': 'disabled'}
        should_stop = False
        metric_for_selection = val_metrics['f1'] if val_metrics is not None else train_metrics['f1']
        if early_stopping is not None:
            should_stop, trend_info = early_stopping(metric_for_selection, epoch)
        history['trends'].append(trend_info)
        
        # Save best model
        if metric_for_selection > best_metric:
            best_metric = metric_for_selection
            logger.info(f"New best model! F1: {metric_for_selection:.4f}")
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics if val_metrics is not None else train_metrics,
                'config': config
            }
            torch.save(best_model_state, checkpoint_path)

            # Print detailed metrics for best model
            print_metrics(val_metrics if val_metrics is not None else train_metrics)
        
        # Check for early stopping
        if should_stop:
            logger.info(f"\nEarly stopping triggered:")
            logger.info(f"Status: {trend_info['status']}")
            logger.info(f"Best F1: {trend_info['best_f1']:.4f} at epoch {trend_info['best_epoch']}")
            break
    
    # Save final training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(_serialize_history(history), f, indent=2)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    return model, history, checkpoint_path

def print_metrics(metrics):
    """Print detailed metrics"""
    logger.info("\nDetailed Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")
    logger.info(f"Loss: {metrics['loss']:.4f}")

def create_cv_splits(site_years):
    """Create blocked cross-validation splits"""
    cv_splits = []
    for test_site in site_years:
        train_sites = [site for site in site_years if site != test_site]
        cv_splits.append((train_sites, test_site))
    return cv_splits

def run_cross_validation(data_dir, site_years, config):
    """Run cross-validation with visualization"""
    device = torch.device(config['training']['device'])
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    cv_splits = create_cv_splits(site_years)
    cv_results = []
    
    logger.info(f"Starting {len(cv_splits)}-fold cross-validation")
    
    for fold, (train_sites, test_site) in enumerate(cv_splits, 1):
        # Initialize wandb run if enabled
        if config['wandb']['enabled'] and WANDB_AVAILABLE:
            import wandb
            wandb.init(
                project=config['wandb']['project'],
                name=f"fold_{fold}_{test_site}",
                config=config
            )
        
        fold_dir = results_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        # Setup data
        val_site = train_sites[-1]
        train_sites = train_sites[:-1]
        
        logger.info(f"Fold {fold}: Training on {train_sites}, validating on {val_site}, testing on {test_site}")
        
        # Create datasets
        train_dataset = LocalizationDataset(data_dir, train_sites, preload_data=False, num_classes=config['model'].get('num_classes', 1))
        val_dataset = LocalizationDataset(data_dir, [val_site], preload_data=False, num_classes=config['model'].get('num_classes', 1))
        test_dataset = LocalizationDataset(data_dir, [test_site], preload_data=False, num_classes=config['model'].get('num_classes', 1))
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        # Initialize model
        model = ImprovedLocalizationMILNet(
            feature_dim=config['model']['feature_dim'],
            num_heads=config['model']['num_heads'],
            num_classes=config['model'].get('num_classes', 1)
        ).to(device)
        
        # Train model
        trained_model, history, checkpoint_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            output_dir=fold_dir
        )
        
        # Initialize loss criterion for evaluation
        criterion = StableMILLoss(
            pos_weight=pos_weight,
            smooth_factor=config['training']['label_smoothing'],
            focal_gamma=config['training']['focal_gamma']
        )
        
        # Evaluate on test set
        evaluation_cfg = config.get('evaluation', {})
        test_threshold = evaluation_cfg.get('task_a_thresholds', 0.5)
        test_metrics = evaluate(trained_model, test_loader, criterion, device, threshold=test_threshold)
        
        # Visualize results
        visualizer = ResultVisualizer(save_dir=fold_dir)
        visualizer.plot_confusion_matrix(test_metrics)
        
        # Store fold results
        fold_results = {
            'fold': fold,
            'test_site': test_site,
            'val_site': val_site,
            'train_sites': train_sites,
            'metrics': test_metrics,
            'history': history
        }
        
        cv_results.append(fold_results)
        
        # Save fold results
        with open(fold_dir / 'fold_results.json', 'w') as f:
            json.dump(fold_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # Log to wandb if enabled
        if config['wandb']['enabled'] and WANDB_AVAILABLE:
            wandb.log({"test_metrics": test_metrics})
            wandb.finish()
    
    # Compute aggregate results
    aggregate_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [fold['metrics'][metric] for fold in cv_results]
        aggregate_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Save final results
    final_results = {
        'folds': cv_results,
        'aggregate': aggregate_metrics
    }
    
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    # Print final results
    logger.info("\nFinal Cross-Validation Results:")
    for metric, stats in aggregate_metrics.items():
        logger.info(f"{metric.capitalize()}:")
        logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return final_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train a whale call detection model with cross-validation')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing processed data')
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save results')
    
    parser.add_argument('--site-years', type=str, nargs='+',
                       help='List of site years to use for cross-validation (default: all available)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Check if wandb is available
    global WANDB_AVAILABLE
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        logger.warning("wandb not installed. Experiment tracking will be disabled.")
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Configuration file {args.config} not found. Using default configuration.")
        config = get_default_config()
    
    # Update config with command line arguments
    config['paths']['data_root'] = args.data_dir
    config['paths']['results_dir'] = args.output_dir
    config['training']['seed'] = args.seed
    
    # Determine site years
    if args.site_years:
        site_years = args.site_years
    else:
        # Get all site years from data directory
        data_dir = Path(args.data_dir)
        site_years = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Using site years: {site_years}")
    
    # Run cross-validation
    try:
        results = run_cross_validation(args.data_dir, site_years, config)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
