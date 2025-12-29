#!/usr/bin/env python
"""Command-line training script for the discriminator model.

Usage:
    python train.py --data /path/to/dataset.npz --output ./checkpoints
    python train.py --data-dir /path/to/datasets --output ./checkpoints
    
Examples:
    # Train with a single dataset file (auto-split into train/val)
    python train.py --data dataset.npz --val-split 0.2
    
    # Train with separate train/val files
    python train.py --train train.npz --val val.npz
    
    # Train from a directory with train.npz and val.npz
    python train.py --data-dir ./my_datasets/
    
    # Custom hyperparameters
    python train.py --data dataset.npz --hidden-dim 256 --num-layers 3 --lr 0.0005
"""

import argparse
import json
from pathlib import Path
import sys

import torch

# Add parent directory to path for imports when running as package
# But also support running directly from model/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Try package import first, fall back to local imports
try:
    from model import (
        SiameseLSTMDiscriminator,
        TrajectoryPairDataset,
        load_dataset_from_directory,
        Trainer,
        TrainingConfig
    )
    from model.dataset import create_train_val_split, create_data_loaders
except ImportError:
    # Running from within model/ directory
    from model import SiameseLSTMDiscriminator
    from dataset import (
        TrajectoryPairDataset,
        load_dataset_from_directory,
        create_train_val_split,
        create_data_loaders
    )
    from trainer import Trainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the discriminator model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", type=str, 
                           help="Path to single .npz dataset file (will auto-split)")
    data_group.add_argument("--data-dir", type=str,
                           help="Path to directory with train.npz, val.npz")
    data_group.add_argument("--train", type=str,
                           help="Path to training .npz file")
    data_group.add_argument("--val", type=str,
                           help="Path to validation .npz file")
    data_group.add_argument("--val-split", type=float, default=0.2,
                           help="Validation split ratio (default: 0.2)")
    
    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--hidden-dim", type=int, default=128,
                            help="LSTM hidden dimension (default: 128)")
    model_group.add_argument("--num-layers", type=int, default=2,
                            help="Number of LSTM layers (default: 2)")
    model_group.add_argument("--dropout", type=float, default=0.2,
                            help="Dropout probability (default: 0.2)")
    model_group.add_argument("--no-bidirectional", action="store_true",
                            help="Use unidirectional LSTM")
    model_group.add_argument("--classifier-dims", type=str, default="128,64",
                            help="Classifier hidden dims (comma-separated, default: 128,64)")
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100,
                            help="Number of training epochs (default: 100)")
    train_group.add_argument("--batch-size", type=int, default=32,
                            help="Batch size (default: 32)")
    train_group.add_argument("--lr", type=float, default=1e-3,
                            help="Learning rate (default: 0.001)")
    train_group.add_argument("--weight-decay", type=float, default=1e-4,
                            help="Weight decay (default: 0.0001)")
    train_group.add_argument("--early-stopping", type=int, default=10,
                            help="Early stopping patience (default: 10)")
    train_group.add_argument("--scheduler", type=str, default="plateau",
                            choices=["plateau", "cosine", "none"],
                            help="LR scheduler (default: plateau)")
    
    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", type=str, default="./checkpoints",
                             help="Output directory for checkpoints (default: ./checkpoints)")
    output_group.add_argument("--experiment-name", type=str,
                             help="Name for this experiment")
    output_group.add_argument("--save-all", action="store_true",
                             help="Save checkpoint at every epoch")
    
    # Other
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--device", type=str, default="auto",
                            choices=["auto", "cuda", "cpu"],
                            help="Device to use (default: auto)")
    other_group.add_argument("--num-workers", type=int, default=0,
                            help="DataLoader workers (default: 0)")
    other_group.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    other_group.add_argument("--quiet", action="store_true",
                            help="Suppress verbose output")
    
    return parser.parse_args()


def load_data(args):
    """Load training and validation datasets based on arguments."""
    
    if args.data_dir:
        # Load from directory with train.npz, val.npz
        datasets = load_dataset_from_directory(args.data_dir)
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        
    elif args.train and args.val:
        # Separate train and val files
        train_dataset = TrajectoryPairDataset(args.train)
        val_dataset = TrajectoryPairDataset(args.val)
        
    elif args.data:
        # Single file with auto-split
        full_dataset = TrajectoryPairDataset(args.data)
        train_dataset, val_dataset = create_train_val_split(
            full_dataset, 
            val_ratio=args.val_split,
            seed=args.seed
        )
        
    else:
        raise ValueError("Must provide --data, --data-dir, or both --train and --val")
        
    return train_dataset, val_dataset


def main():
    args = parse_args()
    
    # Load data
    if not args.quiet:
        print("Loading data...")
    train_dataset, val_dataset = load_data(args)
    
    if not args.quiet:
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Parse classifier dims
    classifier_dims = tuple(int(x) for x in args.classifier_dims.split(","))
    
    # Create model
    model = SiameseLSTMDiscriminator(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=not args.no_bidirectional,
        classifier_hidden_dims=classifier_dims
    )
    
    if not args.quiet:
        print(f"\nModel architecture:")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Num layers: {args.num_layers}")
        print(f"  Bidirectional: {not args.no_bidirectional}")
        print(f"  Classifier: {classifier_dims}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        
    # Create training config
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=not args.no_bidirectional,
        classifier_hidden_dims=classifier_dims,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        scheduler=args.scheduler,
        checkpoint_dir=args.output,
        save_best_only=not args.save_all,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=args.experiment_name
    )
    
    # Train
    history = trainer.train(verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
        print("  - best.pt: Best model checkpoint")
        print("  - latest.pt: Most recent checkpoint")
        print("  - config.json: Training configuration")
        print("  - history.json: Training history")
        

if __name__ == "__main__":
    main()
