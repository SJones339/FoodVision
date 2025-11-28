#!/usr/bin/env python3
"""
Quick training script for both Custom CNN and EfficientNet.
Run this to train both models and compare results.

Usage:
    python quick_train.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--device DEVICE]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import argparse
import json

# Import our modules
from classification.custom_cnn import get_custom_cnn
from classification.train_custom_cnn import Food101Dataset, train_epoch, validate

def train_custom_cnn_quick(epochs=20, batch_size=32, device='auto', max_samples=None):
    """Quick training of custom CNN"""
    print("\n" + "="*60)
    print("TRAINING CUSTOM CNN (FROM SCRATCH)")
    print("="*60)
    
    # Create model with higher dropout for regularization
    model = get_custom_cnn(model_type='v1', num_classes=101, dropout_rate=0.7)  # Increased dropout
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    if max_samples:
        print(f"⚠️  Using only {max_samples} samples - may cause overfitting!")
        print("   Consider removing --max-samples for better results")
    
    # Data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Food101Dataset('train', transform=train_transform, max_samples=max_samples)
    val_dataset = Food101Dataset('validation', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 0 to avoid multiprocessing issues
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/custom_cnn_best.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("CUSTOM CNN TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    return history, {'best_val_acc': best_val_acc, 'training_time': training_time}


def train_efficientnet_quick(epochs=20, batch_size=32, device='auto', max_samples=None):
    """Quick training of EfficientNet with transfer learning"""
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNET (TRANSFER LEARNING)")
    print("="*60)
    
    try:
        from efficientnet_pytorch import EfficientNet
    except ImportError:
        print("Installing efficientnet-pytorch...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet-pytorch"])
        from efficientnet_pytorch import EfficientNet
    
    # Create model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 101)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Data loaders (same as custom CNN)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Food101Dataset('train', transform=train_transform, max_samples=max_samples)
    val_dataset = Food101Dataset('validation', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 0 to avoid multiprocessing issues
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/efficientnet_best.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("EFFICIENTNET TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    return history, {'best_val_acc': best_val_acc, 'training_time': training_time}


def main():
    parser = argparse.ArgumentParser(description='Quick training of both models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device (cuda/mps/cpu/auto)')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples (for speed). WARNING: May cause overfitting if too low!')
    parser.add_argument('--custom-only', action='store_true', help='Train only custom CNN')
    parser.add_argument('--efficientnet-only', action='store_true', help='Train only EfficientNet')
    
    args = parser.parse_args()
    
    # Auto-detect best device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == 'mps':
        print("✓ Using Apple Silicon GPU (MPS) for acceleration!")
    elif device == 'cpu':
        print("⚠️  Using CPU - training will be slower")
    
    results = {}
    
    # Train Custom CNN
    if not args.efficientnet_only:
        history_custom, metrics_custom = train_custom_cnn_quick(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_samples=args.max_samples
        )
        results['custom_cnn'] = {'history': history_custom, 'metrics': metrics_custom}
    
    # Train EfficientNet
    if not args.custom_only:
        history_eff, metrics_eff = train_efficientnet_quick(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_samples=args.max_samples
        )
        results['efficientnet'] = {'history': history_eff, 'metrics': metrics_eff}
    
    # Print comparison
    if 'custom_cnn' in results and 'efficientnet' in results:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Custom CNN - Best Val Acc: {results['custom_cnn']['metrics']['best_val_acc']:.2f}%")
        print(f"EfficientNet - Best Val Acc: {results['efficientnet']['metrics']['best_val_acc']:.2f}%")
        print(f"\nCustom CNN - Training Time: {results['custom_cnn']['metrics']['training_time']/60:.2f} min")
        print(f"EfficientNet - Training Time: {results['efficientnet']['metrics']['training_time']/60:.2f} min")
        improvement = results['efficientnet']['metrics']['best_val_acc'] - results['custom_cnn']['metrics']['best_val_acc']
        print(f"\nAccuracy Improvement (Transfer Learning): {improvement:.2f}%")
        print("="*60)
    
    # Save results
    Path('models').mkdir(exist_ok=True)
    with open('models/training_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            json_results[key] = {
                'metrics': value['metrics'],
                'final_train_acc': value['history']['train_acc'][-1],
                'final_val_acc': value['history']['val_acc'][-1]
            }
        json.dump(json_results, f, indent=2)
    
    # Save full training history for visualization
    with open('models/training_history.json', 'w') as f:
        history_data = {}
        for key, value in results.items():
            history_data[key] = value['history']
        json.dump(history_data, f, indent=2)
    
    print(f"\nResults saved to models/training_results.json")
    print(f"Training history saved to models/training_history.json")


if __name__ == "__main__":
    main()

