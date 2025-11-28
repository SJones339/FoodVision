"""
Training script for custom CNN (from scratch) vs EfficientNet (transfer learning).
This script trains both models and compares their performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Handle both relative and absolute imports
try:
    from .custom_cnn import get_custom_cnn
except (ImportError, ValueError):
    # If running as script directly or if relative import fails
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from custom_cnn import get_custom_cnn
from datasets import load_dataset
from PIL import Image


class Food101Dataset(torch.utils.data.Dataset):
    """PyTorch dataset for Food-101"""
    def __init__(self, split='train', transform=None, max_samples=None):
        self.ds = load_dataset("ethz/food101", split=split)
        self.transform = transform
        self.max_samples = max_samples
        
        # Get class names
        if split == 'train':
            self.class_names = self.ds.features['label'].names
        else:
            # For validation, use same class names
            train_ds = load_dataset("ethz/food101", split='train')
            self.class_names = train_ds.features['label'].names
    
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.ds))
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image']
        label = item['label']
        
        # Convert to RGB if grayscale (some images might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(batch_size=32, num_workers=4, max_train_samples=None):
    """Create data loaders for training and validation"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Food101Dataset('train', transform=train_transform, max_samples=max_train_samples)
    val_dataset = Food101Dataset('validation', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 0 to avoid multiprocessing issues on some systems
        pin_memory=False  # Disable if not using GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset.class_names


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_custom_cnn(
    model_type='v1',
    num_classes=101,
    epochs=30,
    batch_size=32,
    lr=0.001,
    device='cuda',
    save_dir='models/',
    max_train_samples=None  # Limit training samples for faster training
):
    """
    Train custom CNN from scratch.
    
    Returns:
        Dictionary with training history and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Custom CNN ({model_type}) from Scratch")
    print(f"{'='*60}\n")
    
    # Create model
    model = get_custom_cnn(model_type=model_type, num_classes=num_classes)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Get data loaders
    train_loader, val_loader, class_names = get_data_loaders(
        batch_size=batch_size, 
        max_train_samples=max_train_samples
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(save_dir) / f'custom_cnn_{model_type}_best.pth'
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'class_names': class_names
            }, save_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    # Final metrics
    metrics = {
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'training_time': training_time,
        'num_parameters': num_params,
        'epochs_trained': epochs
    }
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    return history, metrics


def plot_training_comparison(history_custom, history_efficientnet, save_path='training_comparison.png'):
    """Plot comparison between custom CNN and EfficientNet"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history_custom['train_loss']) + 1)
    
    # Loss comparison
    axes[0, 0].plot(epochs, history_custom['train_loss'], label='Custom CNN (Train)', linestyle='--')
    axes[0, 0].plot(epochs, history_custom['val_loss'], label='Custom CNN (Val)')
    if history_efficientnet:
        axes[0, 0].plot(epochs, history_efficientnet['train_loss'], label='EfficientNet (Train)', linestyle='--')
        axes[0, 0].plot(epochs, history_efficientnet['val_loss'], label='EfficientNet (Val)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs, history_custom['train_acc'], label='Custom CNN (Train)', linestyle='--')
    axes[0, 1].plot(epochs, history_custom['val_acc'], label='Custom CNN (Val)')
    if history_efficientnet:
        axes[0, 1].plot(epochs, history_efficientnet['train_acc'], label='EfficientNet (Train)', linestyle='--')
        axes[0, 1].plot(epochs, history_efficientnet['val_acc'], label='EfficientNet (Val)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(epochs, history_custom['learning_rates'], label='Custom CNN')
    if history_efficientnet:
        axes[1, 0].plot(epochs, history_efficientnet['learning_rates'], label='EfficientNet')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Final comparison bar chart
    if history_efficientnet:
        models = ['Custom CNN', 'EfficientNet']
        train_accs = [history_custom['train_acc'][-1], history_efficientnet['train_acc'][-1]]
        val_accs = [history_custom['val_acc'][-1], history_efficientnet['val_acc'][-1]]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
        axes[1, 1].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8)
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Final Accuracy Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom CNN from scratch')
    parser.add_argument('--model-type', choices=['v1', 'v2'], default='v1',
                       help='CNN architecture version')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit training samples for faster training')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Train custom CNN
    history, metrics = train_custom_cnn(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_train_samples=args.max_samples
    )
    
    # Save metrics
    metrics_path = Path('models') / f'custom_cnn_{args.model_type}_metrics.json'
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")

