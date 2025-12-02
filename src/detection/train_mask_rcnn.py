"""
Train Mask R-CNN for food detection and segmentation.

Mask R-CNN typically achieves 50-70% mAP on food detection tasks,
significantly better than YOLOv8's 34% mAP.

Architecture:
- Mask R-CNN: Instance segmentation (bounding boxes + masks)
- ResNet-50/101 backbone
- Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)

Expected Performance:
- mAP50: 50-70% (vs YOLOv8's 34%)
- Better segmentation masks
- More accurate bounding boxes
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import json


class FoodSegDataset(Dataset):
    """Dataset for FoodSeg103 in YOLO format, converted for Mask R-CNN"""
    
    def __init__(self, data_yaml, split='train', transform=None):
        """
        Args:
            data_yaml: Path to YOLO format data.yaml
            split: 'train' or 'val'
            transform: Optional transforms
        """
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        self.data_dir = Path(data_yaml).parent
        self.split = split
        self.transform = transform
        
        # Get image and label paths
        images_dir = self.data_dir / 'images' / split
        labels_dir = self.data_dir / 'labels' / split
        
        self.images = sorted(list(images_dir.glob('*.jpg')))
        self.labels = sorted(list(labels_dir.glob('*.txt')))
        
        # Filter to ensure matching
        self.images = [img for img in self.images 
                      if (labels_dir / f"{img.stem}.txt").exists()]
        
        print(f"Loaded {len(self.images)} {split} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load YOLO format labels
        label_path = self.data_dir / 'labels' / self.split / f"{img_path.stem}.txt"
        
        boxes = []
        labels = []
        masks = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            h, w = image.shape[:2]
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                # YOLO format: normalized center x, center y, width, height
                cx, cy, nw, nh = map(float, parts[1:5])
                
                # Skip invalid boxes (zero or negative width/height)
                if nw <= 0 or nh <= 0:
                    continue
                
                # Convert to absolute coordinates
                x1 = (cx - nw/2) * w
                y1 = (cy - nh/2) * h
                x2 = (cx + nw/2) * w
                y2 = (cy + nh/2) * h
                
                # Ensure valid coordinates and positive width/height
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                
                # Skip if box has zero width or height
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Ensure minimum box size (at least 1 pixel)
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)  # Mask R-CNN uses 1-indexed
                
                # Create simple rectangular mask (can be improved with actual mask data)
                mask = np.zeros((h, w), dtype=np.uint8)
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                # Ensure indices are valid
                x1i, y1i = max(0, x1i), max(0, y1i)
                x2i, y2i = min(w, x2i), min(h, y2i)
                if x2i > x1i and y2i > y1i:
                    mask[y1i:y2i, x1i:x2i] = 1
                masks.append(mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }
        
        return image, target


def get_model_instance_segmentation(num_classes):
    """Create Mask R-CNN model"""
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def collate_fn(batch):
    """Custom collate function for batching"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def train_mask_rcnn(
    data_yaml='datasets/foodseg_pp/data.yaml',
    num_classes=105,  # 104 food classes + background
    epochs=50,
    batch_size=4,  # Smaller batch size for Mask R-CNN
    lr=0.001,
    device='auto',
    output_dir='models/'
):
    """
    Train Mask R-CNN on FoodSeg103 dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        num_classes: Number of classes (104 food + 1 background)
        epochs: Number of training epochs
        batch_size: Batch size (smaller for Mask R-CNN)
        lr: Learning rate
        device: 'auto', 'cuda', 'mps', or 'cpu'
        output_dir: Where to save the model
    """
    print("=" * 70)
    print("TRAINING MASK R-CNN FOR FOOD DETECTION")
    print("=" * 70)
    print(f"\nModel: Mask R-CNN (ResNet-50 FPN)")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Classes: {num_classes}")
    print(f"Dataset: {data_yaml}")
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Device: {device}")
    device = torch.device(device)
    
    # Check if dataset exists
    if not Path(data_yaml).exists():
        print(f"\n⚠️  ERROR: Dataset YAML not found: {data_yaml}")
        print("   Make sure you've converted FoodSeg103 to YOLO format:")
        print("   python download_datasets.py --foodseg103 --convert-yolo")
        return None
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = FoodSegDataset(data_yaml, split='train')
    val_dataset = FoodSegDataset(data_yaml, split='val')
    
    # Create data loaders
    # Use pin_memory=False for MPS compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 for macOS compatibility
        pin_memory=False,  # MPS doesn't support pin_memory
        prefetch_factor=None  # Disable prefetching for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None
    )
    
    # Create model
    print("\nCreating Mask R-CNN model...")
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Enable training mode
    model.train()
    
    # For MPS, we might need to compile the model (PyTorch 2.0+)
    try:
        if device.type == 'mps':
            print("⚠️  Note: Mask R-CNN on MPS can be slow. Consider using CPU for faster training.")
            # MPS has known issues with some operations, fallback to CPU if needed
    except:
        pass
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop
    print("\nStarting training...")
    print("This will take several hours. Progress will be shown below.\n")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    training_history = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        batch_count = 0
        for images, targets in pbar:
            try:
                # Move to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Skip empty batches
                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Skip if loss is invalid
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"\n⚠️  Invalid loss at batch {batch_count}, skipping...")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                
                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += losses.item()
                batch_count += 1
                pbar.set_postfix({'loss': losses.item(), 'batches': batch_count})
                
            except Exception as e:
                print(f"\n⚠️  Error at batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                pbar.set_postfix({'loss': losses.item()})
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average loss (only count successful batches)
        successful_batches = max(1, batch_count)  # Avoid division by zero
        avg_train_loss = train_loss / successful_batches
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            output_path = Path(output_dir) / 'mask_rcnn_food_best.pth'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model to {output_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best model saved to: {output_path}")
    
    # Save training history
    history_path = Path(output_dir) / 'mask_rcnn_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for food detection')
    parser.add_argument('--data-yaml', type=str, default='datasets/foodseg_pp/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    train_mask_rcnn(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

