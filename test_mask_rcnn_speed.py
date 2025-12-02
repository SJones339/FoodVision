#!/usr/bin/env python3
"""Quick test to see if Mask R-CNN can process a single batch"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from detection.train_mask_rcnn import FoodSegDataset, get_model_instance_segmentation, collate_fn
from torch.utils.data import DataLoader
import time

print("Testing Mask R-CNN speed on first batch...")
print("=" * 70)

# Check device
device = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
print(f"Device: {device}")

# Load dataset
data_yaml = 'datasets/foodseg_pp/data.yaml'
print(f"\nLoading dataset from {data_yaml}...")
dataset = FoodSegDataset(data_yaml, split='train')
print(f"Dataset size: {len(dataset)}")

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=1,  # Small batch for testing
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False
)

# Create model
print("\nCreating model...")
model = get_model_instance_segmentation(105)
model.to(device)
model.train()

# Test first batch
print("\nTesting first batch...")
start_time = time.time()

try:
    for i, (images, targets) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"  Images: {len(images)}")
        print(f"  Image shape: {images[0].shape}")
        print(f"  Targets: {len(targets)}")
        
        if len(targets) > 0:
            print(f"  Boxes in first target: {len(targets[0]['boxes'])}")
        
        # Move to device
        print("  Moving to device...")
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        print("  Running forward pass (this might take a while on MPS)...")
        forward_start = time.time()
        loss_dict = model(images, targets)
        forward_time = time.time() - forward_start
        
        losses = sum(loss for loss in loss_dict.values())
        print(f"  Forward pass time: {forward_time:.2f} seconds")
        print(f"  Loss: {losses.item():.4f}")
        
        if i >= 0:  # Just test first batch
            break
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Total test time: {total_time:.2f} seconds")
print(f"\nIf this took > 30 seconds, Mask R-CNN is too slow on MPS.")
print(f"Consider using CPU instead: device='cpu'")


