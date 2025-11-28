"""
Custom CNN Architecture for Food Classification
Trained from scratch (no pretrained weights) to compare with transfer learning approaches.

Architecture Design:
- Multiple convolutional blocks with batch normalization
- Max pooling for downsampling
- Dropout for regularization
- Fully connected layers for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class CustomFoodCNN(nn.Module):
    """
    Custom CNN architecture for food classification.
    Designed from scratch without pretrained weights.
    
    Architecture:
    - Input: 224x224x3 RGB images
    - Conv blocks: Progressive feature extraction
    - Global Average Pooling: Reduces parameters
    - Dropout: Prevents overfitting
    - Fully connected: Final classification
    """
    def __init__(self, num_classes=101, dropout_rate=0.5):
        super().__init__()
        
        # Feature extraction layers
        # Block 1: 224x224 -> 112x112
        self.conv1 = ConvBlock(3, 64, kernel_size=3)
        
        # Block 2: 112x112 -> 56x56
        self.conv2 = ConvBlock(64, 128, kernel_size=3)
        
        # Block 3: 56x56 -> 28x28
        self.conv3 = ConvBlock(128, 256, kernel_size=3)
        
        # Block 4: 28x28 -> 14x14
        self.conv4 = ConvBlock(256, 512, kernel_size=3)
        
        # Block 5: 14x14 -> 7x7
        self.conv5 = ConvBlock(512, 512, kernel_size=3)
        
        # Global Average Pooling (reduces parameters vs flattening)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization (good for ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class CustomFoodCNNV2(nn.Module):
    """
    Enhanced custom CNN with residual connections (simplified ResNet-style).
    More sophisticated architecture for better performance.
    """
    def __init__(self, num_classes=101, dropout_rate=0.5):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual-like blocks
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a layer with multiple conv blocks"""
        layers = []
        layers.append(ConvBlock(in_channels, out_channels))
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_custom_cnn(model_type='v1', num_classes=101, **kwargs):
    """
    Factory function to get custom CNN model.
    
    Args:
        model_type: 'v1' (simple) or 'v2' (enhanced)
        num_classes: Number of food classes
    """
    if model_type == 'v1':
        return CustomFoodCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'v2':
        return CustomFoodCNNV2(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    model = CustomFoodCNN(num_classes=101)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

