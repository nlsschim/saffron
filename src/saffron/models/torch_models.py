import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class MicrogliaCNN(torch.nn.Module):
    """
    A simple neural network architecture for microglia classification.
    Handles 512x512 grayscale images.
    """

    def __init__(self, input_size=512, num_classes=6):
        super(MicrogliaCNN, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=12, stride=2, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=12, stride=2, padding=2)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=128, kernel_size=12, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=12, stride=2, padding=2)
        )
        
        # Calculate flattened size dynamically
        self.flattened_size = self._get_flattened_size(input_size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
    
    def _get_flattened_size(self, input_size):
        """Calculate the size after convolutions and pooling."""
        # Simulate forward pass with dummy input
        dummy_input = torch.zeros(1, 1, input_size, input_size)
        x = self.cnn1(dummy_input)
        x = self.cnn2(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc1(out)

        return out


class BackboneEncoder(ABC):
    """Abstract base class for different backbone encoders."""
    
    @abstractmethod
    def __init__(self, input_channels: int = 1, pretrained: bool = True):
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        pass


class ResNetEncoder(BackboneEncoder):
    """ResNet backbone encoder."""
    
    def __init__(self, input_channels: int = 1, pretrained: bool = True, 
                 resnet_type: str = "resnet18"):
        self.resnet_type = resnet_type
        
        # Load pretrained ResNet
        if resnet_type == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.output_dim = 512
        elif resnet_type == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.output_dim = 512
        elif resnet_type == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
    
    def get_output_dim(self) -> int:
        return self.output_dim


class EfficientNetEncoder(BackboneEncoder):
    """EfficientNet backbone encoder."""
    
    def __init__(self, input_channels: int = 1, pretrained: bool = True,
                 efficientnet_type: str = "efficientnet_b0"):
        try:
            from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
        except ImportError:
            raise ImportError("EfficientNet requires torchvision >= 0.11.0")
        
        self.efficientnet_type = efficientnet_type
        
        # Load pretrained EfficientNet
        if efficientnet_type == "efficientnet_b0":
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.output_dim = 1280
        elif efficientnet_type == "efficientnet_b1":
            self.backbone = efficientnet_b1(pretrained=pretrained)
            self.output_dim = 1280
        elif efficientnet_type == "efficientnet_b2":
            self.backbone = efficientnet_b2(pretrained=pretrained)
            self.output_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet type: {efficientnet_type}")
        
        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
        
        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def get_output_dim(self) -> int:
        return self.output_dim


class ViTEncoder(BackboneEncoder):
    """Vision Transformer backbone encoder."""
    
    def __init__(self, input_channels: int = 1, pretrained: bool = True,
                 vit_type: str = "vit_b_16"):
        try:
            from torchvision.models import vit_b_16, vit_b_32, vit_l_16
        except ImportError:
            raise ImportError("Vision Transformer requires torchvision >= 0.13.0")
        
        self.vit_type = vit_type
        
        # Load pretrained ViT
        if vit_type == "vit_b_16":
            self.backbone = vit_b_16(pretrained=pretrained)
            self.output_dim = 768
        elif vit_type == "vit_b_32":
            self.backbone = vit_b_32(pretrained=pretrained)
            self.output_dim = 768
        elif vit_type == "vit_l_16":
            self.backbone = vit_l_16(pretrained=pretrained)
            self.output_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT type: {vit_type}")
        
        # Modify patch embedding for single channel input if needed
        if input_channels != 3:
            original_conv = self.backbone.conv_proj
            self.backbone.conv_proj = nn.Conv2d(
                input_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding
            )
        
        # Remove the final classification head
        self.backbone.heads = nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def get_output_dim(self) -> int:
        return self.output_dim


class SimpleConvEncoder(BackboneEncoder):
    """Simple convolutional encoder for testing/baseline."""
    
    def __init__(self, input_channels: int = 1, pretrained: bool = True):
        # pretrained parameter ignored for custom architecture
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.output_dim = 256
    
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def get_output_dim(self) -> int:
        return self.output_dim


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class ContrastiveModel(nn.Module):
    """
    Main contrastive learning model with interchangeable backbones.
    Handles both masked image and patch encoding for contrastive learning.
    """
    
    def __init__(self,
                 backbone_type: str = "resnet18",
                 input_channels: int = 1,
                 pretrained: bool = True,
                 projection_dim: int = 128,
                 backbone_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.input_channels = input_channels
        self.projection_dim = projection_dim
        
        if backbone_kwargs is None:
            backbone_kwargs = {}
        
        # Initialize backbone encoder
        self.encoder = self._create_backbone(backbone_type, input_channels, pretrained, **backbone_kwargs)
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.get_output_dim(),
            output_dim=projection_dim
        )
        
        logger.info(f"Initialized ContrastiveModel with {backbone_type} backbone")
        logger.info(f"Encoder output dim: {self.encoder.get_output_dim()}")
        logger.info(f"Projection dim: {projection_dim}")

    def _create_backbone(self, backbone_type: str, input_channels: int, 
                        pretrained: bool, **kwargs) -> BackboneEncoder:
        """Factory method to create backbone encoders."""
        
        if backbone_type.startswith("resnet"):
            return ResNetEncoder(input_channels, pretrained, backbone_type, **kwargs)
        elif backbone_type.startswith("efficientnet"):
            return EfficientNetEncoder(input_channels, pretrained, backbone_type, **kwargs)
        elif backbone_type.startswith("vit"):
            return ViTEncoder(input_channels, pretrained, backbone_type, **kwargs)
        elif backbone_type == "simple_conv":
            return SimpleConvEncoder(input_channels, pretrained, **kwargs)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def encode(self, x: Tensor) -> Tensor:
        """Encode input through backbone only (no projection)."""
        return self.encoder.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass through encoder and projection head."""
        features = self.encode(x)
        projections = self.projection_head(features)
        return projections
    
    def encode_masked_image_and_patch(self, masked_image: Tensor, patch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode both masked image and patch for contrastive learning.
        
        Args:
            masked_image: Batch of masked images [B, C, H, W]
            patch: Batch of patches [B, C, patch_H, patch_W]
            
        Returns:
            Tuple of (masked_image_features, patch_features)
        """
        # Encode masked image
        masked_features = self.forward(masked_image)
        
        # Encode patch
        patch_features = self.forward(patch)
        
        return masked_features, patch_features
    
    def compute_similarity(self, features1: Tensor, features2: Tensor) -> Tensor:
        """Compute cosine similarity between feature vectors."""
        return F.cosine_similarity(features1, features2, dim=1)
    
    def get_backbone_name(self) -> str:
        """Get the name of the current backbone."""
        return self.backbone_type
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for different parts of the model."""
        encoder_params = sum(p.numel() for p in self.encoder.backbone.parameters())
        projection_params = sum(p.numel() for p in self.projection_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'encoder': encoder_params,
            'projection_head': projection_params,
            'total': total_params
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.encoder.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.encoder.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters unfrozen")


def create_contrastive_model(backbone_type: str = "resnet18", **kwargs) -> ContrastiveModel:
    """
    Convenience function to create a ContrastiveModel with specified backbone.
    
    Args:
        backbone_type: Type of backbone ('resnet18', 'resnet50', 'efficientnet_b0', 
                      'vit_b_16', 'simple_conv', etc.)
        **kwargs: Additional arguments passed to ContrastiveModel
        
    Returns:
        ContrastiveModel instance
    """
    return ContrastiveModel(backbone_type=backbone_type, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Deep Learning Model module loaded successfully!")
    
    # Test different backbones
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example: Create models with different backbones
    available_backbones = ["resnet18", "simple_conv"]  # Add others as needed
    
    for backbone in available_backbones:
        try:
            model = create_contrastive_model(backbone_type=backbone, input_channels=1)
            print(f"\n{backbone} model created successfully!")
            print(f"Parameters: {model.get_num_parameters()}")

            # Test forward pass
            dummy_input = torch.randn(2, 1, 64, 64)  # Batch of 2, single channel, 64x64
            with torch.no_grad():
                features = model.encode(dummy_input)
                projections = model(dummy_input)
                print(f"Input shape: {dummy_input.shape}")
                print(f"Feature shape: {features.shape}")
                print(f"Projection shape: {projections.shape}")

        except Exception as e:
            print(f"Error with {backbone}: {e}")

    print("\nModel testing completed!")
