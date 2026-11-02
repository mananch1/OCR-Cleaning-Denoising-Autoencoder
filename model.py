"""
Convolutional Denoising Autoencoder (CDAE) for Document Image Restoration
Deep Learning Mini-Project
"""

import torch
import torch.nn as nn


class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER ---
        # 2 pooling stages (4x downsampling) to preserve text stroke fidelity
        self.encoder = nn.Sequential(
            # stage 1: 1 -> 32 channels, H/2 x W/2
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2: 32 -> 64 channels, H/4 x W/4 (spatial bottleneck)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- DECODER ---
        # nearest-neighbor upsample + conv to avoid checkerboard artifacts
        self.decoder = nn.Sequential(
            # stage 1: H/4 x W/4 -> H/2 x W/2
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # stage 2: H/2 x W/2 -> H x W
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # bounds output pixels to [0.0, 1.0]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


if __name__ == "__main__":
    model = DenoisingAutoEncoder()
    x = torch.randn(1, 1, 420, 540)
    out = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
