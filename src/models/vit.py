import torch
from torch import nn

from src.models.components import Block

class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768, d_ff=3072, num_heads=12, num_layers=12):
        super(ViTEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=16, stride=16)
        self.pos_embedding = nn.Parameter(torch.randn(1, 36, hidden_dim))
        self.layers = nn.ModuleList([
            Block(hidden_dim, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        assert x.dim() == 4, "Input must be a 4D tensor (batch_size, channels, height, width)"
        assert x.size(2) % 16 == 0 and x.size(3) % 16 == 0, "Height and width must be divisible by 16"

        # Apply convolution to get patch embeddings
        x = self.conv(x)  # (batch_size, hidden_dim, height/16, width/16)
        x = x.flatten(2).transpose(1, 2) # Flatten to (batch_size, hidden_dim, num_patches) and transpose to (batch_size, num_patches, hidden_dim)

        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
    
if __name__ == "__main__":
    # Test the ViTEncoder with dummy data
    batch_size = 2
    in_channels = 3
    height = 96
    width = 96
    hidden_dim = 32
    d_ff = 32
    num_heads = 4
    num_layers = 4

    x = torch.randn(batch_size, in_channels, height, width)
    model = ViTEncoder(in_channels, hidden_dim, d_ff, num_heads, num_layers)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)