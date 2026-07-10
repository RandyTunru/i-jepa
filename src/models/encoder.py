import torch
from torch import nn

from src.models.components import Block

class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, d_model=768, d_ff=3072, num_heads=12, num_layers=12, max_seq_len=36):
        super(ViTEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=16, stride=16)
        # The actual I-JEPA uses the sinusoidal positional embeddings. For simplicity, i chose to use learnable positional embeddings instead.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            Block(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        # Small-variance init (std 0.02) keeps positional signal from
        # overwhelming the patch embeddings at the start of training.
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x, keep_indices=None):
        """
        Args:
            x: (batch_size, channels, height, width)
            keep_indices: (batch_size, num_kept_patches) or None
        Returns:
            tokens: (batch_size, num_kept_patches, d_model)
        """
        assert x.dim() == 4, "Input must be a 4D tensor (batch_size, channels, height, width)"
        assert x.size(2) % 16 == 0 and x.size(3) % 16 == 0, "Height and width must be divisible by 16"
        assert keep_indices.size(0) == x.size(0), "keep_indices batch size must match input batch size"
        assert keep_indices.size(1) <= (x.size(2) // 16) * (x.size(3) // 16), "keep_indices length must not exceed number of patches"

        # Apply convolution to get patch embeddings
        x = self.conv(x)  # (batch_size, d_model, height/16, width/16)
        x = x.flatten(2).transpose(1, 2) # Flatten to (batch_size, d_model, num_patches) and transpose to (batch_size, num_patches, d_model)

        x = x + self.pos_embedding[:, :x.size(1), :]

        if keep_indices is not None:
            # keep_indices shape: (batch_size, num_kept_patches)
            # torch.gather to pluck out only the tokens we want to keep
            expanded_indices = keep_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, dim=1, index=expanded_indices)

        for layer in self.layers:
            x = layer(x)

        return x
    
if __name__ == "__main__":
    # Sanity-check the ViTEncoder the way I-JEPA actually uses it.
    
    """
     A 96x96 image with a 16x16 patch conv gives a 6x6 = 36 patch grid, which
    matches max_seq_len. The same encoder architecture is used two ways:
      1. Target encoder: sees the full image (keep_indices=None).
      2. Context encoder: sees only the sampled context patches.
    """

    batch_size = 2
    in_channels = 3
    height = width = 96
    num_patches = (height // 16) * (width // 16)  # 36

    d_model = 32
    d_ff = 4 * d_model
    num_heads = 4
    num_layers = 4

    x = torch.randn(batch_size, in_channels, height, width)
    encoder = ViTEncoder(in_channels=in_channels, d_model=d_model, d_ff=d_ff,
                         num_heads=num_heads, num_layers=num_layers,
                         max_seq_len=num_patches)

    """
    This ensures that the patches we are predicting for still attended to the other patches in the image.
    Since the target encoder is used to generate the "ground truth" representations for the target patches.
    After the target encoder pass, we will extract only the patches that were masked out and use them as the target representations for the predictor to learn to predict.
    """
    # Target encoder pass: full sequence, no patches dropped.
    full_output = encoder(x)
    print("Input shape:          ", tuple(x.shape))
    print("Target (full) output: ", tuple(full_output.shape))  # (B, 36, d_model)

    # Context encoder pass: keep a unique subset of patches per sample.
    num_context = num_patches // 2  # 18
    keep_indices = torch.stack([
        torch.randperm(num_patches)[:num_context] for _ in range(batch_size)
    ])
    context_output = encoder(x, keep_indices=keep_indices)
    print("Kept patches / sample:", num_context)
    print("Context output:       ", tuple(context_output.shape))  # (B, 18, d_model)