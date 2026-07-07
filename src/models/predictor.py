import torch
from torch import nn

from src.models.components import Block, RMSNorm

class ViTPredictor(nn.Module):
    def __init__(self, encoder_dim=768, predictor_dim=384, d_ff=1536, num_heads=6, num_layers=6, max_seq_len=36):
        super(ViTPredictor, self).__init__()
        self.predictor_dim = predictor_dim
        self.proj = nn.Linear(encoder_dim, predictor_dim) # Linear projection to downsample from the wider encoder dimension to the narrower predictor dimension

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim)) # Learnable mask token for the target patches, this serve as a placeholder for the masked patches during prediction

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, predictor_dim)) # Positional embeddings for the predictor

        self.layers = nn.ModuleList([
            Block(predictor_dim, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(predictor_dim)

        # Project predictions back up to the encoder dimension so they can be
        # compared against the (encoder_dim-wide) target encoder representations.
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)

        # I-JEPA initializes the mask token and position embeddings from a
        # truncated normal (std 0.02) rather than large-variance randn.
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, context_tokens, context_indices, target_indices):
        """
        Args:
            context_tokens: Shape (batch_size, num_context_patches, encoder_dim)
            context_indices: Shape (batch_size, num_context_patches)
            target_indices: Shape (batch_size, num_target_patches)
        """
        batch_size = context_tokens.size(0)
        num_targets = target_indices.size(1)

        # Map context down to predictor dimension
        context_tokens = self.proj(context_tokens) 

        """
        The positional embeddings are added to both the context and mask tokens.
        Note that the dimension of the positional embeddings is the same as the predictor dimension, not the encoder dimension.
        This is because the predictor operates in its own embedding space, which is narrower than the encoder space.
        Also equally important is that the mask positions are arbitrary hence we need to apply positional embeddings
        According to its original position in the sequence. This is done by gathering the positional embeddings using the target indices.
        """
        
        # Gather and add positional embeddings for the context tokens
        expanded_ctx_indices = context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        context_pos_emb = torch.gather(
            self.pos_embedding.expand(batch_size, -1, -1), 
            dim=1, 
            index=expanded_ctx_indices
        )
        context_tokens = context_tokens + context_pos_emb

        mask_tokens = self.mask_token.expand(batch_size, num_targets, -1)
        
        # Gather and add positional embeddings for the target masks
        expanded_tgt_indices = target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        target_pos_emb = torch.gather(
            self.pos_embedding.expand(batch_size, -1, -1), 
            dim=1, 
            index=expanded_tgt_indices
        )
        mask_tokens = mask_tokens + target_pos_emb

        """
        While they are appended at the end of the sequence, they still inhibit the positional information of their original locations in the sequence. 
        This is crucial for the model to learn the correct relationships between context and target patches.
        Also since the model is using full self-attention and not causal attention, the model can attend to all context tokens when predicting the masked tokens.
        Hence the order of the tokens in the sequence does not matter as long as the positional embeddings are correctly applied.
        """
        x = torch.cat([context_tokens, mask_tokens], dim=1) # Shape: (batch_size, num_context + num_targets, predictor_dim)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Extract only the predictions, which are appended at the end of the sequence
        predictions = x[:, -num_targets:, :]

        # Map back up to encoder_dim to match the target representations
        predictions = self.predictor_proj(predictions)

        return predictions
    
if __name__ == "__main__":
    # Sanity-check the ViTPredictor with a coherent I-JEPA masking split.
    
    """
    The 36 patches are partitioned (per sample) into a disjoint context set
    and target set via a random permutation, so no patch is both context and
    target. The predictor consumes the context encoder's output and predicts
    the target-patch representations from learnable mask tokens.
    """
    
    batch_size = 2
    num_patches = 36
    encoder_dim = 32   # must match the encoder's d_model
    predictor_dim = 16 # narrower than the encoder, as in I-JEPA
    d_ff = 4 * predictor_dim
    num_heads = 4
    num_layers = 4

    num_context = 18
    num_target = 8

    # Disjoint context / target indices per sample.
    perms = torch.stack([torch.randperm(num_patches) for _ in range(batch_size)])
    context_indices = perms[:, :num_context]
    target_indices = perms[:, num_context:num_context + num_target]

    # Stand-in for the context encoder's output (encoder_dim wide).
    context_tokens = torch.randn(batch_size, num_context, encoder_dim)

    model = ViTPredictor(encoder_dim=encoder_dim, predictor_dim=predictor_dim,
                         d_ff=d_ff, num_heads=num_heads, num_layers=num_layers,
                         max_seq_len=num_patches)
    predictions = model(context_tokens, context_indices, target_indices)

    print("Context tokens shape:", tuple(context_tokens.shape))
    print("Context / target:    ", num_context, "/", num_target)
    print("Predictions shape:   ", tuple(predictions.shape))  # (B, num_target, encoder_dim)