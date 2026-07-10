import copy

import torch
from torch import nn
import torch.nn.functional as F

from src.models.modules.encoder import ViTEncoder
from src.models.modules.predictor import ViTPredictor


class IJEPA(nn.Module):
    def __init__(self, encoder_kwargs, predictor_kwargs):
        super().__init__()
        self.context_encoder = ViTEncoder(**encoder_kwargs)
        self.predictor = ViTPredictor(**predictor_kwargs)

        # Target encoder starts as an exact copy of the context encoder and is updated only through EMA
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            # Disable grad so it is excluded from the optimizer and never receives backprop updates.
            param.requires_grad = False

    def trainable_parameters(self):
        """Parameters the optimizer should own (excludes the EMA target)."""
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())

    def train(self, mode=True):
        """
        Keep the target encoder in eval mode even while training.
        The purpose of this is to ensure that the target encoder's batch norm (if any) and dropout (if any) behave consistently during training.
        We can't make let the target encoder have an active dropout because the target encoder is used to generate the targets for the predictor, and we want those targets to be deterministic and not stochastic.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, imgs, context_indices, target_indices):
        """
        Args:
            imgs: (batch_size, channels, height, width)
            context_indices: (batch_size, num_context)
            target_indices:  (batch_size, num_blocks, block_size)
        Returns:
            predictions: (batch_size, num_blocks, block_size, encoder_dim)
            targets:     (batch_size, num_blocks, block_size, encoder_dim)  (stop-grad)
        """
        batch_size, num_blocks, block_size = target_indices.shape

        # Context branch: gradients flow through the context encoder.
        context_repr = self.context_encoder(imgs, keep_indices=context_indices)

        # Target branch: full image through the EMA encoder, no gradients.
        with torch.no_grad():
            target_full = self.target_encoder(imgs)  # (B, num_patches, encoder_dim)

            # I-JEPA layer-norms the target representations before the loss making the predictor regresses onto a normalized target
            # Stabilizes training and discourages collapse to trivial constants.
            target_full = F.layer_norm(target_full, (target_full.size(-1),))

            # Flatten the block axis to gather every target patch in one pass,
            # then unfold it again so targets align with the predictor's output.
            flat_indices = target_indices.reshape(batch_size, num_blocks * block_size)
            expanded = flat_indices.unsqueeze(-1).expand(-1, -1, target_full.size(-1))
            targets = torch.gather(target_full, dim=1, index=expanded)
            targets = targets.view(batch_size, num_blocks, block_size, -1)

        predictions = self.predictor(context_repr, context_indices, target_indices)
        return predictions, targets

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA update: target = momentum * target + (1 - momentum) * context."""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)


if __name__ == "__main__":
    # End-to-end sanity check on a single I-JEPA step.
    from src.utils.masking import MultiBlockMasking

    batch_size = 2
    grid_size = 6
    num_patches = grid_size * grid_size  # 36

    encoder_kwargs = dict(in_channels=3, d_model=32, d_ff=128, num_heads=4,
                          num_layers=4, max_seq_len=num_patches)
    predictor_kwargs = dict(encoder_dim=32, predictor_dim=16, d_ff=64, num_heads=4,
                            num_layers=4, max_seq_len=num_patches)

    model = IJEPA(encoder_kwargs, predictor_kwargs)
    masker = MultiBlockMasking(grid_size=grid_size, generator=torch.Generator().manual_seed(0))

    imgs = torch.randn(batch_size, 3, 96, 96)
    ctx_idx, tgt_idx = masker(batch_size, device="cpu")

    preds, targets = model(imgs, ctx_idx, tgt_idx)

    # Mean over blocks, patches and features: every block has the same patch
    # count, so this equals averaging the per-block losses.
    loss = F.smooth_l1_loss(preds, targets)

    print("Context indices:  ", tuple(ctx_idx.shape))
    print("Target indices:   ", tuple(tgt_idx.shape))
    print("Predictions shape:", tuple(preds.shape))   # (B, num_blocks, block_size, encoder_dim)
    print("Targets shape:    ", tuple(targets.shape))
    print("Loss:             ", loss.item())

    # Confirm the EMA update runs and only trainable params carry grad.
    loss.backward()
    model.update_target_encoder(momentum=0.996)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_target = sum(p.numel() for p in model.target_encoder.parameters())
    print("Trainable params: ", n_train)
    print("Target params:    ", n_target, "(EMA, no grad)")
