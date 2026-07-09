import torch


class MultiBlockMasking:
    """
    I-JEPA multi-block masking on a square patch grid.

    Follows the reference MaskCollator:
      1. Sample ONE target block size and ONE context block size per batch.
         Sharing the size (not the position) is what keeps the resulting index
         tensors rectangular: every target block has exactly h*w patches.
      2. For each image independently, sample `num_target_blocks` target block
         positions. Their union is that image's target set.
      3. For each image, sample a context block position and subtract the target
         set from it, so context and target patches are disjoint.
      4. Context sizes still vary per image (the subtraction removes a different
         number of patches each time), so truncate every context to the shortest
         one in the batch.

    Returns per-block target indices, since the predictor is run once per target
    block rather than on the union of all blocks.
    """

    def __init__(self, grid_size, num_target_blocks=4,
                 target_scale=(0.15, 0.2), target_ratio=(0.75, 1.5),
                 context_scale=(0.85, 1.0), context_ratio=(1.0, 1.0),
                 min_keep=4, max_tries=20, generator=None):
        self.grid_size = grid_size 
        self.num_target_blocks = num_target_blocks
        self.target_scale = target_scale
        self.target_ratio = target_ratio
        self.context_scale = context_scale
        self.context_ratio = context_ratio
        self.min_keep = min_keep      # smallest acceptable context after subtraction
        self.max_tries = max_tries    # context position retries before resampling sizes
        self.generator = generator    # torch.Generator for reproducible/resumable masking

    def _uniform(self, low, high):
        """Scalar drawn from U[low, high) using the torch RNG."""
        return low + (high - low) * torch.rand((), generator=self.generator).item()

    def _sample_block_size(self, scale_range, ratio_range):
        """Block height/width in patch units. `scale` is an area fraction of the
        patch grid; no image resizing is involved."""
        g = self.grid_size
        block_area = self._uniform(*scale_range) * (g * g)
        ratio = self._uniform(*ratio_range)

        h = max(1, min(g, int(round((block_area * ratio) ** 0.5))))
        w = max(1, min(g, int(round((block_area / ratio) ** 0.5))))
        return h, w

    def _sample_block_indices(self, h, w):
        """Flat patch indices of one h x w rectangle placed uniformly at random."""
        g = self.grid_size
        top = torch.randint(0, g - h + 1, (), generator=self.generator).item()
        left = torch.randint(0, g - w + 1, (), generator=self.generator).item()

        rows = torch.arange(top, top + h)
        cols = torch.arange(left, left + w)
        return (rows.view(-1, 1) * g + cols.view(1, -1)).flatten()

    def __call__(self, batch_size, device):
        """
        Returns:
            context_indices: (batch_size, num_context) long tensor
            target_indices:  (batch_size, num_target_blocks, block_size) long tensor
        """
        while True:
            # Block sizes are drawn once per batch and shared across all images.
            th, tw = self._sample_block_size(self.target_scale, self.target_ratio)
            ch, cw = self._sample_block_size(self.context_scale, self.context_ratio)

            per_image_targets = []
            per_image_context = []

            for _ in range(batch_size):
                # Target block positions are independent per image. Blocks may
                # overlap each other; only their union matters for the context.
                blocks = [self._sample_block_indices(th, tw)
                          for _ in range(self.num_target_blocks)]

                target_union = set()
                for block in blocks:
                    target_union |= set(block.tolist())

                # Context block, with every target patch removed. Retry the
                # position if the subtraction leaves too little context.
                context = None
                for _ in range(self.max_tries):
                    candidate = set(self._sample_block_indices(ch, cw).tolist()) - target_union
                    if len(candidate) >= self.min_keep:
                        context = candidate
                        break

                if context is None:
                    break  # give up on these block sizes, resample them

                per_image_targets.append(torch.stack(blocks))
                per_image_context.append(sorted(context))

            if len(per_image_context) == batch_size:
                break

        # Contexts have unequal lengths across the batch; truncate to the shortest
        # so they stack into a rectangular tensor (this is what min_keep guards).
        shortest = min(len(c) for c in per_image_context)
        context_indices = torch.tensor([c[:shortest] for c in per_image_context],
                                       device=device, dtype=torch.long)

        # Every target block has exactly th*tw patches, so no truncation needed.
        target_indices = torch.stack(per_image_targets).to(device=device, dtype=torch.long)

        return context_indices, target_indices


if __name__ == "__main__":
    # Sanity-check the masking on a 6x6 patch grid (96x96 image, 16x16 patches).
    generator = torch.Generator().manual_seed(0)
    masker = MultiBlockMasking(grid_size=6, num_target_blocks=4, generator=generator)

    batch_size = 4
    ctx_idx, tgt_idx = masker(batch_size, device="cpu")

    print("Context indices shape:", tuple(ctx_idx.shape))  # (B, num_context)
    print("Target indices shape: ", tuple(tgt_idx.shape))  # (B, num_blocks, block_size)

    # Per image, context and the union of its target blocks must be disjoint.
    for b in range(batch_size):
        overlap = set(ctx_idx[b].tolist()) & set(tgt_idx[b].flatten().tolist())
        assert not overlap, f"image {b}: context/target overlap {overlap}"
    print("Context/target disjoint per image: True")

    # Positions vary per image; block sizes do not.
    print("Positions differ across batch:", not bool((tgt_idx[0] == tgt_idx[1]).all()))
    print("All blocks same size:        ", tgt_idx.size(-1))

    # Each block is a connected rectangle on the grid; blocks need not touch.
    print("Block 0 of image 0 (flat):  ", sorted(tgt_idx[0, 0].tolist()))
