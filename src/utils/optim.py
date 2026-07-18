"""
Weight-decay parameter grouping for ViT-style architectures.

Standard ViT training practice excludes norm (LayerNorm, RMSNorm) and bias
parameters from weight decay. Applying weight decay to norms is especially
harmful in I-JEPA: the encoder's final norm weight directly scales the output,
and the predictor can absorb any change in that scale via its own projections,
so the loss provides no counter-gradient. Weight decay pulls the norm weight
monotonically downward, which cascades into smaller activations throughout the
encoder and silently kills representation quality while the training loss stays
flat.

Grouping parameters into decay/no-decay sets is the standard fix (used in MAE,
DINO, the reference I‑JEPA, and every modern ViT training pipeline).
"""


def param_groups_with_decay(model, weight_decay, lr,
                            betas=(0.9, 0.95),
                            no_decay_patterns=(".bias", "norm")):
    """
    Return optimizer parameter groups that exclude norms and biases from weight
    decay. `model` should implement `trainable_parameters()` returning a flat
    list of parameters (the IJEPA convention).

    `no_decay_patterns` are substring-matched against each parameter name.
    Defaults catch every norm (norm, norm1, norm2, predictor_norm, ...) and
    every bias, matching the standard ViT convention used in MAE, DINO, and
    the reference I‑JEPA.
    """
    decay_params = []
    no_decay_params = []

    named = {id(p): name for name, p in model.named_parameters()}

    for param in model.trainable_parameters():
        name = named.get(id(param), "")
        if any(pat in name for pat in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params,    "weight_decay": weight_decay, "lr": lr, "betas": betas},
        {"params": no_decay_params, "weight_decay": 0.0,          "lr": lr, "betas": betas},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay:,} params w/ weight_decay={weight_decay}, "
          f"{n_nodecay:,} params w/ weight_decay=0.0")

    return groups
