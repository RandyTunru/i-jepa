import torch
from torch import nn

from src.models.modules.encoder import ViTEncoder

class STL10Classifier(nn.Module):
    def __init__(self, encoder_kwargs, num_classes=10):
        super().__init__()
        self.encoder = ViTEncoder(**encoder_kwargs)
        self.classifier = nn.Linear(encoder_kwargs['d_model'], num_classes)

        for param in self.encoder.parameters():
            # Disable grad so it is excluded from the optimizer and never receives backprop updates.
            param.requires_grad = False

    def trainable_parameters(self):
        """Only the linear head trains; the encoder is frozen (linear probe)."""
        return [param for param in self.parameters() if param.requires_grad]

    def train(self, mode=True):
        """
        Keep the encoder in eval mode even while training.
        The purpose of this is to ensure that the encoder's batch norm (if any) and dropout (if any) behave consistently during training.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, imgs):
        """
        Args:
            imgs: (batch_size, channels, height, width)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Get the representation from the encoder. ViTEncoder ends with its own
        # final norm, so these tokens are already in a sane scale for the head.
        with torch.no_grad():  # Ensure no gradients are computed for the encoder
            x = self.encoder(imgs)  # (batch_size, num_patches, d_model)

        # Use global average pooling to get a single representation per image
        # (the I-JEPA paper average-pools the target-encoder output).
        x = x.mean(dim=1)  # (B, encoder_dim)

        # Pass through the classifier
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    @torch.no_grad()
    def predict(self, imgs):
        """
        Args:
            imgs: (batch_size, channels, height, width)
        Returns:
            predicted_labels: (batch_size,)
        """
        self.eval()  # Set the model to evaluation mode
        logits = self.forward(imgs)  # (B, num_classes)
        predicted_labels = torch.argmax(logits, dim=1)  # (B,)
        return predicted_labels