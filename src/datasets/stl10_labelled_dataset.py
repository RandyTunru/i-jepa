import numpy as np
import os
import torch
from torch.utils.data import Dataset

class STL10LabelledDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r') # Each value is a uint8 representing pixel intensity
        self.labels = np.memmap(labels_path, dtype=np.uint8, mode='r') # Each value is a uint8 representing the label

        self.per_image_value = 96 * 96 * 3  # Each image is 96x96 pixels with 3 color channels (RGB)
        self.num_samples = len(self.data) // self.per_image_value  # Each image is 96x96x3 
    
    def __len__(self):
        # Calculate the number of images in the dataset
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.per_image_value
        end = start + self.per_image_value

        image = self.data[start:end].reshape((3, 96, 96))  # Reshape to (C, H, W)
        image = np.transpose(image, (2, 1, 0))  # Fix the column-major orientation -> (H, W, C)

        tensor_image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        label = int(self.labels[idx]) - 1  # Convert to zero-based index

        # Return (C, H, W), the PyTorch-standard layout the models consume directly.
        return tensor_image.permute(2, 0, 1).contiguous(), label
    
if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='STL10 Labelled Dataset')
    parser.add_argument('--data-path', type=str, default='./data/raw/stl10_binary/train_X.bin', help='Path to the STL-10 training images binary file')
    parser.add_argument('--labels-path', type=str, default='./data/raw/stl10_binary/train_y.bin', help='Path to the STL-10 training labels binary file')
    parser.add_argument('--index', type=int, default=0, help='Index of the image to retrieve from the dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    args = parser.parse_args()

    dataset = STL10LabelledDataset(args.data_path, args.labels_path)
    print(f"Total number of images in the dataset: {len(dataset)}")
    image, label = dataset[args.index]
    print(f"Shape of the image at index {args.index}: {image.shape}, Label: {label}")  # (C, H, W)
    plt.imshow(image.permute(1, 2, 0))  # back to (H, W, C) for matplotlib

    plt.title(f"Image at index {args.index} with Label {label}")
    plt.axis('off')
    plt.savefig(f"labelled_image_{args.index}.png")

    # Create a DataLoader to iterate through the dataset in batches
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Number of batches with batch size {args.batch_size}: {len(dataloader)}")

    for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
        plt.figure(figsize=(12, 6))
        for i in range(min(args.batch_size, len(batch_images))):
            plt.subplot(2, args.batch_size // 2, i + 1)
            plt.imshow(batch_images[i].permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C)
            plt.title(f"Label: {batch_labels[i].item()}")
            plt.axis('off')
        plt.suptitle("First Batch of Images with Labels")
        plt.savefig(f"labelled_batch_{batch_idx}.png")
        break