import numpy as np
import os
import torch
from torch.utils.data import Dataset

class STL10UnlabelledDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, 'stl10_binary/unlabeled_X.bin')
        self.data = np.memmap(self.data_path, dtype=np.uint8, mode='r') # Each value is a uint8 representing pixel intensity

        self.per_image_value = 96 * 96 * 3  # Each image is 96x96 pixels with 3 color channels (RGB)
        self.num_samples = len(self.data) // self.per_image_value  # Each image is 96x96x3 
    
    def __len__(self):
        # Calculate the number of images in the dataset
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.per_image_value
        end = start + self.per_image_value

        image = self.data[start:end].reshape((3, 96, 96))  # Reshape to (C, H, W)
        image = np.transpose(image, (2, 1, 0))  # Convert to (H, W, C) format for image processing libraries

        tensor_image = torch.tensor(image, dtype=torch.float32) # Convert to a PyTorch tensor
        return tensor_image / 255.0 # Normalize pixel values to [0, 1] range
    
if __name__ == "__main__":
    import argparse
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='STL10 Unlabelled Dataset')
    parser.add_argument('--data-dir', type=str, default='./data/raw/', help='Path to the directory containing the STL-10 dataset')
    parser.add_argument('--index', type=int, default=0, help='Index of the image to retrieve from the dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    args = parser.parse_args()

    dataset = STL10UnlabelledDataset(args.data_dir)
    print(f"Total number of images in the dataset: {len(dataset)}")
    image = dataset[args.index]
    print(f"Shape of the image at index {args.index}: {image.shape}")
    plt.imshow(image)

    plt.title(f"Image at index {args.index}")
    plt.axis('off')
    plt.savefig(f"image_{args.index}.png")

    # Create a DataLoader to iterate through the dataset in batches
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Number of batches with batch size {args.batch_size}: {len(dataloader)}")

    for batch_idx, batch in enumerate(dataloader):
        plt.figure(figsize=(12, 6))
        for i in range(min(args.batch_size, len(batch))):
            plt.subplot(2, args.batch_size // 2, i + 1)
            plt.imshow(batch[i].numpy())
            plt.axis('off')
        plt.suptitle("First Batch of Images")
        plt.savefig("first_batch.png")
        break
