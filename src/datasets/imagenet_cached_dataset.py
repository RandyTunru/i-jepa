"""
Locally-cached ImageNet for I-JEPA.

The streaming dataset pulls shards over HTTP on every step, which caps parallelism
(shuffling collapses the 52 shards to 5, so >5 workers idle) and makes training
hostage to network hiccups and HTTP-client lifetime bugs.

Downloading the compressed parquet instead costs ~22.5 GB for the whole train
split -- an order of magnitude less than materialising raw uint8 at 256px (252 GB)
-- and turns this into an ordinary map-style Dataset: real random access, so
shuffle=True, any num_workers, and persistent_workers all work normally.

Run `python -m scripts.imagenet_download --cache` first to populate the cache.
Yields (C, H, W) float in [0, 1], matching every other dataset in this project.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.datasets.imagenet_stream_dataset import resize_center_crop

class ImageNetCachedDataset(Dataset):
    def __init__(self, dataset_name="evanarlian/imagenet_1k_resized_256", split="train",
                 image_size=256, return_label=False):
        # Reads from the local HF cache; downloads on first use.
        self.data = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.return_label = return_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Same transform as the streaming path: RGB, resize short side, center crop.
        arr = resize_center_crop(example["image"], self.image_size)  # (H, W, C) uint8
        image = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (C, H, W)

        if self.return_label:
            return image, int(example["label"])
        return image


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Cached ImageNet dataset sanity check")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    dataset = ImageNetCachedDataset(split=args.split, image_size=args.image_size,
                                    return_label=True)
    print(f"split '{args.split}': {len(dataset)} images")

    # Map-style: shuffle=True and persistent_workers work, unlike the stream.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    images, labels = next(iter(dataloader))
    print(f"batch: images {tuple(images.shape)}  labels {tuple(labels.shape)}  "
          f"min {images.min():.3f} max {images.max():.3f}")
    print("OK")
