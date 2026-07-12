from dotenv import load_dotenv
load_dotenv()

import os
import argparse

import numpy as np
from datasets import load_dataset

from src.datasets.imagenet_stream_dataset import resize_center_crop

DATASET_NAME = "evanarlian/imagenet_1k_resized_256"


def verify(dataset_name, num_examples):

    print(f"Streaming from: {dataset_name}")
    stream = load_dataset(dataset_name, split="train", streaming=True)

    for i, example in enumerate(stream):
        print(f"  example {i}: size {example['image'].size}, "
              f"mode {example['image'].mode}, label {example.get('label')}")
        if i + 1 >= num_examples:
            break
    print(f"Verified {num_examples} examples. Streaming works; no local download needed.")


def cache(dataset_name, splits):
    """Download the compressed parquet into the local HF cache.

    This is the recommended path. The whole train split is only ~22.5 GB, versus
    ~252 GB to materialize it as raw uint8 at 256px. Once cached, training reads
    locally with a map-style Dataset: no network, no HTTP-client failures, and no
    shard cap on num_workers (streaming's .shuffle() collapses 52 shards to 5).
    """
    for split in splits:
        print(f"Caching split '{split}' from {dataset_name} ...")
        data = load_dataset(dataset_name, split=split)  # downloads, then caches
        print(f"  split '{split}': {len(data)} images cached")

    print("\nDone. Train with the cached dataset via ImageNetCachedDataset.")
    print("Cache location: $HF_HOME or ~/.cache/huggingface (set HF_HOME to relocate).")


def materialize(dataset_name, image_size, num_images, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, "imagenet_X.bin")

    # Preallocate the memmap so we write in place rather than growing a buffer.
    x = np.memmap(x_path, dtype=np.uint8, mode="w+",
                  shape=(num_images, image_size, image_size, 3))

    stream = load_dataset(dataset_name, split="train", streaming=True)
    written = 0
    for example in stream:
        x[written] = resize_center_crop(example["image"], image_size)
        written += 1
        if written % 5000 == 0:
            print(f"  wrote {written}/{num_images}")
        if written >= num_images:
            break

    x.flush()
    with open(os.path.join(out_dir, "imagenet_meta.txt"), "w") as f:
        f.write(f"{written} {image_size}\n")
    print(f"Materialized {written} images -> {x_path} "
          f"({os.path.getsize(x_path) / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare streaming ImageNet for I-JEPA")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Target square size. The default source is 256x256, so 256 avoids resizing.")
    parser.add_argument("--verify-examples", type=int, default=5)
    parser.add_argument("--cache", action="store_true",
                        help="Download the compressed parquet to the local HF cache (~22.5 GB for train). Recommended.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to cache with --cache")
    parser.add_argument("--materialize", type=int, default=0,
                        help="If > 0, write this many images to a local memmap instead of just verifying")
    parser.add_argument("--out", type=str, default="data/raw/imagenet")
    args = parser.parse_args()

    if args.cache:
        cache(args.dataset, args.splits)
    elif args.materialize > 0:
        materialize(args.dataset, args.image_size, args.materialize, args.out)
    else:
        verify(args.dataset, args.verify_examples)


if __name__ == "__main__":
    main()
