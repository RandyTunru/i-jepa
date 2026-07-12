import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import IterableDataset


def resize_center_crop(pil_image, size):
    """
    Standard ImageNet eval transform: force RGB, resize the short side to
    `size` (preserving aspect ratio), then center-crop a square `size`x`size`.
    Returns a (size, size, 3) uint8 array. This avoids the aspect-ratio squashing
    of a direct resize -- the source mirror keeps the short side at 256 but lets
    the long side vary (e.g. 341x256).
    """
    img = pil_image.convert("RGB")  # ImageNet has grayscale/CMYK/RGBA images
    w, h = img.size

    scale = size / min(w, h)
    if scale != 1.0:
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))))

    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    return np.asarray(img, dtype=np.uint8)

class ImageNetStreamDataset(IterableDataset):
    def __init__(self, dataset_name="evanarlian/imagenet_1k_resized_256", split="train",
                 image_size=256, shuffle_buffer=10000, seed=0, return_label=False,
                 loop=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.return_label = return_label
        # loop=True re-opens the stream in-place when it is exhausted, so the
        # DataLoader iterator never raises StopIteration. Without it, the trainer
        # rebuilds the iterator on every pass, which reuses workers whose HTTP
        # clients are already closed ("Cannot send a request, as the client has
        # been closed"). Pretraining wants loop=True; eval wants loop=False.
        self.loop = loop

    def _build_stream(self, epoch=0):
        stream = load_dataset(self.dataset_name, split=self.split, streaming=True)

        # NOTE: do not shard by worker here. A HuggingFace streaming IterableDataset
        # already distributes its shards across DataLoader workers automatically
        if self.shuffle_buffer > 0:
            # Vary the seed per pass so a looping stream does not replay the same order.
            stream = stream.shuffle(seed=self.seed + epoch, buffer_size=self.shuffle_buffer)

        return stream

    def _to_tensor(self, pil_image):
        arr = resize_center_crop(pil_image, self.image_size)  # (H, W, C) uint8
        # tensor = torch.from_numpy(arr).float() / 255.0  # normalize to [0, 1]
        tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (C, H, W) float32
        return tensor

    def __iter__(self):
        epoch = 0
        while True:
            # Rebuilt inside the worker each pass, so every pass gets a fresh HTTP
            # client rather than reusing one that was closed when the last pass ended.
            for example in self._build_stream(epoch):
                image = self._to_tensor(example["image"])
                if self.return_label:
                    yield image, int(example["label"])
                else:
                    yield image

            if not self.loop:
                return
            epoch += 1


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Streaming ImageNet dataset sanity check")
    parser.add_argument("--dataset", type=str, default="evanarlian/imagenet_1k_resized_256")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()

    dataset = ImageNetStreamDataset(dataset_name=args.dataset, image_size=args.image_size)
    # IterableDataset -> shuffle must be False; the dataset shuffles internally.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(f"Streaming from: {args.dataset}")
    for i, batch in enumerate(dataloader):
        print(f"batch {i}: shape {tuple(batch.shape)}  "
              f"min {batch.min():.3f}  max {batch.max():.3f}  mean {batch.mean():.3f}")
        if i + 1 >= args.batches:
            break
    print("OK")
