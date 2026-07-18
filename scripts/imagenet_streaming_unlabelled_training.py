from dotenv import load_dotenv
load_dotenv()

import yaml
import torch
from torch.utils.data import DataLoader
import wandb

from src.datasets.imagenet_stream_dataset import ImageNetStreamDataset
from src.datasets.imagenet_cached_dataset import ImageNetCachedDataset
from src.models.ijepa import IJEPA
from src.training.unlabelled_trainer import Trainer
from src.utils.optim import param_groups_with_decay


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    assert config['image_size'] % config['patch_size'] == 0, \
        "image_size must be divisible by patch_size"

    config['grid_size'] = config['image_size'] // config['patch_size']
    config['max_seq_len'] = config['grid_size'] ** 2
    print(f"Patch grid: {config['grid_size']}x{config['grid_size']} = {config['max_seq_len']} tokens")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    # Cached (default) reads the locally downloaded parquet: a map-style Dataset,
    # so shuffle=True, any num_workers, and persistent_workers all work. Streaming
    # avoids the ~22.5 GB download but caps parallelism (shuffling collapses the
    # 52 shards to 5) and depends on the network every step.
    use_streaming = config.get('use_streaming', False)

    if use_streaming:
        dataset = ImageNetStreamDataset(
            image_size=config['image_size'],
            shuffle_buffer=config.get('shuffle_buffer', 10000),
            seed=config.get('mask_seed', 0),
            loop=True,  # never exhaust: training is driven by steps, not epochs
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,  # IterableDataset: the dataset shuffles, not the loader
            num_workers=min(config['num_workers'], 5),  # shuffled stream exposes only 5 shards
            drop_last=config['drop_last'],
            pin_memory=(device.type == 'cuda'),
            # Must stay False. Reusing a persisted worker across DataLoader iterators
            # resurrects its already-closed HTTP client -> "Cannot send a request,
            # as the client has been closed". loop=True means we never re-iterate.
            persistent_workers=False,
        )
        print("Data: streaming from HuggingFace (no local cache).")
    else:
        dataset = ImageNetCachedDataset(image_size=config['image_size'])
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,  # map-style: real random access
            num_workers=config['num_workers'],
            drop_last=config['drop_last'],
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0),
        )
        print(f"Data: local HF cache, {len(dataset)} images.")

    # --- Model ---
    encoder_kwargs = dict(
        in_channels=config['in_channels'], patch_size=config['patch_size'],
        d_model=config['encoder_d_model'], d_ff=config['encoder_d_ff'], num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'], max_seq_len=config['max_seq_len'],
        dropout=config['encoder_dropout'],
    )
    predictor_kwargs = dict(
        encoder_dim=config['encoder_d_model'], predictor_dim=config['predictor_d_model'], d_ff=config['predictor_d_ff'],
        num_heads=config['predictor_num_heads'], num_layers=config['predictor_num_layers'], max_seq_len=config['max_seq_len'],
        dropout=config['predictor_dropout'],
    )
    model = IJEPA(encoder_kwargs, predictor_kwargs).to(device)

    # Standard ViT practice: exclude norm and bias parameters from weight decay.
    # In I-JEPA this is critical: weight decay on the encoder's final norm pulls
    # it toward zero, and the predictor absorbs the scale change, so the loss
    # provides no counter-gradient — the encoder silently collapses while the
    # training loss stays flat.
    param_groups = param_groups_with_decay(model, lr=config['learning_rate'],
                                           weight_decay=config['weight_decay'],
                                           betas=(0.9, 0.95))
    optimizer = torch.optim.AdamW(param_groups)

    # --- Checkpoint Loading ---
    start_step = 0
    generator_state = None

    if config.get('from_checkpoint'):
        print(f"Loading weights from checkpoint: {config['from_checkpoint']}")
        checkpoint = torch.load(config['from_checkpoint'], map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if config.get('is_resume', False):
            missing = [k for k in ['optimizer_state_dict', 'generator_state_dict', 'step'] if k not in checkpoint]
            if missing:
                raise ValueError(f"is_resume=True but the checkpoint is missing: {missing}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            generator_state = checkpoint['generator_state_dict']

            for param_group in optimizer.param_groups:
                # Current config wins over the checkpoint's stored lr/weight_decay.
                param_group['lr'] = config['learning_rate']
                param_group['weight_decay'] = config['weight_decay']

            start_step = checkpoint['step']
            assert isinstance(start_step, int), \
                f"checkpoint 'step' must be an int to resume, got {start_step!r}"

            print(f"is_resume=True: restored optimizer, resuming from step {start_step}.")
        else:
            print("is_resume=False: weights warm-started, fresh optimizer, step 0.")

    # --- Logging ---
    wandb.init(
        project=config['project_name'],
        name=config['run_name'],
        config=config,
        dir=config['wandb_dir'],
    )

    # --- Train ---
    trainer = Trainer(model, dataloader, optimizer, config, device, start_step)
    if generator_state is not None:
        trainer.masker.generator.set_state(generator_state)
        print("Restored masker generator state from checkpoint.")

    final_step = trainer.train()

    trainer.save_checkpoint(final_step, label="final")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stream-train I-JEPA on ImageNet-1k (HuggingFace)')
    parser.add_argument('--config', type=str, default='configs/imagenet_ijepa.yaml',
                        help='Path to the training configuration YAML file')

    args = parser.parse_args()

    main(args.config)
