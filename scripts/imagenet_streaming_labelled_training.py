from dotenv import load_dotenv
load_dotenv()

import yaml
import torch
from torch.utils.data import DataLoader
import wandb

from src.datasets.imagenet_stream_dataset import ImageNetStreamDataset
from src.datasets.imagenet_cached_dataset import ImageNetCachedDataset
from src.models.classifier import STL10Classifier
from src.training.labelled_trainer import Trainer


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
    # Cached (default) uses the locally downloaded parquet: map-style, so shuffle
    # and persistent_workers work. The 'val' split holds the labelled eval set.
    use_streaming = config.get('use_streaming', False)

    if use_streaming:
        train_dataset = ImageNetStreamDataset(
            split="train", image_size=config['image_size'],
            shuffle_buffer=config.get('shuffle_buffer', 10000),
            seed=config.get('mask_seed', 0), return_label=True,
            loop=True,  # never exhaust: training is driven by steps, not epochs
        )
        val_dataset = ImageNetStreamDataset(
            split="val", image_size=config['image_size'],
            shuffle_buffer=0, return_label=True,  # deterministic, no shuffle for eval
            loop=False,  # eval must terminate so the val pass ends
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=min(config['num_workers'], 5),  # shuffled stream exposes 5 shards
            drop_last=config['drop_last'], pin_memory=(device.type == 'cuda'),
            # Must stay False: a persisted worker's HTTP client is closed once its
            # stream ends, and reusing it raises "client has been closed".
            persistent_workers=False,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=2, drop_last=False,  # val split has 2 shards
            pin_memory=(device.type == 'cuda'),
        )
        print("Data: streaming from HuggingFace (no local cache).")
    else:
        train_dataset = ImageNetCachedDataset(
            split="train", image_size=config['image_size'], return_label=True)
        val_dataset = ImageNetCachedDataset(
            split="val", image_size=config['image_size'], return_label=True)
        train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], drop_last=config['drop_last'],
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0),
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], drop_last=False,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0),
        )
        print(f"Data: local HF cache, train {len(train_dataset)} / val {len(val_dataset)} images.")

    # --- Model (frozen encoder + linear head) ---
    encoder_kwargs = dict(
        in_channels=config['in_channels'], patch_size=config['patch_size'],
        d_model=config['encoder_d_model'], d_ff=config['encoder_d_ff'], num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'], max_seq_len=config['max_seq_len'],
    )
    model = STL10Classifier(encoder_kwargs, num_classes=config['num_classes']).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),   # only the linear head; the encoder is frozen
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    # --- Checkpoint Loading ---
    start_step = 0

    if config.get('from_checkpoint'):
        print(f"Loading weights from checkpoint: {config['from_checkpoint']}")
        checkpoint = torch.load(config['from_checkpoint'], map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Classifier checkpoint (encoder.* + classifier.*) vs I-JEPA pretrain
        # (context_encoder.* / target_encoder.* / predictor.*).
        is_classifier_ckpt = any(k.startswith('classifier.') for k in state_dict)

        if is_classifier_ckpt:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded a classifier checkpoint (encoder + head).")
        else:
            prefix = config.get('encoder_source', 'target_encoder') + "."
            encoder_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if not encoder_state:
                raise ValueError(f"Checkpoint has no '{prefix}' weights to load")
            model.encoder.load_state_dict(encoder_state, strict=True)
            print(f"Warm-started encoder from '{prefix[:-1]}' ({len(encoder_state)} tensors); head is fresh.")

        if config.get('is_resume', False):
            if not is_classifier_ckpt:
                raise ValueError("is_resume=True needs a classifier checkpoint, not an I-JEPA pretrain checkpoint")
            missing = [k for k in ('optimizer_state_dict', 'step') if k not in checkpoint]
            if missing:
                raise ValueError(f"is_resume=True but the checkpoint is missing: {missing}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate']
                param_group['weight_decay'] = config['weight_decay']
            start_step = checkpoint['step']
            assert isinstance(start_step, int), \
                f"checkpoint 'step' must be an int to resume, got {start_step!r}"
            print(f"is_resume=True: restored optimizer, resuming from step {start_step}.")
        else:
            print("is_resume=False: encoder warm-started, fresh head, step 0.")

    # --- Logging ---
    wandb.init(project=config['project_name'], name=config['run_name'], config=config, dir=config['wandb_dir'])

    # --- Train ---
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, config, device, start_step)
    final_step = trainer.train()

    trainer.save_checkpoint(final_step, label="final")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Linear-probe I-JEPA on ImageNet-1k (streaming)')
    parser.add_argument('--config', type=str, default='configs/imagenet_ijepa_classifier.yaml',
                        help='Path to the training configuration YAML file')

    args = parser.parse_args()

    main(args.config)
