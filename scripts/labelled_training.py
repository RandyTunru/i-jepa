from dotenv import load_dotenv
load_dotenv()  

import yaml
import torch
from torch.utils.data import DataLoader
import wandb

from src.datasets.stl10_labelled_dataset import STL10LabelledDataset
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
    train_dataset = STL10LabelledDataset(config['train_data_path'], config['train_labels_path'])

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [int(len(train_dataset) * config['train_split_size']), len(train_dataset) - int(len(train_dataset) * config['train_split_size'])],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=config['drop_last'],
        pin_memory=(device.type == 'cuda'),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,  # never discard val samples; metrics should cover the whole split
        pin_memory=(device.type == 'cuda'),
    )

    encoder_kwargs = dict(
        in_channels=config['in_channels'], patch_size=config['patch_size'],
        d_model=config['encoder_d_model'], d_ff=config['encoder_d_ff'], num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'], max_seq_len=config['max_seq_len'],
    )

    model = STL10Classifier(encoder_kwargs, num_classes=config['num_classes']).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
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

        # A classifier checkpoint stores 'encoder.*' + 'classifier.*'. 
        # the I-JEPA pretrain checkpoint stores 'context_encoder.*' / 'target_encoder.*' /'predictor.*'.
        # Keys are 'classifier.weight' etc., so testing for a bare 'classifier' key never matches.
        is_classifier_ckpt = any(k.startswith('classifier.') for k in state_dict)

        if is_classifier_ckpt:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded a classifier checkpoint (encoder + head).")
        else:
            # The I-JEPA paper states "We use the target-encoder for evaluation and
            # average pool its output to produce a global image representation."
            prefix = config.get('encoder_source', 'target_encoder') + "." # set the prefix to either 'context_encoder.' or 'target_encoder.' depending on the config
            encoder_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not encoder_state:
                raise ValueError(f"Checkpoint has no '{prefix}' weights to load")

            model.encoder.load_state_dict(encoder_state, strict=True)
            print(f"Warm-started encoder from '{prefix[:-1]}' ({len(encoder_state)} tensors); classifier head is fresh.")

        if config.get('is_resume', False):
            if not is_classifier_ckpt:
                raise ValueError("is_resume=True needs a classifier checkpoint, not an I-JEPA pretrain checkpoint")

            missing = [k for k in ('optimizer_state_dict', 'step') if k not in checkpoint]
            if missing:
                raise ValueError(f"is_resume=True but the checkpoint is missing: {missing}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for param_group in optimizer.param_groups:
                # The learning rate in the checkpoint may not match the current config, so we override it with the current config's learning rate.
                # This is important because the learning rate may have changed between runs, and we want to ensure that the optimizer uses the correct learning rate for the current training run.
                # Also note that this is only setting the base learning rate; the actual learning rate may be modified by the learning rate scheduler during training.
                # However the base learning rate is still important to set correctly, as it is used to compute the actual learning rate during training.
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
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, config, device, start_step)

    final_step = trainer.train()

    trainer.save_checkpoint(final_step, label="final")
    wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train I-JEPA on STL-10 Unlabelled Dataset')
    parser.add_argument('--config', type=str, default='configs/stl10_ijepa_classifier.yaml', help='Path to the training configuration YAML file')

    args = parser.parse_args()

    main(args.config)