from dotenv import load_dotenv
load_dotenv()  

import yaml
import torch
from torch.utils.data import DataLoader
import wandb

from src.datasets.stl10_unlabelled_dataset import STL10UnlabelledDataset
from src.models.ijepa import IJEPA
from src.training.unlabelled_trainer import Trainer


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
    dataset = STL10UnlabelledDataset(config['data_dir'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=config['drop_last'],
        pin_memory=(device.type == 'cuda'),
    )

    # --- Model ---
    # Encoder dims follow a small ViT; the predictor is deliberately narrower.
    encoder_kwargs = dict(
        in_channels=config['in_channels'], patch_size=config['patch_size'],
        d_model=config['encoder_d_model'], d_ff=config['encoder_d_ff'], num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'], max_seq_len=config['max_seq_len'],
        dropout=config['encoder_dropout'],
    )
    predictor_kwargs = dict(
        encoder_dim=config['encoder_d_model'], predictor_dim=config['predictor_d_model'], d_ff=config['predictor_d_ff'], num_heads=config['predictor_num_heads'],
        num_layers=config['predictor_num_layers'], max_seq_len=config['max_seq_len'],
        dropout=config['predictor_dropout'],
    )
    model = IJEPA(encoder_kwargs, predictor_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), # Only use trainable parameters (context encoder + predictor) for optimization; EMA target is updated separately.
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    # --- Checkpoint Loading ---
    start_step = 0
    generator_state = None

    if config.get('from_checkpoint'):
        print(f"Loading weights from checkpoint: {config['from_checkpoint']}")
        checkpoint = torch.load(config['from_checkpoint'], map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if config.get('is_resume', False):
            if ['optimizer_state_dict', 'generator_state_dict', 'step'] not in checkpoint:
                raise ValueError("is_resume=True but the checkpoint has no optimizer_state_dict or generator_state_dict or step. Cannot resume training.")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            generator_state = checkpoint['generator_state_dict']

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
    trainer = Trainer(model, dataloader, optimizer, config, device, start_step)
    if generator_state is not None:
        trainer.masker.generator.set_state(generator_state)
        print("Restored masker generator state from checkpoint.")

    final_step = trainer.train()

    trainer.save_checkpoint(final_step, label="final")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train I-JEPA on STL-10 Unlabelled Dataset')
    parser.add_argument('--config', type=str, default='configs/stl10_ijepa.yaml', help='Path to the training configuration YAML file')

    args = parser.parse_args()

    main(args.config)
