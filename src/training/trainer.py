import os
import math
import torch
import torch.nn.functional as F
import wandb

from src.utils.metrics import ThroughputMonitor
from src.utils.masking import MultiBlockMasking


class Trainer:
    def __init__(self, model, dataloader, optimizer, config, device, start_step=0):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.start_step = start_step

        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.monitor = ThroughputMonitor(config['batch_size'] * self.grad_accum_steps, config['max_seq_len'])

        # Multi-block masking (batch-shared). A seeded generator keeps the mask
        # stream deterministic so runs are reproducible and resumable.
        mask_generator = torch.Generator().manual_seed(config.get('mask_seed', 0))
        self.masker = MultiBlockMasking(
            grid_size=config['grid_size'],
            num_target_blocks=config.get('num_target_blocks', 4),
            generator=mask_generator,
        )

    def get_lr(self, step):
        max_lr = self.config['learning_rate']
        min_lr = max_lr * 0.1  # Usually 10% of max_lr
        warmup_steps = self.config.get('warmup_steps', 1000)
        total_steps = self.config['max_steps']

        # 1. Linear warmup
        if step < warmup_steps:
            return max_lr * (step / warmup_steps)

        # 2. Minimum LR after training exceeds max_steps
        if step > total_steps:
            return min_lr

        # 3. Cosine decay down to min_lr
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)

    def get_momentum(self, step):
        """EMA momentum for the target encoder, ramped from ema_start -> ema_end
        over training on a cosine schedule (standard I-JEPA / BYOL practice)."""
        start = self.config.get('ema_start', 0.996)
        end = self.config.get('ema_end', 1.0)
        total = self.config['max_steps']

        step = min(step, total)
        coeff = 0.5 * (1.0 + math.cos(math.pi * step / total))  # 1 at step 0 -> 0 at total
        return end - (end - start) * coeff

    def train(self):
        self.model.train()

        step = self.start_step

        use_amp = self.device.type == 'cuda'
        ctx = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp)

        data_iter = iter(self.dataloader)

        print("Starting training loop...")
        self.monitor.start()

        while step < self.config['max_steps']:
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # --- Gradient Accumulation Loop ---
            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self.config['no_epochs']:
                        print("Dataset exhausted. Stopping training loop.")
                        return
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Dataset yields (B, H, W, C) in [0, 1]; the encoder expects (B, C, H, W).
                imgs = batch.to(self.device).permute(0, 3, 1, 2).contiguous()

                # Sample one block mask and broadcast it across the batch.
                context_indices, target_indices = self.masker(imgs.size(0), self.device)

                with ctx:
                    # preds: (B, num_target, encoder_dim); targets are stop-grad.
                    preds, targets = self.model(imgs, context_indices, target_indices)

                    # Regress predicted target representations onto the EMA targets.
                    loss = F.smooth_l1_loss(preds, targets)

                    # Scale loss based on gradient accumulation steps to maintain correct optimization dynamics
                    loss = loss / self.grad_accum_steps

                # Backward Pass
                loss.backward()
                accum_loss += loss.item()  # Accumulate the scaled loss for logging purposes

            # --- Optimization Step ---
            # Gradient clipping to prevent exploding gradients (once per optimization step, not per micro-step)
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), max_norm=1.0)

            # Step the optimizer (context encoder + predictor only)
            self.optimizer.step()

            # EMA update of the target encoder AFTER the optimizer step.
            momentum = self.get_momentum(step)
            self.model.update_target_encoder(momentum)

            # Logging & Checkpointing
            if step % self.config['log_every'] == 0:
                tps = self.monitor.get_tps()
                wandb.log({
                    "train/loss": accum_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train/ema_momentum": momentum,
                    "metrics/patches_per_sec": tps,
                    "step": step
                })
                print(f"Step {step:05d} | Loss: {accum_loss:.4f} | EMA: {momentum:.4f} | PPS: {tps:.0f}")

            if step > 0 and step % self.config['save_every'] == 0:
                self.save_checkpoint(step)

            step += 1

    def save_checkpoint(self, step):
        filepath = os.path.join(self.checkpoint_dir, f"model_step_{step}.pt")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # context + target encoders + predictor
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
