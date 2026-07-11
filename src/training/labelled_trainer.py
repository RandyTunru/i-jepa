import os
import math
import torch
import torch.nn.functional as F
import wandb

from src.utils.metrics import ThroughputMonitor
from src.utils.performance import f1_score, accuracy

class Trainer: 
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, config, device, start_step=0):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.start_step = start_step

        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.monitor = ThroughputMonitor(config['batch_size'] * self.grad_accum_steps, config['max_seq_len'])

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

    def train(self):
        self.model.train()

        step = self.start_step

        use_amp = self.device.type == 'cuda'
        ctx = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp)

        data_iter = iter(self.train_dataloader)

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
                        return step
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                imgs, labels = batch

                # All datasets yield (B, C, H, W), the layout the encoder expects.
                imgs = imgs.to(self.device).contiguous()

                with ctx:
                    # Forward Pass
                    outputs = self.model(imgs)
                    loss = F.cross_entropy(outputs, labels.to(self.device))

                    # Scale by the accumulation steps so the effective gradient is
                    # the mean over micro-batches rather than their sum.
                    loss = loss / self.grad_accum_steps

                # Backward Pass
                loss.backward()
                accum_loss += loss.item()  # Accumulate the scaled loss for logging purposes

            # --- Optimization Step ---
            # Gradient clipping to prevent exploding gradients (once per optimization step, not per micro-step)
            torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), max_norm=1.0)

            # Step the optimizer (the linear head only; the encoder is frozen)
            self.optimizer.step()

            # Logging & Checkpointing
            if step % self.config['log_every'] == 0:
                tps = self.monitor.get_tps()

                with torch.no_grad():
                    self.model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    all_preds, all_labels = [], []

                    # Cap the eval for large/streaming val sets; None => full pass.
                    val_max_batches = self.config.get('val_max_batches')

                    for vi, val_batch in enumerate(self.val_dataloader):
                        if val_max_batches is not None and vi >= val_max_batches:
                            break
                        val_imgs, val_labels = val_batch
                        val_imgs = val_imgs.to(self.device).contiguous()
                        val_labels = val_labels.to(self.device)

                        val_outputs = self.model(val_imgs)
                        val_loss += F.cross_entropy(val_outputs, val_labels).item()
                        val_steps += 1

                        all_preds.append(torch.argmax(val_outputs, dim=1).cpu())
                        all_labels.append(val_labels.cpu())

                    self.model.train()

                avg_val_loss = val_loss / max(val_steps, 1)

                # Metrics over the whole validation split, not just the last batch.
                preds = torch.cat(all_preds)
                targets = torch.cat(all_labels)
                f1 = f1_score(targets, preds)
                acc = accuracy(targets, preds)

                wandb.log({
                    "train/loss": accum_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "val/loss": avg_val_loss,
                    "val/f1_score": f1,
                    "val/accuracy": acc,
                    "metrics/patches_per_sec": tps,
                    "step": step
                })
                print(f"Step {step:05d} | Loss: {accum_loss:.4f} | Val Loss: {avg_val_loss:.4f} | PPS: {tps:.0f}")

            if step > 0 and step % self.config['save_every'] == 0:
                self.save_checkpoint(step)

            step += 1

        return step

    def save_checkpoint(self, step, label=None):
        assert isinstance(step, int), f"step must be an int, got {type(step).__name__}"

        name = label if label is not None else str(step)
        filepath = os.path.join(self.checkpoint_dir, f"model_step_{name}.pt")
        checkpoint = {
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")