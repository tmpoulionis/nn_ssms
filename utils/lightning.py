import os
import torch
import copy
import lightning as L
from torchmetrics.functional import accuracy
from utils.utils import create_scheduler
from utils.noise_injection import NoiseInjector
from utils.non_negativity import compute_negative_penalty, NonNegativityScheduler, check_non_negativity

class LightningMamba(L.LightningModule):
    def __init__(self, model, total_steps, optimizer, loss_fn, lr_scheduler=None, opt_hyperparams=None, noise_injection=None, non_negative=None):
        super().__init__()
        self.saved_weights = {}
        self.model = model
        self.total_steps = total_steps
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.opt_hyperparams = opt_hyperparams if opt_hyperparams is not None else {}
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        # Non-Negativity
        if non_negative is not None:
            self.nn_enabled = non_negative["enabled"]
            self.nn_penalty = non_negative["penalty_type"]
            self.nn_weight = non_negative["penalty_weight"]
            self.live_clip = non_negative["live_clipping"]
            self.clip_interval = non_negative["clip_interval"]
            if non_negative["scheduler"] is not None:
                self.nn_scheduler = NonNegativityScheduler(total_steps, self.nn_penalty, **non_negative["scheduler"])
            else:
                self.nn_scheduler = None
        else:
            self.nn_enabled = False
        
        # Noise Injector
        if noise_injection is not None:
            self.noise_injector = NoiseInjector(
                model=self.model,
                noise_config=noise_injection["noise_config"],
                noise_std=noise_injection["noise_std"]
            )
            self.noise_schedule = noise_injection["noise_schedule"]
        else:
            self.noise_injector = None

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["train"]:
                self.noise_injector.attach()
            
    def on_train_epoch_end(self):
        if self.nn_enabled and self.live_clip:
            if self.current_epoch % self.clip_interval == 0:
                # Store unclipped state
                self._last_unclipped_state = copy.deepcopy(self.model.state_dict())
                
                # Clip in-place
                for param in self.model.parameters():
                    param.data.clamp_(min=0.0)
            print("train non-negativity:")
            check_non_negativity(self.model, verbose=False)

    def on_validation_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()
                
        # Clip model
        if self.nn_enabled:
            for param in self.model.parameters():
                param.data.clamp_(min=0.0)
        print("validation non-negativity:")
        check_non_negativity(self.model, verbose=False)

    def on_test_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()
        print("test non-negativity:")
        check_non_negativity(self.model, verbose=False)
    
    def on_save_checkpoint(self, checkpoint):
        if self.noise_injector is not None:
            if self.noise_injector._is_attached:
                self.noise_injector.dettach()
        
        if self.nn_enabled and self.live_clip and hasattr(self, '_last_unclipped_state'):
            # Save unclipped model
            path = self.trainer.checkpoint_callback.dirpath
            os.makedirs(path, exist_ok=True)
            unclipped_ckpt = copy.deepcopy(checkpoint)
            for k, v in self._last_unclipped_state.items():
                unclipped_ckpt["state_dict"][f"model.{k}"] = v
            torch.save(unclipped_ckpt, os.path.join(path, "unclipped.ckpt"))


    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)

        train_metrics = {f"train_{k}": v for k, v in metrics.items()}
        train_metrics["train_loss"] = loss

        if self.nn_enabled and self.nn_penalty is not None:
            if self.nn_scheduler is not None: # If there is a scheduler, use it to get L2 & L1 weights
                l2_weight, l1_weight = self.nn_scheduler.get_weights(self.global_step)
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=l2_weight, l1_weight=l1_weight)
            else:
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=self.nn_weight, l1_weight=self.nn_weight)

            loss = loss + self.nn_weight * neg_penalty
            train_metrics["nn_penalty"] = neg_penalty
            train_metrics["train_loss"] = loss  # log penalised loss

        self.log_dict(train_metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        val_metrics["val_loss"] = loss
        self.log_dict(val_metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return val_metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)
        test_metrics = {f"test_{k}": v for k, v in metrics.items()}
        test_metrics["test_loss"] = loss
        self.log_dict(test_metrics, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.nn_enabled and self.trainer.ckpt_path is None:
            # Save original weights before loading unclipped model
            _original_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

            # Load unclipped weights
            state_dict = torch.load(os.path.join(self.trainer.checkpoint_callback.dirpath, "unclipped.ckpt"))["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            loss_u, metrics_u = self._shared_eval_step(batch, batch_idx)
            unclipped_metrics = {f"test_{k}_unclipped": v for k, v in metrics_u.items()}
            unclipped_metrics["test_loss_unclipped"] = loss_u
            self.log_dict(unclipped_metrics, prog_bar=True, on_epoch=True, sync_dist=True)

            # Restore original weights
            self.model.load_state_dict(_original_weights)

        return test_metrics
    
    def _shared_eval_step(self, batch, batch_idx):
        if self.model.task == 'generation':
            x, y = batch # (B, L)
            logits = self.model(x) # (B, L, vocab_size)

            # Flatten for cross_entropy
            logits_flat = logits.reshape(-1, self.model.vocab_size) # (B*L, vocab_size)
            targets_flat = y.reshape(-1) # (B*L)

            loss = self.loss_fn(logits_flat, targets_flat)
            perplexity = torch.exp(loss)

            preds = logits.argmax(dim=-1) # (B, L)
            token_acc = (preds == y).float().mean()

            # Second-half accuracy: the induction head recall signal
            mid = y.shape[1] // 2
            second_half_acc = (preds[:, mid:] == y[:, mid:]).float().mean()

            return loss, {"perplexity": perplexity, "token_acc": token_acc, "second_half_acc": second_half_acc}

        else:
            x, y = batch
            y_hat = self.model(x) # (B, D_out)
            loss = self.loss_fn(y_hat, y)
            acc = accuracy(y_hat, y, task="multiclass", num_classes=self.model.d_out)
            return loss, {"acc": acc}
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **self.opt_hyperparams)
        
        if self.lr_scheduler is None:
            return optimizer
        else:
            scheduler = create_scheduler(optimizer, self.total_steps, **self.lr_scheduler)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }