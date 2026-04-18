import os
import torch
import copy
import lightning as L
from utils.utils import create_scheduler
from utils.noise_injection import NoiseInjector
from utils.non_negativity import compute_negative_penalty, NonNegativityScheduler, check_non_negativity

class LightningMamba(L.LightningModule):
    def __init__(self, model, total_steps, optimizer, lr_scheduler=None, opt_hyperparams=None, noise_injection=None, non_negative=None, config=None):
        super().__init__()
        self.saved_weights = {}
        self.model = model
        self.total_steps = total_steps
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.opt_hyperparams = opt_hyperparams if opt_hyperparams is not None else {}
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        # Non-Negativity
        if non_negative is not None:
            self.nn_enabled = non_negative["enabled"]
            self.nn_penalty = non_negative["penalty_type"]
            self.nn_weight = non_negative["penalty_weight"]
            self.live_clip = non_negative["live_clipping"]
            self.clip_mode = non_negative["clip_mode"]
            self.clip_interval = non_negative["clip_interval"]
            self.exclude_bias = non_negative["exclude_bias"]
            self._pre_val_params = None
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Non-negativity
        if self.nn_enabled and self.live_clip and self.clip_mode == "step" and (self.global_step % self.clip_interval == 0):
            self.clip_weights(exclude_biases=self.exclude_bias)

    def on_validation_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()
                
        # Non-negativity
        if self.nn_enabled:
            if not self.live_clip:
                # Save unclipped state to restore after validation
                self._pre_val_params = copy.deepcopy(self.model.state_dict())
            self.clip_weights(exclude_biases=self.exclude_bias)

    def on_validation_epoch_end(self):
        # Restore unclipped weights if not live clipping
        if self.nn_enabled and not self.live_clip and self._pre_val_params is not None:
            self.model.load_state_dict(self._pre_val_params)
            self._pre_val_params = None

    def on_test_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()
        
        # Non-negativity
        if self.nn_enabled:
            if not self.live_clip:
                self._unclipped_state = copy.deepcopy(self.model.state_dict())
            self.clip_weights(exclude_biases=self.exclude_bias)
            
            if not self.live_clip:
                self._clipped_state = copy.deepcopy(self.model.state_dict())
                self._clipped_path = os.path.join(self.trainer.checkpoint_callback.dirpath, "clipped_model.ckpt")
                torch.save(self._clipped_state, self._clipped_path)
    
    def on_save_checkpoint(self, checkpoint):
        if self.config is not None:
            checkpoint["experiment_config"] = self.config
        if self.noise_injector is not None:
            if self.noise_injector._is_attached:
                self.noise_injector.dettach()

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)

        train_metrics = {f"train_{k}": v for k, v in metrics.items()}
        train_metrics["train_loss"] = loss

        if self.nn_enabled and self.nn_penalty is not None:
            if self.nn_scheduler is not None: # If there is a scheduler, use it to get L2 & L1 weights
                l2_weight, l1_weight = self.nn_scheduler.get_weights(self.global_step)
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=l2_weight, l1_weight=l1_weight, exclude_biases=self.exclude_bias)
            else:
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=1.0, l1_weight=1.0, exclude_biases=self.exclude_bias)

            nn_loss_contribution = self.nn_weight * neg_penalty
            loss = loss + nn_loss_contribution
            train_metrics["nn_penalty"] = nn_loss_contribution
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

        if self.nn_enabled and not self.live_clip:
            # Load unclipped weights
            self.model.load_state_dict(self._unclipped_state)
            loss_u, metrics_u = self._shared_eval_step(batch, batch_idx)
            unclipped_metrics = {f"test_{k}_unclipped": v for k, v in metrics_u.items()}
            unclipped_metrics["test_loss_unclipped"] = loss_u
            self.log_dict(unclipped_metrics, prog_bar=True, on_epoch=True, sync_dist=True)

            # Restore original weights
            self.model.load_state_dict(self._clipped_state)

        return test_metrics
    
    # Utility functions
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.model.compute_loss(logits, y)
        metrics = self.model.compute_metrics(logits, y)
        return loss, metrics
    
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
            
    def clip_weights(self, exclude_biases = False):
        for name, param in self.model.named_parameters():
            if exclude_biases and name.endswith('dt_proj.bias'):
                continue
            param.data.clamp_(min=0.0)