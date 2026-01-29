import os
import torch
import copy
import lightning as L
from torchmetrics.functional import accuracy
from utils.utils import create_scheduler
from utils.noise_injection import NoiseInjector
from utils.non_negativity import compute_negative_penalty, NonNegativityScheduler

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
            
    def on_validation_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()

    def on_test_epoch_start(self):
        if self.noise_injector is not None:
            if self.noise_schedule["eval"]:
                self.noise_injector.attach()
            else:
                self.noise_injector.dettach()
    
    def on_save_checkpoint(self, checkpoint):
        if self.noise_injector is not None:
            if self.noise_injector._is_attached:
                self.noise_injector.dettach()
        
        if self.nn_enabled:
            path = self.trainer.checkpoint_callback.dirpath
            os.makedirs(path, exist_ok=True)
            
            # Save unclipped version
            unclipped_path = os.path.join(path, "unclipped.ckpt")
            unclipped = copy.deepcopy(checkpoint)
            torch.save(unclipped, unclipped_path)
            
            # Clip negative weights
            for name, param in checkpoint["state_dict"].items():
                checkpoint["state_dict"][name] = torch.clamp(param, min=0.0) # Lightning will save the clipped version automatically
            
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        
        if self.nn_enabled and self.nn_penalty is not None:
            
            if self.nn_scheduler is not None: # If there is a scheduler, use it to get L2 & L1 weights
                l2_weight, l1_weight = self.nn_scheduler.get_weights(self.global_step)
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=l2_weight, l1_weight=l1_weight)
            else:
                neg_penalty = compute_negative_penalty(self.model, penalty_type=self.nn_penalty, l2_weight=self.nn_weight, l1_weight=self.nn_weight)
                
            loss = loss + self.nn_weight * neg_penalty
            metrics = {'nn_penalty': neg_penalty, 'train_loss': loss, 'train_acc': acc}
            self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            metrics = {'train_loss': loss, 'train_acc': acc}
            self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        if self.model.task == 'generation':
            metrics = {"val_loss": loss, 'val_per': acc}
        else:
            metrics = {'val_loss': loss, 'val_acc': acc}
            
        self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics
        
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        if self.model.task == 'generation':
            metrics = {"test_loss": loss, 'test_per': acc}
        else:
            metrics = {'test_loss': loss, 'test_acc': acc}
            
        self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        
        if self.nn_enabled:
            # Load unclipped weights
            state_dict = torch.load(os.path.join(self.trainer.checkpoint_callback.dirpath, "unclipped.ckpt"))["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            loss, acc = self._shared_eval_step(batch, batch_idx)
            metrics = {'test_loss_unclipped': loss, 'test_acc_unclipped': acc}
            self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return metrics
    
    def _shared_eval_step(self, batch, batch_idx):
        if self.model.task == 'generation':
            x, y = batch # (B, L)
            logits = self.model(x) # (B, L, vocab_size)
            
            # Flatten for cross_entropy
            logits_flat = logits.reshape(-1, self.model.vocab_size) # (B*L, vocab_size)
            targets_flat = y.reshape(-1) # (B*L)
            
            loss = self.loss_fn(logits_flat, targets_flat)
            perplexity = torch.exp(loss)
            return loss, perplexity
        
        else:
            x, y = batch
            y_hat = self.model(x) # (B, D_out)
            loss = self.loss_fn(y_hat, y)
            acc = accuracy(y_hat, y, task="multiclass", num_classes=self.model.d_out)
            return loss, acc
    
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