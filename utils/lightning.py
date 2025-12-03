import torch
import lightning as L
from torchmetrics.functional import accuracy
from utils.noise_injection import NoiseInjector

class LightningMamba(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn, scheduler_config=None, opt_hyperparams=None, noise_injector_config=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_hyperparams = opt_hyperparams if opt_hyperparams is not None else {}
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        self.noise_injector = noise_injector_config["injector"]
        self.noise_schedule = noise_injector_config["schedule"]
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        if self.noise_schedule["train"] and self.noise_injector is not None:
            self.noise_injector.attach()

    def on_validation_epoch_start(self):
        if self.noise_injector is None:
            return 
        
        if self.noise_schedule["eval"]:
            self.noise_injector.attach()
        else:
            self.noise_injector.dettach()
            
    def on_test_epoch_start(self):
        if self.noise_injector is None:
            return 
        
        if self.noise_schedule["eval"]:
            self.noise_injector.attach()
        else:
            self.noise_injector.dettach()
    
    def on_save_checkpoint(self, checkpoint):
        if self.noise_injector is not None and self.noise_injector._is_attached:
            self.noise_injector.dettach()
            
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        if self.model.task == 'generation':
            metrics = {"train_loss": loss, 'train_per': acc}
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
        
        if self.scheduler_config is None:
            return optimizer
        else:
            scheduler = self.scheduler_config["scheduler"](
                optimizer,
                **self.scheduler_config["params"]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
    
    # # Debugging
    # def on_fit_start(self):
    #     def hook(module, input, output):
    #         print(f"[HOOK] Activation {module} input mean = ({input[0].mean().item():.2f})")
        
    #     self.model.mamba.act.register_forward_hook(hook)