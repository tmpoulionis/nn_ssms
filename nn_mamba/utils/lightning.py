import torch
import lightning as L
from torchmetrics.functional import accuracy

class LightningMamba(L.LightningModule):
    def __init__(self, model, optimizer, loss_fn, scheduler_config=None, opt_hyperparams=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_hyperparams = opt_hyperparams if opt_hyperparams is not None else {}
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics
        
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x) # (B, L, D_out)
        y_hat = torch.mean(y_hat, dim=1)  # (B, D_out) pooling over sequence length
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
            