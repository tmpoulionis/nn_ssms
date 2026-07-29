import lightning as L
from utils.utils import create_scheduler
from utils.quantization import QuantizationManager

class LightningMamba(L.LightningModule):
    def __init__(self, model, total_steps, optimizer, lr_scheduler=None, opt_hyperparams=None, quantization=None, config=None):
        super().__init__()
        self.model = model
        self.total_steps = total_steps
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.opt_hyperparams = opt_hyperparams if opt_hyperparams is not None else {}
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.save_hyperparameters(ignore=['model'])

        # Quantization Aware Training
        self.qat_cfg = quantization
        self.qat_manager = None
        if self.qat_cfg is not None and self.qat_cfg.get("enabled", False):
            self.qat_manager = QuantizationManager(self.model, self.qat_cfg)
            self.qat_manager.set_quant_mode(self.qat_cfg.get("train_quant", True))
            self.qat_manager.enable_observer()
            self.qat_manager.disable_fake_quant()  # calibration phase

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        if self.qat_manager is not None:
            self.qat_manager.set_quant_mode(self.qat_cfg.get("train_quant", True))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # QAT: switch from calibration to fake-quant; optionally freeze observers
        if self.qat_manager is not None:
            if self.global_step == self.qat_cfg.get("calibration_steps", 0):
                self.qat_manager.enable_fake_quant()
            fos = self.qat_cfg.get("freeze_observer_step")
            if fos is not None and self.global_step == fos:
                self.qat_manager.freeze_observers()

    def on_validation_epoch_start(self):
        if self.qat_manager is not None:
            self.qat_manager.set_quant_mode(self.qat_cfg.get("eval_quant", True))

    def on_test_epoch_start(self):
        if self.qat_manager is not None:
            self.qat_manager.set_quant_mode(self.qat_cfg.get("eval_quant", True))

    def on_save_checkpoint(self, checkpoint):
        if self.config is not None:
            checkpoint["experiment_config"] = self.config
        try:
            run_id = getattr(self.logger, "experiment", None) and self.logger.experiment.id
        except Exception:
            run_id = None
        if run_id is not None:
            checkpoint["wandb_run_id"] = run_id

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)

        train_metrics = {f"train_{k}": v for k, v in metrics.items()}
        train_metrics["train_loss"] = loss

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