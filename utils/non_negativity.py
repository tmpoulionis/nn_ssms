import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNegativityHooks:
    def __init__(
        self,
        model,
        layer_types=(nn.Linear, nn.Conv1d, nn.LayerNorm)
    ):
        self.model = model
        self.layer_types = layer_types
        self.hooks = []
        
    def clip_hook(self, module, input):
        with torch.no_grad():
            module.weight.data.clamp_(min=0.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.clamp_(min=0.0)
                
    def register(self):
        for module in self.model.modules():
            if isinstance(module, self.layer_types):
                hook = module.register_forward_pre_hook(self.clip_hook)
                self.hooks.append(hook)
    
    def detach(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
            
    