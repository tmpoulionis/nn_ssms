from torch import nn
import torch

class NoiseInjector(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        noise_config: dict,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.noise_config = noise_config
        self.noise_std = noise_std
    
    def _make_pre_hook(self, name: str):
        def pre_hook(module, x):
            if self.noise_config["input"]:
                x = x + torch.randn_like(x) * self.noise_std
            
            if self.noise_config["weight"] and hasattr(module, 'weight'):
                module.weight.data = module.weight.data + torch.randn_like(module.weight.data) * self.noise_std
            
            if self.noise_config["bias"] and hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                module.bias.data = module.bias.data + torch.randn_like(module.bias.data) * self.noise_std
                
            return x
        return pre_hook
    
    def _make_post_hook(self, name: str):
        def post_hook(module, input, output):
            if self.noise_config["output"]:
                output = output + torch.randn_like(output) * self.noise_std
            return output
        return post_hook
     
    def attach(self):
        for name, module in self.model.named_modules():
            module.register_forward_pre_hook(self._make_pre_hook(name))
            module.register_forward_hook(self._make_post_hook(name))
    
    def detach(self):
        for _, module in self.model.named_modules():
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()
    
    