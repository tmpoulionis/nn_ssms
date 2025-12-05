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
        
        self._hooks = []
        self._is_attached = False
        self._original_weights = {}
        self._original_biases = {}
        
    def _make_pre_hook(self, name: str):
        def pre_hook(module, x):
            # Inject noise into input, weights, and biases
            if self.noise_config["input"]:
                x = tuple(
                    i + torch.randn_like(i) * self.noise_std
                    if isinstance(i, torch.Tensor)
                    else i
                    for i in x
                )
            
            if self.noise_config["weight"] and hasattr(module, 'weight'):
                self._original_weights[name] = module.weight.data.clone() # Store original weights and biases
                module.weight.data = module.weight.data + torch.randn_like(module.weight.data) * self.noise_std
            
            if self.noise_config["bias"] and hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                self._original_biases[name] = module.bias.data.clone()
                module.bias.data = module.bias.data + torch.randn_like(module.bias.data) * self.noise_std
                
                
            return x
        return pre_hook
    
    def _make_post_hook(self, name: str):
        def post_hook(module, input, output):
            # Restore original weights and biases
            if self.noise_config["weight"] and hasattr(module, 'weight'):
                module.weight.data = self._original_weights[name]
            if self.noise_config["bias"] and hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                module.bias.data = self._original_biases[name]
            
            # Inject noise into output
            if self.noise_config["output"]:
                output = output + torch.randn_like(output) * self.noise_std
            return output
        return post_hook
     
    def attach(self):
        if self._is_attached:
            return self
        
        for name, module in self.model.named_modules():
            if isinstance(module,  (nn.Linear, nn.Conv1d)):
                pre_handle = module.register_forward_pre_hook(self._make_pre_hook(name))
                post_handle = module.register_forward_hook(self._make_post_hook(name))
                self._hooks.append(pre_handle)
                self._hooks.append(post_handle)
                
        self._is_attached = True
        return self
    
    def dettach(self):
        for handle in self._hooks:
            handle.remove()
            
        self._hooks = []
        self._original_weights = {}
        self._original_biases = {}
        self._is_attached = False
        return self