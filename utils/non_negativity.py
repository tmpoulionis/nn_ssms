import torch
import torch.nn as nn

class NonNegativityScheduler:
    def __init__(self,
        total_steps,
        l2_weight_start,
        l2_weight_end,
        delay, # Delayed non-negative penalty for the initial training fraction 
        warmup # Warm up on penalty weights to not destroy the initial representation
    ):
        self.total_steps = total_steps
        self.l2_weight_start = l2_weight_start
        self.l2_weight_end = l2_weight_end
        self.delay_steps = int(delay*total_steps)
        self.warmup_steps = int(warmup*total_steps)
        self.penalty_steps = self.total_steps - self.delay_steps - self.warmup_steps
        
    def get_weights(self, current_step): # Weights a, b, c --> c(aL2 + bL1)
        a = self.l2_weight_start
        b = (1 - self.l2_weight_start)
        
        if current_step < self.delay_steps: # Don't apply penalty for the first delay_steps
            return 0, 0
        
        if current_step < self.delay_steps + self.warmup_steps: # After delay_steps, gradually increase weights over warmup_steps for stability
            c = (current_step - self.delay_steps)/self.warmup_steps
            return c*a, c*b
    
        # Gradualy decrease L2 weight and increase L1 weight over penalty_steps
        current_penalty_step = current_step - self.delay_steps - self.warmup_steps
        step = min(current_penalty_step/self.penalty_steps, 1)
        frac = self.l2_weight_start - (self.l2_weight_start - self.l2_weight_end)*step
        a = frac
        b = (1-frac)
        return a, b
        
def compute_negative_penalty(model, penalty_type='l2', l2_weight=0, l1_weight=0):
    penalty = 0
    
    for name, param in model.named_parameters():
        if penalty_type == 'l1':
            negative_val = torch.clamp(param, max=0)
            penalty = penalty + torch.sum(torch.abs(negative_val))

        elif penalty_type == 'l2':
            negative_val = torch.clamp(param, max=0)
            penalty = penalty + torch.sum(negative_val**2)
        
        elif penalty_type == 'elastic':
            negative_val = torch.clamp(param, max=0)
            penalty = penalty + l1_weight*torch.sum(torch.abs(negative_val)) + l2_weight*torch.sum(negative_val**2)
    
    return penalty

def check_non_negativity(model, verbose=True):
    results={}
    total_params = 0
    total_negative = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        num_negative = (param.data < 0).sum().item()
        total_params += num_params
        total_negative += num_negative
        min = param.data.min().item()
        max = param.data.max().item()
        
        results[name] = {
            'total_params': num_params,
            'negative_params': num_negative,
            'ratio': num_negative / num_params,
            'min': min,
            'max': max
        }
        
        if verbose and num_negative > 0:
            print(f"❌ Parameter '{name}")
            print(f"\t {num_negative}/{num_params} negative parameters.")
            print(f"\t min: {min}, max: {max}")
        elif verbose and num_negative == 0:
            print(f"✔️ Parameter '{name}' has no negative values. ({num_params})")
            
    print("Overall Negative Weights Summary:")
    print(f"\t {total_negative}/{total_params} negative parameters.")
    print(f"\t Overall Ratio: {total_negative / total_params}")
    print(f"Negative Weights found in:")
    for name, stats in results.items():
        if stats['negative_params'] > 0:
            print(f" - {name}")
        
    return results