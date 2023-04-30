import torch as th
from torch.distributions.utils import _standard_normal, broadcast_all

class GradNormal(th.distributions.Normal):
    
    def __init__(self, loc, scale, validate_args=None):
        
        super().__init__(loc, scale, validate_args)
    
    def sample_eps(self, sample_shape=th.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        
        return eps
    
    def eps_to_action(self, eps):
        
        return self.loc + eps * self.scale