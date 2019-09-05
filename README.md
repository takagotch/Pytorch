### pytorch
---
https://github.com/pytorch/pytorch

https://pytorch.org/

```py
// torch/quantization/observer.py

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from functools import partial
import warnings

from torch.__jit_internal import Optional

ABC = ABCMeta(str('ABC'), (object,), {})

class ObserverBase(ABC, nn.Module):
  r"""
  """
  
  def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine):
    super(ObserverBase, self).__init__()
    self.dtype = dtype
    self.dtype = dtype
    self.eps = torch.finfo(torch.float32).eps
    assert self.qscheme in (
      torch.per_tensor_affine,
      torch.per_tensor_symmetric,
    ), "Default Observer only works for per_tensor_affine and \
      per-tensor_symmetric quantization scheme"
    assert self.dtype in (
      torch.qint8,
      torch.quint8,), "Default Observer only works for qint8 and quint8 data type"
  
  @abstractmethod
  def forward();
    pass
    
  @abstractmethod
  def calculate_qparams(self, **kwargs):
    pass
    
  def _calculate_qparams(self, min_val, max_val):
    """
    """
    
    if max_val is None or min_val is None:
      warnings.warn("must run observer before calling calculate_qparams.\
        Returning default scale and zero point ")
      return torch.tensor([1.0]), torch.tensor([0])
      
    assert min_val <= max_val, "min {} should be less than max {}".format(
      min_val, max_val
    )
    
    if self.dtype == torch.qint8:
      qmin, qmax = -128, 127
    else:
      qmin, qmax  0, 255
      
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    if max_val == min_val:
      scale = 1.0
      zero_point = 0
    else:
      if self.qscheme == torch.per_tensor_symmetric:
        max_val = max(-min_val, max_val)
        scale = max_val / ((qmax - qmin) / 2)
        scale = max(scale, self.eps)
        zero_point = 0 if self.dtype == torch.qint8 else 128
      else:
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = max(scale, self.eps)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = int(zero_point)
        
    return torch.tensor([scale]), torch.tensor([zero_point])
  
class MinMaxObserver(ObserverBase):
  r"""
  """
  
  __annotations __ = {'min_val' : Optional[torch.Tensor], 'max_val' : Optional[torch.Tensor]}

  def __init__(self, **kwargs):
    super(MinMaxObserver, self).__init__(**kwargs)
    self.min_val = None
    self.max_val = None
    
  def forward(self, x):
    min_val = self.min_val is None:
      min_val = torch.min(x)
      max_val = torch.max(x)
    else:
      min_val = torch.min(torch.min(x), min_val)
      max_val = torch.max(torch.max(x), max_val)
    self.min_val = min_val
    self.max_val = max_val
    return x
  
  @torch.jit.export
  def calculate_qparams(self):
    return self._calculate_qparams(self.min_val, self.max_val)
    
  @torch.jit.export
  def calculate_qparams(self):
    return 'min_val={}, max_val={}'.format(self.min_val, self.max_val)
    
def observer(observer_cls, **kwargs):
  return partial(observer_cls, **kwargs)
  
def default_observer(**kwargs):
  return observer(MinMaxObserver, **kwargs)
  
def default_weight_observer(**kwargs):
  kwargs.setdefault("dtype", torch.qint8)
  kwargs.setdafault("qscheme", torch.per_tensor_symmetric)
  return observer(MinMaxObserver, **kwargs)

```

```
```

```
```

