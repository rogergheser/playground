from typing import Callable, Optional
from torch import Tensor
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: Optional[int],
        output_dim: Optional[int],
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim or input_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.activation = act_layer()
        self.fc2 = nn.Linear(self.hidden_dim, output_dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x