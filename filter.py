import torch as t
from torch import nn
from typing import Tuple
from kf import KF
from mcf import MCF
from mcckf import MCCKF

class Filter(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int
    ):
        super().__init__()
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._F = None
        self._B = None
        self._H = None
        self._Q = None
        self._R = None
        self._P_0 = None
        self._x_0 = None
        self._sigma = None
        
    @property
    def x(self):
        return self._x
    
    @property
    def F(self):
        return self._F
    @F.setter
    def F(self, new_F:t.Tensor):
        if(new_F.shape == (self._dim_x, self._dim_x)):
            self._F = new_F
        else: 
            raise ValueError(f"Incorrect dimension for F {new_F.shape}")
    @property
    def B(self):
        return self._B
    @B.setter
    def B(self, new_B:t.Tensor):
        if(new_B.ndim  == 1 and new_B.shape[0] == self._dim_x):
            self._B = new_B
        else: 
            raise ValueError(f"Incorrect dimension for B {new_B.shape}")

    @property
    def Q(self):
        return self._Q
    @Q.setter
    def Q(self, new_Q:t.Tensor):
        if(new_Q.shape == (self._dim_x, self._dim_x)):
            self._Q = new_Q
        else: 
            raise ValueError(f"Incorrect dimension for Q {new_Q.shape}")
    
    @property
    def H(self):
        return self._H
    @H.setter
    def H(self, new_H:t.Tensor):
        if(new_H.shape == (self._dim_y, self._dim_x)):
            self._H = new_H
        else: 
            raise ValueError(f"Incorrect dimension for H {new_H.shape}")
    
    @property
    def R(self):
        return self._R
    @R.setter
    def R(self, new_R:t.Tensor):
        if(new_R.shape == (self._dim_y, self._dim_y)):
            self._R = new_R
        else: 
            raise ValueError(f"Incorrect dimension for R {new_R.shape}")
    
    @property
    def P_0(self):
        return self._P_0
    @P_0.setter
    def P_0(self, new_P_0:t.Tensor):
        if(new_P_0.shape == (self._dim_x, self._dim_x)):
            self._P_0 = new_P_0
        else: 
            raise ValueError(f"Incorrect dimension for P_0 {new_P_0.shape}")
    
    @property    
    def x_0(self):
        return self._x_0
    @x_0.setter
    def x_0(self, new_x_0):
        self._x_0 = new_x_0
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        if(new_sigma > 0):
            self._sigma = new_sigma
        else: 
            raise ValueError(f"Non-positive value for sigma {new_sigma}")

    def _init_values(self) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, float]:
        
        if self.F is None:
            raise ValueError(f"The matrix F is not declared!")
        
        if self.B is None:
            self.B = t.zeros(self._dim_x, dtype=t.float64)
        
        if self.H is None:
            raise ValueError(f"The matrix H is not declared!")
        
        if self.Q is None:
            self.Q = t.eye(self._dim_x, dtype=t.float64)
            
        if self.R is None:
            self.R = t.eye(self._dim_y, dtype=t.float64)
        
        if self.P_0 is None:
            self.P_0 = t.eye(self._dim_x, dtype=t.float64)
    
        if self.x_0 is not None:
            self.x[:, 0] = self.x_0
            
        if self.sigma is None:
            self.sigma = 1

        return self.F, self.B, self.H, self.Q, self.R, self.P_0, self.sigma
        
    def forward(self, y:t.Tensor, filter:str) -> t.Tensor:
        with t.no_grad():
            params = self._init_values()

            if filter == "KF":
                self._x = KF(y, self._dim_x, params) 
            elif filter == "MCF":
                self._x = MCF(y, self._dim_x, params)
            elif filter == "MCCKF":
                self._x = MCCKF(y, self._dim_x, params)
            else:
                raise ValueError(f"Unknown method!")

            return self._x
