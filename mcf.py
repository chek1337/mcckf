import torch as t
from common_filter import *

def MCF(y:t.Tensor, dim_x, params):
    F, B, H, Q, R, P_k1_k1, sigma = params
    
    T = y.shape[1]
    I = t.eye(dim_x, dtype=t.float64)

    x = t.zeros((dim_x, T), dtype=t.float64)
    u_t = t.ones(dim_x, dtype=t.float64)

    for k in range(1, T):
        # Prior estimation
        x_k_pr = F @ x[:, k-1] + B @ u_t
        P_k_k1 = F @ P_k1_k1 @ F.T + Q
        
        # Posterior estimation
        L = G_sigma(norm(y[:, k], H, x_k_pr), sigma)
        K = inv(I + L * H.T @ H) * L @ H.T
        x[:, k] = x_k_pr + K @ (y[:, k] - H @ x_k_pr)
        P_k_k = (I - K @ H) @ P_k_k1 @ (I - K @ H).T + K @ R @ K.T
        
        P_k1_k1 = P_k_k

    return x


