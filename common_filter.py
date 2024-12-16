import torch as t

def inv(M):
    return t.linalg.inv(M)

def G_sigma(dist, sigma):
    tmp =  t.exp(-dist**2 / (2 * sigma**2))
    # print(f"G = {tmp}")
    return tmp

def norm(left, M, right): 
    vec = left - M @ right
    tmp =  t.sqrt(vec @ vec)
    return tmp

def cov_norm(left, M, right, COV): 
    vec = left - M @ right
    tmp =  t.sqrt(vec @ COV @ vec)
    # print(f"mah_norm = {tmp}")
    return tmp