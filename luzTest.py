import numpy as np

def ceil(x,d):
    print(f'x: {x}, d: {d}, frac:{x/d}')
    return np.ceil(x/d)*d

def floor(x,d):
    print(f'x: {x}, d: {d}, frac:{x/d}')
    return np.floor(x/d)*d