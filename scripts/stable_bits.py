import numpy as np

def select_stable_bits(codes, threshold=0.85):
    codes = np.array(codes)
    bit_mean = np.mean(codes, axis=0)
    stable_mask = (bit_mean > threshold) | (bit_mean < 1 - threshold)
    stable_codes = codes[:, stable_mask]
    return stable_codes, stable_mask