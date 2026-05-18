import numpy as np

def majority_vote(codes):

    codes = np.array(codes)

    mean_bits = np.mean(codes,axis=0)

    result = (mean_bits > 0.5).astype(int)

    return result