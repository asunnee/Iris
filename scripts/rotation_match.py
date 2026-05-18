import numpy as np

def hamming_distance(a, b):
    """
    计算两个IrisCode的HD距离
    """
    return np.sum(a != b) / len(a)


def circular_shift(code, shift):
    """
    虹膜旋转补偿
    """
    return np.roll(code, shift)


def best_rotation_match(code1, code2, max_shift=10):
    """
    在旋转范围内寻找最小HD
    """

    best_hd = 1.0

    for shift in range(-max_shift, max_shift+1):

        shifted = circular_shift(code2, shift)

        hd = hamming_distance(code1, shifted)

        if hd < best_hd:
            best_hd = hd

    return best_hd