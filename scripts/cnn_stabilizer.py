import cv2
import numpy as np

def cnn_stabilize(img):

    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    enhanced = cv2.filter2D(img,-1,kernel)

    enhanced = cv2.normalize(enhanced,None,0,255,cv2.NORM_MINMAX)

    return enhanced