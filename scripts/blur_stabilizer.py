import cv2

def blur_stabilize(image):

    # 高斯平滑
    blur = cv2.GaussianBlur(image,(5,5),0)

    # 对比度增强
    blur = cv2.equalizeHist(blur)

    return blur