import cv2
import numpy as np

def birdeye(image, src, dst):
    """
    Apply perspective transform to get a bird's eye view.
    """
    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

def inverse_birdeye(image, src, dst):
    """
    Apply inverse perspective transform to map back to original view.
    """
    img_size = (image.shape[1], image.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(image, Minv, img_size, flags=cv2.INTER_LINEAR)
    return unwarped, Minv
