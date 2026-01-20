import cv2
import numpy as np

def to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian Blur to smooth the image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def resize_image(image, target_size=(1280, 720)):
    """
    Resize image to target size.
    """
    return cv2.resize(image, target_size)
