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

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Applies Sobel x or y, then takes an absolute value and applies a threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def hls_select(img, thresh=(170, 255)):
    """
    Thresholds the S-channel of HLS.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def combined_threshold(image):
    """
    Combines Sobel X and HLS S-channel thresholding.
    """
    gradx = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    s_binary = hls_select(image, thresh=(170, 255))
    
    combined_binary = np.zeros_like(gradx)
    # Combine the two binary thresholds
    combined_binary[(s_binary == 1) | (gradx == 1)] = 255
    return combined_binary
