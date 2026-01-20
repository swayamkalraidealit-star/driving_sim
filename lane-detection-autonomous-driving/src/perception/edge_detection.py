import cv2

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny Edge Detection.
    """
    return cv2.Canny(image, low_threshold, high_threshold)
