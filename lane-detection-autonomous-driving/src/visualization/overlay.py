import cv2
import numpy as np
from src.perception.lane_detection import pixel_points

def draw_lane_lines(image, left_line, right_line, color=(0, 255, 0), thickness=10):
    """
    Draw the detected lane lines on the image.
    """
    line_image = np.zeros_like(image)
    height, width, _ = image.shape
    
    if left_line is not None:
        p1, p2 = pixel_points(height, height*0.6, left_line)
        cv2.line(line_image, p1, p2, color, thickness)
        
    if right_line is not None:
         p1, p2 = pixel_points(height, height*0.6, right_line)
         cv2.line(line_image, p1, p2, color, thickness)
         
    return cv2.addWeighted(image, 1, line_image, 1, 0)

def draw_info(image, steering_angle, offset):
    """
    Draw steering angle and offset information on the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Steering Angle: {steering_angle:.2f} deg", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Offset: {offset:.2f} px", (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image
