import cv2
import numpy as np

def detect_lane_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Detect lane lines using Probabilistic Hough Transform.
    """
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines = [] # (slope, intercept)
    left_weights = [] # (length,)
    right_lines = [] # (slope, intercept)
    right_weights = [] # (length,)

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: # Vertical lines
                continue
            
            # Avoid division by zero
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out horizontal-ish lines (slope close to 0)
            # In bird's eye view, lanes should be vertical (high slope)
            if abs(slope) < 0.5:
                continue

            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if slope < 0: # Left lane
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else: # Right lane
                right_lines.append((slope, intercept))
                right_weights.append(length)

    # Add more weights to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    """
    if line is None:
        return None
    slope, intercept = line
    
    # Prevent division by zero for horizontal lines
    if abs(slope) < 1e-3:
        return None
        
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except OverflowError:
        return None
        
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
