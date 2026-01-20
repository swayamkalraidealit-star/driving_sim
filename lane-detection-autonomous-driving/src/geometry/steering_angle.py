import numpy as np

def calculate_vehicle_offset(frame_width, left_lane, right_lane, eval_y):
    """
    Calculate the offset of the vehicle from the center of the lane.
    Positive value means vehicle is to the right of center.
    Negative value means vehicle is to the left of center.
    """
    if left_lane is None or right_lane is None:
        return 0.0

    # Calculate x positions of lanes at the bottom of the image (eval_y)
    left_slope, left_intercept = left_lane
    right_slope, right_intercept = right_lane
    
    x_left = (eval_y - left_intercept) / left_slope
    x_right = (eval_y - right_intercept) / right_slope
    
    lane_center = (x_left + x_right) / 2.0
    image_center = frame_width / 2.0
    
    offset_pixels = image_center - lane_center
    
    # Return offset in pixels (conversion to meters can be done if XM_PER_PIX is available)
    return offset_pixels

def calculate_steering_angle(offset_pixels, max_steering_angle=25.0, max_offset_pixels=200):
    """
    Calculate steering angle based on offset.
    (Simple Proportional Controller)
    """
    steering_angle = (offset_pixels / max_offset_pixels) * max_steering_angle
    return np.clip(steering_angle, -max_steering_angle, max_steering_angle)
