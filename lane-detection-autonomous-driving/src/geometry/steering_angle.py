import numpy as np

def calculate_vehicle_offset(frame_width, left_fit, right_fit, eval_y, xm_per_pix=3.7/700):
    """
    Calculate the offset of the vehicle from the center of the lane.
    Positive value means vehicle is to the right of center.
    Negative value means vehicle is to the left of center.
    Using polynomial evaluation x = Ay^2 + By + C
    Returns offset in METERS.
    """
    if left_fit is None or right_fit is None:
        return 0.0

    # Calculate x positions of lanes at the bottom of the image (eval_y)
    # x = Ay^2 + By + C
    
    x_left = left_fit[0]*eval_y**2 + left_fit[1]*eval_y + left_fit[2]
    x_right = right_fit[0]*eval_y**2 + right_fit[1]*eval_y + right_fit[2]
    
    lane_center = (x_left + x_right) / 2.0
    image_center = frame_width / 2.0
    
    offset_pixels = image_center - lane_center
    offset_meters = offset_pixels * xm_per_pix
    
    return offset_meters

def calculate_steering_angle(offset_meters, max_steering_angle=25.0, max_offset_meters=1.0):
    """
    Calculate steering angle based on offset in METERS.
    (Simple Proportional Controller)
    """
    steering_angle = (offset_meters / max_offset_meters) * max_steering_angle
    return np.clip(steering_angle, -max_steering_angle, max_steering_angle)
