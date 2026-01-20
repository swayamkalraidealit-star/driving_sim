import cv2
import numpy as np

def draw_info(image, steering_angle, offset, curvature=None, confidence_text=None, color=(0, 255, 0)):
    """
    Draw steering angle, offset, and curvature information on the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Steering Angle: {steering_angle:.2f} deg", (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Offset: {offset:.2f} m", (50, 100), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    if curvature:
        cv2.putText(image, f"Radius of Curvature: {curvature:.0f} m", (50, 150), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    if confidence_text:
        cv2.putText(image, f"Status: {confidence_text}", (50, 200), font, 1.2, color, 2, cv2.LINE_AA)
    
    return image

def draw_lane_area(image, binary_warped, left_fit, right_fit, Minv):
    """
    Draw the lane area (filled polygon) back onto the original image.
    """
    if left_fit is None or right_fit is None:
        return image

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        return image

    # Create an image to draw the lanes on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return result
