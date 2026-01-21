import cv2
import numpy as np
from src.config import settings
from src.preprocessing import image_utils
from src.perception import edge_detection, roi, perspective_transform, lane_detection
from src.geometry import steering_angle, lane_geometry
from src.tracking import lane_line
from src.visualization import overlay
from src.control.pid_controller import PIDController

class LanePipeline:
    def __init__(self):
        self.settings = settings
        # Initialize LaneLine trackers
        self.left_lane = lane_line.LaneLine(alpha=0.2)
        self.right_lane = lane_line.LaneLine(alpha=0.2)
        
        # Initialize PID Controller
        self.pid_controller = PIDController(
            kp=settings.STEERING_KP,
            ki=settings.STEERING_KI,
            kd=settings.STEERING_KD
        )

    def process_frame(self, frame):
        # 0. Resize
        frame = image_utils.resize_image(frame)

        # 1. Preprocessing & Edge Detection
        # Use combined thresholding (HLS S-Channel + Sobel X) instead of simple Grayscale + Canny
        edges = image_utils.combined_threshold(frame)

        # 3. ROI Masking
        masked_edges = roi.region_of_interest(edges, [self.settings.ROI_VERTICES])

        # 4. Perspective Transform (Bird's Eye)
        warped_edges, M = perspective_transform.birdeye(masked_edges, self.settings.SRC_POINTS, self.settings.DST_POINTS)
        Minv = np.linalg.inv(M)

        # 5. Lane Detection (Polynomial Fit)
        # Check if we have a valid prior track to search around
        left_fit_prior = self.left_lane.get_fit()
        right_fit_prior = self.right_lane.get_fit()

        if left_fit_prior is not None and right_fit_prior is not None:
             # Search around previous detection
             left_fit, right_fit, _, _ = lane_detection.search_around_poly(warped_edges, left_fit_prior, right_fit_prior)
        else:
             # Full sliding window search
             left_fit, right_fit, _, _ = lane_detection.fit_polynomial(warped_edges)
        
        # SANITY CHECK
        if self.validate_lines(left_fit, right_fit):
            self.left_lane.update(left_fit)
            self.right_lane.update(right_fit)
        else:
            # If invalid, we ignore ANY new detection and stick to the history
            # (LaneLine.update is NOT called, so it keeps old state or decays confidence)
            # Or we can explicitly tell it "dropped frame"
            self.left_lane.update(None)
            self.right_lane.update(None)
        
        # Get smoothed fits for visualization and geometry
        best_left = self.left_lane.get_fit()
        best_right = self.right_lane.get_fit()

        # 6. Geometry & Steering
        # Calculate offset in pixels at the bottom of the image
        offset = steering_angle.calculate_vehicle_offset(frame.shape[1], best_left, best_right, frame.shape[0], self.settings.XM_PER_PIX)
        
        # PID Control Update
        # Assuming 30 FPS, dt = 1/30 approx 0.033
        st_angle = self.pid_controller.update(offset, dt=0.033)
        
        # Calculate Curvature
        left_curverad, right_curverad = lane_geometry.measure_curvature_real(best_left, best_right, frame.shape[0], self.settings.XM_PER_PIX, self.settings.YM_PER_PIX)
        avg_curvature = (left_curverad + right_curverad) / 2
        
        # Confidence logic
        if self.left_lane.lost_count > 0 or self.right_lane.lost_count > 0:
            confidence_text = "Tracking (Coast)"
            color = (0, 165, 255) # Orange
        else:
            confidence_text = "High Confidence"
            color = (0, 255, 0)   # Green

        # 7. Visualization
        # Draw the filled lane area using the smoothed fits
        result = overlay.draw_lane_area(frame, warped_edges, best_left, best_right, Minv)
        
        # Add Text with curvature
        result = overlay.draw_info(result, st_angle, offset, avg_curvature, confidence_text, color)
        
        return result

    def validate_lines(self, left_fit, right_fit):
        """
        Checks if the detected lines are valid lanes.
        """
        if left_fit is None or right_fit is None:
            return False

        # Check 1: Calculate Lane Width at bottom and middle
        # We expect around 700 pixels (roughly) based on settings.
        # Let's say valid is 500 to 900.
        ploty = np.linspace(0, 719, 720)
        
        # Bottom
        left_x_bottom = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
        right_x_bottom = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2]
        lane_width_bottom = right_x_bottom - left_x_bottom

        # Middle
        left_x_mid = left_fit[0]*360**2 + left_fit[1]*360 + left_fit[2]
        right_x_mid = right_fit[0]*360**2 + right_fit[1]*360 + right_fit[2]
        lane_width_mid = right_x_mid - left_x_mid

        if not (500 < lane_width_bottom < 900):
            return False
        if not (500 < lane_width_mid < 900):
            return False

        # Check 2: Parallelism (similar slopes)
        # derivative = 2*A*y + B
        # Evaluate slope at middle
        left_slope = 2*left_fit[0]*360 + left_fit[1]
        right_slope = 2*right_fit[0]*360 + right_fit[1]
        
        if abs(left_slope - right_slope) > 0.5: # Tune this threshold
            return False
            
        return True
