import cv2
import numpy as np
from src.config import settings
from src.preprocessing import image_utils
from src.perception import edge_detection, roi, perspective_transform, lane_detection
from src.geometry import steering_angle
from src.visualization import overlay

class LanePipeline:
    def __init__(self):
        self.settings = settings

    def process_frame(self, frame):
        # 0. Resize
        frame = image_utils.resize_image(frame)

        # 1. Preprocessing
        gray = image_utils.to_grayscale(frame)
        blur = image_utils.apply_gaussian_blur(gray, self.settings.GAUSSIAN_KERNEL_SIZE)

        # 2. Edge Detection
        edges = edge_detection.detect_edges(blur, self.settings.CANNY_LOW_THRESHOLD, self.settings.CANNY_HIGH_THRESHOLD)

        # 3. ROI Masking
        masked_edges = roi.region_of_interest(edges, [self.settings.ROI_VERTICES])

        # 4. Perspective Transform (Bird's Eye)
        # Note: The user requested Perspective Transform BEFORE Hough Transform.
        # This is strictly followed here.
        warped_edges, M = perspective_transform.birdeye(masked_edges, self.settings.SRC_POINTS, self.settings.DST_POINTS)

        # 5. Lane Detection (Hough Transform on Warped Edges)
        lines = lane_detection.detect_lane_lines(warped_edges, 
                                                 self.settings.HOUGH_RHO, 
                                                 self.settings.HOUGH_THETA, 
                                                 self.settings.HOUGH_THRESHOLD, 
                                                 self.settings.HOUGH_MIN_LINE_LEN, 
                                                 self.settings.HOUGH_MAX_LINE_GAP)
        
        # Average lines to get left and right lane
        left_line, right_line = lane_detection.average_slope_intercept(lines)

        # 6. Geometry & Steering
        # Calculate offset in bird's eye view
        offset = steering_angle.calculate_vehicle_offset(frame.shape[1], left_line, right_line, frame.shape[0])
        st_angle = steering_angle.calculate_steering_angle(offset)

        # 7. Visualization
        # We need to draw the lines. Since we detected them in Warped Space, we ideally should warp them back.
        # But `draw_lane_lines` currently takes simple slope/intercept.
        # For visualization simply on the warped image for now or project back if possible.
        # For simplicity in this straight-line Hough version:
        # We will visualize on the original frame by just showing the computed text info.
        # If we want to show lines on original frame, we need to map points back.
        
        # Let's create a visual debug image or just final output
        # To keep it simple and effective:
        # We will draw lines on a blank warped image, unwarp it, and combine.
        
        line_warp = np.zeros_like(frame) # Blank RGB
        if left_line is not None:
             p1, p2 = lane_detection.pixel_points(frame.shape[0], 0, left_line)
             cv2.line(line_warp, p1, p2, (0, 255, 0), 30)
        if right_line is not None:
             p1, p2 = lane_detection.pixel_points(frame.shape[0], 0, right_line)
             cv2.line(line_warp, p1, p2, (0, 255, 0), 30)

        # Unwarp
        unwarped_lines, Minv = perspective_transform.inverse_birdeye(line_warp, self.settings.SRC_POINTS, self.settings.DST_POINTS)
        
        # Combine
        result = cv2.addWeighted(frame, 1, unwarped_lines, 0.9, 0)
        
        # Add Text
        result = overlay.draw_info(result, st_angle, offset)
        
        return result
