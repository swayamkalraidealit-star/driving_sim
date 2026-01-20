import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception import lane_detection

class TestPolyFit(unittest.TestCase):
    def test_fit_polynomial(self):
        # Create a synthetic image with a curved line
        # x = 0.01 * y^2 - 0.5 * y + 50
        image = np.zeros((100, 100), dtype=np.uint8)
        
        # Generate points
        ploty = np.linspace(0, 99, 100)
        true_fit = [0.005, -0.2, 50]
        true_x = true_fit[0]*ploty**2 + true_fit[1]*ploty + true_fit[2]
        
        # Draw the line
        for i in range(len(ploty)):
            x = int(true_x[i])
            y = int(ploty[i])
            if 0 <= x < 100:
                image[y, x] = 255
                
        # Run polyfit (we need to pass this as a "warped binary" so edges are just pixels)
        # In our pipeline, we pass an edge map. Our synthetic image IS an edge map.
        
        left_fit, right_fit, _, _ = lane_detection.fit_polynomial(image)
        
        # We only drew one line, but fit_polynomial looks for two based on histogram peaks.
        # Since we have one line in the middle, it might be classified as left or right depending on the split.
        # Let's adjust the image to have two lines clearly separated.
        
        image = np.zeros((100, 100), dtype=np.uint8)
        # Left lane (x around 25)
        for i in range(len(ploty)):
            x = int(true_x[i] - 25)
            y = int(ploty[i])
            if 0 <= x < 100:
                image[y, x] = 255
        
        # Right lane (x around 75)
        for i in range(len(ploty)):
            x = int(true_x[i] + 25)
            y = int(ploty[i])
            if 0 <= x < 100:
                image[y, x] = 255
                
        left_fit, right_fit, _, _ = lane_detection.fit_polynomial(image)
        
        self.assertIsNotNone(left_fit)
        self.assertIsNotNone(right_fit)
        
        # Check coefficients are reasonably close (simple check)
        # Leading coeff should be positive (curving right)
        self.assertTrue(left_fit[0] > 0)
        self.assertTrue(right_fit[0] > 0)

if __name__ == '__main__':
    unittest.main()
