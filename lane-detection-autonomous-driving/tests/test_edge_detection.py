import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception import edge_detection

class TestEdgeDetection(unittest.TestCase):
    def test_detect_edges(self):
        # Create a synthetic image with a white square on black background
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)
        
        edges = edge_detection.detect_edges(image)
        
        # Check if edges are detected (should be non-zero)
        self.assertTrue(np.sum(edges) > 0)
        
        # specific check: corners should be edges
        self.assertEqual(edges[25, 25], 255)
        self.assertEqual(edges[75, 75], 255)

if __name__ == '__main__':
    unittest.main()
