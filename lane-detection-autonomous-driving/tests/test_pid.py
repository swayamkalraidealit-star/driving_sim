import unittest
import sys
import os
import numpy as np

# Add src to python path to import correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.control.pid_controller import PIDController

class TestPIDController(unittest.TestCase):
    def test_proportional_updates(self):
        """Test simple P-controller behavior"""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        # Error = 1.0, Expect output = 1.0
        output = pid.update(1.0, dt=1.0)
        self.assertAlmostEqual(output, 1.0)
        
        # Error = -0.5, Expect output = -0.5
        output = pid.update(-0.5, dt=1.0)
        self.assertAlmostEqual(output, -0.5)

    def test_integral_updates(self):
        """Test I-term accumulation"""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        
        # First update: error=1.0, dt=1.0 -> Integral=1.0 -> Output=1.0
        output = pid.update(1.0, dt=1.0)
        self.assertAlmostEqual(output, 1.0)
        
        # Second update: error=1.0, dt=1.0 -> Integral=2.0 -> Output=2.0
        output = pid.update(1.0, dt=1.0)
        self.assertAlmostEqual(output, 2.0)

    def test_derivative_updates(self):
        """Test D-term response to change"""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        
        # First update: prev_error=0, current=1.0, dt=1.0 => derivative=(1-0)/1 = 1.0 -> Output=1.0
        output = pid.update(1.0, dt=1.0)
        self.assertAlmostEqual(output, 1.0)
        
        # Second update: prev_error=1.0, current=1.0 => derivative=0 -> Output=0.0
        output = pid.update(1.0, dt=1.0)
        self.assertAlmostEqual(output, 0.0)
        
        # Third update: prev_error=1.0, current=2.0 => derivative=1.0 -> Output=1.0
        output = pid.update(2.0, dt=1.0)
        self.assertAlmostEqual(output, 1.0)

    def test_clamping(self):
        """Test input/output saturation"""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, min_output=-10, max_output=10)
        
        # High error -> High P-term -> Clamped to 10
        output = pid.update(1.0)
        self.assertEqual(output, 10.0)
        
        output = pid.update(-1.0)
        self.assertEqual(output, -10.0)

if __name__ == '__main__':
    unittest.main()
