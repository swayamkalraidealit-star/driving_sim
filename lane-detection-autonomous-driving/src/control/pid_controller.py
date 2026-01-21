import numpy as np

class PIDController:
    """
    PID Controller for smooth steering control.
    """
    def __init__(self, kp, ki, kd, min_output=-25.0, max_output=25.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.min_output = min_output
        self.max_output = max_output
        
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error, dt=0.033):
        """
        Update the PID controller.
        
        Args:
            error: The current error (target - current).
            dt: Time step in seconds.
            
        Returns:
            Control output (e.g., steering angle).
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Simple clamping for integral windup - optional, but good practice
        # Limiting integral contribution to 50% of max output can be a safe heuristic
        max_integral = abs(self.max_output) / (2.0 * self.ki) if self.ki > 0 else 0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Total output
        output = p_term + i_term + d_term
        
        # Update previous error
        self.prev_error = error
        
        # Clip output
        output = np.clip(output, self.min_output, self.max_output)
        
        return output

    def reset(self):
        """Reset the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
