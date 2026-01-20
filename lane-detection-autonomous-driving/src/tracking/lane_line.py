import numpy as np

class LaneLine:
    """
    Class to track the state of a lane line over time using Exponential Moving Average (EMA).
    """
    def __init__(self, alpha=0.2, max_lost=10):
        self.alpha = alpha
        self.max_lost = max_lost
        
        # Was the line detected in the last iteration?
        self.detected = False  
        
        # Count of consecutive lost frames
        self.lost_count = 0
        
        # Polynomial coefficients for the most recent fit
        self.current_fit = None  
        
        # Polynomial coefficients averaged over the last n iterations (EMA)
        self.best_fit = None  
        
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 

    def update(self, new_fit):
        """
        Update the lane state with a new fit.
        """
        if new_fit is None:
            self.detected = False
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.reset()
            return

        self.detected = True
        self.lost_count = 0
        self.current_fit = new_fit
        
        if self.best_fit is None:
            # First detection
            self.best_fit = self.current_fit
        else:
            # Apply Exponential Moving Average
            # new_average = alpha * new_val + (1 - alpha) * old_average
            self.best_fit = self.alpha * self.current_fit + (1 - self.alpha) * self.best_fit
            
    def get_fit(self):
        """
        Return the best fit if available, otherwise currents or None.
        """
        if self.best_fit is not None:
             return self.best_fit
        return self.current_fit

    def reset(self):
        """
        Reset state (e.g., if outlier detected or track lost for too long).
        """
        self.detected = False
        self.lost_count = 0
        self.best_fit = None
        self.current_fit = None
